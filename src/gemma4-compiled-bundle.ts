import { createHash } from "node:crypto";
import { constants as fsConstants, createReadStream, createWriteStream } from "node:fs";
import { copyFile, mkdir, open, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { createGemma4PagedRuntimeIndex, openGemma4PagedRuntimeArtifact } from "./gemma4-paged-runtime-index.js";
import { validateGemma4VectorizedRealLoweringPlan, type Gemma4VectorizedRealLoweringPlan } from "./gemma4-vectorized-real-lowering.js";

export interface Gemma4CompiledBundleManifest {
  kind: "gemma4-compiled-shared-dag-bundle";
  schemaVersion: 2 | 3;
  execution: "vectorized-literal-runtime-with-global-formula-reference";
  runtimeLowering: {
    engine: "mlx-f32-real-decoder-stack-v1";
    directlyExecutesGlobalFormula: false;
    executesPersistedLoweringPlan: true;
    plan: "vectorized-real-lowering.json";
    functionBindingsSha256: string;
    outputBindingsSha256: string;
    standaloneSsaOutputsSha256: string;
    outputFunctions: number;
    realSimplifiedProgramSha256: string;
    globalFormulaRole: "algebraic-source-and-scalar-reference";
  };
  formula: { family: string; dimension: number; root: string; expressionNodes: number; inputTensor: "x"; inputLength: number; file: string };
  globalProgram?: { file: string; terminalLogits: number; constantPool: string };
  runtimeIndex?: { file: "constants.runtime-index.json"; schemaVersion: 1 | 2; constantPoolSha256: string; integrityRootSha256: string };
  files: Array<{ role: "formula-graph" | "global-formulas" | "vectorized-real-lowering" | "constant-pool" | "literal-runtime-index" | "runtime-weights" | "tokenizer" | "tokenizer-config" | "generation-config" | "model-config"; file: string; bytes: number; sha256: string }>;
}

export async function createGemma4CompiledBundle(options: { graph: string; globalSsa: string; realLoweringPlan: string; runtimeModel?: string; constantArtifact: string; tokenizerDirectory: string; outputDirectory: string; createRuntimeIndex?: boolean }): Promise<Gemma4CompiledBundleManifest> {
  const graph = resolve(options.graph), constants = resolve(options.constantArtifact), tokenizerDirectory = resolve(options.tokenizerDirectory), output = resolve(options.outputDirectory);
  await mkdir(output);
  const summary = await readClosedGraphSummary(graph);
  if (summary.remainingFunctionCalls.length !== 0 || summary.inputVector.tensor !== "x") throw new Error("Bundle compilado requer closure sem funções e com entrada exclusiva x.");
  const loweringPlan = await readLoweringPlan(resolve(options.realLoweringPlan));
  const sources = [
    { source: graph, file: "formula.graph.json", role: "formula-graph" as const },
    { source: constants, file: "constants.literal.json", role: "constant-pool" as const },
    { source: join(tokenizerDirectory, "tokenizer.json"), file: "tokenizer.json", role: "tokenizer" as const },
    { source: join(tokenizerDirectory, "tokenizer_config.json"), file: "tokenizer_config.json", role: "tokenizer-config" as const },
    { source: join(tokenizerDirectory, "generation_config.json"), file: "generation_config.json", role: "generation-config" as const },
    { source: join(tokenizerDirectory, "config.json"), file: "config.json", role: "model-config" as const },
    { source: resolve(options.realLoweringPlan), file: "vectorized-real-lowering.json", role: "vectorized-real-lowering" as const },
    ...(options.runtimeModel ? [{ source: resolve(options.runtimeModel), file: "model.safetensors", role: "runtime-weights" as const }] : []),
  ];
  const files: Gemma4CompiledBundleManifest["files"] = [];
  for (const source of sources) {
    const info = await stat(source.source); if (!info.isFile() || info.size === 0) throw new Error(`${source.source}: arquivo obrigatório do bundle ausente.`);
    await copyFile(source.source, join(output, source.file), fsConstants.COPYFILE_FICLONE);
    files.push({ role: source.role, file: source.file, bytes: info.size, sha256: await sha256File(source.source) });
  }
  const constantFile = files.find((entry) => entry.role === "constant-pool")!;
  const loweringFile = files.find((entry) => entry.role === "vectorized-real-lowering")!;
  const runtimeIndexDescriptor = options.createRuntimeIndex === false ? undefined : await createGemma4PagedRuntimeIndex(join(output, constantFile.file), join(output, "constants.runtime-index.json"), constantFile.sha256, { path: join(output, loweringFile.file), sha256: loweringFile.sha256 });
  if (runtimeIndexDescriptor) files.push({ role: "literal-runtime-index", file: "constants.runtime-index.json", bytes: runtimeIndexDescriptor.bytes, sha256: runtimeIndexDescriptor.sha256 });
  const destination = join(output, "global-formulas.ssa.json");
  await copyPortableGlobalSsa(resolve(options.globalSsa), destination);
  const info = await stat(destination), standaloneSsaOutputsSha256 = await hashGlobalSsaOutputs(destination);
  if (standaloneSsaOutputsSha256 !== loweringPlan.contract.source.standaloneSsaOutputsSha256) throw new Error("SSA global não corresponde às saídas finais autenticadas pelo plano vetorizado.");
  const terminalLogits = loweringPlan.contract.source.outputFamilies.terminal_logit?.dimensions ?? 0;
  if (terminalLogits === 0) throw new Error("Plano vetorizado não contém logits terminais.");
  files.push({ role: "global-formulas", file: "global-formulas.ssa.json", bytes: info.size, sha256: await sha256File(destination) });
  const globalProgram: NonNullable<Gemma4CompiledBundleManifest["globalProgram"]> = { file: "global-formulas.ssa.json", terminalLogits, constantPool: "constants.literal.json" };
  const manifest: Gemma4CompiledBundleManifest = {
    kind: "gemma4-compiled-shared-dag-bundle", schemaVersion: runtimeIndexDescriptor ? 3 : 2, execution: "vectorized-literal-runtime-with-global-formula-reference",
    runtimeLowering: {
      engine: "mlx-f32-real-decoder-stack-v1", directlyExecutesGlobalFormula: false, executesPersistedLoweringPlan: true,
      plan: "vectorized-real-lowering.json", functionBindingsSha256: loweringPlan.contract.functionBindingsSha256,
      outputBindingsSha256: loweringPlan.contract.source.outputBindingsSha256, standaloneSsaOutputsSha256,
      outputFunctions: loweringPlan.contract.source.outputFunctions,
      realSimplifiedProgramSha256: loweringPlan.contract.source.realSimplifiedProgramSha256,
      globalFormulaRole: "algebraic-source-and-scalar-reference",
    },
    formula: { family: summary.output.family, dimension: summary.output.dimension, root: summary.root, expressionNodes: summary.expressionNodes, inputTensor: "x", inputLength: summary.inputVector.length, file: "formula.graph.json" },
    globalProgram,
    ...(runtimeIndexDescriptor ? { runtimeIndex: { file: "constants.runtime-index.json" as const, schemaVersion: runtimeIndexDescriptor.schemaVersion, constantPoolSha256: constantFile.sha256, integrityRootSha256: runtimeIndexDescriptor.integrityRootSha256 } } : {}),
    files,
  };
  await writeFile(join(output, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  return manifest;
}

/** Atomically upgrades an existing v1 bundle after compiling its real plan. */
export async function bindGemma4VectorizedRealLoweringPlan(bundleDirectory: string, planPath: string): Promise<Gemma4CompiledBundleManifest> {
  const bundle = resolve(bundleDirectory), source = resolve(planPath), destination = join(bundle, "vectorized-real-lowering.json");
  const plan = await readLoweringPlan(source);
  const manifestPath = join(bundle, "manifest.json");
  const current = JSON.parse(await readFile(manifestPath, "utf8")) as Record<string, unknown> & { files?: Gemma4CompiledBundleManifest["files"] };
  if (current.kind !== "gemma4-compiled-shared-dag-bundle" || current.execution !== "vectorized-literal-runtime-with-global-formula-reference" || !Array.isArray(current.files)) {
    throw new Error("Bundle existente não possui manifesto Gemma 4 compilado atualizável.");
  }
  const globalFile = current.files.find((entry) => entry.role === "global-formulas");
  if (!globalFile || await hashGlobalSsaOutputs(join(bundle, globalFile.file)) !== plan.contract.source.standaloneSsaOutputsSha256) {
    throw new Error("SSA global existente não corresponde às saídas finais do novo plano vetorizado.");
  }
  if (source !== destination) {
    const temporaryPlan = `${destination}.next-${process.pid}`;
    try { await copyFile(source, temporaryPlan, fsConstants.COPYFILE_EXCL); await rename(temporaryPlan, destination); }
    catch (error) { await rm(temporaryPlan, { force: true }); throw error; }
  }
  const info = await stat(destination), sha256 = await sha256File(destination);
  const existing = current.files.filter((entry) => entry.role !== "vectorized-real-lowering");
  const manifest = {
    ...current,
    schemaVersion: current.runtimeIndex ? 3 : 2,
    runtimeLowering: {
      engine: "mlx-f32-real-decoder-stack-v1", directlyExecutesGlobalFormula: false, executesPersistedLoweringPlan: true,
      plan: "vectorized-real-lowering.json", functionBindingsSha256: plan.contract.functionBindingsSha256,
      outputBindingsSha256: plan.contract.source.outputBindingsSha256, standaloneSsaOutputsSha256: plan.contract.source.standaloneSsaOutputsSha256,
      outputFunctions: plan.contract.source.outputFunctions,
      realSimplifiedProgramSha256: plan.contract.source.realSimplifiedProgramSha256,
      globalFormulaRole: "algebraic-source-and-scalar-reference",
    },
    files: [...existing, { role: "vectorized-real-lowering" as const, file: "vectorized-real-lowering.json", bytes: info.size, sha256 }],
  } as Gemma4CompiledBundleManifest;
  const temporary = `${manifestPath}.next-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(manifest, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  await rename(temporary, manifestPath);
  return manifest;
}

/** Atomically adds or upgrades the authenticated execution index in a compiled bundle. */
export async function bindGemma4LiteralRuntimeIndex(bundleDirectory: string): Promise<Gemma4CompiledBundleManifest> {
  const bundle = resolve(bundleDirectory), manifestPath = join(bundle, "manifest.json"), destination = join(bundle, "constants.runtime-index.json");
  const current = JSON.parse(await readFile(manifestPath, "utf8")) as Record<string, unknown> & { files?: Gemma4CompiledBundleManifest["files"]; runtimeIndex?: Gemma4CompiledBundleManifest["runtimeIndex"] };
  if (current.kind !== "gemma4-compiled-shared-dag-bundle" || current.execution !== "vectorized-literal-runtime-with-global-formula-reference" || !Array.isArray(current.files)) throw new Error("Bundle existente não possui manifesto Gemma 4 compilado atualizável.");
  const constantFile = current.files.find((entry) => entry.role === "constant-pool");
  const loweringFile = current.files.find((entry) => entry.role === "vectorized-real-lowering");
  if (!constantFile || constantFile.file !== "constants.literal.json" || !/^[0-9a-f]{64}$/.test(constantFile.sha256)) throw new Error("Bundle existente não vincula o constant pool literal esperado.");
  if (!loweringFile || loweringFile.file !== "vectorized-real-lowering.json" || !/^[0-9a-f]{64}$/.test(loweringFile.sha256)) throw new Error("Bundle existente não vincula o plano de lowering vetorizado esperado.");
  if (current.runtimeIndex?.schemaVersion === 2) {
    const existing = current.files.find((entry) => entry.role === "literal-runtime-index");
    if (!existing || existing.file !== current.runtimeIndex.file || await sha256File(join(bundle, existing.file)) !== existing.sha256) throw new Error("Índice runtime paginado existente diverge do manifesto do bundle.");
    const planBytes = await readFile(join(bundle, loweringFile.file)), plan = JSON.parse(planBytes.toString("utf8")) as unknown;
    validateGemma4VectorizedRealLoweringPlan(plan);
    const indexPath = join(bundle, existing.file), indexInfo = await stat(indexPath);
    if (!indexInfo.isFile() || indexInfo.size < 1 || indexInfo.size > 64 * 1024 * 1024) throw new Error("Índice runtime paginado existente possui tamanho inválido.");
    const indexed = JSON.parse(await readFile(indexPath, "utf8")) as { realLowering?: { planSha256?: unknown } };
    if (indexed.realLowering?.planSha256 === loweringFile.sha256) {
      const opened = await openGemma4PagedRuntimeArtifact(join(bundle, constantFile.file), { path: join(bundle, existing.file), sha256: existing.sha256, constantPoolSha256: constantFile.sha256 });
      try {
        if (!("runtimeIndexSchemaVersion" in opened) || opened.integrityManifest.rootSha256 !== current.runtimeIndex.integrityRootSha256 || !isDeepStrictEqual(opened.realLowering?.contract, plan.contract)) throw new Error("Índice runtime paginado existente diverge da raiz ou do plano declarado no bundle.");
        return current as unknown as Gemma4CompiledBundleManifest;
      } finally { await opened.close(); }
    }
  }
  else if (current.runtimeIndex && current.runtimeIndex.schemaVersion !== 1) throw new Error("Bundle existente declara versão de índice runtime desconhecida.");
  const descriptor = await createGemma4PagedRuntimeIndex(join(bundle, constantFile.file), destination, constantFile.sha256, { path: join(bundle, loweringFile.file), sha256: loweringFile.sha256 });
  const files = current.files.filter((entry) => entry.role !== "literal-runtime-index");
  const manifest = {
    ...current, schemaVersion: 3,
    runtimeIndex: { file: "constants.runtime-index.json", schemaVersion: descriptor.schemaVersion, constantPoolSha256: constantFile.sha256, integrityRootSha256: descriptor.integrityRootSha256 },
    files: [...files, { role: "literal-runtime-index" as const, file: "constants.runtime-index.json", bytes: descriptor.bytes, sha256: descriptor.sha256 }],
  } as Gemma4CompiledBundleManifest;
  const temporary = `${manifestPath}.next-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(manifest, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  await rename(temporary, manifestPath);
  return manifest;
}

async function readLoweringPlan(path: string): Promise<Gemma4VectorizedRealLoweringPlan> {
  const info = await stat(path);
  if (!info.isFile() || info.size < 1 || info.size > 16 * 1024 * 1024) throw new Error("Plano de lowering vetorizado deve ser arquivo JSON não vazio de até 16 MiB.");
  const value = JSON.parse(await readFile(path, "utf8")) as unknown;
  validateGemma4VectorizedRealLoweringPlan(value);
  return value;
}

async function copyPortableGlobalSsa(source: string, destination: string): Promise<void> {
  const handle = await open(source, "r");
  try {
    const prefix = Buffer.alloc(64 * 1024), read = await handle.read(prefix, 0, prefix.length, 0), text = prefix.subarray(0, read.bytesRead).toString("utf8");
    const match = text.match(/"constantMemory":\{"kind":"authenticated-gemma4-literal-artifact","artifact":"[^"]+"/);
    if (!match || match.index === undefined) throw new Error("SSA global não declara constantMemory.artifact no cabeçalho.");
    const portable = match[0].replace(/"artifact":"[^"]+"/, '"artifact":"constants.literal.json"');
    const rewritten = text.slice(0, match.index) + portable + text.slice(match.index + match[0].length);
    const output = createWriteStream(destination, { flags: "wx" }); output.write(rewritten);
    await new Promise<void>((accept, reject) => { const remainder = createReadStream(source, { start: read.bytesRead }); remainder.once("error", reject); output.once("error", reject); output.once("finish", accept); remainder.pipe(output); });
  } finally { await handle.close(); }
}

async function hashGlobalSsaOutputs(path: string): Promise<string> {
  const marker = Buffer.from('"outputs":['), digest = createHash("sha256");
  const stream = createReadStream(path); let carry: Buffer<ArrayBufferLike> = Buffer.alloc(0), found = false, depth = 0, inString = false, escaped = false, complete = false;
  for await (const chunk_ of stream) {
    const chunk = chunk_ as Buffer; let data = chunk;
    if (!found) {
      const searchable = carry.length ? Buffer.concat([carry, chunk]) : chunk, markerIndex = searchable.indexOf(marker);
      if (markerIndex < 0) { carry = searchable.subarray(Math.max(0, searchable.length - marker.length + 1)); continue; }
      found = true; data = searchable.subarray(markerIndex + marker.length - 1);
    }
    let end = data.length;
    for (let index = 0; index < data.length; index += 1) {
      const byte = data[index]!;
      if (inString) {
        if (escaped) escaped = false;
        else if (byte === 0x5c) escaped = true;
        else if (byte === 0x22) inString = false;
      } else if (byte === 0x22) inString = true;
      else if (byte === 0x5b) depth += 1;
      else if (byte === 0x5d) {
        depth -= 1;
        if (depth === 0) { end = index + 1; complete = true; break; }
      }
    }
    digest.update(data.subarray(0, end));
    if (complete) { stream.destroy(); break; }
  }
  if (!found || !complete || depth !== 0 || inString) throw new Error("SSA global não contém um array outputs JSON completo.");
  return digest.digest("hex");
}

interface ClosedGraphSummary {
  output: { family: string; dimension: number };
  root: string;
  expressionNodes: number;
  remainingFunctionCalls: string[];
  inputVector: { tensor: "x"; length: number };
}

async function readClosedGraphSummary(path: string): Promise<ClosedGraphSummary> {
  const info = await stat(path), handle = await open(path, "r");
  try {
    const head = Buffer.alloc(Math.min(info.size, 4096)); await handle.read(head, 0, head.length, 0);
    const outputMatch = head.toString("utf8").match(/"output":\{"family":"([^"]+)","dimension":(\d+)/);
    if (!outputMatch) throw new Error(`${basename(path)}: cabeçalho de output ausente.`);
    const tailBytes = Math.min(info.size, 2 * 1024 * 1024), tail = Buffer.alloc(tailBytes); await handle.read(tail, 0, tailBytes, info.size - tailBytes);
    const text = tail.toString("utf8"), marker = text.lastIndexOf(']},"root":');
    if (marker < 0) throw new Error(`${basename(path)}: cauda do grafo ausente.`);
    const suffix = JSON.parse(`{${text.slice(marker + 3)}`) as Omit<ClosedGraphSummary, "output" | "expressionNodes">;
    const expressionNodes = await countOccurrences(path, '"id":"sha256:');
    return { output: { family: outputMatch[1]!, dimension: Number(outputMatch[2]) }, expressionNodes, ...suffix };
  } finally { await handle.close(); }
}

async function countOccurrences(path: string, needle: string): Promise<number> {
  let count = 0, carry = "";
  for await (const chunk of createReadStream(path, { encoding: "utf8" })) {
    const text = carry + chunk; let index = 0;
    while ((index = text.indexOf(needle, index)) >= 0) { count += 1; index += needle.length; }
    carry = text.slice(-(needle.length - 1));
  }
  return count;
}

async function sha256File(path: string): Promise<string> {
  const digest = createHash("sha256"); for await (const chunk of createReadStream(path)) digest.update(chunk as Buffer); return digest.digest("hex");
}
