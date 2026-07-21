import { createHash } from "node:crypto";
import { constants as fsConstants, createReadStream, createWriteStream } from "node:fs";
import { copyFile, mkdir, open, stat, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

export interface Gemma4CompiledBundleManifest {
  kind: "gemma4-compiled-shared-dag-bundle";
  schemaVersion: 1;
  execution: "shared-dag-constant-pool-multithread-logit-tiles";
  formula: { family: string; dimension: number; root: string; expressionNodes: number; inputTensor: "x"; inputLength: number; file: string };
  globalProgram?: { file: string; terminalLogits: number; constantPool: string };
  files: Array<{ role: "formula-graph" | "global-formulas" | "constant-pool" | "runtime-weights" | "tokenizer" | "tokenizer-config" | "generation-config" | "model-config"; file: string; bytes: number; sha256: string }>;
}

export async function createGemma4CompiledBundle(options: { graph: string; globalSsa?: string; runtimeModel?: string; constantArtifact: string; tokenizerDirectory: string; outputDirectory: string }): Promise<Gemma4CompiledBundleManifest> {
  const graph = resolve(options.graph), constants = resolve(options.constantArtifact), tokenizerDirectory = resolve(options.tokenizerDirectory), output = resolve(options.outputDirectory);
  await mkdir(output);
  const summary = await readClosedGraphSummary(graph);
  if (summary.remainingFunctionCalls.length !== 0 || summary.inputVector.tensor !== "x") throw new Error("Bundle compilado requer closure sem funções e com entrada exclusiva x.");
  const sources = [
    { source: graph, file: "formula.graph.json", role: "formula-graph" as const },
    { source: constants, file: "constants.literal.json", role: "constant-pool" as const },
    { source: join(tokenizerDirectory, "tokenizer.json"), file: "tokenizer.json", role: "tokenizer" as const },
    { source: join(tokenizerDirectory, "tokenizer_config.json"), file: "tokenizer_config.json", role: "tokenizer-config" as const },
    { source: join(tokenizerDirectory, "generation_config.json"), file: "generation_config.json", role: "generation-config" as const },
    { source: join(tokenizerDirectory, "config.json"), file: "config.json", role: "model-config" as const },
    ...(options.runtimeModel ? [{ source: resolve(options.runtimeModel), file: "model.safetensors", role: "runtime-weights" as const }] : []),
  ];
  const files: Gemma4CompiledBundleManifest["files"] = [];
  for (const source of sources) {
    const info = await stat(source.source); if (!info.isFile() || info.size === 0) throw new Error(`${source.source}: arquivo obrigatório do bundle ausente.`);
    await copyFile(source.source, join(output, source.file), fsConstants.COPYFILE_FICLONE);
    files.push({ role: source.role, file: source.file, bytes: info.size, sha256: await sha256File(source.source) });
  }
  let globalProgram: Gemma4CompiledBundleManifest["globalProgram"];
  if (options.globalSsa) {
    const destination = join(output, "global-formulas.ssa.json");
    await copyPortableGlobalSsa(resolve(options.globalSsa), destination);
    const info = await stat(destination), terminalLogits = await countOccurrences(destination, '"assignment":"calc_terminal_logit_');
    if (terminalLogits === 0) throw new Error("SSA global não contém logits terminais.");
    files.push({ role: "global-formulas", file: "global-formulas.ssa.json", bytes: info.size, sha256: await sha256File(destination) });
    globalProgram = { file: "global-formulas.ssa.json", terminalLogits, constantPool: "constants.literal.json" };
  }
  const manifest: Gemma4CompiledBundleManifest = {
    kind: "gemma4-compiled-shared-dag-bundle", schemaVersion: 1, execution: "shared-dag-constant-pool-multithread-logit-tiles",
    formula: { family: summary.output.family, dimension: summary.output.dimension, root: summary.root, expressionNodes: summary.expressionNodes, inputTensor: "x", inputLength: summary.inputVector.length, file: "formula.graph.json" },
    ...(globalProgram ? { globalProgram } : {}),
    files,
  };
  await writeFile(join(output, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  return manifest;
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
