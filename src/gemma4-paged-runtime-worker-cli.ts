import { createHash } from "node:crypto";
import { createInterface } from "node:readline";
import { resolve } from "node:path";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { Gemma4BinaryConstantPool } from "./gemma4-binary-constant-pool.js";
import { Gemma4PagedNativeLinearWorker } from "./gemma4-paged-native-linear.js";
import { executeGemma4PagedTextLiteralF32 } from "./gemma4-paged-text.js";
import { selectGemma4LiteralGenerationToken } from "./gemma4-literal-generation-control.js";
import { rankGemma4TerminalLogits } from "./gemma4-terminal-logits.js";

const args = parseArguments(process.argv.slice(2));
const initializationStarted = performance.now();
const artifact = await openGemma4CompositeLiteralArtifact(args.artifact);
const pool = await Gemma4BinaryConstantPool.open(args.binaryPool);
const linear = new Gemma4PagedNativeLinearWorker({ python: args.python, helper: args.linearHelper, threads: args.threads, binaryPool: args.binaryPool, storageTensors: pool.catalog.tensors, backend: args.linearBackend, mlxHelper: args.mlxHelper });
const options = { maxReadBytes: args.maxReadBytes, finalHeadMaxReadBytes: args.finalHeadMaxReadBytes, allowUnverifiedFidelity: true, tensorReader: pool, linearTileKernel: linear, finalHeadCompute: args.finalHeadCompute, ...(args.fusedMlpRounding === "off" ? {} : { fusedMlpRounding: args.fusedMlpRounding }), ...(args.fusedFfnRounding === "off" ? {} : { fusedFfnRounding: args.fusedFfnRounding }), ...(args.fusedDecoderLayerRounding === "off" ? {} : { fusedDecoderLayerRounding: args.fusedDecoderLayerRounding }), ...(args.fusedPleRounding === "off" ? {} : { fusedPleRounding: args.fusedPleRounding }), ...(args.fusedPlePreludeRounding === "off" ? {} : { fusedPlePreludeRounding: args.fusedPlePreludeRounding }), ...(args.nativeAttentionRounding === "off" ? {} : { nativeAttentionRounding: args.nativeAttentionRounding }), ...(args.fusedAttentionRounding === "off" ? {} : { fusedAttentionRounding: args.fusedAttentionRounding }) };
process.stdout.write(`${JSON.stringify({ ready: true, initializationSeconds: (performance.now() - initializationStarted) / 1000, backend: "paged-binary-native", linearBackend: linear.backend, fusedMlpRounding: args.fusedMlpRounding, fusedFfnRounding: args.fusedFfnRounding, fusedDecoderLayerRounding: args.fusedDecoderLayerRounding, fusedPleRounding: args.fusedPleRounding, fusedPlePreludeRounding: args.fusedPlePreludeRounding, finalHeadCompute: args.finalHeadCompute, nativeAttentionRounding: args.nativeAttentionRounding, fusedAttentionRounding: args.fusedAttentionRounding, threads: args.threads, maxReadMiB: args.maxReadBytes / (1024 * 1024), finalHeadReadMiB: args.finalHeadMaxReadBytes / (1024 * 1024) })}\n`);

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  let request: { id?: unknown; inputIds?: unknown; maxNewTokens?: unknown } = {};
  try {
    request = JSON.parse(line) as typeof request;
    const inputIds = validateIds(request.inputIds), maxNewTokens = validateTokens(request.maxNewTokens);
    const report = await generate(inputIds, maxNewTokens);
    process.stdout.write(`${JSON.stringify({ id: request.id, report })}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ id: request.id, error: error instanceof Error ? error.message : String(error) })}\n`);
  }
}
await linear.close(); await pool.close(); await artifact.close();

async function generate(inputIds: number[], maxNewTokens: number): Promise<Record<string, unknown>> {
  const dispatchesBefore = linear.dispatchMetrics();
  const started = performance.now(), generatedTokenIds: number[] = [], steps: Array<Record<string, unknown>> = [];
  let forwardStarted = performance.now();
  let current = await executeGemma4PagedTextLiteralF32(artifact, { inputIds: [inputIds] }, options);
  let forwardSeconds = (performance.now() - forwardStarted) / 1000;
  for (let step = 0; step < maxNewTokens; step += 1) {
    const top = rankGemma4TerminalLogits(current.logits), tokenId = selectGemma4LiteralGenerationToken(artifact.generation.controlProgram, current.logits);
    generatedTokenIds.push(tokenId);
    steps.push({ step, tokenId, contextLength: inputIds.length + step, forwardSeconds, topLogits: top });
    if (step + 1 === maxNewTokens) break;
    forwardStarted = performance.now();
    current = await executeGemma4PagedTextLiteralF32(artifact, { inputIds: [[tokenId]], positionIds: [[inputIds.length + step]], pastKeyValues: current.pastKeyValues }, options);
    forwardSeconds = (performance.now() - forwardStarted) / 1000;
  }
  const elapsedSeconds = (performance.now() - started) / 1000;
  const terminalBytes = Buffer.from(current.logits.values.buffer, current.logits.values.byteOffset, current.logits.values.byteLength);
  const dispatchesAfter = linear.dispatchMetrics();
  return {
    kind: "gemma4-paged-binary-native-generation", schemaVersion: 2, backend: "paged-binary-native", linearBackend: linear.backend,
    sourceCheckpointAccessed: false, compatibilityBinaryWeightsAccessed: true,
    inputIds, maxNewTokens, generatedTokenIds, fullTokenIds: [...inputIds, ...generatedTokenIds], steps,
    terminalLogitsSha256: createHash("sha256").update(terminalBytes).digest("hex"),
    elapsedSeconds, tokensPerSecond: maxNewTokens / elapsedSeconds, linearThreads: args.threads,
    maxReadMiB: args.maxReadBytes / (1024 * 1024),
    finalHeadReadMiB: args.finalHeadMaxReadBytes / (1024 * 1024),
    linearReferenceDispatches: dispatchesAfter.referenceDispatches - dispatchesBefore.referenceDispatches,
    wholeNativeBf16Dispatches: dispatchesAfter.wholeNativeBf16Dispatches - dispatchesBefore.wholeNativeBf16Dispatches,
    linearBatchDispatches: dispatchesAfter.batchDispatches - dispatchesBefore.batchDispatches,
    linearBatchedProjectionTiles: dispatchesAfter.batchedProjectionTiles - dispatchesBefore.batchedProjectionTiles,
    fusedMlpRounding: args.fusedMlpRounding,
    fusedMlpDispatches: dispatchesAfter.fusedMlpDispatches - dispatchesBefore.fusedMlpDispatches,
    fusedFfnRounding: args.fusedFfnRounding,
    fusedFfnDispatches: dispatchesAfter.fusedFfnDispatches - dispatchesBefore.fusedFfnDispatches,
    fusedDecoderLayerRounding: args.fusedDecoderLayerRounding,
    fusedDecoderLayerDispatches: dispatchesAfter.fusedDecoderLayerDispatches - dispatchesBefore.fusedDecoderLayerDispatches,
    fusedPleRounding: args.fusedPleRounding,
    fusedPlePreludeRounding: args.fusedPlePreludeRounding,
    fusedPleDispatches: dispatchesAfter.fusedPleDispatches - dispatchesBefore.fusedPleDispatches,
    fusedPlePreludeDispatches: dispatchesAfter.fusedPlePreludeDispatches - dispatchesBefore.fusedPlePreludeDispatches,
    finalHeadCompute: args.finalHeadCompute,
    nativeAttentionRounding: args.nativeAttentionRounding,
    nativeAttentionDispatches: dispatchesAfter.nativeAttentionDispatches - dispatchesBefore.nativeAttentionDispatches,
    fusedAttentionRounding: args.fusedAttentionRounding,
    fusedAttentionDispatches: dispatchesAfter.fusedAttentionDispatches - dispatchesBefore.fusedAttentionDispatches,
    referenceSeconds: dispatchesAfter.referenceSeconds - dispatchesBefore.referenceSeconds,
    batchSeconds: dispatchesAfter.batchSeconds - dispatchesBefore.batchSeconds,
    fusedMlpSeconds: dispatchesAfter.fusedMlpSeconds - dispatchesBefore.fusedMlpSeconds,
    fusedFfnSeconds: dispatchesAfter.fusedFfnSeconds - dispatchesBefore.fusedFfnSeconds,
    fusedDecoderLayerSeconds: dispatchesAfter.fusedDecoderLayerSeconds - dispatchesBefore.fusedDecoderLayerSeconds,
    fusedPleSeconds: dispatchesAfter.fusedPleSeconds - dispatchesBefore.fusedPleSeconds,
    fusedPlePreludeSeconds: dispatchesAfter.fusedPlePreludeSeconds - dispatchesBefore.fusedPlePreludeSeconds,
    nativeAttentionSeconds: dispatchesAfter.nativeAttentionSeconds - dispatchesBefore.nativeAttentionSeconds,
    fusedAttentionSeconds: dispatchesAfter.fusedAttentionSeconds - dispatchesBefore.fusedAttentionSeconds,
    processRssBytes: process.memoryUsage().rss, processMaxRssKiB: process.resourceUsage().maxRSS,
  };
}

function validateIds(value: unknown): number[] {
  if (!Array.isArray(value) || value.length === 0 || value.some((token) => !Number.isSafeInteger(token) || (token as number) < 0)) throw new Error("inputIds deve ser array não vazio de inteiros não negativos.");
  return value as number[];
}
function validateTokens(value: unknown): number {
  if (!Number.isSafeInteger(value) || (value as number) < 1 || (value as number) > 64) throw new Error("maxNewTokens deve estar entre 1 e 64.");
  return value as number;
}
function parseArguments(argv: string[]): { artifact: string; binaryPool: string; python: string; linearHelper: string; mlxHelper: string; linearBackend: "pytorch" | "mlx"; fusedMlpRounding: "off" | "bf16" | "real" | "native-bf16"; fusedFfnRounding: "off" | "native-bf16"; fusedDecoderLayerRounding: "off" | "native-bf16"; fusedPleRounding: "off" | "bf16" | "real"; fusedPlePreludeRounding: "off" | "bf16" | "real"; finalHeadCompute: "f32" | "native-bf16" | "native-bf16-whole"; nativeAttentionRounding: "off" | "bf16" | "real"; fusedAttentionRounding: "off" | "bf16" | "real" | "native-bf16"; threads: number; maxReadBytes: number; finalHeadMaxReadBytes: number } {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 2) { const flag = argv[index], value = argv[index + 1]; if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`); values.set(flag, value); }
  const known = new Set(["--artifact", "--binary-pool", "--python", "--linear-helper", "--mlx-helper", "--linear-backend", "--fused-mlp", "--fused-ffn", "--fused-decoder-layer", "--fused-ple", "--fused-ple-prelude", "--final-head", "--final-head-read-mib", "--native-attention", "--fused-attention", "--threads", "--max-read-mib"]); for (const flag of values.keys()) if (!known.has(flag)) throw new Error(`Flag desconhecida: ${flag}.`);
  const required = (flag: string): string => { const value = values.get(flag); if (!value) throw new Error(`${flag} é obrigatório.`); return resolve(value); };
  const threads = Number(values.get("--threads") ?? "8"), maxReadMiB = Number(values.get("--max-read-mib") ?? "16");
  const linearBackend = values.get("--linear-backend") ?? "pytorch"; if (linearBackend !== "pytorch" && linearBackend !== "mlx") throw new Error("--linear-backend deve ser pytorch ou mlx.");
  const finalHeadReadMiB = Number(values.get("--final-head-read-mib") ?? (linearBackend === "pytorch" ? "32" : String(maxReadMiB)));
  if (!Number.isSafeInteger(threads) || threads < 1 || threads > 256 || !Number.isSafeInteger(maxReadMiB) || maxReadMiB < 1 || maxReadMiB > 1024 || !Number.isSafeInteger(finalHeadReadMiB) || finalHeadReadMiB < 1 || finalHeadReadMiB > 1024) throw new Error("threads/max-read-mib/final-head-read-mib inválidos.");
  const fusedMlpRounding = values.get("--fused-mlp") ?? (linearBackend === "pytorch" ? "native-bf16" : "real"); if (fusedMlpRounding !== "off" && fusedMlpRounding !== "bf16" && fusedMlpRounding !== "real" && fusedMlpRounding !== "native-bf16") throw new Error("--fused-mlp deve ser off, bf16, real ou native-bf16.");
  if (linearBackend === "mlx" && fusedMlpRounding === "native-bf16") throw new Error("--fused-mlp native-bf16 requer --linear-backend pytorch.");
  const fusedFfnRounding = values.get("--fused-ffn") ?? (linearBackend === "pytorch" ? "native-bf16" : "off"); if (fusedFfnRounding !== "off" && fusedFfnRounding !== "native-bf16") throw new Error("--fused-ffn deve ser off ou native-bf16.");
  if (linearBackend === "mlx" && fusedFfnRounding !== "off") throw new Error("--fused-ffn requer --linear-backend pytorch.");
  const fusedDecoderLayerRounding = values.get("--fused-decoder-layer") ?? (linearBackend === "pytorch" ? "native-bf16" : "off"); if (fusedDecoderLayerRounding !== "off" && fusedDecoderLayerRounding !== "native-bf16") throw new Error("--fused-decoder-layer deve ser off ou native-bf16.");
  if (linearBackend === "mlx" && fusedDecoderLayerRounding !== "off") throw new Error("--fused-decoder-layer requer --linear-backend pytorch.");
  const fusedPleRounding = values.get("--fused-ple") ?? (linearBackend === "pytorch" ? "bf16" : "off"); if (fusedPleRounding !== "off" && fusedPleRounding !== "bf16" && fusedPleRounding !== "real") throw new Error("--fused-ple deve ser off, bf16 ou real.");
  if (linearBackend === "mlx" && fusedPleRounding !== "off") throw new Error("--fused-ple requer --linear-backend pytorch.");
  const fusedPlePreludeRounding = values.get("--fused-ple-prelude") ?? "off"; if (fusedPlePreludeRounding !== "off" && fusedPlePreludeRounding !== "bf16" && fusedPlePreludeRounding !== "real") throw new Error("--fused-ple-prelude deve ser off, bf16 ou real.");
  if (linearBackend === "mlx" && fusedPlePreludeRounding !== "off") throw new Error("--fused-ple-prelude requer --linear-backend pytorch.");
  const finalHeadCompute = values.get("--final-head") ?? (linearBackend === "pytorch" ? "native-bf16" : "f32"); if (finalHeadCompute !== "f32" && finalHeadCompute !== "native-bf16" && finalHeadCompute !== "native-bf16-whole") throw new Error("--final-head deve ser f32, native-bf16 ou native-bf16-whole.");
  if (linearBackend === "mlx" && finalHeadCompute !== "f32") throw new Error("--final-head BF16 requer --linear-backend pytorch.");
  const nativeAttentionRounding = values.get("--native-attention") ?? (linearBackend === "pytorch" ? "real" : "off"); if (nativeAttentionRounding !== "off" && nativeAttentionRounding !== "bf16" && nativeAttentionRounding !== "real") throw new Error("--native-attention deve ser off, bf16 ou real.");
  if (linearBackend === "mlx" && nativeAttentionRounding !== "off") throw new Error("--native-attention requer --linear-backend pytorch.");
  const fusedAttentionRounding = values.get("--fused-attention") ?? (linearBackend === "pytorch" ? "native-bf16" : "off"); if (fusedAttentionRounding !== "off" && fusedAttentionRounding !== "bf16" && fusedAttentionRounding !== "real" && fusedAttentionRounding !== "native-bf16") throw new Error("--fused-attention deve ser off, bf16, real ou native-bf16.");
  if (linearBackend === "mlx" && fusedAttentionRounding !== "off") throw new Error("--fused-attention requer --linear-backend pytorch.");
  return { artifact: required("--artifact"), binaryPool: required("--binary-pool"), python: values.get("--python") ?? resolve("venv/bin/python"), linearHelper: resolve(values.get("--linear-helper") ?? "scripts/gemma4-paged-linear-worker.py"), mlxHelper: resolve(values.get("--mlx-helper") ?? "scripts/gemma4-mlx-linear-worker.py"), linearBackend, fusedMlpRounding, fusedFfnRounding, fusedDecoderLayerRounding, fusedPleRounding, fusedPlePreludeRounding, finalHeadCompute, nativeAttentionRounding, fusedAttentionRounding, threads, maxReadBytes: maxReadMiB * 1024 * 1024, finalHeadMaxReadBytes: finalHeadReadMiB * 1024 * 1024 };
}
