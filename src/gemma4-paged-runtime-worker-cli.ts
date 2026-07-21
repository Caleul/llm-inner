import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { createInterface } from "node:readline";
import { join, resolve } from "node:path";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { Gemma4BinaryConstantPool } from "./gemma4-binary-constant-pool.js";
import { Gemma4PagedNativeLinearWorker } from "./gemma4-paged-native-linear.js";
import { executeGemma4PagedTextEpilogueLiteralF32, executeGemma4PagedTextHiddenLiteralF32, executeGemma4PagedTextLiteralF32, generateGemma4PagedTextLiteralNativeF32 } from "./gemma4-paged-text.js";
import { selectGemma4LiteralGenerationToken } from "./gemma4-literal-generation-control.js";
import { rankGemma4TerminalLogits } from "./gemma4-terminal-logits.js";
import { assertGemma4VectorizedRealLoweringPlanMatches, Gemma4VectorizedRealExecutionGuard, type Gemma4VectorizedRealLoweringPlan } from "./gemma4-vectorized-real-lowering.js";

const args = parseArguments(process.argv.slice(2));
const initializationStarted = performance.now();
const artifact = await openGemma4CompositeLiteralArtifact(args.artifact);
const pool = await Gemma4BinaryConstantPool.open(args.binaryPool);
let vectorizedRealLowering: Gemma4VectorizedRealLoweringPlan["contract"] | undefined;
let vectorizedRealExecutionGuard: Gemma4VectorizedRealExecutionGuard | undefined;
if (args.fusedDecoderStackRounding === "real") {
  const planInfo = await stat(args.realLoweringPlan!);
  if (!planInfo.isFile() || planInfo.size < 1 || planInfo.size > 16 * 1024 * 1024) throw new Error("Plano de lowering real persistido deve ter entre 1 byte e 16 MiB.");
  const persisted = JSON.parse(await readFile(args.realLoweringPlan!, "utf8")) as unknown;
  assertGemma4VectorizedRealLoweringPlanMatches(persisted, artifact.realSimplifiedProgram, artifact.calculationGraph, artifact.integrityManifest);
  vectorizedRealLowering = persisted.contract;
  vectorizedRealExecutionGuard = new Gemma4VectorizedRealExecutionGuard(persisted);
}
const linear = new Gemma4PagedNativeLinearWorker({ python: args.python, helper: args.linearHelper, threads: args.threads, binaryPool: args.binaryPool, storageTensors: pool.catalog.tensors, backend: args.linearBackend, mlxHelper: args.mlxHelper });
const options = { maxReadBytes: args.maxReadBytes, finalHeadMaxReadBytes: args.finalHeadMaxReadBytes, allowUnverifiedFidelity: true, tensorReader: pool, linearTileKernel: linear, finalHeadCompute: args.finalHeadCompute, ...(vectorizedRealExecutionGuard ? { vectorizedRealExecutionGuard } : {}), ...(args.fusedMlpRounding === "off" ? {} : { fusedMlpRounding: args.fusedMlpRounding }), ...(args.fusedFfnRounding === "off" ? {} : { fusedFfnRounding: args.fusedFfnRounding }), ...(args.fusedDecoderLayerRounding === "off" ? {} : { fusedDecoderLayerRounding: args.fusedDecoderLayerRounding }), ...(args.fusedDecoderStackRounding === "off" ? {} : { fusedDecoderStackRounding: args.fusedDecoderStackRounding }), ...(args.fusedPleRounding === "off" ? {} : { fusedPleRounding: args.fusedPleRounding }), ...(args.fusedPlePreludeRounding === "off" ? {} : { fusedPlePreludeRounding: args.fusedPlePreludeRounding }), ...(args.fusedTokenForwardRounding === "off" ? {} : { fusedTokenForwardRounding: args.fusedTokenForwardRounding }), ...(args.nativeAttentionRounding === "off" ? {} : { nativeAttentionRounding: args.nativeAttentionRounding }), ...(args.fusedAttentionRounding === "off" ? {} : { fusedAttentionRounding: args.fusedAttentionRounding }) };
process.stdout.write(`${JSON.stringify({ ready: true, initializationSeconds: (performance.now() - initializationStarted) / 1000, backend: "paged-binary-native", linearBackend: linear.backend, fusedMlpRounding: args.fusedMlpRounding, fusedFfnRounding: args.fusedFfnRounding, fusedDecoderLayerRounding: args.fusedDecoderLayerRounding, fusedDecoderStackRounding: args.fusedDecoderStackRounding, fusedPleRounding: args.fusedPleRounding, fusedPlePreludeRounding: args.fusedPlePreludeRounding, fusedTokenForwardRounding: args.fusedTokenForwardRounding, residentGeneration: args.residentGeneration, finalHeadCompute: args.finalHeadCompute, nativeAttentionRounding: args.nativeAttentionRounding, fusedAttentionRounding: args.fusedAttentionRounding, threads: args.threads, maxReadMiB: args.maxReadBytes / (1024 * 1024), finalHeadReadMiB: args.finalHeadMaxReadBytes / (1024 * 1024), ...(vectorizedRealLowering ? { vectorizedRealLowering } : {}) })}\n`);

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  let request: { id?: unknown; inputIds?: unknown; maxNewTokens?: unknown; sessionId?: unknown; stream?: unknown; verificationFastPath?: unknown } = {};
  try {
    request = JSON.parse(line) as typeof request;
    const inputIds = validateIds(request.inputIds), maxNewTokens = validateTokens(request.maxNewTokens);
    const sessionId = validateSessionId(request.sessionId), stream = validateStream(request.stream), verificationFastPath = validateVerificationFastPath(request.verificationFastPath, maxNewTokens);
    const report = verificationFastPath
      ? await generateSelectiveVerification(inputIds, maxNewTokens, verificationFastPath)
      : await generate(inputIds, maxNewTokens, sessionId, stream ? (event) => process.stdout.write(`${JSON.stringify({ id: request.id, event: { type: "token", ...event } })}\n`) : undefined);
    process.stdout.write(`${JSON.stringify({ id: request.id, report })}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ id: request.id, error: error instanceof Error ? error.message : String(error) })}\n`);
  }
}
await linear.close(); await pool.close(); await artifact.close();

async function generate(inputIds: number[], maxNewTokens: number, sessionId?: number, onToken?: (event: Record<string, unknown>) => void): Promise<Record<string, unknown>> {
  const dispatchesBefore = linear.dispatchMetrics();
  const started = performance.now(), generatedTokenIds: number[] = [], steps: Array<Record<string, unknown>> = [];
  if (args.residentGeneration === "on") {
    const generated = await generateGemma4PagedTextLiteralNativeF32(artifact, { inputIds: [inputIds], maxNewTokens }, options, 5, { ...(sessionId === undefined ? {} : { sessionId }), ...(onToken === undefined ? {} : { onToken: (event) => onToken({ step: event.step, tokenId: event.tokenId, forwardSeconds: event.forwardSeconds, topLogits: Array.from(event.topTokenIds, (tokenId, rank) => ({ tokenId, value: event.topLogits[rank]! })) }) }) });
    const elapsedSeconds = (performance.now() - started) / 1000, dispatchesAfter = linear.dispatchMetrics();
    for (let step = 0; step < generated.generatedTokenIds.length; step += 1) {
      generatedTokenIds.push(generated.generatedTokenIds[step]!);
      steps.push({ step, tokenId: generated.generatedTokenIds[step], contextLength: inputIds.length + step, forwardSeconds: generated.forwardSeconds[step], topLogits: generated.topLogits[step] });
    }
    return buildReport(inputIds, maxNewTokens, generatedTokenIds, steps, generated.terminalLogitsSha256, elapsedSeconds, dispatchesBefore, dispatchesAfter, { residentKvBytes: generated.residentKvBytes, kvCacheTransportBytes: 0, externalForwardRequests: 1, prefixTokensReused: generated.prefixTokensReused, prefillTokensComputed: generated.prefillTokensComputed, sessionCacheHit: generated.sessionCacheHit, cachedContextTokens: generated.cachedContextTokens });
  }
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
  return buildReport(inputIds, maxNewTokens, generatedTokenIds, steps, createHash("sha256").update(terminalBytes).digest("hex"), elapsedSeconds, dispatchesBefore, dispatchesAfter, { externalForwardRequests: generatedTokenIds.length });
}

interface VerificationFastPath {
  generatedTokenIds: number[];
  sensitiveSteps: number[];
  terminalLogitsSha256: string;
}

async function generateSelectiveVerification(inputIds: number[], maxNewTokens: number, fast: VerificationFastPath): Promise<Record<string, unknown>> {
  if (args.linearBackend !== "pytorch" || args.residentGeneration !== "off" || args.fusedTokenForwardRounding !== "off") throw new Error("Verificação seletiva requer backend PyTorch não residente.");
  const dispatchesBefore = linear.dispatchMetrics(), started = performance.now();
  const sensitive = new Set(fast.sensitiveSteps), generatedTokenIds: number[] = [], steps: Array<Record<string, unknown>> = [];
  let forwardStarted = performance.now();
  let current = await executeGemma4PagedTextHiddenLiteralF32(artifact, { inputIds: [inputIds] }, options);
  let forwardSeconds = (performance.now() - forwardStarted) / 1000, terminalLogitsSha256 = fast.terminalLogitsSha256, divergenceStep: number | null = null;
  for (let step = 0; step < maxNewTokens; step += 1) {
    const verify = divergenceStep !== null || sensitive.has(step);
    let tokenId = fast.generatedTokenIds[step]!, topLogits: unknown[] = [], verificationSkipped = true;
    if (verify) {
      const epilogueStarted = performance.now();
      const logits = await executeGemma4PagedTextEpilogueLiteralF32(artifact, step === 0 ? [inputIds] : [[generatedTokenIds.at(-1)!]], current.hidden, options);
      forwardSeconds += (performance.now() - epilogueStarted) / 1000;
      topLogits = rankGemma4TerminalLogits(logits);
      tokenId = selectGemma4LiteralGenerationToken(artifact.generation.controlProgram, logits);
      verificationSkipped = false;
      const terminalBytes = Buffer.from(logits.values.buffer, logits.values.byteOffset, logits.values.byteLength);
      terminalLogitsSha256 = createHash("sha256").update(terminalBytes).digest("hex");
      if (divergenceStep === null && tokenId !== fast.generatedTokenIds[step]) divergenceStep = step;
    }
    generatedTokenIds.push(tokenId);
    steps.push({ step, tokenId, contextLength: inputIds.length + step, forwardSeconds, topLogits, verificationSkipped });
    if (step + 1 === maxNewTokens) break;
    forwardStarted = performance.now();
    current = await executeGemma4PagedTextHiddenLiteralF32(artifact, { inputIds: [[tokenId]], positionIds: [[inputIds.length + step]], pastKeyValues: current.pastKeyValues }, options);
    forwardSeconds = (performance.now() - forwardStarted) / 1000;
  }
  const elapsedSeconds = (performance.now() - started) / 1000, dispatchesAfter = linear.dispatchMetrics();
  return buildReport(inputIds, maxNewTokens, generatedTokenIds, steps, terminalLogitsSha256, elapsedSeconds, dispatchesBefore, dispatchesAfter, {
    selectiveVerification: true, sensitiveSteps: fast.sensitiveSteps, trustedFastPathSteps: steps.filter((step) => step.verificationSkipped === true).length,
    verificationHeadSteps: steps.filter((step) => step.verificationSkipped === false).length, verificationDivergenceStep: divergenceStep, externalForwardRequests: generatedTokenIds.length,
  });
}

function buildReport(inputIds: number[], maxNewTokens: number, generatedTokenIds: number[], steps: Array<Record<string, unknown>>, terminalLogitsSha256: string, elapsedSeconds: number, dispatchesBefore: ReturnType<typeof linear.dispatchMetrics>, dispatchesAfter: ReturnType<typeof linear.dispatchMetrics>, transport: Record<string, unknown>): Record<string, unknown> {
  return {
    kind: "gemma4-paged-binary-native-generation", schemaVersion: 3, backend: "paged-binary-native", linearBackend: linear.backend,
    sourceCheckpointAccessed: false, compatibilityBinaryWeightsAccessed: true,
    inputIds, maxNewTokens, generatedTokenIds, fullTokenIds: [...inputIds, ...generatedTokenIds], steps,
    terminalLogitsSha256,
    elapsedSeconds, tokensPerSecond: maxNewTokens / elapsedSeconds, linearThreads: args.threads,
    residentGeneration: args.residentGeneration,
    ...transport,
    maxReadMiB: args.maxReadBytes / (1024 * 1024),
    finalHeadReadMiB: args.finalHeadMaxReadBytes / (1024 * 1024),
    linearReferenceDispatches: dispatchesAfter.referenceDispatches - dispatchesBefore.referenceDispatches,
    wholeNativeBf16Dispatches: dispatchesAfter.wholeNativeBf16Dispatches - dispatchesBefore.wholeNativeBf16Dispatches,
    streamedNativeBf16Dispatches: dispatchesAfter.streamedNativeBf16Dispatches - dispatchesBefore.streamedNativeBf16Dispatches,
    linearBatchDispatches: dispatchesAfter.batchDispatches - dispatchesBefore.batchDispatches,
    linearBatchedProjectionTiles: dispatchesAfter.batchedProjectionTiles - dispatchesBefore.batchedProjectionTiles,
    fusedMlpRounding: args.fusedMlpRounding,
    fusedMlpDispatches: dispatchesAfter.fusedMlpDispatches - dispatchesBefore.fusedMlpDispatches,
    fusedFfnRounding: args.fusedFfnRounding,
    fusedFfnDispatches: dispatchesAfter.fusedFfnDispatches - dispatchesBefore.fusedFfnDispatches,
    fusedDecoderLayerRounding: args.fusedDecoderLayerRounding,
    fusedDecoderLayerDispatches: dispatchesAfter.fusedDecoderLayerDispatches - dispatchesBefore.fusedDecoderLayerDispatches,
    fusedDecoderStackRounding: args.fusedDecoderStackRounding,
    ...(vectorizedRealLowering ? { vectorizedRealLowering } : {}),
    ...(vectorizedRealExecutionGuard?.summary() ? { vectorizedRealExecution: vectorizedRealExecutionGuard.summary() } : {}),
    fusedDecoderStackDispatches: dispatchesAfter.fusedDecoderStackDispatches - dispatchesBefore.fusedDecoderStackDispatches,
    fusedDecoderStackEpilogueDispatches: dispatchesAfter.fusedDecoderStackEpilogueDispatches - dispatchesBefore.fusedDecoderStackEpilogueDispatches,
    fusedTokenForwardRounding: args.fusedTokenForwardRounding,
    fusedTokenForwardDispatches: dispatchesAfter.fusedTokenForwardDispatches - dispatchesBefore.fusedTokenForwardDispatches,
    fusedTokenGenerationDispatches: dispatchesAfter.fusedTokenGenerationDispatches - dispatchesBefore.fusedTokenGenerationDispatches,
    fusedDecoderStackGateUpPairs: dispatchesAfter.fusedDecoderStackGateUpPairs - dispatchesBefore.fusedDecoderStackGateUpPairs,
    fusedDecoderStackWidenedCacheHits: dispatchesAfter.fusedDecoderStackWidenedCacheHits - dispatchesBefore.fusedDecoderStackWidenedCacheHits,
    widenedTensorCacheEntries: dispatchesAfter.widenedTensorCacheEntries,
    widenedTensorCacheBytes: dispatchesAfter.widenedTensorCacheBytes,
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
    streamedNativeBf16Seconds: dispatchesAfter.streamedNativeBf16Seconds - dispatchesBefore.streamedNativeBf16Seconds,
    batchSeconds: dispatchesAfter.batchSeconds - dispatchesBefore.batchSeconds,
    fusedMlpSeconds: dispatchesAfter.fusedMlpSeconds - dispatchesBefore.fusedMlpSeconds,
    fusedFfnSeconds: dispatchesAfter.fusedFfnSeconds - dispatchesBefore.fusedFfnSeconds,
    fusedDecoderLayerSeconds: dispatchesAfter.fusedDecoderLayerSeconds - dispatchesBefore.fusedDecoderLayerSeconds,
    fusedDecoderStackSeconds: dispatchesAfter.fusedDecoderStackSeconds - dispatchesBefore.fusedDecoderStackSeconds,
    fusedDecoderStackAttentionSeconds: dispatchesAfter.fusedDecoderStackAttentionSeconds - dispatchesBefore.fusedDecoderStackAttentionSeconds,
    fusedDecoderStackFfnSeconds: dispatchesAfter.fusedDecoderStackFfnSeconds - dispatchesBefore.fusedDecoderStackFfnSeconds,
    fusedDecoderStackPleSeconds: dispatchesAfter.fusedDecoderStackPleSeconds - dispatchesBefore.fusedDecoderStackPleSeconds,
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
function validateSessionId(value: unknown): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || (value as number) < 1 || (value as number) > 0xffff_ffff) throw new Error("sessionId deve ser inteiro entre 1 e 4.294.967.295.");
  return value as number;
}
function validateStream(value: unknown): boolean {
  if (value === undefined) return false;
  if (typeof value !== "boolean") throw new Error("stream deve ser booleano.");
  return value;
}
function validateVerificationFastPath(value: unknown, maxNewTokens: number): VerificationFastPath | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("verificationFastPath deve ser objeto.");
  const candidate = value as Record<string, unknown>;
  const generatedTokenIds = candidate.generatedTokenIds;
  const sensitiveSteps = candidate.sensitiveSteps;
  const terminalLogitsSha256 = candidate.terminalLogitsSha256;
  if (!Array.isArray(generatedTokenIds) || generatedTokenIds.length !== maxNewTokens || generatedTokenIds.some((token) => !Number.isSafeInteger(token) || (token as number) < 0)) throw new Error("verificationFastPath.generatedTokenIds deve conter exatamente maxNewTokens inteiros não negativos.");
  if (!Array.isArray(sensitiveSteps) || sensitiveSteps.length === 0 || sensitiveSteps.some((step) => !Number.isSafeInteger(step) || (step as number) < 0 || (step as number) >= maxNewTokens)) throw new Error("verificationFastPath.sensitiveSteps deve conter passos válidos.");
  const normalizedSteps = [...sensitiveSteps] as number[];
  if (normalizedSteps.some((step, index) => index > 0 && step <= normalizedSteps[index - 1]!)) throw new Error("verificationFastPath.sensitiveSteps deve estar ordenado e sem duplicatas.");
  if (typeof terminalLogitsSha256 !== "string" || !/^[0-9a-f]{64}$/.test(terminalLogitsSha256)) throw new Error("verificationFastPath.terminalLogitsSha256 deve ser SHA-256 hexadecimal minúsculo.");
  return { generatedTokenIds: generatedTokenIds as number[], sensitiveSteps: normalizedSteps, terminalLogitsSha256 };
}
function parseArguments(argv: string[]): { artifact: string; binaryPool: string; realLoweringPlan?: string; python: string; linearHelper: string; mlxHelper: string; linearBackend: "pytorch" | "mlx"; fusedMlpRounding: "off" | "bf16" | "real" | "native-bf16"; fusedFfnRounding: "off" | "native-bf16"; fusedDecoderLayerRounding: "off" | "native-bf16"; fusedDecoderStackRounding: "off" | "real" | "native-bf16" | "native-bf16-ple"; fusedPleRounding: "off" | "bf16" | "real"; fusedPlePreludeRounding: "off" | "bf16" | "real"; fusedTokenForwardRounding: "off" | "bf16"; residentGeneration: "off" | "on"; finalHeadCompute: "f32" | "native-bf16" | "native-bf16-stream" | "native-bf16-whole"; nativeAttentionRounding: "off" | "bf16" | "real"; fusedAttentionRounding: "off" | "bf16" | "real" | "native-bf16"; threads: number; maxReadBytes: number; finalHeadMaxReadBytes: number } {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 2) { const flag = argv[index], value = argv[index + 1]; if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`); values.set(flag, value); }
  const known = new Set(["--artifact", "--binary-pool", "--real-lowering-plan", "--python", "--linear-helper", "--mlx-helper", "--linear-backend", "--fused-mlp", "--fused-ffn", "--fused-decoder-layer", "--fused-decoder-stack", "--fused-ple", "--fused-ple-prelude", "--fused-token-forward", "--resident-generation", "--final-head", "--final-head-read-mib", "--native-attention", "--fused-attention", "--threads", "--max-read-mib"]); for (const flag of values.keys()) if (!known.has(flag)) throw new Error(`Flag desconhecida: ${flag}.`);
  const required = (flag: string): string => { const value = values.get(flag); if (!value) throw new Error(`${flag} é obrigatório.`); return resolve(value); };
  const threads = Number(values.get("--threads") ?? "10"), maxReadMiB = Number(values.get("--max-read-mib") ?? "16");
  const linearBackend = values.get("--linear-backend") ?? "mlx"; if (linearBackend !== "pytorch" && linearBackend !== "mlx") throw new Error("--linear-backend deve ser pytorch ou mlx.");
  const finalHeadReadMiB = Number(values.get("--final-head-read-mib") ?? (linearBackend === "pytorch" ? "32" : String(maxReadMiB)));
  if (!Number.isSafeInteger(threads) || threads < 1 || threads > 256 || !Number.isSafeInteger(maxReadMiB) || maxReadMiB < 1 || maxReadMiB > 1024 || !Number.isSafeInteger(finalHeadReadMiB) || finalHeadReadMiB < 1 || finalHeadReadMiB > 1024) throw new Error("threads/max-read-mib/final-head-read-mib inválidos.");
  const fusedMlpRounding = values.get("--fused-mlp") ?? (linearBackend === "pytorch" ? "native-bf16" : "real"); if (fusedMlpRounding !== "off" && fusedMlpRounding !== "bf16" && fusedMlpRounding !== "real" && fusedMlpRounding !== "native-bf16") throw new Error("--fused-mlp deve ser off, bf16, real ou native-bf16.");
  if (linearBackend === "mlx" && fusedMlpRounding === "native-bf16") throw new Error("--fused-mlp native-bf16 requer --linear-backend pytorch.");
  const fusedFfnRounding = values.get("--fused-ffn") ?? (linearBackend === "pytorch" ? "native-bf16" : "off"); if (fusedFfnRounding !== "off" && fusedFfnRounding !== "native-bf16") throw new Error("--fused-ffn deve ser off ou native-bf16.");
  if (linearBackend === "mlx" && fusedFfnRounding !== "off") throw new Error("--fused-ffn requer --linear-backend pytorch.");
  const fusedDecoderLayerRounding = values.get("--fused-decoder-layer") ?? (linearBackend === "pytorch" ? "native-bf16" : "off"); if (fusedDecoderLayerRounding !== "off" && fusedDecoderLayerRounding !== "native-bf16") throw new Error("--fused-decoder-layer deve ser off ou native-bf16.");
  if (linearBackend === "mlx" && fusedDecoderLayerRounding !== "off") throw new Error("--fused-decoder-layer requer --linear-backend pytorch.");
  const fusedDecoderStackRounding = values.get("--fused-decoder-stack") ?? (linearBackend === "mlx" ? "real" : "native-bf16"); if (fusedDecoderStackRounding !== "off" && fusedDecoderStackRounding !== "real" && fusedDecoderStackRounding !== "native-bf16" && fusedDecoderStackRounding !== "native-bf16-ple") throw new Error("--fused-decoder-stack deve ser off, real, native-bf16 ou native-bf16-ple.");
  if (linearBackend === "mlx" && fusedDecoderStackRounding === "native-bf16-ple") throw new Error("--fused-decoder-stack native-bf16-ple requer --linear-backend pytorch.");
  if (linearBackend !== "mlx" && fusedDecoderStackRounding === "real") throw new Error("--fused-decoder-stack real requer --linear-backend mlx.");
  const fusedPleRounding = values.get("--fused-ple") ?? (linearBackend === "pytorch" ? "bf16" : "off"); if (fusedPleRounding !== "off" && fusedPleRounding !== "bf16" && fusedPleRounding !== "real") throw new Error("--fused-ple deve ser off, bf16 ou real.");
  if (linearBackend === "mlx" && fusedPleRounding !== "off") throw new Error("--fused-ple requer --linear-backend pytorch.");
  const fusedPlePreludeRounding = values.get("--fused-ple-prelude") ?? (linearBackend === "mlx" ? "bf16" : "off"); if (fusedPlePreludeRounding !== "off" && fusedPlePreludeRounding !== "bf16" && fusedPlePreludeRounding !== "real") throw new Error("--fused-ple-prelude deve ser off, bf16 ou real.");
  const fusedTokenForwardRounding = values.get("--fused-token-forward") ?? (linearBackend === "mlx" && fusedDecoderStackRounding !== "off" ? "bf16" : "off"); if (fusedTokenForwardRounding !== "off" && fusedTokenForwardRounding !== "bf16") throw new Error("--fused-token-forward deve ser off ou bf16.");
  if (linearBackend !== "mlx" && fusedTokenForwardRounding !== "off") throw new Error("--fused-token-forward requer --linear-backend mlx.");
  if (fusedDecoderStackRounding === "off" && fusedTokenForwardRounding !== "off") throw new Error("--fused-token-forward requer --fused-decoder-stack habilitado.");
  const residentGeneration = values.get("--resident-generation") ?? (fusedTokenForwardRounding === "bf16" ? "on" : "off"); if (residentGeneration !== "off" && residentGeneration !== "on") throw new Error("--resident-generation deve ser off ou on.");
  if (residentGeneration === "on" && (linearBackend !== "mlx" || fusedTokenForwardRounding !== "bf16")) throw new Error("--resident-generation on requer token forward MLX bf16.");
  const finalHeadCompute = values.get("--final-head") ?? (linearBackend === "pytorch" ? "native-bf16-stream" : "native-bf16-whole"); if (finalHeadCompute !== "f32" && finalHeadCompute !== "native-bf16" && finalHeadCompute !== "native-bf16-stream" && finalHeadCompute !== "native-bf16-whole") throw new Error("--final-head deve ser f32, native-bf16, native-bf16-stream ou native-bf16-whole.");
  if (linearBackend === "mlx" && finalHeadCompute !== "f32" && finalHeadCompute !== "native-bf16-whole") throw new Error("--final-head no backend MLX deve ser f32 ou native-bf16-whole.");
  const nativeAttentionRounding = values.get("--native-attention") ?? (linearBackend === "pytorch" ? "real" : "off"); if (nativeAttentionRounding !== "off" && nativeAttentionRounding !== "bf16" && nativeAttentionRounding !== "real") throw new Error("--native-attention deve ser off, bf16 ou real.");
  if (linearBackend === "mlx" && nativeAttentionRounding !== "off") throw new Error("--native-attention requer --linear-backend pytorch.");
  const fusedAttentionRounding = values.get("--fused-attention") ?? (linearBackend === "pytorch" ? "native-bf16" : "off"); if (fusedAttentionRounding !== "off" && fusedAttentionRounding !== "bf16" && fusedAttentionRounding !== "real" && fusedAttentionRounding !== "native-bf16") throw new Error("--fused-attention deve ser off, bf16, real ou native-bf16.");
  if (linearBackend === "mlx" && fusedAttentionRounding !== "off") throw new Error("--fused-attention requer --linear-backend pytorch.");
  const binaryPool = required("--binary-pool");
  const realLoweringPlan = fusedDecoderStackRounding === "real" ? resolve(values.get("--real-lowering-plan") ?? join(binaryPool, "vectorized-real-lowering.json")) : undefined;
  return { artifact: required("--artifact"), binaryPool, ...(realLoweringPlan ? { realLoweringPlan } : {}), python: values.get("--python") ?? resolve("venv/bin/python"), linearHelper: resolve(values.get("--linear-helper") ?? "scripts/gemma4-paged-linear-worker.py"), mlxHelper: resolve(values.get("--mlx-helper") ?? "scripts/gemma4-mlx-linear-worker.py"), linearBackend, fusedMlpRounding, fusedFfnRounding, fusedDecoderLayerRounding, fusedDecoderStackRounding, fusedPleRounding, fusedPlePreludeRounding, fusedTokenForwardRounding, residentGeneration, finalHeadCompute, nativeAttentionRounding, fusedAttentionRounding, threads, maxReadBytes: maxReadMiB * 1024 * 1024, finalHeadMaxReadBytes: finalHeadReadMiB * 1024 * 1024 };
}
