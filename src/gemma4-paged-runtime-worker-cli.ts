import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { createInterface } from "node:readline";
import { planGemma4SelectiveVerificationTail } from "./gemma4-selective-verification.js";
import { Gemma4VerificationSessionCache } from "./gemma4-verification-session-cache.js";
import { join, resolve } from "node:path";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { assertGemma4VectorizedRealLoweringPlanMatchesRuntime, openGemma4PagedRuntimeArtifact } from "./gemma4-paged-runtime-index.js";
import { Gemma4BinaryConstantPool } from "./gemma4-binary-constant-pool.js";
import { Gemma4PagedNativeLinearWorker, normalizeGemma4DecoderQuantizationLayers, type Gemma4MlxDecoderQuantization } from "./gemma4-paged-native-linear.js";
import { executeGemma4PagedTextEpilogueLiteralF32, executeGemma4PagedTextHiddenLiteralF32, executeGemma4PagedTextLiteralF32, generateGemma4PagedTextLiteralNativeF32, selectGemma4PagedTerminalHidden, type Gemma4PagedTextHiddenResult } from "./gemma4-paged-text.js";
import { selectGemma4LiteralGenerationToken } from "./gemma4-literal-generation-control.js";
import { rankGemma4TerminalLogits } from "./gemma4-terminal-logits.js";
import { assertGemma4VectorizedRealLoweringPlanMatches, Gemma4VectorizedRealExecutionGuard, type Gemma4VectorizedRealLoweringPlan } from "./gemma4-vectorized-real-lowering.js";
import { validateGemma4FinalFormulaMapFile, type Gemma4FinalFormulaMapValidation } from "./gemma4-final-formula-map.js";
import { validateGemma4FinalFormulaRuntimeFile } from "./gemma4-final-formula-runtime.js";
import type { Operation } from "./types.js";

const args = parseArguments(process.argv.slice(2));
const initializationStarted = performance.now();
const artifact = args.artifactIndex
  ? await openGemma4PagedRuntimeArtifact(args.artifact, args.artifactIndex)
  : await openGemma4CompositeLiteralArtifact(args.artifact);
const pool = await Gemma4BinaryConstantPool.open(args.binaryPool);
let vectorizedRealLowering: Gemma4VectorizedRealLoweringPlan["contract"] | undefined;
let vectorizedRealLoweringSha256: string | undefined;
let vectorizedRealExecutionGuard: Gemma4VectorizedRealExecutionGuard | undefined;
let vectorizedRealCompiledOutputOperations: readonly Operation[] | undefined;
if (args.fusedDecoderStackRounding === "real") {
  const planInfo = await stat(args.realLoweringPlan!);
  if (!planInfo.isFile() || planInfo.size < 1 || planInfo.size > 16 * 1024 * 1024) throw new Error("Plano de lowering real persistido deve ter entre 1 byte e 16 MiB.");
  const persistedBytes = await readFile(args.realLoweringPlan!);
  vectorizedRealLoweringSha256 = createHash("sha256").update(persistedBytes).digest("hex");
  const persisted = JSON.parse(persistedBytes.toString("utf8")) as unknown;
  if ("runtimeIndexSchemaVersion" in artifact) {
    assertGemma4VectorizedRealLoweringPlanMatchesRuntime(persisted, createHash("sha256").update(persistedBytes).digest("hex"), artifact);
  } else {
    assertGemma4VectorizedRealLoweringPlanMatches(persisted, artifact.realSimplifiedProgram, artifact.calculationGraph, artifact.integrityManifest);
  }
  vectorizedRealLowering = persisted.contract;
  vectorizedRealExecutionGuard = new Gemma4VectorizedRealExecutionGuard(persisted);
  const availableOutputOperations = [...artifact.program.textProgram.layers.flatMap((layer) => layer.operations), ...artifact.program.textProgram.epilogue];
  const operationsById = new Map(availableOutputOperations.map((operation) => [operation.id, operation]));
  if (operationsById.size !== availableOutputOperations.length || availableOutputOperations.length !== persisted.functionBindings.length) {
    throw new Error("Programa textual runtime não corresponde um-a-um à closure compilada persistida.");
  }
  const compiledOutputOperations = persisted.functionBindings.map((binding) => {
    const operation = operationsById.get(binding.operationId);
    if (!operation) throw new Error(`${binding.operationId}: operação do programa de saída compilado ausente no runtime.`);
    return operation;
  });
  vectorizedRealCompiledOutputOperations = vectorizedRealExecutionGuard.bindCompiledOutputProgram(compiledOutputOperations);
}
let finalFormulaProgram: (Gemma4FinalFormulaMapValidation & { execution: "vectorized-shared-dag-output-program"; evaluator: "EVAL_EXACT_DAG"; runtimeContractSha256: string }) | undefined;
if (args.finalFormulaMap) {
  const validated = await validateGemma4FinalFormulaMapFile(args.finalFormulaMap.path, args.finalFormulaMap);
  if (!args.finalFormulaRuntime) throw new Error("Mapa final requer runtime EVAL_EXACT_DAG autenticado.");
  const runtime = await validateGemma4FinalFormulaRuntimeFile(args.finalFormulaRuntime.path, args.finalFormulaRuntime.fileSha256);
  const terminalLogits = vectorizedRealLowering?.source.outputFamilies.terminal_logit?.dimensions;
  if (!vectorizedRealExecutionGuard || terminalLogits !== validated.functions || runtime.output.functions !== validated.functions ||
    runtime.artifacts.formulaMap.sha256 !== validated.fileSha256 || runtime.artifacts.formulaMap.orderedRootsSha256 !== validated.orderedRootsSha256 ||
    runtime.artifacts.loweringPlan.sha256 !== vectorizedRealLoweringSha256 || runtime.artifacts.loweringPlan.functionBindingsSha256 !== vectorizedRealLowering?.functionBindingsSha256 ||
    runtime.artifacts.loweringPlan.outputBindingsSha256 !== vectorizedRealLowering?.source.outputBindingsSha256 || runtime.artifacts.loweringPlan.realSimplifiedProgramSha256 !== vectorizedRealLowering?.source.realSimplifiedProgramSha256 ||
    JSON.stringify(runtime.execution.compiledOutputProgram) !== JSON.stringify(vectorizedRealLowering?.compiledOutputProgram) ||
    (args.artifactIndex && runtime.artifacts.constantPool.sha256 !== args.artifactIndex.constantPoolSha256)) throw new Error("Runtime EVAL_EXACT_DAG não corresponde ao mapa, constantes e programa vetorial executável.");
  finalFormulaProgram = { ...validated, execution: "vectorized-shared-dag-output-program", evaluator: "EVAL_EXACT_DAG", runtimeContractSha256: runtime.fileSha256 };
}
const createLinear = () => new Gemma4PagedNativeLinearWorker({ python: args.python, helper: args.linearHelper, threads: args.threads, binaryPool: args.binaryPool, storageTensors: pool.catalog.tensors, backend: args.linearBackend, mlxHelper: args.mlxHelper, mlxHeadQuantization: args.mlxHeadQuantization, mlxDecoderQuantization: args.mlxDecoderQuantization, ...(args.mlxDecoderQuantizationLayers === undefined ? {} : { mlxDecoderQuantizationLayers: args.mlxDecoderQuantizationLayers }) });
let linear = createLinear();
const createExecutionOptions = () => ({
  maxReadBytes: args.maxReadBytes, finalHeadMaxReadBytes: args.finalHeadMaxReadBytes,
  allowUnverifiedFidelity: true, tensorReader: pool, linearTileKernel: linear, finalHeadCompute: args.finalHeadCompute,
  ...(vectorizedRealExecutionGuard && vectorizedRealCompiledOutputOperations ? { vectorizedRealExecutionGuard, vectorizedRealCompiledOutputOperations } : {}),
  ...(args.fusedMlpRounding === "off" ? {} : { fusedMlpRounding: args.fusedMlpRounding }), ...(args.fusedFfnRounding === "off" ? {} : { fusedFfnRounding: args.fusedFfnRounding }),
  ...(args.fusedDecoderLayerRounding === "off" ? {} : { fusedDecoderLayerRounding: args.fusedDecoderLayerRounding }), ...(args.fusedDecoderStackRounding === "off" ? {} : { fusedDecoderStackRounding: args.fusedDecoderStackRounding }),
  ...(args.fusedPleRounding === "off" ? {} : { fusedPleRounding: args.fusedPleRounding }), ...(args.fusedPlePreludeRounding === "off" ? {} : { fusedPlePreludeRounding: args.fusedPlePreludeRounding }),
  ...(args.fusedTokenForwardRounding === "off" ? {} : { fusedTokenForwardRounding: args.fusedTokenForwardRounding }), ...(args.nativeAttentionRounding === "off" ? {} : { nativeAttentionRounding: args.nativeAttentionRounding }),
  ...(args.fusedAttentionRounding === "off" ? {} : { fusedAttentionRounding: args.fusedAttentionRounding }),
});
let options = createExecutionOptions();
type VerificationCachedState = Pick<Gemma4PagedTextHiddenResult, "hidden" | "pastKeyValues">;
const verificationSessionCache = new Gemma4VerificationSessionCache<VerificationCachedState>(8);
process.stdout.write(`${JSON.stringify({ ready: true, initializationSeconds: (performance.now() - initializationStarted) / 1000, artifactOpenStrategy: "runtimeIndexSchemaVersion" in artifact ? "authenticated-execution-index-v2" : args.artifactIndex ? "authenticated-runtime-index-v1" : "streamed-literal-scan-v1", ...(args.artifactIndex ? { artifactIndexSha256: args.artifactIndex.sha256 } : {}), backend: "paged-binary-native", linearBackend: linear.backend, mlxHeadQuantization: args.mlxHeadQuantization, fusedMlpRounding: args.fusedMlpRounding, fusedFfnRounding: args.fusedFfnRounding, fusedDecoderLayerRounding: args.fusedDecoderLayerRounding, fusedDecoderStackRounding: args.fusedDecoderStackRounding, fusedPleRounding: args.fusedPleRounding, fusedPlePreludeRounding: args.fusedPlePreludeRounding, fusedTokenForwardRounding: args.fusedTokenForwardRounding, residentGeneration: args.residentGeneration, finalHeadCompute: args.finalHeadCompute, nativeAttentionRounding: args.nativeAttentionRounding, fusedAttentionRounding: args.fusedAttentionRounding, threads: args.threads, maxReadMiB: args.maxReadBytes / (1024 * 1024), finalHeadReadMiB: args.finalHeadMaxReadBytes / (1024 * 1024), ...(vectorizedRealLowering ? { vectorizedRealLowering } : {}), ...(finalFormulaProgram ? { finalFormulaProgram } : {}) })}\n`);

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  let request: { id?: unknown; control?: unknown; inputIds?: unknown; maxNewTokens?: unknown; eosTokenId?: unknown; sessionId?: unknown; stream?: unknown; verificationFastPath?: unknown; verificationControl?: unknown; captureLayerHidden?: unknown; captureLayerStages?: unknown } = {};
  try {
    request = JSON.parse(line) as typeof request;
    if (request.control === "trim-memory") {
      if (args.linearBackend !== "pytorch") throw new Error("trim-memory requer backend PyTorch.");
      await linear.close();
      linear = createLinear();
      options = createExecutionOptions();
      process.stdout.write(`${JSON.stringify({ id: request.id, report: { trimmed: true, strategy: "restart-linear-kernel", processRssBytes: process.memoryUsage().rss, processMaxRssKiB: process.resourceUsage().maxRSS } })}\n`);
      continue;
    }
    if (request.control !== undefined) throw new Error("Controle de worker desconhecido.");
    const inputIds = validateIds(request.inputIds), maxNewTokens = validateTokens(request.maxNewTokens), eosTokenId = validateEosTokenId(request.eosTokenId);
    const sessionId = validateSessionId(request.sessionId), stream = validateStream(request.stream), verificationFastPath = validateVerificationFastPath(request.verificationFastPath, maxNewTokens), verificationControl = validateVerificationControl(request.verificationControl), captureLayerHidden = validateCaptureLayerHidden(request.captureLayerHidden), captureLayerStages = validateCaptureLayerStages(request.captureLayerStages);
    if (verificationFastPath && verificationControl) throw new Error("verificationFastPath e verificationControl são mutuamente exclusivos.");
    if (verificationControl && (args.linearBackend !== "mlx" || args.mlxDecoderQuantization !== "q8-ffn-gate-up-down" || sessionId !== undefined || stream)) throw new Error("verificationControl requer decoder MLX q8-ffn-gate-up-down sem sessão nem streaming.");
    if ((captureLayerHidden || captureLayerStages !== undefined) && (!verificationFastPath || args.linearBackend !== "pytorch" || sessionId !== undefined || stream)) throw new Error("Captura decoder requer verificationFastPath no backend PyTorch sem sessão nem streaming.");
    const report = verificationFastPath
      ? await generateSelectiveVerification(inputIds, maxNewTokens, verificationFastPath, eosTokenId, sessionId, captureLayerHidden, captureLayerStages)
      : await generate(inputIds, maxNewTokens, eosTokenId, sessionId, stream ? (event) => process.stdout.write(`${JSON.stringify({ id: request.id, event: { type: "token", ...event } })}\n`) : undefined, verificationControl);
    process.stdout.write(`${JSON.stringify({ id: request.id, report })}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ id: request.id, error: error instanceof Error ? error.message : String(error) })}\n`);
  }
}
await linear.close(); await pool.close(); await artifact.close();

async function generate(inputIds: number[], maxNewTokens: number, eosTokenId?: number, sessionId?: number, onToken?: (event: Record<string, unknown>) => void, controlDecoder = false): Promise<Record<string, unknown>> {
  const dispatchesBefore = linear.dispatchMetrics();
  const started = performance.now(), generatedTokenIds: number[] = [], steps: Array<Record<string, unknown>> = [];
  if (args.residentGeneration === "on") {
    const generated = await generateGemma4PagedTextLiteralNativeF32(artifact, { inputIds: [inputIds], maxNewTokens, ...(eosTokenId === undefined ? {} : { eosTokenId }) }, options, 5, { ...(sessionId === undefined ? {} : { sessionId }), ...(controlDecoder ? { controlDecoder: true } : {}), ...(onToken === undefined ? {} : { onToken: (event) => onToken({ step: event.step, tokenId: event.tokenId, forwardSeconds: event.forwardSeconds, topLogits: Array.from(event.topTokenIds, (tokenId, rank) => ({ tokenId, value: event.topLogits[rank]! })) }) }) });
    const elapsedSeconds = (performance.now() - started) / 1000, dispatchesAfter = linear.dispatchMetrics();
    for (let step = 0; step < generated.generatedTokenIds.length; step += 1) {
      generatedTokenIds.push(generated.generatedTokenIds[step]!);
      steps.push({ step, tokenId: generated.generatedTokenIds[step], contextLength: inputIds.length + step, forwardSeconds: generated.forwardSeconds[step], topLogits: generated.topLogits[step] });
    }
    return buildReport(inputIds, maxNewTokens, generatedTokenIds, steps, generated.terminalLogitsSha256, elapsedSeconds, dispatchesBefore, dispatchesAfter, { terminalLogitMaterializations: generated.terminalLogitMaterializations, gpuRankedTokenSteps: generated.gpuRankedTokenSteps, fullLogitTransfersAvoided: generated.fullLogitTransfersAvoided, terminalLogitVectorBytes: generated.terminalLogitVectorBytes, ropeFactorBuilds: generated.ropeFactorBuilds, ropeFactorBuildsAvoided: generated.ropeFactorBuildsAvoided, topologyMaskBuilds: generated.topologyMaskBuilds, topologyMaskBuildsAvoided: generated.topologyMaskBuildsAvoided, redundantLogitFiniteScansAvoided: generated.redundantLogitFiniteScansAvoided, kvPrefixValidationScansAvoided: generated.kvPrefixValidationScansAvoided, decoderLayerValidityScansAvoided: generated.decoderLayerValidityScansAvoided, compiledIncrementalDecoderSteps: generated.compiledIncrementalDecoderSteps, incrementalCompilerCacheHit: generated.incrementalCompilerCacheHit, residentKvBytes: generated.residentKvBytes, kvCacheTransportBytes: 0, externalForwardRequests: 1, prefixTokensReused: generated.prefixTokensReused, prefillTokensComputed: generated.prefillTokensComputed, sessionCacheHit: generated.sessionCacheHit, cachedContextTokens: generated.cachedContextTokens, quantizedHeadCertifiedSteps: generated.quantizedHeadCertifiedSteps, quantizedHeadExactFallbackSteps: generated.quantizedHeadExactFallbackSteps, tokenPreludeCacheHits: generated.tokenPreludeCacheHits, tokenPreludeCacheMisses: generated.tokenPreludeCacheMisses, prefillSeconds: generated.prefillSeconds, incrementalDecoderSeconds: generated.incrementalDecoderSeconds, tokenPreludeSeconds: generated.tokenPreludeSeconds, compiledDecoderGraphSeconds: generated.compiledDecoderGraphSeconds, tokenSelectionSeconds: generated.tokenSelectionSeconds, terminalLogitTransferSeconds: generated.terminalLogitTransferSeconds });
  }
  let forwardStarted = performance.now();
  let current = await executeGemma4PagedTextLiteralF32(artifact, { inputIds: [inputIds] }, options);
  let forwardSeconds = (performance.now() - forwardStarted) / 1000;
  for (let step = 0; step < maxNewTokens; step += 1) {
    const top = rankGemma4TerminalLogits(current.logits), tokenId = selectGemma4LiteralGenerationToken(artifact.generation.controlProgram, current.logits);
    generatedTokenIds.push(tokenId);
    steps.push({ step, tokenId, contextLength: inputIds.length + step, forwardSeconds, topLogits: top });
    if (tokenId === eosTokenId || step + 1 === maxNewTokens) break;
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

async function generateSelectiveVerification(inputIds: number[], maxNewTokens: number, fast: VerificationFastPath, eosTokenId?: number, sessionId?: number, captureLayerHidden = false, captureLayerStages?: number): Promise<Record<string, unknown>> {
  if (args.linearBackend !== "pytorch" || args.residentGeneration !== "off" || args.fusedTokenForwardRounding !== "off") throw new Error("Verificação seletiva requer backend PyTorch não residente.");
  const dispatchesBefore = linear.dispatchMetrics(), started = performance.now();
  const sensitive = new Set(fast.sensitiveSteps), generatedTokenIds: number[] = [], steps: Array<Record<string, unknown>> = [];
  let forwardStarted = performance.now();
  const layerHiddenCaptures: string[][] = [];
  const layerHiddenFullCaptures: string[][] = [];
  const layerStageCaptures: Array<{ names: readonly string[]; values: string[]; attentionNames?: readonly string[]; attentionValues?: string[]; pleNames?: readonly string[]; pleValues?: string[] }> = [];
  const executionOptions = captureLayerHidden || captureLayerStages !== undefined ? {
    ...options,
    ...(captureLayerHidden ? { onDecoderLayerHidden: (layers: readonly Float32Array[]) => { layerHiddenCaptures.push(layers.map(encodeTerminalHidden)); layerHiddenFullCaptures.push(layers.map(encodeWholeF32)); } } : {}),
    ...(captureLayerStages === undefined ? {} : { decoderLayerStageCapture: {
      layerIndex: captureLayerStages,
      accept: (names: readonly string[], values: readonly Float32Array[]) => layerStageCaptures.push({ names, values: values.map(encodeWholeF32) }),
      acceptAttention: (names: readonly string[], values: readonly Float32Array[]) => Object.assign(layerStageCaptures.at(-1)!, { attentionNames: names, attentionValues: values.map(encodeWholeF32) }),
      acceptPle: (names: readonly string[], values: readonly Float32Array[]) => Object.assign(layerStageCaptures.at(-1)!, { pleNames: names, pleValues: values.map(encodeWholeF32) }),
    } }),
  } : options;
  const cached = sessionId === undefined ? undefined : verificationSessionCache.resolve(sessionId, inputIds);
  const suffixTokenIds = cached?.suffixTokenIds ?? inputIds;
  let current: VerificationCachedState, currentInputIds: number[], verificationDecoderSteps: number;
  if (cached && suffixTokenIds.length === 0) {
    current = cached.state; currentInputIds = cached.currentInputIds; verificationDecoderSteps = 0;
  } else {
    currentInputIds = [...suffixTokenIds]; verificationDecoderSteps = 1;
    current = await executeGemma4PagedTextHiddenLiteralF32(artifact, {
      inputIds: [currentInputIds],
      ...(cached ? { positionIds: [currentInputIds.map((_token, index) => cached.prefixTokensReused + index)], pastKeyValues: cached.state.pastKeyValues } : {}),
    }, executionOptions);
  }
  const currentContextTokens = [...inputIds];
  let verificationEarlyExitStep: number | null = null, verificationDecoderStepsAvoided = 0, verificationHeadPositionsComputed = 0, verificationHeadPositionsAvoided = 0;
  const lastSensitiveStep = fast.sensitiveSteps.at(-1)!;
  let forwardSeconds = (performance.now() - forwardStarted) / 1000, terminalLogitsSha256 = fast.terminalLogitsSha256, divergenceStep: number | null = null;
  for (let step = 0; step < fast.generatedTokenIds.length; step += 1) {
    const verify = divergenceStep !== null || sensitive.has(step);
    let tokenId = fast.generatedTokenIds[step]!, topLogits: unknown[] = [], verificationSkipped = true;
    if (verify) {
      const epilogueStarted = performance.now();
      const terminal = selectGemma4PagedTerminalHidden([currentInputIds], current.hidden);
      const logits = await executeGemma4PagedTextEpilogueLiteralF32(artifact, terminal.inputIds, terminal.hidden, options);
      verificationHeadPositionsComputed += terminal.inputIds.length;
      verificationHeadPositionsAvoided += terminal.positionsAvoided;
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
    if (tokenId === eosTokenId || step + 1 === maxNewTokens) break;
    const trustedTail = planGemma4SelectiveVerificationTail({
      fastGeneratedTokenIds: fast.generatedTokenIds,
      fastTerminalLogitsSha256: fast.terminalLogitsSha256,
      inputLength: inputIds.length,
      currentStep: step,
      lastSensitiveStep,
      divergenceStep,
      captureEnabled: captureLayerHidden || captureLayerStages !== undefined,
    });
    if (trustedTail) {
      generatedTokenIds.push(...trustedTail.generatedTokenIds);
      steps.push(...trustedTail.steps);
      terminalLogitsSha256 = trustedTail.terminalLogitsSha256;
      verificationEarlyExitStep = trustedTail.earlyExitStep;
      verificationDecoderStepsAvoided = trustedTail.decoderStepsAvoided;
      break;
    }
    forwardStarted = performance.now();
    current = await executeGemma4PagedTextHiddenLiteralF32(artifact, { inputIds: [[tokenId]], positionIds: [[inputIds.length + step]], pastKeyValues: current.pastKeyValues }, executionOptions);
    currentInputIds = [tokenId]; currentContextTokens.push(tokenId);
    verificationDecoderSteps += 1;
    forwardSeconds = (performance.now() - forwardStarted) / 1000;
  }
  if (sessionId !== undefined) verificationSessionCache.update(sessionId, currentContextTokens, currentInputIds, { hidden: current.hidden, pastKeyValues: current.pastKeyValues }, verificationStateBytes(current));
  const elapsedSeconds = (performance.now() - started) / 1000, dispatchesAfter = linear.dispatchMetrics();
  return buildReport(inputIds, maxNewTokens, generatedTokenIds, steps, terminalLogitsSha256, elapsedSeconds, dispatchesBefore, dispatchesAfter, {
    selectiveVerification: true, sensitiveSteps: fast.sensitiveSteps, trustedFastPathSteps: steps.filter((step) => step.verificationSkipped === true).length,
    verificationHeadSteps: steps.filter((step) => step.verificationSkipped === false).length, verificationDivergenceStep: divergenceStep, externalForwardRequests: verificationDecoderSteps,
    verificationDecoderSteps, verificationDecoderStepsAvoided,
    verificationHeadPositionsComputed, verificationHeadPositionsAvoided,
    verificationSessionCacheHit: cached !== undefined,
    verificationPrefixTokensReused: cached?.prefixTokensReused ?? 0,
    verificationPrefillTokensComputed: suffixTokenIds.length,
    verificationCachedContextTokens: currentContextTokens.length,
    verificationResidentKvBytes: verificationSessionCache.residentBytes,
    verificationCachedSessions: verificationSessionCache.sessions,
    ...(verificationEarlyExitStep === null ? {} : { verificationEarlyExitStep }),
    ...(captureLayerHidden ? { layerHiddenEncoding: "terminal-token-f32le-base64", layerHiddenCaptures, layerHiddenFullEncoding: "whole-tensor-f32le-base64", layerHiddenFullCaptures } : {}),
    ...(captureLayerStages === undefined ? {} : { layerStageEncoding: "whole-tensor-f32le-base64", layerStageLayer: captureLayerStages, layerStageCaptures }),
  });
}

function verificationStateBytes(state: VerificationCachedState): number {
  let bytes = state.hidden.values.byteLength;
  for (const cache of state.pastKeyValues.values()) bytes += cache.key.values.byteLength + cache.value.values.byteLength;
  return bytes;
}

function encodeTerminalHidden(values: Float32Array): string {
  const hiddenSize = artifact.program.contract.text.hiddenSize;
  const terminal = values.subarray(values.length - hiddenSize);
  return Buffer.from(terminal.buffer, terminal.byteOffset, terminal.byteLength).toString("base64");
}
function encodeWholeF32(values: Float32Array): string {
  return Buffer.from(values.buffer, values.byteOffset, values.byteLength).toString("base64");
}

function buildReport(inputIds: number[], maxNewTokens: number, generatedTokenIds: number[], steps: Array<Record<string, unknown>>, terminalLogitsSha256: string, elapsedSeconds: number, dispatchesBefore: ReturnType<typeof linear.dispatchMetrics>, dispatchesAfter: ReturnType<typeof linear.dispatchMetrics>, transport: Record<string, unknown>): Record<string, unknown> {
  return {
    kind: "gemma4-paged-binary-native-generation", schemaVersion: 3, backend: "paged-binary-native", linearBackend: linear.backend,
    sourceCheckpointAccessed: false, compatibilityBinaryWeightsAccessed: true,
    inputIds, maxNewTokens, generatedTokenIds, fullTokenIds: [...inputIds, ...generatedTokenIds], steps,
    terminalLogitsSha256,
    elapsedSeconds, tokensPerSecond: generatedTokenIds.length / elapsedSeconds, linearThreads: args.threads,
    parallelExecutionBackend: args.linearBackend === "mlx" ? "metal-vectorized-kernels" : "pytorch-cpu-thread-pool",
    configuredHostThreads: args.threads,
    hostThreadSettingApplied: args.linearBackend === "pytorch",
    metalCommandStreams: args.linearBackend === "mlx" ? 1 : 0,
    parallelTerminalOutputDimensions: artifact.program.contract.text.vocabSize,
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
    ...(finalFormulaProgram ? { finalFormulaProgram } : {}),
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
    mlxHeadQuantization: args.mlxHeadQuantization,
    mlxHeadQuantizationStrategy: args.mlxHeadQuantization === "off" ? "exact-bf16" : args.mlxHeadQuantization === "q8-shortlist" ? "single-stage-affine-q8-shortlist-refined-v1" : "two-stage-residual-affine-certified-v1",
    mlxDecoderQuantization: args.mlxDecoderQuantization,
    ...(args.mlxDecoderQuantizationLayers === undefined ? {} : { mlxDecoderQuantizationLayers: args.mlxDecoderQuantizationLayers }),
    mlxDecoderQuantizationStrategy: args.mlxDecoderQuantization === "off" ? "exact-bf16" : args.mlxDecoderQuantization === "q8-ffn-gate-up" && args.mlxDecoderQuantizationLayers === undefined ? "single-stage-affine-q8-calibrated-v1" : args.mlxDecoderQuantization === "q4-ffn-gate-up" ? "experimental-affine-q4" : "experimental-affine-q8",
    mlxDecoderGateUpProjectionStrategy: args.mlxDecoderQuantization === "off" || args.mlxDecoderQuantization === "q8-ffn-down" || args.mlxDecoderQuantization === "q8-attention" ? "separate-projections" : args.mlxDecoderQuantization === "q4-ffn-gate-up" ? "concatenated-affine-q4-v1" : "concatenated-affine-q8-v1",
    mlxDecoderGateUpSplitStrategy: args.mlxDecoderQuantization === "off" || args.mlxDecoderQuantization === "q8-ffn-down" || args.mlxDecoderQuantization === "q8-attention" ? "separate-projections" : "static-index-take-v1",
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
function validateEosTokenId(value: unknown): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || (value as number) < 0) throw new Error("eosTokenId deve ser inteiro não negativo.");
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
function validateCaptureLayerHidden(value: unknown): boolean {
  if (value === undefined) return false;
  if (typeof value !== "boolean") throw new Error("captureLayerHidden deve ser booleano.");
  return value;
}
function validateCaptureLayerStages(value: unknown): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || (value as number) < 0 || (value as number) >= 42) throw new Error("captureLayerStages deve estar entre 0 e 41.");
  return value as number;
}
function validateVerificationControl(value: unknown): boolean {
  if (value === undefined) return false;
  if (typeof value !== "boolean") throw new Error("verificationControl deve ser booleano.");
  return value;
}
function validateVerificationFastPath(value: unknown, maxNewTokens: number): VerificationFastPath | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("verificationFastPath deve ser objeto.");
  const candidate = value as Record<string, unknown>;
  const generatedTokenIds = candidate.generatedTokenIds;
  const sensitiveSteps = candidate.sensitiveSteps;
  const terminalLogitsSha256 = candidate.terminalLogitsSha256;
  if (!Array.isArray(generatedTokenIds) || generatedTokenIds.length < 1 || generatedTokenIds.length > maxNewTokens || generatedTokenIds.some((token) => !Number.isSafeInteger(token) || (token as number) < 0)) throw new Error("verificationFastPath.generatedTokenIds deve conter entre 1 e maxNewTokens inteiros não negativos.");
  if (!Array.isArray(sensitiveSteps) || sensitiveSteps.length === 0 || sensitiveSteps.some((step) => !Number.isSafeInteger(step) || (step as number) < 0 || (step as number) >= generatedTokenIds.length)) throw new Error("verificationFastPath.sensitiveSteps deve conter passos gerados válidos.");
  const normalizedSteps = [...sensitiveSteps] as number[];
  if (normalizedSteps.some((step, index) => index > 0 && step <= normalizedSteps[index - 1]!)) throw new Error("verificationFastPath.sensitiveSteps deve estar ordenado e sem duplicatas.");
  if (typeof terminalLogitsSha256 !== "string" || !/^[0-9a-f]{64}$/.test(terminalLogitsSha256)) throw new Error("verificationFastPath.terminalLogitsSha256 deve ser SHA-256 hexadecimal minúsculo.");
  return { generatedTokenIds: generatedTokenIds as number[], sensitiveSteps: normalizedSteps, terminalLogitsSha256 };
}
function parseArguments(argv: string[]): { artifact: string; artifactIndex?: { path: string; sha256: string; constantPoolSha256: string }; binaryPool: string; realLoweringPlan?: string; finalFormulaMap?: { path: string; fileSha256: string; functions: number; orderedRootsSha256: string }; finalFormulaRuntime?: { path: string; fileSha256: string }; python: string; linearHelper: string; mlxHelper: string; linearBackend: "pytorch" | "mlx"; mlxHeadQuantization: "off" | "q8" | "q8-shortlist" | "q4"; mlxDecoderQuantization: Gemma4MlxDecoderQuantization; mlxDecoderQuantizationLayers?: string; fusedMlpRounding: "off" | "bf16" | "real" | "native-bf16"; fusedFfnRounding: "off" | "native-bf16"; fusedDecoderLayerRounding: "off" | "native-bf16"; fusedDecoderStackRounding: "off" | "real" | "native-bf16" | "native-bf16-ple"; fusedPleRounding: "off" | "bf16" | "real"; fusedPlePreludeRounding: "off" | "bf16" | "real"; fusedTokenForwardRounding: "off" | "bf16"; residentGeneration: "off" | "on"; finalHeadCompute: "f32" | "native-bf16" | "native-bf16-stream" | "native-bf16-whole"; nativeAttentionRounding: "off" | "bf16" | "real"; fusedAttentionRounding: "off" | "bf16" | "real" | "native-bf16"; threads: number; maxReadBytes: number; finalHeadMaxReadBytes: number } {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 2) { const flag = argv[index], value = argv[index + 1]; if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`); values.set(flag, value); }
  const known = new Set(["--artifact", "--artifact-index", "--artifact-index-sha256", "--artifact-sha256", "--binary-pool", "--real-lowering-plan", "--final-formula-map", "--final-formula-map-sha256", "--final-formula-functions", "--final-formula-roots-sha256", "--final-formula-runtime", "--final-formula-runtime-sha256", "--python", "--linear-helper", "--mlx-helper", "--linear-backend", "--mlx-head-quantization", "--mlx-decoder-quantization", "--mlx-decoder-quantization-layers", "--fused-mlp", "--fused-ffn", "--fused-decoder-layer", "--fused-decoder-stack", "--fused-ple", "--fused-ple-prelude", "--fused-token-forward", "--resident-generation", "--final-head", "--final-head-read-mib", "--native-attention", "--fused-attention", "--threads", "--max-read-mib"]); for (const flag of values.keys()) if (!known.has(flag)) throw new Error(`Flag desconhecida: ${flag}.`);
  const required = (flag: string): string => { const value = values.get(flag); if (!value) throw new Error(`${flag} é obrigatório.`); return resolve(value); };
  const threads = Number(values.get("--threads") ?? "10"), maxReadMiB = Number(values.get("--max-read-mib") ?? "16");
  const linearBackend = values.get("--linear-backend") ?? "mlx"; if (linearBackend !== "pytorch" && linearBackend !== "mlx") throw new Error("--linear-backend deve ser pytorch ou mlx.");
  const mlxHeadQuantization = values.get("--mlx-head-quantization") ?? "off"; if (mlxHeadQuantization !== "off" && mlxHeadQuantization !== "q8" && mlxHeadQuantization !== "q8-shortlist" && mlxHeadQuantization !== "q4") throw new Error("--mlx-head-quantization deve ser off, q8, q8-shortlist ou q4.");
  if (linearBackend !== "mlx" && mlxHeadQuantization !== "off") throw new Error("--mlx-head-quantization requer backend MLX.");
  const mlxDecoderQuantization = values.get("--mlx-decoder-quantization") ?? (linearBackend === "mlx" ? "q8-ffn-gate-up" : "off"); if (mlxDecoderQuantization !== "off" && mlxDecoderQuantization !== "q8-ffn" && mlxDecoderQuantization !== "q8-ffn-gate-up" && mlxDecoderQuantization !== "q8-ffn-gate-up-attention" && mlxDecoderQuantization !== "q8-ffn-gate-up-down" && mlxDecoderQuantization !== "q4-ffn-gate-up" && mlxDecoderQuantization !== "q8-ffn-gate-up-first-half" && mlxDecoderQuantization !== "q8-ffn-gate-up-last-half" && mlxDecoderQuantization !== "q8-ffn-down" && mlxDecoderQuantization !== "q8-attention" && mlxDecoderQuantization !== "q8-all") throw new Error("--mlx-decoder-quantization possui modo inválido.");
  if (linearBackend !== "mlx" && mlxDecoderQuantization !== "off") throw new Error("--mlx-decoder-quantization requer backend MLX.");
  const mlxDecoderQuantizationLayers = values.has("--mlx-decoder-quantization-layers") ? normalizeGemma4DecoderQuantizationLayers(values.get("--mlx-decoder-quantization-layers")!) : undefined;
  if (mlxDecoderQuantizationLayers !== undefined && mlxDecoderQuantization !== "q8-ffn-gate-up" && mlxDecoderQuantization !== "q8-ffn-gate-up-down" && mlxDecoderQuantization !== "q4-ffn-gate-up") throw new Error("--mlx-decoder-quantization-layers requer q8-ffn-gate-up, q8-ffn-gate-up-down ou q4-ffn-gate-up.");
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
  const indexValues = [values.get("--artifact-index"), values.get("--artifact-index-sha256"), values.get("--artifact-sha256")];
  if (indexValues.some((value) => value !== undefined) && indexValues.some((value) => value === undefined)) throw new Error("--artifact-index, --artifact-index-sha256 e --artifact-sha256 devem ser informados juntos.");
  const artifactIndex = indexValues[0] ? { path: resolve(indexValues[0]), sha256: indexValues[1]!, constantPoolSha256: indexValues[2]! } : undefined;
  const realLoweringPlan = fusedDecoderStackRounding === "real" ? resolve(values.get("--real-lowering-plan") ?? join(binaryPool, "vectorized-real-lowering.json")) : undefined;
  const finalFormulaValues = [values.get("--final-formula-map"), values.get("--final-formula-map-sha256"), values.get("--final-formula-functions"), values.get("--final-formula-roots-sha256")];
  if (finalFormulaValues.some((value) => value !== undefined) && finalFormulaValues.some((value) => value === undefined)) throw new Error("Flags do mapa final devem ser informadas juntas.");
  const finalFormulaFunctions = finalFormulaValues[2] === undefined ? undefined : Number(finalFormulaValues[2]);
  if (finalFormulaValues[0] !== undefined && (fusedDecoderStackRounding !== "real" || !/^[0-9a-f]{64}$/.test(finalFormulaValues[1]!) || !Number.isSafeInteger(finalFormulaFunctions) || finalFormulaFunctions! < 1 || !/^[0-9a-f]{64}$/.test(finalFormulaValues[3]!))) throw new Error("Contrato CLI do mapa final é inválido ou não usa lowering real.");
  const finalFormulaMap = finalFormulaValues[0] === undefined ? undefined : { path: resolve(finalFormulaValues[0]), fileSha256: finalFormulaValues[1]!, functions: finalFormulaFunctions!, orderedRootsSha256: finalFormulaValues[3]! };
  const finalRuntimeValues = [values.get("--final-formula-runtime"), values.get("--final-formula-runtime-sha256")];
  if (finalRuntimeValues.some((value) => value !== undefined) && finalRuntimeValues.some((value) => value === undefined)) throw new Error("Flags do runtime final devem ser informadas juntas.");
  const finalFormulaRuntime = finalRuntimeValues[0] === undefined ? undefined : { path: resolve(finalRuntimeValues[0]), fileSha256: finalRuntimeValues[1]! };
  if ((finalFormulaMap === undefined) !== (finalFormulaRuntime === undefined) || (finalFormulaRuntime && !/^[0-9a-f]{64}$/.test(finalFormulaRuntime.fileSha256))) throw new Error("Mapa e runtime final autenticado devem ser informados juntos.");
  return { artifact: required("--artifact"), ...(artifactIndex ? { artifactIndex } : {}), binaryPool, ...(realLoweringPlan ? { realLoweringPlan } : {}), ...(finalFormulaMap ? { finalFormulaMap } : {}), ...(finalFormulaRuntime ? { finalFormulaRuntime } : {}), python: values.get("--python") ?? resolve("venv/bin/python"), linearHelper: resolve(values.get("--linear-helper") ?? "scripts/gemma4-paged-linear-worker.py"), mlxHelper: resolve(values.get("--mlx-helper") ?? "scripts/gemma4-mlx-linear-worker.py"), linearBackend, mlxHeadQuantization, mlxDecoderQuantization, ...(mlxDecoderQuantizationLayers === undefined ? {} : { mlxDecoderQuantizationLayers }), fusedMlpRounding, fusedFfnRounding, fusedDecoderLayerRounding, fusedDecoderStackRounding, fusedPleRounding, fusedPlePreludeRounding, fusedTokenForwardRounding, residentGeneration, finalHeadCompute, nativeAttentionRounding, fusedAttentionRounding, threads, maxReadBytes: maxReadMiB * 1024 * 1024, finalHeadMaxReadBytes: finalHeadReadMiB * 1024 * 1024 };
}
