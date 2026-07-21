import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createGemma4RealComparisonServer, parseGemma4RealServerOptions, type DirectLogitAgreement, type Gemma4RealComparisonRunnerOptions } from "./gemma4-real-compare-server.js";

export const GEMMA4_CALIBRATION_PROMPTS = [
  "The capital of France is",
  "Two plus two equals",
  "Write one short sentence about the Moon:",
  "Translate to Portuguese: Good morning",
  "Complete the Python expression: sum([1, 2, 3]) ==",
  "Water freezes at",
  "Era uma vez um pequeno robô que",
  "Answer yes or no: Is the Earth round?",
] as const;

export interface Gemma4CalibrationCliOptions {
  prompts: string[];
  output: string;
  maxNewTokens: number;
  requestThreads: number;
  precision: "f32" | "f64";
  roundingPolicy: "none" | "layer-bf16" | "operation-bf16";
  runner: Gemma4RealComparisonRunnerOptions & { host: string; port: number };
}

interface ThreeWayCase {
  prompt: string;
  inputIds: number[][];
  baselineGeneratedTokenIds: number[];
  baselineGeneratedText: string;
  candidateGeneratedTokenIds: number[];
  candidateGeneratedText: string;
  generatedTokensEqual: boolean;
  firstDivergentStep: number | null;
  performance: { baselineSeconds: number; candidateSeconds: number; baselineTokensPerSecond: number; candidateTokensPerSecond: number; candidateSpeedup: number };
  steps: Array<{ step: number; contextsEqualBeforeStep: boolean; baselineToken: number; candidateToken: number; baselineTopLogits: unknown; candidateTopLogits: unknown }>;
  direct: {
    generatedTokenIds: number[];
    generatedText: string;
    elapsedSeconds: number;
    tokensPerSecond: number;
    linearThreads: number;
    maxReadMiB?: number;
    finalHeadReadMiB?: number;
    linearReferenceDispatches?: number;
    wholeNativeBf16Dispatches?: number;
    processRssBytes?: number;
    processMaxRssKiB?: number;
    linearBatchDispatches?: number;
    linearBatchedProjectionTiles?: number;
    fusedMlpRounding?: "off" | "bf16" | "real" | "native-bf16";
    fusedMlpDispatches?: number;
    fusedFfnRounding?: "off" | "native-bf16";
    fusedFfnDispatches?: number;
    fusedDecoderLayerRounding?: "off" | "native-bf16";
    fusedDecoderLayerDispatches?: number;
    fusedDecoderStackRounding?: "off" | "real" | "native-bf16" | "native-bf16-ple";
    fusedDecoderStackDispatches?: number;
    fusedDecoderStackEpilogueDispatches?: number;
    fusedTokenForwardRounding?: "off" | "bf16";
    fusedTokenForwardDispatches?: number;
    residentGeneration?: "off" | "on";
    fusedTokenGenerationDispatches?: number;
    externalForwardRequests?: number;
    kvCacheTransportBytes?: number;
    residentKvBytes?: number;
    terminalLogitMaterializations?: number;
    gpuRankedTokenSteps?: number;
    fullLogitTransfersAvoided?: number;
    terminalLogitVectorBytes?: number;
    ropeFactorBuilds?: number;
    ropeFactorBuildsAvoided?: number;
    topologyMaskBuilds?: number;
    topologyMaskBuildsAvoided?: number;
    redundantLogitFiniteScansAvoided?: number;
    kvPrefixValidationScansAvoided?: number;
    compiledIncrementalDecoderSteps?: number;
    incrementalCompilerCacheHit?: boolean;
    fusedDecoderStackGateUpPairs?: number;
    fusedDecoderStackWidenedCacheHits?: number;
    widenedTensorCacheEntries?: number;
    widenedTensorCacheBytes?: number;
    fusedPleRounding?: "off" | "bf16" | "real";
    fusedPlePreludeRounding?: "off" | "bf16" | "real";
    fusedPleDispatches?: number;
    fusedPlePreludeDispatches?: number;
    finalHeadCompute?: "f32" | "native-bf16" | "native-bf16-stream" | "native-bf16-whole";
    streamedNativeBf16Dispatches?: number;
    nativeAttentionRounding?: "off" | "bf16" | "real";
    nativeAttentionDispatches?: number;
    fusedAttentionRounding?: "off" | "bf16" | "real" | "native-bf16";
    fusedAttentionDispatches?: number;
    referenceSeconds?: number;
    streamedNativeBf16Seconds?: number;
    batchSeconds?: number;
    fusedMlpSeconds?: number;
    fusedFfnSeconds?: number;
    fusedDecoderLayerSeconds?: number;
    fusedDecoderStackSeconds?: number;
    fusedDecoderStackAttentionSeconds?: number;
    fusedDecoderStackFfnSeconds?: number;
    fusedDecoderStackPleSeconds?: number;
    fusedPleSeconds?: number;
    fusedPlePreludeSeconds?: number;
    nativeAttentionSeconds?: number;
    fusedAttentionSeconds?: number;
    tokensEqualBaseline: boolean;
    firstDivergentStep: number | null;
    logitAgreement?: DirectLogitAgreement;
    selectionPolicy?: string;
    selectedBackend?: "mlx" | "pytorch";
    fallbackTriggered?: boolean;
    fallbackReason?: string;
    fastPathMinimumMargin?: number | null;
    verificationMarginThreshold?: number;
    fastPathSeconds?: number;
    verificationSeconds?: number;
    workerVerificationSeconds?: number;
    hybridSeconds?: number;
    selectiveVerification?: boolean;
    sensitiveSteps?: number[];
    trustedFastPathSteps?: number;
    verificationHeadSteps?: number;
    verificationDivergenceStep?: number | null;
    fastPath?: { generatedTokenIds?: number[]; elapsedSeconds?: number; tokensPerSecond?: number; linearBackend?: string };
    steps: Array<{ step: number; tokenId: number; forwardSeconds: number; topLogits: unknown }>;
  };
}

export interface Gemma4ThreeWayCalibrationReport {
  kind: "gemma4-three-way-calibration";
  schemaVersion: 1;
  source: string;
  configuration: Record<string, unknown>;
  initialization: Record<string, unknown>;
  summary: Record<string, number | null>;
  cases: Array<Record<string, unknown>>;
}

export async function runGemma4ThreeWayCalibration(options: Gemma4CalibrationCliOptions): Promise<Gemma4ThreeWayCalibrationReport> {
  const server = createGemma4RealComparisonServer(options.runner);
  await new Promise<void>((accept, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => accept());
  });
  try {
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("Servidor de calibração não expôs uma porta TCP.");
    const endpoint = `http://127.0.0.1:${address.port}`;
    let status = await waitUntilReady(endpoint);
    const cases: ThreeWayCase[] = [];
    for (const prompt of options.prompts) {
      const response = await fetch(`${endpoint}/api/compare`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt, maxNewTokens: options.maxNewTokens, threads: options.requestThreads, precision: options.precision, roundingPolicy: options.roundingPolicy }),
      });
      const body = await response.json() as ThreeWayCase | { error?: string };
      if (!response.ok) throw new Error(`Calibração falhou para ${JSON.stringify(prompt)}: ${"error" in body ? body.error : response.statusText}`);
      assertThreeWayCase(body, prompt);
      cases.push(body);
    }
    status = await fetch(`${endpoint}/api/status`).then((response) => response.json()) as Record<string, unknown>;
    const report = summarizeGemma4ThreeWayCalibration(cases, options, status);
    await writeFile(options.output, `${JSON.stringify(report, null, 2)}\n`);
    return report;
  } finally {
    await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept()));
  }
}

export function summarizeGemma4ThreeWayCalibration(cases: ThreeWayCase[], options: { maxNewTokens: number; requestThreads: number; precision: "f32" | "f64"; roundingPolicy: "none" | "layer-bf16" | "operation-bf16"; runner: Gemma4RealComparisonRunnerOptions }, status: Record<string, unknown>): Gemma4ThreeWayCalibrationReport {
  if (cases.length === 0) throw new Error("Calibração requer ao menos um caso.");
  const totalSteps = cases.reduce((total, entry) => total + entry.baselineGeneratedTokenIds.length, 0);
  const compatibilityEqualSteps = cases.reduce((total, entry) => total + equalTokenSteps(entry.baselineGeneratedTokenIds, entry.candidateGeneratedTokenIds), 0);
  const directEqualSteps = cases.reduce((total, entry) => total + equalTokenSteps(entry.baselineGeneratedTokenIds, entry.direct.generatedTokenIds), 0);
  const baselineSeconds = cases.reduce((total, entry) => total + entry.performance.baselineSeconds, 0);
  const compatibilitySeconds = cases.reduce((total, entry) => total + entry.performance.candidateSeconds, 0);
  const directSeconds = cases.reduce((total, entry) => total + entry.direct.elapsedSeconds, 0);
  const compatibilityPromptsEqual = cases.filter((entry) => entry.generatedTokensEqual).length;
  const directPromptsEqual = cases.filter((entry) => entry.direct.tokensEqualBaseline).length;
  const directFallbackPrompts = cases.filter((entry) => entry.direct.fallbackTriggered === true).length;
  const promptsThreeWayEqual = cases.filter((entry) => entry.generatedTokensEqual && entry.direct.tokensEqualBaseline).length;
  const directReportedLogitSteps = cases.flatMap((entry) => entry.direct.logitAgreement?.steps ?? []);
  const directLogitSteps = directReportedLogitSteps.filter((step) => step.contextsEqualBeforeStep);
  const directRootDivergences = directLogitSteps.filter((step) => step.baselineArgmaxToken !== step.directArgmaxToken).length;
  const selectedLogitErrors = directLogitSteps.flatMap((step) => step.baselineArgmaxLogitAbsError === null ? [] : [step.baselineArgmaxLogitAbsError]);
  const marginErrors = directLogitSteps.flatMap((step) => step.greedyMarginAbsError === null ? [] : [step.greedyMarginAbsError]);
  const topKOverlapRates = directLogitSteps.flatMap((step) => step.topKOverlapRate === null ? [] : [step.topKOverlapRate]);
  const commonTopKErrors = directLogitSteps.flatMap((step) => step.topKCommonLogitMaxAbsError === null ? [] : [step.topKCommonLogitMaxAbsError]);
  const directBackend = options.runner.directLinearBackend ?? "mlx";
  return {
    kind: "gemma4-three-way-calibration",
    schemaVersion: 1,
    source: options.runner.source,
    configuration: {
      prompts: cases.length, tokensPerPrompt: options.maxNewTokens, requestThreads: options.requestThreads,
      directThreads: options.runner.directThreads, directMaxReadMiB: options.runner.directMaxReadMiB ?? 16, directFinalHeadReadMiB: options.runner.directFinalHeadReadMiB ?? (directBackend === "pytorch" ? 32 : options.runner.directMaxReadMiB ?? 16), directLinearBackend: directBackend, directVerificationMargin: options.runner.directVerificationMargin ?? null, directFusedMlp: options.runner.directFusedMlp ?? (directBackend === "pytorch" ? "native-bf16" : "real"), directFusedFfn: options.runner.directFusedFfn ?? (directBackend === "pytorch" ? "native-bf16" : "off"), directFusedDecoderLayer: options.runner.directFusedDecoderLayer ?? (directBackend === "pytorch" ? "native-bf16" : "off"), directFusedDecoderStack: options.runner.directFusedDecoderStack ?? "native-bf16", directFusedPle: options.runner.directFusedPle ?? (directBackend === "pytorch" ? "bf16" : "off"), directFusedPlePrelude: options.runner.directFusedPlePrelude ?? (directBackend === "mlx" ? "bf16" : "off"), directFusedTokenForward: options.runner.directFusedTokenForward ?? (directBackend === "mlx" && options.runner.directFusedDecoderStack !== "off" ? "bf16" : "off"), directResidentGeneration: options.runner.directResidentGeneration ?? (directBackend === "mlx" && options.runner.directFusedDecoderStack !== "off" ? "on" : "off"), directFinalHead: options.runner.directFinalHead ?? (directBackend === "pytorch" ? "native-bf16-stream" : "native-bf16-whole"), directNativeAttention: options.runner.directNativeAttention ?? (directBackend === "pytorch" ? "real" : "off"), directFusedAttention: options.runner.directFusedAttention ?? (directBackend === "pytorch" ? "native-bf16" : "off"), precision: options.precision, roundingPolicy: options.roundingPolicy,
    },
    initialization: status,
    summary: {
      promptsThreeWayEqual,
      promptThreeWayAgreementRate: promptsThreeWayEqual / cases.length,
      compatibilityPromptsEqual,
      compatibilityPromptAgreementRate: compatibilityPromptsEqual / cases.length,
      directPromptsEqual,
      directPromptAgreementRate: directPromptsEqual / cases.length,
      directFallbackPrompts,
      directFallbackRate: directFallbackPrompts / cases.length,
      comparedTokenSteps: totalSteps,
      compatibilityEqualTokenSteps: compatibilityEqualSteps,
      compatibilityTokenAgreementRate: compatibilityEqualSteps / totalSteps,
      directEqualTokenSteps: directEqualSteps,
      directTokenAgreementRate: directEqualSteps / totalSteps,
      directLogitReportedSteps: directReportedLogitSteps.length,
      directLogitMeasuredSteps: directLogitSteps.length,
      directRootDivergences,
      directComparableTokenAgreementRate: directLogitSteps.length === 0 ? null : (directLogitSteps.length - directRootDivergences) / directLogitSteps.length,
      directPostDivergenceSteps: directReportedLogitSteps.length - directLogitSteps.length,
      directSelectedLogitMeasuredSteps: selectedLogitErrors.length,
      directMeanBaselineArgmaxLogitAbsError: meanOrNull(selectedLogitErrors),
      directMaxBaselineArgmaxLogitAbsError: maxOrNull(selectedLogitErrors),
      directMarginMeasuredSteps: marginErrors.length,
      directMeanGreedyMarginAbsError: meanOrNull(marginErrors),
      directMaxGreedyMarginAbsError: maxOrNull(marginErrors),
      directMeanTopKOverlapRate: meanOrNull(topKOverlapRates),
      directMaxTopKCommonLogitAbsError: maxOrNull(commonTopKErrors),
      baselineSeconds,
      compatibilitySeconds,
      directSeconds,
      baselineTokensPerSecond: totalSteps / baselineSeconds,
      compatibilityTokensPerSecond: totalSteps / compatibilitySeconds,
      directTokensPerSecond: totalSteps / directSeconds,
      compatibilityVsBaselineThroughputRatio: baselineSeconds / compatibilitySeconds,
      directVsBaselineThroughputRatio: baselineSeconds / directSeconds,
    },
    cases: cases.map((entry) => ({
      prompt: entry.prompt,
      inputIds: entry.inputIds,
      threeWayTokensEqual: entry.generatedTokensEqual && entry.direct.tokensEqualBaseline,
      baseline: { tokenIds: entry.baselineGeneratedTokenIds, text: entry.baselineGeneratedText, seconds: entry.performance.baselineSeconds, tokensPerSecond: entry.performance.baselineTokensPerSecond },
      compatibility: { tokenIds: entry.candidateGeneratedTokenIds, text: entry.candidateGeneratedText, tokensEqualBaseline: entry.generatedTokensEqual, firstDivergentStep: entry.firstDivergentStep, seconds: entry.performance.candidateSeconds, tokensPerSecond: entry.performance.candidateTokensPerSecond },
      direct: { tokenIds: entry.direct.generatedTokenIds, text: entry.direct.generatedText, tokensEqualBaseline: entry.direct.tokensEqualBaseline, firstDivergentStep: entry.direct.firstDivergentStep, seconds: entry.direct.elapsedSeconds, tokensPerSecond: entry.direct.tokensPerSecond, linearThreads: entry.direct.linearThreads,
        ...(entry.direct.selectionPolicy === undefined ? {} : { selectionPolicy: entry.direct.selectionPolicy }),
        ...(entry.direct.selectedBackend === undefined ? {} : { selectedBackend: entry.direct.selectedBackend }),
        ...(entry.direct.fallbackTriggered === undefined ? {} : { fallbackTriggered: entry.direct.fallbackTriggered }),
        ...(entry.direct.fallbackReason === undefined ? {} : { fallbackReason: entry.direct.fallbackReason }),
        ...(entry.direct.fastPathMinimumMargin === undefined ? {} : { fastPathMinimumMargin: entry.direct.fastPathMinimumMargin }),
        ...(entry.direct.verificationMarginThreshold === undefined ? {} : { verificationMarginThreshold: entry.direct.verificationMarginThreshold }),
        ...(entry.direct.fastPathSeconds === undefined ? {} : { fastPathSeconds: entry.direct.fastPathSeconds }),
        ...(entry.direct.verificationSeconds === undefined ? {} : { verificationSeconds: entry.direct.verificationSeconds }),
        ...(entry.direct.workerVerificationSeconds === undefined ? {} : { workerVerificationSeconds: entry.direct.workerVerificationSeconds }),
        ...(entry.direct.hybridSeconds === undefined ? {} : { hybridSeconds: entry.direct.hybridSeconds }),
        ...(entry.direct.selectiveVerification === undefined ? {} : { selectiveVerification: entry.direct.selectiveVerification }),
        ...(entry.direct.sensitiveSteps === undefined ? {} : { sensitiveSteps: entry.direct.sensitiveSteps }),
        ...(entry.direct.trustedFastPathSteps === undefined ? {} : { trustedFastPathSteps: entry.direct.trustedFastPathSteps }),
        ...(entry.direct.verificationHeadSteps === undefined ? {} : { verificationHeadSteps: entry.direct.verificationHeadSteps }),
        ...(entry.direct.verificationDivergenceStep === undefined ? {} : { verificationDivergenceStep: entry.direct.verificationDivergenceStep }),
        ...(entry.direct.fastPath === undefined ? {} : { fastPath: { tokenIds: entry.direct.fastPath.generatedTokenIds, seconds: entry.direct.fastPath.elapsedSeconds, tokensPerSecond: entry.direct.fastPath.tokensPerSecond, linearBackend: entry.direct.fastPath.linearBackend } }),
        ...(entry.direct.logitAgreement === undefined ? {} : { logitAgreement: entry.direct.logitAgreement }),
        ...(entry.direct.maxReadMiB === undefined ? {} : { maxReadMiB: entry.direct.maxReadMiB }),
        ...(entry.direct.finalHeadReadMiB === undefined ? {} : { finalHeadReadMiB: entry.direct.finalHeadReadMiB }),
        ...(entry.direct.linearReferenceDispatches === undefined ? {} : { linearReferenceDispatches: entry.direct.linearReferenceDispatches }),
        ...(entry.direct.wholeNativeBf16Dispatches === undefined ? {} : { wholeNativeBf16Dispatches: entry.direct.wholeNativeBf16Dispatches }),
        ...(entry.direct.processRssBytes === undefined ? {} : { processRssBytes: entry.direct.processRssBytes }),
        ...(entry.direct.processMaxRssKiB === undefined ? {} : { processMaxRssKiB: entry.direct.processMaxRssKiB }),
        ...(entry.direct.linearBatchDispatches === undefined ? {} : { linearBatchDispatches: entry.direct.linearBatchDispatches }),
        ...(entry.direct.linearBatchedProjectionTiles === undefined ? {} : { linearBatchedProjectionTiles: entry.direct.linearBatchedProjectionTiles }),
        ...(entry.direct.fusedMlpRounding === undefined ? {} : { fusedMlpRounding: entry.direct.fusedMlpRounding }),
        ...(entry.direct.fusedMlpDispatches === undefined ? {} : { fusedMlpDispatches: entry.direct.fusedMlpDispatches }),
        ...(entry.direct.fusedFfnRounding === undefined ? {} : { fusedFfnRounding: entry.direct.fusedFfnRounding }),
        ...(entry.direct.fusedFfnDispatches === undefined ? {} : { fusedFfnDispatches: entry.direct.fusedFfnDispatches }),
        ...(entry.direct.fusedDecoderLayerRounding === undefined ? {} : { fusedDecoderLayerRounding: entry.direct.fusedDecoderLayerRounding }),
        ...(entry.direct.fusedDecoderLayerDispatches === undefined ? {} : { fusedDecoderLayerDispatches: entry.direct.fusedDecoderLayerDispatches }),
        ...(entry.direct.fusedDecoderStackRounding === undefined ? {} : { fusedDecoderStackRounding: entry.direct.fusedDecoderStackRounding }),
        ...(entry.direct.fusedDecoderStackDispatches === undefined ? {} : { fusedDecoderStackDispatches: entry.direct.fusedDecoderStackDispatches }),
        ...(entry.direct.fusedDecoderStackEpilogueDispatches === undefined ? {} : { fusedDecoderStackEpilogueDispatches: entry.direct.fusedDecoderStackEpilogueDispatches }),
        ...(entry.direct.fusedTokenForwardRounding === undefined ? {} : { fusedTokenForwardRounding: entry.direct.fusedTokenForwardRounding }),
        ...(entry.direct.fusedTokenForwardDispatches === undefined ? {} : { fusedTokenForwardDispatches: entry.direct.fusedTokenForwardDispatches }),
        ...(entry.direct.residentGeneration === undefined ? {} : { residentGeneration: entry.direct.residentGeneration }),
        ...(entry.direct.fusedTokenGenerationDispatches === undefined ? {} : { fusedTokenGenerationDispatches: entry.direct.fusedTokenGenerationDispatches }),
        ...(entry.direct.externalForwardRequests === undefined ? {} : { externalForwardRequests: entry.direct.externalForwardRequests }),
        ...(entry.direct.kvCacheTransportBytes === undefined ? {} : { kvCacheTransportBytes: entry.direct.kvCacheTransportBytes }),
        ...(entry.direct.residentKvBytes === undefined ? {} : { residentKvBytes: entry.direct.residentKvBytes }),
        ...(entry.direct.terminalLogitMaterializations === undefined ? {} : { terminalLogitMaterializations: entry.direct.terminalLogitMaterializations }),
        ...(entry.direct.gpuRankedTokenSteps === undefined ? {} : { gpuRankedTokenSteps: entry.direct.gpuRankedTokenSteps }),
        ...(entry.direct.fullLogitTransfersAvoided === undefined ? {} : { fullLogitTransfersAvoided: entry.direct.fullLogitTransfersAvoided }),
        ...(entry.direct.terminalLogitVectorBytes === undefined ? {} : { terminalLogitVectorBytes: entry.direct.terminalLogitVectorBytes }),
        ...(entry.direct.ropeFactorBuilds === undefined ? {} : { ropeFactorBuilds: entry.direct.ropeFactorBuilds }),
        ...(entry.direct.ropeFactorBuildsAvoided === undefined ? {} : { ropeFactorBuildsAvoided: entry.direct.ropeFactorBuildsAvoided }),
        ...(entry.direct.topologyMaskBuilds === undefined ? {} : { topologyMaskBuilds: entry.direct.topologyMaskBuilds }),
        ...(entry.direct.topologyMaskBuildsAvoided === undefined ? {} : { topologyMaskBuildsAvoided: entry.direct.topologyMaskBuildsAvoided }),
        ...(entry.direct.redundantLogitFiniteScansAvoided === undefined ? {} : { redundantLogitFiniteScansAvoided: entry.direct.redundantLogitFiniteScansAvoided }),
        ...(entry.direct.kvPrefixValidationScansAvoided === undefined ? {} : { kvPrefixValidationScansAvoided: entry.direct.kvPrefixValidationScansAvoided }),
        ...(entry.direct.compiledIncrementalDecoderSteps === undefined ? {} : { compiledIncrementalDecoderSteps: entry.direct.compiledIncrementalDecoderSteps }),
        ...(entry.direct.incrementalCompilerCacheHit === undefined ? {} : { incrementalCompilerCacheHit: entry.direct.incrementalCompilerCacheHit }),
        ...(entry.direct.fusedDecoderStackGateUpPairs === undefined ? {} : { fusedDecoderStackGateUpPairs: entry.direct.fusedDecoderStackGateUpPairs }),
        ...(entry.direct.fusedDecoderStackWidenedCacheHits === undefined ? {} : { fusedDecoderStackWidenedCacheHits: entry.direct.fusedDecoderStackWidenedCacheHits }),
        ...(entry.direct.widenedTensorCacheEntries === undefined ? {} : { widenedTensorCacheEntries: entry.direct.widenedTensorCacheEntries }),
        ...(entry.direct.widenedTensorCacheBytes === undefined ? {} : { widenedTensorCacheBytes: entry.direct.widenedTensorCacheBytes }),
        ...(entry.direct.fusedPleRounding === undefined ? {} : { fusedPleRounding: entry.direct.fusedPleRounding }),
        ...(entry.direct.fusedPlePreludeRounding === undefined ? {} : { fusedPlePreludeRounding: entry.direct.fusedPlePreludeRounding }),
        ...(entry.direct.fusedPleDispatches === undefined ? {} : { fusedPleDispatches: entry.direct.fusedPleDispatches }),
        ...(entry.direct.fusedPlePreludeDispatches === undefined ? {} : { fusedPlePreludeDispatches: entry.direct.fusedPlePreludeDispatches }),
        ...(entry.direct.finalHeadCompute === undefined ? {} : { finalHeadCompute: entry.direct.finalHeadCompute }),
        ...(entry.direct.streamedNativeBf16Dispatches === undefined ? {} : { streamedNativeBf16Dispatches: entry.direct.streamedNativeBf16Dispatches }),
        ...(entry.direct.nativeAttentionRounding === undefined ? {} : { nativeAttentionRounding: entry.direct.nativeAttentionRounding }),
        ...(entry.direct.nativeAttentionDispatches === undefined ? {} : { nativeAttentionDispatches: entry.direct.nativeAttentionDispatches }),
        ...(entry.direct.fusedAttentionRounding === undefined ? {} : { fusedAttentionRounding: entry.direct.fusedAttentionRounding }),
        ...(entry.direct.fusedAttentionDispatches === undefined ? {} : { fusedAttentionDispatches: entry.direct.fusedAttentionDispatches }),
        ...(entry.direct.referenceSeconds === undefined ? {} : { referenceSeconds: entry.direct.referenceSeconds }),
        ...(entry.direct.streamedNativeBf16Seconds === undefined ? {} : { streamedNativeBf16Seconds: entry.direct.streamedNativeBf16Seconds }),
        ...(entry.direct.batchSeconds === undefined ? {} : { batchSeconds: entry.direct.batchSeconds }),
        ...(entry.direct.fusedMlpSeconds === undefined ? {} : { fusedMlpSeconds: entry.direct.fusedMlpSeconds }),
        ...(entry.direct.fusedFfnSeconds === undefined ? {} : { fusedFfnSeconds: entry.direct.fusedFfnSeconds }),
        ...(entry.direct.fusedDecoderLayerSeconds === undefined ? {} : { fusedDecoderLayerSeconds: entry.direct.fusedDecoderLayerSeconds }),
        ...(entry.direct.fusedDecoderStackSeconds === undefined ? {} : { fusedDecoderStackSeconds: entry.direct.fusedDecoderStackSeconds }),
        ...(entry.direct.fusedDecoderStackAttentionSeconds === undefined ? {} : { fusedDecoderStackAttentionSeconds: entry.direct.fusedDecoderStackAttentionSeconds }),
        ...(entry.direct.fusedDecoderStackFfnSeconds === undefined ? {} : { fusedDecoderStackFfnSeconds: entry.direct.fusedDecoderStackFfnSeconds }),
        ...(entry.direct.fusedDecoderStackPleSeconds === undefined ? {} : { fusedDecoderStackPleSeconds: entry.direct.fusedDecoderStackPleSeconds }),
        ...(entry.direct.fusedPleSeconds === undefined ? {} : { fusedPleSeconds: entry.direct.fusedPleSeconds }),
        ...(entry.direct.fusedPlePreludeSeconds === undefined ? {} : { fusedPlePreludeSeconds: entry.direct.fusedPlePreludeSeconds }),
        ...(entry.direct.nativeAttentionSeconds === undefined ? {} : { nativeAttentionSeconds: entry.direct.nativeAttentionSeconds }),
        ...(entry.direct.fusedAttentionSeconds === undefined ? {} : { fusedAttentionSeconds: entry.direct.fusedAttentionSeconds }),
        forwardSeconds: entry.direct.steps.map((step) => step.forwardSeconds) },
      steps: entry.steps.map((step, index) => ({
        step: step.step, contextsEqualBeforeStep: step.contextsEqualBeforeStep,
        baselineToken: step.baselineToken, compatibilityToken: step.candidateToken, directToken: entry.direct.steps[index]?.tokenId,
        baselineTopLogits: step.baselineTopLogits, compatibilityTopLogits: step.candidateTopLogits, directTopLogits: entry.direct.steps[index]?.topLogits,
      })),
    })),
  };
}

export async function parseGemma4CalibrationCliOptions(argv: readonly string[]): Promise<Gemma4CalibrationCliOptions> {
  const pairs: Array<[string, string]> = [];
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index], value = argv[index + 1];
    if (!flag?.startsWith("--") || value === undefined || pairs.some(([existing]) => existing === flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`);
    pairs.push([flag, value]);
  }
  const value = (flag: string): string | undefined => pairs.find(([candidate]) => candidate === flag)?.[1];
  const output = value("--output"); if (!output) throw new Error("--output é obrigatório.");
  const maxNewTokens = Number(value("--tokens") ?? "2"), requestThreads = Number(value("--request-threads") ?? "1");
  if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 64) throw new Error("--tokens deve estar entre 1 e 64.");
  if (!Number.isSafeInteger(requestThreads) || requestThreads < 0 || requestThreads > 256) throw new Error("--request-threads deve estar entre 0 e 256.");
  const precision = value("--precision") ?? "f32"; if (precision !== "f32" && precision !== "f64") throw new Error("--precision deve ser f32 ou f64.");
  const roundingPolicy = value("--rounding-policy") ?? "none"; if (roundingPolicy !== "none" && roundingPolicy !== "layer-bf16" && roundingPolicy !== "operation-bf16") throw new Error("--rounding-policy inválido.");
  const promptsPath = value("--prompts-json");
  const prompts = promptsPath ? JSON.parse(await readFile(resolve(promptsPath), "utf8")) as unknown : [...GEMMA4_CALIBRATION_PROMPTS];
  if (!Array.isArray(prompts) || prompts.length === 0 || prompts.some((prompt) => typeof prompt !== "string" || prompt.length === 0 || prompt.length > 16_384)) throw new Error("--prompts-json deve conter um array não vazio de prompts válidos.");
  const calibrationFlags = new Set(["--output", "--tokens", "--request-threads", "--precision", "--rounding-policy", "--prompts-json"]);
  const runnerArguments = pairs.filter(([flag]) => !calibrationFlags.has(flag)).flat();
  return { prompts: prompts as string[], output: resolve(output), maxNewTokens, requestThreads, precision, roundingPolicy, runner: parseGemma4RealServerOptions(runnerArguments) };
}

async function waitUntilReady(endpoint: string): Promise<Record<string, unknown>> {
  const deadline = Date.now() + 240_000;
  while (Date.now() < deadline) {
    const response = await fetch(`${endpoint}/api/status`);
    const status = await response.json() as Record<string, unknown>;
    if (typeof status.error === "string") throw new Error(`Inicialização dos workers falhou: ${status.error}`);
    if (status.ready === true) {
      const direct = status.direct as Record<string, unknown> | undefined;
      if (direct?.enabled !== true) throw new Error("Calibração três-vias requer o backend compilado direto habilitado.");
      return status;
    }
    await new Promise((accept) => setTimeout(accept, 100));
  }
  throw new Error("Workers persistentes não ficaram prontos em 240 segundos.");
}

function assertThreeWayCase(value: ThreeWayCase | { error?: string }, prompt: string): asserts value is ThreeWayCase {
  const candidate = value as Partial<ThreeWayCase>;
  if (!Array.isArray(candidate.baselineGeneratedTokenIds) || !Array.isArray(candidate.candidateGeneratedTokenIds) || !candidate.direct || !Array.isArray(candidate.direct.generatedTokenIds)) throw new Error(`Resposta três-vias inválida para ${JSON.stringify(prompt)}.`);
}

function equalTokenSteps(reference: readonly number[], candidate: readonly number[]): number {
  return reference.reduce((equal, token, index) => equal + Number(token === candidate[index]), 0);
}
function meanOrNull(values: readonly number[]): number | null { return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length; }
function maxOrNull(values: readonly number[]): number | null { return values.length === 0 ? null : Math.max(...values); }
