import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { parseGemma4CalibrationCliOptions, summarizeGemma4ThreeWayCalibration } from "../src/gemma4-real-calibration.js";

const baseCase = {
  prompt: "x", inputIds: [[2]], baselineGeneratedTokenIds: [7, 8], baselineGeneratedText: "a",
  candidateGeneratedTokenIds: [7, 9], candidateGeneratedText: "b", generatedTokensEqual: false, firstDivergentStep: 1,
  performance: { baselineSeconds: 4, candidateSeconds: 2, baselineTokensPerSecond: 0.5, candidateTokensPerSecond: 1, candidateSpeedup: 2 },
  steps: [{ step: 0, contextsEqualBeforeStep: true, baselineToken: 7, candidateToken: 7, baselineTopLogits: [], candidateTopLogits: [] }, { step: 1, contextsEqualBeforeStep: true, baselineToken: 8, candidateToken: 9, baselineTopLogits: [], candidateTopLogits: [] }],
  direct: { generatedTokenIds: [7, 8], generatedText: "a", elapsedSeconds: 1, tokensPerSecond: 2, linearThreads: 4, fusedDecoderStackEpilogueDispatches: 2, residentGeneration: "on" as const, fusedTokenGenerationDispatches: 1, externalForwardRequests: 1, kvCacheTransportBytes: 0, residentKvBytes: 512, terminalLogitMaterializations: 1, gpuRankedTokenSteps: 2, fullLogitTransfersAvoided: 1, terminalLogitVectorBytes: 1024, ropeFactorBuilds: 2, ropeFactorBuildsAvoided: 82, topologyMaskBuilds: 4, topologyMaskBuildsAvoided: 80, redundantLogitFiniteScansAvoided: 2, kvPrefixValidationScansAvoided: 14, compiledIncrementalDecoderSteps: 1, incrementalCompilerCacheHit: true, fusedDecoderStackGateUpPairs: 42, fusedDecoderStackWidenedCacheHits: 588, widenedTensorCacheEntries: 294, widenedTensorCacheBytes: 1024, fusedDecoderStackAttentionSeconds: 0.2, fusedDecoderStackFfnSeconds: 0.5, fusedDecoderStackPleSeconds: 0.1, tokensEqualBaseline: true, firstDivergentStep: null, steps: [{ step: 0, tokenId: 7, forwardSeconds: 0.6, topLogits: [] }, { step: 1, tokenId: 8, forwardSeconds: 0.4, topLogits: [] }] },
};

test("calibração três-vias agrega acordo token a token e throughput", () => {
  const report = summarizeGemma4ThreeWayCalibration([baseCase], { maxNewTokens: 2, requestThreads: 1, precision: "f32", roundingPolicy: "none", runner: { source: "/model", python: "python", helper: "helper", directThreads: 4 } }, { ready: true });
  assert.equal(report.configuration.directMlxHeadQuantization, "off");
  assert.equal(report.configuration.directMlxDecoderQuantization, "q8-ffn-gate-up");
  assert.deepEqual(report.summary, {
    promptsThreeWayEqual: 0, promptThreeWayAgreementRate: 0,
    compatibilityPromptsEqual: 0, compatibilityPromptAgreementRate: 0,
    directPromptsEqual: 1, directPromptAgreementRate: 1, directFallbackPrompts: 0, directFallbackRate: 0, comparedTokenSteps: 2,
    compatibilityEqualTokenSteps: 1, compatibilityTokenAgreementRate: 0.5,
    directEqualTokenSteps: 2, directTokenAgreementRate: 1,
    directLogitReportedSteps: 0, directLogitMeasuredSteps: 0, directRootDivergences: 0, directComparableTokenAgreementRate: null, directPostDivergenceSteps: 0, directSelectedLogitMeasuredSteps: 0,
    directMeanBaselineArgmaxLogitAbsError: null, directMaxBaselineArgmaxLogitAbsError: null,
    directMarginMeasuredSteps: 0, directMeanGreedyMarginAbsError: null, directMaxGreedyMarginAbsError: null,
    directMeanTopKOverlapRate: null, directMaxTopKCommonLogitAbsError: null,
    baselineSeconds: 4, compatibilitySeconds: 2, directSeconds: 1,
    baselineTokensPerSecond: 0.5, compatibilityTokensPerSecond: 1, directTokensPerSecond: 2,
    compatibilityVsBaselineThroughputRatio: 2, directVsBaselineThroughputRatio: 4,
  });
  assert.deepEqual(report.cases[0]?.direct, { tokenIds: [7, 8], text: "a", tokensEqualBaseline: true, firstDivergentStep: null, seconds: 1, tokensPerSecond: 2, linearThreads: 4, fusedDecoderStackEpilogueDispatches: 2, residentGeneration: "on", fusedTokenGenerationDispatches: 1, externalForwardRequests: 1, kvCacheTransportBytes: 0, residentKvBytes: 512, terminalLogitMaterializations: 1, gpuRankedTokenSteps: 2, fullLogitTransfersAvoided: 1, terminalLogitVectorBytes: 1024, ropeFactorBuilds: 2, ropeFactorBuildsAvoided: 82, topologyMaskBuilds: 4, topologyMaskBuildsAvoided: 80, redundantLogitFiniteScansAvoided: 2, kvPrefixValidationScansAvoided: 14, compiledIncrementalDecoderSteps: 1, incrementalCompilerCacheHit: true, fusedDecoderStackGateUpPairs: 42, fusedDecoderStackWidenedCacheHits: 588, widenedTensorCacheEntries: 294, widenedTensorCacheBytes: 1024, fusedDecoderStackAttentionSeconds: 0.2, fusedDecoderStackFfnSeconds: 0.5, fusedDecoderStackPleSeconds: 0.1, forwardSeconds: [0.6, 0.4] });
});

test("calibração preserva métricas da cabeça quantizada certificada", () => {
  const report = summarizeGemma4ThreeWayCalibration([
    { ...baseCase, direct: { ...baseCase.direct, quantizedHeadCertifiedSteps: 1, quantizedHeadExactFallbackSteps: 1 } },
  ], { maxNewTokens: 2, requestThreads: 1, precision: "f32", roundingPolicy: "none", runner: { source: "/model", python: "python", helper: "helper", directThreads: 4 } }, { ready: true });
  const direct = report.cases[0]?.direct as { quantizedHeadCertifiedSteps?: number; quantizedHeadExactFallbackSteps?: number };
  assert.equal(direct.quantizedHeadCertifiedSteps, 1);
  assert.equal(direct.quantizedHeadExactFallbackSteps, 1);
});

test("opções da calibração validam corpus e preservam opções do runtime", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-calibration-options-"));
  const prompts = join(directory, "prompts.json");
  await writeFile(prompts, JSON.stringify(["um", "dois"]));
  try {
    const options = await parseGemma4CalibrationCliOptions(["--source", "./bundle", "--output", "./report.json", "--prompts-json", prompts, "--tokens", "3", "--direct-threads", "8"]);
    assert.deepEqual(options.prompts, ["um", "dois"]); assert.equal(options.maxNewTokens, 3); assert.equal(options.runner.directThreads, undefined);
    assert.match(options.runner.source, /\/bundle$/); assert.match(options.output, /\/report\.json$/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("opções da calibração rejeitam corpus vazio e limites inválidos", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-calibration-invalid-"));
  const prompts = join(directory, "prompts.json"); await writeFile(prompts, "[]");
  try {
    await assert.rejects(parseGemma4CalibrationCliOptions(["--output", "x", "--prompts-json", prompts]), /array não vazio/);
    await assert.rejects(parseGemma4CalibrationCliOptions(["--output", "x", "--tokens", "0"]), /entre 1 e 64/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
