import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { parseGemma4CalibrationCliOptions, summarizeGemma4ThreeWayCalibration } from "../src/gemma4-real-calibration.js";

const baseCase = {
  prompt: "x", inputIds: [[2]], baselineGeneratedTokenIds: [7, 8], baselineGeneratedText: "a",
  candidateGeneratedTokenIds: [7, 9], candidateGeneratedText: "b", generatedTokensEqual: false, firstDivergentStep: 1,
  performance: { baselineSeconds: 4, candidateSeconds: 2, baselineTokensPerSecond: 0.5, candidateTokensPerSecond: 1, candidateSpeedup: 2 },
  steps: [{ step: 0, contextsEqualBeforeStep: true, baselineToken: 7, candidateToken: 7, baselineTopLogits: [], candidateTopLogits: [] }, { step: 1, contextsEqualBeforeStep: true, baselineToken: 8, candidateToken: 9, baselineTopLogits: [], candidateTopLogits: [] }],
  direct: { generatedTokenIds: [7, 8], generatedText: "a", elapsedSeconds: 1, tokensPerSecond: 2, linearThreads: 4, parallelExecutionBackend: "metal-vectorized-kernels" as const, configuredHostThreads: 4, hostThreadSettingApplied: false, metalCommandStreams: 1, parallelTerminalOutputDimensions: 262_144, fusedDecoderStackEpilogueDispatches: 2, residentGeneration: "on" as const, fusedTokenGenerationDispatches: 1, externalForwardRequests: 1, kvCacheTransportBytes: 0, residentKvBytes: 512, terminalLogitMaterializations: 1, gpuRankedTokenSteps: 2, fullLogitTransfersAvoided: 1, terminalLogitVectorBytes: 1024, ropeFactorBuilds: 2, ropeFactorBuildsAvoided: 82, topologyMaskBuilds: 4, topologyMaskBuildsAvoided: 80, redundantLogitFiniteScansAvoided: 2, kvPrefixValidationScansAvoided: 14, decoderLayerValidityScansAvoided: 41, compiledIncrementalDecoderSteps: 1, incrementalCompilerCacheHit: true, fusedDecoderStackGateUpPairs: 42, fusedDecoderStackWidenedCacheHits: 588, widenedTensorCacheEntries: 294, widenedTensorCacheBytes: 1024, fusedDecoderStackAttentionSeconds: 0.2, fusedDecoderStackFfnSeconds: 0.5, fusedDecoderStackPleSeconds: 0.1, tokensEqualBaseline: true, firstDivergentStep: null, steps: [{ step: 0, tokenId: 7, forwardSeconds: 0.6, topLogits: [] }, { step: 1, tokenId: 8, forwardSeconds: 0.4, topLogits: [] }] },
};

test("evidência do default de oito threads vincula corpus e tokens autoritativos", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-kv-rewind-thread8-calibration-32x8.json", "utf8")) as any;
  assert.equal(artifact.configuration.selectedThreads, 8);
  assert.deepEqual([artifact.canonicalCorpus.promptsEqual, artifact.canonicalCorpus.equalTokenSteps, artifact.canonicalCorpus.rootDivergences, artifact.canonicalCorpus.compiledContinuationsAccepted, artifact.canonicalCorpus.compiledContinuationPrefixRetries, artifact.canonicalCorpus.compiledContinuationsRejected], [32, 256, 2, 2, 0, 0]);
  const digest = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");
  assert.equal(await digest(`artifacts/${artifact.sources.prompts.file}`), artifact.sources.prompts.sha256);
  assert.equal(await digest(`artifacts/${artifact.sources.authoritativeTokenIds.file}`), artifact.sources.authoritativeTokenIds.sha256);
  const selected = artifact.threadSweep.results.find((entry: { threads: number }) => entry.threads === artifact.configuration.selectedThreads);
  const ten = artifact.threadSweep.results.find((entry: { threads: number }) => entry.threads === 10);
  assert.ok(selected.lateTieSeconds < ten.lateTieSeconds); assert.ok(selected.rootCorrectionSeconds < ten.rootCorrectionSeconds);
});

test("evidência rejeita prefill exato eager quando ele preserva tokens mas piora todas as classes de latência", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-eager-verification-prefill-rejection-3x8.json", "utf8")) as any;
  assert.equal(artifact.decision.selectedPolicy, "sensitive-margin-triggered");
  assert.equal(artifact.decision.rejectedPolicy, "eager-before-compiled-generation");
  assert.equal(artifact.configuration.warmRepetitionsPerPrompt, 2);
  assert.ok(artifact.tokenParity.every((entry: { equalAcrossPolicies: boolean }) => entry.equalAcrossPolicies));
  assert.equal(artifact.tokenParity.filter((entry: { matchesAuthoritative?: boolean }) => entry.matchesAuthoritative).length, 2);
  const digest = createHash("sha256").update(await readFile(`artifacts/${artifact.sources.authoritativeTokenIds.file}`)).digest("hex");
  assert.equal(digest, artifact.sources.authoritativeTokenIds.sha256);
  const selected = artifact.results.find((entry: { policy: string }) => entry.policy === artifact.decision.selectedPolicy);
  for (const rejected of artifact.results.filter((entry: { policy: string }) => entry.policy === artifact.decision.rejectedPolicy)) {
    assert.ok(rejected.meanSeconds.lateFallback > selected.meanSeconds.lateFallback);
    assert.ok(rejected.meanSeconds.earlyFallback > selected.meanSeconds.earlyFallback);
    assert.ok(rejected.meanSeconds.safe > selected.meanSeconds.safe);
  }
});

test("evidência do corte de prefill terminal preserva corpus e correção de raiz", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-terminal-prefill-skip-parity-32x8.json", "utf8")) as any;
  assert.deepEqual([artifact.result.promptsEqual, artifact.result.comparedTokenSteps, artifact.result.equalTokenSteps, artifact.result.terminalPrefillSkips], [32, 256, 256, 1]);
  assert.deepEqual([artifact.result.terminalSkipCase.verificationPrefillAhead, artifact.result.terminalSkipCase.verificationPrefillSkippedTerminal, artifact.result.terminalSkipCase.sensitiveStep], [false, true, 7]);
  assert.deepEqual([artifact.result.rootCorrectionGuardCase.verificationDivergenceStep, artifact.result.rootCorrectionGuardCase.compiledContinuationAccepted], [3, true]);
  const digest = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");
  assert.equal(await digest(`artifacts/${artifact.sources.prompts.file}`), artifact.sources.prompts.sha256);
  assert.equal(await digest(`artifacts/${artifact.sources.authoritativeTokenIds.file}`), artifact.sources.authoritativeTokenIds.sha256);
});

test("evidência promove antecipação do prefixo exato somente em sessão com cache", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-cached-prefix-ahead-promotion-8-token.json", "utf8")) as any;
  const digest = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");
  assert.equal(await digest(`artifacts/${artifact.sources.prompts.file}`), artifact.sources.prompts.sha256);
  assert.equal(await digest(`artifacts/${artifact.sources.authoritativeTokenIds.file}`), artifact.sources.authoritativeTokenIds.sha256);
  assert.equal(artifact.cacheMiss.verificationPrefixAhead, false);
  assert.equal(artifact.liveOriginalComparison.generatedTokensEqual, true);
  assert.deepEqual(artifact.liveOriginalComparison.baselineGeneratedTokenIds, artifact.liveOriginalComparison.compiledGeneratedTokenIds);
  for (const policy of [artifact.pairedBenchmark.control, artifact.pairedBenchmark.candidate]) {
    assert.ok(policy.repetitions.every((entry: any) => entry.generatedTokenIds.length === 8 && entry.generatedTokenIds.every((token: number, index: number) => token === artifact.authoritativeGeneratedTokenIds[index])));
  }
  assert.ok(artifact.pairedBenchmark.candidate.meanElapsedSeconds < artifact.pairedBenchmark.control.meanElapsedSeconds);
  assert.ok(artifact.pairedBenchmark.throughputGainRate > 0);
  assert.ok(artifact.pairedBenchmark.candidate.repetitions.every((entry: any) => entry.verificationPrefixAhead === true && entry.verificationPrefillWaitSeconds === 0 && entry.verificationPrefillOverlapSeconds === entry.verificationPrefillAheadSeconds));
});

test("evidência promove o maior prefixo exato entre sessões sem alterar logits", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-cross-session-prefix-promotion.json", "utf8")) as any;
  const digest = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");
  assert.equal(await digest(artifact.sources.sessionCache.file), artifact.sources.sessionCache.sha256);
  assert.equal(await digest(artifact.sources.verificationWorker.file), artifact.sources.verificationWorker.sha256);
  assert.equal(await digest(artifact.sources.runtimeIndex.file), artifact.sources.runtimeIndex.sha256);
  assert.ok(artifact.isolatedExactPrefill.identicalInputRepetitions.every((entry: any) => entry.sharedTokensReused === 9 && entry.sharedTokensComputed === 0 && entry.sharedPrefixSeconds < entry.cacheMissSeconds));
  assert.deepEqual([artifact.partialPrefixParity.prefixTokensReused, artifact.partialPrefixParity.tokensComputed], [8, 1]);
  assert.equal(artifact.partialPrefixParity.cachedGeneratedTokenId, artifact.partialPrefixParity.cleanGeneratedTokenId);
  assert.equal(artifact.partialPrefixParity.cachedTerminalLogitsSha256, artifact.partialPrefixParity.cleanTerminalLogitsSha256);
  assert.deepEqual(artifact.liveOriginalComparison.compiledGeneratedTokenIds, artifact.liveOriginalComparison.baselineGeneratedTokenIds);
  assert.equal(artifact.liveOriginalComparison.tokensEqual, true);
  assert.equal(artifact.decision.selectedPolicy, "cross-session-longest-exact-token-prefix");
});

test("evidência promove a fronteira Q8 0-19 com menos fallbacks e paridade integral", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-q8-layer-boundary-promotion-32x8.json", "utf8")) as any;
  assert.deepEqual([artifact.configuration.prompts, artifact.configuration.tokensPerPrompt, artifact.configuration.decoderQuantization, artifact.configuration.verificationMargin], [32, 8, "q8-ffn-gate-up", 0]);
  assert.deepEqual([artifact.promoted.quantizedLayers, artifact.promoted.promptsEqual, artifact.promoted.equalTokenSteps, artifact.promoted.comparedTokenSteps, artifact.promoted.rootDivergences], ["0-19", 32, 256, 256, 0]);
  assert.ok(artifact.promoted.fallbackPrompts < artifact.previous.fallbackPrompts);
  assert.ok(artifact.promoted.directVsBaselineThroughputRatio > 1);
  assert.equal(artifact.decision.selectedQuantizedLayers, "0-19");
  const digest = createHash("sha256").update(await readFile(`artifacts/${artifact.sources.prompts.file}`)).digest("hex");
  assert.equal(digest, artifact.sources.prompts.sha256);
});

test("evidência rejeita fronteiras Q8 esparsas que reduzem fallbacks mas pioram o híbrido", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-sparse-q8-boundary-rejection-32x8.json", "utf8")) as any;
  assert.equal(artifact.lateBlockSweep.every((entry: any) => entry.silentDivergences > 0), true);
  assert.equal(artifact.singleLayerRemovalSweep.testedRemovedLayers, 20);
  assert.equal(artifact.singleLayerRemovalSweep.safeCandidates.every((entry: any) => entry.silentDivergences === 0), true);
  const { current, candidate, rejectedCandidate } = artifact.hybridPairedControl;
  assert.deepEqual([current.promptsEqual, current.equalTokenSteps, candidate.promptsEqual, candidate.equalTokenSteps, rejectedCandidate.promptsEqual, rejectedCandidate.equalTokenSteps], [32, 256, 32, 256, 32, 256]);
  assert.ok(candidate.fallbackPrompts < current.fallbackPrompts);
  assert.ok(candidate.meanTokensPerSecond < current.meanTokensPerSecond);
  assert.ok(candidate.meanVerificationSeconds > current.meanVerificationSeconds);
  assert.ok(rejectedCandidate.tokensPerSecond < current.meanTokensPerSecond);
  assert.equal(artifact.decision.selectedQuantizedLayers, "0-19");
  const digest = createHash("sha256").update(await readFile(`artifacts/${artifact.sources.prompts.file}`)).digest("hex");
  assert.equal(digest, artifact.sources.prompts.sha256);
});

test("evidência rejeita Q8 na projeção down quando velocidade e verificação deixam de coexistir", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-q8-down-projection-rejection-32x8.json", "utf8")) as any;
  const coupled = artifact.candidates.find((entry: any) => entry.id === "coupled-gate-up-down-0-19");
  const safeBlock = artifact.candidates.find((entry: any) => entry.id === "orthogonal-down-5-9");
  const combined = artifact.candidates.find((entry: any) => entry.id === "orthogonal-down-5-9-30-34");
  assert.ok(coupled.fastPath.repetitions.every((entry: any) => entry.tokensPerSecond > artifact.controlFastPath.repetitions[1].tokensPerSecond));
  assert.ok(coupled.hybrid.equalTokenSteps < coupled.hybrid.comparedTokenSteps);
  assert.ok(coupled.hybrid.uncoveredPositiveMarginDivergences.length > 0);
  assert.equal(safeBlock.fastPath.uncoveredPositiveMarginDivergences.length, 0);
  assert.ok(safeBlock.fastPath.tokensPerSecond < artifact.controlFastPath.repetitions[1].tokensPerSecond);
  assert.ok(combined.fastPath.uncoveredPositiveMarginDivergences.length > 0);
  assert.deepEqual([artifact.decision.selectedGateUpLayers, artifact.decision.selectedDownLayers, artifact.decision.experimentalRuntimeCodeRetained], ["0-19", "off", false]);
  const digest = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");
  assert.equal(await digest(`artifacts/${artifact.sources.prompts.file}`), artifact.sources.prompts.sha256);
  assert.equal(await digest(`artifacts/${artifact.sources.authoritativeTokenIds.file}`), artifact.sources.authoritativeTokenIds.sha256);
});

test("evidência rejeita grupos Q8 que divergem ou ampliam o fallback", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-q8-group-size-rejection-32x8.json", "utf8")) as any;
  const digest = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");
  assert.equal(await digest(`artifacts/${artifact.sources.compiledBundleManifest.file}`), artifact.sources.compiledBundleManifest.sha256);
  assert.equal(await digest(`artifacts/${artifact.sources.authoritativeTokenIds.file}`), artifact.sources.authoritativeTokenIds.sha256);
  assert.equal(artifact.warmSmoke.group32.firstAuthoritativeDivergenceStep, 2);
  assert.ok(artifact.warmSmoke.group32.meanElapsedSeconds > artifact.warmSmoke.group64.meanElapsedSeconds);
  assert.equal(artifact.fastPathMatrix.group128.uncoveredDivergence.candidateGreedyMargin, 0.125);
  assert.ok(artifact.selectedRuntimeMatrix.unsafeCandidate.equalTokenSteps < artifact.selectedRuntimeMatrix.unsafeCandidate.comparedTokenSteps);
  assert.deepEqual([artifact.selectedRuntimeMatrix.control.equalTokenSteps, artifact.selectedRuntimeMatrix.safeCandidate.equalTokenSteps], [256, 256]);
  assert.ok(artifact.selectedRuntimeMatrix.safeCandidate.fallbackPrompts > artifact.selectedRuntimeMatrix.control.fallbackPrompts);
  assert.ok(artifact.selectedRuntimeMatrix.safeCandidate.tokensPerSecond < artifact.selectedRuntimeMatrix.control.tokensPerSecond);
  assert.deepEqual([artifact.decision.selectedGroupSize, artifact.decision.rejectedGroupSizes], [64, [32, 128]]);
});

test("calibração três-vias agrega acordo token a token e throughput", () => {
  const report = summarizeGemma4ThreeWayCalibration([baseCase], { maxNewTokens: 2, requestThreads: 1, precision: "f32", roundingPolicy: "none", runner: { source: "/model", python: "python", helper: "helper", directThreads: 4 } }, { ready: true });
  assert.equal(report.configuration.directMlxHeadQuantization, "off");
  assert.equal(report.configuration.directMlxDecoderQuantization, "q8-ffn-gate-up");
  assert.deepEqual(report.summary, {
    promptsThreeWayEqual: 0, promptThreeWayAgreementRate: 0,
    compatibilityPromptsEqual: 0, compatibilityPromptAgreementRate: 0,
    directPromptsEqual: 1, directPromptAgreementRate: 1, directFallbackPrompts: 0, directFallbackRate: 0, directControlCorrectionPrompts: 0, directControlVerifiedTokenSteps: 0, comparedTokenSteps: 2,
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
  assert.deepEqual(report.cases[0]?.direct, { tokenIds: [7, 8], text: "a", tokensEqualBaseline: true, firstDivergentStep: null, seconds: 1, tokensPerSecond: 2, linearThreads: 4, parallelExecutionBackend: "metal-vectorized-kernels", configuredHostThreads: 4, hostThreadSettingApplied: false, metalCommandStreams: 1, parallelTerminalOutputDimensions: 262_144, fusedDecoderStackEpilogueDispatches: 2, residentGeneration: "on", fusedTokenGenerationDispatches: 1, externalForwardRequests: 1, kvCacheTransportBytes: 0, residentKvBytes: 512, terminalLogitMaterializations: 1, gpuRankedTokenSteps: 2, fullLogitTransfersAvoided: 1, terminalLogitVectorBytes: 1024, ropeFactorBuilds: 2, ropeFactorBuildsAvoided: 82, topologyMaskBuilds: 4, topologyMaskBuildsAvoided: 80, redundantLogitFiniteScansAvoided: 2, kvPrefixValidationScansAvoided: 14, decoderLayerValidityScansAvoided: 41, compiledIncrementalDecoderSteps: 1, incrementalCompilerCacheHit: true, fusedDecoderStackGateUpPairs: 42, fusedDecoderStackWidenedCacheHits: 588, widenedTensorCacheEntries: 294, widenedTensorCacheBytes: 1024, fusedDecoderStackAttentionSeconds: 0.2, fusedDecoderStackFfnSeconds: 0.5, fusedDecoderStackPleSeconds: 0.1, forwardSeconds: [0.6, 0.4] });
});

test("calibração preserva métricas da cabeça quantizada certificada", () => {
  const report = summarizeGemma4ThreeWayCalibration([
    { ...baseCase, direct: { ...baseCase.direct, tokenPreludeCacheHits: 2, tokenPreludeCacheMisses: 1, prefillExecutionTokens: 8, prefillPaddingTokens: 2, prefillSeconds: 0.6, incrementalDecoderSeconds: 0.3, tokenPreludeSeconds: 0.04, compiledDecoderGraphSeconds: 0.25, tokenSelectionSeconds: 0.09, terminalLogitTransferSeconds: 0.01, mlxHeadQuantizationStrategy: "two-stage-residual-affine-certified-v1" as const, mlxDecoderQuantizationStrategy: "single-stage-affine-q8-calibrated-v1" as const, mlxDecoderGateUpProjectionStrategy: "concatenated-affine-q8-v1" as const, mlxDecoderGateUpSplitStrategy: "static-index-take-v1" as const, quantizedHeadCertifiedSteps: 1, quantizedHeadExactFallbackSteps: 1, verificationDecoderSteps: 1, verificationDecoderStepsAvoided: 1, verificationHeadPositionsComputed: 1, verificationHeadPositionsAvoided: 5, verificationPrefillAhead: true, verificationPrefixAhead: true, verificationPrefixAheadStep: 0, verificationPrefixAheadTokenSteps: 1, verificationPrefillAheadSeconds: 0.4, verificationPrefillAheadWorkerSeconds: 0.39, verificationPrefillOverlapSeconds: 0.3, verificationPrefillWaitSeconds: 0.1, verificationPrefillAheadCacheHit: false, verificationPrefillAheadPrefixTokensReused: 0, verificationPrefillAheadTokensComputed: 6, verificationEarlyExitStep: 0, verificationSessionCacheHit: true, verificationPrefixTokensReused: 4, verificationPrefillTokensComputed: 2, verificationCachedContextTokens: 6, verificationResidentKvBytes: 1024, verificationCachedSessions: 1 } },
  ], { maxNewTokens: 2, requestThreads: 1, precision: "f32", roundingPolicy: "none", runner: { source: "/model", python: "python", helper: "helper", directThreads: 4 } }, { ready: true });
  const direct = report.cases[0]?.direct as { tokenPreludeCacheHits?: number; tokenPreludeCacheMisses?: number; prefillExecutionTokens?: number; prefillPaddingTokens?: number; prefillSeconds?: number; incrementalDecoderSeconds?: number; tokenPreludeSeconds?: number; compiledDecoderGraphSeconds?: number; tokenSelectionSeconds?: number; terminalLogitTransferSeconds?: number; mlxHeadQuantizationStrategy?: string; mlxDecoderQuantizationStrategy?: string; mlxDecoderGateUpProjectionStrategy?: string; mlxDecoderGateUpSplitStrategy?: string; quantizedHeadCertifiedSteps?: number; quantizedHeadExactFallbackSteps?: number; verificationDecoderSteps?: number; verificationDecoderStepsAvoided?: number; verificationHeadPositionsComputed?: number; verificationHeadPositionsAvoided?: number; verificationPrefillAhead?: boolean; verificationPrefixAhead?: boolean; verificationPrefixAheadStep?: number; verificationPrefixAheadTokenSteps?: number; verificationPrefillAheadSeconds?: number; verificationPrefillAheadWorkerSeconds?: number; verificationPrefillOverlapSeconds?: number; verificationPrefillWaitSeconds?: number; verificationPrefillAheadCacheHit?: boolean; verificationPrefillAheadPrefixTokensReused?: number; verificationPrefillAheadTokensComputed?: number; verificationEarlyExitStep?: number; verificationSessionCacheHit?: boolean; verificationPrefixTokensReused?: number; verificationPrefillTokensComputed?: number; verificationCachedContextTokens?: number; verificationResidentKvBytes?: number; verificationCachedSessions?: number };
  assert.equal(direct.mlxHeadQuantizationStrategy, "two-stage-residual-affine-certified-v1");
  assert.equal(direct.mlxDecoderQuantizationStrategy, "single-stage-affine-q8-calibrated-v1");
  assert.equal(direct.mlxDecoderGateUpProjectionStrategy, "concatenated-affine-q8-v1");
  assert.equal(direct.mlxDecoderGateUpSplitStrategy, "static-index-take-v1");
  assert.equal(direct.quantizedHeadCertifiedSteps, 1);
  assert.equal(direct.quantizedHeadExactFallbackSteps, 1);
  assert.deepEqual([direct.verificationDecoderSteps, direct.verificationDecoderStepsAvoided, direct.verificationEarlyExitStep], [1, 1, 0]);
  assert.deepEqual([direct.verificationHeadPositionsComputed, direct.verificationHeadPositionsAvoided], [1, 5]);
  assert.deepEqual([direct.verificationPrefillAhead, direct.verificationPrefillAheadSeconds, direct.verificationPrefillAheadWorkerSeconds, direct.verificationPrefillOverlapSeconds, direct.verificationPrefillWaitSeconds, direct.verificationPrefillAheadCacheHit, direct.verificationPrefillAheadPrefixTokensReused, direct.verificationPrefillAheadTokensComputed], [true, 0.4, 0.39, 0.3, 0.1, false, 0, 6]);
  assert.deepEqual([direct.verificationPrefixAhead, direct.verificationPrefixAheadStep, direct.verificationPrefixAheadTokenSteps], [true, 0, 1]);
  assert.deepEqual([direct.verificationSessionCacheHit, direct.verificationPrefixTokensReused, direct.verificationPrefillTokensComputed, direct.verificationCachedContextTokens, direct.verificationResidentKvBytes, direct.verificationCachedSessions], [true, 4, 2, 6, 1024, 1]);
  assert.deepEqual([direct.prefillSeconds, direct.incrementalDecoderSeconds, direct.tokenSelectionSeconds, direct.terminalLogitTransferSeconds], [0.6, 0.3, 0.09, 0.01]);
  assert.deepEqual([direct.prefillExecutionTokens, direct.prefillPaddingTokens], [8, 2]);
  assert.deepEqual([direct.tokenPreludeCacheHits, direct.tokenPreludeCacheMisses, direct.tokenPreludeSeconds, direct.compiledDecoderGraphSeconds], [2, 1, 0.04, 0.25]);
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
