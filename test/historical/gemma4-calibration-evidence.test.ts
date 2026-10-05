import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

// Historical benchmark evidence. Execute explicitly with npm run test:historical.
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

test("evidência promove contexto sensível exato sem replay do decoder", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-exact-sensitive-context-promotion.json", "utf8")) as any;
  const digest = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");
  for (const source of [artifact.sources.sessionCache, artifact.sources.verificationWorker, artifact.sources.comparisonServer, artifact.sources.runtimeIndex]) assert.equal(await digest(source.file), source.sha256);
  assert.equal(artifact.isolatedExactWorker.controlTerminalLogitsSha256, artifact.isolatedExactWorker.candidateTerminalLogitsSha256);
  assert.ok(artifact.isolatedExactWorker.candidate.every((entry: any) => entry.exactContextCacheHit && entry.trustedPrefixStepsReused === 3 && entry.decoderSteps === 0 && entry.headSteps === 1));
  assert.ok(artifact.isolatedExactWorker.candidateMeanSeconds < artifact.isolatedExactWorker.controlMeanSeconds);
  assert.ok(artifact.isolatedExactWorker.latencyReductionRate > 0.9);
  assert.deepEqual(artifact.liveOriginalComparison.compiledGeneratedTokenIds, artifact.liveOriginalComparison.baselineGeneratedTokenIds);
  assert.ok(artifact.liveOriginalComparison.compiledDirectSeconds < artifact.liveOriginalComparison.baselineModelSeconds);
  assert.equal(artifact.decision.selectedPolicy, "reuse-exact-sensitive-context-before-head");
});

test("evidência adia prefill exato sem cache e preserva antecipação do contexto exato", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-uncached-prefill-defer-promotion.json", "utf8")) as any;
  const digest = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");
  for (const source of Object.values(artifact.sources) as Array<{ file: string; sha256: string }>) assert.equal(await digest(source.file), source.sha256);
  assert.equal(artifact.pairedBenchmark.allGeneratedTokensEqual, true);
  assert.equal(artifact.pairedBenchmark.allTerminalLogitsSha256Equal, true);
  assert.ok(artifact.pairedBenchmark.candidate.meanElapsedSeconds < artifact.pairedBenchmark.control.meanElapsedSeconds);
  assert.ok(artifact.pairedBenchmark.meanWallReductionRate > 0.2);
  assert.deepEqual(artifact.pairedBenchmark.candidate.uncachedPrefillDeferred, [true, true]);
  assert.equal(artifact.cachedExactContext.verificationPrefixAhead, true);
  assert.equal(artifact.cachedExactContext.verificationExactContextCacheHit, true);
  assert.equal(artifact.cachedExactContext.verificationDecoderSteps, 0);
  assert.deepEqual(artifact.authoritativeComparison.compiledGeneratedTokenIds, artifact.authoritativeComparison.baselineGeneratedTokenIds);
  assert.equal(artifact.authoritativeComparison.tokensEqual, true);
  assert.equal(artifact.decision.selectedDefault, "defer-uncached-prefill");
});

test("evidência promove cabeça BF16 integral e rejeita prefill agrupado divergente", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-verification-whole-head-promotion.json", "utf8")) as any;
  const digest = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");
  for (const source of Object.values(artifact.sources) as Array<{ file: string; sha256: string }>) assert.equal(await digest(source.file), source.sha256);
  assert.equal(artifact.pairedBenchmark.allGeneratedTokensEqual, true);
  assert.equal(artifact.pairedBenchmark.allTerminalLogitsSha256Equal, true);
  assert.ok(artifact.pairedBenchmark.candidate.meanElapsedSeconds < artifact.pairedBenchmark.control.meanElapsedSeconds);
  assert.ok(artifact.pairedBenchmark.meanWallReductionRate > 0.25);
  assert.deepEqual([artifact.authoritativeCorpus.promptsEqual, artifact.authoritativeCorpus.prompts, artifact.authoritativeCorpus.equalTokenSteps, artifact.authoritativeCorpus.comparedTokenSteps], [5, 5, 40, 40]);
  assert.ok(artifact.authoritativeCorpus.cases.every((entry: any) => JSON.stringify(entry.compiledTokenIds) === JSON.stringify(entry.baselineTokenIds)));
  assert.equal(artifact.rejectedTrustedPrefixBatch.rejected, true);
  assert.notDeepEqual(artifact.rejectedTrustedPrefixBatch.candidateTokenIds, artifact.rejectedTrustedPrefixBatch.baselineTokenIds);
  assert.equal(artifact.rejectedTrustedPrefixBatch.firstDivergentStep, 3);
  assert.equal(artifact.decision.selectedDefault, "native-bf16-whole");
});

test("evidência promove cache de decisão exata e rejeita fronteira Q8 0-18", async () => {
  const artifact = JSON.parse(await readFile("artifacts/gemma4-exact-decision-cache-promotion.json", "utf8")) as any;
  const digest = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");
  for (const source of Object.values(artifact.sources) as Array<{ file: string; sha256: string }>) assert.equal(await digest(source.file), source.sha256);
  assert.deepEqual([artifact.rejectedQuantizationBoundary.control.promptsEqual, artifact.rejectedQuantizationBoundary.control.equalTokenSteps], [32, 256]);
  assert.deepEqual([artifact.rejectedQuantizationBoundary.candidate.promptsEqual, artifact.rejectedQuantizationBoundary.candidate.equalTokenSteps], [32, 256]);
  assert.ok(artifact.rejectedQuantizationBoundary.candidate.fallbacks > artifact.rejectedQuantizationBoundary.control.fallbacks);
  assert.ok(artifact.rejectedQuantizationBoundary.candidate.directSeconds > artifact.rejectedQuantizationBoundary.control.directSeconds);
  assert.equal(artifact.rejectedQuantizationBoundary.rejected, true);
  assert.equal(artifact.repeatedExactConfirmation.tokensEqual, true);
  assert.deepEqual(artifact.repeatedExactConfirmation.repeatedAcrossSession, {
    elapsedSeconds: 0.3910765410000004,
    selectedBackend: "exact-decision-cache",
    selectionPolicy: "margin-verified-exact-decision-cache-v1",
    exactDecisionCacheHit: true,
    exactDecisionCacheMatchedSteps: 1,
    verificationRequestCount: 0,
    verificationRequestWallSeconds: 0,
    verificationDecoderSteps: 0,
    verificationHeadPositionsComputed: 0,
  });
  assert.ok(artifact.repeatedExactConfirmation.selectedComputeSpeedup > 7);
  assert.equal(artifact.negativeCorrectionControl.repeatedExactDecisionCacheHit, false);
  assert.equal(artifact.negativeCorrectionControl.correctionPreserved, true);
  assert.notDeepEqual(artifact.negativeCorrectionControl.selectedGeneratedTokenIds, artifact.negativeCorrectionControl.fastGeneratedTokenIds);
  assert.equal(artifact.decision.selectedPolicy, "exact-token-context-decision-lru");
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
