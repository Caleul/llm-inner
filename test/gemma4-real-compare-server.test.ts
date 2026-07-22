import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Script } from "node:vm";
import { gemma4RealCompareHtml, parseGemma4PromptMatrixInput, projectGemma4PrimaryComparison, summarizeGemma4PromptMatrix } from "../src/gemma4-real-compare-ui.js";
import { GEMMA4_FINAL_FORMULA_NODE_SEMANTICS, writeGemma4FinalFormulaRuntime } from "../src/gemma4-final-formula-runtime.js";
import { assessDirectVerification, assertDirectFinalFormulaProgram, attachDirectFallbackOutcome, computeDirectExecutionMetrics, computeDirectLogitAgreement, createGemma4RealComparisonServer, hasVerificationCachedPrefix, parseGemma4RealServerOptions, shouldStartDirectVerificationPrefill, type Gemma4CompiledProgramStatus } from "../src/gemma4-real-compare-server.js";

test("interface diferencial contém controles e apresentação dos dois executores", () => {
  assert.match(gemma4RealCompareHtml, /Enviar e comparar/);
  assert.match(gemma4RealCompareHtml, /Original — BF16/);
  assert.match(gemma4RealCompareHtml, /LLM compilada — funções finais vetorizadas/);
  assert.match(gemma4RealCompareHtml, /Tokens original \/ compilada/);
  assert.match(gemma4RealCompareHtml, /Diagnóstico histórico de compatibilidade/);
  assert.match(gemma4RealCompareHtml, /linearBackend/);
  assert.match(gemma4RealCompareHtml, /lotes lineares/);
  assert.match(gemma4RealCompareHtml, /MLPs fundidos/);
  assert.match(gemma4RealCompareHtml, /projeção gate\/up/);
  assert.match(gemma4RealCompareHtml, /head BF16 exato: sem aproximação ou fallback/);
  assert.match(gemma4RealCompareHtml, /head Q8 rápido: shortlist top-16 refinada com pesos BF16 exatos \(experimental\)/);
  assert.match(gemma4RealCompareHtml, /runtime MLX calibrado/);
  assert.match(gemma4RealCompareHtml, /programa final direto/);
  assert.match(gemma4RealCompareHtml, /mapa executado/);
  assert.match(gemma4RealCompareHtml, /Inspetor das dimensões finais compiladas/);
  assert.match(gemma4RealCompareHtml, /Dimensão \/ token ID/);
  assert.match(gemma4RealCompareHtml, /\/api\/final-formula/);
  assert.match(gemma4RealCompareHtml, /EVAL_EXACT_DAG/);
  assert.match(gemma4RealCompareHtml, /finalFormulaRuntime/);
  assert.match(gemma4RealCompareHtml, /única entrada variável/);
  assert.match(gemma4RealCompareHtml, /despacho lógico\/forward/);
  assert.match(gemma4RealCompareHtml, /dimensões finais vinculadas/);
  assert.match(gemma4RealCompareHtml, /FFNs completos/);
  assert.match(gemma4RealCompareHtml, /Decoder layers completas/);
  assert.match(gemma4RealCompareHtml, /Pilhas decoder completas/);
  assert.match(gemma4RealCompareHtml, /pilhas até logits/);
  assert.match(gemma4RealCompareHtml, /gerações residentes/);
  assert.match(gemma4RealCompareHtml, /chamadas externas/);
  assert.match(gemma4RealCompareHtml, /transporte KV/);
  assert.match(gemma4RealCompareHtml, /Nova conversa/);
  assert.match(gemma4RealCompareHtml, /Cancelar/);
  assert.match(gemma4RealCompareHtml, /Conclusão bruta \(fiel\)/);
  assert.match(gemma4RealCompareHtml, /Chat IT \(contrato oficial Gemma 4\)/);
  assert.match(gemma4RealCompareHtml, /não transforma pesos base em pesos instruction-tuned/);
  assert.match(gemma4RealCompareHtml, /razão compilada\/original/);
  assert.match(gemma4RealCompareHtml, /Executor direto ausente: gere ou informe o bundle compilado/);
  assert.match(gemma4RealCompareHtml, /clearComparison/);
  assert.match(gemma4RealCompareHtml, /AbortController/);
  assert.match(gemma4RealCompareHtml, /Δlogit \/ Δmargem \/ top-K/);
  assert.match(gemma4RealCompareHtml, /divergência direta top-K/);
  assert.match(gemma4RealCompareHtml, /Original e LLM compilada geraram exatamente os mesmos tokens/);
  assert.match(gemma4RealCompareHtml, /Este resultado não altera o veredito da LLM compilada/);
  assert.match(gemma4RealCompareHtml, /prefixo KV/);
  assert.match(gemma4RealCompareHtml, /Primeiro token/);
  assert.match(gemma4RealCompareHtml, /prefillAvoidedRate/);
  assert.match(gemma4RealCompareHtml, /\/api\/compare-stream/);
  assert.match(gemma4RealCompareHtml, /\/api\/generate-stream/);
  assert.match(gemma4RealCompareHtml, /Somente LLM compilada/);
  assert.match(gemma4RealCompareHtml, /referência permaneceu descarregada/);
  assert.match(gemma4RealCompareHtml, /Aquecendo o forward compilado no Metal/);
  assert.match(gemma4RealCompareHtml, /gate\+up unidos/);
  assert.match(gemma4RealCompareHtml, /cache de constantes F32/);
  assert.match(gemma4RealCompareHtml, /ranking final GPU/);
  assert.match(gemma4RealCompareHtml, /fullLogitTransfersAvoided/);
  assert.match(gemma4RealCompareHtml, /CSE RoPE/);
  assert.match(gemma4RealCompareHtml, /ropeFactorBuildsAvoided/);
  assert.match(gemma4RealCompareHtml, /validações redundantes evitadas/);
  assert.match(gemma4RealCompareHtml, /kvPrefixValidationScansAvoided/);
  assert.match(gemma4RealCompareHtml, /decoderLayerValidityScansAvoided/);
  assert.match(gemma4RealCompareHtml, /prefillExecutionTokens/);
  assert.match(gemma4RealCompareHtml, /parallelExecutionBackend/);
  assert.match(gemma4RealCompareHtml, /paralelismo compilado/);
  assert.match(gemma4RealCompareHtml, /decoder→logits compilado/);
  assert.match(gemma4RealCompareHtml, /compiledIncrementalDecoderSteps/);
  assert.match(gemma4RealCompareHtml, /incrementalCompilerCacheHit/);
  assert.match(gemma4RealCompareHtml, /Isolada \(fiel\)/);
  assert.match(gemma4RealCompareHtml, /Paralela \(stress\)/);
  assert.match(gemma4RealCompareHtml, /direct-complete/);
  assert.match(gemma4RealCompareHtml, /message\.generatedText/);
  assert.match(gemma4RealCompareHtml, /assistantNode\.textContent=message\.data\.generatedText/);
  assert.match(gemma4RealCompareHtml, /reaquecimento Metal agendado em segundo plano/);
  assert.match(gemma4RealCompareHtml, /directBackgroundRecoveryWaitSeconds/);
  assert.match(gemma4RealCompareHtml, /reference-loading/);
  assert.match(gemma4RealCompareHtml, /direct-rewarming/);
  assert.match(gemma4RealCompareHtml, /modelo original ainda não carregado/);
  assert.match(gemma4RealCompareHtml, /verifier\?\.enabled===false\?'desabilitado'/);
  assert.match(gemma4RealCompareHtml, /parede até o resultado/);
  assert.match(gemma4RealCompareHtml, /fases stack attn\/FFN\/PLE/);
  assert.match(gemma4RealCompareHtml, /fases stack: fundidas no grafo Metal/);
  assert.match(gemma4RealCompareHtml, /fases compiladas prefill\/decoder incremental\/prelude token\/grafo decoder\/seleção\/transferência terminal/);
  assert.match(gemma4RealCompareHtml, /cache prelude token/);
  assert.match(gemma4RealCompareHtml, /PLEs fundidos/);
  assert.match(gemma4RealCompareHtml, /head:/);
  assert.match(gemma4RealCompareHtml, /attention core:/);
  assert.match(gemma4RealCompareHtml, /attention fused:/);
  assert.match(gemma4RealCompareHtml, /tempo worker ref\/head\/attn\/MLP\/FFN\/layer\/stack\/PLE/);
  assert.match(gemma4RealCompareHtml, /heads verificados/);
  assert.match(gemma4RealCompareHtml, /bundle compilado autocontido/);
  assert.match(gemma4RealCompareHtml, /worker autenticou as/);
  assert.match(gemma4RealCompareHtml, /Threads/);
  assert.match(gemma4RealCompareHtml, /\/api\/compare/);
  assert.match(gemma4RealCompareHtml, /Matriz reproduzível de prompts/);
  assert.match(gemma4RealCompareHtml, /function finiteNonNegative/);
  assert.match(gemma4RealCompareHtml, /Executar matriz/);
  assert.match(gemma4RealCompareHtml, /passos executados/);
  assert.match(gemma4RealCompareHtml, /Baixar relatório JSON/);
  assert.match(gemma4RealCompareHtml, /summarizeGemma4PromptMatrix/);
  assert.match(gemma4RealCompareHtml, /melhorou.*piorou.*continuou divergente/);
  const embedded = gemma4RealCompareHtml.match(/<script>([\s\S]*)<\/script>/)?.[1]; assert.ok(embedded); assert.doesNotThrow(() => new Script(embedded), "JavaScript embutido deve ser sintaticamente executável pelo navegador");
});

test("interface projeta o veredito somente de original × LLM compilada", () => {
  const report = {
    baselineGeneratedText: "original",
    baselineGeneratedTokenIds: [1, 2],
    candidateGeneratedText: "legado divergente",
    candidateGeneratedTokenIds: [1, 9],
    candidatePrecision: "f32",
    roundingPolicy: "none",
    direct: { generatedText: "compilado", generatedTokenIds: [1, 2] },
  };
  assert.deepEqual(projectGemma4PrimaryComparison(report), {
    baselineText: "original", compiledText: "compilado", baselineTokenIds: [1, 2], compiledTokenIds: [1, 2], tokensEqual: true, firstDivergentStep: null,
    compatibility: { text: "legado divergente", tokenIds: [1, 9], tokensEqual: false, firstDivergentStep: 1, precision: "f32", roundingPolicy: "none" },
  });
  const divergent = projectGemma4PrimaryComparison({ ...report, direct: { generatedText: "compilado divergente", generatedTokenIds: [1, 3] } });
  assert.equal(divergent.tokensEqual, false); assert.equal(divergent.firstDivergentStep, 1);
  assert.throws(() => projectGemma4PrimaryComparison({ ...report, direct: undefined }), /não contém as saídas/);
});

test("matriz da interface aceita linhas ou JSON e rejeita entradas ambíguas", () => {
  assert.deepEqual(parseGemma4PromptMatrixInput("um\n\ndois"), ["um", "dois"]);
  assert.deepEqual(parseGemma4PromptMatrixInput('["linha 1\\nlinha 2", "  preservar  "]'), ["linha 1\nlinha 2", "  preservar  "]);
  assert.throws(() => parseGemma4PromptMatrixInput(""), /entre 1 e 32/);
  assert.throws(() => parseGemma4PromptMatrixInput("["), /JSON.*inválido/);
  assert.throws(() => parseGemma4PromptMatrixInput(JSON.stringify(Array.from({ length: 33 }, (_, index) => String(index)))), /entre 1 e 32/);
});

test("matriz da interface agrega paridade, desempenho e divergência de logits sem esconder casos", () => {
  const report = summarizeGemma4PromptMatrix([
    { prompt: "igual", baselineGeneratedTokenIds: [7, 8], performance: { baselineSeconds: 2 }, directExecutionMetrics: { firstTokenWallSeconds: 0.1 }, direct: { generatedTokenIds: [7, 8], elapsedSeconds: 0.5, selectedBackend: "mlx", fallbackTriggered: false, logitAgreement: { steps: [{ contextsEqualBeforeStep: true, baselineArgmaxToken: 7, directArgmaxToken: 7, topKOverlapRate: 1, baselineArgmaxLogitAbsError: 0.25 }] } } },
    { prompt: "diverge", baselineGeneratedTokenIds: [9, 10], performance: { baselineSeconds: 3 }, directExecutionMetrics: { firstTokenForwardSeconds: 0.2 }, direct: { generatedTokenIds: [9, 11], elapsedSeconds: 1, selectedBackend: "pytorch", fallbackTriggered: true, fallbackChangedTokens: true, fallbackOutcome: "worsened", logitAgreement: { steps: [{ contextsEqualBeforeStep: true, baselineArgmaxToken: 10, directArgmaxToken: 11, topKOverlapRate: 0.5, baselineArgmaxLogitAbsError: 0.75 }, { contextsEqualBeforeStep: false, baselineArgmaxToken: 1, directArgmaxToken: 2 }] } } },
  ]);
  assert.deepEqual({ ...report.summary, meanFirstTokenWallSeconds: undefined }, { prompts: 2, promptsEqual: 1, promptAgreementRate: 0.5, comparedTokenSteps: 4, equalTokenSteps: 3, tokenAgreementRate: 0.75, firstDivergentPrompt: 1, fallbackPrompts: 1, fallbackRate: 0.5, correctionPrompts: 1, fallbackImprovedPrompts: 0, fallbackWorsenedPrompts: 1, fallbackChangedStillDivergentPrompts: 0, verificationEarlyExitPrompts: 0, verificationDecoderSteps: 0, verificationDecoderStepsAvoided: 0, verificationHeadPositionsComputed: 0, verificationHeadPositionsAvoided: 0, verificationPrefillAheadPrompts: 0, verificationPrefixAheadPrompts: 0, verificationExactContextHitPrompts: 0, verificationTrustedPrefixStepsReused: 0, verificationPrefillSkippedTerminalPrompts: 0, verificationUncachedPrefillDeferredPrompts: 0, verificationPrefillAheadSeconds: 0, verificationPrefillOverlapSeconds: 0, verificationPrefillWaitSeconds: 0, verificationSessionCacheHitPrompts: 0, verificationSharedPrefixHitPrompts: 0, verificationPrefixTokensReused: 0, verificationPrefillTokensComputed: 0, compiledContinuationAcceptedPrompts: 0, compiledContinuationTokenSteps: 0, compiledContinuationSeconds: 0, compiledContinuationPrefixRetryPrompts: 0, compiledContinuationPrefixAttemptSeconds: 0, baselineSeconds: 5, directSeconds: 1.5, baselineTokensPerSecond: 0.8, directTokensPerSecond: 4 / 1.5, directVsBaselineThroughputRatio: 5 / 1.5, meanFirstTokenWallSeconds: undefined, directRootDivergences: 1, meanTopKOverlapRate: 0.75, maxBaselineArgmaxLogitAbsError: 0.75 });
  assert.ok(Math.abs(report.summary.meanFirstTokenWallSeconds! - 0.15) < Number.EPSILON);
  assert.equal(report.cases[1]?.firstDivergentStep, 1); assert.equal(report.cases[1]?.selectedBackend, "pytorch"); assert.equal(report.cases[1]?.fallbackTriggered, true); assert.equal(report.reports.length, 2);
  const lengthMismatch = summarizeGemma4PromptMatrix([{ prompt: "EOS desigual", baselineGeneratedTokenIds: [1], performance: { baselineSeconds: 1 }, direct: { generatedTokenIds: [1, 2], elapsedSeconds: 1 } }]);
  assert.equal(lengthMismatch.summary.comparedTokenSteps, 2); assert.equal(lengthMismatch.summary.equalTokenSteps, 1); assert.equal(lengthMismatch.summary.tokenAgreementRate, 0.5); assert.equal(lengthMismatch.cases[0]?.tokensEqual, false);
  const earlyExit = summarizeGemma4PromptMatrix([{ prompt: "empate confirmado", baselineGeneratedTokenIds: [1, 2], performance: { baselineSeconds: 1 }, direct: { generatedTokenIds: [1, 2], elapsedSeconds: 0.5, fallbackTriggered: true, verificationDecoderSteps: 1, verificationDecoderStepsAvoided: 1, verificationHeadPositionsComputed: 1, verificationHeadPositionsAvoided: 3, verificationPrefillAhead: true, verificationPrefixAhead: true, verificationExactContextCacheHit: true, verificationTrustedPrefixStepsReused: 2, verificationUncachedPrefillDeferred: true, verificationPrefillAheadSeconds: 0.4, verificationPrefillOverlapSeconds: 0.3, verificationPrefillWaitSeconds: 0.1, verificationEarlyExitStep: 0, verificationSessionCacheHit: true, verificationCacheScope: "shared-prefix", verificationPrefixTokensReused: 4, verificationPrefillTokensComputed: 2, compiledContinuationAccepted: true, compiledContinuationTokenSteps: 1, compiledContinuationSeconds: 0.2, compiledContinuationMinimumMargin: 0.5, compiledContinuationPrefixRetry: true, compiledContinuationPrefixAttemptSeconds: 0.05, compiledContinuationPrefixAttemptMinimumMargin: 0 } }]);
  assert.deepEqual([earlyExit.summary.verificationEarlyExitPrompts, earlyExit.summary.verificationDecoderSteps, earlyExit.summary.verificationDecoderStepsAvoided, earlyExit.summary.verificationHeadPositionsComputed, earlyExit.summary.verificationHeadPositionsAvoided, earlyExit.summary.verificationPrefillAheadPrompts, earlyExit.summary.verificationPrefixAheadPrompts, earlyExit.summary.verificationUncachedPrefillDeferredPrompts, earlyExit.summary.verificationPrefillAheadSeconds, earlyExit.summary.verificationPrefillOverlapSeconds, earlyExit.summary.verificationPrefillWaitSeconds, earlyExit.summary.verificationSessionCacheHitPrompts, earlyExit.summary.verificationPrefixTokensReused, earlyExit.summary.verificationPrefillTokensComputed], [1, 1, 1, 1, 3, 1, 1, 1, 0.4, 0.3, 0.1, 1, 4, 2]);
  assert.equal(earlyExit.summary.verificationSharedPrefixHitPrompts, 1);
  assert.deepEqual([earlyExit.summary.verificationExactContextHitPrompts, earlyExit.summary.verificationTrustedPrefixStepsReused], [1, 2]);
  assert.deepEqual([earlyExit.summary.compiledContinuationAcceptedPrompts, earlyExit.summary.compiledContinuationTokenSteps, earlyExit.summary.compiledContinuationSeconds, earlyExit.cases[0]?.compiledContinuationMinimumMargin], [1, 1, 0.2, 0.5]);
  assert.deepEqual([earlyExit.summary.compiledContinuationPrefixRetryPrompts, earlyExit.summary.compiledContinuationPrefixAttemptSeconds, earlyExit.cases[0]?.compiledContinuationPrefixAttemptMinimumMargin], [1, 0.05, 0]);
  assert.throws(() => summarizeGemma4PromptMatrix([]), /ao menos um/);
  assert.throws(() => summarizeGemma4PromptMatrix([{ prompt: "x" }]), /comparação original.*compilada/);
});

test("avalia o efeito do fallback somente depois da referência sem usá-la na seleção", () => {
  const improved: any = { generatedTokenIds: [7], fullTokenIds: [2, 7], fallbackTriggered: true, fastPath: { generatedTokenIds: [8] } };
  attachDirectFallbackOutcome(improved, [7]);
  assert.deepEqual({ outcome: improved.fallbackOutcome, changed: improved.fallbackChangedTokens, fastEqual: improved.fastPathTokensEqualBaseline }, { outcome: "improved", changed: true, fastEqual: false });
  const worsened: any = { generatedTokenIds: [8], fullTokenIds: [2, 8], fallbackTriggered: true, fastPath: { generatedTokenIds: [7] } };
  attachDirectFallbackOutcome(worsened, [7]);
  assert.deepEqual({ outcome: worsened.fallbackOutcome, changed: worsened.fallbackChangedTokens, fastEqual: worsened.fastPathTokensEqualBaseline }, { outcome: "worsened", changed: true, fastEqual: true });
  const unchanged: any = { generatedTokenIds: [7], fullTokenIds: [2, 7], fallbackTriggered: true, fastPath: { generatedTokenIds: [7] } };
  attachDirectFallbackOutcome(unchanged, [7]);
  assert.equal(unchanged.fallbackOutcome, "confirmed-equal");
});

test("modo compilado exige que o worker autentique exatamente o mapa final", () => {
  const expected = { file: "final-formulas.json", sha256: "a".repeat(64), functions: 262_144, inputTensor: "x" as const, evaluator: "BF16_RNE(EVAL_EXACT_DAG(root,x))" as const, orderedRootsSha256: "b".repeat(64) };
  const runtime = { file: "final-formulas.runtime.json" as const, sha256: "c".repeat(64), schemaVersion: 1 as const, evaluator: "EVAL_EXACT_DAG" as const };
  assert.doesNotThrow(() => assertDirectFinalFormulaProgram({ finalFormulaProgram: { execution: "vectorized-shared-dag-output-program", evaluator: "EVAL_EXACT_DAG", runtimeContractSha256: "c".repeat(64), functions: 262_144, fileSha256: "a".repeat(64), orderedRootsSha256: "b".repeat(64) } }, expected, runtime));
  assert.throws(() => assertDirectFinalFormulaProgram({}, expected, runtime), /não autenticou/);
  assert.throws(() => assertDirectFinalFormulaProgram({ finalFormulaProgram: { execution: "vectorized-shared-dag-output-program", evaluator: "EVAL_EXACT_DAG", runtimeContractSha256: "c".repeat(64), functions: 262_143, fileSha256: "a".repeat(64), orderedRootsSha256: "b".repeat(64) } }, expected, runtime), /vínculo divergente/);
});

test("métricas diretas separam primeiro token, decode, speedup e prefill evitado", () => {
  const metrics = computeDirectExecutionMetrics({
    generatedTokenIds: [7, 8, 9], fullTokenIds: [2, 3, 7, 8, 9], elapsedSeconds: 0.25,
    sessionCacheHit: true, prefixTokensReused: 6, prefillTokensComputed: 2, tokensEqualBaseline: true,
    steps: [{ forwardSeconds: 0.12 }, { forwardSeconds: 0.04 }, { forwardSeconds: 0.03 }],
  }, 8, 2, 0.13);
  assert.deepEqual(metrics, {
    inputTokens: 8, generatedTokens: 3, firstTokenForwardSeconds: 0.12, firstTokenWallSeconds: 0.13,
    decodeForwardSeconds: 0.07, totalForwardSeconds: 0.19, sessionCacheHit: true,
    prefixTokensReused: 6, prefillTokensComputed: 2, prefillAvoidedRate: 0.75,
    baselineSpeedup: 8, tokensEqualBaseline: true,
  });
  assert.equal(computeDirectExecutionMetrics({ generatedTokenIds: [7], fullTokenIds: [2, 7] }, 1, undefined, 0.1, false).tokensEqualBaseline, null);
});

test("métricas top-K diretas quantificam logit escolhido, margem greedy e sobreposição", () => {
  const agreement = computeDirectLogitAgreement(
    [
      { baselineToken: 7, baselineTopLogits: [{ tokenId: 7, logit: 10 }, { tokenId: 8, logit: 9 }, { tokenId: 9, logit: 8 }] },
      { baselineToken: 8, baselineTopLogits: [{ tokenId: 8, logit: 5 }, { tokenId: 7, logit: 4 }] },
      { baselineToken: 1, baselineTopLogits: [{ tokenId: 1, logit: 7 }, { tokenId: 2, logit: 6 }] },
    ],
    [
      { tokenId: 7, topLogits: [{ tokenId: 7, value: 9.75 }, { tokenId: 8, value: 8.75 }, { tokenId: 9, value: 8.25 }] },
      { tokenId: 9, topLogits: [{ tokenId: 9, value: 5.1 }, { tokenId: 8, value: 4.8 }] },
      { tokenId: 2, topLogits: [{ tokenId: 2, value: 8 }, { tokenId: 1, value: 7.5 }] },
    ],
  );
  assert.equal(agreement.reportedSteps, 3); assert.equal(agreement.measuredSteps, 2); assert.equal(agreement.rootDivergences, 1); assert.equal(agreement.postDivergenceSteps, 1); assert.equal(agreement.selectedLogitMeasuredSteps, 2); assert.ok(Math.abs(agreement.meanBaselineArgmaxLogitAbsError! - 0.225) < 1e-12); assert.equal(agreement.maxBaselineArgmaxLogitAbsError, 0.25);
  assert.equal(agreement.marginMeasuredSteps, 2); assert.ok(Math.abs(agreement.meanGreedyMarginAbsError! - 0.35) < 1e-12); assert.ok(Math.abs(agreement.maxGreedyMarginAbsError! - 0.7) < 1e-12);
  assert.equal(agreement.meanTopKOverlapRate, 0.75); assert.equal(agreement.maxTopKCommonLogitAbsError, 0.25); assert.deepEqual(agreement.steps.map((step) => [step.contextsEqualBeforeStep, step.topKOverlapCount, step.topK]), [[true, 3, 3], [true, 1, 2], [false, 0, 2]]);
});

test("servidor integra geração direta persistente e decodifica seus tokens", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-direct-worker-test-"));
  const transformer = join(directory, "transformer.mjs"), direct = join(directory, "direct.mjs");
  await writeFile(transformer, `import readline from "node:readline";
console.log(JSON.stringify({ready:true,initializationSeconds:0.1}));
readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);const report=r.mode==='decode'?{text:r.tokenIds.join('|')}:r.mode==='encode'?{tokenIds:[2]}:{inputIds:[[2]],baselineGeneratedTokenIds:[7]};console.log(JSON.stringify({id:r.id,report}));});\n`);
  await writeFile(direct, `import readline from "node:readline";
console.log(JSON.stringify({ready:true,initializationSeconds:0.2}));
let requests=0; readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);requests++;console.log(JSON.stringify({id:r.id,report:{generatedTokenIds:[7],fullTokenIds:[2,7],elapsedSeconds:1,tokensPerSecond:1,linearThreads:4,requests}}));});\n`);
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper: transformer, tokenizerHelper: transformer, literalArtifact: "literal.json", binaryPool: directory, directWorker: direct, directLinearHelper: "unused.py", directThreads: 4 });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP de teste ausente.");
    const response = await fetch(`http://127.0.0.1:${address.port}/api/compare`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "x", maxNewTokens: 1 }) });
    assert.equal(response.status, 200);
    const body = await response.json() as { direct: { generatedText: string; fullText: string; tokensEqualBaseline: boolean; firstDivergentStep: number | null } };
    assert.deepEqual(body.direct, { generatedTokenIds: [7], fullTokenIds: [2, 7], elapsedSeconds: 1, tokensPerSecond: 1, linearThreads: 4, requests: 3, generatedText: "7", fullText: "2|7", tokensEqualBaseline: true, firstDivergentStep: null });
    const status = await fetch(`http://127.0.0.1:${address.port}/api/status`).then((entry) => entry.json()) as { ready: boolean; chatTemplate: string; checkpointChatTemplateDeclared: boolean; direct: { warmupComplete: boolean; warmupSeconds: number } };
    assert.equal(status.ready, true); assert.equal(status.chatTemplate, "google-gemma4-it-text-turn-v1"); assert.equal(status.checkpointChatTemplateDeclared, false); assert.equal(status.direct.warmupComplete, true); assert.ok(status.direct.warmupSeconds >= 0);
  } finally { await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept())); await rm(directory, { recursive: true, force: true }); }
});

test("streaming entrega tokens antes do relatório e reutiliza prefixo comprovado da sessão", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-stream-session-test-"));
  const transformer = join(directory, "transformer.mjs"), direct = join(directory, "direct.mjs");
  await writeFile(transformer, `import readline from "node:readline";
console.log(JSON.stringify({ready:true}));
readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);let report;if(r.mode==='encode'){const first=r.text==='<|turn>user\\nprimeiro<turn|>\\n<|turn>model\\n',next=r.text==='\\n<|turn>user\\ncontinuação<turn|>\\n<|turn>model\\n';report={tokenIds:first?[2,105]:next?[105]:[999]};}else if(r.mode==='decode')report={text:r.tokenIds.join('|')};else{const ids=String(r.inputIds).split(',').map(Number),token=ids.includes(106)?8:106;report={inputIds:[ids],baselineGeneratedTokenIds:[token],generatedTokensEqual:true,steps:[],performance:{baselineTokensPerSecond:1,candidateTokensPerSecond:1,candidateSpeedup:1,processPeakRssBytes:0},executionThreads:1,candidatePrecision:'f32',roundingPolicy:'none',receivedEosTokenId:r.eosTokenId};}console.log(JSON.stringify({id:r.id,report}));});\n`);
  await writeFile(direct, `import readline from "node:readline";
console.log(JSON.stringify({ready:true}));
let generations=0;readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line),token=r.inputIds.includes(106)?8:106,hit=r.inputIds.includes(106),reused=hit?r.inputIds.indexOf(106)+1:0,emit=()=>console.log(JSON.stringify({id:r.id,report:{generatedTokenIds:[token],fullTokenIds:[...r.inputIds,token],elapsedSeconds:0.01,tokensPerSecond:100,linearThreads:4,sessionCacheHit:hit,prefixTokensReused:reused,prefillTokensComputed:r.inputIds.length-reused,cachedContextTokens:r.inputIds.length,receivedInputIds:r.inputIds,receivedEosTokenId:r.eosTokenId,workerGenerations:++generations}}));if(r.stream){console.log(JSON.stringify({id:r.id,event:{type:'token',step:0,tokenId:token,forwardSeconds:0.01,topLogits:[]}}));setTimeout(emit,40);}else emit();});\n`);
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper: transformer, tokenizerHelper: transformer, literalArtifact: "literal.json", binaryPool: directory, directWorker: direct, directLinearHelper: "unused.py" });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP ausente.");
    const endpoint = `http://127.0.0.1:${address.port}/api/compare-stream`;
    let initialStatus: { ready?: boolean; reference?: { state?: string }; tokenizer?: { ready?: boolean } } = {};
    for (let index = 0; index < 100 && !initialStatus.ready; index += 1) { initialStatus = await fetch(`http://127.0.0.1:${address.port}/api/status`).then((response) => response.json()) as typeof initialStatus; await new Promise((accept) => setTimeout(accept, 5)); }
    assert.equal(initialStatus.tokenizer?.ready, true); assert.equal(initialStatus.reference?.state, "unloaded");
    const compiledStream = async (prompt: string, continueSession: boolean) => {
      const response = await fetch(`http://127.0.0.1:${address.port}/api/generate-stream`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt, maxNewTokens: 1, sessionId: 91, continueSession, conversationMode: "chat" }) });
      assert.equal(response.status, 200); return (await response.text()).trim().split("\n").map((line) => JSON.parse(line) as { type: string; generatedTokenIds?: number[]; generatedText?: string; deltaText?: string; textReset?: boolean; data?: { executionMode?: string; referenceExecuted?: boolean; direct?: { receivedInputIds: number[]; sessionCacheHit: boolean; prefixTokensReused: number }; directExecutionMetrics?: { tokensEqualBaseline: boolean | null; baselineSpeedup: number | null }; comparisonTiming?: { schedule: string } } });
    };
    const compiledFirst = await compiledStream("primeiro", false); assert.deepEqual(compiledFirst.map((entry) => entry.type), ["direct-token", "direct-complete", "result"]); assert.deepEqual(compiledFirst[0]!.generatedTokenIds, [106]); assert.equal(compiledFirst[0]!.generatedText, "106"); assert.equal(compiledFirst[0]!.deltaText, "106"); assert.equal(compiledFirst[0]!.textReset, false); assert.equal(compiledFirst.at(-1)!.data!.executionMode, "compiled-only"); assert.equal(compiledFirst.at(-1)!.data!.referenceExecuted, false); assert.equal(compiledFirst.at(-1)!.data!.directExecutionMetrics!.tokensEqualBaseline, null); assert.equal(compiledFirst.at(-1)!.data!.directExecutionMetrics!.baselineSpeedup, null); assert.equal(compiledFirst.at(-1)!.data!.comparisonTiming!.schedule, "compiled-only");
    const compiledSecond = await compiledStream("continuação", true); assert.deepEqual(compiledSecond.at(-1)!.data!.direct!.receivedInputIds, [2, 105, 106, 105]); assert.equal(compiledSecond.at(-1)!.data!.direct!.sessionCacheHit, true); assert.equal(compiledSecond.at(-1)!.data!.direct!.prefixTokensReused, 3);
    const compiledStatus = await fetch(`http://127.0.0.1:${address.port}/api/status`).then((response) => response.json()) as { reference: { state: string } }; assert.equal(compiledStatus.reference.state, "unloaded");
    const stream = async (prompt: string, continueSession: boolean) => {
      const response = await fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt, maxNewTokens: 1, sessionId: 41, continueSession, conversationMode: "chat" }) });
      assert.equal(response.status, 200); const lines = (await response.text()).trim().split("\n").map((line) => JSON.parse(line) as { type: string; serverElapsedSeconds?: number; event?: { tokenId: number }; generatedTokenIds?: number[]; generatedText?: string; deltaText?: string; textReset?: boolean; data?: { chatTemplate?: string; generatedText?: string; direct?: { receivedInputIds: number[]; receivedEosTokenId: number; sessionCacheHit: boolean; prefixTokensReused: number; workerGenerations: number }; directExecutionMetrics?: { inputTokens: number; firstTokenWallSeconds: number | null; prefixTokensReused: number; prefillTokensComputed: number; prefillAvoidedRate: number; tokensEqualBaseline: boolean }; comparisonTiming?: { schedule: string; directBackgroundRecoveryWaitSeconds: number; awaitedBackgroundRecoverySeconds?: number; backgroundRecoveryScheduled: boolean; directPhaseSeconds: number; referencePhaseSeconds: number; referenceReleased: boolean; directRecoverySeconds?: number; totalWallSeconds: number } } }); return lines;
    };
    const first = await stream("primeiro", false); assert.deepEqual(first.map((entry) => entry.type), ["direct-token", "direct-complete", "reference-loading", "direct-rewarming", "result"]); assert.equal(first[0]!.event!.tokenId, 106); assert.deepEqual(first[0]!.generatedTokenIds, [106]); assert.equal(first[0]!.generatedText, "106"); assert.equal(first[0]!.deltaText, "106"); assert.equal(first[0]!.textReset, false); assert.ok(first[0]!.serverElapsedSeconds! >= 0); assert.equal(first[1]!.data!.generatedText, "106"); assert.deepEqual(first[4]!.data!.direct!.receivedInputIds, [2, 105]); assert.equal(first[4]!.data!.direct!.receivedEosTokenId, 106); assert.equal(first[4]!.data!.chatTemplate, "google-gemma4-it-text-turn-v1"); assert.deepEqual(first[4]!.data!.directExecutionMetrics, { inputTokens: 2, generatedTokens: 1, firstTokenForwardSeconds: null, firstTokenWallSeconds: first[4]!.data!.directExecutionMetrics!.firstTokenWallSeconds, decodeForwardSeconds: 0, totalForwardSeconds: 0, sessionCacheHit: false, prefixTokensReused: 0, prefillTokensComputed: 2, prefillAvoidedRate: 0, baselineSpeedup: null, tokensEqualBaseline: true }); assert.ok(first[4]!.data!.directExecutionMetrics!.firstTokenWallSeconds! >= 0); assert.equal(first[4]!.data!.comparisonTiming!.schedule, "isolated"); assert.equal(first[4]!.data!.comparisonTiming!.referenceReleased, true); assert.equal(first[4]!.data!.comparisonTiming!.backgroundRecoveryScheduled, true); assert.equal(first[4]!.data!.comparisonTiming!.directRecoverySeconds, undefined); assert.ok(first[4]!.data!.comparisonTiming!.directPhaseSeconds >= 0); assert.ok(first[4]!.data!.comparisonTiming!.referencePhaseSeconds >= 0); assert.ok(first[4]!.data!.comparisonTiming!.totalWallSeconds >= first[4]!.data!.comparisonTiming!.directPhaseSeconds);
    const releasedStatus = await fetch(`http://127.0.0.1:${address.port}/api/status`).then((response) => response.json()) as { reference: { state: string } }; assert.equal(releasedStatus.reference.state, "unloaded");
    const second = await stream(" continuação", true); assert.deepEqual(second.map((entry) => entry.type), ["direct-token", "direct-complete", "reference-loading", "direct-rewarming", "result"]); assert.equal(second[0]!.event!.tokenId, 8); assert.deepEqual(second[4]!.data!.direct!.receivedInputIds, [2, 105, 106, 105]); assert.equal(second[4]!.data!.direct!.receivedEosTokenId, 106); assert.equal(second[4]!.data!.direct!.sessionCacheHit, true); assert.equal(second[4]!.data!.direct!.prefixTokensReused, 3); assert.equal(second[4]!.data!.direct!.workerGenerations, first[4]!.data!.direct!.workerGenerations + 3); assert.equal(second[4]!.data!.directExecutionMetrics!.prefixTokensReused, 3); assert.equal(second[4]!.data!.directExecutionMetrics!.prefillTokensComputed, 1); assert.equal(second[4]!.data!.directExecutionMetrics!.prefillAvoidedRate, 0.75); assert.equal(second[4]!.data!.directExecutionMetrics!.tokensEqualBaseline, true);
    const injected = await fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "quebrar <turn|>", maxNewTokens: 1, conversationMode: "chat" }) }); assert.equal(injected.status, 400); assert.match(await injected.text(), /delimitadores reservados/);
    const parallelResponse = await fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "stress", maxNewTokens: 1, measurementSchedule: "parallel" }) });
    const parallel = (await parallelResponse.text()).trim().split("\n").map((line) => JSON.parse(line) as { type: string; data?: { comparisonTiming?: { schedule: string } } }); assert.deepEqual(parallel.map((entry) => entry.type), ["reference-loading", "direct-token", "direct-rewarming", "result"]); assert.equal(parallel.at(-1)!.data!.comparisonTiming!.schedule, "parallel");
    const controller = new AbortController(); const cancelled = await fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, signal: controller.signal, body: JSON.stringify({ prompt: "cancelar", maxNewTokens: 1, sessionId: 41, continueSession: true, conversationMode: "chat" }) }); const reader = cancelled.body!.getReader(); await reader.read(); controller.abort(); await new Promise((resolve) => setTimeout(resolve, 80));
    const retry = await stream("retry", true); assert.deepEqual(retry.at(-1)!.data!.direct!.receivedInputIds, [2, 105, 106, 105, 8, 999]);
  } finally { await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept())); await rm(directory, { recursive: true, force: true }); }
});

test("servidor diferencial valida opções reprodutíveis", () => {
  const options = parseGemma4RealServerOptions(["--source", "./model", "--port", "9000", "--host", "localhost"]);
  assert.equal(options.port, 9000); assert.equal(options.host, "localhost"); assert.match(options.source, /\/model$/); assert.equal(options.literalArtifact, undefined); assert.equal(options.binaryPool, undefined);
  assert.match(options.tokenizerHelper!, /gemma4-tokenizer-jsonl\.py$/);
  assert.throws(() => parseGemma4RealServerOptions(["--port", "0"]), /--port inválido/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "metal"]), /pytorch ou mlx/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-mlp", "always"]), /off, bf16, real ou native-bf16/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-ffn", "real"]), /off ou native-bf16/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-decoder-layer", "real"]), /off ou native-bf16/);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-fused-decoder-stack", "real"]).directFusedDecoderStack, "real");
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-ple", "always"]), /off, bf16 ou real/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-ple-prelude", "always"]), /off, bf16 ou real/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-token-forward", "always"]), /off ou bf16/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-resident-generation", "always"]), /off ou on/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-final-head", "fp16"]), /f32, native-bf16, native-bf16-stream ou native-bf16-whole/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-mlx-head-quantization", "q2"]), /off, q8, q8-shortlist ou q4/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-native-attention", "auto"]), /off, bf16 ou real/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-attention", "always"]), /off, bf16, real ou native-bf16/);
  const directDefaults = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool"]);
  assert.equal(directDefaults.directThreads, 8); assert.equal(directDefaults.directMaxReadMiB, 16); assert.equal(directDefaults.directFinalHeadReadMiB, 16); assert.equal(directDefaults.directVerificationMargin, 0);
  assert.equal(directDefaults.directVerificationPrefixAhead, true);
  assert.equal(directDefaults.directVerificationUncachedPrefillAhead, false);
  assert.equal(directDefaults.directVerificationFinalHead, "native-bf16-whole");
  assert.equal(directDefaults.directMlxHeadQuantization, "off");
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-mlx-head-quantization", "q8-shortlist"]).directMlxHeadQuantization, "q8-shortlist");
  assert.equal(directDefaults.directMlxDecoderQuantization, "q8-ffn-gate-up");
  assert.equal(directDefaults.directMlxDecoderQuantizationLayers, "0-19");
  assert.equal(directDefaults.directMlxDecoderQuantizationGroupSize, 64);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-mlx-decoder-quantization-group-size", "128"]).directMlxDecoderQuantizationGroupSize, 128);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-mlx-decoder-quantization-group-size", "96"]), /32, 64 ou 128/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "pytorch", "--direct-mlx-decoder-quantization-group-size", "128"]), /requer backend MLX/);
  const unquantized = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-mlx-decoder-quantization", "off"]); assert.equal(unquantized.directMlxDecoderQuantization, "off"); assert.equal(unquantized.directMlxDecoderQuantizationLayers, undefined); assert.equal(unquantized.directVerificationMargin, 0.25);
  const customQ8Layers = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-mlx-decoder-quantization-layers", "3,1-2,2"]); assert.equal(customQ8Layers.directMlxDecoderQuantizationLayers, "1-3"); assert.equal(customQ8Layers.directVerificationMargin, 0.25);
  const q4Layers = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-mlx-decoder-quantization", "q4-ffn-gate-up", "--direct-mlx-decoder-quantization-layers", "0-9"]);
  assert.equal(q4Layers.directMlxDecoderQuantization, "q4-ffn-gate-up"); assert.equal(q4Layers.directMlxDecoderQuantizationLayers, "0-9");
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-mlx-decoder-quantization", "q8-ffn-gate-up-attention"]).directMlxDecoderQuantization, "q8-ffn-gate-up-attention");
  const q8GateUpDownLayers = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-mlx-decoder-quantization", "q8-ffn-gate-up-down", "--direct-mlx-decoder-quantization-layers", "10-20"]);
  assert.equal(q8GateUpDownLayers.directMlxDecoderQuantization, "q8-ffn-gate-up-down"); assert.equal(q8GateUpDownLayers.directMlxDecoderQuantizationLayers, "10-20"); assert.equal(q8GateUpDownLayers.directVerificationMargin, 0.125); assert.equal(q8GateUpDownLayers.directVerificationBackend, "mlx-shared-control");
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-mlx-decoder-quantization-layers", "3,1-2,2"]).directMlxDecoderQuantizationLayers, "1-3");
  assert.throws(() => parseGemma4RealServerOptions(["--direct-mlx-decoder-quantization-layers", "42"]), /entre 0 e 41/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-mlx-decoder-quantization", "off", "--direct-mlx-decoder-quantization-layers", "0"]), /requer q8-ffn-gate-up, q8-ffn-gate-up-down ou q4-ffn-gate-up/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-mlx-decoder-quantization", "q2"]), /modo inválido/);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-verification-margin", "off"]).directVerificationMargin, undefined);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-verification-backend", "mlx-control"]).directVerificationBackend, "mlx-control");
  assert.throws(() => parseGemma4RealServerOptions(["--direct-verification-backend", "cuda"]), /pytorch, mlx-control ou mlx-shared-control/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "pytorch", "--direct-verification-backend", "mlx-control"]), /requer backend mlx/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-verification-margin", "-1"]), /finito não negativo/);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-verification-prefix-ahead", "off"]).directVerificationPrefixAhead, false);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-verification-prefix-ahead", "auto"]), /on ou off/);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-verification-uncached-prefill-ahead", "on"]).directVerificationUncachedPrefillAhead, true);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-verification-uncached-prefill-ahead", "auto"]), /on ou off/);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-verification-final-head", "native-bf16-whole"]).directVerificationFinalHead, "native-bf16-whole");
  assert.throws(() => parseGemma4RealServerOptions(["--direct-verification-final-head", "f32"]), /native-bf16-stream ou native-bf16-whole/);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "pytorch", "--direct-fused-decoder-stack", "native-bf16-ple"]).directFusedDecoderStack, "native-bf16-ple");
  const splitTiles = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-max-read-mib", "16", "--direct-final-head-read-mib", "32"]);
  assert.equal(splitTiles.directFinalHeadReadMiB, 32);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-final-head-read-mib", "0"]), /Configuração direta inválida/);
  assert.equal(directDefaults.directLinearBackend, "mlx"); assert.equal(directDefaults.directFusedMlp, "real"); assert.equal(directDefaults.directFusedFfn, "off"); assert.equal(directDefaults.directFusedDecoderLayer, "off"); assert.equal(directDefaults.directFusedDecoderStack, "real"); assert.equal(directDefaults.directFusedPle, "off"); assert.equal(directDefaults.directFusedPlePrelude, "bf16"); assert.equal(directDefaults.directFusedTokenForward, "bf16"); assert.equal(directDefaults.directResidentGeneration, "on"); assert.equal(directDefaults.directFinalHead, "native-bf16-whole"); assert.equal(directDefaults.directNativeAttention, "off"); assert.equal(directDefaults.directFusedAttention, "off");
  const mlxDefaults = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "mlx"]);
  assert.equal(mlxDefaults.directFinalHeadReadMiB, 16); assert.equal(mlxDefaults.directFusedMlp, "real"); assert.equal(mlxDefaults.directFusedFfn, "off"); assert.equal(mlxDefaults.directFusedDecoderLayer, "off"); assert.equal(mlxDefaults.directFusedDecoderStack, "real"); assert.equal(mlxDefaults.directFusedPle, "off"); assert.equal(mlxDefaults.directFusedPlePrelude, "bf16"); assert.equal(mlxDefaults.directFusedTokenForward, "bf16"); assert.equal(mlxDefaults.directResidentGeneration, "on"); assert.equal(mlxDefaults.directFinalHead, "native-bf16-whole"); assert.equal(mlxDefaults.directNativeAttention, "off"); assert.equal(mlxDefaults.directFusedAttention, "off");
  const pytorchDefaults = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "pytorch"]);
  assert.equal(pytorchDefaults.directMlxDecoderQuantization, "off");
  assert.equal(pytorchDefaults.directFinalHeadReadMiB, 32); assert.equal(pytorchDefaults.directVerificationMargin, undefined); assert.equal(pytorchDefaults.directFusedMlp, "native-bf16"); assert.equal(pytorchDefaults.directFusedFfn, "native-bf16"); assert.equal(pytorchDefaults.directFusedDecoderLayer, "native-bf16"); assert.equal(pytorchDefaults.directFusedDecoderStack, "native-bf16"); assert.equal(pytorchDefaults.directFusedPle, "bf16"); assert.equal(pytorchDefaults.directFusedPlePrelude, "bf16"); assert.equal(pytorchDefaults.directFusedTokenForward, "off"); assert.equal(pytorchDefaults.directResidentGeneration, "off"); assert.equal(pytorchDefaults.directFinalHead, "native-bf16-stream"); assert.equal(pytorchDefaults.directNativeAttention, "real"); assert.equal(pytorchDefaults.directFusedAttention, "native-bf16");
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-fused-mlp", "native-bf16"]), /requer backend pytorch/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-fused-ffn", "native-bf16"]), /requer backend pytorch/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-fused-decoder-layer", "native-bf16"]), /requer backend pytorch/);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "mlx", "--direct-fused-decoder-stack", "native-bf16"]).directFusedDecoderStack, "native-bf16");
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "mlx", "--direct-fused-decoder-stack", "real"]).directFusedDecoderStack, "real");
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "pytorch", "--direct-fused-decoder-stack", "real"]), /requer backend mlx/);
  const stackOff = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "mlx", "--direct-fused-decoder-stack", "off"]);
  assert.equal(stackOff.directFusedTokenForward, "off"); assert.equal(stackOff.directResidentGeneration, "off");
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-fused-decoder-stack", "off", "--direct-fused-token-forward", "bf16"]), /decoder stack habilitada/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-fused-decoder-stack", "native-bf16-ple"]), /requer backend pytorch/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-fused-attention", "native-bf16"]), /requer backend pytorch/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "pytorch", "--direct-fused-token-forward", "bf16"]), /requer backend mlx/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-resident-generation", "on", "--direct-fused-token-forward", "off"]), /token forward MLX bf16/);
  assert.throws(() => parseGemma4RealServerOptions(["--unknown", "x"]), /Flag desconhecida/);
});

test("bundle compilado fornece modelo, constantes e tokenizer sem o diretório original", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-compiled-bundle-options-"));
  try {
    await Promise.all(["constants.literal.json", "model.safetensors", "tokenizer.json", "config.json", "global-formulas.ssa.json", "formula.graph.json"].map((file) => writeFile(join(directory, file), "fixture")));
    const finalFormulaBytes = Buffer.from('{"calc_final_0":"BF16_RNE(EVAL_EXACT_DAG(\\"root:output\\", x))"}\n'), finalFormulaSha256 = createHash("sha256").update(finalFormulaBytes).digest("hex"); await writeFile(join(directory, "final-formulas.json"), finalFormulaBytes);
    const functionBindings = [{ functionId: "op:0", operationId: "operation_0", ordinal: 0, output: "value_0", operation: "activation", kernel: "mlx-real-activation", root: "root:0", predecessorFunctions: [] }];
    const functionBindingsSha256 = createHash("sha256").update(JSON.stringify(functionBindings)).digest("hex"), realSimplifiedProgramSha256 = "e".repeat(64);
    const orderedDispatchSha256 = createHash("sha256").update(JSON.stringify([{ id: "operation_0", op: "activation", output: "value_0" }])).digest("hex");
    const compiledOutputProgram = { id: "gemma4-text-real-final-vectors" as const, semantics: "shared-dag-parametric-output-functions-v1" as const, logicalDispatchesPerForward: 1 as const, operationFunctions: 1, outputFunctions: 1, firstOperationId: "operation_0", terminalOperationId: "operation_0", orderedDispatchSha256 };
    const outputBinding = { name: "terminal_logit", operationId: "operation_0", fixedDimension: 0, coordinate: [], parameters: [], root: "root:output", finalQuantization: "BF16-round-to-nearest-ties-to-even" };
    const outputBindingsSha256 = createHash("sha256").update(JSON.stringify([outputBinding])).digest("hex");
    const standaloneSsaOutputsSha256 = createHash("sha256").update(JSON.stringify([{ assignment: "calc_terminal_logit_0", value: "root:output", parameters: [], coordinate: [], finalQuantization: "BF16-round-to-nearest-ties-to-even" }])).digest("hex");
    const planBytes = Buffer.from(JSON.stringify({ kind: "gemma4-vectorized-real-lowering-plan", schemaVersion: 3, contract: { kind: "gemma4-vectorized-real-lowering-contract", schemaVersion: 3, semantics: "gemma4-exact-real-simplified-v1", execution: { engine: "mlx-f32-real-decoder-stack-v1", mode: "compiled-parametric-output-program", intermediateBf16Boundaries: 0, finalQuantization: "BF16-round-to-nearest-ties-to-even", directlyLoadsStandaloneSsaFile: false, sourceProgramValidated: true, directlyExecutesGlobalFormula: true }, source: { artifactIntegritySha256: "f".repeat(64), realSimplifiedProgramSha256, sourceAssignments: 1, expressionNodes: 1, operationFunctions: 1, globalClosureFunctions: 1, standaloneRuntimeReductions: 0, outputFunctions: 1, outputBindingsSha256, standaloneSsaOutputsSha256, outputFamilies: { terminal_logit: { dimensions: 1, operationId: "operation_0", parameters: [], finalQuantization: "BF16-round-to-nearest-ties-to-even" } } }, coverage: { unresolvedRuntimeReductions: 0, intermediateIeeeRoundingNodes: 0, unsupportedOperations: 0, globalClosuresDependingOnStandaloneReductions: 0, kernels: { activation: 1, elementwise: 0, linear: 0, reshape_heads: 0, rms_norm: 0, rotary_embedding: 0, scaled_dot_product_attention: 0, select_per_layer: 0, tensor_scale: 0 } }, functionBindingsSha256, compiledOutputProgram }, functionBindings }));
    const planSha256 = createHash("sha256").update(planBytes).digest("hex"); await writeFile(join(directory, "vectorized-real-lowering.json"), planBytes);
    const finalFormulaRuntime = await writeGemma4FinalFormulaRuntime(join(directory, "final-formulas.runtime.json"), {
      kind: "gemma4-final-formula-runtime", schemaVersion: 1, evaluator: "EVAL_EXACT_DAG", semantics: "gemma4-exact-real-simplified-v1", publicFormula: "BF16_RNE(EVAL_EXACT_DAG(root,x))",
      input: { tensor: "x", length: 1, onlyFreeInput: true }, output: { family: "terminal_logit", functions: 1, finalQuantization: "BF16-round-to-nearest-ties-to-even" },
      artifacts: { formulaMap: { file: "final-formulas.json", sha256: finalFormulaSha256, orderedRootsSha256: "e".repeat(64) }, globalSsa: { file: "global-formulas.ssa.json", sha256: "a".repeat(64), standaloneOutputsSha256: standaloneSsaOutputsSha256 }, constantPool: { file: "constants.literal.json", sha256: "b".repeat(64) }, loweringPlan: { file: "vectorized-real-lowering.json", sha256: planSha256, functionBindingsSha256, outputBindingsSha256, realSimplifiedProgramSha256 } },
      evaluation: { rootResolution: "calc_final_n -> ordered terminal_logit output root -> dependency-ordered SSA node", functionCalls: "lexical parameter binding followed by evaluation of the referenced operation-function root", reductionOrder: "ascending integer index; arguments retain serialized order", intermediateIeeeRounding: "none", nodeSemantics: GEMMA4_FINAL_FORMULA_NODE_SEMANTICS },
      execution: { engine: "mlx-f32-real-decoder-stack-v1", mode: "compiled-parametric-output-program", parallelism: "metal-vectorized-output-dimensions", compiledOutputProgram },
    });
    await writeFile(join(directory, "constants.runtime-index.json"), "fixture-index");
    await writeFile(join(directory, "manifest.json"), JSON.stringify({ kind: "gemma4-compiled-shared-dag-bundle", schemaVersion: 4, execution: "compiled-parametric-output-program-runtime", runtimeLowering: { engine: "mlx-f32-real-decoder-stack-v1", directlyExecutesGlobalFormula: true, executesPersistedLoweringPlan: true, plan: "vectorized-real-lowering.json", functionBindingsSha256, outputBindingsSha256, standaloneSsaOutputsSha256, outputFunctions: 1, realSimplifiedProgramSha256, globalFormulaRole: "compiled-executable-shared-dag", compiledOutputProgram }, formula: { family: "terminal_logit", dimension: 0, root: `sha256:${"1".repeat(64)}`, expressionNodes: 1, inputTensor: "x", inputLength: 1, file: "formula.graph.json" }, globalProgram: { file: "global-formulas.ssa.json", terminalLogits: 1, constantPool: "constants.literal.json" }, finalFormulaMap: { file: "final-formulas.json", functions: 1, inputTensor: "x", evaluator: "BF16_RNE(EVAL_EXACT_DAG(root,x))", globalFormulaSha256: "a".repeat(64), orderedRootsSha256: "e".repeat(64) }, finalFormulaRuntime, runtimeIndex: { file: "constants.runtime-index.json", schemaVersion: 2, constantPoolSha256: "b".repeat(64), integrityRootSha256: "f".repeat(64) }, files: [{ role: "formula-graph", file: "formula.graph.json", sha256: "c".repeat(64) }, { role: "global-formulas", file: "global-formulas.ssa.json", sha256: "a".repeat(64) }, { role: "final-formulas", file: "final-formulas.json", sha256: finalFormulaSha256 }, { role: "final-formula-runtime", file: "final-formulas.runtime.json", sha256: finalFormulaRuntime.sha256 }, { role: "constant-pool", file: "constants.literal.json", sha256: "b".repeat(64) }, { role: "literal-runtime-index", file: "constants.runtime-index.json", sha256: "d".repeat(64) }, { role: "vectorized-real-lowering", file: "vectorized-real-lowering.json", sha256: planSha256 }] }));
    const bundled = parseGemma4RealServerOptions(["--compiled-bundle", directory]);
    assert.equal(bundled.source, directory); assert.equal(bundled.literalArtifact, join(directory, "constants.literal.json")); assert.equal(bundled.binaryPool, directory);
    assert.equal(bundled.directArtifactIndex, join(directory, "constants.runtime-index.json")); assert.equal(bundled.directArtifactIndexSha256, "d".repeat(64)); assert.equal(bundled.directArtifactSha256, "b".repeat(64));
    assert.equal(bundled.compiledProgram?.runtimeIndex?.schemaVersion, 2);
    assert.deepEqual(bundled.compiledProgram?.finalFormulaMap, { file: "final-formulas.json", sha256: finalFormulaSha256, functions: 1, inputTensor: "x", evaluator: "BF16_RNE(EVAL_EXACT_DAG(root,x))", orderedRootsSha256: "e".repeat(64) });
    assert.deepEqual(bundled.compiledProgram?.finalFormulaRuntime, finalFormulaRuntime);
    assert.deepEqual(bundled.compiledProgram?.exampleFormula, { key: "calc_final_0", family: "terminal_logit", dimension: 0, root: `sha256:${"1".repeat(64)}`, expressionNodes: 1, inputTensor: "x", inputLength: 1, file: "formula.graph.json", sha256: "c".repeat(64) });
    assert.equal(bundled.compiledProgram?.directRuntime.executesPersistedLoweringPlan, true); assert.equal(bundled.compiledProgram?.directRuntime.directlyExecutesGlobalFormula, true); assert.equal(bundled.compiledProgram?.directRuntime.compiledOutputProgram.logicalDispatchesPerForward, 1); assert.equal(bundled.compiledProgram?.directRuntime.plan.sha256, planSha256); assert.equal(bundled.compiledProgram?.globalFormula.sha256, "a".repeat(64));
    const split = parseGemma4RealServerOptions(["--source", "./authoritative", "--compiled-bundle", directory]);
    assert.match(split.source, /\/authoritative$/); assert.equal(split.literalArtifact, join(directory, "constants.literal.json")); assert.equal(split.binaryPool, directory);
    await writeFile(join(directory, "final-formulas.json"), "tampered");
    assert.throws(() => parseGemma4RealServerOptions(["--compiled-bundle", directory]), /Mapa final de fórmulas diverge/);
    await writeFile(join(directory, "final-formulas.json"), finalFormulaBytes);
    await writeFile(join(directory, "vectorized-real-lowering.json"), "tampered");
    assert.throws(() => parseGemma4RealServerOptions(["--compiled-bundle", directory]), /diverge do SHA-256/);
  } finally { await rm(directory, { recursive: true, force: true }); }
  assert.throws(() => parseGemma4RealServerOptions(["--compiled-bundle", join(tmpdir(), "gemma4-missing-bundle")]), /não contém constants/);
});

test("servidor reutiliza um worker carregado para múltiplos prompts", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-worker-test-"));
  const helper = join(directory, "worker.mjs");
  await writeFile(helper, `import readline from "node:readline";
let requests=0; console.log(JSON.stringify({ready:true}));
readline.createInterface({input:process.stdin}).on("line",line=>{const request=JSON.parse(line); const report=request.mode==='encode'?{tokenIds:[2]}:(requests++,{prompt:request.prompt,requests,threads:request.threads}); console.log(JSON.stringify({id:request.id,report}));});\n`);
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper, tokenizerHelper: helper });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP de teste ausente.");
    const endpoint = `http://127.0.0.1:${address.port}/api/compare`;
    const request = (prompt: string, threads: number) => fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt, maxNewTokens: 1, threads }) }).then((response) => response.json()) as Promise<{ prompt: string; requests: number; threads: number; conversationMode: string; chatTemplate: null; comparisonTiming: { schedule: string; totalWallSeconds: number } }>;
    const first = await request("primeiro", 2); assert.deepEqual({ ...first, comparisonTiming: undefined }, { prompt: "primeiro", requests: 1, threads: 2, conversationMode: "raw", chatTemplate: null, comparisonTiming: undefined }); assert.equal(first.comparisonTiming.schedule, "reference-only"); assert.ok(first.comparisonTiming.totalWallSeconds >= 0);
    const second = await request("segundo", 4); assert.deepEqual({ ...second, comparisonTiming: undefined }, { prompt: "segundo", requests: 2, threads: 4, conversationMode: "raw", chatTemplate: null, comparisonTiming: undefined }); assert.equal(second.comparisonTiming.schedule, "reference-only");
    const invalid = await fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "inválido", maxNewTokens: 1, precision: "f16" }) });
    assert.equal(invalid.status, 400);
    assert.deepEqual(await invalid.json(), { error: "precision deve ser f32 ou f64." });
    const invalidSession = await fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "continuação", maxNewTokens: 1, continueSession: true }) });
    assert.equal(invalidSession.status, 400); assert.deepEqual(await invalidSession.json(), { error: "continueSession requer sessionId." });
    const invalidSchedule = await fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "x", maxNewTokens: 1, measurementSchedule: "mixed" }) });
    assert.equal(invalidSchedule.status, 400); assert.deepEqual(await invalidSchedule.json(), { error: "measurementSchedule deve ser isolated ou parallel." });
  } finally {
    await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept()));
    await rm(directory, { recursive: true, force: true });
  }
});

test("endpoint final liga qualquer dimensão de logit ao token decodificado", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-final-formula-server-")), helper = join(directory, "worker.mjs"), globalFile = join(directory, "global-formulas.ssa.json");
  const root = (digit: string) => `sha256:${digit.repeat(64)}`;
  const node = <T extends Record<string, unknown>>(payload: T) => ({ id: `sha256:${createHash("sha256").update(JSON.stringify(payload)).digest("hex")}`, ...payload });
  const outputFeature = node({ kind: "integer-constant", value: 1 });
  const scale = node({ kind: "rational", value: { numerator: "30", denominator: "1" } });
  const body = node({ kind: "multiply", arguments: [outputFeature.id, scale.id] });
  const call = node({ kind: "function-call", functionId: "operation:final_logit_softcap", arguments: [outputFeature.id] });
  const output = (family: string, dimension: number, digit: string) => ({ assignment: `calc_${family}_${dimension}`, value: root(digit), parameters: ["batch", "sequence"], coordinate: [root("a"), root("b"), root(digit)], finalQuantization: "BF16-round-to-nearest-ties-to-even" });
  await writeFile(globalFile, JSON.stringify({
    statements: [
      { target: outputFeature.id, expression: "1", node: outputFeature }, { target: scale.id, expression: "30", node: scale },
      { target: body.id, expression: `(${outputFeature.id} * ${scale.id})`, node: body },
      { target: call.id, expression: `operation:final_logit_softcap(${outputFeature.id})`, node: call },
    ],
    functions: [{ functionId: "operation:final_logit_softcap", operationId: "final_logit_softcap", ordinal: 1, output: "softcapped_logits", parameters: [{ name: "output_feature", node: outputFeature.id }], root: body.id, predecessorFunctions: ["operation:lm_head"], closureKind: "global-output-closure", operandBoundaries: [] }],
    outputs: [output("final_hidden_dimension", 0, "1"), output("terminal_logit", 0, "2"), { ...output("terminal_logit", 1, "3"), value: call.id }],
  }));
  await writeFile(helper, `import readline from "node:readline"; console.log(JSON.stringify({ready:true})); readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);const report=r.mode==='decode'?{text:'token:'+r.tokenIds[0]}:r.mode==='encode'?{tokenIds:[2]}:{inputIds:[[2]],baselineGeneratedTokenIds:[1]};console.log(JSON.stringify({id:r.id,report}));});\n`);
  const compiledProgram: Gemma4CompiledProgramStatus = {
    bundle: directory, execution: "compiled-parametric-output-program-runtime", formulaSemantics: "gemma4-exact-real-simplified-v1",
    globalFormula: { file: "global-formulas.ssa.json", sha256: "4".repeat(64), terminalLogits: 2, role: "compiled-executable-shared-dag" }, constantPool: { file: "constants.literal.json", sha256: "5".repeat(64) },
    finalFormulaMap: { file: "final-formulas.json", sha256: "d".repeat(64), functions: 2, inputTensor: "x", evaluator: "BF16_RNE(EVAL_EXACT_DAG(root,x))", orderedRootsSha256: "e".repeat(64) },
    finalFormulaRuntime: { file: "final-formulas.runtime.json", sha256: "f".repeat(64), schemaVersion: 1, evaluator: "EVAL_EXACT_DAG" },
    exampleFormula: { key: "calc_final_0", family: "terminal_logit", dimension: 0, root: root("2"), expressionNodes: 1, inputTensor: "x", inputLength: 1, file: "formula.graph.json", sha256: "6".repeat(64) },
    terminalLogitOutputs: { offset: 1, dimensions: 2, operationId: "final", finalQuantization: "BF16-round-to-nearest-ties-to-even" },
    directRuntime: { engine: "mlx-f32-real-decoder-stack-v1", directlyExecutesGlobalFormula: true, executesPersistedLoweringPlan: true, compiledOutputProgram: { id: "gemma4-text-real-final-vectors", semantics: "shared-dag-parametric-output-functions-v1", logicalDispatchesPerForward: 1, operationFunctions: 1, outputFunctions: 3, firstOperationId: "first", terminalOperationId: "final", orderedDispatchSha256: "7".repeat(64) }, plan: { file: "vectorized-real-lowering.json", sha256: "8".repeat(64), functionBindingsSha256: "9".repeat(64), outputBindingsSha256: "a".repeat(64), standaloneSsaOutputsSha256: "b".repeat(64), outputFunctions: 3, realSimplifiedProgramSha256: "c".repeat(64) } },
  };
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper, tokenizerHelper: helper, compiledProgram });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP de teste ausente.");
    const response = await fetch(`http://127.0.0.1:${address.port}/api/final-formula?dimension=1`); assert.equal(response.status, 200);
    const formula = await response.json() as { key: string; root: string; token: { id: number; text: string }; onlyFreeInput: string; formulaMap: { file: string; key: string; sha256: string }; formulaRuntime: { file: string; sha256: string; evaluator: string }; resolvedMath: { expression: string; node: { kind: string }; operation: { operationId: string; bodyExpression: string; bodyNode: { kind: string }; bodyFragment: { expandedExpression: string; truncated: boolean } } } };
    assert.equal(formula.key, "calc_final_1"); assert.equal(formula.root, call.id); assert.deepEqual(formula.token, { id: 1, text: "token:1" }); assert.equal(formula.onlyFreeInput, "x"); assert.deepEqual(formula.formulaMap, { file: "final-formulas.json", key: "calc_final_1", sha256: "d".repeat(64) });
    assert.deepEqual(formula.formulaRuntime, { file: "final-formulas.runtime.json", sha256: "f".repeat(64), schemaVersion: 1, evaluator: "EVAL_EXACT_DAG" });
    assert.equal(formula.resolvedMath.expression, `operation:final_logit_softcap(${outputFeature.id})`); assert.equal(formula.resolvedMath.node.kind, "function-call");
    assert.equal(formula.resolvedMath.operation.operationId, "final_logit_softcap"); assert.equal(formula.resolvedMath.operation.bodyExpression, `(${outputFeature.id} * ${scale.id})`); assert.equal(formula.resolvedMath.operation.bodyNode.kind, "multiply");
    assert.equal(formula.resolvedMath.operation.bodyFragment.expandedExpression, "((1) * (30))"); assert.equal(formula.resolvedMath.operation.bodyFragment.truncated, false);
    const invalid = await fetch(`http://127.0.0.1:${address.port}/api/final-formula?dimension=2`); assert.equal(invalid.status, 400); assert.deepEqual(await invalid.json(), { error: "dimension deve estar entre 0 e 1." });
  } finally { await new Promise<void>((accept) => server.close(() => accept())); await rm(directory, { recursive: true, force: true }); }
});

test("verificação seletiva é fail-closed e usa somente a margem do caminho rápido", () => {
  const report = (topLogits: unknown, generatedTokenIds = [7]) => ({ generatedTokenIds, fullTokenIds: [2, ...generatedTokenIds], steps: generatedTokenIds.map((tokenId) => ({ tokenId, topLogits })) });
  assert.deepEqual(assessDirectVerification(report([{ tokenId: 7, value: 2 }, { tokenId: 8, value: 2 }]), 0), { trigger: true, reason: "margin-at-or-below-threshold", minimumMargin: 0, marginThreshold: 0, sensitiveSteps: [0] });
  assert.deepEqual(assessDirectVerification(report([{ tokenId: 7, value: 2.125 }, { tokenId: 8, value: 2 }]), 0), { trigger: false, reason: "margin-at-or-below-threshold", minimumMargin: 0.125, marginThreshold: 0, sensitiveSteps: [] });
  assert.deepEqual(assessDirectVerification(report([], [7]), 0), { trigger: true, reason: "margin-unavailable", minimumMargin: null, marginThreshold: 0, sensitiveSteps: [] });
  assert.throws(() => createGemma4RealComparisonServer({ source: ".", python: "python", helper: "helper", directLinearBackend: "pytorch", directVerificationMargin: 0 }), /requer backend MLX/);
});

test("prefill exato antecipado exige ao menos um token Metal futuro", () => {
  assert.equal(shouldStartDirectVerificationPrefill({ step: 6, tokenId: 7 }, 8), true);
  assert.equal(shouldStartDirectVerificationPrefill({ step: 7, tokenId: 7 }, 8), false);
  assert.equal(shouldStartDirectVerificationPrefill({ step: 2, tokenId: 106 }, 8, 106), false);
  assert.throws(() => shouldStartDirectVerificationPrefill({ step: -1 }, 8), /inválido/);
});

test("prefixo exato antecipado reconhece contexto sensível idêntico em outra sessão", () => {
  const sessions = new Map<number, readonly number[]>([[1, [2, 10]], [2, [2, 20, 30, 40]]]);
  assert.equal(hasVerificationCachedPrefix(sessions, 1, [2, 10, 11]), true);
  assert.equal(hasVerificationCachedPrefix(sessions, 3, [2, 20], [30, 40]), true);
  assert.equal(hasVerificationCachedPrefix(sessions, 3, [2, 20], [30, 41]), false);
});

test("servidor substitui empate Metal pelo resultado do verificador compilado", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-selective-verification-test-"));
  const transformer = join(directory, "transformer.mjs"), direct = join(directory, "direct.mjs");
  await writeFile(transformer, `import readline from "node:readline";
console.log(JSON.stringify({ready:true}));
readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);const report=r.mode==='encode'?{tokenIds:[2,99]}:r.mode==='decode'?{text:r.tokenIds.join('|')}:{baselineGeneratedTokenIds:[7],candidateGeneratedTokenIds:[7],baselineGeneratedText:'7',candidateGeneratedText:'7',generatedTokensEqual:true,firstDivergentStep:null,inputIds:[[2,99]],steps:[{step:0,baselineToken:7,candidateToken:7,baselineTopLogits:[{tokenId:7,logit:2.125},{tokenId:8,logit:2}],candidateTopLogits:[],metrics:{argmaxEqual:true,divergenceRate:0,maxAbsError:0},baselineSeconds:1,candidateSeconds:1,...(r.captureLayerHidden?{layerHiddenEncoding:'terminal-token-f32le-base64',baselineLayerHidden:['AAAAAA=='],candidateLayerHidden:['AAAAAA==']}: {})}],performance:{baselineSeconds:1,candidateSeconds:1,baselineTokensPerSecond:1,candidateTokensPerSecond:1,candidateSpeedup:1,processPeakRssBytes:0},executionThreads:1,candidatePrecision:'f32',roundingPolicy:'none'};console.log(JSON.stringify({id:r.id,report}));});\n`);
  await writeFile(direct, `import readline from "node:readline";
const args=process.argv,backend=args[args.indexOf('--linear-backend')+1],decoderStack=args[args.indexOf('--fused-decoder-stack')+1];console.log(JSON.stringify({ready:true,linearBackend:backend}));
let generations=0;readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);if(r.control){console.log(JSON.stringify({id:r.id,report:{trimmed:true}}));return;}generations++;if(r.verificationPrefill){console.log(JSON.stringify({id:r.id,report:{verificationPrefill:true,elapsedSeconds:0.005,verificationSessionCacheHit:false,verificationPrefixTokensReused:0,verificationPrefillTokensComputed:r.inputIds.length}}));return;}const token=backend==='pytorch'?7:8,top=backend==='pytorch'?[{tokenId:7,value:2.125},{tokenId:8,value:2}]:[{tokenId:8,value:2},{tokenId:7,value:2}],report={generatedTokenIds:[token],fullTokenIds:[...r.inputIds,token],elapsedSeconds:0.01,tokensPerSecond:100,linearThreads:4,linearBackend:backend,decoderStack,workerPid:process.pid,workerGenerations:generations,terminalLogitsSha256:'${"0".repeat(64)}',steps:[{step:0,tokenId:token,forwardSeconds:0.01,topLogits:top}],...(r.captureLayerHidden?{layerHiddenEncoding:'terminal-token-f32le-base64',layerHiddenCaptures:[['AAAAAA==']]}:{}),...(r.verificationFastPath?{selectiveVerification:true,sensitiveSteps:r.verificationFastPath.sensitiveSteps,trustedFastPathSteps:0,verificationHeadSteps:1,verificationDivergenceStep:0}:{})};if(r.stream)console.log(JSON.stringify({id:r.id,event:{type:'token',step:0,tokenId:token,forwardSeconds:0.01,topLogits:top}}));console.log(JSON.stringify({id:r.id,report}));});\n`);
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper: transformer, tokenizerHelper: transformer, literalArtifact: "literal.json", binaryPool: directory, directWorker: direct, directLinearHelper: "unused.py", directVerificationMargin: 0, directVerificationUncachedPrefillAhead: true });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP ausente.");
    const endpoint = `http://127.0.0.1:${address.port}`;
    let status: { ready?: boolean; direct?: { backgroundRecovery?: { state?: string; lastSeconds?: number }; verification?: { ready?: boolean; state?: string; marginThreshold?: number; lifecycle?: string; warmupComplete?: boolean; warmupSeconds?: number } } } = {};
    for (let index = 0; index < 50 && !status.ready; index += 1) { status = await fetch(`${endpoint}/api/status`).then((response) => response.json()) as typeof status; await new Promise((accept) => setTimeout(accept, 10)); }
    assert.equal(status.direct?.verification?.ready, true); assert.equal(status.direct?.verification?.state, "ready"); assert.equal(status.direct?.verification?.marginThreshold, 0); assert.equal(status.direct?.verification?.warmupComplete, true); assert.ok((status.direct?.verification?.warmupSeconds ?? -1) >= 0);
    const response = await fetch(`${endpoint}/api/compare-stream`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "empate", maxNewTokens: 2, sessionId: 101 }) });
    const messages = (await response.text()).trim().split("\n").map((line) => JSON.parse(line) as { type: string; provisional?: boolean; generatedText?: string; generatedTokenIds?: number[]; data?: { generatedText?: string; direct: Record<string, unknown> } });
    assert.deepEqual(messages.map((message) => message.type), ["direct-token", "direct-verification-prefill", "direct-fallback", "direct-complete", "reference-loading", "direct-rewarming", "result"]); assert.equal(messages[0]?.provisional, true);
    assert.equal(messages[0]?.generatedText, "8"); assert.deepEqual(messages[0]?.generatedTokenIds, [8]); assert.equal(messages[3]?.data?.generatedText, "7");
    const selected = messages[6]!.data!.direct; assert.deepEqual(selected.generatedTokenIds, [7]); assert.equal(selected.selectedBackend, "pytorch"); assert.equal(selected.fallbackTriggered, true); assert.equal(selected.fastPathMinimumMargin, 0); assert.deepEqual((selected.fastPath as { generatedTokenIds: number[] }).generatedTokenIds, [8]); assert.equal(selected.tokensEqualBaseline, true); assert.equal(selected.selectiveVerification, true); assert.deepEqual(selected.sensitiveSteps, [0]); assert.equal(selected.verificationHeadSteps, 1);
    assert.equal(selected.verificationPrefillAhead, true); assert.equal(selected.verificationPrefixAhead, undefined); assert.equal(selected.verificationPrefillAheadTokensComputed, 2); assert.equal(selected.verificationPrefillAheadCacheHit, false); assert.ok((selected.verificationPrefillOverlapSeconds as number) >= 0); assert.ok((selected.verificationPrefillWaitSeconds as number) >= 0);
    assert.equal(selected.verificationRequestCount, 1); assert.ok((selected.verificationAcquireSeconds as number) >= 0); assert.ok((selected.verificationRequestWallSeconds as number) >= 0); assert.ok((selected.verificationOrchestrationSeconds as number) >= 0);
    const generateCompiled = async () => {
      const compiledResponse = await fetch(`${endpoint}/api/generate-stream`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "empate compilado", maxNewTokens: 1 }) });
      assert.equal(compiledResponse.status, 200);
      const stream = (await compiledResponse.text()).trim().split("\n").map((line) => JSON.parse(line) as { type: string; data?: { direct: { fastPath?: { workerGenerations?: number } }; comparisonTiming: { backgroundRecoveryScheduled?: boolean; directBackgroundRecoveryWaitSeconds?: number } } });
      return stream.at(-1)!.data!;
    };
    const firstCompiled = await generateCompiled(); assert.equal(firstCompiled.comparisonTiming.backgroundRecoveryScheduled, true); assert.equal(firstCompiled.comparisonTiming.directBackgroundRecoveryWaitSeconds, 0);
    const secondCompiled = await generateCompiled(); assert.equal(secondCompiled.comparisonTiming.backgroundRecoveryScheduled, true); assert.equal(secondCompiled.direct.fastPath!.workerGenerations, firstCompiled.direct.fastPath!.workerGenerations! + 3);
    status = await fetch(`${endpoint}/api/status`).then((entry) => entry.json()) as typeof status; assert.ok(status.direct?.backgroundRecovery?.state === "idle" || status.direct?.backgroundRecovery?.state === "running"); assert.ok((status.direct?.backgroundRecovery?.lastSeconds ?? -1) >= 0); assert.equal(status.direct?.verification?.state, "ready"); assert.equal(status.direct?.verification?.ready, true); assert.equal(status.direct?.verification?.lifecycle, "warmed-persistent-exact-kernel-v1");
    const secondResponse = await fetch(`${endpoint}/api/compare`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "outro empate", maxNewTokens: 1 }) });
    const second = await secondResponse.json() as { direct: Record<string, unknown> };
    assert.equal(second.direct.workerPid, selected.workerPid); assert.equal(second.direct.workerGenerations, (selected.workerGenerations as number) + 3); assert.deepEqual(second.direct.generatedTokenIds, [7]); assert.equal(second.direct.tokensEqualBaseline, true);
    const diagnosticResponse = await fetch(`${endpoint}/api/diagnose-layers`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "diagnóstico", maxNewTokens: 1 }) });
    const diagnostic = await diagnosticResponse.json() as { kind: string; reference: { steps: Array<{ baselineLayerHidden?: string[] }> }; verifier: { decoderStack?: string; layerHiddenCaptures?: string[][] } };
    assert.equal(diagnostic.kind, "gemma4-decoder-layer-differential"); assert.deepEqual(diagnostic.reference.steps[0]?.baselineLayerHidden, ["AAAAAA=="]); assert.deepEqual(diagnostic.verifier.layerHiddenCaptures, [["AAAAAA=="]]); assert.equal(diagnostic.verifier.decoderStack, "native-bf16-ple");
  } finally { await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept())); await rm(directory, { recursive: true, force: true }); }
});

test("sessão com KV exato antecipa o prefixo sensível e reutiliza o restante Metal", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-prefix-ahead-test-"));
  const transformer = join(directory, "transformer.mjs"), direct = join(directory, "direct.mjs");
  await writeFile(transformer, `import readline from "node:readline";
console.log(JSON.stringify({ready:true}));
readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);const report=r.mode==='encode'?{tokenIds:r.text==='entrada diferente'?[2,100]:[2,99]}:r.mode==='decode'?{text:r.tokenIds.join('|')}:{baselineGeneratedTokenIds:[8,9,10],candidateGeneratedTokenIds:[8,9,10],baselineGeneratedText:'8|9|10',candidateGeneratedText:'8|9|10',generatedTokensEqual:true,firstDivergentStep:null,inputIds:[r.inputIds],steps:[],performance:{baselineSeconds:1,candidateSeconds:1,baselineTokensPerSecond:3,candidateTokensPerSecond:3,candidateSpeedup:1,processPeakRssBytes:0},executionThreads:1,candidatePrecision:'f32',roundingPolicy:'none'};console.log(JSON.stringify({id:r.id,report}));});\n`);
  await writeFile(direct, `import readline from "node:readline";
const args=process.argv,backend=args[args.indexOf('--linear-backend')+1];let prefixRequests=0;console.log(JSON.stringify({ready:true,linearBackend:backend}));
readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);if(r.verificationPrefill){console.log(JSON.stringify({id:r.id,report:{verificationPrefill:true,elapsedSeconds:0.01,verificationSessionCacheHit:false,verificationPrefixTokensReused:0,verificationPrefillTokensComputed:r.inputIds.length}}));return;}const prefixAhead=r.verificationFastPath&&r.verificationFastPath.terminalLogitsSha256===undefined;if(prefixAhead)prefixRequests++;const base=[8,9,10],tokens=(r.verificationFastPath?r.verificationFastPath.generatedTokenIds:base).slice(0,r.maxNewTokens),steps=tokens.map((tokenId,step)=>({step,tokenId,forwardSeconds:0.01,topLogits:step===0?[{tokenId:8,value:2},{tokenId:7,value:2}]:[{tokenId,value:3},{tokenId:7,value:2}],...(r.verificationFastPath&&step!==0?{verificationSkipped:true}:{})})),report={generatedTokenIds:tokens,fullTokenIds:[...r.inputIds,...tokens],elapsedSeconds:0.03,tokensPerSecond:tokens.length/0.03,linearBackend:backend,prefixRequests,terminalLogitsSha256:'${"0".repeat(64)}',steps,...(backend==='pytorch'?{verificationCachedContextTokens:r.inputIds.length,verificationSessionCacheHit:prefixAhead,verificationPrefixTokensReused:prefixAhead?r.inputIds.length:0,verificationPrefillTokensComputed:prefixAhead?0:r.inputIds.length}:{}),...(r.verificationFastPath?{selectiveVerification:true,sensitiveSteps:r.verificationFastPath.sensitiveSteps,trustedFastPathSteps:0,verificationHeadSteps:1,verificationDivergenceStep:null,verificationDecoderSteps:1,verificationDecoderStepsAvoided:0,verificationHeadPositionsComputed:1,verificationHeadPositionsAvoided:0}:{})};if(r.stream)for(const event of steps)console.log(JSON.stringify({id:r.id,event:{type:'token',...event}}));console.log(JSON.stringify({id:r.id,report}));});\n`);
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper: transformer, tokenizerHelper: transformer, literalArtifact: "literal.json", binaryPool: directory, directWorker: direct, directLinearHelper: "unused.py", directVerificationMargin: 0 });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP ausente."); const endpoint = `http://127.0.0.1:${address.port}`;
    let ready = false; for (let index = 0; index < 50 && !ready; index += 1) { ready = (await fetch(`${endpoint}/api/status`).then((response) => response.json()) as { ready: boolean }).ready; await new Promise((accept) => setTimeout(accept, 10)); }
    const generate = async (prompt = "empate confirmado") => { const response = await fetch(`${endpoint}/api/compare-stream`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt, maxNewTokens: 3, sessionId: 77 }) }); return (await response.text()).trim().split("\n").map((line) => JSON.parse(line)).at(-1).data.direct as Record<string, unknown>; };
    const first = await generate(); assert.deepEqual(first.generatedTokenIds, [8, 9, 10]); assert.equal(first.verificationPrefixAhead, undefined); assert.equal(first.verificationUncachedPrefillDeferred, true); assert.equal(first.verificationPrefillAhead, undefined);
    const second = await generate(); assert.deepEqual(second.generatedTokenIds, [8, 9, 10]); assert.equal(second.verificationPrefixAhead, true); assert.equal(second.verificationUncachedPrefillDeferred, undefined); assert.deepEqual([second.verificationPrefixAheadStep, second.verificationPrefixAheadTokenSteps, second.verificationEarlyExitStep], [0, 1, 0]); assert.equal(second.prefixRequests, 1); assert.equal(second.trustedFastPathSteps, 2);
    const changed = await generate("entrada diferente"); assert.deepEqual(changed.generatedTokenIds, [8, 9, 10]); assert.equal(changed.verificationPrefixAhead, undefined); assert.equal(changed.verificationUncachedPrefillDeferred, true); assert.equal(changed.prefixRequests, 1);
  } finally { await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept())); await rm(directory, { recursive: true, force: true }); }
});

test("modo down recupera margem sensível no mesmo worker MLX compilado", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-mlx-control-verification-test-"));
  const transformer = join(directory, "transformer.mjs"), direct = join(directory, "direct.mjs");
  await writeFile(transformer, `import readline from "node:readline";
console.log(JSON.stringify({ready:true}));
readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);const report=r.mode==='encode'?{tokenIds:[2,99]}:r.mode==='decode'?{text:r.tokenIds.join('|')}:{baselineGeneratedTokenIds:[7],candidateGeneratedTokenIds:[7],baselineGeneratedText:'7',candidateGeneratedText:'7',generatedTokensEqual:true,firstDivergentStep:null,inputIds:[[2,99]],steps:[{step:0,baselineToken:7,candidateToken:7,baselineTopLogits:[{tokenId:7,logit:2.125},{tokenId:8,logit:2}],candidateTopLogits:[],metrics:{argmaxEqual:true,divergenceRate:0,maxAbsError:0},baselineSeconds:1,candidateSeconds:1}],performance:{baselineSeconds:1,candidateSeconds:1,baselineTokensPerSecond:1,candidateTokensPerSecond:1,candidateSpeedup:1,processPeakRssBytes:0},executionThreads:1,candidatePrecision:'f32',roundingPolicy:'none'};console.log(JSON.stringify({id:r.id,report}));});\n`);
  await writeFile(direct, `import readline from "node:readline";
const args=process.argv,backend=args[args.indexOf('--linear-backend')+1],decoder=args[args.indexOf('--mlx-decoder-quantization')+1];console.log(JSON.stringify({ready:true,linearBackend:backend,mlxDecoderQuantization:decoder}));
let generations=0;readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);generations++;const control=r.verificationControl===true,token=control?7:8,top=control?[{tokenId:7,value:2.125},{tokenId:8,value:2}]:[{tokenId:8,value:2},{tokenId:7,value:2}],report={generatedTokenIds:[token],fullTokenIds:[...r.inputIds,token],elapsedSeconds:0.01,tokensPerSecond:100,linearThreads:4,linearBackend:backend,workerPid:process.pid,workerGenerations:generations,terminalLogitsSha256:'${"0".repeat(64)}',steps:[{step:0,tokenId:token,forwardSeconds:0.01,topLogits:top}]};if(r.stream)console.log(JSON.stringify({id:r.id,event:{type:'token',step:0,tokenId:token,forwardSeconds:0.01,topLogits:top}}));console.log(JSON.stringify({id:r.id,report}));});\n`);
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper: transformer, tokenizerHelper: transformer, literalArtifact: "literal.json", binaryPool: directory, directWorker: direct, directLinearHelper: "unused.py", directMlxDecoderQuantization: "q8-ffn-gate-up-down", directVerificationMargin: 0.125, directVerificationBackend: "mlx-shared-control" });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP ausente.");
    const endpoint = `http://127.0.0.1:${address.port}`;
    let status: { ready?: boolean; direct?: { verification?: { backend?: string; lifecycle?: string; warmupComplete?: boolean } } } = {};
    for (let index = 0; index < 50 && !status.ready; index += 1) { status = await fetch(`${endpoint}/api/status`).then((response) => response.json()) as typeof status; await new Promise((accept) => setTimeout(accept, 10)); }
    assert.equal(status.direct?.verification?.backend, "mlx-shared-control"); assert.equal(status.direct?.verification?.lifecycle, "shared-worker-compiled-control-v1"); assert.equal(status.direct?.verification?.warmupComplete, true);
    const response = await fetch(`${endpoint}/api/compare`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "margem sensível", maxNewTokens: 1 }) });
    const result = await response.json() as { direct: Record<string, unknown> };
    assert.deepEqual(result.direct.generatedTokenIds, [7]); assert.equal(result.direct.selectedBackend, "mlx-shared-control"); assert.equal(result.direct.selectionPolicy, "margin-verified-mlx-shared-control-v1"); assert.equal(result.direct.fallbackTriggered, true); assert.equal(result.direct.fastPathMinimumMargin, 0); assert.deepEqual((result.direct.fastPath as { generatedTokenIds: number[] }).generatedTokenIds, [8]); assert.equal(result.direct.tokensEqualBaseline, true); assert.equal(result.direct.selectiveVerification, undefined); assert.equal(result.direct.controlVerificationTokenSteps, 1); assert.equal(result.direct.controlCorrectionTriggered, true);
  } finally { await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept())); await rm(directory, { recursive: true, force: true }); }
});

test("verificador exato corrige a raiz e retoma a continuação compilada quando suas margens são seguras", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-root-correction-test-"));
  const transformer = join(directory, "transformer.mjs"), direct = join(directory, "direct.mjs");
  await writeFile(transformer, `import readline from "node:readline";
console.log(JSON.stringify({ready:true}));
readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);const report=r.mode==='encode'?{tokenIds:[2,99]}:r.mode==='decode'?{text:r.tokenIds.join('|')}:{baselineGeneratedTokenIds:[7,70,71],candidateGeneratedTokenIds:[7,70,71],generatedTokensEqual:true,firstDivergentStep:null,inputIds:[[2,99]],steps:[],performance:{baselineSeconds:1,candidateSeconds:1,baselineTokensPerSecond:3,candidateTokensPerSecond:3,candidateSpeedup:1,processPeakRssBytes:0},executionThreads:1,candidatePrecision:'f32',roundingPolicy:'none'};console.log(JSON.stringify({id:r.id,report}));});\n`);
  await writeFile(direct, `import readline from "node:readline";
const args=process.argv,backend=args[args.indexOf('--linear-backend')+1];console.log(JSON.stringify({ready:true,linearBackend:backend}));
let generations=0;readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);generations++;if(r.verificationPrefill){console.log(JSON.stringify({id:r.id,report:{verificationPrefill:true,elapsedSeconds:0.001}}));return;}const exact=backend==='pytorch',corrected=r.inputIds.includes(7),stop=r.verificationFastPath?.stopAfterDivergence===true;let tokens=exact?(stop?[7]:[7,70,71]):corrected?[70,71]:[8,80,81];tokens=tokens.slice(0,r.maxNewTokens);const rewound=corrected&&r.sessionId===3,sensitive=corrected&&(r.sessionId===2||rewound);const steps=tokens.map((tokenId,step)=>({step,tokenId,forwardSeconds:0.01,topLogits:tokenId===8||sensitive?[{tokenId,value:2},{tokenId:999,value:2}]:[{tokenId,value:3},{tokenId:999,value:2}]}));const report={generatedTokenIds:tokens,fullTokenIds:[...r.inputIds,...tokens],elapsedSeconds:exact?0.05:0.02,tokensPerSecond:tokens.length/(exact?0.05:0.02),linearBackend:backend,workerGenerations:generations,terminalLogitsSha256:'${"0".repeat(64)}',steps,sessionCacheHit:rewound,prefixTokensReused:rewound?2:0,prefillTokensComputed:rewound?1:r.inputIds.length,...(r.verificationFastPath?{selectiveVerification:true,sensitiveSteps:r.verificationFastPath.sensitiveSteps,trustedFastPathSteps:0,verificationHeadSteps:1,verificationDivergenceStep:0,verificationDecoderSteps:stop?1:3,verificationDecoderStepsAvoided:0,verificationHeadPositionsAvoided:0,...(stop?{verificationStoppedAfterDivergence:true}:{})}:{})};if(r.stream)for(const event of steps)console.log(JSON.stringify({id:r.id,event:{type:'token',...event}}));console.log(JSON.stringify({id:r.id,report}));});\n`);
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper: transformer, tokenizerHelper: transformer, literalArtifact: "literal.json", binaryPool: directory, directWorker: direct, directLinearHelper: "unused.py", directVerificationMargin: 0 });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP ausente.");
    let ready = false; for (let index = 0; index < 50 && !ready; index += 1) { ready = (await fetch(`http://127.0.0.1:${address.port}/api/status`).then((response) => response.json()) as { ready: boolean }).ready; await new Promise((accept) => setTimeout(accept, 10)); }
    const response = await fetch(`http://127.0.0.1:${address.port}/api/compare`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "corrigir", maxNewTokens: 3 }) });
    const report = await response.json() as { direct: Record<string, unknown> };
    assert.deepEqual(report.direct.generatedTokenIds, [7, 70, 71]); assert.equal(report.direct.tokensEqualBaseline, true);
    assert.equal(report.direct.selectionPolicy, "margin-verified-pytorch-root-mlx-continuation-v1"); assert.equal(report.direct.selectedBackend, "pytorch-root+mlx-continuation");
    assert.equal(report.direct.compiledContinuationAccepted, true); assert.equal(report.direct.compiledContinuationTokenSteps, 2); assert.equal(report.direct.compiledContinuationMinimumMargin, 1);
    assert.equal(report.direct.verificationDecoderSteps, 1); assert.equal(report.direct.verificationDecoderStepsAvoided, 2); assert.deepEqual((report.direct.fastPath as { generatedTokenIds: number[] }).generatedTokenIds, [8, 80, 81]);
    const rejectedResponse = await fetch(`http://127.0.0.1:${address.port}/api/compare`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "rejeitar", maxNewTokens: 3, sessionId: 2 }) });
    const rejected = await rejectedResponse.json() as { direct: Record<string, unknown> };
    assert.deepEqual(rejected.direct.generatedTokenIds, [7, 70, 71]); assert.equal(rejected.direct.tokensEqualBaseline, true);
    assert.equal(rejected.direct.selectionPolicy, "margin-verified-pytorch-v1"); assert.equal(rejected.direct.selectedBackend, "pytorch"); assert.equal(rejected.direct.compiledContinuationAccepted, false);
    assert.equal(rejected.direct.compiledContinuationRejectedReason, "margin-at-or-below-threshold"); assert.equal(rejected.direct.compiledContinuationMinimumMargin, 0); assert.equal(rejected.direct.verificationDecoderSteps, 3);
    const retriedResponse = await fetch(`http://127.0.0.1:${address.port}/api/compare`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "recompor", maxNewTokens: 3, sessionId: 3 }) });
    const retried = await retriedResponse.json() as { direct: Record<string, unknown> };
    assert.deepEqual(retried.direct.generatedTokenIds, [7, 70, 71]); assert.equal(retried.direct.tokensEqualBaseline, true);
    assert.equal(retried.direct.selectionPolicy, "margin-verified-pytorch-root-mlx-continuation-v1"); assert.equal(retried.direct.compiledContinuationAccepted, true); assert.equal(retried.direct.compiledContinuationPrefixRetry, true);
    assert.equal(retried.direct.compiledContinuationPrefixAttemptMinimumMargin, 0); assert.equal(retried.direct.compiledContinuationPrefixAttemptSeconds, 0.02); assert.equal(retried.direct.compiledContinuationSeconds, 0.04);
  } finally { await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept())); await rm(directory, { recursive: true, force: true }); }
});
