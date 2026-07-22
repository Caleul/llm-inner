import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Script } from "node:vm";
import { gemma4RealCompareHtml } from "../src/gemma4-real-compare-ui.js";
import { assessDirectVerification, assertDirectFinalFormulaProgram, computeDirectExecutionMetrics, computeDirectLogitAgreement, createGemma4RealComparisonServer, parseGemma4RealServerOptions, type Gemma4CompiledProgramStatus } from "../src/gemma4-real-compare-server.js";

test("interface diferencial contém controles e apresentação dos dois executores", () => {
  assert.match(gemma4RealCompareHtml, /Enviar e comparar/);
  assert.match(gemma4RealCompareHtml, /Original — BF16/);
  assert.match(gemma4RealCompareHtml, /Compilado \(compatibilidade\) — F32\/F64/);
  assert.match(gemma4RealCompareHtml, /Compilado direto — forward integral até logits/);
  assert.match(gemma4RealCompareHtml, /Tokens orig\. \/ compat\. \/ direto/);
  assert.match(gemma4RealCompareHtml, /linearBackend/);
  assert.match(gemma4RealCompareHtml, /lotes lineares/);
  assert.match(gemma4RealCompareHtml, /MLPs fundidos/);
  assert.match(gemma4RealCompareHtml, /projeção gate\/up/);
  assert.match(gemma4RealCompareHtml, /head BF16 exato: sem aproximação ou fallback/);
  assert.match(gemma4RealCompareHtml, /programa final direto/);
  assert.match(gemma4RealCompareHtml, /mapa executado/);
  assert.match(gemma4RealCompareHtml, /Inspetor das dimensões finais compiladas/);
  assert.match(gemma4RealCompareHtml, /Dimensão \/ token ID/);
  assert.match(gemma4RealCompareHtml, /\/api\/final-formula/);
  assert.match(gemma4RealCompareHtml, /EVAL_EXACT_DAG/);
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
  assert.match(gemma4RealCompareHtml, /Chat IT \(experimental\)/);
  assert.match(gemma4RealCompareHtml, /não transforma pesos base em pesos instruction-tuned/);
  assert.match(gemma4RealCompareHtml, /razão direto\/original/);
  assert.match(gemma4RealCompareHtml, /Executor direto ausente: gere ou informe o bundle compilado/);
  assert.match(gemma4RealCompareHtml, /clearComparison/);
  assert.match(gemma4RealCompareHtml, /AbortController/);
  assert.match(gemma4RealCompareHtml, /Direto: Δlogit \/ Δmargem \/ top-K/);
  assert.match(gemma4RealCompareHtml, /divergência direta top-K/);
  assert.match(gemma4RealCompareHtml, /compatibilidade:.*direto:/);
  assert.match(gemma4RealCompareHtml, /prefixo KV/);
  assert.match(gemma4RealCompareHtml, /Primeiro token/);
  assert.match(gemma4RealCompareHtml, /prefillAvoidedRate/);
  assert.match(gemma4RealCompareHtml, /\/api\/compare-stream/);
  assert.match(gemma4RealCompareHtml, /Aquecendo o forward compilado no Metal/);
  assert.match(gemma4RealCompareHtml, /gate\+up unidos/);
  assert.match(gemma4RealCompareHtml, /cache de constantes F32/);
  assert.match(gemma4RealCompareHtml, /ranking final GPU/);
  assert.match(gemma4RealCompareHtml, /fullLogitTransfersAvoided/);
  assert.match(gemma4RealCompareHtml, /CSE RoPE/);
  assert.match(gemma4RealCompareHtml, /ropeFactorBuildsAvoided/);
  assert.match(gemma4RealCompareHtml, /validações redundantes evitadas/);
  assert.match(gemma4RealCompareHtml, /kvPrefixValidationScansAvoided/);
  assert.match(gemma4RealCompareHtml, /decoder→logits compilado/);
  assert.match(gemma4RealCompareHtml, /compiledIncrementalDecoderSteps/);
  assert.match(gemma4RealCompareHtml, /incrementalCompilerCacheHit/);
  assert.match(gemma4RealCompareHtml, /Isolada \(fiel\)/);
  assert.match(gemma4RealCompareHtml, /Paralela \(stress\)/);
  assert.match(gemma4RealCompareHtml, /direct-complete/);
  assert.match(gemma4RealCompareHtml, /reference-loading/);
  assert.match(gemma4RealCompareHtml, /direct-rewarming/);
  assert.match(gemma4RealCompareHtml, /modelo original ainda não carregado/);
  assert.match(gemma4RealCompareHtml, /parede total/);
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
  const embedded = gemma4RealCompareHtml.match(/<script>([\s\S]*)<\/script>/)?.[1]; assert.ok(embedded); assert.doesNotThrow(() => new Script(embedded), "JavaScript embutido deve ser sintaticamente executável pelo navegador");
});

test("modo compilado exige que o worker autentique exatamente o mapa final", () => {
  const expected = { file: "final-formulas.json", sha256: "a".repeat(64), functions: 262_144, inputTensor: "x" as const, evaluator: "BF16_RNE(EVAL_EXACT_DAG(root,x))" as const, orderedRootsSha256: "b".repeat(64) };
  assert.doesNotThrow(() => assertDirectFinalFormulaProgram({ finalFormulaProgram: { execution: "vectorized-shared-dag-output-program", functions: 262_144, fileSha256: "a".repeat(64), orderedRootsSha256: "b".repeat(64) } }, expected));
  assert.throws(() => assertDirectFinalFormulaProgram({}, expected), /não autenticou/);
  assert.throws(() => assertDirectFinalFormulaProgram({ finalFormulaProgram: { execution: "vectorized-shared-dag-output-program", functions: 262_143, fileSha256: "a".repeat(64), orderedRootsSha256: "b".repeat(64) } }, expected), /vínculo divergente/);
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
    assert.equal(status.ready, true); assert.equal(status.chatTemplate, "llm-inner-gemma4-it-text-turn-v1"); assert.equal(status.checkpointChatTemplateDeclared, false); assert.equal(status.direct.warmupComplete, true); assert.ok(status.direct.warmupSeconds >= 0);
  } finally { await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept())); await rm(directory, { recursive: true, force: true }); }
});

test("streaming entrega tokens antes do relatório e reutiliza prefixo comprovado da sessão", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-stream-session-test-"));
  const transformer = join(directory, "transformer.mjs"), direct = join(directory, "direct.mjs");
  await writeFile(transformer, `import readline from "node:readline";
console.log(JSON.stringify({ready:true}));
readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);let report;if(r.mode==='encode'){const first=r.text==='<|turn>user\\nprimeiro<turn|>\\n<|turn>model\\n',next=r.text==='<turn|>\\n<|turn>user\\ncontinuação<turn|>\\n<|turn>model\\n';report={tokenIds:first?[2,105]:next?[106,105]:[999]};}else if(r.mode==='decode')report={text:r.tokenIds.join('|')};else{const ids=String(r.inputIds).split(',').map(Number),token=ids.includes(7)?8:7;report={inputIds:[ids],baselineGeneratedTokenIds:[token],generatedTokensEqual:true,steps:[],performance:{baselineTokensPerSecond:1,candidateTokensPerSecond:1,candidateSpeedup:1,processPeakRssBytes:0},executionThreads:1,candidatePrecision:'f32',roundingPolicy:'none'};}console.log(JSON.stringify({id:r.id,report}));});\n`);
  await writeFile(direct, `import readline from "node:readline";
console.log(JSON.stringify({ready:true}));
readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line),token=r.inputIds.includes(7)?8:7,hit=r.inputIds.includes(7),reused=hit?r.inputIds.indexOf(7)+1:0,emit=()=>console.log(JSON.stringify({id:r.id,report:{generatedTokenIds:[token],fullTokenIds:[...r.inputIds,token],elapsedSeconds:0.01,tokensPerSecond:100,linearThreads:4,sessionCacheHit:hit,prefixTokensReused:reused,prefillTokensComputed:r.inputIds.length-reused,cachedContextTokens:r.inputIds.length,receivedInputIds:r.inputIds}}));if(r.stream){console.log(JSON.stringify({id:r.id,event:{type:'token',step:0,tokenId:token,forwardSeconds:0.01,topLogits:[]}}));setTimeout(emit,40);}else emit();});\n`);
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper: transformer, tokenizerHelper: transformer, literalArtifact: "literal.json", binaryPool: directory, directWorker: direct, directLinearHelper: "unused.py" });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP ausente.");
    const endpoint = `http://127.0.0.1:${address.port}/api/compare-stream`;
    let initialStatus: { ready?: boolean; reference?: { state?: string }; tokenizer?: { ready?: boolean } } = {};
    for (let index = 0; index < 100 && !initialStatus.ready; index += 1) { initialStatus = await fetch(`http://127.0.0.1:${address.port}/api/status`).then((response) => response.json()) as typeof initialStatus; await new Promise((accept) => setTimeout(accept, 5)); }
    assert.equal(initialStatus.tokenizer?.ready, true); assert.equal(initialStatus.reference?.state, "unloaded");
    const stream = async (prompt: string, continueSession: boolean) => {
      const response = await fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt, maxNewTokens: 1, sessionId: 41, continueSession, conversationMode: "chat" }) });
      assert.equal(response.status, 200); const lines = (await response.text()).trim().split("\n").map((line) => JSON.parse(line) as { type: string; serverElapsedSeconds?: number; event?: { tokenId: number }; data?: { chatTemplate?: string; generatedText?: string; direct?: { receivedInputIds: number[]; sessionCacheHit: boolean; prefixTokensReused: number }; directExecutionMetrics?: { inputTokens: number; firstTokenWallSeconds: number | null; prefixTokensReused: number; prefillTokensComputed: number; prefillAvoidedRate: number; tokensEqualBaseline: boolean }; comparisonTiming?: { schedule: string; directPhaseSeconds: number; referencePhaseSeconds: number; referenceReleased: boolean; directRecoverySeconds: number; totalWallSeconds: number } } }); return lines;
    };
    const first = await stream("primeiro", false); assert.deepEqual(first.map((entry) => entry.type), ["direct-token", "direct-complete", "reference-loading", "direct-rewarming", "result"]); assert.equal(first[0]!.event!.tokenId, 7); assert.ok(first[0]!.serverElapsedSeconds! >= 0); assert.equal(first[1]!.data!.generatedText, "7"); assert.deepEqual(first[4]!.data!.direct!.receivedInputIds, [2, 105]); assert.equal(first[4]!.data!.chatTemplate, "llm-inner-gemma4-it-text-turn-v1"); assert.deepEqual(first[4]!.data!.directExecutionMetrics, { inputTokens: 2, generatedTokens: 1, firstTokenForwardSeconds: null, firstTokenWallSeconds: first[4]!.data!.directExecutionMetrics!.firstTokenWallSeconds, decodeForwardSeconds: 0, totalForwardSeconds: 0, sessionCacheHit: false, prefixTokensReused: 0, prefillTokensComputed: 2, prefillAvoidedRate: 0, baselineSpeedup: null, tokensEqualBaseline: true }); assert.ok(first[4]!.data!.directExecutionMetrics!.firstTokenWallSeconds! >= 0); assert.equal(first[4]!.data!.comparisonTiming!.schedule, "isolated"); assert.equal(first[4]!.data!.comparisonTiming!.referenceReleased, true); assert.ok(first[4]!.data!.comparisonTiming!.directRecoverySeconds >= 0); assert.ok(first[4]!.data!.comparisonTiming!.directPhaseSeconds >= 0); assert.ok(first[4]!.data!.comparisonTiming!.referencePhaseSeconds >= 0); assert.ok(first[4]!.data!.comparisonTiming!.totalWallSeconds >= first[4]!.data!.comparisonTiming!.directPhaseSeconds);
    const releasedStatus = await fetch(`http://127.0.0.1:${address.port}/api/status`).then((response) => response.json()) as { reference: { state: string } }; assert.equal(releasedStatus.reference.state, "unloaded");
    const second = await stream(" continuação", true); assert.deepEqual(second.map((entry) => entry.type), ["direct-token", "direct-complete", "reference-loading", "direct-rewarming", "result"]); assert.equal(second[0]!.event!.tokenId, 8); assert.deepEqual(second[4]!.data!.direct!.receivedInputIds, [2, 105, 7, 106, 105]); assert.equal(second[4]!.data!.direct!.sessionCacheHit, true); assert.equal(second[4]!.data!.direct!.prefixTokensReused, 3); assert.equal(second[4]!.data!.directExecutionMetrics!.prefixTokensReused, 3); assert.equal(second[4]!.data!.directExecutionMetrics!.prefillTokensComputed, 2); assert.equal(second[4]!.data!.directExecutionMetrics!.prefillAvoidedRate, 0.6); assert.equal(second[4]!.data!.directExecutionMetrics!.tokensEqualBaseline, true);
    const parallelResponse = await fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "stress", maxNewTokens: 1, measurementSchedule: "parallel" }) });
    const parallel = (await parallelResponse.text()).trim().split("\n").map((line) => JSON.parse(line) as { type: string; data?: { comparisonTiming?: { schedule: string } } }); assert.deepEqual(parallel.map((entry) => entry.type), ["reference-loading", "direct-token", "direct-rewarming", "result"]); assert.equal(parallel.at(-1)!.data!.comparisonTiming!.schedule, "parallel");
    const controller = new AbortController(); const cancelled = await fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, signal: controller.signal, body: JSON.stringify({ prompt: "cancelar", maxNewTokens: 1, sessionId: 41, continueSession: true, conversationMode: "chat" }) }); const reader = cancelled.body!.getReader(); await reader.read(); controller.abort(); await new Promise((resolve) => setTimeout(resolve, 80));
    const retry = await stream("retry", true); assert.deepEqual(retry.at(-1)!.data!.direct!.receivedInputIds, [2, 105, 7, 106, 105, 8, 999]);
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
  assert.throws(() => parseGemma4RealServerOptions(["--direct-mlx-head-quantization", "q2"]), /off, q8 ou q4/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-native-attention", "auto"]), /off, bf16 ou real/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-attention", "always"]), /off, bf16, real ou native-bf16/);
  const directDefaults = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool"]);
  assert.equal(directDefaults.directThreads, 10); assert.equal(directDefaults.directMaxReadMiB, 16); assert.equal(directDefaults.directFinalHeadReadMiB, 16); assert.equal(directDefaults.directVerificationMargin, 0);
  assert.equal(directDefaults.directMlxHeadQuantization, "off");
  assert.equal(directDefaults.directMlxDecoderQuantization, "q8-ffn-gate-up");
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-mlx-decoder-quantization", "off"]).directMlxDecoderQuantization, "off");
  const q4Layers = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-mlx-decoder-quantization", "q4-ffn-gate-up", "--direct-mlx-decoder-quantization-layers", "0-9"]);
  assert.equal(q4Layers.directMlxDecoderQuantization, "q4-ffn-gate-up"); assert.equal(q4Layers.directMlxDecoderQuantizationLayers, "0-9");
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-mlx-decoder-quantization-layers", "3,1-2,2"]).directMlxDecoderQuantizationLayers, "1-3");
  assert.throws(() => parseGemma4RealServerOptions(["--direct-mlx-decoder-quantization-layers", "42"]), /entre 0 e 41/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-mlx-decoder-quantization", "off", "--direct-mlx-decoder-quantization-layers", "0"]), /requer q8-ffn-gate-up ou q4-ffn-gate-up/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-mlx-decoder-quantization", "q2"]), /modo inválido/);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-verification-margin", "off"]).directVerificationMargin, undefined);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-verification-margin", "-1"]), /finito não negativo/);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "pytorch", "--direct-fused-decoder-stack", "native-bf16-ple"]).directFusedDecoderStack, "native-bf16-ple");
  const splitTiles = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-max-read-mib", "16", "--direct-final-head-read-mib", "32"]);
  assert.equal(splitTiles.directFinalHeadReadMiB, 32);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-final-head-read-mib", "0"]), /Configuração direta inválida/);
  assert.equal(directDefaults.directLinearBackend, "mlx"); assert.equal(directDefaults.directFusedMlp, "real"); assert.equal(directDefaults.directFusedFfn, "off"); assert.equal(directDefaults.directFusedDecoderLayer, "off"); assert.equal(directDefaults.directFusedDecoderStack, "real"); assert.equal(directDefaults.directFusedPle, "off"); assert.equal(directDefaults.directFusedPlePrelude, "bf16"); assert.equal(directDefaults.directFusedTokenForward, "bf16"); assert.equal(directDefaults.directResidentGeneration, "on"); assert.equal(directDefaults.directFinalHead, "native-bf16-whole"); assert.equal(directDefaults.directNativeAttention, "off"); assert.equal(directDefaults.directFusedAttention, "off");
  const mlxDefaults = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "mlx"]);
  assert.equal(mlxDefaults.directFinalHeadReadMiB, 16); assert.equal(mlxDefaults.directFusedMlp, "real"); assert.equal(mlxDefaults.directFusedFfn, "off"); assert.equal(mlxDefaults.directFusedDecoderLayer, "off"); assert.equal(mlxDefaults.directFusedDecoderStack, "real"); assert.equal(mlxDefaults.directFusedPle, "off"); assert.equal(mlxDefaults.directFusedPlePrelude, "bf16"); assert.equal(mlxDefaults.directFusedTokenForward, "bf16"); assert.equal(mlxDefaults.directResidentGeneration, "on"); assert.equal(mlxDefaults.directFinalHead, "native-bf16-whole"); assert.equal(mlxDefaults.directNativeAttention, "off"); assert.equal(mlxDefaults.directFusedAttention, "off");
  const pytorchDefaults = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "pytorch"]);
  assert.equal(pytorchDefaults.directMlxDecoderQuantization, "off");
  assert.equal(pytorchDefaults.directFinalHeadReadMiB, 32); assert.equal(pytorchDefaults.directVerificationMargin, undefined); assert.equal(pytorchDefaults.directFusedMlp, "native-bf16"); assert.equal(pytorchDefaults.directFusedFfn, "native-bf16"); assert.equal(pytorchDefaults.directFusedDecoderLayer, "native-bf16"); assert.equal(pytorchDefaults.directFusedDecoderStack, "native-bf16"); assert.equal(pytorchDefaults.directFusedPle, "bf16"); assert.equal(pytorchDefaults.directFusedPlePrelude, "off"); assert.equal(pytorchDefaults.directFusedTokenForward, "off"); assert.equal(pytorchDefaults.directResidentGeneration, "off"); assert.equal(pytorchDefaults.directFinalHead, "native-bf16-stream"); assert.equal(pytorchDefaults.directNativeAttention, "real"); assert.equal(pytorchDefaults.directFusedAttention, "native-bf16");
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
    const compiledOutputProgram = { id: "gemma4-text-real-final-vectors", semantics: "shared-dag-parametric-output-functions-v1", logicalDispatchesPerForward: 1, operationFunctions: 1, outputFunctions: 1, firstOperationId: "operation_0", terminalOperationId: "operation_0", orderedDispatchSha256 };
    const outputBinding = { name: "terminal_logit", operationId: "operation_0", fixedDimension: 0, coordinate: [], parameters: [], root: "root:output", finalQuantization: "BF16-round-to-nearest-ties-to-even" };
    const outputBindingsSha256 = createHash("sha256").update(JSON.stringify([outputBinding])).digest("hex");
    const standaloneSsaOutputsSha256 = createHash("sha256").update(JSON.stringify([{ assignment: "calc_terminal_logit_0", value: "root:output", parameters: [], coordinate: [], finalQuantization: "BF16-round-to-nearest-ties-to-even" }])).digest("hex");
    const planBytes = Buffer.from(JSON.stringify({ kind: "gemma4-vectorized-real-lowering-plan", schemaVersion: 3, contract: { kind: "gemma4-vectorized-real-lowering-contract", schemaVersion: 3, semantics: "gemma4-exact-real-simplified-v1", execution: { engine: "mlx-f32-real-decoder-stack-v1", mode: "compiled-parametric-output-program", intermediateBf16Boundaries: 0, finalQuantization: "BF16-round-to-nearest-ties-to-even", directlyLoadsStandaloneSsaFile: false, sourceProgramValidated: true, directlyExecutesGlobalFormula: true }, source: { artifactIntegritySha256: "f".repeat(64), realSimplifiedProgramSha256, sourceAssignments: 1, expressionNodes: 1, operationFunctions: 1, globalClosureFunctions: 1, standaloneRuntimeReductions: 0, outputFunctions: 1, outputBindingsSha256, standaloneSsaOutputsSha256, outputFamilies: { terminal_logit: { dimensions: 1, operationId: "operation_0", parameters: [], finalQuantization: "BF16-round-to-nearest-ties-to-even" } } }, coverage: { unresolvedRuntimeReductions: 0, intermediateIeeeRoundingNodes: 0, unsupportedOperations: 0, globalClosuresDependingOnStandaloneReductions: 0, kernels: { activation: 1, elementwise: 0, linear: 0, reshape_heads: 0, rms_norm: 0, rotary_embedding: 0, scaled_dot_product_attention: 0, select_per_layer: 0, tensor_scale: 0 } }, functionBindingsSha256, compiledOutputProgram }, functionBindings }));
    const planSha256 = createHash("sha256").update(planBytes).digest("hex"); await writeFile(join(directory, "vectorized-real-lowering.json"), planBytes);
    await writeFile(join(directory, "constants.runtime-index.json"), "fixture-index");
    await writeFile(join(directory, "manifest.json"), JSON.stringify({ kind: "gemma4-compiled-shared-dag-bundle", schemaVersion: 3, execution: "compiled-parametric-output-program-runtime", runtimeLowering: { engine: "mlx-f32-real-decoder-stack-v1", directlyExecutesGlobalFormula: true, executesPersistedLoweringPlan: true, plan: "vectorized-real-lowering.json", functionBindingsSha256, outputBindingsSha256, standaloneSsaOutputsSha256, outputFunctions: 1, realSimplifiedProgramSha256, globalFormulaRole: "compiled-executable-shared-dag", compiledOutputProgram }, formula: { family: "terminal_logit", dimension: 0, root: `sha256:${"1".repeat(64)}`, expressionNodes: 1, inputTensor: "x", inputLength: 1, file: "formula.graph.json" }, globalProgram: { file: "global-formulas.ssa.json", terminalLogits: 1, constantPool: "constants.literal.json" }, finalFormulaMap: { file: "final-formulas.json", functions: 1, inputTensor: "x", evaluator: "BF16_RNE(EVAL_EXACT_DAG(root,x))", globalFormulaSha256: "a".repeat(64), orderedRootsSha256: "e".repeat(64) }, runtimeIndex: { file: "constants.runtime-index.json", schemaVersion: 2, constantPoolSha256: "b".repeat(64), integrityRootSha256: "f".repeat(64) }, files: [{ role: "formula-graph", file: "formula.graph.json", sha256: "c".repeat(64) }, { role: "global-formulas", file: "global-formulas.ssa.json", sha256: "a".repeat(64) }, { role: "final-formulas", file: "final-formulas.json", sha256: finalFormulaSha256 }, { role: "constant-pool", file: "constants.literal.json", sha256: "b".repeat(64) }, { role: "literal-runtime-index", file: "constants.runtime-index.json", sha256: "d".repeat(64) }, { role: "vectorized-real-lowering", file: "vectorized-real-lowering.json", sha256: planSha256 }] }));
    const bundled = parseGemma4RealServerOptions(["--compiled-bundle", directory]);
    assert.equal(bundled.source, directory); assert.equal(bundled.literalArtifact, join(directory, "constants.literal.json")); assert.equal(bundled.binaryPool, directory);
    assert.equal(bundled.directArtifactIndex, join(directory, "constants.runtime-index.json")); assert.equal(bundled.directArtifactIndexSha256, "d".repeat(64)); assert.equal(bundled.directArtifactSha256, "b".repeat(64));
    assert.equal(bundled.compiledProgram?.runtimeIndex?.schemaVersion, 2);
    assert.deepEqual(bundled.compiledProgram?.finalFormulaMap, { file: "final-formulas.json", sha256: finalFormulaSha256, functions: 1, inputTensor: "x", evaluator: "BF16_RNE(EVAL_EXACT_DAG(root,x))", orderedRootsSha256: "e".repeat(64) });
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
  const output = (family: string, dimension: number, digit: string) => ({ assignment: `calc_${family}_${dimension}`, value: root(digit), parameters: ["batch", "sequence"], coordinate: [root("a"), root("b"), root(digit)], finalQuantization: "BF16-round-to-nearest-ties-to-even" });
  await writeFile(globalFile, JSON.stringify({ statements: [], functions: [], outputs: [output("final_hidden_dimension", 0, "1"), output("terminal_logit", 0, "2"), output("terminal_logit", 1, "3")] }));
  await writeFile(helper, `import readline from "node:readline"; console.log(JSON.stringify({ready:true})); readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);const report=r.mode==='decode'?{text:'token:'+r.tokenIds[0]}:r.mode==='encode'?{tokenIds:[2]}:{inputIds:[[2]],baselineGeneratedTokenIds:[1]};console.log(JSON.stringify({id:r.id,report}));});\n`);
  const compiledProgram: Gemma4CompiledProgramStatus = {
    bundle: directory, execution: "compiled-parametric-output-program-runtime", formulaSemantics: "gemma4-exact-real-simplified-v1",
    globalFormula: { file: "global-formulas.ssa.json", sha256: "4".repeat(64), terminalLogits: 2, role: "compiled-executable-shared-dag" }, constantPool: { file: "constants.literal.json", sha256: "5".repeat(64) },
    finalFormulaMap: { file: "final-formulas.json", sha256: "d".repeat(64), functions: 2, inputTensor: "x", evaluator: "BF16_RNE(EVAL_EXACT_DAG(root,x))", orderedRootsSha256: "e".repeat(64) },
    exampleFormula: { key: "calc_final_0", family: "terminal_logit", dimension: 0, root: root("2"), expressionNodes: 1, inputTensor: "x", inputLength: 1, file: "formula.graph.json", sha256: "6".repeat(64) },
    terminalLogitOutputs: { offset: 1, dimensions: 2, operationId: "final", finalQuantization: "BF16-round-to-nearest-ties-to-even" },
    directRuntime: { engine: "mlx-f32-real-decoder-stack-v1", directlyExecutesGlobalFormula: true, executesPersistedLoweringPlan: true, compiledOutputProgram: { id: "gemma4-text-real-final-vectors", semantics: "shared-dag-parametric-output-functions-v1", logicalDispatchesPerForward: 1, operationFunctions: 1, outputFunctions: 3, firstOperationId: "first", terminalOperationId: "final", orderedDispatchSha256: "7".repeat(64) }, plan: { file: "vectorized-real-lowering.json", sha256: "8".repeat(64), functionBindingsSha256: "9".repeat(64), outputBindingsSha256: "a".repeat(64), standaloneSsaOutputsSha256: "b".repeat(64), outputFunctions: 3, realSimplifiedProgramSha256: "c".repeat(64) } },
  };
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper, tokenizerHelper: helper, compiledProgram });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP de teste ausente.");
    const response = await fetch(`http://127.0.0.1:${address.port}/api/final-formula?dimension=1`); assert.equal(response.status, 200);
    const formula = await response.json() as { key: string; root: string; token: { id: number; text: string }; onlyFreeInput: string; formulaMap: { file: string; key: string; sha256: string } };
    assert.equal(formula.key, "calc_final_1"); assert.equal(formula.root, root("3")); assert.deepEqual(formula.token, { id: 1, text: "token:1" }); assert.equal(formula.onlyFreeInput, "x"); assert.deepEqual(formula.formulaMap, { file: "final-formulas.json", key: "calc_final_1", sha256: "d".repeat(64) });
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

test("servidor substitui empate Metal pelo resultado do verificador compilado", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-selective-verification-test-"));
  const transformer = join(directory, "transformer.mjs"), direct = join(directory, "direct.mjs");
  await writeFile(transformer, `import readline from "node:readline";
console.log(JSON.stringify({ready:true}));
readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);const report=r.mode==='encode'?{tokenIds:[2,99]}:r.mode==='decode'?{text:r.tokenIds.join('|')}:{baselineGeneratedTokenIds:[7],candidateGeneratedTokenIds:[7],baselineGeneratedText:'7',candidateGeneratedText:'7',generatedTokensEqual:true,firstDivergentStep:null,inputIds:[[2,99]],steps:[{step:0,baselineToken:7,candidateToken:7,baselineTopLogits:[{tokenId:7,logit:2.125},{tokenId:8,logit:2}],candidateTopLogits:[],metrics:{argmaxEqual:true,divergenceRate:0,maxAbsError:0},baselineSeconds:1,candidateSeconds:1}],performance:{baselineSeconds:1,candidateSeconds:1,baselineTokensPerSecond:1,candidateTokensPerSecond:1,candidateSpeedup:1,processPeakRssBytes:0},executionThreads:1,candidatePrecision:'f32',roundingPolicy:'none'};console.log(JSON.stringify({id:r.id,report}));});\n`);
  await writeFile(direct, `import readline from "node:readline";
const args=process.argv,backend=args[args.indexOf('--linear-backend')+1];console.log(JSON.stringify({ready:true,linearBackend:backend}));
let generations=0;readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);if(r.control){console.log(JSON.stringify({id:r.id,report:{trimmed:true}}));return;}generations++;const token=backend==='pytorch'?7:8,top=backend==='pytorch'?[{tokenId:7,value:2.125},{tokenId:8,value:2}]:[{tokenId:8,value:2},{tokenId:7,value:2}],report={generatedTokenIds:[token],fullTokenIds:[...r.inputIds,token],elapsedSeconds:0.01,tokensPerSecond:100,linearThreads:4,linearBackend:backend,workerPid:process.pid,workerGenerations:generations,terminalLogitsSha256:'${"0".repeat(64)}',steps:[{step:0,tokenId:token,forwardSeconds:0.01,topLogits:top}],...(r.verificationFastPath?{selectiveVerification:true,sensitiveSteps:r.verificationFastPath.sensitiveSteps,trustedFastPathSteps:0,verificationHeadSteps:1,verificationDivergenceStep:0}:{})};if(r.stream)console.log(JSON.stringify({id:r.id,event:{type:'token',step:0,tokenId:token,forwardSeconds:0.01,topLogits:top}}));console.log(JSON.stringify({id:r.id,report}));});\n`);
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper: transformer, tokenizerHelper: transformer, literalArtifact: "literal.json", binaryPool: directory, directWorker: direct, directLinearHelper: "unused.py", directVerificationMargin: 0 });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP ausente.");
    const endpoint = `http://127.0.0.1:${address.port}`;
    let status: { ready?: boolean; direct?: { verification?: { ready?: boolean; state?: string; marginThreshold?: number; lifecycle?: string } } } = {};
    for (let index = 0; index < 50 && !status.ready; index += 1) { status = await fetch(`${endpoint}/api/status`).then((response) => response.json()) as typeof status; await new Promise((accept) => setTimeout(accept, 10)); }
    assert.equal(status.direct?.verification?.ready, false); assert.equal(status.direct?.verification?.state, "unloaded"); assert.equal(status.direct?.verification?.marginThreshold, 0);
    const response = await fetch(`${endpoint}/api/compare-stream`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "empate", maxNewTokens: 1 }) });
    const messages = (await response.text()).trim().split("\n").map((line) => JSON.parse(line) as { type: string; provisional?: boolean; data?: { direct: Record<string, unknown> } });
    assert.deepEqual(messages.map((message) => message.type), ["direct-token", "direct-fallback", "direct-complete", "reference-loading", "direct-rewarming", "result"]); assert.equal(messages[0]?.provisional, true);
    const selected = messages[5]!.data!.direct; assert.deepEqual(selected.generatedTokenIds, [7]); assert.equal(selected.selectedBackend, "pytorch"); assert.equal(selected.fallbackTriggered, true); assert.equal(selected.fastPathMinimumMargin, 0); assert.deepEqual((selected.fastPath as { generatedTokenIds: number[] }).generatedTokenIds, [8]); assert.equal(selected.tokensEqualBaseline, true); assert.equal(selected.selectiveVerification, true); assert.deepEqual(selected.sensitiveSteps, [0]); assert.equal(selected.verificationHeadSteps, 1);
    status = await fetch(`${endpoint}/api/status`).then((entry) => entry.json()) as typeof status; assert.equal(status.direct?.verification?.state, "ready"); assert.equal(status.direct?.verification?.ready, true); assert.equal(status.direct?.verification?.lifecycle, "retained-index-restarted-kernel-v1");
    const secondResponse = await fetch(`${endpoint}/api/compare`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "outro empate", maxNewTokens: 1 }) });
    const second = await secondResponse.json() as { direct: Record<string, unknown> };
    assert.equal(second.direct.workerPid, selected.workerPid); assert.equal(second.direct.workerGenerations, 2); assert.deepEqual(second.direct.generatedTokenIds, [7]); assert.equal(second.direct.tokensEqualBaseline, true);
  } finally { await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept())); await rm(directory, { recursive: true, force: true }); }
});
