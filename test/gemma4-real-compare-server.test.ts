import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gemma4RealCompareHtml } from "../src/gemma4-real-compare-ui.js";
import { createGemma4RealComparisonServer, parseGemma4RealServerOptions } from "../src/gemma4-real-compare-server.js";

test("interface diferencial contém controles e apresentação dos dois executores", () => {
  assert.match(gemma4RealCompareHtml, /Gerar e comparar/);
  assert.match(gemma4RealCompareHtml, /Original — BF16/);
  assert.match(gemma4RealCompareHtml, /Compilado \(compatibilidade\) — F32\/F64/);
  assert.match(gemma4RealCompareHtml, /Compilado direto — forward integral até logits/);
  assert.match(gemma4RealCompareHtml, /Tokens orig\. \/ compat\. \/ direto/);
  assert.match(gemma4RealCompareHtml, /linearBackend/);
  assert.match(gemma4RealCompareHtml, /lotes lineares/);
  assert.match(gemma4RealCompareHtml, /MLPs fundidos/);
  assert.match(gemma4RealCompareHtml, /FFNs completos/);
  assert.match(gemma4RealCompareHtml, /Decoder layers completas/);
  assert.match(gemma4RealCompareHtml, /Pilhas decoder completas/);
  assert.match(gemma4RealCompareHtml, /pilhas até logits/);
  assert.match(gemma4RealCompareHtml, /gerações residentes/);
  assert.match(gemma4RealCompareHtml, /chamadas externas/);
  assert.match(gemma4RealCompareHtml, /transporte KV/);
  assert.match(gemma4RealCompareHtml, /Aquecendo o forward compilado no Metal/);
  assert.match(gemma4RealCompareHtml, /gate\+up unidos/);
  assert.match(gemma4RealCompareHtml, /cache de constantes F32/);
  assert.match(gemma4RealCompareHtml, /fases stack attn\/FFN\/PLE/);
  assert.match(gemma4RealCompareHtml, /fases stack: fundidas no grafo Metal/);
  assert.match(gemma4RealCompareHtml, /PLEs fundidos/);
  assert.match(gemma4RealCompareHtml, /head:/);
  assert.match(gemma4RealCompareHtml, /attention core:/);
  assert.match(gemma4RealCompareHtml, /attention fused:/);
  assert.match(gemma4RealCompareHtml, /tempo worker ref\/head\/attn\/MLP\/FFN\/layer\/stack\/PLE/);
  assert.match(gemma4RealCompareHtml, /Threads/);
  assert.match(gemma4RealCompareHtml, /\/api\/compare/);
});

test("servidor integra geração direta persistente e decodifica seus tokens", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-direct-worker-test-"));
  const transformer = join(directory, "transformer.mjs"), direct = join(directory, "direct.mjs");
  await writeFile(transformer, `import readline from "node:readline";
console.log(JSON.stringify({ready:true,initializationSeconds:0.1}));
readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);const report=r.mode==='decode'?{text:r.tokenIds.join('|')}:{inputIds:[[2]],baselineGeneratedTokenIds:[7]};console.log(JSON.stringify({id:r.id,report}));});\n`);
  await writeFile(direct, `import readline from "node:readline";
console.log(JSON.stringify({ready:true,initializationSeconds:0.2}));
let requests=0; readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);requests++;console.log(JSON.stringify({id:r.id,report:{generatedTokenIds:[7],fullTokenIds:[2,7],elapsedSeconds:1,tokensPerSecond:1,linearThreads:4,requests}}));});\n`);
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper: transformer, literalArtifact: "literal.json", binaryPool: directory, directWorker: direct, directLinearHelper: "unused.py", directThreads: 4 });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP de teste ausente.");
    const response = await fetch(`http://127.0.0.1:${address.port}/api/compare`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "x", maxNewTokens: 1 }) });
    assert.equal(response.status, 200);
    const body = await response.json() as { direct: { generatedText: string; fullText: string; tokensEqualBaseline: boolean; firstDivergentStep: number | null } };
    assert.deepEqual(body.direct, { generatedTokenIds: [7], fullTokenIds: [2, 7], elapsedSeconds: 1, tokensPerSecond: 1, linearThreads: 4, requests: 2, generatedText: "7", fullText: "2|7", tokensEqualBaseline: true, firstDivergentStep: null });
    const status = await fetch(`http://127.0.0.1:${address.port}/api/status`).then((entry) => entry.json()) as { ready: boolean; direct: { warmupComplete: boolean; warmupSeconds: number } };
    assert.equal(status.ready, true); assert.equal(status.direct.warmupComplete, true); assert.ok(status.direct.warmupSeconds >= 0);
  } finally { await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept())); await rm(directory, { recursive: true, force: true }); }
});

test("servidor diferencial valida opções reprodutíveis", () => {
  const options = parseGemma4RealServerOptions(["--source", "./model", "--port", "9000", "--host", "localhost"]);
  assert.equal(options.port, 9000); assert.equal(options.host, "localhost"); assert.match(options.source, /\/model$/);
  assert.throws(() => parseGemma4RealServerOptions(["--port", "0"]), /--port inválido/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "metal"]), /pytorch ou mlx/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-mlp", "always"]), /off, bf16, real ou native-bf16/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-ffn", "real"]), /off ou native-bf16/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-decoder-layer", "real"]), /off ou native-bf16/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-decoder-stack", "real"]), /off, native-bf16 ou native-bf16-ple/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-ple", "always"]), /off, bf16 ou real/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-ple-prelude", "always"]), /off, bf16 ou real/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-token-forward", "always"]), /off ou bf16/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-resident-generation", "always"]), /off ou on/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-final-head", "fp16"]), /f32, native-bf16, native-bf16-stream ou native-bf16-whole/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-native-attention", "auto"]), /off, bf16 ou real/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-attention", "always"]), /off, bf16, real ou native-bf16/);
  const directDefaults = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool"]);
  assert.equal(directDefaults.directThreads, 10); assert.equal(directDefaults.directMaxReadMiB, 16); assert.equal(directDefaults.directFinalHeadReadMiB, 16);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "pytorch", "--direct-fused-decoder-stack", "native-bf16-ple"]).directFusedDecoderStack, "native-bf16-ple");
  const splitTiles = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-max-read-mib", "16", "--direct-final-head-read-mib", "32"]);
  assert.equal(splitTiles.directFinalHeadReadMiB, 32);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-final-head-read-mib", "0"]), /Configuração direta inválida/);
  assert.equal(directDefaults.directLinearBackend, "mlx"); assert.equal(directDefaults.directFusedMlp, "real"); assert.equal(directDefaults.directFusedFfn, "off"); assert.equal(directDefaults.directFusedDecoderLayer, "off"); assert.equal(directDefaults.directFusedDecoderStack, "native-bf16"); assert.equal(directDefaults.directFusedPle, "off"); assert.equal(directDefaults.directFusedPlePrelude, "bf16"); assert.equal(directDefaults.directFusedTokenForward, "bf16"); assert.equal(directDefaults.directResidentGeneration, "on"); assert.equal(directDefaults.directFinalHead, "native-bf16-whole"); assert.equal(directDefaults.directNativeAttention, "off"); assert.equal(directDefaults.directFusedAttention, "off");
  const mlxDefaults = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "mlx"]);
  assert.equal(mlxDefaults.directFinalHeadReadMiB, 16); assert.equal(mlxDefaults.directFusedMlp, "real"); assert.equal(mlxDefaults.directFusedFfn, "off"); assert.equal(mlxDefaults.directFusedDecoderLayer, "off"); assert.equal(mlxDefaults.directFusedDecoderStack, "native-bf16"); assert.equal(mlxDefaults.directFusedPle, "off"); assert.equal(mlxDefaults.directFusedPlePrelude, "bf16"); assert.equal(mlxDefaults.directFusedTokenForward, "bf16"); assert.equal(mlxDefaults.directResidentGeneration, "on"); assert.equal(mlxDefaults.directFinalHead, "native-bf16-whole"); assert.equal(mlxDefaults.directNativeAttention, "off"); assert.equal(mlxDefaults.directFusedAttention, "off");
  const pytorchDefaults = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "pytorch"]);
  assert.equal(pytorchDefaults.directFinalHeadReadMiB, 32); assert.equal(pytorchDefaults.directFusedMlp, "native-bf16"); assert.equal(pytorchDefaults.directFusedFfn, "native-bf16"); assert.equal(pytorchDefaults.directFusedDecoderLayer, "native-bf16"); assert.equal(pytorchDefaults.directFusedDecoderStack, "native-bf16"); assert.equal(pytorchDefaults.directFusedPle, "bf16"); assert.equal(pytorchDefaults.directFusedPlePrelude, "off"); assert.equal(pytorchDefaults.directFusedTokenForward, "off"); assert.equal(pytorchDefaults.directResidentGeneration, "off"); assert.equal(pytorchDefaults.directFinalHead, "native-bf16-stream"); assert.equal(pytorchDefaults.directNativeAttention, "real"); assert.equal(pytorchDefaults.directFusedAttention, "native-bf16");
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-fused-mlp", "native-bf16"]), /requer backend pytorch/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-fused-ffn", "native-bf16"]), /requer backend pytorch/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-fused-decoder-layer", "native-bf16"]), /requer backend pytorch/);
  assert.equal(parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "mlx", "--direct-fused-decoder-stack", "native-bf16"]).directFusedDecoderStack, "native-bf16");
  const stackOff = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "mlx", "--direct-fused-decoder-stack", "off"]);
  assert.equal(stackOff.directFusedTokenForward, "off"); assert.equal(stackOff.directResidentGeneration, "off");
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-fused-decoder-stack", "off", "--direct-fused-token-forward", "bf16"]), /decoder stack habilitada/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-fused-decoder-stack", "native-bf16-ple"]), /requer backend pytorch/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-fused-attention", "native-bf16"]), /requer backend pytorch/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "pytorch", "--direct-fused-token-forward", "bf16"]), /requer backend mlx/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "mlx", "--direct-resident-generation", "on", "--direct-fused-token-forward", "off"]), /token forward MLX bf16/);
  assert.throws(() => parseGemma4RealServerOptions(["--unknown", "x"]), /Flag desconhecida/);
});

test("servidor reutiliza um worker carregado para múltiplos prompts", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-worker-test-"));
  const helper = join(directory, "worker.mjs");
  await writeFile(helper, `import readline from "node:readline";
let requests=0; console.log(JSON.stringify({ready:true}));
readline.createInterface({input:process.stdin}).on("line",line=>{const request=JSON.parse(line); requests++; console.log(JSON.stringify({id:request.id,report:{prompt:request.prompt,requests,threads:request.threads}}));});\n`);
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP de teste ausente.");
    const endpoint = `http://127.0.0.1:${address.port}/api/compare`;
    const request = (prompt: string, threads: number) => fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt, maxNewTokens: 1, threads }) }).then((response) => response.json()) as Promise<{ prompt: string; requests: number; threads: number }>;
    assert.deepEqual(await request("primeiro", 2), { prompt: "primeiro", requests: 1, threads: 2 });
    assert.deepEqual(await request("segundo", 4), { prompt: "segundo", requests: 2, threads: 4 });
    const invalid = await fetch(endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "inválido", maxNewTokens: 1, precision: "f16" }) });
    assert.equal(invalid.status, 400);
    assert.deepEqual(await invalid.json(), { error: "precision deve ser f32 ou f64." });
  } finally {
    await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept()));
    await rm(directory, { recursive: true, force: true });
  }
});
