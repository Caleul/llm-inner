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
  assert.match(gemma4RealCompareHtml, /Compilado direto — pool binário/);
  assert.match(gemma4RealCompareHtml, /Tokens orig\. \/ compat\. \/ direto/);
  assert.match(gemma4RealCompareHtml, /linearBackend/);
  assert.match(gemma4RealCompareHtml, /lotes lineares/);
  assert.match(gemma4RealCompareHtml, /MLPs fundidos/);
  assert.match(gemma4RealCompareHtml, /PLEs fundidos/);
  assert.match(gemma4RealCompareHtml, /head:/);
  assert.match(gemma4RealCompareHtml, /attention core:/);
  assert.match(gemma4RealCompareHtml, /attention fused:/);
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
readline.createInterface({input:process.stdin}).on("line",line=>{const r=JSON.parse(line);console.log(JSON.stringify({id:r.id,report:{generatedTokenIds:[7],fullTokenIds:[2,7],elapsedSeconds:1,tokensPerSecond:1,linearThreads:4}}));});\n`);
  const server = createGemma4RealComparisonServer({ source: directory, python: process.execPath, helper: transformer, literalArtifact: "literal.json", binaryPool: directory, directWorker: direct, directLinearHelper: "unused.py", directThreads: 4 });
  await new Promise<void>((accept) => server.listen(0, "127.0.0.1", accept));
  try {
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Endereço HTTP de teste ausente.");
    const response = await fetch(`http://127.0.0.1:${address.port}/api/compare`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: "x", maxNewTokens: 1 }) });
    assert.equal(response.status, 200);
    const body = await response.json() as { direct: { generatedText: string; fullText: string; tokensEqualBaseline: boolean; firstDivergentStep: number | null } };
    assert.deepEqual(body.direct, { generatedTokenIds: [7], fullTokenIds: [2, 7], elapsedSeconds: 1, tokensPerSecond: 1, linearThreads: 4, generatedText: "7", fullText: "2|7", tokensEqualBaseline: true, firstDivergentStep: null });
  } finally { await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept())); await rm(directory, { recursive: true, force: true }); }
});

test("servidor diferencial valida opções reprodutíveis", () => {
  const options = parseGemma4RealServerOptions(["--source", "./model", "--port", "9000", "--host", "localhost"]);
  assert.equal(options.port, 9000); assert.equal(options.host, "localhost"); assert.match(options.source, /\/model$/);
  assert.throws(() => parseGemma4RealServerOptions(["--port", "0"]), /--port inválido/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-linear-backend", "metal"]), /pytorch ou mlx/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-mlp", "always"]), /off, bf16 ou real/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-ple", "always"]), /off, bf16 ou real/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-final-head", "fp16"]), /f32, native-bf16 ou native-bf16-whole/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-native-attention", "auto"]), /off, bf16 ou real/);
  assert.throws(() => parseGemma4RealServerOptions(["--direct-fused-attention", "always"]), /off, bf16 ou real/);
  const directDefaults = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool"]);
  assert.equal(directDefaults.directFusedMlp, "real"); assert.equal(directDefaults.directFusedPle, "bf16"); assert.equal(directDefaults.directFinalHead, "native-bf16"); assert.equal(directDefaults.directNativeAttention, "real"); assert.equal(directDefaults.directFusedAttention, "bf16");
  const mlxDefaults = parseGemma4RealServerOptions(["--literal-artifact", "literal.json", "--binary-pool", "pool", "--direct-linear-backend", "mlx"]);
  assert.equal(mlxDefaults.directFusedPle, "off"); assert.equal(mlxDefaults.directFinalHead, "f32"); assert.equal(mlxDefaults.directNativeAttention, "off"); assert.equal(mlxDefaults.directFusedAttention, "off");
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
