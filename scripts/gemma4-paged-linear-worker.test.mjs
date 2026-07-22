import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("worker PyTorch preserva a redução pow(-0.5) do RMSNorm Gemma 4", async () => {
  const source = await readFile(new URL("./gemma4-paged-linear-worker.py", import.meta.url), "utf8");
  const start = source.indexOf("def rms_norm_real(tensor, weight, epsilon):");
  const end = source.indexOf("\ndef ", start + 1);
  assert.ok(start >= 0 && end > start, "função rms_norm_real ausente");
  const body = source.slice(start, end);
  assert.match(body, /torch\.pow\(mean_squared, torch\.tensor\(-0\.5, dtype=torch\.float32\)\)/);
  assert.doesNotMatch(body, /torch\.rsqrt/);
});

test("worker PyTorch compartilha as fronteiras eager BF16 entre atenção isolada e decoder stack", async () => {
  const source = await readFile(new URL("./gemma4-paged-linear-worker.py", import.meta.url), "utf8");
  const start = source.indexOf("def eager_bf16_attention(query, key, value, mask, scale):");
  const end = source.indexOf("\ndef ", start + 1);
  assert.ok(start >= 0 && end > start, "função eager_bf16_attention ausente");
  const body = source.slice(start, end);
  assert.match(body, /query = query\.to\(torch\.bfloat16\)/);
  assert.match(body, /scores = \(torch\.matmul\(query, key\.transpose\(-1, -2\)\)\.float\(\) \* torch\.tensor\(scale, dtype=torch\.float32\)\)\.to\(torch\.bfloat16\)/);
  assert.match(body, /scores = \(scores \+ mask\.to\(torch\.bfloat16\)\)\.to\(torch\.bfloat16\)/);
  assert.match(body, /torch\.softmax\(scores\.float\(\), dim=-1\)\.to\(torch\.bfloat16\)/);
  assert.match(body, /torch\.matmul\(probabilities, value\)\.to\(torch\.bfloat16\)\.float\(\)/);
  assert.equal(source.match(/eager_bf16_attention\(/g)?.length, 3, "definição, decoder stack e atenção isolada devem compartilhar a mesma função");
});
