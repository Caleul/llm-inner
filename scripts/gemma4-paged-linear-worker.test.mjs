import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("worker PyTorch preserva a redução pow(-0.5) do RMSNorm Gemma 4", async () => {
  const source = await readFile(new URL("./gemma4-paged-linear-worker.py", import.meta.url), "utf8");
  const start = source.indexOf("def rms_norm_real(tensor, weight, epsilon):");
  const end = source.indexOf("\ndef ", start + 1);
  assert.ok(start >= 0 && end > start, "função rms_norm_real ausente");
  const body = source.slice(start, end);
  assert.match(body, /tensor\.pow\(2\)\.mean\(dim=-1, keepdim=True\)/);
  assert.doesNotMatch(body, /tensor \* tensor/);
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
  assert.match(body, /scores = torch\.matmul\(query, key\.transpose\(-1, -2\)\) \* scale/);
  assert.doesNotMatch(body, /matmul\(query, key[^\n]+\.float\(\)/);
  assert.match(body, /scores = scores \+ mask\.to\(torch\.bfloat16\)/);
  assert.match(body, /torch\.softmax\(scores, dim=-1, dtype=torch\.float32\)\.to\(query\.dtype\)/);
  assert.match(body, /torch\.matmul\(probabilities, value\)\.float\(\)/);
  assert.equal(source.match(/eager_bf16_attention\(/g)?.length, 3, "definição, decoder stack e atenção isolada devem compartilhar a mesma função");
});

test("worker PyTorch constrói RoPE pela mesma frequência inversa e matmul do Transformers", async () => {
  const source = await readFile(new URL("./gemma4-paged-linear-worker.py", import.meta.url), "utf8");
  const start = source.indexOf("def rope_real(");
  const end = source.indexOf("\ndef ", start + 1);
  assert.ok(start >= 0 && end > start, "função rope_real ausente");
  const body = source.slice(start, end);
  assert.match(body, /rotated_inverse_frequency = 1\.0 \/ \(torch\.tensor\(theta, dtype=torch\.float32\) \*\* exponents\)/);
  assert.match(body, /torch\.matmul\(inverse_frequency\[None, :, None\]\.expand\(positions\.shape\[0\], -1, 1\), positions\[:, None, :\]\.float\(\)\)\.transpose\(1, 2\)/);
  assert.doesNotMatch(body, /positions\[[^\n]+\/ denominator/);
});

test("worker PyTorch preserva BSHD até RoPE e usa o repeat_kv físico do Transformers", async () => {
  const source = await readFile(new URL("./gemma4-paged-linear-worker.py", import.meta.url), "utf8");
  const start = source.indexOf("def execute_decoder_layer(");
  const end = source.indexOf("\ndef ", start + 1);
  assert.ok(start >= 0 && end > start, "função execute_decoder_layer ausente");
  const body = source.slice(start, end);
  assert.match(body, /reshape\(batch, query_sequence, query_heads, head_dim\)\n    query = rope_real\([^\n]+, True, True\)\.transpose\(1, 2\)/);
  assert.match(body, /attention_key = repeat_key_value\(key, group\)/);
  assert.match(body, /attention_value = repeat_key_value\(value, group\)/);
  const repeatStart = source.indexOf("def repeat_key_value(");
  const repeatEnd = source.indexOf("\ndef ", repeatStart + 1);
  const repeatBody = source.slice(repeatStart, repeatEnd);
  assert.match(repeatBody, /hidden_states\[:, :, None, :, :\]\.expand/);
  assert.match(repeatBody, /expanded\.reshape\(batch, key_value_heads \* groups, sequence, head_dim\)/);
  assert.doesNotMatch(body, /repeat_interleave/);
});
