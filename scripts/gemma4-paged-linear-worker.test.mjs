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
