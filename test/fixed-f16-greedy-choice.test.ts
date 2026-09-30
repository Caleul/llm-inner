import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { test } from "node:test";
import { chooseFixedF16GreedyToken, prepareFixedF16GreedyChoiceSource } from "../src/fixed-f16-greedy-choice.js";
import { compileFixedF16ScalarDimensionFromDirectory,
  compileFixedF16ScalarModelFromDirectory } from "../src/fixed-f16-ir-scalar-compiler.js";
import { compileFixedF16EmbeddingParametricFormulas, evaluateFixedF16CachedScalarSource,
  scalarSourceFromFormulas } from "../src/fixed-f16-parametric-formulas.js";
import { SafetensorsCatalogReader } from "../src/safetensors.js";

test("greedy F16 choice compares values and keeps the first index on ties", () => {
  assert.deepEqual(chooseFixedF16GreedyToken([0x3c00, 0x4000, 0x4000]),
    { tokenId: 1, logitBits: 0x4000, logitValue: 2 });
  assert.deepEqual(chooseFixedF16GreedyToken([0xbc00, 0x8000, 0x0000]),
    { tokenId: 1, logitBits: 0x8000, logitValue: 0 });
  assert.throws(() => chooseFixedF16GreedyToken([]), /sem logits/);
  const generic = prepareFixedF16GreedyChoiceSource({
    kind: "fixed-f16-cached-scalar-source", inputSize: 1, outputSize: 3,
    declarations: "", nextCacheId: 0, maxSequenceLength: 5,
    formulas: ["x[t][0]", "15360", "16384"],
  });
  assert.deepEqual(generic([[0], [0x4200]]), { tokenId: 0, logitBits: 0x4200, logitValue: 3 });
  assert.deepEqual(generic([[0]]), { tokenId: 2, logitBits: 16384, logitValue: 2 });
  assert.throws(() => generic([]), /Comprimento da sequência/);
  assert.throws(() => generic(Array.from({ length: 6 }, () => [0])), /Comprimento da sequência/);
});

test("selected token and winning logit match the PyTorch forward for varied lengths", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); }
  catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-parametric-attention.json"), "utf8")) as
    { token_cases: Record<string, { ids: number[] }>;
      random_token_cases: Record<string, { ids: number[] }> };
  const expected: Record<string, { tokenId: number; logitBits: number }> = {
    "1": { tokenId: 5971, logitBits: 13748 },
    "4": { tokenId: 5995, logitBits: 13742 },
    "8": { tokenId: 24181, logitBits: 13596 },
    "42": { tokenId: 27488, logitBits: 13579 },
    "2026": { tokenId: 18171, logitBits: 13765 },
  };
  const source = await compileFixedF16ScalarModelFromDirectory(directory);
  assert.equal(source.outputSize, 32_000);
  const choose = prepareFixedF16GreedyChoiceSource(source);
  const reader = new SafetensorsCatalogReader(directory);
  let embedding: Awaited<ReturnType<typeof compileFixedF16EmbeddingParametricFormulas>>;
  try {
    embedding = await compileFixedF16EmbeddingParametricFormulas(reader, "model.embed_tokens.weight");
  } finally { await reader.close(); }
  for (const [key, sample] of Object.entries({ ...fixture.token_cases, ...fixture.random_token_cases })) {
    const vectors: number[][] = sample.ids.map((id) =>
      evaluateFixedF16CachedScalarSource(scalarSourceFromFormulas(embedding), [[id]])[0]!);
    const result = choose(vectors);
    assert.equal(result.tokenId, expected[key]!.tokenId, `sample=${key}`);
    assert.equal(result.logitBits, expected[key]!.logitBits, `sample=${key}`);
    const selectedDimension = await compileFixedF16ScalarDimensionFromDirectory(directory, result.tokenId);
    const selectedLogit: number = evaluateFixedF16CachedScalarSource(selectedDimension, vectors).at(-1)![0]!;
    assert.equal(selectedLogit, expected[key]!.logitBits, `selected formula sample=${key}`);
  }
});
