import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { inlineFixedF16HiddenCall, rewriteFixedF16HiddenFile, rewriteFixedF16ContextFile,
  rewriteFixedF16ScoreFile, fixedF16ContextReplacer, fixedF16ScoreReplacer } from "../src/fixed-f16-hidden-inline.js";
import type { FixedF16CachedScalarSource } from "../src/fixed-f16-parametric-formulas.js";

test("streaming substitutes hidden calls across read boundaries without changing surrounding text", async () => {
  const dir = await mkdtemp(join(tmpdir(), "f16-hidden-stream-"));
  try {
    const input = join(dir, "input.txt"), output = join(dir, "output.txt");
    const source = `before${"x".repeat(251)}+hidden_0(t,31)+hidden_1(t,7)${"y".repeat(273)}after`;
    await writeFile(input, source);
    const result = await rewriteFixedF16HiddenFile(input, output, 7);
    const expected = source.replace(/\bhidden_(\d+)\(t,(\d+)\)/g,
      (_, layer: string, dimension: string) => inlineFixedF16HiddenCall(layer, dimension));
    assert.equal(await readFile(output, "utf8"), expected);
    assert.deepEqual(result, { replacements: 2, bytes: Buffer.byteLength(expected) });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("streaming expands context then score across file boundaries", async () => {
  const dir = await mkdtemp(join(tmpdir(), "f16-attention-stream-"));
  try {
    const source: FixedF16CachedScalarSource = {
      kind: "fixed-f16-cached-scalar-source", inputSize: 1, outputSize: 1,
      formulas: ["context_0(t,0)"], nextCacheId: 0,
      declarations: `const context_0=(p,coordinate)=>{const h=Math.floor(coordinate/1),d=coordinate%1,kvh=0; let acc=Math.fround(0); for(let j=0;j<=p;j++){acc=Math.fround(acc+score_0(p,j,h));} const value=f16Bits(acc); return value;};\n` +
        `const score_0=(p,j,h)=>{let total=Math.fround(0); for(let d=0;d<1;d++){total=Math.fround(total+Math.fround(j+h));} const value=mul16(f16Bits(total),f16Bits(1)); return value;};`,
    };
    const input = join(dir, "input"), context = join(dir, "context"), score = join(dir, "score");
    await writeFile(input, `(${source.formulas[0]})`);
    assert.equal((await rewriteFixedF16ContextFile(input, context, source, 3)).replacements, 1);
    assert.equal((await rewriteFixedF16ScoreFile(context, score, source, 3)).replacements, 1);
    const expectedContext = `(${fixedF16ContextReplacer(source)("0", "0")})`;
    const expected = expectedContext.replace("score_0(p,j,h)", fixedF16ScoreReplacer(source)("0", ""));
    assert.equal(await readFile(score, "utf8"), expected);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
