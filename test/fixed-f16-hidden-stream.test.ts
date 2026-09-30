import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { inlineFixedF16HiddenCall, rewriteFixedF16HiddenFile, rewriteFixedF16ContextFile,
  rewriteFixedF16ScoreFile, fixedF16ContextReplacer, fixedF16ScoreReplacer } from "../src/fixed-f16-hidden-inline.js";
import { foldFixedF16ConstantCall, rewriteFixedF16NumericConstantsFile } from "../src/fixed-f16-hidden-inline.js";
import { rewriteFixedF16DecodeFile } from "../src/fixed-f16-hidden-inline.js";
import { fixedF16BitArithmeticPrefix, rewriteFixedF16BitArithmeticFile } from "../src/fixed-f16-hidden-inline.js";
import { rewriteFixedF16RopeFile } from "../src/fixed-f16-hidden-inline.js";
import type { FixedF16CachedScalarSource } from "../src/fixed-f16-parametric-formulas.js";
import { auditFixedF16StreamFile } from "../src/fixed-f16-stream-diagnostic.js";

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

test("streaming audit counts unresolved numeric calls across chunks", async () => {
  const dir = await mkdtemp(join(tmpdir(), "f16-audit-"));
  try {
    const path = join(dir, "formula");
    await writeFile(path, `${"x".repeat(65530)} Math.exp(0)+f16Bits(1)+Math.exp(2)+q_0(t,0)`);
    assert.deepEqual(await auditFixedF16StreamFile(path),
      { "Math.exp": 2, f16Bits: 1, "q_*": 1 });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("streaming folds literal F32/F16 boundaries and keeps signed zero behavior", async () => {
  const dir = await mkdtemp(join(tmpdir(), "f16-constants-"));
  try {
    const input = join(dir, "input"), output = join(dir, "output");
    await writeFile(input, "Math.fround(-0)+f16Bits(0.5)+Math.fround(1e-10)+f16(15360)+neg16(0)+f16Bits(x)");
    const result = await rewriteFixedF16NumericConstantsFile(input, output, 2);
    assert.equal(result.replacements, 5);
    assert.equal(await readFile(output, "utf8"),
      `-0+${foldFixedF16ConstantCall("f16Bits", "0.5")}+${Math.fround(1e-10)}+1+32768+f16Bits(x)`);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("streaming replaces generic F16 decoding with affine guards", async () => {
  const dir = await mkdtemp(join(tmpdir(), "f16-decode-"));
  try {
    const input = join(dir, "input"), output = join(dir, "output");
    await writeFile(input, "f16(0)+f16(32768)+f16Bits(0)");
    const result = await rewriteFixedF16DecodeFile(input, output, 2);
    const expanded = await readFile(output, "utf8");
    assert.equal(result.replacements, 2);
    assert.doesNotMatch(expanded, /\bf16\(/);
    assert.equal(new Function("f16Bits", `return ${expanded}`)((value: number) => value), 0 + 0 + 0);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("streaming embeds F16 add, multiply and sign branches", async () => {
  const dir = await mkdtemp(join(tmpdir(), "f16-arithmetic-"));
  try {
    const input = join(dir, "input"), output = join(dir, "output");
    await writeFile(input, "add16(1,2)+mul16(3,4)+neg16(5)");
    const result = await rewriteFixedF16BitArithmeticFile(input, output, 2);
    assert.equal(result.replacements, 3);
    assert.equal(await readFile(output, "utf8"),
      `${fixedF16BitArithmeticPrefix("add16")}1,2)+` +
      `${fixedF16BitArithmeticPrefix("mul16")}3,4)+` +
      `${fixedF16BitArithmeticPrefix("neg16")}5)`);
    const negate = new Function("bits", `return ${fixedF16BitArithmeticPrefix("neg16")}bits)`) as
      (bits: number) => number;
    for (let bits = 0; bits <= 0xffff; bits++) assert.equal(negate(bits), bits ^ 0x8000);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("streaming lowers RoPE calls for the configured context", async () => {
  const dir = await mkdtemp(join(tmpdir(), "f16-rope-stream-"));
  try {
    const input = join(dir, "input"), output = join(dir, "output");
    await writeFile(input, "ropeBits(p,d,4,10000,0)+ropeBits(j,d,4,10000,1)");
    const result = await rewriteFixedF16RopeFile(input, output, 8, 3);
    const expanded = await readFile(output, "utf8");
    assert.equal(result.replacements, 2);
    assert.doesNotMatch(expanded, /\bropeBits\(/);
    assert.match(expanded, /pos>=8/);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
