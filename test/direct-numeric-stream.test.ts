import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { lowerDirectNumericExpressionFile, rewriteDirectNumericFile, rewriteDirectRopeFile } from "../src/direct-numeric-stream.js";
import { fixedF16RopeLiteral } from "../src/fixed-f16-rope-branches.js";
import { sleefExpF32 } from "../src/sleef-f32.js";

test("independent streaming numeric passes leave executable direct expressions", async () => {
  const directory = await mkdtemp(join(tmpdir(), "direct-numeric-"));
  try {
    let input = join(directory, "input");
    await writeFile(input, "f16(15360)+ropeBits(p,d,4,10000,0)+Math.exp(-1)+Math.sqrt(4)+f16Bits(0.5)");
    let index = 0;
    for (const operation of ["exp", "sqrt", "rope", "round", "decode"] as const) {
      const output = join(directory, `pass-${index++}`);
      if (operation === "rope") await rewriteDirectRopeFile(input, output, 8, 7);
      else await rewriteDirectNumericFile(input, output, operation, 7);
      input = output;
    }
    const source = await readFile(input, "utf8");
    assert.doesNotMatch(source, /\b(?:f16|f16Bits|ropeBits|Math\.exp|Math\.sqrt|Math\.fround)\(/);
    const evaluate = new Function("p", "d", `return ${source};`) as (position: number, dimension: number) => number;
    assert.equal(evaluate(2, 1), 1 + fixedF16RopeLiteral(2, 1, 4, 10000, 0) +
      sleefExpF32(-1) + 2 + 14336);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("numeric pipeline emits audited executable source from an independent expression file", async () => {
  const directory = await mkdtemp(join(tmpdir(), "direct-pipeline-"));
  try {
    const input = join(directory, "input"), output = join(directory, "output");
    await writeFile(input, "f16(x)+ropeBits(p,d,4,10000,0)+Math.exp(y)+Math.sqrt(z)+f16Bits(w)");
    const result = await lowerDirectNumericExpressionFile(input, output, 8);
    assert.deepEqual(result.unresolved, {});
    assert.deepEqual(result.passes.map((pass) => pass.name),
      ["fold", "exp", "sqrt", "rope", "round", "decode"]);
    const source = await readFile(output, "utf8");
    const evaluate = new Function("x", "p", "d", "y", "z", "w", `return ${source};`) as
      (bits: number, position: number, dimension: number, exponent: number, square: number, value: number) => number;
    assert.equal(evaluate(15360, 2, 1, -1, 4, 0.5),
      1 + fixedF16RopeLiteral(2, 1, 4, 10000, 0) + sleefExpF32(-1) + 2 + 14336);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("literal folding runs before branch expansion and preserves signed zero", async () => {
  const directory = await mkdtemp(join(tmpdir(), "direct-fold-"));
  try {
    const input = join(directory, "input"), output = join(directory, "output");
    await writeFile(input, "f16(32768)+f16Bits(-0)+Math.fround(0.5)+Math.exp(-1)+Math.sqrt(4)+f16Bits(x)");
    const result = await rewriteDirectNumericFile(input, output, "fold", 2);
    assert.equal(result.replacements, 5);
    assert.equal(await readFile(output, "utf8"),
      `-0+32768+0.5+${sleefExpF32(-1)}+2+f16Bits(x)`);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("literal RoPE dimension is simplified before position branching", async () => {
  const directory = await mkdtemp(join(tmpdir(), "direct-rope-dimension-"));
  try {
    const input = join(directory, "input"), output = join(directory, "output");
    await writeFile(input, "ropeBits(p,1,4,10000,0)+ropeBits(p,2,4,10000,1)");
    assert.equal((await rewriteDirectRopeFile(input, output, 8, 3)).replacements, 2);
    const source = await readFile(output, "utf8");
    assert.doesNotMatch(source, /\bdim\b/);
    const evaluate = new Function("p", `return ${source};`) as (position: number) => number;
    for (let position = 0; position < 8; position++) {
      assert.equal(evaluate(position), fixedF16RopeLiteral(position, 1, 4, 10000, 0) +
        fixedF16RopeLiteral(position, 2, 4, 10000, 1));
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
