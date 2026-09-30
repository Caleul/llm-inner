import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { directSqrtPrefix } from "../src/fixed-f16-direct-sqrt.js";
import { rewriteDirectNumericFile } from "../src/direct-numeric-stream.js";

test("direct square root agrees with F32 RMS normalization results", () => {
  const source = `${directSqrtPrefix()}x)`;
  assert.doesNotMatch(source, /\bMath\.sqrt\(/);
  const evaluate = new Function("x", `return ${source};`) as (value: number) => number;
  const values = [0, -0, 2 ** -149, 2 ** -126, 1e-6, 1, 2, 4, 65504, 3.4028234663852886e38];
  let state = 0x5194fae7;
  for (let index = 0; index < 200000; index++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values.push(Math.fround((state / 0xffffffff) * 10000 + 1e-6));
  }
  for (const value of values) {
    const expected = Math.fround(1 / Math.fround(Math.sqrt(value)));
    const actual = Math.fround(1 / evaluate(value));
    assert.ok(Object.is(actual, expected), `rsqrt(${value}): ${actual} !== ${expected}`);
  }
  const f32 = new DataView(new ArrayBuffer(4));
  f32.setFloat32(0, Math.fround(1 / evaluate(2540.686767578125)), true);
  assert.equal(f32.getInt32(0, true), 1017284057);
});

test("streaming replaces square root without a model program", async () => {
  const directory = await mkdtemp(join(tmpdir(), "direct-sqrt-"));
  try {
    const input = join(directory, "input"), output = join(directory, "output");
    await writeFile(input, "Math.sqrt(x)+Math.sqrt(y)");
    assert.equal((await rewriteDirectNumericFile(input, output, "sqrt", 3)).replacements, 2);
    const source = await readFile(output, "utf8");
    assert.doesNotMatch(source, /\bMath\.sqrt\(/);
    const evaluate = new Function("x", "y", `return ${source};`) as (x: number, y: number) => number;
    assert.equal(Math.fround(evaluate(4, 9)), 5);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
