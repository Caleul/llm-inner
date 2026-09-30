import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { directRoundPrefix, lowerDirectRoundingInSource } from "../src/fixed-f16-direct-rounding.js";
import { rewriteFixedF16DirectRoundingFile } from "../src/fixed-f16-hidden-inline.js";
import { f32BitsToDyadic, roundDyadicToF16IfElse } from "../src/fixed-f16-projection.js";
import { compileFixedF16ScalarDimensionFromDirectory } from "../src/fixed-f16-ir-scalar-compiler.js";
import { prepareFixedF16CachedScalarDimensionSource } from "../src/fixed-f16-parametric-formulas.js";

function make(kind: "f16Bits" | "Math.fround"): (value: number) => number {
  return new Function("x", `return ${directRoundPrefix(kind)}x);`) as (value: number) => number;
}

test("direct F32 rounding agrees with Math.fround at boundaries and sampled doubles", () => {
  const round = make("Math.fround");
  const values = [-0, 0, 2 ** -150, 2 ** -149, 2 ** -126,
    1 + 2 ** -24, 1 + 3 * 2 ** -24, 3.4028234663852886e38, 3.4028235677973366e38];
  let state = 0x12345678;
  for (let index = 0; index < 100000; index++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values.push((state / 0xffffffff - 0.5) * 2 ** ((index % 280) - 140));
  }
  for (const value of values) assert.ok(Object.is(round(value), Math.fround(value)), `F32 ${value}`);
});

test("direct F16 bits agree with the declared finite F32 to F16 rule", () => {
  const round = make("f16Bits");
  const buffer = new DataView(new ArrayBuffer(4));
  const values = [-0, 0, 2 ** -25, 2 ** -24, 2 ** -14, 1 + 2 ** -11, 65504, 65520];
  let state = 0xabcdef01;
  for (let index = 0; index < 100000; index++) {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    buffer.setUint32(0, state, true);
    const value = buffer.getFloat32(0, true);
    if (Number.isFinite(value)) values.push(value);
  }
  for (const value of values) {
    buffer.setFloat32(0, value, true);
    const expected = Object.is(value, -0) ? 32768 : roundDyadicToF16IfElse(f32BitsToDyadic(buffer.getUint32(0, true)));
    assert.equal(round(value), expected, `F16 ${value}`);
  }
});

test("streaming substitutes input-dependent rounding calls across chunks", async () => {
  const directory = await mkdtemp(join(tmpdir(), "direct-rounding-"));
  try {
    const input = join(directory, "input"), output = join(directory, "output");
    await writeFile(input, "Math.fround(x)+f16Bits(y)+Math.fround(-0)");
    const result = await rewriteFixedF16DirectRoundingFile(input, output, 3);
    const source = await readFile(output, "utf8");
    assert.equal(result.replacements, 3);
    assert.doesNotMatch(source, /\b(?:Math\.fround|f16Bits)\(/);
    const evaluate = new Function("x", "y", `return ${source};`) as (x: number, y: number) => number;
    assert.equal(evaluate(1.25, 0.5), Math.fround(1.25) + 14336);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("in-place rounding preserves a complete diagnostic scalar forward", async () => {
  const program = await compileFixedF16ScalarDimensionFromDirectory("artifacts/tiny-random-llama", 3);
  const lowered = { ...program, declarations: lowerDirectRoundingInSource(program.declarations),
    formulas: program.formulas.map(lowerDirectRoundingInSource) };
  const baseline = prepareFixedF16CachedScalarDimensionSource(program, 0);
  const direct = prepareFixedF16CachedScalarDimensionSource(lowered, 0);
  const input = Array.from({ length: 4 }, (_, position) =>
    Array.from({ length: program.inputSize }, (_, dimension) =>
      ((position * 17 + dimension * 31) % 23) * 32 + 0x3000));
  assert.deepEqual(direct(input), baseline(input));
});
