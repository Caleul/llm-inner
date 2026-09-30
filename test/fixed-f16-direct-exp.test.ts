import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { directF32ExpPrefix } from "../src/fixed-f16-direct-exp.js";
import { sleefExpF32 } from "../src/sleef-f32.js";
import { rewriteDirectNumericFile } from "../src/direct-numeric-stream.js";

test("inlined exponential matches pinned CPU F32 exponential", () => {
  const source = `${directF32ExpPrefix()}x)`;
  assert.doesNotMatch(source, /\bMath\.exp\(|\bexp\(/);
  const evaluate = new Function("x", `return ${source};`) as (value: number) => number;
  const values = [-104, -103.99999, -87, -80, -10, -1, -0, 0, 1, 20, 100];
  let state = 0x1278abce;
  for (let index = 0; index < 200000; index++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values.push(-104 + (state / 0xffffffff) * 104);
  }
  for (const value of values) {
    assert.ok(Object.is(evaluate(value), sleefExpF32(value)), `exp(${value})`);
  }
});

test("streaming replaces exp and all internal rounding calls", async () => {
  const directory = await mkdtemp(join(tmpdir(), "direct-exp-"));
  try {
    const input = join(directory, "input"), expanded = join(directory, "expanded"), output = join(directory, "output");
    await writeFile(input, "Math.exp(x)+Math.exp(y)");
    assert.equal((await rewriteDirectNumericFile(input, expanded, "exp", 3)).replacements, 2);
    await rewriteDirectNumericFile(expanded, output, "round", 7);
    const source = await readFile(output, "utf8");
    assert.doesNotMatch(source, /\b(?:Math\.exp|Math\.fround|f16Bits)\(/);
    const evaluate = new Function("x", "y", `return ${source};`) as (x: number, y: number) => number;
    for (const [x, y] of [[-1, -2], [-10, 0], [-80, -0.5]] as const) {
      assert.equal(evaluate(x, y), sleefExpF32(x) + sleefExpF32(y));
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
