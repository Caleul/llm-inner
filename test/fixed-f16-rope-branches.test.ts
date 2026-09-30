import assert from "node:assert/strict";
import { test } from "node:test";
import { compileFixedF16RopeBranches, fixedF16RopeLiteral } from "../src/fixed-f16-rope-branches.js";

test("RoPE branches cover every position and dimension in the compiled context", () => {
  for (const sine of [0, 1]) {
    const body = compileFixedF16RopeBranches(2048, 4, 10_000, sine);
    const evaluate = new Function("pos", "dim", body) as (position: number, dimension: number) => number;
    for (let position = 0; position < 2048; position++) {
      for (let dimension = 0; dimension < 4; dimension++) {
        assert.equal(evaluate(position, dimension),
          fixedF16RopeLiteral(position, dimension, 4, 10_000, sine));
      }
    }
    assert.throws(() => evaluate(2048, 0), /fora do contexto/);
    assert.throws(() => evaluate(0, 4), /Dimensão RoPE inválida/);
  }
});
