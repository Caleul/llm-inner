import assert from "node:assert/strict";
import { test } from "node:test";
import { compileF16BitDecodeBranches } from "../src/fixed-f16-bit-decode-branches.js";
import { f16BitsToDyadic } from "../src/fixed-f16-projection.js";

test("finite F16 decode has affine branches exactly equal to the reference", () => {
  const branch = new Function("bits", compileF16BitDecodeBranches()) as (bits: number) => number;
  let checked = 0;
  for (let bits = 0; bits <= 0xffff; bits++) {
    if ((bits & 0x7c00) === 0x7c00) {
      assert.throws(() => branch(bits), /F16 não finito/);
      continue;
    }
    const dyadic = f16BitsToDyadic(bits);
    assert.equal(branch(bits), Number(dyadic.coefficient) * 2 ** dyadic.exponent, `bits=${bits}`);
    checked++;
  }
  assert.equal(checked, 63_488);
});
