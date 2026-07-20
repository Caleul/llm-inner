import assert from "node:assert/strict";
import test from "node:test";
import {
  Gemma4RealExpressionBuilder,
  gemma4ExactRationalFromIeeeBits,
  validateGemma4RealExpressionGraph,
} from "../src/gemma4-real-expression.js";
import { simplifyGemma4RealExpressionGraph } from "../src/gemma4-real-expression-simplifier.js";
import { evaluateGemma4RealExpressionFastF64 } from "../src/gemma4-real-expression-evaluator.js";

test("decodifica BF16, F32 e F64 finitos como racionais exatos", () => {
  assert.deepEqual(gemma4ExactRationalFromIeeeBits("BF16", "0x3e80"), { numerator: "1", denominator: "4" });
  assert.deepEqual(gemma4ExactRationalFromIeeeBits("F32", "0xbf900000"), { numerator: "-9", denominator: "8" });
  assert.deepEqual(gemma4ExactRationalFromIeeeBits("F64", "0x4012000000000000"), { numerator: "9", denominator: "2" });
  assert.deepEqual(gemma4ExactRationalFromIeeeBits("F16", "0x3c00"), { numerator: "1", denominator: "1" });
  assert.deepEqual(gemma4ExactRationalFromIeeeBits("F32", "0x00000001"), { numerator: "1", denominator: (1n << 149n).toString() });
});

test("normaliza constantes, termos semelhantes e fatores idênticos", () => {
  const builder = new Gemma4RealExpressionBuilder();
  const x = builder.input("embedding", ["0", "0"]);
  const first = builder.multiply(builder.rational(-1n, 4n), x);
  const second = builder.multiply(builder.rational(-9n, 8n), x);
  const combined = builder.add(first, second);
  const expected = builder.multiply(builder.rational(-11n, 8n), x);
  assert.equal(combined, expected);
  assert.equal(builder.multiply(x, x), builder.integerPower(x, 2));
  validateGemma4RealExpressionGraph(builder.build());
});

test("remove casts conceitualmente ao construir uma expressão real fatorada", () => {
  const builder = new Gemma4RealExpressionBuilder();
  const x0 = builder.input("embedding", ["0", "0"]);
  const x1 = builder.input("embedding", ["0", "1"]);
  const result = builder.multiply(
    builder.rational(1n, 4n),
    builder.add(builder.negate(x0), builder.multiply(builder.rational(-9n, 2n), x1)),
  );
  const expanded = builder.add(
    builder.multiply(builder.rational(-1n, 4n), x0),
    builder.multiply(builder.rational(-9n, 8n), x1),
  );
  // A expansão e a fatoração são ambas construíveis sobre a mesma semântica
  // real. O passe de fatoração escolhe a forma de menor custo posteriormente.
  assert.notEqual(result, expanded);
  const simplified = simplifyGemma4RealExpressionGraph(builder.build(), { y0: expanded });
  validateGemma4RealExpressionGraph(simplified.graph);
  assert.ok(simplified.statistics.outputNodes > 0);
});

test("rejeita valores IEEE não finitos", () => {
  assert.throws(() => gemma4ExactRationalFromIeeeBits("F32", "0x7f800000"), /não é finito/);
});

test("avalia em F64 sem casts intermediários e arredonda apenas o resultado", () => {
  const builder = new Gemma4RealExpressionBuilder();
  const x = builder.input("x");
  const root = builder.add(builder.multiply(builder.rational(1n, 3n), x), builder.rational(1n, 10n));
  const result = evaluateGemma4RealExpressionFastF64(builder.build(), root, { x: 3 });
  assert.equal(result.realApproximation, 1.1);
  assert.equal(result.finalF32, Math.fround(1.1));
  assert.match(result.finalF32BitsHex, /^0x[0-9a-f]{8}$/);
  assert.match(result.finalBF16BitsHex, /^0x[0-9a-f]{4}$/);
});
