import assert from "node:assert/strict";
import test from "node:test";
import { Gemma4RealExpressionBuilder } from "../src/gemma4-real-expression.js";
import { evaluateGemma4RealExpressionFastF64 } from "../src/gemma4-real-expression-evaluator.js";
import { composeGemma4ReverseAlgebra } from "../src/gemma4-reverse-algebraic-composer.js";

test("compõe camadas de trás para frente e simplifica coeficientes entre substituições", () => {
  const output = new Gemma4RealExpressionBuilder(), previous = output.input("previous");
  const outputRoot = output.multiply(output.rational(-1n, 4n), previous);
  const predecessor = new Gemma4RealExpressionBuilder(), x = predecessor.input("x");
  const predecessorRoot = predecessor.multiply(predecessor.rational(2n), x);
  const result = composeGemma4ReverseAlgebra(output.build(), outputRoot, [{ id: "previous-layer", target: { name: "previous" }, replacementGraph: predecessor.build(), replacementRoot: predecessorRoot }]);
  assert.deepEqual(result.freeInputs, [{ name: "x", coordinates: [] }]);
  assert.equal(result.steps.length, 1); assert.equal(result.steps[0]!.substitutedOccurrences, 1);
  assert.equal(evaluateGemma4RealExpressionFastF64(result.graph, result.root, { x: 3 }).realApproximation, -1.5);
  const root = result.graph.nodes.find((node) => node.id === result.root)!;
  assert.equal(root.kind, "multiply");
  if (root.kind === "multiply") {
    const coefficient = result.graph.nodes.find((node) => node.id === root.arguments[0]);
    assert.deepEqual(coefficient?.kind === "rational" ? coefficient.value : undefined, { numerator: "-1", denominator: "2" });
  }
});

test("aplica identidades trigonométricas exatas e paridade transcendente", () => {
  const builder = new Gemma4RealExpressionBuilder(), x = builder.input("x");
  const sin = builder.unaryFunction("sin", x), cos = builder.unaryFunction("cos", x);
  assert.equal(builder.add(builder.integerPower(sin, 2), builder.integerPower(cos, 2)), builder.rational(1n));
  const tangent = builder.divide(sin, cos), tangentNode = builder.requiredNode(tangent);
  assert.equal(tangentNode.kind, "unary-function");
  assert.equal(tangentNode.kind === "unary-function" ? tangentNode.function : undefined, "tan");
  const negative = builder.negate(x);
  assert.equal(builder.unaryFunction("cos", negative), cos);
  assert.equal(builder.unaryFunction("abs", negative), builder.unaryFunction("abs", x));
  const oddTanh = builder.unaryFunction("tanh", negative);
  const expected = builder.negate(builder.unaryFunction("tanh", x));
  assert.equal(oddTanh, expected);
  const value = evaluateGemma4RealExpressionFastF64(builder.build(), tangent, { x: 0.3 }).realApproximation;
  assert.ok(Math.abs(value - Math.tan(0.3)) < 1e-15);
});

test("preserva domínios usando abs em sqrt de quadrado", () => {
  const builder = new Gemma4RealExpressionBuilder(), x = builder.input("x");
  const root = builder.unaryFunction("sqrt", builder.integerPower(x, 2));
  const node = builder.requiredNode(root);
  assert.equal(node.kind, "unary-function");
  assert.equal(node.kind === "unary-function" ? node.function : undefined, "abs");
  assert.equal(evaluateGemma4RealExpressionFastF64(builder.build(), root, { x: -3 }).realApproximation, 3);
});
