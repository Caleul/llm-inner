import assert from "node:assert/strict";
import test from "node:test";
import { Gemma4ParametricRealBuilder, validateGemma4ParametricRealExpressionGraph } from "../src/gemma4-parametric-real-expression.js";

test("representa um dot product de dimensão simbólica sem expandir seus termos", () => {
  const builder = new Gemma4ParametricRealBuilder();
  const hidden = builder.boundIndex("hidden");
  const end = builder.integerConstant(2560);
  const input = builder.inputElement("hidden_states", [hidden]);
  const weight = builder.learnedElement("lm_head.weight", "BF16", [builder.integerConstant(0), hidden], "decode:lm_head.weight");
  const root = builder.finiteReduction("finite-sum", "hidden", builder.integerConstant(0), end, builder.multiply(input, weight));
  const graph = builder.build();
  validateGemma4ParametricRealExpressionGraph(graph);
  assert.equal(builder.requiredNode(root).kind, "finite-sum");
  assert.ok(graph.nodes.length < 10);
});
