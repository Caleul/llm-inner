import assert from "node:assert/strict";
import test from "node:test";
import { Gemma4RealExpressionBuilder, validateGemma4RealExpressionGraph } from "../src/gemma4-real-expression.js";
import { lowerGemma4ScalarExpressionToExactReal } from "../src/gemma4-real-statement-lowering.js";
import { buildGemma4LiteralScalarStatementPrograms } from "../src/gemma4-literal-scalar-statement-programs.js";

test("lowering real remove casts e transforma BMM runtime em soma matemática", () => {
  const formula = "y[batch,output_feature] = BF16(F32(REDUCE(input_feature=0..3,exact_product(x[batch,input_feature]*decode(weight)[output_feature,input_feature]))+decode(bias)[output_feature]))";
  const program = buildGemma4LiteralScalarStatementPrograms([formula], "y", "real-lowering")[0]!;
  const builder = new Gemma4RealExpressionBuilder();
  const weights = [1n, 2n, 3n, 4n].map((value) => builder.rational(value, 4n));
  const root = lowerGemma4ScalarExpressionToExactReal(program.expression, {
    builder,
    integerBindings: { batch: 0, output_feature: 0 },
    resolveTensorElement: (name, coordinates) => builder.input(name, coordinates.map(String)),
    resolveLearnedElement: (role, coordinates) => role === "weight" ? weights[coordinates[1]!]! : builder.rational(1n, 8n),
  });
  const graph = builder.build();
  validateGemma4RealExpressionGraph(graph);
  const rootNode = builder.requiredNode(root);
  assert.equal(rootNode.kind, "add");
  assert.ok(graph.nodes.every((node) => !JSON.stringify(node).match(/BF16|F32|runtime|unpublished/)));
  assert.equal(graph.nodes.filter((node) => node.kind === "input").length, 4);
});

test("lowering real converte SiLU para divisão e exp matemáticos", () => {
  const formula = "y[batch,hidden] = F32(x[batch,hidden]/F32(1+SLEEF_EXP_F32(F32(-x[batch,hidden]))))";
  const program = buildGemma4LiteralScalarStatementPrograms([formula], "y", "real-silu")[0]!;
  const builder = new Gemma4RealExpressionBuilder();
  const root = lowerGemma4ScalarExpressionToExactReal(program.expression, {
    builder,
    integerBindings: { batch: 0, hidden: 3 },
    resolveTensorElement: (name, coordinates) => builder.input(name, coordinates.map(String)),
    resolveLearnedElement: () => { throw new Error("SiLU não possui peso."); },
  });
  assert.equal(builder.requiredNode(root).kind, "divide");
  assert.ok(builder.build().nodes.some((node) => node.kind === "unary-function" && node.function === "exp"));
});
