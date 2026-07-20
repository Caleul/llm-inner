import assert from "node:assert/strict";
import test from "node:test";
import type { Gemma4ParametricExactRealProgram } from "../src/gemma4-parametric-global-real-program.js";
import { evaluateGemma4ParametricOutput } from "../src/gemma4-parametric-real-evaluator.js";
import { closeGemma4ParametricReverseTextInputs, composeGemma4ParametricReverse } from "../src/gemma4-parametric-reverse-composer.js";
import { Gemma4ParametricRealBuilder } from "../src/gemma4-parametric-real-expression.js";

test("expande operações paramétricas da saída para a entrada preservando o valor", async () => {
  const builder = new Gemma4ParametricRealBuilder(), previousParameter = builder.integerParameter("previous:i"), finalParameter = builder.integerParameter("final:i"), zero = builder.integerConstant(0);
  const input = builder.inputElement("x", [previousParameter]);
  const previousRoot = builder.multiply(builder.rational(2n), input);
  const previousCall = builder.functionCall("operation:previous", [finalParameter]);
  const finalRoot = builder.multiply(builder.rational(-1n, 4n), previousCall);
  const outputRoot = builder.functionCall("operation:final", [zero]);
  const program: Gemma4ParametricExactRealProgram = {
    kind: "gemma4-parametric-exact-real-simplified-program", schemaVersion: 1, semantics: "gemma4-exact-real-simplified-v1", inputBoundaries: ["x"], expressionGraph: builder.build(),
    operationFunctions: [
      { functionId: "operation:previous", operationId: "previous", ordinal: 0, output: "previous", parameters: [{ name: "i", node: previousParameter }], root: previousRoot, predecessorFunctions: [], closureKind: "global-output-closure", operandBoundaries: ["x"] },
      { functionId: "operation:final", operationId: "final", ordinal: 1, output: "final", parameters: [{ name: "i", node: finalParameter }], root: finalRoot, predecessorFunctions: ["operation:previous"], closureKind: "global-output-closure", operandBoundaries: [] },
    ],
    outputFunctions: [{ name: "y", operationId: "final", fixedDimension: 0, coordinate: [zero], parameters: [], root: outputRoot, finalQuantization: "none" }],
    coverage: { sourceAssignments: 2, compiledOperationTemplates: 2, learnedRationalTableReads: 0, runtimeDefinedReductionsLowered: 0, unresolvedRuntimeReductions: 0, intermediateIeeeRoundingNodes: 0 },
  };
  const composed = composeGemma4ParametricReverse(program, program.outputFunctions[0]!, {}, 2);
  assert.equal(composed.steps.length, 2); assert.deepEqual(composed.remainingFunctionCalls, []);
  const executable = { ...program, expressionGraph: composed.graph, operationFunctions: [], outputFunctions: [{ ...program.outputFunctions[0]!, root: composed.root }] };
  const value = await evaluateGemma4ParametricOutput(executable, "y", 0, { parameters: {}, inputs: { axis: () => 1, element: () => 3 }, learned: { element: () => { throw new Error("peso inesperado"); } } });
  assert.equal(value, -1.5);
});

test("não cancela x/x sem prova de não-zero", () => {
  const builder = new Gemma4ParametricRealBuilder(), zero = builder.integerConstant(0), x = builder.inputElement("x", [zero]);
  assert.equal(builder.requiredNode(builder.divide(x, x)).kind, "divide");
});

test("fecha boundaries textuais em um único vetor x e dobra posição e máscara", () => {
  const builder = new Gemma4ParametricRealBuilder(), zero = builder.integerConstant(0), one = builder.integerConstant(1);
  const hidden = builder.inputElement("hidden_states_0", [zero, one, zero]);
  const ple = builder.inputElement("ple_inputs", [zero, one, one, zero]);
  const position = builder.inputElement("position_ids", [zero, one], "integer");
  const mask = builder.inputElement("full_attention_mask", [zero, zero, one, zero]);
  const root = builder.add(hidden, ple, position, mask);
  const closed = closeGemma4ParametricReverseTextInputs({
    kind: "gemma4-parametric-reverse-algebraic-composition", schemaVersion: 1,
    output: { family: "y", dimension: 0, parameters: {} }, graph: builder.build(), root,
    steps: [], remainingFunctionCalls: [],
  }, { sequenceLength: 2, hiddenSize: 3, layers: 2, pleFeatures: 2 });
  assert.equal(closed.inputVector.length, 14);
  assert.deepEqual(closed.inputVector.foldedInputs, ["position_ids", "full_attention_mask", "sliding_attention_mask"]);
  const inputs = closed.graph.nodes.filter((node) => node.kind === "input-element");
  assert.equal(inputs.length, 2);
  assert.ok(inputs.every((node) => node.tensor === "x"));
  assert.deepEqual(inputs.map((node) => (closed.graph.nodes.find((candidate) => candidate.id === node.coordinates[0]) as { value: number }).value).sort((left, right) => left - right), [3, 12]);
});
