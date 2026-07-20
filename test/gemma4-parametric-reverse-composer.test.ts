import assert from "node:assert/strict";
import test from "node:test";
import type { Gemma4ParametricExactRealProgram } from "../src/gemma4-parametric-global-real-program.js";
import { evaluateGemma4ParametricOutput } from "../src/gemma4-parametric-real-evaluator.js";
import { composeGemma4ParametricReverse } from "../src/gemma4-parametric-reverse-composer.js";
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
