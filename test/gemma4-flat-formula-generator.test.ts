import assert from "node:assert/strict";
import test from "node:test";
import type { Gemma4ParametricExactRealProgram } from "../src/gemma4-parametric-global-real-program.js";
import { generateGemma4FlatFormulaObject, Gemma4FlatFormulaLimitError } from "../src/gemma4-flat-formula-generator.js";

test("substitui funções intermediárias e retorna somente calc_final_n e x[i]", async () => {
  const nodes = [
    { id: "p", kind: "integer-parameter" as const, name: "i" },
    { id: "zero", kind: "integer-constant" as const, value: 0 }, { id: "one", kind: "integer-constant" as const, value: 1 },
    { id: "x0", kind: "input-element" as const, tensor: "embedding", coordinates: ["zero"], valueType: "real" as const },
    { id: "x1", kind: "input-element" as const, tensor: "embedding", coordinates: ["one"], valueType: "real" as const },
    { id: "quarter", kind: "rational" as const, value: { numerator: "-1", denominator: "4" } },
    { id: "weight", kind: "learned-rational-element" as const, tensor: "layer.weight", storageDtype: "BF16" as const, coordinates: ["zero"], decoderId: "decode:layer.weight" },
    { id: "left", kind: "multiply" as const, arguments: ["quarter", "x0"] },
    { id: "right", kind: "multiply" as const, arguments: ["weight", "x1"] },
    { id: "body", kind: "add" as const, arguments: ["left", "right"] },
    { id: "call", kind: "function-call" as const, functionId: "operation:layer", arguments: ["zero"] },
  ];
  const program = {
    kind: "gemma4-parametric-exact-real-simplified-program", schemaVersion: 1, semantics: "gemma4-exact-real-simplified-v1", inputBoundaries: ["embedding"],
    expressionGraph: { kind: "gemma4-parametric-exact-real-expression-graph", schemaVersion: 1, semantics: "gemma4-exact-real-simplified-v1", nodeOrder: "dependency-order", nodes },
    operationFunctions: [{ functionId: "operation:layer", operationId: "layer", ordinal: 0, output: "y", parameters: [{ name: "i", node: "p" }], root: "body", predecessorFunctions: [], closureKind: "global-output-closure", operandBoundaries: ["embedding"] }],
    outputFunctions: [{ name: "terminal_logit", operationId: "layer", fixedDimension: 0, coordinate: ["zero"], parameters: [], root: "call", finalQuantization: "none" }],
    coverage: { sourceAssignments: 1, compiledOperationTemplates: 1, learnedRationalTableReads: 1, runtimeDefinedReductionsLowered: 0, unresolvedRuntimeReductions: 0, intermediateIeeeRoundingNodes: 0 },
  } satisfies Gemma4ParametricExactRealProgram;
  const result = await generateGemma4FlatFormulaObject(program, program.outputFunctions, {
    maxCharacters: 10_000n, maxUnrolledReductionTerms: 10,
    inputVariable: (_tensor, coordinates) => `x[${coordinates[0]}]`,
    learnedLiteral: (tensor, dtype, coordinates) => tensor === "layer.weight" && dtype === "BF16" && coordinates[0] === 0 ? "-1.125" : Promise.reject(new Error("peso inesperado")),
  });
  assert.deepEqual(result.formulas, { calc_final_0: "(((-1/4) * x[0]) + (-1.125 * x[1]))" });
  assert.deepEqual(result.freeVariables, ["x[0]", "x[1]"]);
  assert.doesNotMatch(result.formulas.calc_final_0!, /operation:|sha256:|layer/);
});

test("recusa expansão maior que o limite antes de retornar JSON parcial", async () => {
  const program = { kind: "gemma4-parametric-exact-real-simplified-program", schemaVersion: 1, semantics: "gemma4-exact-real-simplified-v1", inputBoundaries: [], expressionGraph: { kind: "gemma4-parametric-exact-real-expression-graph", schemaVersion: 1, semantics: "gemma4-exact-real-simplified-v1", nodeOrder: "dependency-order", nodes: [{ id: "c", kind: "rational", value: { numerator: "1", denominator: "1" } }] }, operationFunctions: [], outputFunctions: [{ name: "y", operationId: "y", fixedDimension: 0, coordinate: [], parameters: [], root: "c", finalQuantization: "none" }], coverage: { sourceAssignments: 0, compiledOperationTemplates: 0, learnedRationalTableReads: 0, runtimeDefinedReductionsLowered: 0, unresolvedRuntimeReductions: 0, intermediateIeeeRoundingNodes: 0 } } as Gemma4ParametricExactRealProgram;
  await assert.rejects(() => generateGemma4FlatFormulaObject(program, program.outputFunctions, { maxCharacters: 1n, maxUnrolledReductionTerms: 1, inputVariable: () => "x[0]", learnedLiteral: () => "1" }), Gemma4FlatFormulaLimitError);
});
