import assert from "node:assert/strict";
import test from "node:test";
import type { Gemma4ParametricExactRealProgram } from "../src/gemma4-parametric-global-real-program.js";
import { estimateGemma4FlatExpansion, gemma4SsaExpression } from "../src/gemma4-real-ssa-export.js";

test("estima substituição física e emite atribuição SSA legível", () => {
  const constant = { id: "constant", kind: "rational" as const, value: { numerator: "-1", denominator: "4" } };
  const call = { id: "call", kind: "function-call" as const, functionId: "operation:linear", arguments: [] };
  const program = {
    kind: "gemma4-parametric-exact-real-simplified-program", schemaVersion: 1, semantics: "gemma4-exact-real-simplified-v1", inputBoundaries: [],
    expressionGraph: { kind: "gemma4-parametric-exact-real-expression-graph", schemaVersion: 1, semantics: "gemma4-exact-real-simplified-v1", nodeOrder: "dependency-order", nodes: [constant, call] },
    operationFunctions: [{ functionId: "operation:linear", operationId: "linear", ordinal: 0, output: "y", parameters: [], root: constant.id, predecessorFunctions: [], closureKind: "global-output-closure", operandBoundaries: [] }],
    outputFunctions: [{ name: "terminal_logit", operationId: "linear", fixedDimension: 0, coordinate: [], parameters: [], root: call.id, finalQuantization: "BF16-round-to-nearest-ties-to-even" }],
    coverage: { sourceAssignments: 1, compiledOperationTemplates: 1, learnedRationalTableReads: 0, runtimeDefinedReductionsLowered: 0, unresolvedRuntimeReductions: 0, intermediateIeeeRoundingNodes: 0 },
  } satisfies Gemma4ParametricExactRealProgram;
  assert.equal(gemma4SsaExpression(constant), "-1/4");
  assert.deepEqual(estimateGemma4FlatExpansion(program, program.outputFunctions[0]!), {
    family: "terminal_logit", dimension: 0, symbolicNodeOccurrences: "2", decimalDigits: 1, conservativeMinimumBytes: "4", reductionsUnrolled: false,
  });
});
