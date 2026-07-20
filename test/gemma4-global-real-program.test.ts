import assert from "node:assert/strict";
import test from "node:test";
import { buildGemma4ExactRealSimplifiedProgram, validateGemma4ExactRealSimplifiedProgram } from "../src/gemma4-global-real-program.js";
import type { Gemma4LiteralCalculationGraph, Gemma4LiteralInstantiatedCalculation } from "../src/gemma4-literal-calculation-graph.js";
import { buildGemma4LiteralScalarStatementDataflow, type Gemma4LiteralScalarCalculation } from "../src/gemma4-literal-scalar-calculations.js";
import { buildGemma4LiteralScalarStatementEnvironment, buildGemma4LiteralScalarStatementPrograms } from "../src/gemma4-literal-scalar-statement-programs.js";
import { buildGemma4LiteralOutputCoordinateNavigation, gemma4LiteralCoordinateExpressionLanguage } from "../src/gemma4-literal-coordinate-accesses.js";

test("substitui produtores entre camadas e deixa uma raiz fechada por output", () => {
  const first = calculation("layer_0", "h", ["x"], "h[hidden]=F32(x[hidden]*decode(weight)[hidden])", ["weight"]);
  const second = calculation("layer_1", "y", ["h"], "y[hidden]=BF16(F32(h[hidden]*decode(weight)[hidden]+decode(bias)[hidden]))", ["weight", "bias"]);
  const assignments = [instantiate(first, 0), instantiate(second, 1)];
  assignments[0]!.consumers = ["layer_1"];
  assignments[1]!.predecessors = [{ input: "h", producerOperationId: "layer_0", accesses: [], scalarUse: "addressed" }];
  const graph: Gemma4LiteralCalculationGraph = {
    kind: "gemma4-literal-instantiated-calculation-graph", schemaVersion: 8, order: "dependency-order",
    coordinateLanguage: gemma4LiteralCoordinateExpressionLanguage(),
    assignments,
  };
  const program = buildGemma4ExactRealSimplifiedProgram(graph, [{
    name: "y", operationId: "layer_1", coordinate: [0], finalQuantization: "BF16-round-to-nearest-ties-to-even",
  }], {
    declaredInputs: new Set(["x"]),
    resolveLearnedElement: (_assignment, role, coordinates, _bindings, builder) => {
      if (role === "bias") return builder.rational(1n, 8n);
      return builder.rational(BigInt(coordinates[0]! + 1), 4n);
    },
  });
  validateGemma4ExactRealSimplifiedProgram(program);
  assert.equal(program.outputFunctions.length, 1);
  assert.equal(program.coverage.compiledOperationCoordinates, 2);
  assert.deepEqual(program.coverage.reachableIntermediateTensorLeaves, []);
  assert.deepEqual(program.expressionGraph.nodes.filter((node) => node.kind === "input").map((node) => node.name), ["x"]);
  assert.ok(!JSON.stringify(program).match(/unpublished-provider|runtime-defined|F32\(|BF16\(/));
});

function calculation(id: string, output: string, inputs: string[], formula: string, learnedRoles: string[]): Gemma4LiteralScalarCalculation {
  const statements = [formula];
  const programs = buildGemma4LiteralScalarStatementPrograms(statements, output, id);
  return {
    scope: "text-layer", definitionId: id, operation: "fixture", orderedInputs: inputs, output, outputCoordinates: ["hidden"],
    learnedOperandRoles: learnedRoles as Gemma4LiteralScalarCalculation["learnedOperandRoles"], scalarAssignments: statements,
    statementDataflow: buildGemma4LiteralScalarStatementDataflow(statements, output, id), statementPrograms: programs,
    statementEnvironment: buildGemma4LiteralScalarStatementEnvironment(programs, {
      output, outputCoordinates: ["hidden"], orderedInputs: inputs, learnedOperandRoles: learnedRoles, reductions: [],
    }, id),
    formula, dtypePolicy: { inputDtype: "BF16", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "BF16" }, reproducibility: "literal",
  };
}

function instantiate(scalarCalculation: Gemma4LiteralScalarCalculation, ordinal: number): Gemma4LiteralInstantiatedCalculation {
  return {
    ordinal, operationId: scalarCalculation.definitionId, definitionId: scalarCalculation.definitionId, operation: scalarCalculation.operation,
    scope: "text-layer", orderedInputs: scalarCalculation.orderedInputs, output: scalarCalculation.output,
    outputCoordinate: buildGemma4LiteralOutputCoordinateNavigation(scalarCalculation.output, scalarCalculation.scalarAssignments),
    outputDomain: { structure: "tensor", dtype: "BF16", layout: "row-major", axes: [{ axis: 0, name: "hidden", size: "1", indexDomain: "0 <= hidden < 1" }], shape: ["1"] },
    scalarCalculation, predecessors: scalarCalculation.orderedInputs.map((input) => ({ input, accesses: [], scalarUse: "addressed" as const })),
    consumers: [], consumerCoordinates: [],
  };
}
