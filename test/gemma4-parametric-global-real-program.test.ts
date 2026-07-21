import assert from "node:assert/strict";
import test from "node:test";
import { buildGemma4ParametricExactRealProgram } from "../src/gemma4-parametric-global-real-program.js";
import { evaluateGemma4ParametricOutput } from "../src/gemma4-parametric-real-evaluator.js";
import type { Gemma4LiteralCalculationGraph, Gemma4LiteralInstantiatedCalculation } from "../src/gemma4-literal-calculation-graph.js";
import { gemma4LiteralCoordinateExpressionLanguage, buildGemma4LiteralOutputCoordinateNavigation } from "../src/gemma4-literal-coordinate-accesses.js";
import { buildGemma4LiteralScalarStatementDataflow, type Gemma4LiteralScalarCalculation } from "../src/gemma4-literal-scalar-calculations.js";
import { buildGemma4LiteralScalarStatementEnvironment, buildGemma4LiteralScalarStatementPrograms } from "../src/gemma4-literal-scalar-statement-programs.js";
import { Gemma4ParametricRealBuilder } from "../src/gemma4-parametric-real-expression.js";
import type { Gemma4ParametricExactRealProgram } from "../src/gemma4-parametric-global-real-program.js";

test("emite e executa uma função paramétrica fechada por dimensão sem desenrolar o dot product", async () => {
  const scalarAssignments = ["y[batch,sequence,output_feature]=BF16(REDUCE(input_feature=0..in_features-1,exact_product(x[batch,sequence,input_feature]*decode(weight)[output_feature,input_feature])))"];
  const programs = buildGemma4LiteralScalarStatementPrograms(scalarAssignments, "y", "parametric-linear");
  const scalar: Gemma4LiteralScalarCalculation = {
    scope: "text-layer", definitionId: "linear", operation: "linear", orderedInputs: ["x"], output: "y",
    outputCoordinates: ["batch", "sequence", "output_feature"], learnedOperandRoles: ["weight"], scalarAssignments,
    statementDataflow: buildGemma4LiteralScalarStatementDataflow(scalarAssignments, "y", "parametric-linear"), statementPrograms: programs,
    statementEnvironment: buildGemma4LiteralScalarStatementEnvironment(programs, {
      output: "y", outputCoordinates: ["batch", "sequence", "output_feature"], orderedInputs: ["x"], learnedOperandRoles: ["weight"], reductions: [{ indices: ["input_feature=0..in_features-1"], source: "assignment" }],
    }, "parametric-linear"),
    formula: scalarAssignments[0]!, dtypePolicy: { inputDtype: "BF16", computeDtype: "F32", accumulationDtype: "runtime-defined", outputDtype: "BF16" },
    reduction: { indices: ["input_feature=0..in_features-1"], domains: [{ index: "input_feature", startInclusive: 0, endExclusive: { kind: "constant", value: 2 }, order: "ascending" }], order: "runtime-defined" },
    reproducibility: "fail-closed-runtime-reduction",
  };
  const assignment: Gemma4LiteralInstantiatedCalculation = {
    ordinal: 0, operationId: "linear", definitionId: "linear", operation: "linear", scope: "text-layer", orderedInputs: ["x"], output: "y",
    outputCoordinate: buildGemma4LiteralOutputCoordinateNavigation("y", scalarAssignments),
    outputDomain: { structure: "tensor", dtype: "BF16", layout: "row-major", shape: ["B", "S", "2"], axes: [
      { axis: 0, name: "batch", size: "B", indexDomain: "0 <= batch < B" },
      { axis: 1, name: "sequence", size: "S", indexDomain: "0 <= sequence < S" },
      { axis: 2, name: "output_feature", size: "2", indexDomain: "0 <= output_feature < 2" },
    ] }, scalarCalculation: scalar,
    learnedOperands: [{ role: "weight", tensor: { name: "weight", shape: [2, 2], storageDtype: "BF16" }, logicalIndices: [], decoderId: "decode:weight" }],
    predecessors: [{ input: "x", accesses: [], scalarUse: "addressed" }], consumers: [], consumerCoordinates: [],
  };
  const graph: Gemma4LiteralCalculationGraph = { kind: "gemma4-literal-instantiated-calculation-graph", schemaVersion: 8, order: "dependency-order", coordinateLanguage: gemma4LiteralCoordinateExpressionLanguage(), assignments: [assignment] };
  const program = buildGemma4ParametricExactRealProgram(graph, [{ name: "y", operationId: "linear", dimensions: 2, parameterAxes: ["batch", "sequence"], finalQuantization: "BF16-round-to-nearest-ties-to-even" }], new Set(["x"]));
  assert.equal(program.outputFunctions.length, 2);
  assert.equal(program.coverage.runtimeDefinedReductionsLowered, 1);
  assert.equal(program.expressionGraph.nodes.filter((node) => node.kind === "finite-sum").length, 1);
  assert.ok(program.expressionGraph.nodes.length < 40);
  assert.ok(!JSON.stringify(program).match(/unpublished-provider-boundary|runtime-defined|F32\(|BF16\(/));
  const weights = [[-0.25, -1.125], [2, 3]];
  const result = await evaluateGemma4ParametricOutput(program, "y", 0, {
    parameters: { batch: 0, sequence: 0 },
    inputs: {
      axis: (_tensor, axis) => [1, 1, 2][axis]!,
      element: (_tensor, coordinates) => [2, 4][coordinates[2]!]!,
    },
    learned: { element: (_tensor, _dtype, coordinates) => weights[coordinates[0]!]![coordinates[1]!]! },
  });
  assert.equal(result, -5);
});

test("memoiza subexpressões somente pelos índices livres e valida o limite de memória", async () => {
  const builder = new Gemma4ParametricRealBuilder();
  const zero = builder.integerConstant(0), end = builder.integerConstant(20), index = builder.boundIndex("i");
  const invariant = builder.inputElement("x", [zero]);
  const body = builder.add(invariant, index);
  const root = builder.finiteReduction("finite-sum", "i", zero, end, body);
  const program: Gemma4ParametricExactRealProgram = {
    kind: "gemma4-parametric-exact-real-simplified-program", schemaVersion: 1, semantics: "gemma4-exact-real-simplified-v1", inputBoundaries: ["x"],
    expressionGraph: builder.build(), operationFunctions: [],
    outputFunctions: [{ name: "y", operationId: "sum", fixedDimension: 0, coordinate: [], parameters: [], root, finalQuantization: "none" }],
    coverage: { sourceAssignments: 1, compiledOperationTemplates: 0, learnedRationalTableReads: 0, runtimeDefinedReductionsLowered: 1, unresolvedRuntimeReductions: 0, intermediateIeeeRoundingNodes: 0 },
  };
  let reads = 0;
  const value = await evaluateGemma4ParametricOutput(program, "y", 0, {
    parameters: {}, maximumMemoEntries: 100,
    inputs: { axis: () => 1, element: () => { reads += 1; return 2; } },
    learned: { element: () => { throw new Error("peso inesperado"); } },
  });
  assert.equal(value, 230);
  assert.equal(reads, 1);
  await assert.rejects(() => evaluateGemma4ParametricOutput(program, "y", 0, {
    parameters: {}, maximumMemoEntries: 0,
    inputs: { axis: () => 1, element: () => 2 },
    learned: { element: () => { throw new Error("peso inesperado"); } },
  }), /Limite de memoização paramétrica inválido/);
});
