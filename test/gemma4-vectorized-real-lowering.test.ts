import assert from "node:assert/strict";
import test from "node:test";
import type { Gemma4LiteralArtifactIntegrityManifest } from "../src/gemma4-literal-artifact-integrity.js";
import type { Gemma4LiteralCalculationGraph } from "../src/gemma4-literal-calculation-graph.js";
import type { Gemma4ParametricExactRealProgram } from "../src/gemma4-parametric-global-real-program.js";
import { assertGemma4VectorizedRealLoweringPlanMatches, buildGemma4VectorizedRealLoweringContract, buildGemma4VectorizedRealLoweringPlan, validateGemma4VectorizedRealLoweringPlan } from "../src/gemma4-vectorized-real-lowering.js";

const operations = ["activation", "elementwise", "linear", "reshape_heads", "rms_norm", "rotary_embedding", "scaled_dot_product_attention", "select_per_layer", "tensor_scale"];

test("vincula cada closure real autenticada a uma família vetorizada de kernel", () => {
  const { program, graph, manifest } = fixture();
  const contract = buildGemma4VectorizedRealLoweringContract(program, graph, manifest);
  assert.equal(contract.source.globalClosureFunctions, operations.length);
  assert.equal(contract.execution.intermediateBf16Boundaries, 0);
  assert.equal(contract.execution.directlyLoadsStandaloneSsaFile, false);
  assert.deepEqual(Object.values(contract.coverage.kernels), operations.map(() => 1));
  assert.match(contract.functionBindingsSha256, /^[0-9a-f]{64}$/);
  assert.equal(buildGemma4VectorizedRealLoweringContract(program, graph, manifest).functionBindingsSha256, contract.functionBindingsSha256);
});

test("falha fechado quando a closure contém operação sem lowering", () => {
  const { program, graph, manifest } = fixture();
  graph.assignments[0]!.operation = "unknown_runtime_operator";
  assert.throws(() => buildGemma4VectorizedRealLoweringContract(program, graph, manifest), /operação sem lowering vetorizado/);
});

test("plano persistido precisa preservar cada binding e coincidir integralmente com o artefato", () => {
  const { program, graph, manifest } = fixture();
  const plan = buildGemma4VectorizedRealLoweringPlan(program, graph, manifest);
  validateGemma4VectorizedRealLoweringPlan(structuredClone(plan));
  assert.doesNotThrow(() => assertGemma4VectorizedRealLoweringPlanMatches(structuredClone(plan), program, graph, manifest));
  const tampered = structuredClone(plan);
  tampered.functionBindings[0]!.output = "tampered";
  assert.throws(() => validateGemma4VectorizedRealLoweringPlan(tampered), /compromissos e contagens/);
  const stale = structuredClone(plan);
  stale.contract.source.artifactIntegritySha256 = "c".repeat(64);
  assert.throws(() => assertGemma4VectorizedRealLoweringPlanMatches(stale, program, graph, manifest), /não corresponde ao programa real autenticado/);
});

function fixture(): {
  program: Gemma4ParametricExactRealProgram;
  graph: Gemma4LiteralCalculationGraph;
  manifest: Gemma4LiteralArtifactIntegrityManifest;
} {
  const operationFunctions = operations.map((operation, ordinal) => ({
    functionId: `op:${ordinal}`, operationId: `operation_${ordinal}`, ordinal, output: `value_${ordinal}`,
    parameters: [], root: "rational:one", predecessorFunctions: [], closureKind: "global-output-closure" as const, operandBoundaries: [],
  }));
  const program = {
    kind: "gemma4-parametric-exact-real-simplified-program", schemaVersion: 1, semantics: "gemma4-exact-real-simplified-v1", inputBoundaries: [],
    expressionGraph: { kind: "gemma4-parametric-real-expression-graph", schemaVersion: 1, nodes: [{ id: "rational:one", kind: "rational", numerator: "1", denominator: "1" }] },
    operationFunctions,
    outputFunctions: [{ name: "terminal_logits", operationId: "operation_0", fixedDimension: 0, coordinate: [], parameters: [], root: "rational:one", finalQuantization: "BF16-round-to-nearest-ties-to-even" }],
    coverage: { sourceAssignments: operations.length, compiledOperationTemplates: operations.length, learnedRationalTableReads: 0, runtimeDefinedReductionsLowered: 0, unresolvedRuntimeReductions: 0, intermediateIeeeRoundingNodes: 0 },
  } as unknown as Gemma4ParametricExactRealProgram;
  const graph = {
    kind: "gemma4-literal-instantiated-calculation-graph", schemaVersion: 8, order: "dependency-order", coordinateLanguage: {} as Gemma4LiteralCalculationGraph["coordinateLanguage"],
    assignments: operations.map((operation, ordinal) => ({ ordinal, operationId: `operation_${ordinal}`, operation, output: `value_${ordinal}` })),
  } as Gemma4LiteralCalculationGraph;
  const manifest: Gemma4LiteralArtifactIntegrityManifest = {
    kind: "gemma4-literal-artifact-integrity-manifest", schemaVersion: 1, algorithm: "SHA-256",
    canonicalization: "UTF-8 bytes of ECMAScript JSON.stringify for each named section, in declared order",
    sections: [{ name: "realSimplifiedProgram", canonicalBytes: 1, sha256: "a".repeat(64) }], rootSha256: "b".repeat(64),
  };
  return { program, graph, manifest };
}
