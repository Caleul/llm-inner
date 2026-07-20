import type { Gemma4LiteralCalculationGraph, Gemma4LiteralInstantiatedCalculation } from "./gemma4-literal-calculation-graph.js";
import type { Gemma4RealExpressionGraph } from "./gemma4-real-expression.js";
import { Gemma4RealExpressionBuilder, realNodeDependencies, validateGemma4RealExpressionGraph } from "./gemma4-real-expression.js";
import { compileGemma4ScalarCalculationToExactReal } from "./gemma4-real-scalar-program-compiler.js";
import { simplifyGemma4RealExpressionGraph } from "./gemma4-real-expression-simplifier.js";

export interface Gemma4GlobalRealOutputFunction {
  name: string;
  coordinate: number[];
  root: string;
  finalQuantization: "F32-round-to-nearest-ties-to-even" | "BF16-round-to-nearest-ties-to-even" | "none";
}

export interface Gemma4ExactRealSimplifiedProgram {
  kind: "gemma4-exact-real-simplified-program";
  schemaVersion: 1;
  semantics: {
    id: "gemma4-exact-real-simplified-v1";
    learnedConstants: "exact-dyadic-rationals-from-embedded-storage-bits";
    algebra: "real-number-identities-with-finite-input-domain";
    intermediateIeeeRounding: "removed";
    bmm: "provider-independent-real-sum-of-products";
    transcendentalFunctions: "mathematical-real-functions";
    quantization: "only-at-declared-output-boundary";
    relationshipToIeeeBaseline: "candidate-requiring-differential-validation";
  };
  sourceGraph: {
    kind: Gemma4LiteralCalculationGraph["kind"];
    schemaVersion: Gemma4LiteralCalculationGraph["schemaVersion"];
    assignmentCount: number;
  };
  expressionGraph: Gemma4RealExpressionGraph;
  outputFunctions: Gemma4GlobalRealOutputFunction[];
  coverage: {
    compiledOperationCoordinates: number;
    runtimeDefinedBmmDependenciesLoweredAsRealSums: number;
    unresolvedRuntimeReductionDependencies: 0;
    reachableIntermediateTensorLeaves: string[];
    reachableLearnedOperandLeaves: string[];
    reachableIeeeCastNodes: 0;
  };
}

export interface Gemma4GlobalRealOutputRequest {
  name: string;
  operationId: string;
  coordinate: number[];
  finalQuantization: Gemma4GlobalRealOutputFunction["finalQuantization"];
}

export interface Gemma4GlobalRealCompilationContext {
  /** Resolves one already-addressed learned scalar to an exact rational node. */
  resolveLearnedElement: (
    assignment: Gemma4LiteralInstantiatedCalculation,
    role: string,
    coordinates: readonly number[],
    reductionBindings: Readonly<Record<string, number>>,
    builder: Gemma4RealExpressionBuilder,
  ) => string;
  /** Inputs not produced by the calculation graph are the only legal variable leaves. */
  declaredInputs: ReadonlySet<string>;
}

/**
 * Recursively substitutes every producer coordinate into each requested output
 * coordinate. Memoization and content addressing preserve a DAG physically,
 * while every output root denotes one closed mathematical function.
 */
export function buildGemma4ExactRealSimplifiedProgram(
  calculationGraph: Gemma4LiteralCalculationGraph,
  requests: readonly Gemma4GlobalRealOutputRequest[],
  context: Gemma4GlobalRealCompilationContext,
): Gemma4ExactRealSimplifiedProgram {
  const byOperation = new Map(calculationGraph.assignments.map((assignment) => [assignment.operationId, assignment]));
  const byOutput = new Map(calculationGraph.assignments.map((assignment) => [assignment.output, assignment]));
  const builder = new Gemma4RealExpressionBuilder();
  const memo = new Map<string, string>();
  const active = new Set<string>();
  const compiledOperations = new Set<string>();
  const runtimeBmmDependencies = new Set<string>();
  const compileCoordinate = (assignment: Gemma4LiteralInstantiatedCalculation, coordinate: readonly number[]): string => {
    const key = `${assignment.operationId}[${coordinate.join(",")}]`;
    const cached = memo.get(key);
    if (cached) return cached;
    if (active.has(key)) throw new Error(`Grafo real Gemma 4 possui ciclo em ${key}.`);
    active.add(key);
    const result = compileGemma4ScalarCalculationToExactReal(assignment.scalarCalculation, coordinate, {
      builder,
      integerBindings: staticIntegerBindings(assignment, byOutput),
      resolveTensorElement: (name, inputCoordinate) => {
        if (context.declaredInputs.has(name)) return builder.input(name, inputCoordinate.map(String), inferInputType(name));
        const producer = byOutput.get(name);
        if (producer) return compileCoordinate(producer, inputCoordinate);
        throw new Error(`${key}: folha intermediária ou input não declarado ${name}[${inputCoordinate.join(",")}].`);
      },
      resolveTensorAxis: (name, axis) => {
        const producer = byOutput.get(name);
        const token = producer?.outputDomain.shape[axis];
        if (token && /^\d+$/.test(token)) return builder.rational(token);
        if (!producer && !context.declaredInputs.has(name)) throw new Error(`${key}: shape de tensor não declarado ${name}.shape[${axis}].`);
        return builder.input(`${name}.shape`, [String(axis)], "integer");
      },
      resolveLearnedElement: (role, learnedCoordinate, bindings) =>
        context.resolveLearnedElement(assignment, role, learnedCoordinate, bindings, builder),
    });
    active.delete(key);
    memo.set(key, result.root);
    compiledOperations.add(key);
    if (assignment.scalarCalculation.reduction?.order === "runtime-defined") runtimeBmmDependencies.add(key);
    return result.root;
  };
  const roots: Record<string, string> = {};
  const requestedMetadata = requests.map((request): Omit<Gemma4GlobalRealOutputFunction, "root"> => {
    const assignment = byOperation.get(request.operationId);
    if (!assignment) throw new Error(`Output real Gemma 4 requer operação ausente ${request.operationId}.`);
    const key = outputKey(request.name, request.coordinate);
    if (roots[key]) throw new Error(`Output real Gemma 4 duplicado ${key}.`);
    roots[key] = compileCoordinate(assignment, request.coordinate);
    return { name: request.name, coordinate: [...request.coordinate], finalQuantization: request.finalQuantization };
  });
  const simplified = simplifyGemma4RealExpressionGraph(builder.build(), roots);
  const outputFunctions = requestedMetadata.map((metadata) => ({
    ...metadata,
    root: simplified.roots[outputKey(metadata.name, metadata.coordinate)]!,
  }));
  const reachableInputs = reachableInputNames(simplified.graph, outputFunctions.map((output) => output.root));
  const intermediateLeaves = [...reachableInputs].filter((name) =>
    !context.declaredInputs.has(name) && !name.endsWith(".shape"));
  if (intermediateLeaves.length > 0) throw new Error(`Programa real Gemma 4 reteve folhas intermediárias: ${intermediateLeaves.join(", ")}.`);
  return {
    kind: "gemma4-exact-real-simplified-program",
    schemaVersion: 1,
    semantics: {
      id: "gemma4-exact-real-simplified-v1",
      learnedConstants: "exact-dyadic-rationals-from-embedded-storage-bits",
      algebra: "real-number-identities-with-finite-input-domain",
      intermediateIeeeRounding: "removed",
      bmm: "provider-independent-real-sum-of-products",
      transcendentalFunctions: "mathematical-real-functions",
      quantization: "only-at-declared-output-boundary",
      relationshipToIeeeBaseline: "candidate-requiring-differential-validation",
    },
    sourceGraph: {
      kind: calculationGraph.kind,
      schemaVersion: calculationGraph.schemaVersion,
      assignmentCount: calculationGraph.assignments.length,
    },
    expressionGraph: simplified.graph,
    outputFunctions,
    coverage: {
      compiledOperationCoordinates: compiledOperations.size,
      runtimeDefinedBmmDependenciesLoweredAsRealSums: runtimeBmmDependencies.size,
      unresolvedRuntimeReductionDependencies: 0,
      reachableIntermediateTensorLeaves: [],
      reachableLearnedOperandLeaves: [],
      reachableIeeeCastNodes: 0,
    },
  };
}

export function validateGemma4ExactRealSimplifiedProgram(program: Gemma4ExactRealSimplifiedProgram): void {
  if (program.kind !== "gemma4-exact-real-simplified-program" || program.schemaVersion !== 1 ||
    program.semantics.id !== "gemma4-exact-real-simplified-v1" || program.semantics.intermediateIeeeRounding !== "removed" ||
    program.coverage.unresolvedRuntimeReductionDependencies !== 0 || program.coverage.reachableIeeeCastNodes !== 0 ||
    program.coverage.reachableIntermediateTensorLeaves.length !== 0 || program.coverage.reachableLearnedOperandLeaves.length !== 0) {
    throw new Error("Programa global real Gemma 4 possui contrato ou cobertura incompleta.");
  }
  validateGemma4RealExpressionGraph(program.expressionGraph);
  const ids = new Set(program.expressionGraph.nodes.map((node) => node.id));
  const outputs = new Set<string>();
  for (const output of program.outputFunctions) {
    const key = outputKey(output.name, output.coordinate);
    if (outputs.has(key) || !ids.has(output.root)) throw new Error(`Programa global real Gemma 4 possui output inválido ${key}.`);
    outputs.add(key);
  }
}

function reachableInputNames(graph: Gemma4RealExpressionGraph, roots: readonly string[]): Set<string> {
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const inputs = new Set<string>(), visited = new Set<string>();
  const visit = (id: string): void => {
    if (visited.has(id)) return;
    const node = nodes.get(id);
    if (!node) throw new Error(`Closure global real Gemma 4 possui nó ausente ${id}.`);
    visited.add(id);
    if (node.kind === "input") inputs.add(node.name);
    realNodeDependencies(node).forEach(visit);
  };
  roots.forEach(visit);
  return inputs;
}

function inferInputType(name: string): "real" | "integer" | "boolean" {
  if (name.endsWith("_ids") || name.includes("position")) return "integer";
  if (name === "input_features_mask") return "boolean";
  return "real";
}

function outputKey(name: string, coordinate: readonly number[]): string {
  return `${name}[${coordinate.join(",")}]`;
}

function staticIntegerBindings(
  assignment: Gemma4LiteralInstantiatedCalculation,
  byOutput: ReadonlyMap<string, Gemma4LiteralInstantiatedCalculation>,
): Record<string, number> {
  const bindings: Record<string, number> = {};
  for (const binding of assignment.scalarCalculation.statementEnvironment.reductions) {
    if (!binding.extentAlias) continue;
    const reduction = binding.source === "assignment"
      ? assignment.scalarCalculation.reduction
      : assignment.scalarCalculation.reductionStages?.find((stage) => stage.id === binding.stageId);
    const domain = reduction?.domains[binding.domainOrdinal];
    if (!domain) throw new Error(`${assignment.operationId}: domínio real ausente para ${binding.index}.`);
    const end = domain.endExclusive;
    const extent = end.kind === "constant" ? end.value : staticTensorAxis(end.tensor, end.axis, byOutput);
    bindings[binding.extentAlias] = extent;
  }
  return bindings;
}

function staticTensorAxis(
  tensor: string,
  axis: number,
  byOutput: ReadonlyMap<string, Gemma4LiteralInstantiatedCalculation>,
): number {
  const producer = byOutput.get(tensor);
  const token = producer?.outputDomain.shape[axis];
  if (!token || !/^\d+$/.test(token)) {
    throw new Error(`Programa real Gemma 4 requer extent estático para ${tensor}.shape[${axis}], recebeu ${token ?? "ausente"}.`);
  }
  const value = Number(token);
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`Extent real Gemma 4 inválido para ${tensor}.shape[${axis}].`);
  return value;
}
