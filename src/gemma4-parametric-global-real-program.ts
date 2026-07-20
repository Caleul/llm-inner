import type { Gemma4LiteralCalculationGraph, Gemma4LiteralInstantiatedCalculation } from "./gemma4-literal-calculation-graph.js";
import { Gemma4ParametricRealBuilder, type Gemma4ParametricRealExpressionGraph, validateGemma4ParametricRealExpressionGraph } from "./gemma4-parametric-real-expression.js";
import { compileGemma4ScalarCalculationToParametricReal } from "./gemma4-parametric-real-scalar-compiler.js";

export interface Gemma4ParametricOutputFunction {
  name: string;
  operationId: string;
  fixedDimension: number;
  coordinate: string[];
  parameters: string[];
  root: string;
  finalQuantization: "F32-round-to-nearest-ties-to-even" | "BF16-round-to-nearest-ties-to-even" | "none";
}

export interface Gemma4ParametricOperationFunction {
  functionId: string;
  operationId: string;
  ordinal: number;
  output: string;
  parameters: Array<{ name: string; node: string }>;
  root: string;
  predecessorFunctions: string[];
  closureKind: "global-output-closure" | "standalone-runtime-reduction";
  operandBoundaries: string[];
}

export interface Gemma4ParametricExactRealProgram {
  kind: "gemma4-parametric-exact-real-simplified-program";
  schemaVersion: 1;
  semantics: "gemma4-exact-real-simplified-v1";
  inputBoundaries: string[];
  expressionGraph: Gemma4ParametricRealExpressionGraph;
  operationFunctions: Gemma4ParametricOperationFunction[];
  outputFunctions: Gemma4ParametricOutputFunction[];
  coverage: {
    sourceAssignments: number;
    compiledOperationTemplates: number;
    learnedRationalTableReads: number;
    runtimeDefinedReductionsLowered: number;
    unresolvedRuntimeReductions: 0;
    intermediateIeeeRoundingNodes: 0;
  };
}

export interface Gemma4ParametricOutputFamilyRequest {
  name: string;
  operationId: string;
  dimensions: number;
  /** Output axes before the last feature/token axis become function parameters. */
  parameterAxes: string[];
  finalQuantization: Gemma4ParametricOutputFunction["finalQuantization"];
}

export function buildGemma4ParametricExactRealProgram(
  graph: Gemma4LiteralCalculationGraph,
  requests: readonly Gemma4ParametricOutputFamilyRequest[],
  inputBoundaries: ReadonlySet<string>,
  additionalOperationIds: readonly string[] = [],
): Gemma4ParametricExactRealProgram {
  const builder = new Gemma4ParametricRealBuilder();
  const byOperation = new Map(graph.assignments.map((assignment) => [assignment.operationId, assignment]));
  const byOutput = new Map(graph.assignments.map((assignment) => [assignment.output, assignment]));
  const targetAssignments = requests.map((request) => {
    const assignment = byOperation.get(request.operationId);
    if (!assignment) throw new Error(`Família paramétrica requer operação ausente ${request.operationId}.`);
    return assignment;
  });
  const additionalAssignments = additionalOperationIds.map((operationId) => {
    const assignment = byOperation.get(operationId);
    if (!assignment) throw new Error(`Cobertura paramétrica requer operação ausente ${operationId}.`);
    return assignment;
  });
  const reachable = reachableAssignments(targetAssignments, byOutput, inputBoundaries);
  const compilationAssignments = new Set([...reachable, ...additionalAssignments]);
  const operationFunctions: Gemma4ParametricOperationFunction[] = [];
  const runtimeReductions = new Set<string>();
  let learnedReads = 0;
  for (const assignment of [...compilationAssignments].sort((left, right) => left.ordinal - right.ordinal)) {
    const functionId = operationFunctionId(assignment.operationId);
    const standaloneRuntimeReduction = !reachable.has(assignment);
    const parameters = assignment.scalarCalculation.outputCoordinates.map((name) => ({
      name,
      node: builder.integerParameter(`${functionId}:${name}`),
    }));
    const integerBindings = reductionExtentBindings(assignment, builder, byOutput);
    const predecessorFunctions = new Set<string>();
    const operandBoundaries = new Set<string>();
    const result = compileGemma4ScalarCalculationToParametricReal(assignment.scalarCalculation, parameters.map((parameter) => parameter.node), {
      builder,
      scope: functionId,
      integerBindings,
      resolveTensorElement: (name, inputCoordinates) => {
        if (inputBoundaries.has(name) || standaloneRuntimeReduction) {
          operandBoundaries.add(name);
          return builder.inputElement(name, inputCoordinates, inferInputType(name));
        }
        const producer = byOutput.get(name);
        if (!producer || !reachable.has(producer)) throw new Error(`${functionId}: folha paramétrica não declarada ${name}.`);
        const predecessorFunction = operationFunctionId(producer.operationId);
        predecessorFunctions.add(predecessorFunction);
        return builder.functionCall(predecessorFunction, inputCoordinates);
      },
      resolveLearnedElement: (role, learnedCoordinates) => {
        const operand = assignment.learnedOperands?.find((candidate) => candidate.role === role);
        if (!operand || !isDenseStorageDtype(operand.tensor.storageDtype)) throw new Error(`${functionId}: operando racional aprendido ausente ${role}.`);
        learnedReads += 1;
        return builder.learnedElement(operand.tensor.name, operand.tensor.storageDtype, learnedCoordinates, operand.decoderId);
      },
      resolveTensorAxis: (name, axis) => {
        const producer = byOutput.get(name);
        const token = producer?.outputDomain.shape[axis];
        return token && /^\d+$/.test(token) ? builder.integerConstant(Number(token)) : builder.tensorAxis(name, axis);
      },
    });
    operationFunctions.push({
      functionId,
      operationId: assignment.operationId,
      ordinal: assignment.ordinal,
      output: assignment.output,
      parameters,
      root: result.root,
      predecessorFunctions: [...predecessorFunctions].sort(),
      closureKind: standaloneRuntimeReduction ? "standalone-runtime-reduction" : "global-output-closure",
      operandBoundaries: [...operandBoundaries].sort(),
    });
    if (assignment.scalarCalculation.reduction?.order === "runtime-defined") runtimeReductions.add(functionId);
  }
  const outputFunctions: Gemma4ParametricOutputFunction[] = [];
  for (const request of requests) {
    const assignment = byOperation.get(request.operationId);
    if (!assignment) throw new Error(`Família paramétrica requer operação ausente ${request.operationId}.`);
    if (assignment.scalarCalculation.outputCoordinates.length !== request.parameterAxes.length + 1) {
      throw new Error(`${request.operationId}: família paramétrica requer eixos ${assignment.scalarCalculation.outputCoordinates.join(",")}.`);
    }
    const parameters = request.parameterAxes.map((axis) => builder.integerParameter(`${request.name}:${axis}`));
    for (let dimension = 0; dimension < request.dimensions; dimension += 1) {
      const coordinate = [...parameters, builder.integerConstant(dimension)];
      outputFunctions.push({
        name: request.name,
        operationId: request.operationId,
        fixedDimension: dimension,
        coordinate,
        parameters: [...request.parameterAxes],
        root: builder.functionCall(operationFunctionId(assignment.operationId), coordinate),
        finalQuantization: request.finalQuantization,
      });
    }
  }
  const expressionGraph = builder.build();
  const program: Gemma4ParametricExactRealProgram = {
    kind: "gemma4-parametric-exact-real-simplified-program",
    schemaVersion: 1,
    semantics: "gemma4-exact-real-simplified-v1",
    inputBoundaries: [...inputBoundaries].sort(),
    expressionGraph,
    operationFunctions,
    outputFunctions,
    coverage: {
      sourceAssignments: graph.assignments.length,
      compiledOperationTemplates: operationFunctions.length,
      learnedRationalTableReads: learnedReads,
      runtimeDefinedReductionsLowered: runtimeReductions.size,
      unresolvedRuntimeReductions: 0,
      intermediateIeeeRoundingNodes: 0,
    },
  };
  validateGemma4ParametricExactRealProgram(program);
  return program;
}

export function validateGemma4ParametricExactRealProgram(program: Gemma4ParametricExactRealProgram): void {
  if (program.kind !== "gemma4-parametric-exact-real-simplified-program" || program.schemaVersion !== 1 || program.semantics !== "gemma4-exact-real-simplified-v1" ||
    program.coverage.unresolvedRuntimeReductions !== 0 || program.coverage.intermediateIeeeRoundingNodes !== 0 || program.outputFunctions.length === 0) {
    throw new Error("Programa paramétrico global Gemma 4 possui contrato incompleto.");
  }
  validateGemma4ParametricRealExpressionGraph(program.expressionGraph);
  const ids = new Set(program.expressionGraph.nodes.map((node) => node.id));
  const functions = new Map(program.operationFunctions.map((entry) => [entry.functionId, entry]));
  if (functions.size !== program.operationFunctions.length || functions.size !== program.coverage.compiledOperationTemplates) {
    throw new Error("Programa paramétrico Gemma 4 possui funções duplicadas ou cobertura divergente.");
  }
  for (const entry of program.operationFunctions) {
    if (!ids.has(entry.root) || entry.predecessorFunctions.some((functionId) => !functions.has(functionId))) {
      throw new Error(`${entry.functionId}: closure de função paramétrica incompleta.`);
    }
  }
  for (const node of program.expressionGraph.nodes) if (node.kind === "function-call" && !functions.has(node.functionId)) {
    throw new Error(`Chamada paramétrica referencia função ausente ${node.functionId}.`);
  }
  const families = new Map<string, number[]>();
  for (const output of program.outputFunctions) {
    if (!ids.has(output.root)) throw new Error(`${output.name}[${output.fixedDimension}]: raiz paramétrica ausente.`);
    const dimensions = families.get(output.name) ?? [];
    dimensions.push(output.fixedDimension);
    families.set(output.name, dimensions);
  }
  for (const [name, dimensions] of families) dimensions.sort((a, b) => a - b).forEach((dimension, index) => {
    if (dimension !== index) throw new Error(`${name}: dimensões paramétricas não são completas desde zero.`);
  });
}

export function validateGemma4ParametricExactRealAlignment(
  program: Gemma4ParametricExactRealProgram,
  graph: Gemma4LiteralCalculationGraph,
  expectedFamilies: Readonly<Record<string, { dimensions: number; finalQuantization: Gemma4ParametricOutputFunction["finalQuantization"] }>>,
): void {
  validateGemma4ParametricExactRealProgram(program);
  const graphOperations = new Set(graph.assignments.map((assignment) => assignment.operationId));
  const runtimeOperations = new Set(graph.assignments
    .filter((assignment) => assignment.scalarCalculation.reduction?.order === "runtime-defined")
    .map((assignment) => assignment.operationId));
  const standalone = program.operationFunctions.filter((entry) => entry.closureKind === "standalone-runtime-reduction");
  if (program.coverage.sourceAssignments !== graph.assignments.length ||
    program.coverage.runtimeDefinedReductionsLowered !== runtimeOperations.size ||
    standalone.some((entry) => !runtimeOperations.has(entry.operationId)) ||
    [...runtimeOperations].some((operationId) => !program.operationFunctions.some((entry) => entry.operationId === operationId)) ||
    program.operationFunctions.some((entry) => !graphOperations.has(entry.operationId))) {
    throw new Error("Programa real simplificado diverge da cobertura de operações/reduções do grafo literal.");
  }
  const actualFamilies = new Map<string, number>();
  for (const output of program.outputFunctions) actualFamilies.set(output.name, (actualFamilies.get(output.name) ?? 0) + 1);
  if (actualFamilies.size !== Object.keys(expectedFamilies).length || Object.entries(expectedFamilies).some(([name, expected]) =>
    actualFamilies.get(name) !== expected.dimensions || program.outputFunctions.some((output) => output.name === name && output.finalQuantization !== expected.finalQuantization))) {
    throw new Error("Programa real simplificado não declara exatamente uma função por dimensão pública.");
  }
}

function reductionExtentBindings(
  assignment: Gemma4LiteralInstantiatedCalculation,
  builder: Gemma4ParametricRealBuilder,
  byOutput: ReadonlyMap<string, Gemma4LiteralInstantiatedCalculation>,
): Record<string, string> {
  const bindings: Record<string, string> = {};
  for (const binding of assignment.scalarCalculation.statementEnvironment.reductions) {
    if (!binding.extentAlias) continue;
    const reduction = binding.source === "assignment" ? assignment.scalarCalculation.reduction
      : assignment.scalarCalculation.reductionStages?.find((stage) => stage.id === binding.stageId);
    const domain = reduction?.domains[binding.domainOrdinal];
    if (!domain) throw new Error(`${assignment.operationId}: extent paramétrico ausente para ${binding.index}.`);
    const extent = domain.endExclusive;
    if (extent.kind === "constant") bindings[binding.extentAlias] = builder.integerConstant(extent.value);
    else {
      const producer = byOutput.get(extent.tensor), token = producer?.outputDomain.shape[extent.axis];
      bindings[binding.extentAlias] = token && /^\d+$/.test(token)
        ? builder.integerConstant(Number(token))
        : builder.tensorAxis(extent.tensor, extent.axis);
    }
  }
  return bindings;
}

function isDenseStorageDtype(dtype: string): dtype is "BF16" | "F16" | "F32" {
  return dtype === "BF16" || dtype === "F16" || dtype === "F32";
}

function inferInputType(name: string): "real" | "integer" | "boolean" {
  if (name.endsWith("_ids") || name.includes("position")) return "integer";
  if (name === "input_features_mask") return "boolean";
  return "real";
}

function operationFunctionId(operationId: string): string {
  return `operation:${operationId}`;
}

function reachableAssignments(
  targets: readonly Gemma4LiteralInstantiatedCalculation[],
  byOutput: ReadonlyMap<string, Gemma4LiteralInstantiatedCalculation>,
  inputBoundaries: ReadonlySet<string>,
): Set<Gemma4LiteralInstantiatedCalculation> {
  const reachable = new Set<Gemma4LiteralInstantiatedCalculation>();
  const visit = (assignment: Gemma4LiteralInstantiatedCalculation): void => {
    if (reachable.has(assignment)) return;
    reachable.add(assignment);
    for (const input of assignment.orderedInputs) {
      if (inputBoundaries.has(input)) continue;
      const producer = byOutput.get(input);
      if (!producer) throw new Error(`${assignment.operationId}: input paramétrico sem produtor ou boundary ${input}.`);
      visit(producer);
    }
  };
  targets.forEach(visit);
  return reachable;
}
