import type { Gemma4ParametricExactRealProgram, Gemma4ParametricOutputFunction } from "./gemma4-parametric-global-real-program.js";
import { Gemma4ParametricRealBuilder, parametricNodeDependencies, type Gemma4ParametricRealExpressionGraph, type Gemma4ParametricRealNode } from "./gemma4-parametric-real-expression.js";

export interface Gemma4ParametricReverseStep {
  step: number;
  functionId: string;
  operationId: string;
  ordinal: number;
  expandedCalls: number;
  nodesBefore: number;
  nodesAfterSimplification: number;
  remainingFunctionCalls: number;
}

export interface Gemma4ParametricReverseComposition {
  kind: "gemma4-parametric-reverse-algebraic-composition";
  schemaVersion: 1;
  output: { family: string; dimension: number; parameters: Record<string, number> };
  graph: Gemma4ParametricRealExpressionGraph;
  root: string;
  steps: Gemma4ParametricReverseStep[];
  remainingFunctionCalls: string[];
}

/** Expands the closest-to-output operation, simplifies through the canonical builder, then repeats. */
export function composeGemma4ParametricReverse(
  program: Gemma4ParametricExactRealProgram,
  output: Gemma4ParametricOutputFunction,
  parameters: Readonly<Record<string, number>>,
  maximumSteps: number,
): Gemma4ParametricReverseComposition {
  if (!Number.isSafeInteger(maximumSteps) || maximumSteps < 0) throw new Error("maximumSteps paramétrico reverso inválido.");
  const globalNodes = new Map(program.expressionGraph.nodes.map((node) => [node.id, node]));
  const functions = new Map(program.operationFunctions.map((entry) => [entry.functionId, entry]));
  const initialBuilder = new Gemma4ParametricRealBuilder(), initialMemo = new Map<string, string>();
  const initialEnvironment = new Map<string, string>();
  output.parameters.forEach((parameter, index) => {
    const value = parameters[parameter]; if (value === undefined || !Number.isSafeInteger(value) || value < 0) throw new Error(`${output.name}: parâmetro ${parameter} inválido.`);
    initialEnvironment.set(output.coordinate[index]!, initialBuilder.integerConstant(value));
  });
  const rebuildInitial = (id: string): string => {
    const substituted = initialEnvironment.get(id); if (substituted) return substituted;
    const known = initialMemo.get(id); if (known) return known;
    const result = rebuildParametric(required(globalNodes, id), initialBuilder, rebuildInitial, initialEnvironment); initialMemo.set(id, result); return result;
  };
  let root = rebuildInitial(output.root), graph = initialBuilder.build();
  const steps: Gemma4ParametricReverseStep[] = [];
  for (let step = 0; step < maximumSteps; step += 1) {
    const reachable = reachableIds(graph, root);
    const reachableCalls = graph.nodes.filter((node): node is Extract<Gemma4ParametricRealNode, { kind: "function-call" }> => node.kind === "function-call" && reachable.has(node.id));
    if (!reachableCalls.length) break;
    const target = reachableCalls.map((call) => functions.get(call.functionId)).filter((entry) => entry !== undefined).sort((left, right) => right.ordinal - left.ordinal)[0];
    if (!target) throw new Error("Composição reversa encontrou chamada sem função declarada.");
    const currentNodes = new Map(graph.nodes.map((node) => [node.id, node]));
    const builder = new Gemma4ParametricRealBuilder(), currentMemo = new Map<string, string>(); let expandedCalls = 0;
    const rebuildGlobal = (id: string, environment: ReadonlyMap<string, string>, memo = new Map<string, string>()): string => {
      const substituted = environment.get(id); if (substituted) return substituted;
      const known = memo.get(id); if (known) return known;
      const result = rebuildParametric(required(globalNodes, id), builder, (dependency) => rebuildGlobal(dependency, environment, memo), environment);
      memo.set(id, result); return result;
    };
    const rebuildCurrent = (id: string): string => {
      const known = currentMemo.get(id); if (known) return known;
      const node = required(currentNodes, id); let result: string;
      if (node.kind === "function-call" && node.functionId === target.functionId) {
        expandedCalls += 1;
        const arguments_ = node.arguments.map(rebuildCurrent);
        const environment = new Map(target.parameters.map((parameter, index) => [parameter.node, arguments_[index]!]));
        result = rebuildGlobal(target.root, environment);
      } else result = rebuildParametric(node, builder, rebuildCurrent, new Map());
      currentMemo.set(id, result); return result;
    };
    const nodesBefore = graph.nodes.length; root = rebuildCurrent(root); graph = builder.build();
    const remaining = graph.nodes.filter((node) => node.kind === "function-call").length;
    steps.push({ step, functionId: target.functionId, operationId: target.operationId, ordinal: target.ordinal, expandedCalls, nodesBefore, nodesAfterSimplification: graph.nodes.length, remainingFunctionCalls: remaining });
  }
  return { kind: "gemma4-parametric-reverse-algebraic-composition", schemaVersion: 1, output: { family: output.name, dimension: output.fixedDimension, parameters: { ...parameters } }, graph, root, steps, remainingFunctionCalls: [...new Set(graph.nodes.filter((node): node is Extract<Gemma4ParametricRealNode, { kind: "function-call" }> => node.kind === "function-call").map((node) => node.functionId))].sort() };
}

function rebuildParametric(node: Gemma4ParametricRealNode, builder: Gemma4ParametricRealBuilder, dependency: (id: string) => string, substitutions: ReadonlyMap<string, string>): string {
  const substituted = substitutions.get(node.id); if (substituted) return substituted;
  switch (node.kind) {
    case "integer-constant": return builder.integerConstant(node.value); case "integer-parameter": return builder.integerParameter(node.name); case "integer-bound-index": return builder.boundIndex(node.name);
    case "integer-add": case "integer-subtract": case "integer-multiply": case "integer-floor-divide": case "integer-modulo": return builder.integerBinary(node.kind, dependency(node.left), dependency(node.right));
    case "tensor-axis": return builder.tensorAxis(node.tensor, node.axis); case "stable-true-prefix-rank": return builder.stableTruePrefixRank(node.tensor, node.equals, dependency(node.batch), dependency(node.sequence));
    case "rational": return builder.rational(node.value.numerator, node.value.denominator); case "boolean": return builder.boolean(node.value); case "negative-infinity": return builder.negativeInfinity(); case "positive-infinity": return builder.positiveInfinity();
    case "input-element": return builder.inputElement(node.tensor, node.coordinates.map(dependency), node.valueType); case "learned-rational-element": return builder.learnedElement(node.tensor, node.storageDtype, node.coordinates.map(dependency), node.decoderId);
    case "function-call": return builder.functionCall(node.functionId, node.arguments.map(dependency)); case "add": return builder.add(...node.arguments.map(dependency)); case "multiply": return builder.multiply(...node.arguments.map(dependency));
    case "minimum": return builder.minimum(...node.arguments.map(dependency)); case "maximum": return builder.maximum(...node.arguments.map(dependency)); case "divide": return builder.divide(dependency(node.numerator), dependency(node.denominator));
    case "integer-power": return builder.integerPower(dependency(node.base), node.exponent); case "power": return builder.power(dependency(node.base), dependency(node.exponent)); case "unary-function": return builder.unaryFunction(node.function, dependency(node.argument));
    case "compare": return builder.compare(node.comparison, dependency(node.left), dependency(node.right)); case "select": return builder.select(dependency(node.condition), dependency(node.whenTrue), dependency(node.whenFalse));
    case "finite-sum": case "finite-maximum": return builder.finiteReduction(node.kind, node.index, dependency(node.startInclusive), dependency(node.endExclusive), dependency(node.body), node.predicate ? dependency(node.predicate) : undefined);
  }
}

function reachableIds(graph: Gemma4ParametricRealExpressionGraph, root: string): Set<string> {
  const nodes = new Map(graph.nodes.map((node) => [node.id, node])), visited = new Set<string>(), stack = [root];
  while (stack.length) { const id = stack.pop()!; if (visited.has(id)) continue; visited.add(id); const node = nodes.get(id); if (node) stack.push(...parametricNodeDependencies(node)); }
  return visited;
}
function required(map: ReadonlyMap<string, Gemma4ParametricRealNode>, id: string): Gemma4ParametricRealNode { const node = map.get(id); if (!node) throw new Error(`${id}: nó paramétrico ausente na composição reversa.`); return node; }
