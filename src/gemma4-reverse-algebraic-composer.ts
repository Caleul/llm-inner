import {
  Gemma4RealExpressionBuilder,
  type Gemma4RealExpressionGraph,
  type Gemma4RealExpressionNode,
  validateGemma4RealExpressionGraph,
} from "./gemma4-real-expression.js";
import { simplifyGemma4RealExpressionGraph } from "./gemma4-real-expression-simplifier.js";

export interface Gemma4ReverseCompositionStage {
  id: string;
  /** Input occurrence in the current expression that this predecessor defines. */
  target: { name: string; coordinates?: string[] };
  replacementGraph: Gemma4RealExpressionGraph;
  replacementRoot: string;
}

export interface Gemma4ReverseCompositionStep {
  stageId: string;
  target: string;
  substitutedOccurrences: number;
  nodesBefore: number;
  nodesAfterSubstitution: number;
  nodesAfterSimplification: number;
  nodesRemovedBySimplification: number;
  factoredLinearCombinations: number;
}

export interface Gemma4ReverseCompositionResult {
  graph: Gemma4RealExpressionGraph;
  root: string;
  steps: Gemma4ReverseCompositionStep[];
  freeInputs: Array<{ name: string; coordinates: string[] }>;
}

/**
 * Walks from an output expression toward its inputs. Each predecessor is
 * physically substituted and the exact-real graph is normalized immediately,
 * before the next predecessor is introduced.
 */
export function composeGemma4ReverseAlgebra(
  outputGraph: Gemma4RealExpressionGraph,
  outputRoot: string,
  stages: readonly Gemma4ReverseCompositionStage[],
): Gemma4ReverseCompositionResult {
  validateGemma4RealExpressionGraph(outputGraph);
  let initial = simplifyGemma4RealExpressionGraph(outputGraph, { output: outputRoot });
  let graph = initial.graph, root = initial.roots.output!;
  const steps: Gemma4ReverseCompositionStep[] = [];
  for (const stage of stages) {
    validateGemma4RealExpressionGraph(stage.replacementGraph);
    const replacementNodes = new Map(stage.replacementGraph.nodes.map((node) => [node.id, node]));
    if (!replacementNodes.has(stage.replacementRoot)) throw new Error(`${stage.id}: raiz de predecessor ausente.`);
    const currentNodes = new Map(graph.nodes.map((node) => [node.id, node]));
    const builder = new Gemma4RealExpressionBuilder();
    const currentMemo = new Map<string, string>(), replacementMemo = new Map<string, string>();
    let substitutedOccurrences = 0;
    const rebuildReplacement = (id: string): string => {
      const known = replacementMemo.get(id); if (known) return known;
      const result = rebuild(required(replacementNodes, id), builder, rebuildReplacement); replacementMemo.set(id, result); return result;
    };
    const rebuildCurrent = (id: string): string => {
      const known = currentMemo.get(id); if (known) return known;
      const node = required(currentNodes, id);
      let result: string;
      if (node.kind === "input" && matchesTarget(node, stage.target)) {
        substitutedOccurrences += 1;
        result = rebuildReplacement(stage.replacementRoot);
      } else result = rebuild(node, builder, rebuildCurrent);
      currentMemo.set(id, result); return result;
    };
    const substitutedRoot = rebuildCurrent(root), substitutedGraph = builder.build();
    if (substitutedOccurrences === 0) throw new Error(`${stage.id}: target ${formatTarget(stage.target)} não aparece na expressão corrente.`);
    const simplified = simplifyGemma4RealExpressionGraph(substitutedGraph, { output: substitutedRoot });
    steps.push({
      stageId: stage.id, target: formatTarget(stage.target), substitutedOccurrences,
      nodesBefore: graph.nodes.length, nodesAfterSubstitution: substitutedGraph.nodes.length,
      nodesAfterSimplification: simplified.graph.nodes.length,
      nodesRemovedBySimplification: substitutedGraph.nodes.length - simplified.graph.nodes.length,
      factoredLinearCombinations: simplified.statistics.factoredLinearCombinations,
    });
    graph = simplified.graph; root = simplified.roots.output!;
  }
  return { graph, root, steps, freeInputs: graph.nodes.filter((node): node is Extract<Gemma4RealExpressionNode, { kind: "input" }> => node.kind === "input").map((node) => ({ name: node.name, coordinates: node.coordinates })) };
}

function rebuild(node: Gemma4RealExpressionNode, builder: Gemma4RealExpressionBuilder, dependency: (id: string) => string): string {
  switch (node.kind) {
    case "rational": return node.sourceBits ? builder.rationalFromBits(node.sourceBits.dtype, node.sourceBits.hex) : builder.rational(node.value.numerator, node.value.denominator);
    case "boolean": return builder.boolean(node.value); case "negative-infinity": return builder.negativeInfinity();
    case "input": return builder.input(node.name, node.coordinates, node.valueType);
    case "add": return builder.add(...node.arguments.map(dependency)); case "multiply": return builder.multiply(...node.arguments.map(dependency));
    case "minimum": return builder.minimum(...node.arguments.map(dependency)); case "maximum": return builder.maximum(...node.arguments.map(dependency));
    case "divide": return builder.divide(dependency(node.numerator), dependency(node.denominator));
    case "modulo": return builder.modulo(dependency(node.left), dependency(node.right));
    case "integer-power": return builder.integerPower(dependency(node.base), node.exponent);
    case "unary-function": return builder.unaryFunction(node.function, dependency(node.argument));
    case "compare": return builder.compare(node.comparison, dependency(node.left), dependency(node.right));
    case "select": return builder.select(dependency(node.condition), dependency(node.whenTrue), dependency(node.whenFalse));
  }
}

function matchesTarget(node: Extract<Gemma4RealExpressionNode, { kind: "input" }>, target: Gemma4ReverseCompositionStage["target"]): boolean {
  return node.name === target.name && (target.coordinates === undefined || target.coordinates.length === node.coordinates.length && target.coordinates.every((coordinate, index) => coordinate === node.coordinates[index]));
}
function formatTarget(target: Gemma4ReverseCompositionStage["target"]): string { return target.coordinates ? `${target.name}[${target.coordinates.join(",")}]` : target.name; }
function required(map: ReadonlyMap<string, Gemma4RealExpressionNode>, id: string): Gemma4RealExpressionNode { const node = map.get(id); if (!node) throw new Error(`${id}: nó ausente no compositor reverso.`); return node; }
