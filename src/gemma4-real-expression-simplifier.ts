import type { Gemma4ExactRational, Gemma4RealExpressionGraph, Gemma4RealExpressionNode } from "./gemma4-real-expression.js";
import { Gemma4RealExpressionBuilder, realNodeDependencies, validateGemma4RealExpressionGraph } from "./gemma4-real-expression.js";

export interface Gemma4RealSimplificationResult {
  graph: Gemma4RealExpressionGraph;
  roots: Record<string, string>;
  statistics: {
    inputNodes: number;
    reachableInputNodes: number;
    outputNodes: number;
    factoredLinearCombinations: number;
  };
}

/**
 * Rebuilds only the closure of the requested roots through the canonical
 * builder. The rebuild performs rational folding, like-term collection,
 * associative flattening, deterministic commutative ordering and CSE.
 */
export function simplifyGemma4RealExpressionGraph(
  graph: Gemma4RealExpressionGraph,
  roots: Readonly<Record<string, string>>,
): Gemma4RealSimplificationResult {
  validateGemma4RealExpressionGraph(graph);
  const source = new Map(graph.nodes.map((node) => [node.id, node]));
  const builder = new Gemma4RealExpressionBuilder();
  const memo = new Map<string, string>();
  let factoredLinearCombinations = 0;
  const rebuild = (id: string): string => {
    const existing = memo.get(id);
    if (existing) return existing;
    const node = source.get(id);
    if (!node) throw new Error(`Raiz real Gemma 4 referencia nó ausente ${id}.`);
    const result = rebuildNode(node, builder, rebuild);
    const factored = factorLinearCombination(result, builder);
    if (factored !== result) factoredLinearCombinations += 1;
    memo.set(id, factored);
    return factored;
  };
  const simplifiedRoots = Object.fromEntries(Object.entries(roots).map(([name, id]) => [name, rebuild(id)]));
  const output = builder.build();
  const reachable = reachableNodeIds(output, Object.values(simplifiedRoots));
  const compact: Gemma4RealExpressionGraph = { ...output, nodes: output.nodes.filter((node) => reachable.has(node.id)) };
  validateGemma4RealExpressionGraph(compact);
  return {
    graph: compact,
    roots: simplifiedRoots,
    statistics: {
      inputNodes: graph.nodes.length,
      reachableInputNodes: memo.size,
      outputNodes: compact.nodes.length,
      factoredLinearCombinations,
    },
  };
}

function rebuildNode(
  node: Gemma4RealExpressionNode,
  builder: Gemma4RealExpressionBuilder,
  rebuild: (id: string) => string,
): string {
  switch (node.kind) {
    case "rational": return node.sourceBits
      ? builder.rationalFromBits(node.sourceBits.dtype, node.sourceBits.hex)
      : builder.rational(node.value.numerator, node.value.denominator);
    case "boolean": return builder.boolean(node.value);
    case "negative-infinity": return builder.negativeInfinity();
    case "input": return builder.input(node.name, node.coordinates, node.valueType);
    case "add": return builder.add(...node.arguments.map(rebuild));
    case "multiply": return builder.multiply(...node.arguments.map(rebuild));
    case "minimum": return builder.minimum(...node.arguments.map(rebuild));
    case "maximum": return builder.maximum(...node.arguments.map(rebuild));
    case "divide": return builder.divide(rebuild(node.numerator), rebuild(node.denominator));
    case "modulo": return builder.modulo(rebuild(node.left), rebuild(node.right));
    case "integer-power": return builder.integerPower(rebuild(node.base), node.exponent);
    case "unary-function": return builder.unaryFunction(node.function, rebuild(node.argument));
    case "compare": return builder.compare(node.comparison, rebuild(node.left), rebuild(node.right));
    case "select": return builder.select(rebuild(node.condition), rebuild(node.whenTrue), rebuild(node.whenFalse));
  }
}

/** Factors the magnitude of the first non-constant coefficient from a linear sum. */
function factorLinearCombination(root: string, builder: Gemma4RealExpressionBuilder): string {
  const node = builder.requiredNode(root);
  if (node.kind !== "add" || node.arguments.length < 2) return root;
  const first = coefficientAndTerm(builder, node.arguments[0]!);
  if (!first.term || isOneMagnitude(first.coefficient)) return root;
  const terms = node.arguments.map((argument) => coefficientAndTerm(builder, argument));
  if (terms.some((term) => term.term === undefined)) return root;
  const factor = magnitude(first.coefficient);
  const inner = terms.map(({ coefficient, term }) => builder.multiply(
    builder.rational(
      BigInt(coefficient.numerator) * BigInt(factor.denominator),
      BigInt(coefficient.denominator) * BigInt(factor.numerator),
    ),
    term!,
  ));
  const candidate = builder.multiply(builder.rational(factor.numerator, factor.denominator), builder.add(...inner));
  return expressionCost(builder, candidate) <= expressionCost(builder, root) ? candidate : root;
}

function coefficientAndTerm(
  builder: Gemma4RealExpressionBuilder,
  id: string,
): { coefficient: Gemma4ExactRational; term?: string } {
  const node = builder.requiredNode(id);
  if (node.kind === "rational") return { coefficient: node.value };
  if (node.kind === "multiply" && node.arguments.length >= 2) {
    const first = builder.requiredNode(node.arguments[0]!);
    if (first.kind === "rational") {
      const rest = node.arguments.slice(1);
      return { coefficient: first.value, term: rest.length === 1 ? rest[0]! : rest.reduce((left, right) => builder.multiply(left, right)) };
    }
  }
  return { coefficient: { numerator: "1", denominator: "1" }, term: id };
}

function expressionCost(builder: Gemma4RealExpressionBuilder, root: string): number {
  const seen = new Set<string>();
  const visit = (id: string): void => {
    if (seen.has(id)) return;
    seen.add(id);
    for (const dependency of realNodeDependencies(builder.requiredNode(id))) visit(dependency);
  };
  visit(root);
  return seen.size;
}

function reachableNodeIds(graph: Gemma4RealExpressionGraph, roots: readonly string[]): Set<string> {
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const reachable = new Set<string>();
  const visit = (id: string): void => {
    if (reachable.has(id)) return;
    const node = nodes.get(id);
    if (!node) throw new Error(`Closure real Gemma 4 possui nó ausente ${id}.`);
    for (const dependency of realNodeDependencies(node)) visit(dependency);
    reachable.add(id);
  };
  roots.forEach(visit);
  return reachable;
}

function magnitude(value: Gemma4ExactRational): Gemma4ExactRational {
  const numerator = BigInt(value.numerator);
  return { numerator: (numerator < 0n ? -numerator : numerator).toString(), denominator: value.denominator };
}

function isOneMagnitude(value: Gemma4ExactRational): boolean {
  const numerator = BigInt(value.numerator);
  return (numerator < 0n ? -numerator : numerator).toString() === value.denominator;
}
