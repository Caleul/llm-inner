import { createHash } from "node:crypto";
import type { Gemma4ExactRational, Gemma4RealComparison, Gemma4RealUnaryFunction } from "./gemma4-real-expression.js";
import { normalizeGemma4ExactRational } from "./gemma4-real-expression.js";

export type Gemma4ParametricRealNode =
  | { id: string; kind: "integer-constant"; value: number }
  | { id: string; kind: "integer-parameter" | "integer-bound-index"; name: string }
  | { id: string; kind: "integer-add" | "integer-subtract" | "integer-multiply" | "integer-floor-divide" | "integer-modulo"; left: string; right: string }
  | { id: string; kind: "tensor-axis"; tensor: string; axis: number }
  | { id: string; kind: "stable-true-prefix-rank"; tensor: string; equals: number; batch: string; sequence: string }
  | { id: string; kind: "rational"; value: Gemma4ExactRational }
  | { id: string; kind: "boolean"; value: boolean }
  | { id: string; kind: "negative-infinity" | "positive-infinity" }
  | { id: string; kind: "input-element"; tensor: string; coordinates: string[]; valueType: "real" | "integer" | "boolean" }
  | { id: string; kind: "learned-rational-element"; tensor: string; storageDtype: "BF16" | "F16" | "F32"; coordinates: string[]; decoderId: string }
  | { id: string; kind: "function-call"; functionId: string; arguments: string[] }
  | { id: string; kind: "add" | "multiply" | "minimum" | "maximum"; arguments: string[] }
  | { id: string; kind: "divide"; numerator: string; denominator: string }
  | { id: string; kind: "integer-power"; base: string; exponent: number }
  | { id: string; kind: "power"; base: string; exponent: string }
  | { id: string; kind: "unary-function"; function: Gemma4RealUnaryFunction; argument: string }
  | { id: string; kind: "compare"; comparison: Gemma4RealComparison; left: string; right: string }
  | { id: string; kind: "select"; condition: string; whenTrue: string; whenFalse: string }
  | { id: string; kind: "finite-sum" | "finite-maximum"; index: string; startInclusive: string; endExclusive: string; body: string; predicate?: string };

export interface Gemma4ParametricRealExpressionGraph {
  kind: "gemma4-parametric-exact-real-expression-graph";
  schemaVersion: 1;
  semantics: "gemma4-exact-real-simplified-v1";
  nodeOrder: "dependency-order";
  nodes: Gemma4ParametricRealNode[];
}

type ParametricPayload = Gemma4ParametricRealNode extends infer Node
  ? Node extends { id: string } ? Omit<Node, "id"> : never
  : never;

/** Content-addressed builder used by the full-dimension symbolic compiler. */
export class Gemma4ParametricRealBuilder {
  readonly #nodes = new Map<string, Gemma4ParametricRealNode>();

  integerConstant(value: number): string {
    if (!Number.isSafeInteger(value)) throw new Error(`Inteiro paramétrico Gemma 4 inválido: ${value}.`);
    return this.#intern({ kind: "integer-constant", value });
  }

  integerParameter(name: string): string {
    if (!name) throw new Error("Parâmetro inteiro Gemma 4 requer nome.");
    return this.#intern({ kind: "integer-parameter", name });
  }

  boundIndex(name: string): string {
    if (!name) throw new Error("Índice ligado Gemma 4 requer nome.");
    return this.#intern({ kind: "integer-bound-index", name });
  }

  integerBinary(kind: "integer-add" | "integer-subtract" | "integer-multiply" | "integer-floor-divide" | "integer-modulo", left: string, right: string): string {
    const leftNode = this.requiredNode(left), rightNode = this.requiredNode(right);
    if (leftNode.kind === "integer-constant" && rightNode.kind === "integer-constant") {
      switch (kind) {
        case "integer-add": return this.integerConstant(leftNode.value + rightNode.value);
        case "integer-subtract": return this.integerConstant(leftNode.value - rightNode.value);
        case "integer-multiply": return this.integerConstant(leftNode.value * rightNode.value);
        case "integer-floor-divide":
          if (rightNode.value === 0) throw new Error("Floor-divide paramétrico Gemma 4 por zero.");
          return this.integerConstant(Math.floor(leftNode.value / rightNode.value));
        case "integer-modulo":
          if (rightNode.value <= 0) throw new Error("Módulo paramétrico Gemma 4 requer divisor positivo.");
          return this.integerConstant(((leftNode.value % rightNode.value) + rightNode.value) % rightNode.value);
      }
    }
    return this.#intern({ kind, left, right });
  }

  tensorAxis(tensor: string, axis: number): string {
    if (!tensor || !Number.isSafeInteger(axis) || axis < 0) throw new Error("Tensor-axis paramétrico Gemma 4 inválido.");
    return this.#intern({ kind: "tensor-axis", tensor, axis });
  }

  stableTruePrefixRank(tensor: string, equals: number, batch: string, sequence: string): string {
    this.requiredNode(batch); this.requiredNode(sequence);
    return this.#intern({ kind: "stable-true-prefix-rank", tensor, equals, batch, sequence });
  }

  rational(numerator: bigint | string | number, denominator: bigint | string | number = 1n): string {
    return this.#intern({ kind: "rational", value: normalizeGemma4ExactRational(BigInt(numerator), BigInt(denominator)) });
  }

  boolean(value: boolean): string { return this.#intern({ kind: "boolean", value }); }
  negativeInfinity(): string { return this.#intern({ kind: "negative-infinity" }); }
  positiveInfinity(): string { return this.#intern({ kind: "positive-infinity" }); }

  inputElement(tensor: string, coordinates: readonly string[], valueType: "real" | "integer" | "boolean" = "real"): string {
    coordinates.forEach((coordinate) => this.requiredNode(coordinate));
    return this.#intern({ kind: "input-element", tensor, coordinates: [...coordinates], valueType });
  }

  learnedElement(tensor: string, storageDtype: "BF16" | "F16" | "F32", coordinates: readonly string[], decoderId: string): string {
    coordinates.forEach((coordinate) => this.requiredNode(coordinate));
    return this.#intern({ kind: "learned-rational-element", tensor, storageDtype, coordinates: [...coordinates], decoderId });
  }

  functionCall(functionId: string, arguments_: readonly string[]): string {
    if (!functionId) throw new Error("Chamada paramétrica Gemma 4 requer functionId.");
    arguments_.forEach((argument) => this.requiredNode(argument));
    return this.#intern({ kind: "function-call", functionId, arguments: [...arguments_] });
  }

  add(...arguments_: readonly string[]): string { return this.#associative("add", arguments_); }
  multiply(...arguments_: readonly string[]): string { return this.#associative("multiply", arguments_); }
  minimum(...arguments_: readonly string[]): string { return this.#associative("minimum", arguments_); }
  maximum(...arguments_: readonly string[]): string { return this.#associative("maximum", arguments_); }

  divide(numerator: string, denominator: string): string {
    this.requiredNode(numerator); this.requiredNode(denominator);
    return this.#intern({ kind: "divide", numerator, denominator });
  }

  integerPower(base: string, exponent: number): string {
    this.requiredNode(base);
    if (!Number.isSafeInteger(exponent)) throw new Error("Expoente paramétrico Gemma 4 inválido.");
    if (exponent === 0) return this.rational(1n);
    if (exponent === 1) return base;
    return this.#intern({ kind: "integer-power", base, exponent });
  }

  power(base: string, exponent: string): string {
    this.requiredNode(base); this.requiredNode(exponent);
    const exponentNode = this.requiredNode(exponent);
    if (exponentNode.kind === "rational" && exponentNode.value.denominator === "1") {
      const value = Number(exponentNode.value.numerator);
      if (Number.isSafeInteger(value)) return this.integerPower(base, value);
    }
    return this.#intern({ kind: "power", base, exponent });
  }

  unaryFunction(function_: Gemma4RealUnaryFunction, argument: string): string {
    this.requiredNode(argument);
    return this.#intern({ kind: "unary-function", function: function_, argument });
  }

  compare(comparison: Gemma4RealComparison, left: string, right: string): string {
    this.requiredNode(left); this.requiredNode(right);
    return this.#intern({ kind: "compare", comparison, left, right });
  }

  select(condition: string, whenTrue: string, whenFalse: string): string {
    const conditionNode = this.requiredNode(condition);
    this.requiredNode(whenTrue); this.requiredNode(whenFalse);
    if (conditionNode.kind === "boolean") return conditionNode.value ? whenTrue : whenFalse;
    if (whenTrue === whenFalse) return whenTrue;
    return this.#intern({ kind: "select", condition, whenTrue, whenFalse });
  }

  finiteReduction(kind: "finite-sum" | "finite-maximum", index: string, startInclusive: string, endExclusive: string, body: string, predicate?: string): string {
    this.requiredNode(startInclusive); this.requiredNode(endExclusive); this.requiredNode(body);
    if (predicate) this.requiredNode(predicate);
    return this.#intern({ kind, index, startInclusive, endExclusive, body, ...(predicate ? { predicate } : {}) });
  }

  requiredNode(id: string): Gemma4ParametricRealNode {
    const node = this.#nodes.get(id);
    if (!node) throw new Error(`Nó paramétrico Gemma 4 ausente: ${id}.`);
    return node;
  }

  build(): Gemma4ParametricRealExpressionGraph {
    return { kind: "gemma4-parametric-exact-real-expression-graph", schemaVersion: 1, semantics: "gemma4-exact-real-simplified-v1", nodeOrder: "dependency-order", nodes: [...this.#nodes.values()] };
  }

  #associative(kind: "add" | "multiply" | "minimum" | "maximum", arguments_: readonly string[]): string {
    const flattened: string[] = [];
    for (const id of arguments_) {
      const node = this.requiredNode(id);
      if (node.kind === kind) flattened.push(...node.arguments);
      else flattened.push(id);
    }
    const argumentsSorted = (kind === "minimum" || kind === "maximum" ? [...new Set(flattened)] : flattened)
      .sort((left, right) => left.localeCompare(right, "en"));
    if (argumentsSorted.length === 0) {
      if (kind === "add") return this.rational(0n);
      if (kind === "multiply") return this.rational(1n);
      throw new Error(`${kind} paramétrico Gemma 4 requer argumento.`);
    }
    if (argumentsSorted.length === 1) return argumentsSorted[0]!;
    return this.#intern({ kind, arguments: argumentsSorted });
  }

  #intern(payload: ParametricPayload): string {
    const id = `sha256:${createHash("sha256").update(JSON.stringify(payload)).digest("hex")}`;
    if (!this.#nodes.has(id)) this.#nodes.set(id, { id, ...payload } as Gemma4ParametricRealNode);
    return id;
  }
}

export function parametricNodeDependencies(node: Gemma4ParametricRealNode): string[] {
  switch (node.kind) {
    case "integer-constant": case "integer-parameter": case "integer-bound-index": case "tensor-axis":
    case "rational": case "boolean": case "negative-infinity": case "positive-infinity": return [];
    case "integer-add": case "integer-subtract": case "integer-multiply": case "integer-floor-divide": case "integer-modulo": return [node.left, node.right];
    case "stable-true-prefix-rank": return [node.batch, node.sequence];
    case "input-element": case "learned-rational-element": return node.coordinates;
    case "function-call": return node.arguments;
    case "add": case "multiply": case "minimum": case "maximum": return node.arguments;
    case "divide": return [node.numerator, node.denominator];
    case "integer-power": return [node.base];
    case "power": return [node.base, node.exponent];
    case "unary-function": return [node.argument];
    case "compare": return [node.left, node.right];
    case "select": return [node.condition, node.whenTrue, node.whenFalse];
    case "finite-sum": case "finite-maximum": return [node.startInclusive, node.endExclusive, node.body, ...(node.predicate ? [node.predicate] : [])];
  }
}

export function validateGemma4ParametricRealExpressionGraph(graph: Gemma4ParametricRealExpressionGraph): void {
  if (graph.kind !== "gemma4-parametric-exact-real-expression-graph" || graph.schemaVersion !== 1 || graph.semantics !== "gemma4-exact-real-simplified-v1" || graph.nodeOrder !== "dependency-order") {
    throw new Error("Grafo paramétrico real Gemma 4 possui cabeçalho inválido.");
  }
  const available = new Set<string>();
  for (const node of graph.nodes) {
    const { id, ...payload } = node;
    const expected = `sha256:${createHash("sha256").update(JSON.stringify(payload)).digest("hex")}`;
    if (id !== expected || available.has(id)) throw new Error(`Nó paramétrico real Gemma 4 não canônico: ${id}.`);
    for (const dependency of parametricNodeDependencies(node)) if (!available.has(dependency)) throw new Error(`${id}: dependência paramétrica ausente ${dependency}.`);
    available.add(id);
  }
}
