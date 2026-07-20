import { createHash } from "node:crypto";

export interface Gemma4ExactRational {
  numerator: string;
  denominator: string;
}

export type Gemma4RealUnaryFunction =
  | "abs"
  | "exp"
  | "log1p"
  | "sin"
  | "cos"
  | "tan"
  | "tanh"
  | "sqrt"
  | "floor";

export type Gemma4RealComparison = "equal" | "not-equal" | "less" | "less-equal" | "greater" | "greater-equal";

export type Gemma4RealExpressionNode =
  | { id: string; kind: "rational"; value: Gemma4ExactRational; sourceBits?: { dtype: "BF16" | "F16" | "F32" | "F64"; hex: string } }
  | { id: string; kind: "boolean"; value: boolean }
  | { id: string; kind: "negative-infinity" }
  | { id: string; kind: "input"; name: string; coordinates: string[]; valueType: "real" | "integer" | "boolean" }
  | { id: string; kind: "add" | "multiply"; arguments: string[] }
  | { id: string; kind: "minimum" | "maximum"; arguments: string[] }
  | { id: string; kind: "divide"; numerator: string; denominator: string }
  | { id: string; kind: "modulo"; left: string; right: string }
  | { id: string; kind: "integer-power"; base: string; exponent: number }
  | { id: string; kind: "unary-function"; function: Gemma4RealUnaryFunction; argument: string }
  | { id: string; kind: "compare"; comparison: Gemma4RealComparison; left: string; right: string }
  | { id: string; kind: "select"; condition: string; whenTrue: string; whenFalse: string };

export interface Gemma4RealExpressionGraph {
  kind: "gemma4-exact-real-expression-graph";
  schemaVersion: 1;
  semantics: "gemma4-exact-real-simplified-v1";
  nodeOrder: "dependency-order";
  nodes: Gemma4RealExpressionNode[];
}

type NodePayload = Gemma4RealExpressionNode extends infer Node
  ? Node extends { id: string } ? Omit<Node, "id"> : never
  : never;

export class Gemma4RealExpressionBuilder {
  readonly #nodes = new Map<string, Gemma4RealExpressionNode>();

  rational(numerator: bigint | number | string, denominator: bigint | number | string = 1n): string {
    const value = normalizeGemma4ExactRational(BigInt(numerator), BigInt(denominator));
    return this.#intern({ kind: "rational", value });
  }

  rationalFromBits(dtype: "BF16" | "F16" | "F32" | "F64", hex: string): string {
    const value = gemma4ExactRationalFromIeeeBits(dtype, hex);
    return this.#intern({ kind: "rational", value, sourceBits: { dtype, hex: normalizeHex(hex, dtype) } });
  }

  boolean(value: boolean): string {
    return this.#intern({ kind: "boolean", value });
  }

  negativeInfinity(): string {
    return this.#intern({ kind: "negative-infinity" });
  }

  input(name: string, coordinates: readonly string[] = [], valueType: "real" | "integer" | "boolean" = "real"): string {
    if (!name) throw new Error("Input real Gemma 4 requer nome.");
    return this.#intern({ kind: "input", name, coordinates: [...coordinates], valueType });
  }

  add(...arguments_: readonly string[]): string {
    const flattened: string[] = [];
    for (const argument of arguments_) {
      const node = this.requiredNode(argument);
      if (node.kind === "add") flattened.push(...node.arguments);
      else flattened.push(argument);
    }
    let constantNumerator = 0n;
    let constantDenominator = 1n;
    const coefficients = new Map<string, Gemma4ExactRational>();
    for (const argument of flattened) {
      const { coefficient, term } = this.#coefficientAndTerm(argument);
      if (term === undefined) {
        const sum = addRationals(
          { numerator: constantNumerator.toString(), denominator: constantDenominator.toString() },
          coefficient,
        );
        constantNumerator = BigInt(sum.numerator);
        constantDenominator = BigInt(sum.denominator);
      } else {
        const current = coefficients.get(term) ?? normalizeGemma4ExactRational(0n, 1n);
        coefficients.set(term, addRationals(current, coefficient));
      }
    }
    const normalized: string[] = [];
    for (const [term, coefficient] of [...coefficients.entries()].sort(([left], [right]) => left.localeCompare(right, "en"))) {
      if (BigInt(coefficient.numerator) === 0n) continue;
      if (isRationalOne(coefficient)) normalized.push(term);
      else normalized.push(this.multiply(this.#rationalValue(coefficient), term));
    }
    const trigSquares = new Map<string, Partial<Record<"sin" | "cos", number>>>();
    normalized.forEach((id, index) => {
      const square = this.#trigonometricSquare(id);
      if (square) trigSquares.set(square.argument, { ...(trigSquares.get(square.argument) ?? {}), [square.function]: index });
    });
    const removed = new Set<number>();
    for (const pair of trigSquares.values()) if (pair.sin !== undefined && pair.cos !== undefined) {
      removed.add(pair.sin); removed.add(pair.cos); constantNumerator += constantDenominator;
    }
    if (removed.size) {
      const retained = normalized.filter((_id, index) => !removed.has(index));
      normalized.length = 0; normalized.push(...retained);
    }
    if (constantNumerator !== 0n) normalized.push(this.rational(constantNumerator, constantDenominator));
    normalized.sort((left, right) => left.localeCompare(right, "en"));
    if (normalized.length === 0) return this.rational(0n);
    if (normalized.length === 1) return normalized[0]!;
    return this.#intern({ kind: "add", arguments: normalized });
  }

  multiply(...arguments_: readonly string[]): string {
    const flattened: string[] = [];
    for (const argument of arguments_) {
      const node = this.requiredNode(argument);
      if (node.kind === "multiply") flattened.push(...node.arguments);
      else flattened.push(argument);
    }
    let coefficient = normalizeGemma4ExactRational(1n, 1n);
    const factors: string[] = [];
    for (const argument of flattened) {
      const node = this.requiredNode(argument);
      if (node.kind === "rational") coefficient = multiplyRationals(coefficient, node.value);
      else factors.push(argument);
    }
    if (BigInt(coefficient.numerator) === 0n) return this.rational(0n);
    factors.sort((left, right) => left.localeCompare(right, "en"));
    const compressed: string[] = [];
    for (let index = 0; index < factors.length;) {
      const factor = factors[index]!;
      let end = index + 1;
      while (end < factors.length && factors[end] === factor) end += 1;
      const count = end - index;
      compressed.push(count === 1 ? factor : this.integerPower(factor, count));
      index = end;
    }
    if (!isRationalOne(coefficient)) compressed.unshift(this.#rationalValue(coefficient));
    if (compressed.length === 0) return this.rational(1n);
    if (compressed.length === 1) return compressed[0]!;
    return this.#intern({ kind: "multiply", arguments: compressed });
  }

  minimum(...arguments_: readonly string[]): string {
    return this.#orderedExtremum("minimum", arguments_);
  }

  maximum(...arguments_: readonly string[]): string {
    return this.#orderedExtremum("maximum", arguments_);
  }

  subtract(left: string, right: string): string {
    return this.add(left, this.negate(right));
  }

  negate(argument: string): string {
    return this.multiply(this.rational(-1n), argument);
  }

  divide(numerator: string, denominator: string): string {
    const left = this.requiredNode(numerator), right = this.requiredNode(denominator);
    if (right.kind === "rational" && BigInt(right.value.numerator) === 0n) throw new Error("Divisão real Gemma 4 por zero.");
    if (left.kind === "rational" && right.kind === "rational") return this.#rationalValue(divideRationals(left.value, right.value));
    if (left.kind === "rational" && BigInt(left.value.numerator) === 0n) return left.id;
    if (right.kind === "rational" && isRationalOne(right.value)) return left.id;
    if (left.kind === "unary-function" && left.function === "sin" && right.kind === "unary-function" && right.function === "cos" && left.argument === right.argument) {
      return this.unaryFunction("tan", left.argument);
    }
    return this.#intern({ kind: "divide", numerator, denominator });
  }

  modulo(left: string, right: string): string {
    const leftNode = this.requiredNode(left), rightNode = this.requiredNode(right);
    if (leftNode.kind === "rational" && rightNode.kind === "rational" &&
      leftNode.value.denominator === "1" && rightNode.value.denominator === "1") {
      const divisor = BigInt(rightNode.value.numerator);
      if (divisor === 0n) throw new Error("Módulo real Gemma 4 por zero.");
      const value = BigInt(leftNode.value.numerator);
      return this.rational(((value % divisor) + divisor) % divisor);
    }
    return this.#intern({ kind: "modulo", left, right });
  }

  integerPower(base: string, exponent: number): string {
    if (!Number.isSafeInteger(exponent)) throw new Error(`Expoente inteiro real Gemma 4 inválido: ${exponent}.`);
    if (exponent === 0) return this.rational(1n);
    if (exponent === 1) return base;
    const node = this.requiredNode(base);
    if (node.kind === "rational") return this.#rationalValue(powerRational(node.value, exponent));
    return this.#intern({ kind: "integer-power", base, exponent });
  }

  unaryFunction(function_: Gemma4RealUnaryFunction, argument: string): string {
    const node = this.requiredNode(argument);
    if (node.kind === "rational" && node.value.numerator === "0") {
      if (function_ === "cos" || function_ === "exp") return this.rational(1n);
      if (function_ === "sin" || function_ === "tan" || function_ === "tanh" || function_ === "log1p" || function_ === "sqrt" || function_ === "abs" || function_ === "floor") return argument;
    }
    if (function_ === "abs") {
      if (node.kind === "rational") return this.rational(abs(BigInt(node.value.numerator)), node.value.denominator);
      if (node.kind === "unary-function" && node.function === "abs") return argument;
    }
    if (function_ === "sqrt" && node.kind === "integer-power" && node.exponent === 2) return this.unaryFunction("abs", node.base);
    const negative = this.#negativeTerm(argument);
    if (negative && function_ === "abs") return this.unaryFunction("abs", negative);
    if (negative && function_ === "cos") return this.unaryFunction("cos", negative);
    if (negative && (function_ === "sin" || function_ === "tan" || function_ === "tanh")) return this.negate(this.unaryFunction(function_, negative));
    return this.#intern({ kind: "unary-function", function: function_, argument });
  }

  compare(comparison: Gemma4RealComparison, left: string, right: string): string {
    this.requiredNode(left);
    this.requiredNode(right);
    return this.#intern({ kind: "compare", comparison, left, right });
  }

  select(condition: string, whenTrue: string, whenFalse: string): string {
    const conditionNode = this.requiredNode(condition);
    this.requiredNode(whenTrue);
    this.requiredNode(whenFalse);
    if (conditionNode.kind === "boolean") return conditionNode.value ? whenTrue : whenFalse;
    if (whenTrue === whenFalse) return whenTrue;
    return this.#intern({ kind: "select", condition, whenTrue, whenFalse });
  }

  requiredNode(id: string): Gemma4RealExpressionNode {
    const node = this.#nodes.get(id);
    if (!node) throw new Error(`Nó real Gemma 4 ausente: ${id}.`);
    return node;
  }

  build(): Gemma4RealExpressionGraph {
    return {
      kind: "gemma4-exact-real-expression-graph",
      schemaVersion: 1,
      semantics: "gemma4-exact-real-simplified-v1",
      nodeOrder: "dependency-order",
      nodes: [...this.#nodes.values()],
    };
  }

  #coefficientAndTerm(id: string): { coefficient: Gemma4ExactRational; term?: string } {
    const node = this.requiredNode(id);
    if (node.kind === "rational") return { coefficient: node.value };
    if (node.kind === "multiply") {
      const first = node.arguments[0] ? this.requiredNode(node.arguments[0]) : undefined;
      if (first?.kind === "rational") {
        const rest = node.arguments.slice(1);
        if (rest.length === 0) return { coefficient: first.value };
        const term = rest.length === 1 ? rest[0]! : this.#intern({ kind: "multiply", arguments: rest });
        return { coefficient: first.value, term };
      }
    }
    return { coefficient: normalizeGemma4ExactRational(1n, 1n), term: id };
  }

  #rationalValue(value: Gemma4ExactRational): string {
    return this.#intern({ kind: "rational", value: normalizeGemma4ExactRational(BigInt(value.numerator), BigInt(value.denominator)) });
  }

  #negativeTerm(id: string): string | undefined {
    const node = this.requiredNode(id);
    if (node.kind !== "multiply" || node.arguments.length < 2) return undefined;
    const coefficient = this.requiredNode(node.arguments[0]!);
    if (coefficient.kind !== "rational" || coefficient.value.numerator !== "-1" || coefficient.value.denominator !== "1") return undefined;
    const rest = node.arguments.slice(1);
    return rest.length === 1 ? rest[0]! : this.#intern({ kind: "multiply", arguments: rest });
  }

  #trigonometricSquare(id: string): { function: "sin" | "cos"; argument: string } | undefined {
    const node = this.requiredNode(id);
    if (node.kind !== "integer-power" || node.exponent !== 2) return undefined;
    const base = this.requiredNode(node.base);
    return base.kind === "unary-function" && (base.function === "sin" || base.function === "cos")
      ? { function: base.function, argument: base.argument }
      : undefined;
  }

  #orderedExtremum(kind: "minimum" | "maximum", arguments_: readonly string[]): string {
    const flattened: string[] = [];
    for (const argument of arguments_) {
      const node = this.requiredNode(argument);
      if (node.kind === kind) flattened.push(...node.arguments);
      else flattened.push(argument);
    }
    const unique = [...new Set(flattened)].sort((left, right) => left.localeCompare(right, "en"));
    if (unique.length === 0) throw new Error(`${kind} real Gemma 4 requer ao menos um argumento.`);
    if (unique.length === 1) return unique[0]!;
    return this.#intern({ kind, arguments: unique });
  }

  #intern(payload: NodePayload): string {
    const canonical = JSON.stringify(payload);
    const id = `sha256:${createHash("sha256").update(canonical).digest("hex")}`;
    if (!this.#nodes.has(id)) this.#nodes.set(id, { id, ...payload } as Gemma4RealExpressionNode);
    return id;
  }
}

export function normalizeGemma4ExactRational(numerator: bigint, denominator: bigint): Gemma4ExactRational {
  if (denominator === 0n) throw new Error("Racional Gemma 4 possui denominador zero.");
  if (numerator === 0n) return { numerator: "0", denominator: "1" };
  if (denominator < 0n) {
    numerator = -numerator;
    denominator = -denominator;
  }
  const divisor = gcd(abs(numerator), denominator);
  return { numerator: (numerator / divisor).toString(), denominator: (denominator / divisor).toString() };
}

export function gemma4ExactRationalFromIeeeBits(dtype: "BF16" | "F16" | "F32" | "F64", hex: string): Gemma4ExactRational {
  const normalized = normalizeHex(hex, dtype);
  const bits = BigInt(normalized);
  const exponentBits = dtype === "F64" ? 11n : dtype === "F16" ? 5n : 8n;
  const fractionBits = dtype === "F64" ? 52n : dtype === "F32" ? 23n : dtype === "F16" ? 10n : 7n;
  const totalBits = dtype === "F64" ? 64n : dtype === "F32" ? 32n : 16n;
  const exponentMask = (1n << exponentBits) - 1n;
  const fractionMask = (1n << fractionBits) - 1n;
  const exponent = (bits >> fractionBits) & exponentMask;
  const fraction = bits & fractionMask;
  if (exponent === exponentMask) throw new Error(`${dtype} ${normalized} não é finito e não pertence à semântica real Gemma 4.`);
  if (exponent === 0n && fraction === 0n) return normalizeGemma4ExactRational(0n, 1n);
  const sign = ((bits >> (totalBits - 1n)) & 1n) === 0n ? 1n : -1n;
  const bias = (1n << (exponentBits - 1n)) - 1n;
  const significand = exponent === 0n ? fraction : (1n << fractionBits) | fraction;
  const unbiased = exponent === 0n ? 1n - bias : exponent - bias;
  const shift = unbiased - fractionBits;
  return shift >= 0n
    ? normalizeGemma4ExactRational(sign * significand * (1n << shift), 1n)
    : normalizeGemma4ExactRational(sign * significand, 1n << -shift);
}

export function validateGemma4RealExpressionGraph(graph: Gemma4RealExpressionGraph): void {
  if (graph.kind !== "gemma4-exact-real-expression-graph" || graph.schemaVersion !== 1 ||
    graph.semantics !== "gemma4-exact-real-simplified-v1" || graph.nodeOrder !== "dependency-order") {
    throw new Error("Grafo real Gemma 4 possui cabeçalho inválido.");
  }
  const available = new Set<string>();
  for (const node of graph.nodes) {
    const { id, ...payload } = node;
    const expected = `sha256:${createHash("sha256").update(JSON.stringify(payload)).digest("hex")}`;
    if (id !== expected || available.has(id)) throw new Error(`Grafo real Gemma 4 possui nó não canônico ou duplicado: ${id}.`);
    for (const dependency of realNodeDependencies(node)) if (!available.has(dependency)) {
      throw new Error(`Grafo real Gemma 4 referencia dependência ausente ou futura ${dependency}.`);
    }
    if (node.kind === "rational") {
      const normalized = normalizeGemma4ExactRational(BigInt(node.value.numerator), BigInt(node.value.denominator));
      if (normalized.numerator !== node.value.numerator || normalized.denominator !== node.value.denominator) {
        throw new Error(`Grafo real Gemma 4 possui racional não normalizado em ${id}.`);
      }
      if (node.sourceBits) {
        const expectedValue = gemma4ExactRationalFromIeeeBits(node.sourceBits.dtype, node.sourceBits.hex);
        if (expectedValue.numerator !== node.value.numerator || expectedValue.denominator !== node.value.denominator) {
          throw new Error(`Grafo real Gemma 4 possui proveniência IEEE divergente em ${id}.`);
        }
      }
    }
    available.add(id);
  }
}

export function realNodeDependencies(node: Gemma4RealExpressionNode): string[] {
  switch (node.kind) {
    case "rational": case "boolean": case "negative-infinity": case "input": return [];
    case "add": case "multiply": case "minimum": case "maximum": return node.arguments;
    case "divide": return [node.numerator, node.denominator];
    case "modulo": return [node.left, node.right];
    case "integer-power": return [node.base];
    case "unary-function": return [node.argument];
    case "compare": return [node.left, node.right];
    case "select": return [node.condition, node.whenTrue, node.whenFalse];
  }
}

function addRationals(left: Gemma4ExactRational, right: Gemma4ExactRational): Gemma4ExactRational {
  return normalizeGemma4ExactRational(
    BigInt(left.numerator) * BigInt(right.denominator) + BigInt(right.numerator) * BigInt(left.denominator),
    BigInt(left.denominator) * BigInt(right.denominator),
  );
}

function multiplyRationals(left: Gemma4ExactRational, right: Gemma4ExactRational): Gemma4ExactRational {
  return normalizeGemma4ExactRational(
    BigInt(left.numerator) * BigInt(right.numerator),
    BigInt(left.denominator) * BigInt(right.denominator),
  );
}

function divideRationals(left: Gemma4ExactRational, right: Gemma4ExactRational): Gemma4ExactRational {
  if (BigInt(right.numerator) === 0n) throw new Error("Divisão racional Gemma 4 por zero.");
  return normalizeGemma4ExactRational(
    BigInt(left.numerator) * BigInt(right.denominator),
    BigInt(left.denominator) * BigInt(right.numerator),
  );
}

function powerRational(value: Gemma4ExactRational, exponent: number): Gemma4ExactRational {
  if (exponent < 0 && BigInt(value.numerator) === 0n) throw new Error("Potência racional Gemma 4 divide por zero.");
  const magnitude = BigInt(Math.abs(exponent));
  const numerator = BigInt(value.numerator) ** magnitude;
  const denominator = BigInt(value.denominator) ** magnitude;
  return exponent < 0
    ? normalizeGemma4ExactRational(denominator, numerator)
    : normalizeGemma4ExactRational(numerator, denominator);
}

function isRationalOne(value: Gemma4ExactRational): boolean {
  return value.numerator === "1" && value.denominator === "1";
}

function abs(value: bigint): bigint {
  return value < 0n ? -value : value;
}

function gcd(left: bigint, right: bigint): bigint {
  while (right !== 0n) [left, right] = [right, left % right];
  return left;
}

function normalizeHex(hex: string, dtype: "BF16" | "F16" | "F32" | "F64"): string {
  const width = dtype === "F64" ? 16 : dtype === "F32" ? 8 : 4;
  if (!/^0x[0-9a-fA-F]+$/.test(hex) || hex.length !== width + 2) throw new Error(`${dtype} possui bits inválidos: ${hex}.`);
  return `0x${hex.slice(2).toLowerCase()}`;
}
