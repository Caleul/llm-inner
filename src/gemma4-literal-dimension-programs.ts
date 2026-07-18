import { isDeepStrictEqual } from "node:util";

export type Gemma4LiteralDimensionExpression =
  | { kind: "constant"; value: number }
  | { kind: "dimension"; name: string }
  | { kind: "tensor-axis"; tensor: string; axis: number; whenAbsent?: 0 }
  | {
      kind: "cache-key-length";
      input: "past_key_values";
      sequenceAxis: 2;
      whenAbsent: 0;
      consistency: "all-present-producer-entries-equal";
    }
  | { kind: "true-count"; tensor: string; order: "row-major" }
  | {
      kind: "add" | "multiply" | "ceil-divide" | "exact-divide";
      left: Gemma4LiteralDimensionExpression;
      right: Gemma4LiteralDimensionExpression;
    };

export type Gemma4LiteralDimensionPrograms = Record<string, Gemma4LiteralDimensionExpression>;

export interface Gemma4LiteralDimensionExpressionLanguage {
  id: "gemma4-safe-integer-dimension-expression-v1";
  schemaVersion: 1;
  resultType: "non-negative-safe-integer";
  evaluationOrder: "dependency-order-then-depth-first-left-to-right";
  bindings: {
    tensorAxis: string;
    optionalTensorAxis: string;
    cacheKeyLength: string;
    trueCount: string;
    dimension: string;
  };
  arithmetic: {
    add: string;
    multiply: string;
    ceilDivide: string;
    exactDivide: string;
  };
  shapeExpressions: {
    terms: string;
    multiplication: string;
  };
  invalidOperation: string;
}

export interface Gemma4LiteralDimensionEnvironment {
  tensorShapes: Readonly<Record<string, readonly number[]>>;
  booleanTensors?: Readonly<Record<string, readonly boolean[]>>;
  /** One BHSD key sequence length for every producer-owned cache entry. */
  cacheKeyLengths?: readonly number[];
}

export function gemma4LiteralDimensionExpressionLanguage(): Gemma4LiteralDimensionExpressionLanguage {
  return {
    id: "gemma4-safe-integer-dimension-expression-v1",
    schemaVersion: 1,
    resultType: "non-negative-safe-integer",
    evaluationOrder: "dependency-order-then-depth-first-left-to-right",
    bindings: {
      tensorAxis: "read the zero-based axis from the named row-major tensor shape",
      optionalTensorAxis: "use whenAbsent only when the named optional tensor is absent; a present tensor must contain the declared axis",
      cacheKeyLength: "read BHSD sequence axis 2 from every present producer-owned past_key_values key and require one common length; use zero when no cache is supplied",
      trueCount: "scan every BOOL element of the named tensor in row-major order and count true values exactly",
      dimension: "evaluate the named dimension program exactly once before its consumer",
    },
    arithmetic: {
      add: "exact safe-integer addition",
      multiply: "exact safe-integer multiplication",
      ceilDivide: "ceil(left/right), requiring right > 0",
      exactDivide: "left/right, requiring right > 0 and left modulo right == 0",
    },
    shapeExpressions: {
      terms: "each shape term is either a non-negative decimal safe integer or one named dimension program",
      multiplication: "asterisk-separated terms are multiplied exactly from left to right with safe-integer overflow checks",
    },
    invalidOperation: "fail closed on a missing tensor, missing BOOL payload, cycle, unknown dimension, non-integer, negative value, overflow, zero divisor, or inexact division",
  };
}

export function validateGemma4LiteralDimensionExpressionLanguage(
  language: Gemma4LiteralDimensionExpressionLanguage,
): void {
  if (!isDeepStrictEqual(language, gemma4LiteralDimensionExpressionLanguage())) {
    throw new Error("Programa Gemma 4 possui linguagem de dimensões ausente ou divergente.");
  }
}

export function validateGemma4LiteralDimensionPrograms(
  programs: Gemma4LiteralDimensionPrograms,
  externalNames: ReadonlySet<string> = new Set(),
): void {
  const names = new Set([...externalNames, ...Object.keys(programs)]);
  if (Object.keys(programs).length === 0) throw new Error("Programa Gemma 4 não declara dimensões executáveis.");
  for (const [name, expression] of Object.entries(programs)) {
    if (!/^[A-Z][A-Z0-9_]*$/.test(name)) throw new Error(`Dimensão Gemma 4 possui nome inválido: ${name}.`);
    validateExpression(expression, names, name);
  }
  detectCycles(programs);
}

export function evaluateGemma4LiteralDimensionPrograms(
  programs: Gemma4LiteralDimensionPrograms,
  environment: Gemma4LiteralDimensionEnvironment,
  externalValues: Readonly<Record<string, number>> = {},
): Record<string, number> {
  validateGemma4LiteralDimensionPrograms(programs, new Set(Object.keys(externalValues)));
  for (const [name, value] of Object.entries(externalValues)) requiredSafeInteger(value, name);
  const { values, evaluateNamed } = dimensionEvaluator(programs, environment, externalValues);
  for (const name of Object.keys(programs)) evaluateNamed(name);
  return values;
}

/** Evaluates one dependency closure without requiring tensors from unused modalities. */
export function evaluateGemma4LiteralDimensionProgram(
  name: string,
  programs: Gemma4LiteralDimensionPrograms,
  environment: Gemma4LiteralDimensionEnvironment,
  externalValues: Readonly<Record<string, number>> = {},
): number {
  validateGemma4LiteralDimensionPrograms(programs, new Set(Object.keys(externalValues)));
  for (const [externalName, value] of Object.entries(externalValues)) requiredSafeInteger(value, externalName);
  return dimensionEvaluator(programs, environment, externalValues).evaluateNamed(name);
}

export function evaluateGemma4LiteralShapeExpression(
  expression: string,
  dimensions: Readonly<Record<string, number>>,
): number {
  if (!/^(?:\d+|[A-Z][A-Z0-9_]*)(?:\*(?:\d+|[A-Z][A-Z0-9_]*))*$/.test(expression)) {
    throw new Error(`Expressão de shape Gemma 4 inválida: ${expression}.`);
  }
  let result = 1;
  for (const term of expression.split("*")) {
    const value = /^\d+$/.test(term) ? Number(term) : dimensions[term];
    if (value === undefined) throw new Error(`Expressão de shape Gemma 4 requer dimensão ausente: ${term}.`);
    result = requiredSafeInteger(result * requiredSafeInteger(value, term), expression);
  }
  return result;
}

function dimensionEvaluator(
  programs: Gemma4LiteralDimensionPrograms,
  environment: Gemma4LiteralDimensionEnvironment,
  externalValues: Readonly<Record<string, number>>,
): { values: Record<string, number>; evaluateNamed: (name: string) => number } {
  const values: Record<string, number> = { ...externalValues };
  const active = new Set<string>();
  const evaluateNamed = (name: string): number => {
    if (Object.hasOwn(values, name)) return requiredSafeInteger(values[name], name);
    const expression = programs[name];
    if (!expression) throw new Error(`Dimensão Gemma 4 não declarada: ${name}.`);
    if (active.has(name)) throw new Error(`Programa Gemma 4 possui ciclo de dimensão em ${name}.`);
    active.add(name);
    const value = evaluateExpression(expression, environment, evaluateNamed);
    active.delete(name);
    values[name] = requiredSafeInteger(value, name);
    return values[name]!;
  };
  return { values, evaluateNamed };
}

function evaluateExpression(
  expression: Gemma4LiteralDimensionExpression,
  environment: Gemma4LiteralDimensionEnvironment,
  evaluateNamed: (name: string) => number,
): number {
  switch (expression.kind) {
    case "constant": return requiredSafeInteger(expression.value, "constante de dimensão");
    case "dimension": return evaluateNamed(expression.name);
    case "tensor-axis": {
      const shape = environment.tensorShapes[expression.tensor];
      if (!shape) {
        if (expression.whenAbsent === 0) return 0;
        throw new Error(`Dimensão Gemma 4 requer shape ausente: ${expression.tensor}.`);
      }
      const value = shape[expression.axis];
      if (value === undefined) throw new Error(`${expression.tensor}: eixo de dimensão ausente ${expression.axis}.`);
      return requiredSafeInteger(value, `${expression.tensor}.shape[${expression.axis}]`);
    }
    case "cache-key-length": {
      const lengths = environment.cacheKeyLengths;
      if (lengths === undefined) return expression.whenAbsent;
      if (lengths.length === 0) throw new Error("past_key_values presente não contém cache produtor.");
      const expected = requiredSafeInteger(lengths[0], "past_key_values key length");
      for (const length of lengths) if (requiredSafeInteger(length, "past_key_values key length") !== expected) {
        throw new Error("past_key_values possui comprimentos de key contraditórios.");
      }
      return expected;
    }
    case "true-count": {
      const values = environment.booleanTensors?.[expression.tensor];
      if (!values) throw new Error(`Dimensão Gemma 4 requer tensor BOOL ausente: ${expression.tensor}.`);
      let total = 0;
      for (const value of values) {
        if (typeof value !== "boolean") throw new Error(`${expression.tensor}: dimensão true-count requer BOOL.`);
        if (value) total = requiredSafeInteger(total + 1, `${expression.tensor}.trueCount`);
      }
      return total;
    }
    case "add": case "multiply": case "ceil-divide": case "exact-divide": {
      const left = evaluateExpression(expression.left, environment, evaluateNamed);
      const right = evaluateExpression(expression.right, environment, evaluateNamed);
      if (expression.kind === "add") return requiredSafeInteger(left + right, "soma de dimensão");
      if (expression.kind === "multiply") return requiredSafeInteger(left * right, "produto de dimensão");
      if (right <= 0) throw new Error("Divisão de dimensão Gemma 4 requer divisor positivo.");
      if (expression.kind === "exact-divide" && left % right !== 0) {
        throw new Error(`Divisão de dimensão Gemma 4 não é exata: ${left}/${right}.`);
      }
      return requiredSafeInteger(expression.kind === "ceil-divide" ? Math.ceil(left / right) : left / right, "divisão de dimensão");
    }
  }
}

function validateExpression(expression: Gemma4LiteralDimensionExpression, names: ReadonlySet<string>, owner: string): void {
  if (!expression || typeof expression !== "object") throw new Error(`${owner}: expressão de dimensão inválida.`);
  switch (expression.kind) {
    case "constant": requiredSafeInteger(expression.value, owner); return;
    case "dimension":
      if (!names.has(expression.name)) throw new Error(`${owner}: referência de dimensão ausente ${expression.name}.`);
      return;
    case "tensor-axis":
      if (!expression.tensor || !Number.isSafeInteger(expression.axis) || expression.axis < 0 ||
        (expression.whenAbsent !== undefined && expression.whenAbsent !== 0)) {
        throw new Error(`${owner}: leitura de eixo de tensor inválida.`);
      }
      return;
    case "cache-key-length":
      if (expression.input !== "past_key_values" || expression.sequenceAxis !== 2 || expression.whenAbsent !== 0 ||
        expression.consistency !== "all-present-producer-entries-equal") {
        throw new Error(`${owner}: leitura de cache KV inválida.`);
      }
      return;
    case "true-count":
      if (!expression.tensor || expression.order !== "row-major") throw new Error(`${owner}: true-count inválido.`);
      return;
    case "add": case "multiply": case "ceil-divide": case "exact-divide":
      validateExpression(expression.left, names, owner);
      validateExpression(expression.right, names, owner);
      return;
    default: {
      const unsupported: never = expression;
      throw new Error(`${owner}: expressão de dimensão desconhecida ${JSON.stringify(unsupported)}.`);
    }
  }
}

function detectCycles(programs: Gemma4LiteralDimensionPrograms): void {
  const complete = new Set<string>(), active = new Set<string>();
  const visit = (name: string): void => {
    if (complete.has(name)) return;
    if (active.has(name)) throw new Error(`Programa Gemma 4 possui ciclo de dimensão em ${name}.`);
    active.add(name);
    for (const dependency of dimensionDependencies(programs[name]!)) if (programs[dependency]) visit(dependency);
    active.delete(name);
    complete.add(name);
  };
  for (const name of Object.keys(programs)) visit(name);
}

function dimensionDependencies(expression: Gemma4LiteralDimensionExpression): string[] {
  if (expression.kind === "dimension") return [expression.name];
  if (expression.kind === "add" || expression.kind === "multiply" || expression.kind === "ceil-divide" || expression.kind === "exact-divide") {
    return [...dimensionDependencies(expression.left), ...dimensionDependencies(expression.right)];
  }
  return [];
}

function requiredSafeInteger(value: unknown, name: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) throw new Error(`${name}: dimensão deve ser inteiro seguro não negativo.`);
  return value as number;
}
