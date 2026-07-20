import type { Gemma4LiteralScalarExpression } from "./gemma4-literal-scalar-statement-programs.js";
import { Gemma4RealExpressionBuilder, type Gemma4RealUnaryFunction } from "./gemma4-real-expression.js";

export interface Gemma4RealStatementLoweringContext {
  builder: Gemma4RealExpressionBuilder;
  integerBindings: Readonly<Record<string, number>>;
  valueBindings?: Readonly<Record<string, string>>;
  resolveTensorElement: (name: string, coordinates: readonly number[]) => string;
  resolveTensorAxis?: (name: string, axis: number) => string;
  resolveLearnedElement: (role: string, coordinates: readonly number[]) => string;
  resolveInteger?: (expression: Gemma4LiteralScalarExpression, bindings: Readonly<Record<string, number>>) => number;
}

/**
 * Removes every IEEE cast from one already parsed scalar AST and lowers the
 * remaining operation to the exact-real expression language. Finite reduction
 * domains are unrolled, so runtime-defined BMMs and ordered reducers share the
 * same mathematical sum-of-products authority in this representation.
 */
export function lowerGemma4ScalarExpressionToExactReal(
  expression: Gemma4LiteralScalarExpression,
  context: Gemma4RealStatementLoweringContext,
): string {
  const lower = (current: Gemma4LiteralScalarExpression, bindings: Readonly<Record<string, number>>): string => {
    switch (current.kind) {
      case "literal": {
        if (current.literalType === "boolean") return context.builder.boolean(current.source === "true");
        if (current.literalType !== "number") throw new Error(`Literal real Gemma 4 não numérico não suportado: ${current.source}.`);
        const value = exactDecimalRational(current.source);
        return context.builder.rational(value.numerator, value.denominator);
      }
      case "identifier": {
        const local = context.valueBindings?.[current.name];
        if (local) return local;
        const integer = bindings[current.name];
        if (integer !== undefined) return context.builder.rational(integer);
        if (current.name === "Infinity") throw new Error("+Infinity não pertence ao domínio real finito Gemma 4.");
        return context.builder.input(current.name);
      }
      case "unary": {
        const operand = lower(current.operand, bindings);
        if (current.operator === "+") return operand;
        if (current.operator === "-") return context.builder.negate(operand);
        return context.builder.select(operand, context.builder.boolean(false), context.builder.boolean(true));
      }
      case "binary": return lowerBinary(current.operator, current.left, current.right, bindings, lower, context);
      case "conditional": return context.builder.select(
        lower(current.condition, bindings), lower(current.whenTrue, bindings), lower(current.whenFalse, bindings),
      );
      case "index": return lowerIndex(current, bindings, context);
      case "call": return lowerCall(current, bindings, lower, context);
      case "member": throw new Error(`Acesso de membro real Gemma 4 requer contexto estrutural: ${renderCoordinate(current)}.`);
      case "array": throw new Error("Array real Gemma 4 não é um valor escalar.");
      case "range-inclusive": throw new Error("Range real Gemma 4 só pode aparecer ligado a uma redução.");
      case "filtered-domain": throw new Error("Domínio filtrado real Gemma 4 só pode aparecer ligado a uma redução.");
      case "named-argument": return lower(current.value, bindings);
      case "ordered-loop": throw new Error("Loop ordenado deve ser expandido pelo compilador de statements, não como expressão isolada.");
      case "evaluate-invocation": throw new Error("Invocação composite deve ser ligada ao grafo global antes do lowering real.");
    }
  };
  return lower(expression, context.integerBindings);
}

function lowerBinary(
  operator: string,
  leftExpression: Gemma4LiteralScalarExpression,
  rightExpression: Gemma4LiteralScalarExpression,
  bindings: Readonly<Record<string, number>>,
  lower: (expression: Gemma4LiteralScalarExpression, bindings: Readonly<Record<string, number>>) => string,
  context: Gemma4RealStatementLoweringContext,
): string {
  const left = lower(leftExpression, bindings), right = lower(rightExpression, bindings);
  switch (operator) {
    case "+": return context.builder.add(left, right);
    case "-": return context.builder.subtract(left, right);
    case "*": return context.builder.multiply(left, right);
    case "/": return context.builder.divide(left, right);
    case "%": return context.builder.modulo(left, right);
    case "**": {
      const exponent = evaluateGemma4RealIntegerExpression(rightExpression, bindings, context);
      return context.builder.integerPower(left, exponent);
    }
    case "==": return context.builder.compare("equal", left, right);
    case "!=": return context.builder.compare("not-equal", left, right);
    case "<": return context.builder.compare("less", left, right);
    case "<=": return context.builder.compare("less-equal", left, right);
    case ">": return context.builder.compare("greater", left, right);
    case ">=": return context.builder.compare("greater-equal", left, right);
    case "&&": return context.builder.select(left, right, context.builder.boolean(false));
    case "||": return context.builder.select(left, context.builder.boolean(true), right);
    default: throw new Error(`Operador escalar real Gemma 4 não suportado: ${operator}.`);
  }
}

function lowerIndex(
  expression: Extract<Gemma4LiteralScalarExpression, { kind: "index" }>,
  bindings: Readonly<Record<string, number>>,
  context: Gemma4RealStatementLoweringContext,
): string {
  const coordinates = expression.coordinates.map((coordinate) => evaluateGemma4RealIntegerExpression(coordinate, bindings, context));
  if (expression.target.kind === "identifier") return context.resolveTensorElement(expression.target.name, coordinates);
  if (expression.target.kind === "member" && expression.target.member === "shape" && expression.target.target.kind === "identifier") {
    if (coordinates.length !== 1 || !context.resolveTensorAxis) throw new Error("Acesso real de shape requer um eixo e resolver declarado.");
    return context.resolveTensorAxis(expression.target.target.name, coordinates[0]!);
  }
  if (expression.target.kind === "call" && expression.target.callee.kind === "identifier" && expression.target.callee.name === "decode") {
    const role = expression.target.arguments[0];
    if (expression.target.arguments.length !== 1 || role?.kind !== "identifier") throw new Error("decode real Gemma 4 requer um papel aprendido.");
    return context.resolveLearnedElement(role.name, coordinates);
  }
  throw new Error(`Target indexado real Gemma 4 não suportado: ${renderCoordinate(expression.target)}.`);
}

function lowerCall(
  expression: Extract<Gemma4LiteralScalarExpression, { kind: "call" }>,
  bindings: Readonly<Record<string, number>>,
  lower: (expression: Gemma4LiteralScalarExpression, bindings: Readonly<Record<string, number>>) => string,
  context: Gemma4RealStatementLoweringContext,
): string {
  if (expression.callee.kind !== "identifier") throw new Error("Callee real Gemma 4 precisa ser identificador.");
  const name = expression.callee.name;
  if (["BF16", "F32", "F64", "I32", "exact_product"].includes(name)) {
    if (expression.arguments.length !== 1) throw new Error(`${name} real Gemma 4 requer um argumento.`);
    return lower(expression.arguments[0]!, bindings);
  }
  const unary = new Map<string, Gemma4RealUnaryFunction>([
    ["SLEEF_EXP_F32", "exp"], ["SLEEF_SIN_F32", "sin"], ["SLEEF_COS_F32", "cos"],
    ["SLEEF_TANH_F32", "tanh"], ["SLEEF_LOG1P_F32", "log1p"], ["ARM_SQRT_F32", "sqrt"], ["floor", "floor"],
  ]);
  const function_ = unary.get(name);
  if (function_) {
    if (expression.arguments.length !== 1) throw new Error(`${name} real Gemma 4 requer um argumento.`);
    return context.builder.unaryFunction(function_, lower(expression.arguments[0]!, bindings));
  }
  if (name === "PYTORCH_POW_NEGATIVE_HALF_F32") {
    const argument = lower(expression.arguments[0]!, bindings);
    return context.builder.divide(context.builder.rational(1n), context.builder.unaryFunction("sqrt", argument));
  }
  if (name === "F32_FMA") {
    if (expression.arguments.length !== 3) throw new Error("F32_FMA real Gemma 4 requer três argumentos.");
    return context.builder.add(
      lower(expression.arguments[0]!, bindings),
      context.builder.multiply(lower(expression.arguments[1]!, bindings), lower(expression.arguments[2]!, bindings)),
    );
  }
  if (name === "min" || name === "max") {
    const arguments_ = expression.arguments.map((argument) => lower(argument, bindings));
    return name === "min" ? context.builder.minimum(...arguments_) : context.builder.maximum(...arguments_);
  }
  if (name === "REDUCE") return lowerReduce(expression.arguments, bindings, lower, context, "sum");
  if (["ORDERED_F32_REDUCE_SUM", "PYTORCH_F32_VECTOR_REDUCE_SUM"].includes(name)) {
    return lowerReduce(expression.arguments, bindings, lower, context, "sum");
  }
  if (["ORDERED_F32_REDUCE_MAX", "PYTORCH_F32_VECTOR_REDUCE_MAX"].includes(name)) {
    return lowerReduce(expression.arguments, bindings, lower, context, "maximum");
  }
  if (["ORDERED_F32_DOT", "ARM_NEON_BF16_DOT_F32"].includes(name)) {
    const domain = expression.arguments.find((argument) => argument.kind === "named-argument");
    const values = expression.arguments.filter((argument) => argument.kind !== "named-argument");
    if (!domain || values.length !== 2) throw new Error(`${name} real Gemma 4 requer dois operandos e um domínio.`);
    return lowerReduce([domain, { kind: "binary", operator: "*", left: values[0]!, right: values[1]! }], bindings, lower, context, "sum");
  }
  throw new Error(`Intrinsic escalar real Gemma 4 não suportado: ${name}.`);
}

function lowerReduce(
  arguments_: readonly Gemma4LiteralScalarExpression[],
  bindings: Readonly<Record<string, number>>,
  lower: (expression: Gemma4LiteralScalarExpression, bindings: Readonly<Record<string, number>>) => string,
  context: Gemma4RealStatementLoweringContext,
  operation: "sum" | "maximum",
): string {
  const domains = arguments_.filter((argument): argument is Extract<Gemma4LiteralScalarExpression, { kind: "named-argument" }> => argument.kind === "named-argument");
  const bodies = arguments_.filter((argument) => argument.kind !== "named-argument");
  if (domains.length === 0 || bodies.length !== 1) throw new Error("Redução real Gemma 4 requer domínios nomeados e um corpo.");
  const terms: string[] = [];
  const visit = (index: number, current: Readonly<Record<string, number>>): void => {
    if (index === domains.length) {
      terms.push(lower(bodies[0]!, current));
      return;
    }
    const domain = domains[index]!;
    if (domain.value.kind !== "range-inclusive") throw new Error(`Domínio real Gemma 4 ${domain.name} não é range inclusivo.`);
    const start = evaluateGemma4RealIntegerExpression(domain.value.start, current, context), end = evaluateGemma4RealIntegerExpression(domain.value.end, current, context);
    if (end < start) throw new Error(`Domínio real Gemma 4 ${domain.name} é invertido.`);
    for (let value = start; value <= end; value += 1) visit(index + 1, { ...current, [domain.name]: value });
  };
  visit(0, bindings);
  return operation === "sum" ? context.builder.add(...terms) : context.builder.maximum(...terms);
}

export function evaluateGemma4RealIntegerExpression(
  expression: Gemma4LiteralScalarExpression,
  bindings: Readonly<Record<string, number>>,
  context: Gemma4RealStatementLoweringContext,
): number {
  if (context.resolveInteger) return context.resolveInteger(expression, bindings);
  switch (expression.kind) {
    case "literal": {
      const value = Number(expression.source);
      if (expression.literalType !== "number" || !Number.isSafeInteger(value)) throw new Error(`Coordenada real Gemma 4 inválida: ${expression.source}.`);
      return value;
    }
    case "identifier": {
      const value = bindings[expression.name];
      if (value === undefined) throw new Error(`Coordenada real Gemma 4 não ligada: ${expression.name}.`);
      return value;
    }
    case "unary": {
      const value = evaluateGemma4RealIntegerExpression(expression.operand, bindings, context);
      if (expression.operator === "+") return value;
      if (expression.operator === "-") return -value;
      throw new Error("Negação booleana não é coordenada inteira.");
    }
    case "binary": {
      const left = evaluateGemma4RealIntegerExpression(expression.left, bindings, context), right = evaluateGemma4RealIntegerExpression(expression.right, bindings, context);
      switch (expression.operator) {
        case "+": return left + right;
        case "-": return left - right;
        case "*": return left * right;
        case "/": if (right !== 0 && left % right === 0) return left / right; break;
        case "%": if (right > 0) return ((left % right) + right) % right; break;
      }
      throw new Error(`Expressão inteira Gemma 4 não exata: ${renderCoordinate(expression)}.`);
    }
    case "call": if (expression.callee.kind === "identifier" && expression.callee.name === "floor" && expression.arguments.length === 1) {
      return Math.floor(evaluateGemma4RealIntegerExpression(expression.arguments[0]!, bindings, context));
    } break;
  }
  throw new Error(`Coordenada real Gemma 4 não suportada: ${renderCoordinate(expression)}.`);
}

function exactDecimalRational(source: string): { numerator: bigint; denominator: bigint } {
  const match = /^([+-]?)(\d+)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(source);
  if (!match) throw new Error(`Decimal real Gemma 4 inválido: ${source}.`);
  const sign = match[1] === "-" ? -1n : 1n;
  const fraction = match[3] ?? "";
  const exponent = Number(match[4] ?? "0") - fraction.length;
  const digits = BigInt(`${match[2]}${fraction}` || "0");
  return exponent >= 0
    ? { numerator: sign * digits * (10n ** BigInt(exponent)), denominator: 1n }
    : { numerator: sign * digits, denominator: 10n ** BigInt(-exponent) };
}

function renderCoordinate(expression: Gemma4LiteralScalarExpression): string {
  return JSON.stringify(expression);
}
