import type { Gemma4LiteralScalarExpression } from "./gemma4-literal-scalar-statement-programs.js";
import { type Gemma4RealUnaryFunction } from "./gemma4-real-expression.js";
import { Gemma4ParametricRealBuilder } from "./gemma4-parametric-real-expression.js";

export interface Gemma4ParametricRealLoweringContext {
  builder: Gemma4ParametricRealBuilder;
  scope: string;
  integerBindings: Readonly<Record<string, string>>;
  resolveScalar: (name: string, bindings: Readonly<Record<string, string>>) => string;
  resolveTensorElement: (name: string, coordinates: readonly string[]) => string;
  resolveLearnedElement: (role: string, coordinates: readonly string[]) => string;
  resolveTensorAxis: (name: string, axis: number) => string;
}

export function lowerGemma4ScalarExpressionToParametricReal(
  expression: Gemma4LiteralScalarExpression,
  context: Gemma4ParametricRealLoweringContext,
): string {
  const lower = (current: Gemma4LiteralScalarExpression, bindings: Readonly<Record<string, string>>, scope: string): string => {
    switch (current.kind) {
      case "literal": {
        if (current.literalType === "boolean") return context.builder.boolean(current.source === "true");
        if (current.literalType !== "number") throw new Error(`${scope}: literal paramétrico não numérico ${current.source}.`);
        const rational = exactDecimalRational(current.source);
        return context.builder.rational(rational.numerator, rational.denominator);
      }
      case "identifier": return bindings[current.name] ?? context.resolveScalar(current.name, bindings);
      case "unary": {
        const operand = lower(current.operand, bindings, scope);
        if (current.operator === "+") return operand;
        if (current.operator === "-") return context.builder.multiply(context.builder.rational(-1n), operand);
        return context.builder.select(operand, context.builder.boolean(false), context.builder.boolean(true));
      }
      case "binary": return lowerBinary(current.operator, current.left, current.right, bindings, scope, lower, context);
      case "conditional": return context.builder.select(
        lower(current.condition, bindings, scope),
        lower(current.whenTrue, bindings, scope),
        lower(current.whenFalse, bindings, scope),
      );
      case "index": return lowerIndex(current, bindings, scope, context);
      case "call": return lowerCall(current, bindings, scope, lower, context);
      case "member": return context.resolveScalar(renderMember(current), bindings);
      case "named-argument": return lower(current.value, bindings, scope);
      case "ordered-loop": throw new Error(`${scope}: ordered-loop deve ser resolvido pelo produtor local paramétrico.`);
      case "evaluate-invocation": throw new Error(`${scope}: evaluate-invocation deve ser ligado pelo grafo global paramétrico.`);
      case "array": case "range-inclusive": case "filtered-domain":
        throw new Error(`${scope}: ${current.kind} não é valor escalar paramétrico isolado.`);
    }
  };
  return lower(expression, context.integerBindings, context.scope);
}

export function lowerGemma4IntegerExpressionToParametricReal(
  expression: Gemma4LiteralScalarExpression,
  context: Pick<Gemma4ParametricRealLoweringContext, "builder" | "scope" | "integerBindings" | "resolveTensorAxis"> & {
    resolveIntegerScalar?: (name: string) => string;
    resolveIntegerTensorElement?: (name: string, coordinates: readonly string[]) => string;
  },
): string {
  const lower = (current: Gemma4LiteralScalarExpression): string => {
    switch (current.kind) {
      case "literal": {
        const value = Number(current.source);
        if (current.literalType !== "number" || !Number.isSafeInteger(value)) throw new Error(`${context.scope}: inteiro paramétrico inválido ${current.source}.`);
        return context.builder.integerConstant(value);
      }
      case "identifier": {
        const value = context.integerBindings[current.name];
        if (value) return value;
        if (context.resolveIntegerScalar) return context.resolveIntegerScalar(current.name);
        throw new Error(`${context.scope}: inteiro paramétrico livre ${current.name}.`);
      }
      case "unary": {
        const value = lower(current.operand);
        if (current.operator === "+") return value;
        if (current.operator === "-") return context.builder.integerBinary("integer-subtract", context.builder.integerConstant(0), value);
        break;
      }
      case "binary": {
        const left = lower(current.left), right = lower(current.right);
        switch (current.operator) {
          case "+": return context.builder.integerBinary("integer-add", left, right);
          case "-": return context.builder.integerBinary("integer-subtract", left, right);
          case "*": return context.builder.integerBinary("integer-multiply", left, right);
          case "/": return context.builder.integerBinary("integer-floor-divide", left, right);
          case "%": return context.builder.integerBinary("integer-modulo", left, right);
        }
        break;
      }
      case "conditional": {
        const condition = lowerIntegerCondition(current.condition, lower, context);
        return context.builder.select(condition, lower(current.whenTrue), lower(current.whenFalse));
      }
      case "call": if (current.callee.kind === "identifier" && current.callee.name === "floor" && current.arguments.length === 1) return lower(current.arguments[0]!); break;
      case "index": if (current.target.kind === "identifier" && context.resolveIntegerTensorElement) {
        const coordinates = current.coordinates.map(lower);
        return context.resolveIntegerTensorElement(current.target.name, coordinates);
      }
      if (current.target.kind === "member" && current.target.member === "shape" && current.target.target.kind === "identifier" && current.coordinates.length === 1) {
        const axisNode = lower(current.coordinates[0]!);
        const axis = context.builder.requiredNode(axisNode);
        if (axis.kind !== "integer-constant") throw new Error(`${context.scope}: eixo de shape precisa ser constante.`);
        return context.resolveTensorAxis(current.target.target.name, axis.value);
      } break;
    }
    throw new Error(`${context.scope}: expressão inteira paramétrica não suportada ${JSON.stringify(expression)}.`);
  };
  return lower(expression);
}

function lowerBinary(
  operator: string,
  leftExpression: Gemma4LiteralScalarExpression,
  rightExpression: Gemma4LiteralScalarExpression,
  bindings: Readonly<Record<string, string>>,
  scope: string,
  lower: (expression: Gemma4LiteralScalarExpression, bindings: Readonly<Record<string, string>>, scope: string) => string,
  context: Gemma4ParametricRealLoweringContext,
): string {
  const left = lower(leftExpression, bindings, scope), right = lower(rightExpression, bindings, scope);
  switch (operator) {
    case "+": return context.builder.add(left, right);
    case "-": return context.builder.add(left, context.builder.multiply(context.builder.rational(-1n), right));
    case "*": return context.builder.multiply(left, right);
    case "/": return context.builder.divide(left, right);
    case "**": return context.builder.power(left, right);
    case "%": return context.builder.integerBinary("integer-modulo", left, right);
    case "==": return context.builder.compare("equal", left, right);
    case "!=": return context.builder.compare("not-equal", left, right);
    case "<": return context.builder.compare("less", left, right);
    case "<=": return context.builder.compare("less-equal", left, right);
    case ">": return context.builder.compare("greater", left, right);
    case ">=": return context.builder.compare("greater-equal", left, right);
    case "&&": return context.builder.select(left, right, context.builder.boolean(false));
    case "||": return context.builder.select(left, context.builder.boolean(true), right);
    default: throw new Error(`${scope}: operador paramétrico real não suportado ${operator}.`);
  }
}

function lowerIndex(
  expression: Extract<Gemma4LiteralScalarExpression, { kind: "index" }>,
  bindings: Readonly<Record<string, string>>,
  scope: string,
  context: Gemma4ParametricRealLoweringContext,
): string {
  const integerContext = {
    builder: context.builder, scope, integerBindings: bindings, resolveTensorAxis: context.resolveTensorAxis,
    resolveIntegerScalar: (name: string) => context.resolveScalar(name, bindings), resolveIntegerTensorElement: context.resolveTensorElement,
  };
  const coordinates = expression.coordinates.map((coordinate) => lowerGemma4IntegerExpressionToParametricReal(coordinate, integerContext));
  if (expression.target.kind === "identifier") return context.resolveTensorElement(expression.target.name, coordinates);
  if (expression.target.kind === "call" && expression.target.callee.kind === "identifier" && expression.target.callee.name === "decode") {
    const role = expression.target.arguments[0];
    if (expression.target.arguments.length !== 1 || role?.kind !== "identifier") throw new Error(`${scope}: decode paramétrico requer papel aprendido.`);
    return context.resolveLearnedElement(role.name, coordinates);
  }
  if (expression.target.kind === "call" && expression.target.callee.kind === "identifier" && expression.target.callee.name === "row_major_alias") {
    const tensor = expression.target.arguments[0];
    if (expression.target.arguments.length !== 1 || tensor?.kind !== "identifier") throw new Error(`${scope}: row_major_alias paramétrico requer tensor nomeado.`);
    return context.resolveTensorElement(tensor.name, coordinates);
  }
  if (expression.target.kind === "member" && expression.target.member === "shape" && expression.target.target.kind === "identifier") {
    if (coordinates.length !== 1) throw new Error(`${scope}: tensor.shape paramétrico requer um eixo.`);
    return context.resolveTensorAxis(expression.target.target.name, integerConstant(coordinates[0]!, context.builder, scope));
  }
  throw new Error(`${scope}: target indexado paramétrico não suportado ${JSON.stringify(expression.target)}.`);
}

function lowerCall(
  expression: Extract<Gemma4LiteralScalarExpression, { kind: "call" }>,
  bindings: Readonly<Record<string, string>>,
  scope: string,
  lower: (expression: Gemma4LiteralScalarExpression, bindings: Readonly<Record<string, string>>, scope: string) => string,
  context: Gemma4ParametricRealLoweringContext,
): string {
  if (expression.callee.kind !== "identifier") throw new Error(`${scope}: callee paramétrico precisa ser identificador.`);
  const name = expression.callee.name;
  if (["BF16", "F32", "F64", "I32", "BOOL", "exact_product"].includes(name)) {
    if (expression.arguments.length !== 1) throw new Error(`${scope}: ${name} requer um argumento.`);
    return lower(expression.arguments[0]!, bindings, scope);
  }
  const unary = new Map<string, Gemma4RealUnaryFunction>([
    ["SLEEF_EXP_F32", "exp"], ["SLEEF_SIN_F32", "sin"], ["SLEEF_COS_F32", "cos"], ["SLEEF_TANH_F32", "tanh"],
    ["SLEEF_LOG1P_F32", "log1p"], ["ARM_SQRT_F32", "sqrt"], ["floor", "floor"],
  ]);
  const function_ = unary.get(name);
  if (function_) return context.builder.unaryFunction(function_, lower(requiredArgument(expression, 0, scope), bindings, scope));
  if (name === "PYTORCH_POW_NEGATIVE_HALF_F32") {
    const argument = lower(requiredArgument(expression, 0, scope), bindings, scope);
    return context.builder.divide(context.builder.rational(1n), context.builder.unaryFunction("sqrt", argument));
  }
  if (name === "F32_FMA") {
    if (expression.arguments.length !== 3) throw new Error(`${scope}: F32_FMA requer três argumentos.`);
    return context.builder.add(
      lower(expression.arguments[0]!, bindings, scope),
      context.builder.multiply(lower(expression.arguments[1]!, bindings, scope), lower(expression.arguments[2]!, bindings, scope)),
    );
  }
  if (name === "min" || name === "max") {
    const arguments_ = expression.arguments.map((argument) => lower(argument, bindings, scope));
    return name === "min" ? context.builder.minimum(...arguments_) : context.builder.maximum(...arguments_);
  }
  if (name === "REDUCE") return lowerReduction(expression.arguments, bindings, scope, lower, context, "finite-sum");
  if (["ORDERED_F32_REDUCE_SUM", "PYTORCH_F32_VECTOR_REDUCE_SUM"].includes(name)) {
    return lowerReduction(expression.arguments.filter(notLaneArgument), bindings, scope, lower, context, "finite-sum");
  }
  if (["ORDERED_F32_REDUCE_MAX", "PYTORCH_F32_VECTOR_REDUCE_MAX"].includes(name)) {
    return lowerReduction(expression.arguments.filter(notLaneArgument), bindings, scope, lower, context, "finite-maximum");
  }
  if (["ORDERED_F32_DOT", "ARM_NEON_BF16_DOT_F32"].includes(name)) {
    const domain = expression.arguments.find((argument) => argument.kind === "named-argument");
    const values = expression.arguments.filter((argument) => argument.kind !== "named-argument" && !isScheduleArgument(argument));
    if (!domain || values.length !== 2) throw new Error(`${scope}: ${name} requer dois operandos e um domínio, recebeu ${values.length}.`);
    return lowerReduction([domain, { kind: "binary", operator: "*", left: values[0]!, right: values[1]! }], bindings, scope, lower, context, "finite-sum");
  }
  throw new Error(`${scope}: intrinsic paramétrico real não suportado ${name}.`);
}

function lowerReduction(
  arguments_: readonly Gemma4LiteralScalarExpression[],
  bindings: Readonly<Record<string, string>>,
  scope: string,
  lower: (expression: Gemma4LiteralScalarExpression, bindings: Readonly<Record<string, string>>, scope: string) => string,
  context: Gemma4ParametricRealLoweringContext,
  kind: "finite-sum" | "finite-maximum",
): string {
  const domains = arguments_.filter((argument): argument is Extract<Gemma4LiteralScalarExpression, { kind: "named-argument" }> => argument.kind === "named-argument");
  const bodies = arguments_.filter((argument) => argument.kind !== "named-argument");
  if (domains.length === 0 || bodies.length !== 1) throw new Error(`${scope}: redução paramétrica requer domínio e corpo únicos.`);
  const bound = { ...bindings };
  const normalized = domains.map((domain, ordinal) => {
    const domainScope = `${scope}/reduce-${ordinal}-${domain.name}`;
    const value = domain.value.kind === "filtered-domain" ? domain.value.domain : domain.value;
    if (value.kind !== "range-inclusive") throw new Error(`${domainScope}: domínio não é range inclusivo.`);
    const index = context.builder.boundIndex(domainScope);
    bound[domain.name] = index;
    const integerContext = {
      builder: context.builder, scope: domainScope, integerBindings: bound, resolveTensorAxis: context.resolveTensorAxis,
      resolveIntegerScalar: (name: string) => context.resolveScalar(name, bound), resolveIntegerTensorElement: context.resolveTensorElement,
    };
    const start = lowerGemma4IntegerExpressionToParametricReal(value.start, integerContext);
    const inclusiveEnd = lowerGemma4IntegerExpressionToParametricReal(value.end, integerContext);
    const endExclusive = context.builder.integerBinary("integer-add", inclusiveEnd, context.builder.integerConstant(1));
    return { domain, indexName: domainScope, start, endExclusive };
  });
  let body = lower(bodies[0]!, bound, `${scope}/body`);
  for (let index = normalized.length - 1; index >= 0; index -= 1) {
    const current = normalized[index]!;
    const predicate = current.domain.value.kind === "filtered-domain"
      ? lower(current.domain.value.predicate, bound, `${scope}/predicate-${index}`)
      : undefined;
    body = context.builder.finiteReduction(kind, current.indexName, current.start, current.endExclusive, body, predicate);
  }
  return body;
}

function integerConstant(id: string, builder: Gemma4ParametricRealBuilder, scope: string): number {
  const node = builder.requiredNode(id);
  if (node.kind === "integer-constant") return node.value;
  if (node.kind === "rational" && node.value.denominator === "1") {
    const value = Number(node.value.numerator);
    if (Number.isSafeInteger(value)) return value;
  }
  throw new Error(`${scope}: inteiro constante esperado em ${id}.`);
}

function exactDecimalRational(source: string): { numerator: bigint; denominator: bigint } {
  const match = /^([+-]?)(\d+)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(source);
  if (!match) throw new Error(`Decimal paramétrico Gemma 4 inválido: ${source}.`);
  const sign = match[1] === "-" ? -1n : 1n, fraction = match[3] ?? "";
  const exponent = Number(match[4] ?? "0") - fraction.length;
  const digits = BigInt(`${match[2]}${fraction}`);
  return exponent >= 0
    ? { numerator: sign * digits * (10n ** BigInt(exponent)), denominator: 1n }
    : { numerator: sign * digits, denominator: 10n ** BigInt(-exponent) };
}

function requiredArgument(expression: Extract<Gemma4LiteralScalarExpression, { kind: "call" }>, index: number, scope: string): Gemma4LiteralScalarExpression {
  const argument = expression.arguments[index];
  if (!argument) throw new Error(`${scope}: argumento ${index} ausente.`);
  return argument;
}

function notLaneArgument(argument: Gemma4LiteralScalarExpression): boolean {
  return argument.kind !== "named-argument" || argument.name !== "lanes";
}

function isScheduleArgument(argument: Gemma4LiteralScalarExpression): boolean {
  return argument.kind === "member" && argument.member === "schedule";
}

function renderMember(expression: Extract<Gemma4LiteralScalarExpression, { kind: "member" }>): string {
  return `${JSON.stringify(expression.target)}.${expression.member}`;
}

function lowerIntegerCondition(
  expression: Gemma4LiteralScalarExpression,
  lowerInteger: (expression: Gemma4LiteralScalarExpression) => string,
  context: Pick<Gemma4ParametricRealLoweringContext, "builder" | "scope">,
): string {
  if (expression.kind === "binary") {
    const comparisons = new Map<string, "equal" | "not-equal" | "less" | "less-equal" | "greater" | "greater-equal">([
      ["==", "equal"], ["!=", "not-equal"], ["<", "less"], ["<=", "less-equal"], [">", "greater"], [">=", "greater-equal"],
    ]);
    const comparison = comparisons.get(expression.operator);
    if (comparison) return context.builder.compare(comparison, lowerInteger(expression.left), lowerInteger(expression.right));
  }
  throw new Error(`${context.scope}: condição inteira paramétrica não suportada ${JSON.stringify(expression)}.`);
}
