export type Gemma4LiteralCoordinateAccess =
  | {
    kind: "tensor-element";
    expression: string;
    coordinates: string[];
    coordinatePrograms: Gemma4LiteralCoordinateExpression[];
  }
  | {
    kind: "tensor-shape";
    expression: string;
    axis: string;
    axisProgram: Gemma4LiteralCoordinateScalarExpression;
  }
  | {
    kind: "whole-value";
    expression: string;
  };

export type Gemma4LiteralCoordinateScalarExpression =
  | { kind: "constant"; value: number }
  | { kind: "symbol"; name: string }
  | { kind: "indexed-symbol"; name: string; indices: Gemma4LiteralCoordinateScalarExpression[] }
  | { kind: "negate"; operand: Gemma4LiteralCoordinateScalarExpression }
  | {
    kind: "add" | "subtract" | "multiply" | "modulo" | "floor-divide";
    left: Gemma4LiteralCoordinateScalarExpression;
    right: Gemma4LiteralCoordinateScalarExpression;
  }
  | {
    kind: "stable-true-prefix-rank";
    tensor: string;
    equals: number;
    batch: Gemma4LiteralCoordinateScalarExpression;
    sequence: Gemma4LiteralCoordinateScalarExpression;
  };

export type Gemma4LiteralCoordinateExpression =
  | Gemma4LiteralCoordinateScalarExpression
  | {
    kind: "inclusive-range";
    start: Gemma4LiteralCoordinateScalarExpression;
    end: Gemma4LiteralCoordinateScalarExpression;
  };

export interface Gemma4LiteralCoordinateExpressionLanguage {
  id: "gemma4-coordinate-expression-v1";
  schemaVersion: 1;
  resultType: "signed-safe-integer-or-inclusive-range";
  evaluationOrder: "depth-first-left-to-right";
  bindings: {
    symbol: string;
    indexedSymbol: string;
    stableTruePrefixRank: string;
  };
  arithmetic: {
    add: string;
    subtract: string;
    multiply: string;
    modulo: string;
    floorDivide: string;
    negate: string;
  };
  range: string;
  invalidOperation: string;
}

export interface Gemma4LiteralCoordinateEnvironment {
  symbols: Readonly<Record<string, number>>;
  integerArrays?: Readonly<Record<string, Gemma4LiteralIntegerArray>>;
  integerTensors?: Readonly<Record<string, readonly (readonly number[])[]>>;
}

export type Gemma4LiteralIntegerArray = readonly (number | Gemma4LiteralIntegerArray)[];

export type Gemma4LiteralEvaluatedCoordinate =
  | number
  | { startInclusive: number; endInclusive: number };

export function gemma4LiteralCoordinateExpressionLanguage(): Gemma4LiteralCoordinateExpressionLanguage {
  return {
    id: "gemma4-coordinate-expression-v1",
    schemaVersion: 1,
    resultType: "signed-safe-integer-or-inclusive-range",
    evaluationOrder: "depth-first-left-to-right",
    bindings: {
      symbol: "read the exact signed safe integer bound to the named scalar, including dotted STRUCT fields",
      indexedSymbol: "evaluate each index left-to-right and read the exact signed safe integer from the named finite integer array",
      stableTruePrefixRank: "compare the named I32 tensor to equals elementwise, then scan the rectangular BOOL mask in stable batch-major order before [batch,sequence]",
    },
    arithmetic: {
      add: "exact safe-integer addition",
      subtract: "exact safe-integer subtraction; negative padding coordinates remain explicit until guarded by the owning formula",
      multiply: "exact safe-integer multiplication",
      modulo: "non-negative remainder with a strictly positive divisor",
      floorDivide: "mathematical floor(left/right) with a non-zero divisor",
      negate: "exact safe-integer additive inverse",
    },
    range: "inclusive start..end evaluated after both signed safe-integer endpoints; start must not exceed end",
    invalidOperation: "fail closed on an unknown symbol or tensor, malformed expression, non-integer tensor value, ragged tensor, unsafe arithmetic, zero divisor, invalid stable-rank coordinate, or inverted range",
  };
}

/** Parses the complete finite coordinate notation used by graph navigation. */
export function parseGemma4LiteralCoordinateExpression(value: string): Gemma4LiteralCoordinateExpression {
  const range = topLevelRange(value);
  if (range) {
    return {
      kind: "inclusive-range",
      start: parseCoordinateScalar(range[0]),
      end: parseCoordinateScalar(range[1]),
    };
  }
  return parseCoordinateScalar(value);
}

/** Executes only the embedded safe-integer coordinate AST, never the source formula string. */
export function evaluateGemma4LiteralCoordinateExpression(
  expression: Gemma4LiteralCoordinateExpression,
  environment: Gemma4LiteralCoordinateEnvironment,
): Gemma4LiteralEvaluatedCoordinate {
  if (expression.kind === "inclusive-range") {
    const startInclusive = evaluateCoordinateScalar(expression.start, environment);
    const endInclusive = evaluateCoordinateScalar(expression.end, environment);
    if (startInclusive > endInclusive) throw new Error(`Range de coordenada invertido: ${startInclusive}..${endInclusive}.`);
    return { startInclusive, endInclusive };
  }
  return evaluateCoordinateScalar(expression, environment);
}

export interface Gemma4LiteralPredecessorCoordinateNavigation {
  input: string;
  producerOperationId?: string;
  accessTemplates: Gemma4LiteralCoordinateAccess[];
  renderedAccesses: Gemma4LiteralCoordinateAccess[];
  renderedCoverage: "complete" | "windowed";
}

export interface Gemma4LiteralOutputCoordinateNavigation {
  /** The unique tensor element written by the assignment's scalar program. */
  write: Extract<Gemma4LiteralCoordinateAccess, { kind: "tensor-element" }>;
  /** Output-shape invariants read before the write, if any. */
  shapeAssertions: Array<Extract<Gemma4LiteralCoordinateAccess, { kind: "tensor-shape" }>>;
}

export interface Gemma4LiteralConsumerCoordinateNavigation {
  operationId: string;
  accesses: Gemma4LiteralCoordinateAccess[];
  scalarUse: "addressed" | "shape-or-control-only";
}

export function buildGemma4LiteralPredecessorCoordinateNavigation(
  predecessors: readonly {
    input: string;
    producerOperationId?: string;
    accesses: Gemma4LiteralCoordinateAccess[];
  }[],
  renderedScalarAssignments: readonly string[],
  complete: boolean,
): Gemma4LiteralPredecessorCoordinateNavigation[] {
  return predecessors.map((predecessor) => ({
    input: predecessor.input,
    ...(predecessor.producerOperationId ? { producerOperationId: predecessor.producerOperationId } : {}),
    accessTemplates: structuredClone(predecessor.accesses),
    renderedAccesses: extractGemma4LiteralCoordinateAccesses(predecessor.input, renderedScalarAssignments),
    renderedCoverage: complete ? "complete" : "windowed",
  }));
}

/**
 * Locates the one output coordinate assigned by a scalar program. Output
 * names appearing only on the right-hand side cannot satisfy this contract.
 */
export function buildGemma4LiteralOutputCoordinateNavigation(
  output: string,
  scalarAssignments: readonly string[],
): Gemma4LiteralOutputCoordinateNavigation {
  const targets = scalarAssignments.flatMap((statement, statementIndex) =>
    assignmentTarget(statement, output).map((access) => ({ access, statementIndex })));
  if (targets.length !== 1) {
    throw new Error(`${output}: programa escalar requer uma única escrita tensorial de saída; encontrou ${targets.length}.`);
  }
  const accesses = extractGemma4LiteralCoordinateAccesses(output, scalarAssignments);
  const tensorElements = accesses.filter((access): access is Extract<Gemma4LiteralCoordinateAccess, { kind: "tensor-element" }> =>
    access.kind === "tensor-element");
  if (tensorElements.length !== 1 || JSON.stringify(tensorElements[0]) !== JSON.stringify(targets[0]!.access)) {
    throw new Error(`${output}: leitura ou escrita adicional da própria saída torna a coordenada de destino ambígua.`);
  }
  const shapeAssertions = scalarAssignments.flatMap((statement, statementIndex) =>
    indexedAccesses(statement, `${output}.shape`, "tensor-shape").map((entry) => ({ ...entry, statement, statementIndex })));
  const shapeAccesses = accesses.filter((access): access is Extract<Gemma4LiteralCoordinateAccess, { kind: "tensor-shape" }> =>
    access.kind === "tensor-shape");
  if (shapeAssertions.length !== shapeAccesses.length || shapeAssertions.some((entry) =>
    !entry.statement.trimStart().startsWith("require ") || entry.statementIndex >= targets[0]!.statementIndex)) {
    throw new Error(`${output}: acesso ao shape da saída deve ser uma precondição anterior à escrita.`);
  }
  return {
    write: targets[0]!.access,
    shapeAssertions: shapeAssertions.map((entry) => entry.access as Extract<Gemma4LiteralCoordinateAccess, { kind: "tensor-shape" }>),
  };
}

/** Builds exact downstream reads without operation, layer, shape or dtype dispatch. */
export function buildGemma4LiteralConsumerCoordinateNavigation(
  output: string,
  consumers: readonly { operationId: string; scalarAssignments: readonly string[] }[],
): Gemma4LiteralConsumerCoordinateNavigation[] {
  return consumers.map((consumer) => {
    const accesses = extractGemma4LiteralCoordinateAccesses(output, consumer.scalarAssignments);
    return {
      operationId: consumer.operationId,
      accesses,
      scalarUse: accesses.length === 0 ? "shape-or-control-only" : "addressed",
    };
  });
}

/**
 * Extracts exact reads of one declared input from the artifact's finite scalar
 * statements. This is intentionally a notation parser, not an operation
 * dispatch table: every Gemma 4 assignment receives the same treatment.
 */
export function extractGemma4LiteralCoordinateAccesses(
  input: string,
  scalarAssignments: readonly string[],
): Gemma4LiteralCoordinateAccess[] {
  if (input.length === 0) throw new Error("Navegação de coordenadas requer nome de input não vazio.");
  const located: Array<{ statementIndex: number; offset: number; access: Gemma4LiteralCoordinateAccess }> = [];
  scalarAssignments.forEach((statement, statementIndex) => {
    for (const access of indexedAccesses(statement, input, "tensor-element")) {
      located.push({ statementIndex, offset: access.offset, access: access.access });
    }
    for (const access of rowMajorAliasAccesses(statement, input)) {
      located.push({ statementIndex, offset: access.offset, access: access.access });
    }
    for (const access of indexedAccesses(statement, `${input}.shape`, "tensor-shape")) {
      located.push({ statementIndex, offset: access.offset, access: access.access });
    }
  });
  const unique = new Map<string, { statementIndex: number; offset: number; access: Gemma4LiteralCoordinateAccess }>();
  for (const entry of located) {
    const key = JSON.stringify(entry.access);
    if (!unique.has(key)) unique.set(key, entry);
  }
  if (unique.size === 0 && scalarAssignments.some((statement) => containsExactName(statement, input))) {
    return [{ kind: "whole-value", expression: input }];
  }
  return [...unique.values()]
    .sort((left, right) => left.statementIndex - right.statementIndex || left.offset - right.offset)
    .map((entry) => entry.access);
}

/**
 * `row_major_alias(x)[...]` is an indexed read of x, not an opaque read of the
 * complete tensor. The wrapper changes only the coordinate spelling and is
 * part of the embedded formula language, so the parser can lower it without
 * dispatching on an operation or assignment.
 */
function rowMajorAliasAccesses(
  statement: string,
  input: string,
): Array<{ offset: number; access: Gemma4LiteralCoordinateAccess }> {
  const base = `row_major_alias(${input})`;
  return indexedAccesses(statement, base, "tensor-element").map((entry) => {
    if (entry.access.kind !== "tensor-element") throw new Error("Alias row-major produziu acesso não tensorial.");
    return {
      offset: entry.offset,
      access: {
        kind: "tensor-element",
        expression: `${input}[${entry.access.coordinates.join(",")}]`,
        coordinates: entry.access.coordinates,
        coordinatePrograms: structuredClone(entry.access.coordinatePrograms),
      },
    };
  });
}

function indexedAccesses(
  statement: string,
  base: string,
  kind: "tensor-element" | "tensor-shape",
): Array<{ offset: number; access: Gemma4LiteralCoordinateAccess }> {
  const marker = `${base}[`;
  const result: Array<{ offset: number; access: Gemma4LiteralCoordinateAccess }> = [];
  let cursor = 0;
  while (cursor < statement.length) {
    const offset = statement.indexOf(marker, cursor);
    if (offset < 0) break;
    cursor = offset + marker.length;
    if (!isNameBoundary(statement[offset - 1])) continue;
    const close = matchingBracket(statement, offset + base.length);
    const expression = statement.slice(offset, close + 1);
    const content = statement.slice(offset + marker.length, close);
    if (kind === "tensor-shape") {
      if (content.length === 0) throw new Error(`Acesso de shape vazio em ${expression}.`);
      const axisProgram = parseGemma4LiteralCoordinateExpression(content);
      if (axisProgram.kind === "inclusive-range") throw new Error(`Eixo de shape não aceita range em ${expression}.`);
      result.push({ offset, access: { kind, expression, axis: content, axisProgram } });
    } else {
      const coordinates = splitCoordinates(content);
      if (coordinates.length === 0) throw new Error(`Acesso tensorial sem coordenadas em ${expression}.`);
      result.push({ offset, access: {
        kind,
        expression,
        coordinates,
        coordinatePrograms: coordinates.map(parseGemma4LiteralCoordinateExpression),
      } });
    }
    cursor = close + 1;
  }
  return result;
}

function assignmentTarget(
  statement: string,
  output: string,
): Array<Extract<Gemma4LiteralCoordinateAccess, { kind: "tensor-element" }>> {
  const leadingWhitespace = statement.length - statement.trimStart().length;
  return indexedAccesses(statement, output, "tensor-element").flatMap((entry) => {
    if (entry.offset !== leadingWhitespace || entry.access.kind !== "tensor-element") return [];
    const suffix = statement.slice(entry.offset + entry.access.expression.length);
    return /^\s*=(?!=)/.test(suffix) ? [entry.access] : [];
  });
}

function matchingBracket(statement: string, open: number): number {
  if (statement[open] !== "[") throw new Error("Parser de coordenadas não recebeu '[' inicial.");
  let depth = 0;
  for (let index = open; index < statement.length; index += 1) {
    if (statement[index] === "[") depth += 1;
    else if (statement[index] === "]") {
      depth -= 1;
      if (depth === 0) return index;
      if (depth < 0) break;
    }
  }
  throw new Error(`Acesso tensorial sem ']' final: ${statement.slice(open)}.`);
}

function splitCoordinates(value: string): string[] {
  const result: string[] = [];
  let start = 0, square = 0, round = 0, brace = 0;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (character === "[") square += 1;
    else if (character === "]") square -= 1;
    else if (character === "(") round += 1;
    else if (character === ")") round -= 1;
    else if (character === "{") brace += 1;
    else if (character === "}") brace -= 1;
    else if (character === "," && square === 0 && round === 0 && brace === 0) {
      result.push(requiredCoordinate(value.slice(start, index)));
      start = index + 1;
    }
    if (square < 0 || round < 0 || brace < 0) throw new Error(`Expressão de coordenada desbalanceada: ${value}.`);
  }
  if (square !== 0 || round !== 0 || brace !== 0) throw new Error(`Expressão de coordenada desbalanceada: ${value}.`);
  result.push(requiredCoordinate(value.slice(start)));
  return result;
}

function requiredCoordinate(value: string): string {
  const coordinate = value.trim();
  if (coordinate.length === 0) throw new Error("Expressão de coordenada vazia.");
  return coordinate;
}

function containsExactName(statement: string, name: string): boolean {
  let cursor = 0;
  while (cursor < statement.length) {
    const offset = statement.indexOf(name, cursor);
    if (offset < 0) return false;
    const before = statement[offset - 1], after = statement[offset + name.length];
    if (isNameBoundary(before) && isNameBoundary(after)) return true;
    cursor = offset + name.length;
  }
  return false;
}

function isNameBoundary(character: string | undefined): boolean {
  return character === undefined || !/[A-Za-z0-9_./:-]/.test(character);
}

type CoordinateToken =
  | { kind: "number"; value: number; offset: number }
  | { kind: "name"; value: string; offset: number }
  | { kind: "operator"; value: "+" | "-" | "*" | "/" | "%" | "==" | "(" | ")" | "[" | "]" | ","; offset: number };

function parseCoordinateScalar(value: string): Gemma4LiteralCoordinateScalarExpression {
  const tokens = coordinateTokens(value);
  let cursor = 0;
  const peek = (): CoordinateToken | undefined => tokens[cursor];
  const take = (): CoordinateToken => {
    const token = tokens[cursor];
    if (!token) throw new Error(`Expressão de coordenada terminou prematuramente: ${value}.`);
    cursor += 1;
    return token;
  };
  const accept = (operator: Extract<CoordinateToken, { kind: "operator" }>["value"]): boolean => {
    const token = peek();
    if (token?.kind !== "operator" || token.value !== operator) return false;
    cursor += 1;
    return true;
  };
  const required = (operator: Extract<CoordinateToken, { kind: "operator" }>["value"]): void => {
    if (!accept(operator)) throw new Error(`Expressão de coordenada requer '${operator}': ${value}.`);
  };
  const additive = (): Gemma4LiteralCoordinateScalarExpression => {
    let expression = multiplicative();
    while (true) {
      if (accept("+")) expression = { kind: "add", left: expression, right: multiplicative() };
      else if (accept("-")) expression = { kind: "subtract", left: expression, right: multiplicative() };
      else return expression;
    }
  };
  const multiplicative = (): Gemma4LiteralCoordinateScalarExpression => {
    let expression = unary();
    while (true) {
      if (accept("*")) expression = { kind: "multiply", left: expression, right: unary() };
      else if (accept("%")) expression = { kind: "modulo", left: expression, right: unary() };
      else if (accept("/")) {
        throw new Error(`Divisão de coordenada requer floor(...): ${value}.`);
      } else return expression;
    }
  };
  const unary = (): Gemma4LiteralCoordinateScalarExpression => {
    if (accept("-")) return { kind: "negate", operand: unary() };
    return primary();
  };
  const primary = (): Gemma4LiteralCoordinateScalarExpression => {
    const token = take();
    if (token.kind === "number") return { kind: "constant", value: token.value };
    if (token.kind === "operator") {
      if (token.value !== "(") throw new Error(`Token inesperado '${token.value}' em coordenada: ${value}.`);
      const nested = additive();
      required(")");
      return nested;
    }
    if (accept("[")) {
      const indices: Gemma4LiteralCoordinateScalarExpression[] = [additive()];
      while (accept(",")) indices.push(additive());
      required("]");
      return { kind: "indexed-symbol", name: token.value, indices };
    }
    if (!accept("(")) return { kind: "symbol", name: token.value };
    if (token.value === "floor") {
      const numerator = additiveUntilDivision(value, tokens, () => cursor, (next) => { cursor = next; });
      required("/");
      const denominator = additive();
      required(")");
      return { kind: "floor-divide", left: numerator, right: denominator };
    }
    if (token.value === "STABLE_TRUE_PREFIX_RANK") {
      const tensor = take();
      if (tensor.kind !== "name") throw new Error(`STABLE_TRUE_PREFIX_RANK requer tensor nomeado: ${value}.`);
      required("==");
      const expected = take();
      if (expected.kind !== "number") throw new Error(`STABLE_TRUE_PREFIX_RANK requer literal I32: ${value}.`);
      required(",");
      const batch = additive();
      required(",");
      const sequence = additive();
      required(")");
      return { kind: "stable-true-prefix-rank", tensor: tensor.value, equals: expected.value, batch, sequence };
    }
    throw new Error(`Helper de coordenada sem programa incorporado: ${token.value}.`);
  };
  const result = additive();
  if (cursor !== tokens.length) throw new Error(`Expressão de coordenada possui sufixo inesperado em ${tokens[cursor]!.offset}: ${value}.`);
  return result;
}

/** Parses the numerator inside floor while preserving nested parentheses. */
function additiveUntilDivision(
  value: string,
  tokens: readonly CoordinateToken[],
  getCursor: () => number,
  setCursor: (cursor: number) => void,
): Gemma4LiteralCoordinateScalarExpression {
  const start = getCursor();
  let depth = 0, division = -1;
  for (let index = start; index < tokens.length; index += 1) {
    const token = tokens[index]!;
    if (token.kind !== "operator") continue;
    if (token.value === "(") depth += 1;
    else if (token.value === ")") {
      if (depth === 0) break;
      depth -= 1;
    } else if (token.value === "/" && depth === 0) {
      division = index;
      break;
    }
  }
  if (division < 0) throw new Error(`floor de coordenada requer divisão explícita: ${value}.`);
  const numeratorText = tokenSlice(value, tokens, start, division);
  setCursor(division);
  return parseCoordinateScalar(numeratorText);
}

function tokenSlice(value: string, tokens: readonly CoordinateToken[], start: number, end: number): string {
  if (start >= end) throw new Error(`Expressão de coordenada vazia em ${value}.`);
  const from = tokens[start]!.offset;
  const last = tokens[end - 1]!;
  const to = last.offset + (last.kind === "number" ? String(last.value).length : last.value.length);
  return value.slice(from, to);
}

function coordinateTokens(value: string): CoordinateToken[] {
  const tokens: CoordinateToken[] = [];
  let cursor = 0;
  while (cursor < value.length) {
    const character = value[cursor]!;
    if (/\s/.test(character)) { cursor += 1; continue; }
    if (/\d/.test(character)) {
      const match = /^\d+/.exec(value.slice(cursor))![0];
      const parsed = Number(match);
      if (!Number.isSafeInteger(parsed)) throw new Error(`Literal de coordenada fora de inteiro seguro: ${match}.`);
      tokens.push({ kind: "number", value: parsed, offset: cursor });
      cursor += match.length;
      continue;
    }
    if (/[A-Za-z_]/.test(character)) {
      const match = /^[A-Za-z_][A-Za-z0-9_.]*/.exec(value.slice(cursor))![0];
      tokens.push({ kind: "name", value: match, offset: cursor });
      cursor += match.length;
      continue;
    }
    const pair = value.slice(cursor, cursor + 2);
    if (pair === "==") {
      tokens.push({ kind: "operator", value: "==", offset: cursor });
      cursor += 2;
      continue;
    }
    if (["+", "-", "*", "/", "%", "(", ")", "[", "]", ","].includes(character)) {
      tokens.push({ kind: "operator", value: character as Extract<CoordinateToken, { kind: "operator" }>["value"], offset: cursor });
      cursor += 1;
      continue;
    }
    throw new Error(`Caractere inválido em coordenada '${character}' na posição ${cursor}: ${value}.`);
  }
  if (tokens.length === 0) throw new Error("Expressão de coordenada vazia.");
  return tokens;
}

function topLevelRange(value: string): [string, string] | undefined {
  let depth = 0;
  for (let index = 0; index < value.length; index += 1) {
    if (value[index] === "(") depth += 1;
    else if (value[index] === ")") depth -= 1;
    else if (index + 1 < value.length && value[index] === "." && value[index + 1] === "." && depth === 0) {
      const start = value.slice(0, index).trim(), end = value.slice(index + 2).trim();
      if (!start || !end) throw new Error(`Range de coordenada incompleto: ${value}.`);
      return [start, end];
    }
    if (depth < 0) throw new Error(`Parênteses desbalanceados em coordenada: ${value}.`);
  }
  if (depth !== 0) throw new Error(`Parênteses desbalanceados em coordenada: ${value}.`);
  return undefined;
}

function evaluateCoordinateScalar(
  expression: Gemma4LiteralCoordinateScalarExpression,
  environment: Gemma4LiteralCoordinateEnvironment,
): number {
  switch (expression.kind) {
    case "constant": return safeSignedInteger(expression.value, "literal");
    case "symbol": {
      const value = environment.symbols[expression.name];
      if (value === undefined) throw new Error(`Binding de coordenada ausente: ${expression.name}.`);
      return safeSignedInteger(value, expression.name);
    }
    case "indexed-symbol": {
      let value: number | Gemma4LiteralIntegerArray | undefined = environment.integerArrays?.[expression.name];
      if (!value) throw new Error(`Array de coordenada ausente: ${expression.name}.`);
      for (const indexExpression of expression.indices) {
        const index = evaluateCoordinateScalar(indexExpression, environment);
        if (!Array.isArray(value) || index < 0 || index >= value.length) {
          throw new Error(`${expression.name}: índice de coordenada fora do array: ${index}.`);
        }
        value = value[index] as number | Gemma4LiteralIntegerArray | undefined;
      }
      if (typeof value !== "number") throw new Error(`${expression.name}: programa de coordenada não terminou em escalar I32.`);
      return safeSignedInteger(value, expression.name);
    }
    case "negate": return safeSignedInteger(-evaluateCoordinateScalar(expression.operand, environment), "negate");
    case "add": case "subtract": case "multiply": case "modulo": case "floor-divide": {
      const left = evaluateCoordinateScalar(expression.left, environment);
      const right = evaluateCoordinateScalar(expression.right, environment);
      if (expression.kind === "add") return safeSignedInteger(left + right, "add");
      if (expression.kind === "subtract") return safeSignedInteger(left - right, "subtract");
      if (expression.kind === "multiply") return safeSignedInteger(left * right, "multiply");
      if (right === 0) throw new Error(`${expression.kind}: divisor de coordenada não pode ser zero.`);
      if (expression.kind === "floor-divide") return safeSignedInteger(Math.floor(left / right), "floor-divide");
      if (right < 0) throw new Error("modulo: divisor de coordenada deve ser positivo.");
      return safeSignedInteger(((left % right) + right) % right, "modulo");
    }
    case "stable-true-prefix-rank": {
      const tensor = environment.integerTensors?.[expression.tensor];
      if (!tensor || tensor.length === 0 || tensor[0]!.length === 0) {
        throw new Error(`STABLE_TRUE_PREFIX_RANK requer tensor I32 ausente: ${expression.tensor}.`);
      }
      const columns = tensor[0]!.length;
      if (tensor.some((row) => row.length !== columns || row.some((value) => !Number.isSafeInteger(value)))) {
        throw new Error(`${expression.tensor}: STABLE_TRUE_PREFIX_RANK requer matriz I32 retangular.`);
      }
      const batch = evaluateCoordinateScalar(expression.batch, environment);
      const sequence = evaluateCoordinateScalar(expression.sequence, environment);
      if (batch < 0 || batch >= tensor.length || sequence < 0 || sequence >= columns) {
        throw new Error("STABLE_TRUE_PREFIX_RANK requer coordenada válida.");
      }
      if (tensor[batch]![sequence] !== expression.equals) return -1;
      let rank = 0;
      for (let priorBatch = 0; priorBatch <= batch; priorBatch += 1) for (let priorSequence = 0; priorSequence < columns; priorSequence += 1) {
        if (priorBatch === batch && priorSequence === sequence) return rank;
        if (tensor[priorBatch]![priorSequence] === expression.equals) rank = safeSignedInteger(rank + 1, "stable-true-prefix-rank");
      }
      throw new Error("STABLE_TRUE_PREFIX_RANK não alcançou a coordenada declarada.");
    }
  }
}

function safeSignedInteger(value: number, context: string): number {
  if (!Number.isSafeInteger(value)) throw new Error(`${context}: coordenada deve ser inteiro seguro; recebeu ${value}.`);
  return value;
}
