import { isDeepStrictEqual } from "node:util";

export type Gemma4LiteralScalarExpression =
  | { kind: "literal"; literalType: "number" | "string" | "boolean"; source: string }
  | { kind: "identifier"; name: string }
  | { kind: "array"; elements: Gemma4LiteralScalarExpression[] }
  | { kind: "unary"; operator: "-" | "+" | "!"; operand: Gemma4LiteralScalarExpression }
  | { kind: "binary"; operator: string; left: Gemma4LiteralScalarExpression; right: Gemma4LiteralScalarExpression }
  | { kind: "conditional"; condition: Gemma4LiteralScalarExpression; whenTrue: Gemma4LiteralScalarExpression; whenFalse: Gemma4LiteralScalarExpression }
  | { kind: "call"; callee: Gemma4LiteralScalarExpression; arguments: Gemma4LiteralScalarExpression[] }
  | { kind: "index"; target: Gemma4LiteralScalarExpression; coordinates: Gemma4LiteralScalarExpression[] }
  | { kind: "member"; target: Gemma4LiteralScalarExpression; member: string }
  | { kind: "range-inclusive"; start: Gemma4LiteralScalarExpression; end: Gemma4LiteralScalarExpression }
  | { kind: "filtered-domain"; domain: Gemma4LiteralScalarExpression; predicate: Gemma4LiteralScalarExpression }
  | { kind: "named-argument"; name: string; value: Gemma4LiteralScalarExpression }
  | { kind: "ordered-loop"; body: Gemma4LiteralScalarExpression; index: string; domain: Gemma4LiteralScalarExpression; order: "ascending" }
  | {
      kind: "evaluate-invocation";
      graph: "calculationGraph.assignments";
      predicateField: "invocationId";
      predicateValue: string;
      order: "ordinal-ascending";
      orderedInputs: string[];
      terminalCoordinates: Gemma4LiteralScalarExpression[];
    };

export interface Gemma4LiteralScalarStatementProgram {
  ordinal: number;
  kind: "assignment" | "require";
  /** Human rendering retained for audit; `expression` is the execution authority. */
  source: string;
  targets: Array<{
    name: string;
    role: "local" | "output";
    coordinates: Gemma4LiteralScalarExpression[];
  }>;
  expression: Gemma4LiteralScalarExpression;
}

interface Token {
  kind: "number" | "string" | "identifier" | "operator" | "punctuation" | "eof";
  value: string;
  offset: number;
}

const BINARY_PRECEDENCE = new Map<string, number>([
  ["where", 0],
  ["||", 1], ["&&", 2],
  ["==", 3], ["!=", 3], [">", 3], [">=", 3], ["<", 3], ["<=", 3],
  ["..", 4],
  ["+", 5], ["-", 5],
  ["*", 6], ["/", 6], ["%", 6],
  ["**", 7],
]);

/**
 * Converts every finite scalar statement into a closed syntax tree. The parser
 * dispatches only on the formula language, never on operation, layer or shape.
 */
export function buildGemma4LiteralScalarStatementPrograms(
  scalarAssignments: readonly string[],
  output: string,
  owner: string,
): Gemma4LiteralScalarStatementProgram[] {
  const programs = scalarAssignments.map((source, ordinal) => parseStatement(source, ordinal, output, owner));
  const outputTargets = programs.flatMap((program) => program.targets.filter((target) => target.role === "output"));
  if (outputTargets.length !== 1 || programs.at(-1)?.targets.some((target) => target.role === "output") !== true) {
    throw new Error(`${owner}: programa sintático requer uma única escrita terminal para ${output}.`);
  }
  return programs;
}

export function validateGemma4LiteralScalarStatementPrograms(
  programs: readonly Gemma4LiteralScalarStatementProgram[],
  scalarAssignments: readonly string[],
  output: string,
  owner: string,
): void {
  const expected = buildGemma4LiteralScalarStatementPrograms(scalarAssignments, output, owner);
  if (!isDeepStrictEqual(programs, expected)) {
    throw new Error(`${owner}: programa sintático escalar ausente ou divergente.`);
  }
}

/** Binds call-site tensor names structurally, preserving `/` inside identifiers. */
export function bindGemma4LiteralScalarStatementPrograms(
  programs: readonly Gemma4LiteralScalarStatementProgram[],
  bindings: ReadonlyMap<string, string>,
  boundSources: readonly string[],
): Gemma4LiteralScalarStatementProgram[] {
  if (programs.length !== boundSources.length) throw new Error("Binding sintático Gemma 4 possui cardinalidade divergente.");
  const bindExpression = (expression: Gemma4LiteralScalarExpression): Gemma4LiteralScalarExpression => {
    switch (expression.kind) {
      case "literal": return structuredClone(expression);
      case "identifier": return { ...expression, name: bindings.get(expression.name) ?? expression.name };
      case "array": return { ...expression, elements: expression.elements.map(bindExpression) };
      case "unary": return { ...expression, operand: bindExpression(expression.operand) };
      case "binary": return { ...expression, left: bindExpression(expression.left), right: bindExpression(expression.right) };
      case "conditional": return {
        ...expression,
        condition: bindExpression(expression.condition),
        whenTrue: bindExpression(expression.whenTrue),
        whenFalse: bindExpression(expression.whenFalse),
      };
      case "call": return { ...expression, callee: bindExpression(expression.callee), arguments: expression.arguments.map(bindExpression) };
      case "index": return { ...expression, target: bindExpression(expression.target), coordinates: expression.coordinates.map(bindExpression) };
      case "member": return { ...expression, target: bindExpression(expression.target) };
      case "range-inclusive": return { ...expression, start: bindExpression(expression.start), end: bindExpression(expression.end) };
      case "named-argument": return { ...expression, value: bindExpression(expression.value) };
      case "filtered-domain": return { ...expression, domain: bindExpression(expression.domain), predicate: bindExpression(expression.predicate) };
      case "ordered-loop": return { ...expression, body: bindExpression(expression.body), domain: bindExpression(expression.domain) };
      case "evaluate-invocation": return {
        ...expression,
        orderedInputs: expression.orderedInputs.map((name) => bindings.get(name) ?? name),
        terminalCoordinates: expression.terminalCoordinates.map(bindExpression),
      };
    }
  };
  return programs.map((program, ordinal) => ({
    ...structuredClone(program),
    source: boundSources[ordinal]!,
    targets: program.targets.map((target) => ({
      ...target,
      name: bindings.get(target.name) ?? target.name,
      coordinates: target.coordinates.map(bindExpression),
    })),
    expression: bindExpression(program.expression),
  }));
}

/** Parses one expression independently for conformance tests and readers. */
export function parseGemma4LiteralScalarExpression(source: string): Gemma4LiteralScalarExpression {
  const evaluate = parseEvaluateInvocation(source);
  if (evaluate) return evaluate;
  const loop = parseOrderedLoop(source);
  if (loop) return loop;
  const parser = new ExpressionParser(tokenize(source), source);
  const expression = parser.parseExpression();
  parser.expectEof();
  return expression;
}

function parseOrderedLoop(source: string): Gemma4LiteralScalarExpression | undefined {
  const parts = splitTopLevel(source, ",");
  if (parts.length === 1) return undefined;
  if (parts.length !== 2) throw new Error(`Loop escalar Gemma 4 possui cláusulas ambíguas: ${source}.`);
  const clause = /^([A-Za-z_][A-Za-z0-9_]*)=(.+) ascending$/.exec(parts[1]!);
  if (!clause) return undefined;
  return {
    kind: "ordered-loop",
    body: parseGemma4LiteralScalarExpression(parts[0]!),
    index: clause[1]!,
    domain: parseGemma4LiteralScalarExpression(clause[2]!),
    order: "ascending",
  };
}

function parseStatement(source: string, ordinal: number, output: string, owner: string): Gemma4LiteralScalarStatementProgram {
  if (source.startsWith("require ")) {
    return {
      ordinal,
      kind: "require",
      source,
      targets: [],
      expression: parseGemma4LiteralScalarExpression(source.slice("require ".length)),
    };
  }
  const equals = topLevelAssignment(source);
  if (equals < 0) throw new Error(`${owner}: statement ${ordinal} não possui atribuição de topo.`);
  const targetSource = source.slice(0, equals).trim();
  const expressionSource = source.slice(equals + 1).trim();
  const targets = parseTargets(targetSource, output, owner, ordinal);
  return {
    ordinal,
    kind: "assignment",
    source,
    targets,
    expression: parseGemma4LiteralScalarExpression(expressionSource),
  };
}

function parseTargets(
  source: string,
  output: string,
  owner: string,
  ordinal: number,
): Gemma4LiteralScalarStatementProgram["targets"] {
  const structure = /^STRUCT\(([^)]*)\)$/.exec(source);
  if (structure) return splitTopLevel(structure[1]!, ",").map((name) => {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new Error(`${owner}: statement ${ordinal} possui alvo STRUCT inválido ${name}.`);
    return { name, role: name === output ? "output" as const : "local" as const, coordinates: [] };
  });
  const open = source.indexOf("[");
  if (open < 0) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(source)) throw new Error(`${owner}: statement ${ordinal} possui alvo inválido ${source}.`);
    return [{ name: source, role: source === output ? "output" : "local", coordinates: [] }];
  }
  if (!source.endsWith("]")) throw new Error(`${owner}: statement ${ordinal} possui alvo indexado malformado ${source}.`);
  const name = source.slice(0, open);
  if (!/^[A-Za-z_][A-Za-z0-9_/:.-]*$/.test(name)) throw new Error(`${owner}: statement ${ordinal} possui nome de alvo inválido ${name}.`);
  const coordinates = splitTopLevel(source.slice(open + 1, -1), ",").map(parseGemma4LiteralScalarExpression);
  return [{ name, role: name === output ? "output" : "local", coordinates }];
}

function parseEvaluateInvocation(source: string): Gemma4LiteralScalarExpression | undefined {
  if (!source.startsWith("EVALUATE(")) return undefined;
  const match = /^EVALUATE\(calculationGraph\.assignments where invocationId==(\"(?:[^\"\\]|\\.)*\") in ordinal order, orderedInputs=\[([^\]]*)\]\)\.terminalOutput\[([\s\S]*)\]$/.exec(source);
  if (!match) throw new Error(`EVALUATE Gemma 4 fora da gramática registrada: ${source}.`);
  const predicateValue = JSON.parse(match[1]!) as unknown;
  if (typeof predicateValue !== "string") throw new Error("EVALUATE Gemma 4 requer invocationId string.");
  const orderedInputs = match[2]!.trim().length === 0 ? [] : splitTopLevel(match[2]!, ",");
  if (orderedInputs.some((name) => !/^[A-Za-z_][A-Za-z0-9_/:.-]*$/.test(name))) {
    throw new Error("EVALUATE Gemma 4 possui orderedInputs inválidos.");
  }
  return {
    kind: "evaluate-invocation",
    graph: "calculationGraph.assignments",
    predicateField: "invocationId",
    predicateValue,
    order: "ordinal-ascending",
    orderedInputs,
    terminalCoordinates: splitTopLevel(match[3]!, ",").map(parseGemma4LiteralScalarExpression),
  };
}

class ExpressionParser {
  private index = 0;

  constructor(private readonly tokens: readonly Token[], private readonly source: string) {}

  parseExpression(minimumPrecedence = 0): Gemma4LiteralScalarExpression {
    let left = this.parsePrefix();
    while (true) {
      const token = this.peek();
      if (token.value === "?" && minimumPrecedence <= 0) {
        this.consume();
        const whenTrue = this.parseExpression();
        this.expect(":");
        const whenFalse = this.parseExpression();
        left = { kind: "conditional", condition: left, whenTrue, whenFalse };
        continue;
      }
      const precedence = BINARY_PRECEDENCE.get(token.value);
      if (precedence === undefined || precedence < minimumPrecedence) break;
      this.consume();
      const right = this.parseExpression(precedence + (token.value === "**" ? 0 : 1));
      left = token.value === ".."
        ? { kind: "range-inclusive", start: left, end: right }
        : token.value === "where"
          ? { kind: "filtered-domain", domain: left, predicate: right }
          : { kind: "binary", operator: token.value, left, right };
    }
    return left;
  }

  expectEof(): void {
    if (this.peek().kind !== "eof") this.fail(`token inesperado ${this.peek().value}`);
  }

  private parsePrefix(): Gemma4LiteralScalarExpression {
    const token = this.consume();
    let expression: Gemma4LiteralScalarExpression;
    if (token.value === "-" || token.value === "+" || token.value === "!") {
      expression = { kind: "unary", operator: token.value, operand: this.parseExpression(8) };
    } else if (token.value === "(") {
      expression = this.parseExpression();
      this.expect(")");
    } else if (token.value === "[") {
      const elements = this.parseDelimited("]");
      expression = { kind: "array", elements };
    } else if (token.kind === "number") {
      expression = { kind: "literal", literalType: "number", source: token.value };
    } else if (token.kind === "string") {
      JSON.parse(token.value);
      expression = { kind: "literal", literalType: "string", source: token.value };
    } else if (token.kind === "identifier" && (token.value === "true" || token.value === "false")) {
      expression = { kind: "literal", literalType: "boolean", source: token.value };
    } else if (token.kind === "identifier") {
      expression = { kind: "identifier", name: token.value };
    } else {
      this.fail(`expressão não pode iniciar com ${token.value}`);
    }

    while (true) {
      if (this.peek().value === "(") {
        this.consume();
        const arguments_ = expression.kind === "identifier" && expression.name === "decode"
          ? [this.parseSymbolicName(")")]
          : this.parseDelimited(")", true);
        expression = { kind: "call", callee: expression, arguments: arguments_ };
      } else if (this.peek().value === "[") {
        this.consume();
        const coordinates = expression.kind === "identifier" && expression.name === "reductionStages"
          ? [this.parseSymbolicName("]")]
          : this.parseDelimited("]");
        expression = { kind: "index", target: expression, coordinates };
      } else if (this.peek().value === ".") {
        this.consume();
        const member = this.consume();
        if (member.kind !== "identifier") this.fail("membro requer identificador");
        expression = { kind: "member", target: expression, member: member.value };
      } else break;
    }
    return expression;
  }

  private parseDelimited(close: string, allowNamed = false): Gemma4LiteralScalarExpression[] {
    const result: Gemma4LiteralScalarExpression[] = [];
    if (this.peek().value === close) { this.consume(); return result; }
    while (true) {
      if (allowNamed && this.peek().kind === "identifier" && this.tokens[this.index + 1]?.value === "=") {
        const name = this.consume().value;
        this.consume();
        result.push({ kind: "named-argument", name, value: this.parseExpression() });
      } else result.push(this.parseExpression());
      if (this.peek().value === close) { this.consume(); return result; }
      this.expect(",");
    }
  }

  private parseSymbolicName(close: string): Gemma4LiteralScalarExpression {
    const first = this.consume();
    if (first.kind !== "identifier") this.fail("nome simbólico requer identificador");
    let name = first.value;
    while (this.peek().value === "-") {
      this.consume();
      const part = this.consume();
      if (part.kind !== "identifier") this.fail("nome simbólico possui hífen sem identificador");
      name += `-${part.value}`;
    }
    this.expect(close);
    return { kind: "identifier", name };
  }

  private expect(value: string): void {
    const token = this.consume();
    if (token.value !== value) this.fail(`esperado ${value}, recebeu ${token.value}`);
  }

  private peek(): Token { return this.tokens[this.index]!; }
  private consume(): Token { return this.tokens[this.index++]!; }
  private fail(message: string): never { throw new Error(`Expressão escalar Gemma 4 inválida (${message}): ${this.source}.`); }
}

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let offset = 0;
  while (offset < source.length) {
    if (/\s/.test(source[offset]!)) { offset += 1; continue; }
    const rest = source.slice(offset);
    const string = /^"(?:[^"\\]|\\.)*"/.exec(rest)?.[0];
    if (string) { tokens.push({ kind: "string", value: string, offset }); offset += string.length; continue; }
    const number = /^(?:\d+\.(?!\.)\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?/.exec(rest)?.[0];
    if (number) { tokens.push({ kind: "number", value: number, offset }); offset += number.length; continue; }
    const identifier = /^[A-Za-z_][A-Za-z0-9_:]*/.exec(rest)?.[0];
    if (identifier) { tokens.push({ kind: "identifier", value: identifier, offset }); offset += identifier.length; continue; }
    const operator = ["**", "==", "!=", ">=", "<=", "&&", "||", ".."].find((candidate) => rest.startsWith(candidate));
    if (operator) { tokens.push({ kind: "operator", value: operator, offset }); offset += operator.length; continue; }
    const single = source[offset]!;
    if ("+-*/%><!?=.,:()[]".includes(single)) {
      tokens.push({ kind: ",()[]".includes(single) ? "punctuation" : "operator", value: single, offset });
      offset += 1;
      continue;
    }
    throw new Error(`Token escalar Gemma 4 desconhecido em ${offset}: ${source}.`);
  }
  tokens.push({ kind: "eof", value: "<eof>", offset: source.length });
  return tokens;
}

function topLevelAssignment(source: string): number {
  let parentheses = 0;
  let brackets = 0;
  let quoted = false;
  let escaped = false;
  for (let index = 0; index < source.length; index += 1) {
    const value = source[index]!;
    if (quoted) {
      if (escaped) escaped = false;
      else if (value === "\\") escaped = true;
      else if (value === '"') quoted = false;
      continue;
    }
    if (value === '"') quoted = true;
    else if (value === "(") parentheses += 1;
    else if (value === ")") parentheses -= 1;
    else if (value === "[") brackets += 1;
    else if (value === "]") brackets -= 1;
    else if (value === "=" && parentheses === 0 && brackets === 0 && source[index - 1] !== "!" && source[index - 1] !== "<" && source[index - 1] !== ">" && source[index + 1] !== "=") return index;
  }
  return -1;
}

function splitTopLevel(source: string, separator: string): string[] {
  const result: string[] = [];
  let start = 0;
  let parentheses = 0;
  let brackets = 0;
  let quoted = false;
  let escaped = false;
  for (let index = 0; index < source.length; index += 1) {
    const value = source[index]!;
    if (quoted) {
      if (escaped) escaped = false;
      else if (value === "\\") escaped = true;
      else if (value === '"') quoted = false;
      continue;
    }
    if (value === '"') quoted = true;
    else if (value === "(") parentheses += 1;
    else if (value === ")") parentheses -= 1;
    else if (value === "[") brackets += 1;
    else if (value === "]") brackets -= 1;
    else if (value === separator && parentheses === 0 && brackets === 0) {
      result.push(source.slice(start, index).trim());
      start = index + 1;
    }
  }
  result.push(source.slice(start).trim());
  if (result.some((entry) => entry.length === 0)) throw new Error(`Lista escalar Gemma 4 malformada: ${source}.`);
  return result;
}
