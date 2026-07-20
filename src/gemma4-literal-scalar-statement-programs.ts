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

export interface Gemma4LiteralScalarReductionBinding {
  index: string;
  source: "assignment" | "stage";
  stageId?: string;
  domainOrdinal: number;
  /** Human range aliases are bound to the executable domain beside the calculation. */
  extentAlias?: string;
}

export interface Gemma4LiteralScalarStatementEnvironment {
  kind: "gemma4-literal-scalar-statement-environment";
  schemaVersion: 1;
  outputCoordinates: string[];
  orderedInputs: Array<{ position: number; name: string }>;
  locals: Array<{ name: string; producerStatementOrdinal: number }>;
  reductions: Gemma4LiteralScalarReductionBinding[];
  learnedOperandRoles: string[];
  intrinsics: string[];
  specialValues: ["Infinity"];
  memberAccesses: Array<{
    target: string;
    member: string;
    contract: "tensor-shape" | "reduction-stage-schedule" | "registered-structured-result";
  }>;
}

export interface Gemma4LiteralScalarStatementEnvironmentContext {
  output: string;
  outputCoordinates: readonly string[];
  orderedInputs: readonly string[];
  learnedOperandRoles: readonly string[];
  reductions: ReadonlyArray<{
    indices: readonly string[];
    source: "assignment" | "stage";
    stageId?: string;
  }>;
}

export const GEMMA4_LITERAL_REGISTERED_FUNCTIONS = new Set([
  "ARM_NEON_BF16_DOT_F32", "ARM_SQRT_F32", "AUDIO_RELATIVE_SHIFT_SOURCE", "BF16", "BOOL", "CONTIGUOUS_VISION_GROUP_ID",
  "EVALUATE", "F32", "F32_FMA", "F64", "I32", "ORDERED_F32_DOT", "ORDERED_F32_REDUCE_MAX",
  "ORDERED_F32_REDUCE_SUM", "PYTORCH_F32_VECTOR_REDUCE_MAX", "PYTORCH_F32_VECTOR_REDUCE_SUM",
  "PYTORCH_POW_NEGATIVE_HALF_F32", "REDUCE", "SLEEF_COS_F32", "SLEEF_EXP_F32", "SLEEF_LOG1P_F32",
  "SLEEF_SIN_F32", "SLEEF_TANH_F32", "STABLE_TRUE_COORDINATE_AT_RANK", "STABLE_TRUE_COUNT",
  "STABLE_TRUE_PREFIX_RANK", "STRUCT", "VISION_POOL_CELL_HAS_PATCH", "VISION_POOL_SLOT", "concat", "decode", "exact_product",
  "exact_safe_integer", "floor", "max", "min", "row_major_alias", "tuple",
]);

const STRUCTURED_RESULT_MEMBERS = new Map<string, ReadonlySet<string>>([
  ["AUDIO_RELATIVE_SHIFT_SOURCE", new Set(["valid", "query_in_block", "relative_index"])],
  ["STABLE_TRUE_COORDINATE_AT_RANK", new Set(["batch", "sequence"])],
]);

const REDUCTION_INTRINSICS_WITH_NAMED_ARGUMENTS = new Set([
  "ARM_NEON_BF16_DOT_F32", "ORDERED_F32_DOT", "ORDERED_F32_REDUCE_MAX", "ORDERED_F32_REDUCE_SUM",
  "PYTORCH_F32_VECTOR_REDUCE_MAX", "PYTORCH_F32_VECTOR_REDUCE_SUM", "REDUCE",
]);

const VECTOR_REDUCTION_INTRINSICS = new Set([
  "PYTORCH_F32_VECTOR_REDUCE_MAX", "PYTORCH_F32_VECTOR_REDUCE_SUM",
]);

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

/**
 * Builds and validates the complete lexical environment for one scalar
 * program. Syntax alone is not executable: every identifier, helper, member,
 * reduction extent and learned role must resolve to artifact data.
 */
export function buildGemma4LiteralScalarStatementEnvironment(
  programs: readonly Gemma4LiteralScalarStatementProgram[],
  context: Gemma4LiteralScalarStatementEnvironmentContext,
  owner: string,
): Gemma4LiteralScalarStatementEnvironment {
  const reductions = reductionBindings(context.reductions, owner);
  const outputCoordinates = uniqueNames(context.outputCoordinates, `${owner}: coordenadas de saída`);
  const orderedInputNames = uniqueNames(context.orderedInputs, `${owner}: orderedInputs`);
  const learnedOperandRoles = uniqueNames(context.learnedOperandRoles, `${owner}: learnedOperandRoles`);
  const reductionIndices = new Set(reductions.map((binding) => binding.index));
  const extentAliases = new Set(reductions.flatMap((binding) => binding.extentAlias ? [binding.extentAlias] : []));
  const stageIds = new Set(context.reductions.flatMap((binding) => binding.stageId ? [binding.stageId] : []));
  const localProducer = new Map<string, number>();
  const localBindings: Gemma4LiteralScalarStatementEnvironment["locals"] = [];
  const structuredLocalContracts = new Map<string, string>();
  const intrinsics = new Set<string>();
  const memberAccesses = new Map<string, Gemma4LiteralScalarStatementEnvironment["memberAccesses"][number]>();
  const baseSymbols = new Set([...outputCoordinates, ...reductionIndices, ...extentAliases, ...orderedInputNames]);

  const visit = (
    expression: Gemma4LiteralScalarExpression,
    availableLocals: ReadonlySet<string>,
    lexicalSymbols: ReadonlySet<string>,
    position: string,
  ): void => {
    switch (expression.kind) {
      case "literal": return;
      case "identifier": {
        if (expression.name === "Infinity" || baseSymbols.has(expression.name) || availableLocals.has(expression.name) || lexicalSymbols.has(expression.name)) return;
        throw new Error(`${owner}: identificador escalar livre ${expression.name} em ${position}.`);
      }
      case "array": expression.elements.forEach((element, index) => visit(element, availableLocals, lexicalSymbols, `${position}.elements[${index}]`)); return;
      case "unary": visit(expression.operand, availableLocals, lexicalSymbols, `${position}.operand`); return;
      case "binary":
        visit(expression.left, availableLocals, lexicalSymbols, `${position}.left`);
        visit(expression.right, availableLocals, lexicalSymbols, `${position}.right`);
        return;
      case "conditional":
        visit(expression.condition, availableLocals, lexicalSymbols, `${position}.condition`);
        visit(expression.whenTrue, availableLocals, lexicalSymbols, `${position}.whenTrue`);
        visit(expression.whenFalse, availableLocals, lexicalSymbols, `${position}.whenFalse`);
        return;
      case "call": {
        if (expression.callee.kind !== "identifier" || !GEMMA4_LITERAL_REGISTERED_FUNCTIONS.has(expression.callee.name)) {
          throw new Error(`${owner}: callee escalar não registrado em ${position}.`);
        }
        const callee = expression.callee.name;
        intrinsics.add(callee);
        if (callee === "decode") {
          if (expression.arguments.length !== 1 || expression.arguments[0]?.kind !== "identifier" ||
            !learnedOperandRoles.includes(expression.arguments[0].name)) {
            throw new Error(`${owner}: decode em ${position} não referencia um learnedOperandRole declarado.`);
          }
          return;
        }
        expression.arguments.forEach((argument, index) => {
          if (argument.kind === "named-argument") {
            if (!REDUCTION_INTRINSICS_WITH_NAMED_ARGUMENTS.has(callee)) {
              throw new Error(`${owner}: intrinsic ${callee} não aceita argumento nomeado em ${position}.`);
            }
            if (argument.name === "lanes" && (!VECTOR_REDUCTION_INTRINSICS.has(callee) ||
              argument.value.kind !== "literal" || argument.value.literalType !== "number" || argument.value.source !== "4")) {
              throw new Error(`${owner}: lanes só pode fixar quatro lanes no redutor vetorial registrado em ${position}.`);
            }
            if (argument.name !== "lanes" && !reductionIndices.has(argument.name)) {
              throw new Error(`${owner}: argumento nomeado ${argument.name} não possui domínio de redução em ${position}.`);
            }
            visit(argument.value, availableLocals, lexicalSymbols, `${position}.arguments[${index}].value`);
          } else visit(argument, availableLocals, lexicalSymbols, `${position}.arguments[${index}]`);
        });
        return;
      }
      case "index": {
        if (expression.target.kind === "identifier" && expression.target.name === "reductionStages") {
          if (expression.coordinates.length !== 1 || expression.coordinates[0]?.kind !== "identifier" || !stageIds.has(expression.coordinates[0].name)) {
            throw new Error(`${owner}: acesso reductionStages inválido em ${position}.`);
          }
        } else visit(expression.target, availableLocals, lexicalSymbols, `${position}.target`);
        expression.coordinates.forEach((coordinate, index) => {
          if (expression.target.kind === "identifier" && expression.target.name === "reductionStages") return;
          visit(coordinate, availableLocals, lexicalSymbols, `${position}.coordinates[${index}]`);
        });
        return;
      }
      case "member": {
        visitMember(expression, availableLocals, lexicalSymbols, position);
        return;
      }
      case "range-inclusive":
        visit(expression.start, availableLocals, lexicalSymbols, `${position}.start`);
        visit(expression.end, availableLocals, lexicalSymbols, `${position}.end`);
        return;
      case "filtered-domain":
        visit(expression.domain, availableLocals, lexicalSymbols, `${position}.domain`);
        visit(expression.predicate, availableLocals, lexicalSymbols, `${position}.predicate`);
        return;
      case "named-argument": throw new Error(`${owner}: argumento nomeado fora de call em ${position}.`);
      case "ordered-loop": {
        visit(expression.domain, availableLocals, lexicalSymbols, `${position}.domain`);
        const nested = new Set(lexicalSymbols);
        nested.add(expression.index);
        visit(expression.body, availableLocals, nested, `${position}.body`);
        return;
      }
      case "evaluate-invocation":
        for (const input of expression.orderedInputs) if (!orderedInputNames.includes(input)) {
          throw new Error(`${owner}: EVALUATE referencia orderedInput livre ${input}.`);
        }
        expression.terminalCoordinates.forEach((coordinate, index) =>
          visit(coordinate, availableLocals, lexicalSymbols, `${position}.terminalCoordinates[${index}]`));
        return;
    }
  };

  const visitMember = (
    expression: Extract<Gemma4LiteralScalarExpression, { kind: "member" }>,
    availableLocals: ReadonlySet<string>,
    lexicalSymbols: ReadonlySet<string>,
    position: string,
  ): void => {
    let target: string;
    let contract: Gemma4LiteralScalarStatementEnvironment["memberAccesses"][number]["contract"];
    if (expression.member === "shape" && expression.target.kind === "identifier" &&
      (orderedInputNames.includes(expression.target.name) || expression.target.name === context.output)) {
      target = expression.target.name;
      contract = "tensor-shape";
    } else if (expression.member === "schedule" && expression.target.kind === "index" &&
      expression.target.target.kind === "identifier" && expression.target.target.name === "reductionStages") {
      visit(expression.target, availableLocals, lexicalSymbols, `${position}.target`);
      target = "reductionStages";
      contract = "reduction-stage-schedule";
    } else if (expression.target.kind === "identifier" && availableLocals.has(expression.target.name)) {
      const producer = structuredLocalContracts.get(expression.target.name);
      if (!producer || !STRUCTURED_RESULT_MEMBERS.get(producer)?.has(expression.member)) {
        throw new Error(`${owner}: membro ${expression.member} não registrado para ${expression.target.name} em ${position}.`);
      }
      target = expression.target.name;
      contract = "registered-structured-result";
    } else {
      throw new Error(`${owner}: acesso de membro ${expression.member} não resolvido em ${position}.`);
    }
    memberAccesses.set(`${target}.${expression.member}:${contract}`, { target, member: expression.member, contract });
  };

  for (const program of programs) {
    const available = new Set(localProducer.keys());
    program.targets.forEach((target, targetIndex) => target.coordinates.forEach((coordinate, coordinateIndex) =>
      visit(coordinate, available, new Set(), `statementPrograms[${program.ordinal}].targets[${targetIndex}].coordinates[${coordinateIndex}]`)));
    visit(program.expression, available, new Set(), `statementPrograms[${program.ordinal}].expression`);
    const structuredProducer = program.expression.kind === "call" && program.expression.callee.kind === "identifier" &&
      STRUCTURED_RESULT_MEMBERS.has(program.expression.callee.name) ? program.expression.callee.name : undefined;
    for (const target of program.targets) if (target.role === "local") {
      localProducer.set(target.name, program.ordinal);
      localBindings.push({ name: target.name, producerStatementOrdinal: program.ordinal });
      if (structuredProducer) structuredLocalContracts.set(target.name, structuredProducer);
      else structuredLocalContracts.delete(target.name);
    }
  }

  return {
    kind: "gemma4-literal-scalar-statement-environment",
    schemaVersion: 1,
    outputCoordinates,
    orderedInputs: orderedInputNames.map((name, position) => ({ position, name })),
    locals: localBindings,
    reductions,
    learnedOperandRoles,
    intrinsics: [...intrinsics].sort(),
    specialValues: ["Infinity"],
    memberAccesses: [...memberAccesses.values()],
  };
}

function reductionBindings(
  reductions: Gemma4LiteralScalarStatementEnvironmentContext["reductions"],
  owner: string,
): Gemma4LiteralScalarReductionBinding[] {
  const result: Gemma4LiteralScalarReductionBinding[] = [];
  for (const reduction of reductions) reduction.indices.forEach((source, domainOrdinal) => {
    const match = /^([A-Za-z_][A-Za-z0-9_]*)=0\.\.(?:([A-Za-z_][A-Za-z0-9_]*)-1|([0-9]+)(?:-1)?)$/.exec(source);
    if (!match) throw new Error(`${owner}: índice de redução não possui range fechado reconhecível: ${source}.`);
    result.push({
      index: match[1]!,
      source: reduction.source,
      ...(reduction.stageId ? { stageId: reduction.stageId } : {}),
      domainOrdinal,
      ...(match[2] ? { extentAlias: match[2] } : {}),
    });
  });
  const keys = new Set<string>();
  for (const binding of result) {
    const key = `${binding.source}:${binding.stageId ?? ""}:${binding.index}`;
    if (keys.has(key)) throw new Error(`${owner}: binding de redução duplicado ${key}.`);
    keys.add(key);
  }
  return result;
}

function uniqueNames(values: readonly string[], owner: string): string[] {
  const result = [...values];
  if (result.some((value) => !/^[A-Za-z_][A-Za-z0-9_/:.-]*$/.test(value)) || new Set(result).size !== result.length) {
    throw new Error(`${owner} possui nomes inválidos ou duplicados.`);
  }
  return result;
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
