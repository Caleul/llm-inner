import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { listGemma4LiteralOperations } from "./gemma4-literal-multimodal-scalar-view.js";
import type { Gemma4LiteralScalarCalculation } from "./gemma4-literal-scalar-calculations.js";
import type {
  Gemma4LiteralOperationNavigation,
  Gemma4LiteralScalarViewRequest,
} from "./gemma4-literal-scalar-view.js";
import {
  buildGemma4LiteralOutputCoordinateNavigation,
  buildGemma4LiteralPredecessorCoordinateNavigation,
  type Gemma4LiteralOutputCoordinateNavigation,
  type Gemma4LiteralPredecessorCoordinateNavigation,
} from "./gemma4-literal-coordinate-accesses.js";
import {
  gemma4LiteralRuntimeReductionOperationClass,
  type Gemma4LiteralRuntimeReductionOperationClass,
} from "./gemma4-literal-fidelity-gate.js";
import type { Gemma4LiteralScalarExpression } from "./gemma4-literal-scalar-statement-programs.js";

export interface Gemma4LiteralRuntimeReductionOperation {
  operationId: string;
  definitionId: string;
  invocationId?: string;
  operation: string;
  operationClass: Gemma4LiteralRuntimeReductionOperationClass;
  output: string;
  outputDomain: Gemma4LiteralOperationNavigation["outputDomain"];
  reductionDomain: NonNullable<Gemma4LiteralScalarCalculation["reduction"]>["domains"][number];
  provider: "Apple Accelerate SGEMM";
  scalarSchedule: "unpublished-fail-closed";
  auditability: "operand-products-addressable-reduction-fail-closed";
}

export interface Gemma4LiteralRuntimeReductionProductTerm {
  reductionIndex: number;
  leftOperand: string;
  rightOperand: string;
  predicate?: string;
  mathematicalProduct: string;
}

/**
 * A deliberately non-executable audit of one address in a native BMM.
 * It exposes every operand address and padding predicate without inventing the
 * provider's unpublished product/accumulation rounding tree.
 */
export interface Gemma4LiteralRuntimeReductionAudit {
  kind: "gemma4-literal-runtime-reduction-product-audit";
  schemaVersion: 1;
  sourceCheckpointAccessed: false;
  navigation: Gemma4LiteralOperationNavigation;
  operationClass: Gemma4LiteralRuntimeReductionOperationClass;
  outputCoordinate: number[];
  output: string;
  status: "fail-closed-runtime-reduction";
  coordinateAssignments: string[];
  renderedOutputCoordinate: Gemma4LiteralOutputCoordinateNavigation;
  predecessorCoordinates: Gemma4LiteralPredecessorCoordinateNavigation[];
  termTemplate: {
    reductionIndex: string;
    leftOperand: string;
    rightOperand: string;
    predicate?: string;
    mathematicalProduct: string;
  };
  terms: Gemma4LiteralRuntimeReductionProductTerm[];
  reduction: {
    index: string;
    domain: NonNullable<Gemma4LiteralScalarCalculation["reduction"]>["domains"][number];
    renderedWindow: { startInclusive: number; endExclusive: number };
    complete: boolean;
    omittedTerms?: number;
    provider: "Apple Accelerate SGEMM";
    scalarSchedule: "unpublished-fail-closed";
    productRounding: "unpublished-provider-boundary";
    accumulationOrder: "unpublished-provider-boundary";
    outputCast: string;
  };
  nonExecutableResult: string;
}

interface RuntimeReductionOperands {
  index: "head_feature" | "key_patch" | "key_slot";
  /** Undefined when the extent depends on a caller-provided tensor axis. */
  extent?: number;
  coordinateAssignments: string[];
  left: (index: number | string) => string;
  right: (index: number | string) => string;
  predicate?: (index: number | string) => string;
}

/** Lists the complete operation-classified native BMM boundary. */
export function listGemma4LiteralRuntimeReductionOperations(
  artifact: OpenGemma4CompositeLiteralArtifact,
): Gemma4LiteralRuntimeReductionOperation[] {
  const declaredClasses = new Set(artifact.authoritativeExecution.unresolvedNativeReduction.operationClasses);
  const operations = listGemma4LiteralOperations(artifact).flatMap((navigation): Gemma4LiteralRuntimeReductionOperation[] => {
    const reduction = navigation.scalarCalculation.reduction;
    if (reduction?.order !== "runtime-defined") return [];
    const operationClass = gemma4LiteralRuntimeReductionOperationClass(navigation.scope, navigation.operation);
    if (!declaredClasses.has(operationClass)) {
      throw new Error(`${navigation.operationId}: classe BMM ${operationClass} ausente do contrato autoritativo.`);
    }
    if (reduction.domains.length !== 1) {
      throw new Error(`${navigation.operationId}: BMM runtime-defined requer exatamente um domínio de redução.`);
    }
    return [{
      operationId: navigation.operationId,
      definitionId: navigation.definitionId ?? navigation.operationId,
      ...(navigation.invocationId ? { invocationId: navigation.invocationId } : {}),
      operation: navigation.operation,
      operationClass,
      output: navigation.output,
      outputDomain: structuredClone(navigation.outputDomain),
      reductionDomain: structuredClone(reduction.domains[0]!),
      provider: artifact.authoritativeExecution.unresolvedNativeReduction.provider,
      scalarSchedule: artifact.authoritativeExecution.unresolvedNativeReduction.scalarSchedule,
      auditability: "operand-products-addressable-reduction-fail-closed",
    }];
  });
  const encounteredClasses = new Set(operations.map((operation) => operation.operationClass));
  for (const operationClass of declaredClasses) {
    if (!encounteredClasses.has(operationClass)) {
      throw new Error(`Contrato autoritativo declara ${operationClass}, mas nenhuma atribuição runtime-defined compatível foi encontrada.`);
    }
  }
  return operations;
}

/**
 * Renders a concrete operand window for any of the five compatible Gemma 4
 * native-BMM classes. Strict scalar rendering still rejects the same operation:
 * this view ends before the unknown provider reduction and cannot yield output.
 */
export function renderGemma4LiteralRuntimeReductionAudit(
  artifact: OpenGemma4CompositeLiteralArtifact,
  request: Gemma4LiteralScalarViewRequest,
): Gemma4LiteralRuntimeReductionAudit {
  assertCoordinate(request.outputCoordinate);
  const navigation = listGemma4LiteralOperations(artifact).find((candidate) => candidate.operationId === request.operationId);
  if (!navigation) throw new Error(`Atribuição Gemma 4 literal não encontrada: ${request.operationId}.`);
  const reduction = navigation.scalarCalculation.reduction;
  if (reduction?.order !== "runtime-defined" || navigation.scalarCalculation.reproducibility !== "fail-closed-runtime-reduction") {
    throw new Error(`${request.operationId}: operação não possui redução runtime-defined para auditoria fail-closed.`);
  }
  if (reduction.domains.length !== 1) throw new Error(`${request.operationId}: auditoria BMM requer um único domínio de redução.`);
  const operationClass = gemma4LiteralRuntimeReductionOperationClass(navigation.scope, navigation.operation);
  if (!artifact.authoritativeExecution.unresolvedNativeReduction.operationClasses.includes(operationClass)) {
    throw new Error(`${request.operationId}: classe ${operationClass} não pertence à fronteira autoritativa incorporada.`);
  }
  const rendered = runtimeReductionOperands(navigation, request.outputCoordinate);
  const dynamicExtent = rendered.extent === undefined;
  if (dynamicExtent && (request.inputStart === undefined || request.inputCount === undefined)) {
    throw new Error(`${request.operationId}: redução dinâmica requer inputStart/inputCount para selecionar uma janela auditável.`);
  }
  const window = dynamicExtent
    ? explicitDynamicReductionWindow(request, request.operationId)
    : reductionWindow(request, rendered.extent!, request.operationId);
  const terms = Array.from({ length: window.end - window.start }, (_, offset) => {
    const reductionIndex = window.start + offset;
    const leftOperand = rendered.left(reductionIndex), rightOperand = rendered.right(reductionIndex);
    const predicate = rendered.predicate?.(reductionIndex);
    return {
      reductionIndex,
      leftOperand,
      rightOperand,
      ...(predicate ? { predicate } : {}),
      mathematicalProduct: predicate
        ? `term[${reductionIndex}] = ${predicate} ? REAL_PRODUCT(${leftOperand} * ${rightOperand}) : REAL(0)`
        : `term[${reductionIndex}] = REAL_PRODUCT(${leftOperand} * ${rightOperand})`,
    };
  });
  const output = indexed(navigation.output, request.outputCoordinate);
  const outputCast = navigation.scalarCalculation.dtypePolicy.outputDtype ?? "operation-declared";
  const templateLeft = rendered.left(rendered.index), templateRight = rendered.right(rendered.index);
  const templatePredicate = rendered.predicate?.(rendered.index);
  const complete = !dynamicExtent && window.start === 0 && window.end === rendered.extent;
  const renderedPrograms = [
    ...rendered.coordinateAssignments,
    ...(templatePredicate ? [templatePredicate] : []),
    templateLeft,
    templateRight,
    ...terms.flatMap((term) => [
      ...(term.predicate ? [term.predicate] : []),
      term.leftOperand,
      term.rightOperand,
      term.mathematicalProduct,
    ]),
  ];
  const nonExecutableResult = `${output} = ${outputCast}(APPLE_ACCELERATE_SGEMM_UNPUBLISHED_REDUCTION(term[complete declared domain])); intentionally unavailable until one authoritative class-wide scalar schedule is proven`;
  return {
    kind: "gemma4-literal-runtime-reduction-product-audit",
    schemaVersion: 1,
    sourceCheckpointAccessed: false,
    navigation,
    operationClass,
    outputCoordinate: [...request.outputCoordinate],
    output,
    status: "fail-closed-runtime-reduction",
    coordinateAssignments: rendered.coordinateAssignments,
    renderedOutputCoordinate: buildGemma4LiteralOutputCoordinateNavigation(
      navigation.output,
      [nonExecutableResult],
    ),
    predecessorCoordinates: buildGemma4LiteralPredecessorCoordinateNavigation(
      navigation.predecessors,
      renderedPrograms,
      complete,
    ),
    termTemplate: {
      reductionIndex: rendered.index,
      leftOperand: templateLeft,
      rightOperand: templateRight,
      ...(templatePredicate ? { predicate: templatePredicate } : {}),
      mathematicalProduct: templatePredicate
        ? `term[${rendered.index}] = ${templatePredicate} ? REAL_PRODUCT(${templateLeft} * ${templateRight}) : REAL(0)`
        : `term[${rendered.index}] = REAL_PRODUCT(${templateLeft} * ${templateRight})`,
    },
    terms,
    reduction: {
      index: rendered.index,
      domain: structuredClone(reduction.domains[0]!),
      renderedWindow: { startInclusive: window.start, endExclusive: window.end },
      complete,
      ...(!dynamicExtent ? { omittedTerms: rendered.extent! - terms.length } : {}),
      provider: artifact.authoritativeExecution.unresolvedNativeReduction.provider,
      scalarSchedule: artifact.authoritativeExecution.unresolvedNativeReduction.scalarSchedule,
      productRounding: "unpublished-provider-boundary",
      accumulationOrder: "unpublished-provider-boundary",
      outputCast,
    },
    nonExecutableResult,
  };
}

function runtimeReductionOperands(
  navigation: Gemma4LiteralOperationNavigation,
  coordinate: number[],
): RuntimeReductionOperands {
  const calculation = navigation.scalarCalculation;
  const reduction = calculation.reduction;
  if (!reduction || reduction.domains.length !== 1) throw new Error(`${navigation.operationId}: BMM requer um domínio de redução serializado.`);
  if (calculation.orderedInputs.length !== 2) throw new Error(`${navigation.operationId}: BMM requer exatamente dois orderedInputs.`);
  if (coordinate.length !== calculation.outputCoordinates.length) {
    throw new Error(`${navigation.operationId}: auditoria BMM requer coordenada [${calculation.outputCoordinates.join(",")}].`);
  }
  navigation.outputDomain.shape.forEach((size, axis) => {
    if (/^[0-9]+$/.test(size) && coordinate[axis]! >= Number(size)) {
      throw new Error(`${navigation.operationId}: coordenada ${calculation.outputCoordinates[axis]}=${coordinate[axis]} fora do domínio 0..${Number(size) - 1}.`);
    }
  });
  const index = reduction.domains[0]!.index;
  if (index !== "head_feature" && index !== "key_patch" && index !== "key_slot") {
    throw new Error(`${navigation.operationId}: índice BMM serializado não reconhecido: ${index}.`);
  }
  const outputSymbols = new Map(calculation.outputCoordinates.map((name, position) => [name, String(coordinate[position]!) ]));
  const localPrograms = new Map<string, Gemma4LiteralScalarExpression>();
  const outputProgram = calculation.statementPrograms.find((program) => program.targets.some((target) => target.role === "output"));
  if (!outputProgram || outputProgram !== calculation.statementPrograms.at(-1)) {
    throw new Error(`${navigation.operationId}: BMM não possui escrita terminal serializada.`);
  }
  for (const program of calculation.statementPrograms.slice(0, -1)) {
    if (program.kind !== "assignment" || program.targets.length !== 1 || program.targets[0]!.role !== "local" || program.targets[0]!.coordinates.length !== 0) {
      throw new Error(`${navigation.operationId}: prelude BMM contém statement não escalar.`);
    }
    localPrograms.set(program.targets[0]!.name, program.expression);
  }
  const located = locateSerializedReduction(outputProgram.expression, navigation.operationId);
  if (located.index !== index) throw new Error(`${navigation.operationId}: índice REDUCE diverge do domínio serializado.`);
  const inputAccesses = calculation.orderedInputs.map((input) => {
    const accesses = collectTensorAccesses(located.term, input);
    if (accesses.length !== 1) throw new Error(`${navigation.operationId}: termo BMM requer um único acesso ao input ${input}; encontrou ${accesses.length}.`);
    return accesses[0]!;
  });
  const renderEnvironment = (reductionIndex: number | string): ScalarRenderEnvironment => ({
    symbols: new Map([...outputSymbols, [index, String(reductionIndex)]]),
    locals: localPrograms,
    stack: new Set(),
  });
  const renderAccess = (access: Extract<Gemma4LiteralScalarExpression, { kind: "index" }>, reductionIndex: number | string): string =>
    renderTensorAccess(access, renderEnvironment(reductionIndex));
  const renderPredicate = (reductionIndex: number | string): string => {
    const environment = renderEnvironment(reductionIndex);
    const namedLocals = { ...environment, locals: new Map<string, Gemma4LiteralScalarExpression>() };
    return located.predicates.map((predicate) => renderScalarExpression(predicate, namedLocals).text).join(" && ");
  };
  const coordinateAssignments = [...localPrograms].map(([name, expression]) =>
    `${name}=${renderScalarExpression(expression, renderEnvironment(index)).text}`);
  const extent = reduction.domains[0]!.endExclusive.kind === "constant"
    ? reduction.domains[0]!.endExclusive.value
    : undefined;
  return {
    index,
    ...(extent === undefined ? {} : { extent }),
    coordinateAssignments,
    left: (reductionIndex) => renderAccess(inputAccesses[0]!, reductionIndex),
    right: (reductionIndex) => renderAccess(inputAccesses[1]!, reductionIndex),
    ...(located.predicates.length === 0 ? {} : { predicate: renderPredicate }),
  };
}

interface LocatedSerializedReduction {
  index: string;
  term: Gemma4LiteralScalarExpression;
  predicates: Gemma4LiteralScalarExpression[];
}

function locateSerializedReduction(expression: Gemma4LiteralScalarExpression, owner: string): LocatedSerializedReduction {
  const found: LocatedSerializedReduction[] = [];
  const visit = (current: Gemma4LiteralScalarExpression, predicates: Gemma4LiteralScalarExpression[]): void => {
    if (current.kind === "conditional") {
      const trueCount = countReductionCalls(current.whenTrue), falseCount = countReductionCalls(current.whenFalse);
      if (trueCount === 1 && falseCount === 0 && isSerializedZero(current.whenFalse)) {
        visit(current.whenTrue, [...predicates, current.condition]);
        return;
      }
    }
    if (current.kind === "call" && current.callee.kind === "identifier" && current.callee.name === "REDUCE") {
      const binding = current.arguments[0], rawTerm = current.arguments[1];
      if (current.arguments.length !== 2 || binding?.kind !== "named-argument" || !rawTerm) {
        throw new Error(`${owner}: REDUCE BMM serializado requer binding e termo.`);
      }
      if (rawTerm.kind === "conditional" && isSerializedZero(rawTerm.whenFalse)) {
        found.push({ index: binding.name, term: rawTerm.whenTrue, predicates: [...predicates, rawTerm.condition] });
      } else found.push({ index: binding.name, term: rawTerm, predicates });
      return;
    }
    for (const child of scalarExpressionChildren(current)) visit(child, predicates);
  };
  visit(expression, []);
  if (found.length !== 1) throw new Error(`${owner}: programa BMM requer exatamente um REDUCE serializado; encontrou ${found.length}.`);
  return found[0]!;
}

function countReductionCalls(expression: Gemma4LiteralScalarExpression): number {
  const self = expression.kind === "call" && expression.callee.kind === "identifier" && expression.callee.name === "REDUCE" ? 1 : 0;
  return self + scalarExpressionChildren(expression).reduce((total, child) => total + countReductionCalls(child), 0);
}

function isSerializedZero(expression: Gemma4LiteralScalarExpression): boolean {
  if (expression.kind === "literal") return expression.literalType === "number" && Number(expression.source) === 0;
  return expression.kind === "call" && expression.callee.kind === "identifier" &&
    (expression.callee.name === "F32" || expression.callee.name === "BF16") && expression.arguments.length === 1 && isSerializedZero(expression.arguments[0]!);
}

function collectTensorAccesses(
  expression: Gemma4LiteralScalarExpression,
  tensor: string,
): Array<Extract<Gemma4LiteralScalarExpression, { kind: "index" }>> {
  const result: Array<Extract<Gemma4LiteralScalarExpression, { kind: "index" }>> = [];
  const visit = (current: Gemma4LiteralScalarExpression): void => {
    if (current.kind === "index" && current.target.kind === "identifier" && current.target.name === tensor) result.push(current);
    scalarExpressionChildren(current).forEach(visit);
  };
  visit(expression);
  return result;
}

function scalarExpressionChildren(expression: Gemma4LiteralScalarExpression): Gemma4LiteralScalarExpression[] {
  switch (expression.kind) {
    case "literal": case "identifier": case "evaluate-invocation": return [];
    case "array": return expression.elements;
    case "unary": return [expression.operand];
    case "binary": return [expression.left, expression.right];
    case "conditional": return [expression.condition, expression.whenTrue, expression.whenFalse];
    case "call": return [expression.callee, ...expression.arguments];
    case "index": return [expression.target, ...expression.coordinates];
    case "member": return [expression.target];
    case "range-inclusive": return [expression.start, expression.end];
    case "filtered-domain": return [expression.domain, expression.predicate];
    case "named-argument": return [expression.value];
    case "ordered-loop": return [expression.body, expression.domain];
  }
}

interface ScalarRenderEnvironment {
  symbols: ReadonlyMap<string, string>;
  locals: ReadonlyMap<string, Gemma4LiteralScalarExpression>;
  stack: Set<string>;
}

interface RenderedScalarExpression {
  text: string;
  number?: number;
  boolean?: boolean;
}

function renderScalarExpression(
  expression: Gemma4LiteralScalarExpression,
  environment: ScalarRenderEnvironment,
): RenderedScalarExpression {
  switch (expression.kind) {
    case "literal": {
      if (expression.literalType === "number") return renderedNumber(Number(expression.source));
      if (expression.literalType === "boolean") return { text: expression.source, boolean: expression.source === "true" };
      return { text: expression.source };
    }
    case "identifier": {
      const symbol = environment.symbols.get(expression.name);
      if (symbol !== undefined) return numericText(symbol);
      const local = environment.locals.get(expression.name);
      if (!local) return { text: expression.name };
      if (environment.stack.has(expression.name)) throw new Error(`Programa BMM possui local recursivo ${expression.name}.`);
      environment.stack.add(expression.name);
      try { return renderScalarExpression(local, environment); } finally { environment.stack.delete(expression.name); }
    }
    case "unary": {
      const operand = renderScalarExpression(expression.operand, environment);
      if (expression.operator === "-" && operand.number !== undefined) return renderedNumber(-operand.number);
      if (expression.operator === "+" && operand.number !== undefined) return renderedNumber(operand.number);
      if (expression.operator === "!" && operand.boolean !== undefined) return { text: String(!operand.boolean), boolean: !operand.boolean };
      return { text: `${expression.operator}${parenthesize(operand.text)}` };
    }
    case "binary": return renderBinaryExpression(expression.operator,
      renderScalarExpression(expression.left, environment), renderScalarExpression(expression.right, environment));
    case "call": {
      const callee = renderScalarExpression(expression.callee, environment).text;
      const arguments_ = expression.arguments.map((argument) => renderScalarExpression(argument, environment));
      if (callee === "floor" && arguments_.length === 1 && arguments_[0]!.number !== undefined) return renderedNumber(Math.floor(arguments_[0]!.number));
      return { text: `${callee}(${arguments_.map((argument) => argument.text).join(",")})` };
    }
    case "index": {
      const target = renderScalarExpression(expression.target, environment).text;
      const coordinates = expression.coordinates.map((coordinate) => renderScalarExpression(coordinate, environment).text);
      return { text: `${target}[${coordinates.join(",")}]` };
    }
    case "member": return { text: `${renderScalarExpression(expression.target, environment).text}.${expression.member}` };
    case "conditional": {
      const condition = renderScalarExpression(expression.condition, environment).text;
      const whenTrue = renderScalarExpression(expression.whenTrue, environment).text;
      const whenFalse = renderScalarExpression(expression.whenFalse, environment).text;
      return { text: `${condition} ? ${whenTrue} : ${whenFalse}` };
    }
    case "array": return { text: `[${expression.elements.map((entry) => renderScalarExpression(entry, environment).text).join(",")}]` };
    case "range-inclusive": return { text: `${renderScalarExpression(expression.start, environment).text}..${renderScalarExpression(expression.end, environment).text}` };
    case "filtered-domain": return { text: `${renderScalarExpression(expression.domain, environment).text} where ${renderScalarExpression(expression.predicate, environment).text}` };
    case "named-argument": return { text: `${expression.name}=${renderScalarExpression(expression.value, environment).text}` };
    case "ordered-loop": return { text: `${renderScalarExpression(expression.body, environment).text},${expression.index}=${renderScalarExpression(expression.domain, environment).text} ascending` };
    case "evaluate-invocation": throw new Error("Programa BMM não aceita EVALUATE na redução nativa.");
  }
}

function renderTensorAccess(
  access: Extract<Gemma4LiteralScalarExpression, { kind: "index" }>,
  environment: ScalarRenderEnvironment,
): string {
  const target = renderScalarExpression(access.target, environment).text;
  const coordinates = access.coordinates.map((coordinate) => renderScalarExpression(coordinate, environment));
  for (const coordinate of coordinates) if (coordinate.number !== undefined && !Number.isSafeInteger(coordinate.number)) {
    throw new Error(`Programa BMM produziu coordenada não inteira segura: ${coordinate.number}.`);
  }
  return `${target}[${coordinates.map((coordinate) => coordinate.text).join(",")}]`;
}

function renderBinaryExpression(operator: string, left: RenderedScalarExpression, right: RenderedScalarExpression): RenderedScalarExpression {
  if (left.number !== undefined && right.number !== undefined) {
    switch (operator) {
      case "+": return renderedNumber(left.number + right.number);
      case "-": return renderedNumber(left.number - right.number);
      case "*": return renderedNumber(left.number * right.number);
      case "/": return renderedNumber(left.number / right.number);
      case "%": return renderedNumber(left.number % right.number);
      case "**": return renderedNumber(left.number ** right.number);
      case "<": return renderedBoolean(left.number < right.number);
      case "<=": return renderedBoolean(left.number <= right.number);
      case ">": return renderedBoolean(left.number > right.number);
      case ">=": return renderedBoolean(left.number >= right.number);
      case "==": return renderedBoolean(left.number === right.number);
      case "!=": return renderedBoolean(left.number !== right.number);
    }
  }
  if (operator === "&&" && left.boolean !== undefined) return left.boolean ? right : renderedBoolean(false);
  if (operator === "||" && left.boolean !== undefined) return left.boolean ? renderedBoolean(true) : right;
  return { text: `${parenthesize(left.text)}${operator}${parenthesize(right.text)}` };
}

function numericText(value: string): RenderedScalarExpression {
  const number = Number(value);
  return Number.isFinite(number) ? renderedNumber(number) : { text: value };
}

function renderedNumber(value: number): RenderedScalarExpression {
  if (!Number.isFinite(value)) throw new Error(`Programa BMM produziu número não finito: ${value}.`);
  return { text: String(value), number: value };
}

function renderedBoolean(value: boolean): RenderedScalarExpression { return { text: String(value), boolean: value }; }
function parenthesize(value: string): string { return /^-?[A-Za-z0-9_./:-]+$/.test(value) ? value : `(${value})`; }

function explicitDynamicReductionWindow(request: Gemma4LiteralScalarViewRequest, id: string): { start: number; end: number } {
  const start = request.inputStart!, count = request.inputCount!;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(count) || start < 0 || count <= 0) {
    throw new Error(`${id}: janela dinâmica ${start}+${count} requer inteiros não negativos e count positivo.`);
  }
  return { start, end: start + count };
}

function reductionWindow(request: Gemma4LiteralScalarViewRequest, width: number, id: string): { start: number; end: number } {
  if ((request.inputStart === undefined) !== (request.inputCount === undefined)) throw new Error(`${id}: inputStart/inputCount devem aparecer juntos.`);
  const start = request.inputStart ?? 0, count = request.inputCount ?? width;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(count) || start < 0 || count <= 0 || start + count > width) {
    throw new Error(`${id}: janela ${start}+${count} fora de 0..${width}.`);
  }
  return { start, end: start + count };
}

function indexed(name: string, coordinate: ReadonlyArray<number | string>): string {
  return `${name}[${coordinate.join(",")}]`;
}

function assertCoordinate(coordinate: number[]): void {
  if (coordinate.length === 0 || coordinate.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new Error("Auditoria BMM requer coordenada não negativa.");
  }
}
