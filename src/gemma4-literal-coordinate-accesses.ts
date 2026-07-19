export type Gemma4LiteralCoordinateAccess =
  | {
    kind: "tensor-element";
    expression: string;
    coordinates: string[];
  }
  | {
    kind: "tensor-shape";
    expression: string;
    axis: string;
  }
  | {
    kind: "whole-value";
    expression: string;
  };

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
      result.push({ offset, access: { kind, expression, axis: content } });
    } else {
      const coordinates = splitCoordinates(content);
      if (coordinates.length === 0) throw new Error(`Acesso tensorial sem coordenadas em ${expression}.`);
      result.push({ offset, access: { kind, expression, coordinates } });
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
