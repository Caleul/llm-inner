import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { parametricNodeDependencies, type Gemma4ParametricRealNode } from "./gemma4-parametric-real-expression.js";

export interface Gemma4GlobalSsaOutput {
  assignment: string;
  value: string;
  parameters: string[];
  coordinate: string[];
  finalQuantization: "BF16-round-to-nearest-ties-to-even";
}

export interface Gemma4GlobalSsaStatement {
  target: string;
  expression: string;
  node: Gemma4ParametricRealNode;
}

export interface Gemma4GlobalSsaFunction {
  functionId: string;
  operationId: string;
  ordinal: number;
  output: string;
  parameters: Array<{ name: string; node: string }>;
  root: string;
  predecessorFunctions: string[];
  closureKind: "global-output-closure" | "standalone-runtime-reduction";
  operandBoundaries: string[];
}

export interface Gemma4GlobalSsaRootResolution {
  root: Gemma4GlobalSsaStatement;
  dependencies: string[];
  fragment: Gemma4GlobalSsaExpressionFragment;
  calledFunction?: Gemma4GlobalSsaFunction & {
    body: Gemma4GlobalSsaStatement;
    bodyDependencies: string[];
    bodyFragment: Gemma4GlobalSsaExpressionFragment;
  };
}

export interface Gemma4GlobalSsaExpressionFragment {
  root: string;
  kind: Gemma4ParametricRealNode["kind"];
  expression: string;
  expandedExpression: string;
  dependencies: string[];
  children: Gemma4GlobalSsaExpressionFragment[];
  functionCall?: string;
  truncated: boolean;
}

/**
 * Reads one output binding from the large, one-line SSA export without parsing
 * its hundreds of MiB of statements and functions into memory.
 */
export async function readGemma4GlobalSsaOutput(path: string, ordinal: number): Promise<Gemma4GlobalSsaOutput> {
  if (!Number.isSafeInteger(ordinal) || ordinal < 0) throw new Error("Ordinal de saída SSA deve ser inteiro não negativo.");
  return readArrayObject(path, "outputs", (value, currentOrdinal) => currentOrdinal === ordinal ? validateOutput(value, ordinal) : undefined,
    `SSA global não possui output no ordinal ${ordinal}.`);
}

/** Resolves a content-addressed node without materializing the complete SSA. */
export async function readGemma4GlobalSsaStatement(path: string, target: string): Promise<Gemma4GlobalSsaStatement> {
  validateRoot(target, "Raiz SSA");
  return readArrayObject(path, "statements", (value) => {
    if (!isRecord(value) || value.target !== target) return undefined;
    return validateStatement(value, target);
  }, `SSA global não possui statement para ${target}.`);
}

/** Resolves the operation closure referenced by a function-call node. */
export async function readGemma4GlobalSsaFunction(path: string, functionId: string): Promise<Gemma4GlobalSsaFunction> {
  if (!/^operation:[A-Za-z0-9_.:/-]+$/.test(functionId)) throw new Error(`functionId SSA inválido: ${functionId}.`);
  return readArrayObject(path, "functions", (value) => {
    if (!isRecord(value) || value.functionId !== functionId) return undefined;
    return validateFunction(value, functionId);
  }, `SSA global não possui função ${functionId}.`);
}

/**
 * Turns EVAL_EXACT_DAG(root, x) into the concrete root expression and, for an
 * operation call, the exact function body root used by the scalar evaluator.
 */
export async function resolveGemma4GlobalSsaRoot(path: string, root: string): Promise<Gemma4GlobalSsaRootResolution> {
  const statement = await readGemma4GlobalSsaStatement(path, root);
  const dependencies = parametricNodeDependencies(statement.node);
  const fragment = await expandGemma4GlobalSsaExpression(path, root, { maximumDepth: 2, maximumNodes: 12 });
  if (statement.node.kind !== "function-call") return { root: statement, dependencies, fragment };
  const function_ = await readGemma4GlobalSsaFunction(path, statement.node.functionId);
  const body = await readGemma4GlobalSsaStatement(path, function_.root);
  const bodyFragment = await expandGemma4GlobalSsaExpression(path, function_.root, { maximumDepth: 4, maximumNodes: 24 });
  return { root: statement, dependencies, fragment, calledFunction: { ...function_, body, bodyDependencies: parametricNodeDependencies(body.node), bodyFragment } };
}

/** Builds a bounded, recursively substituted mathematical view of a DAG root. */
export async function expandGemma4GlobalSsaExpression(
  path: string,
  root: string,
  options: { maximumDepth: number; maximumNodes: number },
): Promise<Gemma4GlobalSsaExpressionFragment> {
  if (!Number.isSafeInteger(options.maximumDepth) || options.maximumDepth < 0 || !Number.isSafeInteger(options.maximumNodes) || options.maximumNodes < 1) {
    throw new Error("Limites de expansão SSA inválidos.");
  }
  const statements = new Map<string, Promise<Gemma4GlobalSsaStatement>>(); let nodes = 0;
  const read = (id: string) => {
    let pending = statements.get(id);
    if (!pending) { pending = readGemma4GlobalSsaStatement(path, id); statements.set(id, pending); }
    return pending;
  };
  const visit = async (id: string, depth: number): Promise<Gemma4GlobalSsaExpressionFragment> => {
    const statement = await read(id), dependencies = parametricNodeDependencies(statement.node);
    nodes += 1;
    const canDescend = depth < options.maximumDepth && nodes < options.maximumNodes;
    const children: Gemma4GlobalSsaExpressionFragment[] = [];
    if (canDescend) for (const dependency of dependencies) {
      if (nodes >= options.maximumNodes) break;
      children.push(await visit(dependency, depth + 1));
    }
    let expandedExpression = statement.expression;
    for (const child of children) expandedExpression = expandedExpression.replaceAll(child.root, `(${child.expandedExpression})`);
    const functionCall = statement.node.kind === "function-call" ? statement.node.functionId : undefined;
    return { root: id, kind: statement.node.kind, expression: statement.expression, expandedExpression, dependencies, children, ...(functionCall ? { functionCall } : {}), truncated: functionCall !== undefined || children.length < dependencies.length || children.some((child) => child.truncated) };
  };
  return visit(root, 0);
}

async function readArrayObject<T>(
  path: string,
  array: "statements" | "functions" | "outputs",
  select: (value: unknown, ordinal: number) => T | undefined,
  absentMessage: string,
): Promise<T> {
  const marker = Buffer.from(`"${array}":[`), stream = createReadStream(path);
  let carry: Buffer<ArrayBufferLike> = Buffer.alloc(0); let found = false, objectDepth = 0, inString = false, escaped = false, currentOrdinal = -1;
  let bytes: number[] = [];
  for await (const chunk_ of stream) {
    const chunk = chunk_ as Buffer; let data = chunk;
    if (!found) {
      const searchable = carry.length ? Buffer.concat([carry, chunk]) : chunk, markerIndex = searchable.indexOf(marker);
      if (markerIndex < 0) { carry = searchable.subarray(Math.max(0, searchable.length - marker.length + 1)); continue; }
      found = true; data = searchable.subarray(markerIndex + marker.length);
    }
    for (const byte of data) {
      if (!inString && objectDepth === 0) {
        if (byte === 0x5d) throw new Error(absentMessage);
        if (byte !== 0x7b) continue;
        currentOrdinal += 1; objectDepth = 1; bytes = [byte];
        continue;
      }
      bytes.push(byte);
      if (inString) {
        if (escaped) escaped = false;
        else if (byte === 0x5c) escaped = true;
        else if (byte === 0x22) inString = false;
        continue;
      }
      if (byte === 0x22) inString = true;
      else if (byte === 0x7b) objectDepth += 1;
      else if (byte === 0x7d) {
        objectDepth -= 1;
        if (objectDepth === 0) {
          const selected = select(JSON.parse(Buffer.from(bytes).toString("utf8")) as unknown, currentOrdinal);
          if (selected !== undefined) return selected;
          bytes = [];
        }
      }
    }
  }
  if (!found) throw new Error(`SSA global não contém o array ${array}.`);
  throw new Error(`SSA global terminou antes de fechar o array ${array}.`);
}

function validateOutput(value: unknown, ordinal: number): Gemma4GlobalSsaOutput {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`Output SSA ${ordinal} não é objeto.`);
  const output = value as Partial<Gemma4GlobalSsaOutput>;
  if (typeof output.assignment !== "string" || !/^calc_[a-z0-9_]+_\d+$/.test(output.assignment) || typeof output.value !== "string" || !/^sha256:[0-9a-f]{64}$/.test(output.value) ||
    !Array.isArray(output.parameters) || output.parameters.some((entry) => typeof entry !== "string") || !Array.isArray(output.coordinate) || output.coordinate.some((entry) => typeof entry !== "string" || !/^sha256:[0-9a-f]{64}$/.test(entry)) ||
    output.finalQuantization !== "BF16-round-to-nearest-ties-to-even") throw new Error(`Output SSA ${ordinal} possui contrato inválido.`);
  return output as Gemma4GlobalSsaOutput;
}

function validateStatement(value: Record<string, unknown>, target: string): Gemma4GlobalSsaStatement {
  if (typeof value.expression !== "string" || value.expression.length === 0 || !isRecord(value.node) || value.node.id !== target || typeof value.node.kind !== "string") {
    throw new Error(`Statement SSA ${target} possui contrato inválido.`);
  }
  const { id, ...payload } = value.node;
  const expected = `sha256:${createHash("sha256").update(JSON.stringify(payload)).digest("hex")}`;
  if (expected !== target) throw new Error(`Statement SSA ${target} não corresponde ao conteúdo autenticado do nó.`);
  let dependencies: string[];
  try { dependencies = parametricNodeDependencies(value.node as unknown as Gemma4ParametricRealNode); }
  catch { throw new Error(`Statement SSA ${target} possui kind não executável.`); }
  if (!Array.isArray(dependencies) || dependencies.some((dependency) => typeof dependency !== "string" || !/^sha256:[0-9a-f]{64}$/.test(dependency))) {
    throw new Error(`Statement SSA ${target} possui dependências inválidas.`);
  }
  return value as unknown as Gemma4GlobalSsaStatement;
}

function validateFunction(value: Record<string, unknown>, functionId: string): Gemma4GlobalSsaFunction {
  if (value.operationId === undefined || typeof value.operationId !== "string" || !Number.isSafeInteger(value.ordinal) || typeof value.output !== "string" ||
    !Array.isArray(value.parameters) || value.parameters.some((parameter) => !isRecord(parameter) || typeof parameter.name !== "string" || typeof parameter.node !== "string" || !/^sha256:[0-9a-f]{64}$/.test(parameter.node)) ||
    typeof value.root !== "string" || !/^sha256:[0-9a-f]{64}$/.test(value.root) || !Array.isArray(value.predecessorFunctions) || value.predecessorFunctions.some((entry) => typeof entry !== "string") ||
    (value.closureKind !== "global-output-closure" && value.closureKind !== "standalone-runtime-reduction") || !Array.isArray(value.operandBoundaries) || value.operandBoundaries.some((entry) => typeof entry !== "string")) {
    throw new Error(`Função SSA ${functionId} possui contrato inválido.`);
  }
  return value as unknown as Gemma4GlobalSsaFunction;
}

function validateRoot(value: string, label: string): void {
  if (!/^sha256:[0-9a-f]{64}$/.test(value)) throw new Error(`${label} inválida: ${value}.`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
