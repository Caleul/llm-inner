import { open, stat, type FileHandle } from "node:fs/promises";
import type {
  Gemma4CompositeLiteralCalculationProgram,
  Gemma4CompositeLiteralInput,
  Gemma4LiteralGreedyGenerationProgram,
  Gemma4CompositeLiteralPayloadIntegrityEntry,
  Gemma4CompositeUnreachableConstant,
} from "./gemma4-composite-literal.js";
import { validateGemma4CompositeLiteralInputs, validateGemma4CompositeLiteralNumericPolicy, validateGemma4CompositeLiteralStructure } from "./gemma4-composite-literal.js";
import type { LiteralConstant, LiteralDenseStorageDecodeAssignment, LiteralTensorReader } from "./literal.js";
import type { Gemma4CompositeProgram } from "./gemma4-composite.js";
import { validateGemma4LiteralCalculationDomains, type Gemma4LiteralCalculationDomains } from "./gemma4-literal-domains.js";
import {
  validateGemma4LiteralLearnedOperandBindings,
  type Gemma4LiteralLearnedOperandBindings,
} from "./gemma4-literal-learned-operands.js";
import {
  validateGemma4LiteralScalarCalculations,
  type Gemma4LiteralScalarCalculations,
} from "./gemma4-literal-scalar-calculations.js";
import {
  validateGemma4LiteralCalculationGraph,
  type Gemma4LiteralCalculationGraph,
} from "./gemma4-literal-calculation-graph.js";
import {
  validateGemma4LiteralNumericLiterals,
  type Gemma4LiteralNumericLiterals,
} from "./gemma4-literal-numeric-literals.js";
import type { TensorInfo } from "./types.js";
import {
  validateGemma4LiteralSourceIdentity,
  type Gemma4LiteralSourceIdentity,
} from "./gemma4-literal-source-identity.js";

const CONSTANTS_MARKER = Buffer.from(",\"constants\":[", "ascii");
const PAYLOAD_MARKER = Buffer.from(",\"payloadBase64\":\"", "ascii");
const QUOTE = '"'.charCodeAt(0);
const SCAN_CHUNK_BYTES = 1024 * 1024;
const MAX_STRUCTURAL_JSON_BYTES = 64 * 1024 * 1024;

export interface IndexedLiteralConstant extends Omit<LiteralConstant, "payloadBase64"> {
  payloadOffset: number;
  payloadBase64Characters: number;
  payloadBytes: number;
}

/**
 * The non-payload portion of a Gemma 4 literal artifact, plus byte offsets
 * for every base64 storage payload. It deliberately has no `payloadBase64`
 * fields, so opening a 20 GiB artifact does not build a 20 GiB V8 object.
 */
export interface Gemma4CompositeLiteralArtifactIndex {
  schemaVersion: 9;
  artifact: string;
  artifactBytes: number;
  sourceIdentity: Gemma4LiteralSourceIdentity;
  constants: ReadonlyMap<string, IndexedLiteralConstant>;
  storageDecoders: LiteralDenseStorageDecodeAssignment[];
  unreachableConstants: Gemma4CompositeUnreachableConstant[];
  program: Gemma4CompositeProgram;
  assignments: Gemma4CompositeLiteralCalculationProgram["assignments"];
  calculationDomains: Gemma4LiteralCalculationDomains;
  learnedOperands: Gemma4LiteralLearnedOperandBindings;
  scalarCalculations: Gemma4LiteralScalarCalculations;
  numericLiterals: Gemma4LiteralNumericLiterals;
  calculationGraph: Gemma4LiteralCalculationGraph;
  outputs: Gemma4CompositeLiteralCalculationProgram["outputs"];
  generation: Gemma4LiteralGreedyGenerationProgram;
  inputs: Gemma4CompositeLiteralInput[];
  numericPolicy: Gemma4CompositeLiteralCalculationProgram["numericPolicy"];
  /** Optional for compatibility with artifacts emitted before payload commitments. */
  payloadIntegrity?: ReadonlyMap<string, Gemma4CompositeLiteralPayloadIntegrityEntry>;
}

export interface OpenGemma4CompositeLiteralArtifact extends Gemma4CompositeLiteralArtifactIndex {
  /** Reads original storage bytes entirely from the literal artifact. */
  readTensorBytes(tensor: TensorInfo): Promise<Buffer>;
  /** Reads an exact storage range without decoding or holding the full tensor. */
  readTensorBytesRange(tensor: TensorInfo, offset: number, byteLength: number): Promise<Buffer>;
  close(): Promise<void>;
}

/**
 * Opens a streamed Gemma 4 literal artifact without JSON.parse. The writer's
 * base64 field has a deliberately unambiguous quote terminator, letting this
 * scanner retain only constant metadata and the final structural tail.
 */
export async function openGemma4CompositeLiteralArtifact(artifact: string): Promise<OpenGemma4CompositeLiteralArtifact> {
  const info = await stat(artifact);
  if (!info.isFile() || info.size <= 0) throw new Error(`Artefato literal Gemma 4 inválido: ${artifact}.`);
  const file = await open(artifact, "r");
  try {
    const constantsMarker = await findSequence(file, 0, info.size, CONSTANTS_MARKER);
    if (constantsMarker === -1) throw new Error("Artefato literal Gemma 4 não declara constants.");
    const header = parsePrefixJson(await readStructuralRange(file, 0, constantsMarker, "cabeçalho"), "cabeçalho") as Partial<Gemma4CompositeLiteralCalculationProgram>;
    assertHeader(header);

    const constants = new Map<string, IndexedLiteralConstant>();
    let cursor = constantsMarker + CONSTANTS_MARKER.length;
    while (true) {
      const delimiter = await readRange(file, cursor, cursor + 1);
      if (delimiter.equals(Buffer.from("]"))) {
        cursor += 1;
        break;
      }
      if (!delimiter.equals(Buffer.from("{"))) throw new Error(`Artefato literal Gemma 4 possui delimitador de constante inválido em ${cursor}.`);
      const payloadMarker = await findSequence(file, cursor, info.size, PAYLOAD_MARKER);
      if (payloadMarker === -1) throw new Error("Artefato literal Gemma 4 terminou sem payloadBase64.");
      const metadata = parsePrefixJson(await readStructuralRange(file, cursor, payloadMarker, "metadados de constante"), "metadados de constante") as Omit<LiteralConstant, "payloadBase64">;
      const payloadOffset = payloadMarker + PAYLOAD_MARKER.length;
      const payloadEnd = await findByte(file, payloadOffset, info.size, QUOTE);
      if (payloadEnd === -1) throw new Error(`${metadata.name ?? "constante"}: payloadBase64 sem aspas finais.`);
      const payloadBase64Characters = payloadEnd - payloadOffset;
      if (payloadBase64Characters === 0 || payloadBase64Characters % 4 !== 0) throw new Error(`${metadata.name ?? "constante"}: tamanho base64 inválido.`);
      const padding = await base64Padding(file, payloadEnd, payloadBase64Characters);
      const payloadBytes = payloadBase64Characters / 4 * 3 - padding;
      const constant: IndexedLiteralConstant = { ...metadata, payloadOffset, payloadBase64Characters, payloadBytes };
      assertDenseConstant(constant);
      if (constants.has(constant.name)) throw new Error(`Artefato literal Gemma 4 contém constante duplicada: ${constant.name}.`);
      constants.set(constant.name, constant);
      const suffix = await readRange(file, payloadEnd, payloadEnd + 2);
      if (suffix.equals(Buffer.from("\"}"))) cursor = payloadEnd + 2;
      else throw new Error(`${constant.name}: payloadBase64 não termina na constante esperada.`);
      const next = await readRange(file, cursor, cursor + 1);
      if (next.equals(Buffer.from(","))) cursor += 1;
      else if (next.equals(Buffer.from("]"))) { cursor += 1; break; }
      else throw new Error(`${constant.name}: delimitador após payloadBase64 inválido.`);
    }
    if (constants.size === 0) throw new Error("Artefato literal Gemma 4 não contém constantes.");
    const tail = parseTailJson(await readStructuralRange(file, cursor, info.size, "cauda estrutural"), "cauda estrutural") as Partial<Gemma4CompositeLiteralCalculationProgram>;
    const index = buildIndex(artifact, info.size, header, tail, constants);
    return {
      ...index,
      readTensorBytes: (tensor) => readTensorBytes(file, index.constants, tensor),
      readTensorBytesRange: (tensor, offset, byteLength) => readTensorBytesRange(file, index.constants, tensor, offset, byteLength),
      close: async () => file.close(),
    };
  } catch (error) {
    await file.close();
    throw error;
  }
}

/** Lets an opened artifact participate in existing storage-reader contracts. */
export function asLiteralTensorReader(artifact: OpenGemma4CompositeLiteralArtifact): LiteralTensorReader {
  return { readTensorBytes: artifact.readTensorBytes, readTensorBytesRange: artifact.readTensorBytesRange };
}

function buildIndex(
  artifact: string,
  artifactBytes: number,
  header: Partial<Gemma4CompositeLiteralCalculationProgram>,
  tail: Partial<Gemma4CompositeLiteralCalculationProgram>,
  constants: ReadonlyMap<string, IndexedLiteralConstant>,
): Gemma4CompositeLiteralArtifactIndex {
  if (!Array.isArray(tail.storageDecoders) || !Array.isArray(tail.unreachableConstants) || !tail.program || !tail.assignments || !tail.calculationDomains || !tail.learnedOperands || !tail.scalarCalculations || !tail.numericLiterals || !tail.calculationGraph || !tail.outputs || !tail.generation) {
    throw new Error("Artefato literal Gemma 4 não declara a cauda semântica completa.");
  }
  const storageDecoders = tail.storageDecoders as LiteralDenseStorageDecodeAssignment[];
  if (storageDecoders.length !== constants.size) throw new Error("Artefato literal Gemma 4 deve declarar um decoder denso por constante.");
  for (const decoder of storageDecoders) assertDenseDecoder(decoder, constants);
  validateGemma4CompositeLiteralNumericPolicy(
    header.numericPolicy as Gemma4CompositeLiteralCalculationProgram["numericPolicy"],
    tail.program as Gemma4CompositeProgram,
  );
  validateGemma4CompositeLiteralInputs(header.inputs as Gemma4CompositeLiteralInput[]);
  validateGemma4LiteralSourceIdentity(header.sourceIdentity as Gemma4LiteralSourceIdentity);
  validateGemma4CompositeLiteralStructure(
    tail.program as Gemma4CompositeProgram,
    tail.assignments as Gemma4CompositeLiteralCalculationProgram["assignments"],
    tail.outputs as Gemma4CompositeLiteralCalculationProgram["outputs"],
    tail.generation as Gemma4LiteralGreedyGenerationProgram,
    new Map([...constants].map(([name, constant]) => [name, { ...constant, payloadBase64: "" }])),
    tail.unreachableConstants as Gemma4CompositeUnreachableConstant[],
  );
  validateGemma4LiteralCalculationDomains(tail.calculationDomains as Gemma4LiteralCalculationDomains, tail.program as Gemma4CompositeProgram);
  validateGemma4LiteralLearnedOperandBindings(tail.learnedOperands as Gemma4LiteralLearnedOperandBindings, tail.program as Gemma4CompositeProgram);
  validateGemma4LiteralScalarCalculations(tail.scalarCalculations as Gemma4LiteralScalarCalculations, tail.program as Gemma4CompositeProgram);
  validateGemma4LiteralNumericLiterals(
    tail.numericLiterals as Gemma4LiteralNumericLiterals,
    tail.scalarCalculations as Gemma4LiteralScalarCalculations,
    (tail.generation as Gemma4LiteralGreedyGenerationProgram).scalarCalculations,
  );
  validateGemma4LiteralCalculationGraph(tail.calculationGraph as Gemma4LiteralCalculationGraph, tail.program as Gemma4CompositeProgram);
  const payloadIntegrity = tail.payloadIntegrity === undefined
    ? undefined
    : validatePayloadIntegrity(tail.payloadIntegrity, constants);
  return {
    schemaVersion: 9,
    artifact,
    artifactBytes,
    sourceIdentity: structuredClone(header.sourceIdentity as Gemma4LiteralSourceIdentity),
    constants,
    storageDecoders,
    unreachableConstants: tail.unreachableConstants as Gemma4CompositeUnreachableConstant[],
    program: tail.program as Gemma4CompositeProgram,
    assignments: tail.assignments as Gemma4CompositeLiteralCalculationProgram["assignments"],
    calculationDomains: tail.calculationDomains as Gemma4LiteralCalculationDomains,
    learnedOperands: tail.learnedOperands as Gemma4LiteralLearnedOperandBindings,
    scalarCalculations: tail.scalarCalculations as Gemma4LiteralScalarCalculations,
    numericLiterals: tail.numericLiterals as Gemma4LiteralNumericLiterals,
    calculationGraph: tail.calculationGraph as Gemma4LiteralCalculationGraph,
    outputs: tail.outputs as Gemma4CompositeLiteralCalculationProgram["outputs"],
    generation: tail.generation as Gemma4LiteralGreedyGenerationProgram,
    inputs: header.inputs as Gemma4CompositeLiteralInput[],
    numericPolicy: header.numericPolicy as Gemma4CompositeLiteralCalculationProgram["numericPolicy"],
    ...(payloadIntegrity ? { payloadIntegrity } : {}),
  };
}

function validatePayloadIntegrity(
  entries: unknown,
  constants: ReadonlyMap<string, IndexedLiteralConstant>,
): ReadonlyMap<string, Gemma4CompositeLiteralPayloadIntegrityEntry> {
  if (!Array.isArray(entries) || entries.length !== constants.size) {
    throw new Error("Artefato literal Gemma 4 possui compromissos de integridade de payload incompletos.");
  }
  const integrity = new Map<string, Gemma4CompositeLiteralPayloadIntegrityEntry>();
  for (const entry of entries) {
    if (!entry || typeof entry !== "object") throw new Error("Artefato literal Gemma 4 possui compromisso de integridade inválido.");
    const candidate = entry as Partial<Gemma4CompositeLiteralPayloadIntegrityEntry>;
    const constant = candidate.name ? constants.get(candidate.name) : undefined;
    if (!constant || integrity.has(constant.name) || candidate.payloadBytes !== constant.payloadBytes ||
      typeof candidate.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(candidate.sha256)) {
      throw new Error(`${candidate.name ?? "constante"}: compromisso de integridade de payload inválido.`);
    }
    integrity.set(constant.name, { name: constant.name, payloadBytes: constant.payloadBytes, sha256: candidate.sha256 });
  }
  if (integrity.size !== constants.size) throw new Error("Artefato literal Gemma 4 não possui compromisso para toda constante incorporada.");
  return integrity;
}

function assertHeader(header: Partial<Gemma4CompositeLiteralCalculationProgram>): void {
  const policy = header.numericPolicy;
  const f32 = policy?.inputDtype === "I32/F32/BOOL" && policy.computeDtype === "F32" && policy.accumulationDtype === "F32" &&
    policy.outputDtype === "F32" && policy.scalarSemantics === "IEEE-754 binary32; host libm results rounded to F32";
  const operationDeclared = policy?.inputDtype === "I32/F32/BOOL" && policy.computeDtype === "F32" && policy.accumulationDtype === "F32" &&
    policy.outputDtype === "operation-declared" && policy.scalarSemantics === "IEEE-754 binary32 reductions; each operation declares its F32 or BF16 result cast";
  const operationAccumulationDeclared = policy?.inputDtype === "I32/F32/BOOL" && policy.computeDtype === "F32" && policy.accumulationDtype === "operation-declared" &&
    policy.outputDtype === "operation-declared" &&
    (policy.scalarSemantics === "IEEE-754 binary32 products; each operation declares its ordered-scalar or interleaved-lane F32/F64 reduction and F32 or BF16 result cast" ||
      policy.scalarSemantics === "IEEE-754 binary32 products; each operation declares its ordered-scalar, contiguous blocked-term, or interleaved-lane F32/F64 reduction and F32 or BF16 result cast" ||
      policy.scalarSemantics === "IEEE-754 binary32; each operation declares ordered-scalar, contiguous blocked-term, separately-rounded F32-lane, or fused-multiply-add reduction and its F32 or BF16 result cast" ||
      policy.scalarSemantics === "IEEE-754 binary32 products; each operation declares its ordered-scalar, contiguous blocked-term, blocked tiled-lane, or interleaved-lane F32/F64 reduction and F32 or BF16 result cast" ||
      policy.scalarSemantics === "IEEE-754 binary32; each operation declares ordered-scalar, contiguous blocked-term, blocked tiled-lane, separately-rounded F32-lane, or fused-multiply-add reduction and its F32 or BF16 result cast");
  if (header.schemaVersion !== 9 || header.kind !== "gemma4-composite-literal-calculation-program" || header.sourceFormat !== "safetensors" ||
    !header.sourceIdentity || !Array.isArray(header.inputs) || !policy || (!f32 && !operationDeclared && !operationAccumulationDeclared)) {
    throw new Error("Artefato literal Gemma 4 possui cabeçalho ou política numérica inválida.");
  }
}

function assertDenseConstant(constant: IndexedLiteralConstant): void {
  if (!constant.name || constant.quantization || (constant.storageDtype !== "F32" && constant.storageDtype !== "F16" && constant.storageDtype !== "BF16") ||
    constant.layout !== "row-major" || constant.byteOrder !== "little-endian" || constant.encoding !== "base64" ||
    !sameShape(constant.storageShape, constant.logicalShape) || !validShape(constant.storageShape)) {
    throw new Error(`${constant.name ?? "constante"}: leitor streaming Gemma 4 requer constante densa F32/F16/BF16 row-major verificável.`);
  }
  const bytes = product(constant.storageShape) * (constant.storageDtype === "F32" ? 4 : 2);
  if (!Number.isSafeInteger(bytes) || bytes !== constant.payloadBytes) throw new Error(`${constant.name}: payload literal não corresponde ao shape e dtype declarados.`);
}

function assertDenseDecoder(decoder: LiteralDenseStorageDecodeAssignment, constants: ReadonlyMap<string, IndexedLiteralConstant>): void {
  const constant = constants.get(decoder.output);
  const operation = constant?.storageDtype === "F32" ? "ieee-f32-little-endian" : constant?.storageDtype === "F16" ? "ieee-f16-to-f32" : "ieee-bf16-to-f32";
  if (!constant || decoder.id !== `decode_${constant.name}` || decoder.input !== `${constant.name}:storage` || decoder.operation !== operation ||
    decoder.storageDtype !== constant.storageDtype || decoder.outputDtype !== "F32" || decoder.byteOrder !== "little-endian" ||
    decoder.semantics !== "exact IEEE-754 storage decode; no arithmetic narrowing") {
    throw new Error(`${decoder.id}: decoder denso do artefato literal não corresponde à constante declarada.`);
  }
}

async function readTensorBytes(file: FileHandle, constants: ReadonlyMap<string, IndexedLiteralConstant>, tensor: TensorInfo): Promise<Buffer> {
  const constant = checkedTensor(constants, tensor);
  return readTensorBytesRange(file, constants, tensor, 0, constant.payloadBytes);
}

async function readTensorBytesRange(
  file: FileHandle,
  constants: ReadonlyMap<string, IndexedLiteralConstant>,
  tensor: TensorInfo,
  offset: number,
  byteLength: number,
): Promise<Buffer> {
  const constant = checkedTensor(constants, tensor);
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(byteLength) || offset < 0 || byteLength < 0 || offset + byteLength > constant.payloadBytes) {
    throw new Error(`${tensor.name}: range literal fora do payload incorporado.`);
  }
  if (byteLength === 0) return Buffer.alloc(0);
  const firstGroup = Math.floor(offset / 3);
  const lastGroupExclusive = Math.ceil((offset + byteLength) / 3);
  const encoded = await readRange(file, constant.payloadOffset + firstGroup * 4, constant.payloadOffset + lastGroupExclusive * 4);
  if (!/^[A-Za-z0-9+/=]+$/.test(encoded.toString("ascii"))) throw new Error(`${tensor.name}: payload literal contém base64 inválido.`);
  const decoded = Buffer.from(encoded.toString("ascii"), "base64");
  const relativeOffset = offset - firstGroup * 3;
  const result = decoded.subarray(relativeOffset, relativeOffset + byteLength);
  if (result.length !== byteLength) throw new Error(`${tensor.name}: base64 literal não decodificou o range solicitado.`);
  return Buffer.from(result);
}

function checkedTensor(constants: ReadonlyMap<string, IndexedLiteralConstant>, tensor: TensorInfo): IndexedLiteralConstant {
  const constant = constants.get(tensor.name);
  if (!constant || tensor.quantization || tensor.storageDtype !== constant.storageDtype || !sameShape(tensor.storageShape, constant.storageShape) ||
    !sameShape(tensor.logicalShape, constant.logicalShape)) {
    throw new Error(`${tensor.name}: tensor solicitado não corresponde à constante literal indexada.`);
  }
  return constant;
}

function parsePrefixJson(bytes: Buffer, label: string): unknown {
  if (bytes.length > MAX_STRUCTURAL_JSON_BYTES) throw new Error(`Artefato literal Gemma 4 possui ${label} estrutural excessivo.`);
  const text = bytes.toString("utf8").trim();
  const json = `${text}}`;
  try { return JSON.parse(json); } catch (error) { throw new Error(`Artefato literal Gemma 4 possui ${label} JSON inválido: ${(error as Error).message}`); }
}

function parseTailJson(bytes: Buffer, label: string): unknown {
  if (bytes.length > MAX_STRUCTURAL_JSON_BYTES) throw new Error(`Artefato literal Gemma 4 possui ${label} estrutural excessivo.`);
  const text = bytes.toString("utf8").trim();
  const json = `{${text.startsWith(",") ? text.slice(1) : text}`;
  try { return JSON.parse(json); } catch (error) { throw new Error(`Artefato literal Gemma 4 possui ${label} JSON inválido: ${(error as Error).message}`); }
}

async function base64Padding(file: FileHandle, payloadEnd: number, payloadCharacters: number): Promise<number> {
  const tail = await readRange(file, payloadEnd - Math.min(2, payloadCharacters), payloadEnd);
  return tail.equals(Buffer.from("==")) ? 2 : tail.subarray(-1).equals(Buffer.from("=")) ? 1 : 0;
}

async function findSequence(file: FileHandle, start: number, end: number, sequence: Buffer): Promise<number> {
  let position = start;
  let carry = Buffer.alloc(0);
  while (position < end) {
    const chunk = await readRange(file, position, Math.min(end, position + SCAN_CHUNK_BYTES));
    const combined = carry.length ? Buffer.concat([carry, chunk]) : chunk;
    const found = combined.indexOf(sequence);
    if (found !== -1) return position - carry.length + found;
    carry = Buffer.from(combined.subarray(Math.max(0, combined.length - sequence.length + 1)));
    position += chunk.length;
  }
  return -1;
}

async function findByte(file: FileHandle, start: number, end: number, value: number): Promise<number> {
  let position = start;
  while (position < end) {
    const chunk = await readRange(file, position, Math.min(end, position + SCAN_CHUNK_BYTES));
    const found = chunk.indexOf(value);
    if (found !== -1) return position + found;
    position += chunk.length;
  }
  return -1;
}

async function readRange(file: FileHandle, start: number, end: number): Promise<Buffer> {
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start) throw new Error("Range de artefato literal inválido.");
  const bytes = Buffer.allocUnsafe(end - start);
  let offset = 0;
  while (offset < bytes.length) {
    const { bytesRead } = await file.read(bytes, offset, bytes.length - offset, start + offset);
    if (bytesRead === 0) throw new Error("Artefato literal terminou antes do range solicitado.");
    offset += bytesRead;
  }
  return bytes;
}

async function readStructuralRange(file: FileHandle, start: number, end: number, label: string): Promise<Buffer> {
  if (end - start > MAX_STRUCTURAL_JSON_BYTES) throw new Error(`Artefato literal Gemma 4 possui ${label} estrutural excessivo.`);
  return readRange(file, start, end);
}

function validShape(shape: readonly number[]): boolean { return shape.every((dimension) => Number.isInteger(dimension) && dimension > 0); }
function sameShape(left: readonly number[], right: readonly number[]): boolean { return left.length === right.length && left.every((value, index) => value === right[index]); }
function product(shape: readonly number[]): number { return shape.reduce((total, dimension) => total * dimension, 1); }
