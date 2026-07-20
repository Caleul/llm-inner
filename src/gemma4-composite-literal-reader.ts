import { open, stat, type FileHandle } from "node:fs/promises";
import { isDeepStrictEqual } from "node:util";
import type {
  Gemma4CompositeLiteralCalculationProgram,
  Gemma4CompositeLiteralInput,
  Gemma4LiteralGreedyGenerationProgram,
  Gemma4CompositeUnreachableConstant,
} from "./gemma4-composite-literal.js";
import {
  GEMMA4_COMPOSITE_LITERAL_SCHEMA_VERSION,
  gemma4LiteralIntegritySections,
  validateGemma4CompositeLiteralInputs,
  validateGemma4CompositeLiteralNumericPolicy,
  validateGemma4CompositeLiteralStructure,
  validateGemma4LiteralGenerationProgram,
} from "./gemma4-composite-literal.js";
import {
  validateGemma4LiteralArtifactIntegrityManifest,
  type Gemma4LiteralArtifactIntegrityManifest,
} from "./gemma4-literal-artifact-integrity.js";
import {
  buildLiteralDenseStorageDecodeAssignment,
  validateLiteralDenseDecoderLanguageContract,
  type LiteralConstant,
  type LiteralDenseDecoderLanguageContract,
  type LiteralDenseStorageDecodeAssignment,
  type LiteralTensorReader,
} from "./literal.js";
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
  validateGemma4LiteralSourceWeightMappings,
  type Gemma4LiteralSourceIdentity,
} from "./gemma4-literal-source-identity.js";
import {
  validateGemma4LiteralFormulaLanguageContract,
  type Gemma4LiteralFormulaLanguageContract,
} from "./gemma4-literal-formula-language.js";
import {
  validateGemma4LiteralTranscendentalCoverage,
  type Gemma4LiteralTranscendentalPrograms,
} from "./gemma4-literal-transcendental-programs.js";
import {
  validateGemma4AuthoritativeExecutionContract,
  type Gemma4AuthoritativeExecutionContract,
} from "./gemma4-authoritative-runtime.js";
import {
  validateGemma4LiteralForwardControlProgram,
  type Gemma4LiteralForwardControlProgram,
} from "./gemma4-literal-forward-control.js";
import {
  validateGemma4LiteralInputDeclarationAlignment,
  validateGemma4LiteralInputContract,
  type Gemma4LiteralInputContract,
} from "./gemma4-literal-input-contract.js";
import {
  validateGemma4LiteralOutputContract,
  type Gemma4LiteralOutputContract,
} from "./gemma4-literal-output-contract.js";
import {
  validateGemma4LiteralFidelityGate,
  type Gemma4LiteralFidelityGate,
} from "./gemma4-literal-fidelity-gate.js";
import {
  assertGemma4LiteralPayloadChunkBytes,
  validateGemma4LiteralPayloadIntegrityMetadata,
  type Gemma4CompositeLiteralPayloadIntegrityEntry,
  type Gemma4LiteralPayloadIntegrityChunk,
} from "./gemma4-literal-payload-integrity.js";

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
  schemaVersion: typeof GEMMA4_COMPOSITE_LITERAL_SCHEMA_VERSION;
  artifact: string;
  artifactBytes: number;
  sourceIdentity: Gemma4LiteralSourceIdentity;
  authoritativeExecution: Gemma4AuthoritativeExecutionContract;
  constants: ReadonlyMap<string, IndexedLiteralConstant>;
  storageDecoders: LiteralDenseStorageDecodeAssignment[];
  denseDecoderLanguage: LiteralDenseDecoderLanguageContract;
  unreachableConstants: Gemma4CompositeUnreachableConstant[];
  program: Gemma4CompositeProgram;
  assignments: Gemma4CompositeLiteralCalculationProgram["assignments"];
  calculationDomains: Gemma4LiteralCalculationDomains;
  learnedOperands: Gemma4LiteralLearnedOperandBindings;
  scalarCalculations: Gemma4LiteralScalarCalculations;
  formulaLanguage: Gemma4LiteralFormulaLanguageContract;
  transcendentalPrograms: Gemma4LiteralTranscendentalPrograms;
  numericLiterals: Gemma4LiteralNumericLiterals;
  calculationGraph: Gemma4LiteralCalculationGraph;
  fidelityGate: Gemma4LiteralFidelityGate;
  forwardControl: Gemma4LiteralForwardControlProgram;
  inputContract: Gemma4LiteralInputContract;
  outputContract: Gemma4LiteralOutputContract;
  outputs: Gemma4CompositeLiteralCalculationProgram["outputs"];
  generation: Gemma4LiteralGreedyGenerationProgram;
  inputs: Gemma4CompositeLiteralInput[];
  numericPolicy: Gemma4CompositeLiteralCalculationProgram["numericPolicy"];
  payloadIntegrity: ReadonlyMap<string, Gemma4CompositeLiteralPayloadIntegrityEntry>;
  integrityManifest: Gemma4LiteralArtifactIntegrityManifest;
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
    const payloadReader = new IntegrityVerifiedPayloadReader(file, index.constants, index.payloadIntegrity);
    return {
      ...index,
      readTensorBytes: (tensor) => payloadReader.readTensorBytes(tensor),
      readTensorBytesRange: (tensor, offset, byteLength) => payloadReader.readTensorBytesRange(tensor, offset, byteLength),
      close: async () => { payloadReader.clear(); await file.close(); },
    };
  } catch (error) {
    await file.close();
    throw error;
  }
}

/** Lets an opened artifact participate in existing storage-reader contracts. */
export function asLiteralTensorReader(artifact: OpenGemma4CompositeLiteralArtifact): LiteralTensorReader {
  return {
    readTensorBytes: artifact.readTensorBytes,
    readTensorBytesRange: artifact.readTensorBytesRange,
    storageDecoders: artifact.storageDecoders,
    denseDecoderLanguage: artifact.denseDecoderLanguage,
  };
}

function buildIndex(
  artifact: string,
  artifactBytes: number,
  header: Partial<Gemma4CompositeLiteralCalculationProgram>,
  tail: Partial<Gemma4CompositeLiteralCalculationProgram>,
  constants: ReadonlyMap<string, IndexedLiteralConstant>,
): Gemma4CompositeLiteralArtifactIndex {
  if (!Array.isArray(tail.storageDecoders) || !tail.denseDecoderLanguage || !Array.isArray(tail.unreachableConstants) || !tail.program || !tail.assignments || !tail.calculationDomains || !tail.learnedOperands || !tail.scalarCalculations || !tail.formulaLanguage || !tail.transcendentalPrograms || !tail.numericLiterals || !tail.calculationGraph || !tail.fidelityGate || !tail.forwardControl || !tail.inputContract || !tail.outputContract || !tail.outputs || !tail.generation) {
    throw new Error("Artefato literal Gemma 4 não declara a cauda semântica completa.");
  }
  const storageDecoders = tail.storageDecoders as LiteralDenseStorageDecodeAssignment[];
  if (storageDecoders.length !== constants.size) throw new Error("Artefato literal Gemma 4 deve declarar um decoder denso por constante.");
  for (const decoder of storageDecoders) assertDenseDecoder(decoder, constants);
  validateLiteralDenseDecoderLanguageContract(tail.denseDecoderLanguage as LiteralDenseDecoderLanguageContract);
  validateGemma4CompositeLiteralNumericPolicy(
    header.numericPolicy as Gemma4CompositeLiteralCalculationProgram["numericPolicy"],
    tail.program as Gemma4CompositeProgram,
  );
  validateGemma4CompositeLiteralInputs(header.inputs as Gemma4CompositeLiteralInput[]);
  validateGemma4LiteralSourceIdentity(header.sourceIdentity as Gemma4LiteralSourceIdentity);
  validateGemma4AuthoritativeExecutionContract(header.authoritativeExecution as Gemma4AuthoritativeExecutionContract);
  if ((tail.program as Gemma4CompositeProgram).audioProgram.attentionMaskContract !== "transformers-eager-additive-mask-logical-not-v1") {
    throw new Error("Artefato Gemma 4 não declara a conversão eager autoritativa da máscara de áudio.");
  }
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
  validateGemma4LiteralGenerationProgram(tail.generation as Gemma4LiteralGreedyGenerationProgram, tail.program as Gemma4CompositeProgram);
  validateGemma4LiteralFormulaLanguageContract(
    tail.formulaLanguage as Gemma4LiteralFormulaLanguageContract,
    tail.scalarCalculations as Gemma4LiteralScalarCalculations,
    (tail.generation as Gemma4LiteralGreedyGenerationProgram).scalarCalculations,
    (tail.generation as Gemma4LiteralGreedyGenerationProgram).forwardCalculation,
  );
  validateGemma4LiteralTranscendentalCoverage(tail.transcendentalPrograms as Gemma4LiteralTranscendentalPrograms, [
    ...(tail.scalarCalculations as Gemma4LiteralScalarCalculations).assignments.flatMap((assignment) => assignment.scalarAssignments),
    ...(tail.generation as Gemma4LiteralGreedyGenerationProgram).scalarCalculations.assignments.flatMap((assignment) => assignment.scalarAssignments),
  ]);
  validateGemma4LiteralNumericLiterals(
    tail.numericLiterals as Gemma4LiteralNumericLiterals,
    tail.scalarCalculations as Gemma4LiteralScalarCalculations,
    (tail.generation as Gemma4LiteralGreedyGenerationProgram).scalarCalculations,
    (tail.generation as Gemma4LiteralGreedyGenerationProgram).forwardCalculation,
  );
  validateGemma4LiteralCalculationGraph(tail.calculationGraph as Gemma4LiteralCalculationGraph, tail.program as Gemma4CompositeProgram);
  validateGemma4LiteralFidelityGate(
    tail.fidelityGate as Gemma4LiteralFidelityGate,
    tail.calculationGraph as Gemma4LiteralCalculationGraph,
    header.authoritativeExecution as Gemma4AuthoritativeExecutionContract,
  );
  validateGemma4LiteralForwardControlProgram(tail.forwardControl as Gemma4LiteralForwardControlProgram, tail.program as Gemma4CompositeProgram);
  validateGemma4LiteralInputContract(tail.inputContract as Gemma4LiteralInputContract, tail.program as Gemma4CompositeProgram);
  validateGemma4LiteralInputDeclarationAlignment(tail.inputContract as Gemma4LiteralInputContract, header.inputs as Gemma4CompositeLiteralInput[]);
  validateGemma4LiteralOutputContract(tail.outputContract as Gemma4LiteralOutputContract, tail.program as Gemma4CompositeProgram);
  const payloadIntegrity = validatePayloadIntegrity(tail.payloadIntegrity, constants);
  validateGemma4LiteralSourceWeightMappings(
    header.sourceIdentity as Gemma4LiteralSourceIdentity,
    [...constants.values()].map((constant) => ({
      name: constant.name,
      storageDtype: constant.storageDtype,
      storageShape: constant.storageShape,
      payloadBytes: constant.payloadBytes,
    })),
  );
  if (!tail.integrityManifest) throw new Error("Artefato literal Gemma 4 não declara compromisso estrutural.");
  const constantMetadata = [...constants.values()].map(({ payloadOffset: _offset, payloadBase64Characters: _characters, payloadBytes: _bytes, ...metadata }) => metadata);
  validateGemma4LiteralArtifactIntegrityManifest(
    tail.integrityManifest as Gemma4LiteralArtifactIntegrityManifest,
    gemma4LiteralIntegritySections({
      sourceIdentity: header.sourceIdentity,
      authoritativeExecution: header.authoritativeExecution,
      numericPolicy: header.numericPolicy,
      inputs: header.inputs,
      constants: constantMetadata,
      unreachableConstants: tail.unreachableConstants,
      storageDecoders: tail.storageDecoders,
      denseDecoderLanguage: tail.denseDecoderLanguage,
      program: tail.program,
      assignments: tail.assignments,
      calculationDomains: tail.calculationDomains,
      learnedOperands: tail.learnedOperands,
      scalarCalculations: tail.scalarCalculations,
      formulaLanguage: tail.formulaLanguage,
      transcendentalPrograms: tail.transcendentalPrograms,
      numericLiterals: tail.numericLiterals,
      calculationGraph: tail.calculationGraph,
      fidelityGate: tail.fidelityGate,
      forwardControl: tail.forwardControl,
      inputContract: tail.inputContract,
      outputContract: tail.outputContract,
      outputs: tail.outputs,
      generation: tail.generation,
      payloadIntegrity: tail.payloadIntegrity,
    }),
  );
  return {
    schemaVersion: GEMMA4_COMPOSITE_LITERAL_SCHEMA_VERSION,
    artifact,
    artifactBytes,
    sourceIdentity: structuredClone(header.sourceIdentity as Gemma4LiteralSourceIdentity),
    authoritativeExecution: structuredClone(header.authoritativeExecution as Gemma4AuthoritativeExecutionContract),
    constants,
    storageDecoders,
    denseDecoderLanguage: structuredClone(tail.denseDecoderLanguage as LiteralDenseDecoderLanguageContract),
    unreachableConstants: tail.unreachableConstants as Gemma4CompositeUnreachableConstant[],
    program: tail.program as Gemma4CompositeProgram,
    assignments: tail.assignments as Gemma4CompositeLiteralCalculationProgram["assignments"],
    calculationDomains: tail.calculationDomains as Gemma4LiteralCalculationDomains,
    learnedOperands: tail.learnedOperands as Gemma4LiteralLearnedOperandBindings,
    scalarCalculations: tail.scalarCalculations as Gemma4LiteralScalarCalculations,
    formulaLanguage: tail.formulaLanguage as Gemma4LiteralFormulaLanguageContract,
    transcendentalPrograms: tail.transcendentalPrograms as Gemma4LiteralTranscendentalPrograms,
    numericLiterals: tail.numericLiterals as Gemma4LiteralNumericLiterals,
    calculationGraph: tail.calculationGraph as Gemma4LiteralCalculationGraph,
    fidelityGate: tail.fidelityGate as Gemma4LiteralFidelityGate,
    forwardControl: tail.forwardControl as Gemma4LiteralForwardControlProgram,
    inputContract: tail.inputContract as Gemma4LiteralInputContract,
    outputContract: tail.outputContract as Gemma4LiteralOutputContract,
    outputs: tail.outputs as Gemma4CompositeLiteralCalculationProgram["outputs"],
    generation: tail.generation as Gemma4LiteralGreedyGenerationProgram,
    inputs: header.inputs as Gemma4CompositeLiteralInput[],
    numericPolicy: header.numericPolicy as Gemma4CompositeLiteralCalculationProgram["numericPolicy"],
    payloadIntegrity,
    integrityManifest: structuredClone(tail.integrityManifest as Gemma4LiteralArtifactIntegrityManifest),
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
    if (!constant || integrity.has(constant.name)) {
      throw new Error(`${candidate.name ?? "constante"}: compromisso de integridade de payload inválido.`);
    }
    validateGemma4LiteralPayloadIntegrityMetadata(candidate as Gemma4CompositeLiteralPayloadIntegrityEntry, constant.name, constant.payloadBytes);
    integrity.set(constant.name, structuredClone(candidate as Gemma4CompositeLiteralPayloadIntegrityEntry));
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
  if (header.schemaVersion !== GEMMA4_COMPOSITE_LITERAL_SCHEMA_VERSION || header.kind !== "gemma4-composite-literal-calculation-program" || header.sourceFormat !== "safetensors" ||
    !header.sourceIdentity || !header.authoritativeExecution || !Array.isArray(header.inputs) || !policy || (!f32 && !operationDeclared && !operationAccumulationDeclared)) {
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
  if (!constant || !isDeepStrictEqual(decoder, buildLiteralDenseStorageDecodeAssignment(constant))) {
    throw new Error(`${decoder.id}: decoder denso do artefato literal não corresponde à constante declarada.`);
  }
}

class IntegrityVerifiedPayloadReader {
  private readonly chunkCache = new Map<string, Buffer>();
  private static readonly MAX_CACHED_CHUNKS = 4;

  constructor(
    private readonly file: FileHandle,
    private readonly constants: ReadonlyMap<string, IndexedLiteralConstant>,
    private readonly integrity: ReadonlyMap<string, Gemma4CompositeLiteralPayloadIntegrityEntry>,
  ) {}

  async readTensorBytes(tensor: TensorInfo): Promise<Buffer> {
    const constant = checkedTensor(this.constants, tensor);
    return this.readTensorBytesRange(tensor, 0, constant.payloadBytes);
  }

  async readTensorBytesRange(tensor: TensorInfo, offset: number, byteLength: number): Promise<Buffer> {
    const constant = checkedTensor(this.constants, tensor);
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(byteLength) || offset < 0 || byteLength < 0 || offset + byteLength > constant.payloadBytes) {
      throw new Error(`${tensor.name}: range literal fora do payload incorporado.`);
    }
    if (byteLength === 0) return Buffer.alloc(0);
    const integrity = this.integrity.get(constant.name);
    if (!integrity) throw new Error(`${constant.name}: compromisso incorporado ausente durante leitura.`);
    const first = Math.floor(offset / integrity.chunking.chunkBytes);
    const last = Math.floor((offset + byteLength - 1) / integrity.chunking.chunkBytes);
    const slices: Buffer[] = [];
    for (let ordinal = first; ordinal <= last; ordinal += 1) {
      const chunk = integrity.chunking.chunks[ordinal]!;
      const bytes = await this.readVerifiedChunk(constant, integrity, chunk);
      const sliceStart = Math.max(offset, chunk.byteOffset) - chunk.byteOffset;
      const sliceEnd = Math.min(offset + byteLength, chunk.byteOffset + chunk.byteLength) - chunk.byteOffset;
      slices.push(bytes.subarray(sliceStart, sliceEnd));
    }
    const result = Buffer.concat(slices, byteLength);
    if (result.length !== byteLength) throw new Error(`${tensor.name}: leitura autenticada não produziu o range solicitado.`);
    return result;
  }

  clear(): void { this.chunkCache.clear(); }

  private async readVerifiedChunk(
    constant: IndexedLiteralConstant,
    integrity: Gemma4CompositeLiteralPayloadIntegrityEntry,
    chunk: Gemma4LiteralPayloadIntegrityChunk,
  ): Promise<Buffer> {
    const key = `${constant.name}\0${chunk.ordinal}`;
    const cached = this.chunkCache.get(key);
    if (cached) {
      this.chunkCache.delete(key);
      this.chunkCache.set(key, cached);
      return cached;
    }
    if (chunk.byteOffset % 3 !== 0) throw new Error(`${constant.name}: chunk literal ${chunk.ordinal} não está alinhado ao Base64.`);
    const encodedOffset = constant.payloadOffset + chunk.byteOffset / 3 * 4;
    const encodedCharacters = Math.ceil(chunk.byteLength / 3) * 4;
    const encodedBytes = await readRange(this.file, encodedOffset, encodedOffset + encodedCharacters);
    const encoded = encodedBytes.toString("ascii");
    if (!/^[A-Za-z0-9+/]+={0,2}$/.test(encoded)) throw new Error(`${constant.name}: chunk literal ${chunk.ordinal} contém Base64 inválido.`);
    const bytes = Buffer.from(encoded, "base64");
    if (bytes.length !== chunk.byteLength || bytes.toString("base64") !== encoded) {
      throw new Error(`${constant.name}: chunk literal ${chunk.ordinal} não usa Base64 RFC 4648 canônico.`);
    }
    assertGemma4LiteralPayloadChunkBytes(integrity, chunk, bytes);
    this.chunkCache.set(key, bytes);
    while (this.chunkCache.size > IntegrityVerifiedPayloadReader.MAX_CACHED_CHUNKS) {
      this.chunkCache.delete(this.chunkCache.keys().next().value!);
    }
    return bytes;
  }
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
