import { isDeepStrictEqual } from "node:util";
import { executeReferenceF32, generateReferenceF32 } from "./executor.js";
import { adaptGgufDecoderCatalog } from "./gguf-llama.js";
import { decodeIeeeBF16ToF32, decodeIeeeF16ToF32, roundF32ToBF16 } from "./utils.js";
import type {
  DenseF32Tensor,
  JsonObject,
  ModelCatalog,
  ModelIR,
  Operation,
  QuantizationSpec,
  ReferenceF32ExecutionRequest,
  ReferenceF32ExecutionResult,
  ReferenceF32GenerationRequest,
  ReferenceF32GenerationResult,
  TensorInfo,
  TensorRef,
} from "./types.js";

/** Storage-level reader required to remove source-checkpoint dependencies. */
export interface LiteralTensorReader {
  readTensorBytes(tensor: TensorInfo): Promise<Buffer>;
  /** Optional bounded range read for writers that must handle multi-GiB tensors. */
  readTensorBytesRange?(tensor: TensorInfo, offset: number, byteLength: number): Promise<Buffer>;
}

export interface LiteralInput {
  name: "input_ids" | "position_ids" | "attention_mask" | "past_key_values";
  description: string;
  required: boolean;
  dtype: string;
  shape: string;
}

export type LiteralDenseStorageDtype = "F32" | "F16" | "BF16";
export type LiteralStorageDtype = LiteralDenseStorageDtype | "U32" | "GGML_Q8_0";
export type LiteralStorageLayout = "row-major" | "ggml-first-axis-contiguous";

/**
 * Exact storage payload embedded in the program. `name` is the logical tensor
 * name used by IR assignments; its storage bytes are only made available to
 * those assignments through the corresponding explicit decoder below.
 */
export interface LiteralConstant {
  name: string;
  storageDtype: LiteralStorageDtype;
  storageShape: number[];
  logicalShape: number[];
  layout: LiteralStorageLayout;
  byteOrder: "little-endian";
  encoding: "base64";
  payloadBase64: string;
  /** Present only for a packed logical tensor with a declared decoder below. */
  quantization?: QuantizationSpec;
}

/**
 * A deterministic storage-to-compute boundary. These are assignments in the
 * literal program, not a convenience performed by the host when loading JSON:
 * F16/BF16 payloads remain in their original representation until this exact
 * IEEE conversion emits the F32 tensor named by `output`.
 */
export interface LiteralDenseStorageDecodeAssignment {
  id: string;
  operation: "ieee-f32-little-endian" | "ieee-f16-to-f32" | "ieee-bf16-to-f32";
  input: string;
  output: string;
  storageDtype: LiteralDenseStorageDtype;
  outputDtype: "F32";
  byteOrder: "little-endian";
  /** Complete logical-index to payload-byte mapping; no reader-owned stride inference is permitted. */
  address: LiteralDenseElementAddressProgram;
  /** Bit-level storage conversion sufficient to reproduce the exact F32 result bits. */
  decode: LiteralDenseIeeeDecodeProgram;
  semantics: "exact IEEE-754 storage decode; no arithmetic narrowing";
}

export interface LiteralDenseElementAddressProgram {
  kind: "row-major-dense-element-address";
  schemaVersion: 1;
  logicalShape: number[];
  stridesElements: number[];
  indexDomains: Array<{ axis: number; minInclusive: 0; endExclusive: number }>;
  elementBytes: 2 | 4;
  arithmetic: "exact-non-negative-safe-integer";
  elementOffset: "sum(indices[axis] * stridesElements[axis]) in ascending axis order";
  byteOffset: "elementOffset * elementBytes";
  byteLength: "elementBytes";
}

export type LiteralDenseIeeeDecodeProgram =
  | {
    kind: "ieee-binary32-bitcast";
    schemaVersion: 1;
    read: "uint32-little-endian";
    resultBits: "sourceBits";
  }
  | {
    kind: "ieee-bfloat16-expand";
    schemaVersion: 1;
    read: "uint16-little-endian";
    resultBits: "sourceBits << 16";
  }
  | {
    kind: "ieee-binary16-expand";
    schemaVersion: 1;
    read: "uint16-little-endian";
    fields: {
      signMask: 0x8000;
      signShift: 15;
      exponentMask: 0x7c00;
      exponentShift: 10;
      fractionMask: 0x03ff;
    };
    extraction: {
      sign: "(sourceBits & signMask) >> signShift";
      exponent: "(sourceBits & exponentMask) >> exponentShift";
      fraction: "sourceBits & fractionMask";
    };
    normal: {
      predicate: "0 < exponent < 31";
      resultBits: "(sign << 31) | ((exponent + 112) << 23) | (fraction << 13)";
    };
    zero: {
      predicate: "exponent == 0 && fraction == 0";
      resultBits: "sign << 31";
    };
    subnormal: {
      predicate: "exponent == 0 && fraction != 0";
      normalization: "left-shift fraction until bit 10 is one; shiftCount starts at zero";
      resultBits: "(sign << 31) | ((113 - shiftCount) << 23) | ((normalizedFraction & 0x03ff) << 13)";
    };
    infinityOrNaN: {
      predicate: "exponent == 31";
      resultBits: "(sign << 31) | 0x7f800000 | (fraction << 13)";
    };
  };

export interface LiteralDenseElementAddress {
  elementOffset: number;
  byteOffset: number;
  byteLength: 2 | 4;
}

export interface LiteralDenseDecodedElement {
  sourceBits: number;
  sourceBitsHex: string;
  decodedF32Bits: number;
  decodedF32BitsHex: string;
  decodedF32: number;
}

type LiteralDenseDecoderConstant = Pick<
  LiteralConstant,
  "name" | "storageDtype" | "storageShape" | "logicalShape" | "layout" | "byteOrder" | "quantization"
>;

/** Builds the complete dense address and IEEE conversion program serialized beside one payload. */
export function buildLiteralDenseStorageDecodeAssignment(
  constant: LiteralDenseDecoderConstant,
): LiteralDenseStorageDecodeAssignment {
  if (constant.quantization || !isSupportedDenseStorageDtype(constant.storageDtype) ||
    constant.layout !== "row-major" || constant.byteOrder !== "little-endian" ||
    !sameShape(constant.storageShape, constant.logicalShape) || !validLiteralStorageShape(constant.logicalShape)) {
    throw new Error(`${constant.name}: decoder denso requer storage IEEE row-major não quantizado.`);
  }
  const elementBytes: 2 | 4 = constant.storageDtype === "F32" ? 4 : 2;
  const stridesElements = new Array<number>(constant.logicalShape.length);
  let stride = 1;
  for (let axis = constant.logicalShape.length - 1; axis >= 0; axis -= 1) {
    stridesElements[axis] = stride;
    const next = stride * constant.logicalShape[axis]!;
    if (!Number.isSafeInteger(next)) throw new Error(`${constant.name}: strides densos excedem inteiros seguros.`);
    stride = next;
  }
  return {
    id: `decode_${constant.name}`,
    operation: decodeOperationFor(constant.storageDtype),
    input: `${constant.name}:storage`,
    output: constant.name,
    storageDtype: constant.storageDtype,
    outputDtype: "F32",
    byteOrder: "little-endian",
    address: {
      kind: "row-major-dense-element-address",
      schemaVersion: 1,
      logicalShape: [...constant.logicalShape],
      stridesElements,
      indexDomains: constant.logicalShape.map((endExclusive, axis) => ({ axis, minInclusive: 0 as const, endExclusive })),
      elementBytes,
      arithmetic: "exact-non-negative-safe-integer",
      elementOffset: "sum(indices[axis] * stridesElements[axis]) in ascending axis order",
      byteOffset: "elementOffset * elementBytes",
      byteLength: "elementBytes",
    },
    decode: denseIeeeDecodeProgram(constant.storageDtype),
    semantics: "exact IEEE-754 storage decode; no arithmetic narrowing",
  };
}

/** Evaluates only the serialized address program and rejects any missing, unsafe or out-of-domain index. */
export function evaluateLiteralDenseElementAddress(
  decoder: LiteralDenseStorageDecodeAssignment,
  indices: readonly number[],
): LiteralDenseElementAddress {
  const address = decoder.address;
  if (address.kind !== "row-major-dense-element-address" || address.schemaVersion !== 1 ||
    address.arithmetic !== "exact-non-negative-safe-integer" ||
    address.elementOffset !== "sum(indices[axis] * stridesElements[axis]) in ascending axis order" ||
    address.byteOffset !== "elementOffset * elementBytes" || address.byteLength !== "elementBytes" ||
    indices.length !== address.logicalShape.length || address.stridesElements.length !== address.logicalShape.length ||
    address.indexDomains.length !== address.logicalShape.length) {
    throw new Error(`${decoder.id}: programa de endereço denso inválido.`);
  }
  let expectedStride = 1;
  for (let axis = address.logicalShape.length - 1; axis >= 0; axis -= 1) {
    if (address.stridesElements[axis] !== expectedStride) {
      throw new Error(`${decoder.id}: stride row-major inválido no eixo ${axis}.`);
    }
    expectedStride *= address.logicalShape[axis]!;
    if (!Number.isSafeInteger(expectedStride)) throw new Error(`${decoder.output}: shape denso excede inteiros seguros.`);
  }
  let elementOffset = 0;
  for (let axis = 0; axis < indices.length; axis += 1) {
    const index = indices[axis]!;
    const dimension = address.logicalShape[axis]!;
    const stride = address.stridesElements[axis]!;
    const domain = address.indexDomains[axis];
    if (!domain || domain.axis !== axis || domain.minInclusive !== 0 || domain.endExclusive !== dimension ||
      !Number.isSafeInteger(index) || index < 0 || index >= dimension || !Number.isSafeInteger(stride) || stride <= 0) {
      throw new Error(`${decoder.output}: índice ${index} fora do eixo ${axis} de tamanho ${dimension}.`);
    }
    const term = index * stride;
    elementOffset += term;
    if (!Number.isSafeInteger(term) || !Number.isSafeInteger(elementOffset)) {
      throw new Error(`${decoder.output}: offset denso excede inteiros seguros.`);
    }
  }
  const byteOffset = elementOffset * address.elementBytes;
  if (!Number.isSafeInteger(byteOffset)) throw new Error(`${decoder.output}: byte offset denso excede inteiros seguros.`);
  return { elementOffset, byteOffset, byteLength: address.elementBytes };
}

/** Executes the decoder's embedded bit program for one already-addressed storage element. */
export function decodeLiteralDenseElementF32(
  decoder: LiteralDenseStorageDecodeAssignment,
  bytes: Buffer,
): LiteralDenseDecodedElement {
  if (bytes.length !== decoder.address.elementBytes) {
    throw new Error(`${decoder.id}: elemento denso possui ${bytes.length} bytes; esperados ${decoder.address.elementBytes}.`);
  }
  const sourceBits = bytes.length === 4 ? bytes.readUInt32LE(0) : bytes.readUInt16LE(0);
  let decodedF32: number;
  let decodedF32Bits: number;
  switch (decoder.decode.kind) {
    case "ieee-binary32-bitcast":
      if (decoder.operation !== "ieee-f32-little-endian" || decoder.decode.schemaVersion !== 1 ||
        decoder.decode.read !== "uint32-little-endian" || decoder.decode.resultBits !== "sourceBits" || bytes.length !== 4) {
        throw new Error(`${decoder.id}: programa binary32 incompatível.`);
      }
      decodedF32 = bytes.readFloatLE(0);
      decodedF32Bits = sourceBits;
      break;
    case "ieee-bfloat16-expand":
      if (decoder.operation !== "ieee-bf16-to-f32" || decoder.decode.schemaVersion !== 1 ||
        decoder.decode.read !== "uint16-little-endian" || decoder.decode.resultBits !== "sourceBits << 16" || bytes.length !== 2) {
        throw new Error(`${decoder.id}: programa bfloat16 incompatível.`);
      }
      decodedF32 = decodeIeeeBF16ToF32(sourceBits);
      decodedF32Bits = (sourceBits << 16) >>> 0;
      break;
    case "ieee-binary16-expand":
      if (decoder.operation !== "ieee-f16-to-f32" || !sameBinary16DecodeProgram(decoder.decode) || bytes.length !== 2) {
        throw new Error(`${decoder.id}: programa binary16 incompatível.`);
      }
      decodedF32 = decodeIeeeF16ToF32(sourceBits);
      decodedF32Bits = ieeeF16ToF32Bits(sourceBits);
      break;
  }
  return {
    sourceBits,
    sourceBitsHex: `0x${sourceBits.toString(16).padStart(bytes.length * 2, "0")}`,
    decodedF32Bits,
    decodedF32BitsHex: `0x${decodedF32Bits.toString(16).padStart(8, "0")}`,
    decodedF32,
  };
}

/**
 * The exact MLX affine-U32 reconstruction contract. Codes are contiguous in
 * each logical row, least-significant-bit first in little-endian U32 words;
 * scale/bias tensors are separately embedded constants decoded earlier in the
 * program. This remains a literal assignment rather than host-side loading.
 */
export interface LiteralMlxAffineStorageDecodeAssignment {
  id: string;
  operation: "mlx-affine-u32-to-f32";
  input: string;
  output: string;
  storageDtype: "U32";
  outputDtype: "F32";
  byteOrder: "little-endian";
  packing: "row-major-contiguous-lsb-first-u32";
  bits: 2 | 3 | 4 | 5 | 6 | 8;
  groupSize: number;
  scaleInput: string;
  biasInput?: string;
  parameterDtype: LiteralDenseStorageDtype;
  semantics: "F32(scale * unsigned_code + bias); BF16 parameters round each affine result to BF16 before F32 output";
}

/**
 * GGML Q8_0 retains the original 34-byte block instead of expanding its
 * values into decimal JSON: F16 scale `d` followed by 32 signed codes.
 */
export interface LiteralGgmlQ8_0StorageDecodeAssignment {
  id: string;
  operation: "ggml-q8-0-to-f32";
  input: string;
  output: string;
  storageDtype: "GGML_Q8_0";
  outputDtype: "F32";
  byteOrder: "little-endian";
  packing: "blocks-of-32-f16-scale-then-i8-codes";
  blockSize: 32;
  blockBytes: 34;
  semantics: "F32(IEEE-754 binary16 scale * signed int8 code) for each consecutive GGML logical value";
}

export type LiteralStorageDecodeAssignment =
  | LiteralDenseStorageDecodeAssignment
  | LiteralMlxAffineStorageDecodeAssignment
  | LiteralGgmlQ8_0StorageDecodeAssignment;

export interface LiteralKvCacheTransition {
  id: string;
  layer: number;
  operation: "append-post-rope" | "reuse-producer";
  keyInput: string;
  valueInput: string;
  cacheOutput: string;
  producerLayer?: number;
}

/**
 * A source-independent literal calculation program for established dense,
 * MLX-affine Safetensors, and GGML Q8_0 storage contracts. The operation graph intentionally
 * retains the same stable operation ids and named dataflow as ModelIR, but
 * every referenced tensor is embedded as exact bytes rather than a shard path.
 */
export interface LiteralCalculationProgram {
  schemaVersion: 1;
  kind: "literal-calculation-program";
  sourceFormat: "safetensors" | "mlx-safetensors" | "gguf";
  architecture: ModelIR["architecture"];
  config: JsonObject;
  numericPolicy: {
    inputDtype: "I32";
    computeDtype: "F32";
    accumulationDtype: "F32";
    outputDtype: "F32";
    scalarSemantics: "IEEE-754 binary32; host libm results rounded to F32";
  };
  inputs: LiteralInput[];
  constants: LiteralConstant[];
  storageDecoders: LiteralStorageDecodeAssignment[];
  assignments: {
    prelude: Operation[];
    layers: ModelIR["layers"];
    epilogue: Operation[];
  };
  stateTransitions: LiteralKvCacheTransition[];
  outputs: { logits: string };
}

/**
 * Reusable, source-independent storage boundary for architecture adapters
 * whose executable assignments are not represented by ModelIR alone (for
 * example Gemma 4's vision and audio towers).  Keeping this boundary here
 * prevents an adapter from embedding ad-hoc F32 arrays or reopening a shard
 * during literal replay.
 */
export interface LiteralStorageBundle {
  constants: LiteralConstant[];
  storageDecoders: LiteralStorageDecodeAssignment[];
}

const STORAGE_BYTES: Record<Exclude<LiteralStorageDtype, "GGML_Q8_0">, number> = {
  F32: 4,
  F16: 2,
  BF16: 2,
  U32: 4,
};

/**
 * Creates a self-contained literal program from an already validated
 * architecture IR. It accepts dense F32/F16/BF16 storage, the established
 * MLX affine-U32 contract, and the established GGML Q8_0 contract.
 * Lower-precision and packed storage are embedded unchanged, then accompanied
 * by explicit decoder assignments.
 */
export async function buildLiteralCalculationProgram(
  ir: ModelIR,
  catalog: ModelCatalog,
  reader: LiteralTensorReader,
): Promise<LiteralCalculationProgram> {
  if (catalog.format !== "safetensors" && catalog.format !== "mlx-safetensors" && catalog.format !== "gguf") {
    throw new Error(`Exportação literal requer Safetensors denso, MLX affine-U32 ou GGUF Q8_0; recebeu formato ${catalog.format}.`);
  }
  const storage = await buildLiteralStorageBundle(catalog, reader, referencedTensors(ir).values());

  const assignments = cloneF32Assignments(ir);
  const program: LiteralCalculationProgram = {
    schemaVersion: 1,
    kind: "literal-calculation-program",
    sourceFormat: catalog.format,
    architecture: structuredClone(ir.architecture),
    config: structuredClone(ir.config),
    numericPolicy: {
      inputDtype: "I32",
      computeDtype: "F32",
      accumulationDtype: "F32",
      outputDtype: "F32",
      scalarSemantics: "IEEE-754 binary32; host libm results rounded to F32",
    },
    inputs: literalInputs(),
    constants: storage.constants,
    storageDecoders: storage.storageDecoders,
    assignments,
    stateTransitions: cacheTransitions(assignments),
    outputs: { logits: assignments.epilogue.some((operation) => operation.output === "softcapped_logits") ? "softcapped_logits" : "logits" },
  };
  validateLiteralCalculationProgram(program);
  return program;
}

/**
 * Embeds each referenced storage payload exactly once and declares every
 * required decoder.  Callers must pass references obtained from a validated
 * adapter assignment; this function verifies them again against the catalog.
 */
export async function buildLiteralStorageBundle(
  catalog: ModelCatalog,
  reader: LiteralTensorReader,
  references: Iterable<TensorRef>,
): Promise<LiteralStorageBundle> {
  const adaptedCatalog = adaptGgufDecoderCatalog(catalog);
  const unique = new Map<string, TensorRef>();
  for (const reference of references) registerReference(unique, reference);
  const constants = new Map<string, LiteralConstant>();
  for (const reference of unique.values()) {
    const tensor = adaptedCatalog.tensors.get(reference.name);
    assertLiteralReference(reference, tensor);
    if (tensor!.quantization) {
      if (isGgmlQ8_0Tensor(tensor!)) {
        await embedLiteralConstant(constants, tensor!, reader);
      } else {
        const { scale, bias } = assertMlxAffineTensor(tensor!, adaptedCatalog);
        await embedLiteralConstant(constants, tensor!, reader);
        await embedLiteralConstant(constants, scale, reader);
        if (bias) await embedLiteralConstant(constants, bias, reader);
      }
    } else {
      await embedLiteralConstant(constants, tensor!, reader);
    }
  }
  const ordered = [...constants.values()].sort((left, right) => Number(Boolean(left.quantization)) - Number(Boolean(right.quantization)));
  const bundle = { constants: ordered, storageDecoders: ordered.map((constant) => storageDecoder(constant, constants)) };
  validateLiteralStorageBundle(bundle);
  return bundle;
}

/** Backward-compatible entry points retained for earlier callers. */
export const buildDenseSafetensorsLiteralProgram = buildLiteralCalculationProgram;
export const buildDenseF32LiteralProgram = buildLiteralCalculationProgram;

/** Replays a validated literal program without opening a source checkpoint. */
export function executeLiteralF32(
  program: LiteralCalculationProgram,
  request: Omit<ReferenceF32ExecutionRequest, "tensors">,
): ReferenceF32ExecutionResult {
  validateLiteralCalculationProgram(program);
  return executeReferenceF32(toEmbeddedModelIR(program), { ...request, tensors: decodeDenseConstantsAsF32(program) });
}

/** Greedily generates from the same embedded constants and explicit KV contract. */
export function generateLiteralF32(
  program: LiteralCalculationProgram,
  request: Omit<ReferenceF32GenerationRequest, "tensors">,
): ReferenceF32GenerationResult {
  validateLiteralCalculationProgram(program);
  return generateReferenceF32(toEmbeddedModelIR(program), { ...request, tensors: decodeDenseConstantsAsF32(program) });
}

/**
 * Defensive validation for programs read from JSON. It proves that every
 * operation reads a declared input, an earlier assignment, or an embedded
 * constant; neither a shard path nor an implicit decoder can enter replay.
 */
export function validateLiteralCalculationProgram(program: LiteralCalculationProgram): void {
  if (program.schemaVersion !== 1 || program.kind !== "literal-calculation-program" ||
    (program.sourceFormat !== "safetensors" && program.sourceFormat !== "mlx-safetensors" && program.sourceFormat !== "gguf")) {
    throw new Error("Programa literal inválido: schemaVersion, kind ou sourceFormat não reconhecido.");
  }
  if (
    program.numericPolicy.inputDtype !== "I32" ||
    program.numericPolicy.computeDtype !== "F32" ||
    program.numericPolicy.accumulationDtype !== "F32" ||
    program.numericPolicy.outputDtype !== "F32"
  ) {
    throw new Error("Programa literal inválido: esta versão requer política explícita F32/I32.");
  }
  const constants = new Map<string, LiteralConstant>();
  for (const constant of program.constants) {
    if (constants.has(constant.name)) throw new Error(`Programa literal contém constante duplicada: ${constant.name}.`);
    validateLiteralConstant(constant);
    const payload = decodeBase64(constant.payloadBase64, constant.name);
    if (payload.length !== literalStorageByteLength(constant.storageDtype, constant.storageShape)) {
      throw new Error(`${constant.name}: payload base64 não corresponde ao shape ${constant.storageDtype} declarado.`);
    }
    constants.set(constant.name, constant);
  }
  if (constants.size === 0) throw new Error("Programa literal não contém constantes incorporadas.");
  if (!Array.isArray(program.storageDecoders)) throw new Error("Programa literal não declara decoders de storage.");
  validateStorageDecoders(program.storageDecoders, constants);

  const declaredInputs = new Set(program.inputs.map((input) => input.name));
  for (const required of ["input_ids", "position_ids", "attention_mask", "past_key_values"] as const) {
    if (!declaredInputs.has(required)) throw new Error(`Programa literal não declara a entrada ${required}.`);
  }
  const available = new Set<string>(declaredInputs);
  for (const operation of allOperations(program.assignments)) {
    validateOperationInputs(operation, available, constants);
    if (available.has(operation.output)) throw new Error(`${operation.id}: saída ${operation.output} já foi declarada.`);
    available.add(operation.output);
  }
  if (!available.has(program.outputs.logits)) throw new Error(`Programa literal não produz logits declarados: ${program.outputs.logits}.`);
  validateCacheTransitions(program.stateTransitions, program.assignments);
}

function literalInputs(): LiteralInput[] {
  return [
    { name: "input_ids", description: "Token IDs supplied by the caller.", required: true, dtype: "I32", shape: "[batch, sequence]" },
    { name: "position_ids", description: "Absolute RoPE positions; defaults to 0..sequence-1 only when omitted by the declared executor contract.", required: false, dtype: "I32", shape: "[batch, sequence]" },
    { name: "attention_mask", description: "Optional canonical additive attention bias [batch, 1|heads, query, key].", required: false, dtype: "F32", shape: "[batch, 1|heads, query, key]" },
    { name: "past_key_values", description: "Optional canonical post-RoPE KV cache keyed by producer layer.", required: false, dtype: "F32", shape: "layer -> { key/value: [batch, kv_heads, cached_sequence, head_dim] }" },
  ];
}

function cloneF32Assignments(ir: ModelIR): LiteralCalculationProgram["assignments"] {
  const clone = structuredClone({ prelude: ir.prelude, layers: ir.layers, epilogue: ir.epilogue });
  for (const operation of allOperations(clone)) {
    operation.dtypePolicy = { inputDtype: "F32", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
    // Display-only previews cannot change the executable literal program.
    if (operation.op === "linear") delete operation.preview;
  }
  return clone;
}

function cacheTransitions(assignments: LiteralCalculationProgram["assignments"]): LiteralKvCacheTransition[] {
  return allOperations(assignments)
    .filter((operation): operation is Extract<Operation, { op: "scaled_dot_product_attention" }> => operation.op === "scaled_dot_product_attention")
    .map((operation) => {
      if (operation.layer === undefined) throw new Error(`${operation.id}: atenção literal sem camada não possui dono de cache KV.`);
      if (operation.kvSharing) {
        if (operation.kvSharing.producerLayer === undefined) throw new Error(`${operation.id}: compartilhamento KV não declara producerLayer.`);
        return {
          id: `${operation.id}_kv_cache`, layer: operation.layer, operation: "reuse-producer",
          keyInput: operation.key, valueInput: operation.value,
          cacheOutput: `past_key_values.${operation.kvSharing.producerLayer}`,
          producerLayer: operation.kvSharing.producerLayer,
        };
      }
      return {
        id: `${operation.id}_kv_cache`, layer: operation.layer, operation: "append-post-rope",
        keyInput: operation.key, valueInput: operation.value, cacheOutput: `past_key_values.${operation.layer}`,
      };
    });
}

function validateCacheTransitions(transitions: readonly LiteralKvCacheTransition[], assignments: LiteralCalculationProgram["assignments"]): void {
  const attention = allOperations(assignments).filter((operation) => operation.op === "scaled_dot_product_attention");
  if (transitions.length !== attention.length) throw new Error("Programa literal não declara uma transição KV para cada atenção.");
  const byId = new Map(transitions.map((transition) => [transition.id, transition]));
  for (const operation of attention) {
    if (operation.layer === undefined) throw new Error(`${operation.id}: atenção sem camada.`);
    const transition = byId.get(`${operation.id}_kv_cache`);
    if (!transition || transition.layer !== operation.layer || transition.keyInput !== operation.key || transition.valueInput !== operation.value) {
      throw new Error(`${operation.id}: transição KV literal não corresponde à atribuição de atenção.`);
    }
    if (operation.kvSharing) {
      if (transition.operation !== "reuse-producer" || transition.producerLayer !== operation.kvSharing.producerLayer) {
        throw new Error(`${operation.id}: compartilhamento KV literal não corresponde ao produtor declarado.`);
      }
    } else if (transition.operation !== "append-post-rope" || transition.cacheOutput !== `past_key_values.${operation.layer}`) {
      throw new Error(`${operation.id}: transição KV append literal inválida.`);
    }
  }
}

function validateOperationInputs(operation: Operation, available: ReadonlySet<string>, constants: ReadonlyMap<string, LiteralConstant>): void {
  const requireValue = (name: string): void => {
    if (name.startsWith("attention_mask:")) {
      if (!available.has("attention_mask")) throw new Error(`${operation.id}: máscara de atenção não declarada.`);
      return;
    }
    if (!available.has(name)) throw new Error(`${operation.id}: entrada ${name} não foi declarada antes do uso.`);
  };
  const requireTensor = (reference: TensorRef): void => {
    const constant = constants.get(reference.name);
    if (!constant || !sameShape(reference.shape, constant.logicalShape) || reference.storageDtype !== constant.storageDtype ||
      !sameQuantization(reference.quantization, constant.quantization)) {
      throw new Error(`${operation.id}: tensor ${reference.name} não possui constante literal F32 compatível.`);
    }
  };
  switch (operation.op) {
    case "embedding": case "per_layer_embedding": requireValue(operation.tokenInput); requireTensor(operation.weight); break;
    case "rms_norm": requireValue(operation.input); if (operation.weight) requireTensor(operation.weight); break;
    case "linear": requireValue(operation.input); requireTensor(operation.weight); if (operation.bias) requireTensor(operation.bias); break;
    case "reshape_heads": case "reshape_per_layer": case "select_per_layer": case "activation": requireValue(operation.input); break;
    case "tensor_scale": requireValue(operation.input); requireTensor(operation.scalar); break;
    case "rotary_embedding": requireValue(operation.input); requireValue(operation.positionInput); break;
    case "scaled_dot_product_attention": requireValue(operation.query); requireValue(operation.key); requireValue(operation.value); requireValue(operation.maskInput); break;
    case "elementwise": operation.inputs.forEach(requireValue); break;
  }
}

function toEmbeddedModelIR(program: LiteralCalculationProgram): ModelIR {
  return {
    schemaVersion: 2,
    source: { path: "embedded://literal-calculation-program", format: program.sourceFormat },
    architecture: structuredClone(program.architecture),
    config: structuredClone(program.config),
    preview: { outputRows: 1, inputTerms: 1, includeWeights: false },
    inputs: program.inputs.map(({ name, description }) => ({ name, description })),
    prelude: structuredClone(program.assignments.prelude),
    layers: structuredClone(program.assignments.layers),
    epilogue: structuredClone(program.assignments.epilogue),
    fidelity: { exactByConstruction: false, assumptions: [], unsupported: [], warnings: [] },
  };
}

function decodeDenseConstantsAsF32(program: LiteralCalculationProgram): ReadonlyMap<string, DenseF32Tensor> {
  const constants = new Map(program.constants.map((constant) => [constant.name, constant]));
  const decoded = new Map<string, DenseF32Tensor>();
  for (const decoder of program.storageDecoders) {
    const constant = constants.get(decoder.output);
    if (!constant) throw new Error(`${decoder.id}: constante de storage não encontrada após validação.`);
    const bytes = decodeBase64(constant.payloadBase64, constant.name);
    if (decoder.operation === "mlx-affine-u32-to-f32") {
      const scales = decoded.get(decoder.scaleInput);
      const biases = decoder.biasInput ? decoded.get(decoder.biasInput) : undefined;
      if (!scales || (decoder.biasInput && !biases)) throw new Error(`${decoder.id}: parâmetros affine não foram decodificados antes do peso.`);
      decoded.set(decoder.output, decodeMlxAffineConstant(constant, bytes, decoder, scales, biases));
    } else if (decoder.operation === "ggml-q8-0-to-f32") {
      decoded.set(decoder.output, decodeGgmlQ8_0Constant(constant, bytes, decoder));
    } else {
      const elements = product(constant.logicalShape);
      const values = new Float32Array(elements);
      for (let index = 0; index < elements; index += 1) {
        const offset = index * decoder.address.elementBytes;
        values[index] = decodeLiteralDenseElementF32(
          decoder,
          bytes.subarray(offset, offset + decoder.address.elementBytes),
        ).decodedF32;
      }
      decoded.set(decoder.output, { shape: [...constant.logicalShape], values });
    }
  }
  return decoded;
}

/** Materializes only already embedded payloads; it never opens a source model. */
export function decodeLiteralStorageBundleF32(bundle: LiteralStorageBundle): ReadonlyMap<string, DenseF32Tensor> {
  validateLiteralStorageBundle(bundle);
  return decodeDenseConstantsAsF32({ constants: bundle.constants, storageDecoders: bundle.storageDecoders } as LiteralCalculationProgram);
}

/** Validates the standalone storage/decoder portion shared by literal adapters. */
export function validateLiteralStorageBundle(bundle: LiteralStorageBundle): void {
  const constants = new Map<string, LiteralConstant>();
  for (const constant of bundle.constants) {
    if (constants.has(constant.name)) throw new Error(`Programa literal contém constante duplicada: ${constant.name}.`);
    validateLiteralConstant(constant);
    const payload = decodeBase64(constant.payloadBase64, constant.name);
    if (payload.length !== literalStorageByteLength(constant.storageDtype, constant.storageShape)) {
      throw new Error(`${constant.name}: payload base64 não corresponde ao shape ${constant.storageDtype} declarado.`);
    }
    constants.set(constant.name, constant);
  }
  if (constants.size === 0) throw new Error("Programa literal não contém constantes incorporadas.");
  if (!Array.isArray(bundle.storageDecoders)) throw new Error("Programa literal não declara decoders de storage.");
  validateStorageDecoders(bundle.storageDecoders, constants);
}

/** Checks that a semantic assignment can only name a compatible embedded tensor. */
export function validateLiteralStorageReference(reference: TensorRef, constants: ReadonlyMap<string, LiteralConstant>): void {
  const constant = constants.get(reference.name);
  if (!constant || !sameShape(reference.shape, constant.logicalShape) || reference.storageDtype !== constant.storageDtype ||
    !sameQuantization(reference.quantization, constant.quantization)) {
    throw new Error(`Referência literal ${reference.name} não possui constante incorporada compatível.`);
  }
}

function referencedTensors(ir: ModelIR): Map<string, TensorRef> {
  const references = new Map<string, TensorRef>();
  for (const operation of allOperations({ prelude: ir.prelude, layers: ir.layers, epilogue: ir.epilogue })) {
    if (operation.op === "embedding" || operation.op === "per_layer_embedding") registerReference(references, operation.weight);
    if (operation.op === "rms_norm" && operation.weight) registerReference(references, operation.weight);
    if (operation.op === "linear") {
      registerReference(references, operation.weight);
      if (operation.bias) registerReference(references, operation.bias);
    }
    if (operation.op === "tensor_scale") registerReference(references, operation.scalar);
  }
  return references;
}

function registerReference(references: Map<string, TensorRef>, reference: TensorRef): void {
  const previous = references.get(reference.name);
  if (previous && (previous.storageDtype !== reference.storageDtype || !sameShape(previous.shape, reference.shape) ||
    !sameQuantization(previous.quantization, reference.quantization))) {
    throw new Error(`IR contém referências incompatíveis para ${reference.name}.`);
  }
  references.set(reference.name, reference);
}

function assertLiteralReference(reference: TensorRef, tensor: TensorInfo | undefined): void {
  if (!tensor || reference.storageDtype !== tensor.storageDtype || !sameShape(reference.shape, tensor.logicalShape) ||
    !sameQuantization(reference.quantization, tensor.quantization)) {
    throw new Error(`${reference.name}: referência literal diverge do catálogo de origem.`);
  }
  if (!tensor.quantization) {
    if (!isSupportedDenseStorageDtype(tensor.storageDtype) || !sameShape(tensor.storageShape, tensor.logicalShape) || !validLiteralStorageShape(tensor.storageShape)) {
      throw new Error(`${reference.name}: exportação literal aceita tensor denso somente em F32/F16/BF16 com shape lógico idêntico ao storage.`);
    }
    return;
  }
  if (isGgmlQ8_0Tensor(tensor)) return;
  if (tensor.storageDtype !== "U32" || !validShape(tensor.storageShape) || !validShape(tensor.logicalShape)) {
    throw new Error(`${reference.name}: storage quantizado literal exige U32 e shapes positivos declarados.`);
  }
}

async function embedLiteralConstant(
  constants: Map<string, LiteralConstant>,
  tensor: TensorInfo,
  reader: LiteralTensorReader,
): Promise<void> {
  const existing = constants.get(tensor.name);
  if (existing) {
    if (existing.storageDtype !== tensor.storageDtype || !sameShape(existing.storageShape, tensor.storageShape) ||
      !sameShape(existing.logicalShape, tensor.logicalShape) || !sameQuantization(existing.quantization, tensor.quantization)) {
      throw new Error(`${tensor.name}: constante literal foi requisitada com contratos incompatíveis.`);
    }
    return;
  }
  const payload = await reader.readTensorBytes(tensor);
  const expectedBytes = literalStorageByteLength(tensor.storageDtype, tensor.storageShape);
  if (payload.length !== expectedBytes) {
    throw new Error(`${tensor.name}: payload literal possui ${payload.length} bytes, esperado ${expectedBytes} para ${tensor.storageDtype}.`);
  }
  constants.set(tensor.name, {
    name: tensor.name,
    storageDtype: tensor.storageDtype as LiteralStorageDtype,
    storageShape: [...tensor.storageShape],
    logicalShape: [...tensor.logicalShape],
    layout: tensor.quantization?.family === "gguf" ? "ggml-first-axis-contiguous" : "row-major",
    byteOrder: "little-endian",
    encoding: "base64",
    payloadBase64: payload.toString("base64"),
    ...(tensor.quantization ? { quantization: structuredClone(tensor.quantization) } : {}),
  });
}

function assertMlxAffineTensor(tensor: TensorInfo, catalog: ModelCatalog): { scale: TensorInfo; bias?: TensorInfo } {
  const q = tensor.quantization;
  if (!q || q.family !== "mlx" || q.mode !== "affine" || tensor.storageDtype !== "U32" ||
    !isMlxAffineBitWidth(q.bits) || !Number.isInteger(q.groupSize) || q.groupSize! <= 0 || !q.scaleTensor || q.globalScaleTensor !== undefined) {
    throw new Error(`${tensor.name}: exportação literal aceita somente MLX affine U32 com bits {2,3,4,5,6,8}, group_size, scales e sem global_scale.`);
  }
  const groupSize = q.groupSize!;
  const bits = q.bits;
  const scale = catalog.tensors.get(q.scaleTensor);
  const bias = q.biasTensor ? catalog.tensors.get(q.biasTensor) : undefined;
  if (!scale || scale.quantization || !isSupportedDenseStorageDtype(scale.storageDtype) ||
    !sameShape(scale.storageShape, scale.logicalShape) || !validShape(scale.logicalShape) ||
    (q.biasTensor !== undefined && (!bias || bias.quantization || bias.storageDtype !== scale.storageDtype ||
      !sameShape(bias.storageShape, bias.logicalShape) || !sameShape(bias.logicalShape, scale.logicalShape)))) {
    throw new Error(`${tensor.name}: MLX affine literal requer scales e biases opcionais densos F32/F16/BF16 de shape/dtype idêntico.`);
  }
  if (tensor.storageShape.length !== 2 || tensor.logicalShape.length !== 2 || scale.logicalShape.length !== 2) {
    throw new Error(`${tensor.name}: MLX affine literal requer peso e parâmetros em matrizes 2D.`);
  }
  const [rows, packedWords] = tensor.storageShape;
  const [logicalRows, logicalColumns] = tensor.logicalShape;
  const [scaleRows, groups] = scale.logicalShape;
  if (rows === undefined || packedWords === undefined || logicalRows !== rows || logicalColumns === undefined ||
    scaleRows !== rows || groups === undefined || groups * groupSize !== logicalColumns ||
    logicalColumns * bits % 32 !== 0 || packedWords !== logicalColumns * bits / 32) {
    throw new Error(`${tensor.name}: MLX affine literal encontrou packing U32, shape lógico ou grupos incompatíveis.`);
  }
  return { scale, ...(bias ? { bias } : {}) };
}

function validateLiteralConstant(constant: LiteralConstant): void {
  if ((constant.layout !== "row-major" && constant.layout !== "ggml-first-axis-contiguous") || constant.byteOrder !== "little-endian" || constant.encoding !== "base64" ||
    !validLiteralStorageShape(constant.storageShape) || !validLiteralStorageShape(constant.logicalShape)) {
    throw new Error(`${constant.name}: contrato de constante literal inválido.`);
  }
  if (!constant.quantization) {
    if (!isSupportedDenseStorageDtype(constant.storageDtype) || !sameShape(constant.storageShape, constant.logicalShape)) {
      throw new Error(`${constant.name}: constante literal densa requer F32/F16/BF16 e shape lógico idêntico ao storage.`);
    }
    return;
  }
  const q = constant.quantization;
  if (constant.storageDtype === "GGML_Q8_0" && q.family === "gguf" && q.mode === "q8_0" && q.bits === 8 &&
    q.groupSize === 32 && q.tensorType === "GGML_TYPE_Q8_0" && constant.layout === "ggml-first-axis-contiguous" &&
    product(constant.storageShape) % 32 === 0) {
    return;
  }
  if (constant.storageDtype !== "U32" || q.family !== "mlx" || q.mode !== "affine" || !isMlxAffineBitWidth(q.bits) ||
    !Number.isInteger(q.groupSize) || q.groupSize! <= 0 || !q.scaleTensor || q.globalScaleTensor !== undefined ||
    constant.storageShape.length !== 2 || constant.logicalShape.length !== 2) {
    throw new Error(`${constant.name}: constante literal quantizada não possui contrato MLX affine U32 verificável.`);
  }
}

function mlxAffineStorageDecoder(
  constant: LiteralConstant,
  constants: ReadonlyMap<string, LiteralConstant>,
): LiteralMlxAffineStorageDecodeAssignment {
  const q = constant.quantization;
  const scale = q?.scaleTensor ? constants.get(q.scaleTensor) : undefined;
  if (!q || !isMlxAffineBitWidth(q.bits) || !Number.isInteger(q.groupSize) || !q.scaleTensor || !scale ||
    !isSupportedDenseStorageDtype(scale.storageDtype)) {
    throw new Error(`${constant.name}: constante MLX affine não possui metadados para decoder literal.`);
  }
  const groupSize = q.groupSize!;
  return {
    id: `decode_${constant.name}`,
    operation: "mlx-affine-u32-to-f32",
    input: `${constant.name}:storage`,
    output: constant.name,
    storageDtype: "U32",
    outputDtype: "F32",
    byteOrder: "little-endian",
    packing: "row-major-contiguous-lsb-first-u32",
    bits: q.bits,
    groupSize,
    scaleInput: q.scaleTensor,
    ...(q.biasTensor ? { biasInput: q.biasTensor } : {}),
    parameterDtype: scale.storageDtype,
    semantics: "F32(scale * unsigned_code + bias); BF16 parameters round each affine result to BF16 before F32 output",
  };
}

function validateDenseStorageDecoder(decoder: LiteralDenseStorageDecodeAssignment, constant: LiteralConstant): void {
  const expected = buildLiteralDenseStorageDecodeAssignment(constant);
  if (!isDeepStrictEqual(decoder, expected)) {
    throw new Error(`${decoder.id}: decoder de storage literal não corresponde à constante declarada.`);
  }
}

function validateMlxAffineStorageDecoder(
  decoder: LiteralMlxAffineStorageDecodeAssignment,
  constant: LiteralConstant,
  constants: ReadonlyMap<string, LiteralConstant>,
  decoded: ReadonlySet<string>,
): void {
  const q = constant.quantization;
  const scale = q?.scaleTensor ? constants.get(q.scaleTensor) : undefined;
  const bias = q?.biasTensor ? constants.get(q.biasTensor) : undefined;
  if (!q || !scale || !isSupportedDenseStorageDtype(scale.storageDtype) || scale.quantization || !decoded.has(scale.name) ||
    (q.biasTensor !== undefined && (!bias || bias.quantization || bias.storageDtype !== scale.storageDtype || !decoded.has(bias.name))) ||
    decoder.id !== `decode_${constant.name}` || decoder.input !== `${constant.name}:storage` || decoder.storageDtype !== "U32" ||
    decoder.outputDtype !== "F32" || decoder.byteOrder !== "little-endian" || decoder.packing !== "row-major-contiguous-lsb-first-u32" ||
    !isMlxAffineBitWidth(decoder.bits) || decoder.bits !== q.bits || !Number.isInteger(decoder.groupSize) || decoder.groupSize !== q.groupSize ||
    decoder.scaleInput !== q.scaleTensor || decoder.biasInput !== q.biasTensor || decoder.parameterDtype !== scale.storageDtype ||
    decoder.semantics !== "F32(scale * unsigned_code + bias); BF16 parameters round each affine result to BF16 before F32 output" ||
    !mlxAffineShapesMatch(constant, scale.logicalShape, bias?.logicalShape, decoder.bits, decoder.groupSize)) {
    throw new Error(`${decoder.id}: decoder MLX affine literal não corresponde ao packing, parâmetros ou shapes declarados.`);
  }
}

function mlxAffineShapesMatch(
  constant: LiteralConstant,
  scaleShape: readonly number[],
  biasShape: readonly number[] | undefined,
  bits: number,
  groupSize: number,
): boolean {
  if (constant.storageShape.length !== 2 || constant.logicalShape.length !== 2 || scaleShape.length !== 2 ||
    (biasShape && !sameShape(biasShape, scaleShape))) return false;
  const [rows, packedWords] = constant.storageShape;
  const [logicalRows, logicalColumns] = constant.logicalShape;
  const [scaleRows, groups] = scaleShape;
  return rows !== undefined && packedWords !== undefined && logicalRows === rows && logicalColumns !== undefined &&
    scaleRows === rows && groups !== undefined && groups * groupSize === logicalColumns &&
    logicalColumns * bits % 32 === 0 && packedWords === logicalColumns * bits / 32;
}

function decodeMlxAffineConstant(
  constant: LiteralConstant,
  bytes: Buffer,
  decoder: LiteralMlxAffineStorageDecodeAssignment,
  scales: DenseF32Tensor,
  biases: DenseF32Tensor | undefined,
): DenseF32Tensor {
  if (!mlxAffineShapesMatch(constant, scales.shape, biases?.shape, decoder.bits, decoder.groupSize)) {
    throw new Error(`${decoder.id}: shapes affine inválidos durante replay literal.`);
  }
  const [rows, packedWords] = constant.storageShape as [number, number];
  const [, columns] = constant.logicalShape as [number, number];
  const [, groups] = scales.shape as [number, number];
  const values = new Float32Array(rows * columns);
  const mask = (1 << decoder.bits) - 1;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const bitOffset = column * decoder.bits;
      const wordIndex = row * packedWords + Math.floor(bitOffset / 32);
      const shift = bitOffset % 32;
      const word = bytes.readUInt32LE(wordIndex * 4);
      const code = shift + decoder.bits <= 32
        ? (word >>> shift) & mask
        : ((word >>> shift) | (bytes.readUInt32LE((wordIndex + 1) * 4) << (32 - shift))) & mask;
      const parameterIndex = row * groups + Math.floor(column / decoder.groupSize);
      const affine = Math.fround(Math.fround(scales.values[parameterIndex]! * code) + (biases?.values[parameterIndex] ?? 0));
      values[row * columns + column] = decoder.parameterDtype === "BF16" ? roundF32ToBF16(affine) : affine;
    }
  }
  if (!constant.quantization) throw new Error(`${decoder.id}: replay affine sem proveniência de quantização.`);
  return { shape: [...constant.logicalShape], values, sourceQuantization: structuredClone(constant.quantization) };
}

function decodeGgmlQ8_0Constant(
  constant: LiteralConstant,
  bytes: Buffer,
  decoder: LiteralGgmlQ8_0StorageDecodeAssignment,
): DenseF32Tensor {
  const elements = product(constant.logicalShape);
  if (elements !== product(constant.storageShape) || elements % decoder.blockSize !== 0 ||
    bytes.length !== elements / decoder.blockSize * decoder.blockBytes) {
    throw new Error(`${decoder.id}: payload GGML Q8_0 não corresponde aos blocos e shapes declarados.`);
  }
  const values = new Float32Array(elements);
  for (let block = 0; block < elements / decoder.blockSize; block += 1) {
    const offset = block * decoder.blockBytes;
    const scale = decodeIeeeF16ToF32(bytes.readUInt16LE(offset));
    for (let index = 0; index < decoder.blockSize; index += 1) {
      values[block * decoder.blockSize + index] = Math.fround(scale * bytes.readInt8(offset + 2 + index));
    }
  }
  if (!constant.quantization) throw new Error(`${decoder.id}: replay Q8_0 sem proveniência de quantização.`);
  return { shape: [...constant.logicalShape], values, sourceQuantization: structuredClone(constant.quantization) };
}

function isMlxAffineBitWidth(bits: number | undefined): bits is LiteralMlxAffineStorageDecodeAssignment["bits"] {
  return bits === 2 || bits === 3 || bits === 4 || bits === 5 || bits === 6 || bits === 8;
}

function isGgmlQ8_0Tensor(tensor: TensorInfo): boolean {
  const q = tensor.quantization;
  return tensor.storageDtype === "GGML_Q8_0" && q?.family === "gguf" && q.mode === "q8_0" && q.bits === 8 &&
    q.groupSize === 32 && q.tensorType === "GGML_TYPE_Q8_0" && validShape(tensor.storageShape) &&
    sameShape(tensor.storageShape, tensor.logicalShape) && product(tensor.storageShape) % 32 === 0;
}

function sameQuantization(left: QuantizationSpec | undefined, right: QuantizationSpec | undefined): boolean {
  if (!left || !right) return left === right;
  return left.family === right.family && left.mode === right.mode && left.bits === right.bits && left.groupSize === right.groupSize &&
    left.tensorType === right.tensorType && left.scaleTensor === right.scaleTensor && left.biasTensor === right.biasTensor &&
    left.globalScaleTensor === right.globalScaleTensor;
}

function storageDecoder(constant: LiteralConstant, constants: ReadonlyMap<string, LiteralConstant>): LiteralStorageDecodeAssignment {
  if (constant.quantization) {
    if (constant.quantization.family === "gguf" && constant.quantization.mode === "q8_0") return ggmlQ8_0StorageDecoder(constant);
    return mlxAffineStorageDecoder(constant, constants);
  }
  if (!isSupportedDenseStorageDtype(constant.storageDtype)) throw new Error(`${constant.name}: constante densa literal não possui dtype IEEE suportado.`);
  return buildLiteralDenseStorageDecodeAssignment(constant);
}

function denseIeeeDecodeProgram(dtype: LiteralDenseStorageDtype): LiteralDenseIeeeDecodeProgram {
  if (dtype === "F32") {
    return { kind: "ieee-binary32-bitcast", schemaVersion: 1, read: "uint32-little-endian", resultBits: "sourceBits" };
  }
  if (dtype === "BF16") {
    return { kind: "ieee-bfloat16-expand", schemaVersion: 1, read: "uint16-little-endian", resultBits: "sourceBits << 16" };
  }
  return {
    kind: "ieee-binary16-expand",
    schemaVersion: 1,
    read: "uint16-little-endian",
    fields: { signMask: 0x8000, signShift: 15, exponentMask: 0x7c00, exponentShift: 10, fractionMask: 0x03ff },
    extraction: {
      sign: "(sourceBits & signMask) >> signShift",
      exponent: "(sourceBits & exponentMask) >> exponentShift",
      fraction: "sourceBits & fractionMask",
    },
    normal: {
      predicate: "0 < exponent < 31",
      resultBits: "(sign << 31) | ((exponent + 112) << 23) | (fraction << 13)",
    },
    zero: { predicate: "exponent == 0 && fraction == 0", resultBits: "sign << 31" },
    subnormal: {
      predicate: "exponent == 0 && fraction != 0",
      normalization: "left-shift fraction until bit 10 is one; shiftCount starts at zero",
      resultBits: "(sign << 31) | ((113 - shiftCount) << 23) | ((normalizedFraction & 0x03ff) << 13)",
    },
    infinityOrNaN: {
      predicate: "exponent == 31",
      resultBits: "(sign << 31) | 0x7f800000 | (fraction << 13)",
    },
  };
}

function sameBinary16DecodeProgram(program: Extract<LiteralDenseIeeeDecodeProgram, { kind: "ieee-binary16-expand" }>): boolean {
  const expected = denseIeeeDecodeProgram("F16") as Extract<LiteralDenseIeeeDecodeProgram, { kind: "ieee-binary16-expand" }>;
  return isDeepStrictEqual(program, expected);
}

function ieeeF16ToF32Bits(sourceBits: number): number {
  const sign = (sourceBits >>> 15) & 1;
  const exponent = (sourceBits >>> 10) & 0x1f;
  const fraction = sourceBits & 0x03ff;
  if (exponent === 0x1f) return ((sign << 31) | 0x7f800000 | (fraction << 13)) >>> 0;
  if (exponent !== 0) return ((sign << 31) | ((exponent + 112) << 23) | (fraction << 13)) >>> 0;
  if (fraction === 0) return (sign << 31) >>> 0;
  let normalizedFraction = fraction;
  let shiftCount = 0;
  while ((normalizedFraction & 0x0400) === 0) {
    normalizedFraction <<= 1;
    shiftCount += 1;
  }
  return ((sign << 31) | ((113 - shiftCount) << 23) | ((normalizedFraction & 0x03ff) << 13)) >>> 0;
}

function ggmlQ8_0StorageDecoder(constant: LiteralConstant): LiteralGgmlQ8_0StorageDecodeAssignment {
  if (constant.storageDtype !== "GGML_Q8_0" || constant.quantization?.family !== "gguf" ||
    constant.quantization.mode !== "q8_0" || constant.quantization.bits !== 8 || constant.quantization.groupSize !== 32 ||
    constant.quantization.tensorType !== "GGML_TYPE_Q8_0" || constant.layout !== "ggml-first-axis-contiguous") {
    throw new Error(`${constant.name}: constante GGML Q8_0 não possui contrato literal verificável.`);
  }
  return {
    id: `decode_${constant.name}`,
    operation: "ggml-q8-0-to-f32",
    input: `${constant.name}:storage`,
    output: constant.name,
    storageDtype: "GGML_Q8_0",
    outputDtype: "F32",
    byteOrder: "little-endian",
    packing: "blocks-of-32-f16-scale-then-i8-codes",
    blockSize: 32,
    blockBytes: 34,
    semantics: "F32(IEEE-754 binary16 scale * signed int8 code) for each consecutive GGML logical value",
  };
}

function validateStorageDecoders(
  decoders: readonly LiteralStorageDecodeAssignment[],
  constants: ReadonlyMap<string, LiteralConstant>,
): void {
  if (decoders.length !== constants.size) throw new Error("Programa literal deve declarar exatamente um decoder de storage por constante.");
  const seenIds = new Set<string>();
  const decoded = new Set<string>();
  for (const decoder of decoders) {
    if (seenIds.has(decoder.id)) throw new Error(`Programa literal contém decoder duplicado: ${decoder.id}.`);
    seenIds.add(decoder.id);
    const constant = constants.get(decoder.output);
    if (!constant || decoded.has(decoder.output)) throw new Error(`${decoder.id}: decoder de storage literal não corresponde à constante declarada.`);
    if (decoder.operation === "mlx-affine-u32-to-f32") validateMlxAffineStorageDecoder(decoder, constant, constants, decoded);
    else if (decoder.operation === "ggml-q8-0-to-f32") validateGgmlQ8_0StorageDecoder(decoder, constant);
    else validateDenseStorageDecoder(decoder, constant);
    decoded.add(decoder.output);
  }
}

function validateGgmlQ8_0StorageDecoder(
  decoder: LiteralGgmlQ8_0StorageDecodeAssignment,
  constant: LiteralConstant,
): void {
  const q = constant.quantization;
  if (constant.storageDtype !== "GGML_Q8_0" || constant.layout !== "ggml-first-axis-contiguous" || !q ||
    q.family !== "gguf" || q.mode !== "q8_0" || q.bits !== 8 || q.groupSize !== 32 || q.tensorType !== "GGML_TYPE_Q8_0" ||
    decoder.id !== `decode_${constant.name}` || decoder.input !== `${constant.name}:storage` || decoder.storageDtype !== "GGML_Q8_0" ||
    decoder.outputDtype !== "F32" || decoder.byteOrder !== "little-endian" || decoder.packing !== "blocks-of-32-f16-scale-then-i8-codes" ||
    decoder.blockSize !== 32 || decoder.blockBytes !== 34 ||
    decoder.semantics !== "F32(IEEE-754 binary16 scale * signed int8 code) for each consecutive GGML logical value" ||
    product(constant.storageShape) !== product(constant.logicalShape) || product(constant.storageShape) % decoder.blockSize !== 0) {
    throw new Error(`${decoder.id}: decoder GGML Q8_0 literal não corresponde ao packing, escala ou shapes declarados.`);
  }
}

function decodeOperationFor(storageDtype: LiteralDenseStorageDtype): LiteralDenseStorageDecodeAssignment["operation"] {
  switch (storageDtype) {
    case "F32": return "ieee-f32-little-endian";
    case "F16": return "ieee-f16-to-f32";
    case "BF16": return "ieee-bf16-to-f32";
  }
}

function decodeStoredValueAsF32(bytes: Buffer, offset: number, operation: LiteralDenseStorageDecodeAssignment["operation"]): number {
  switch (operation) {
    case "ieee-f32-little-endian": return bytes.readFloatLE(offset);
    case "ieee-f16-to-f32": return decodeIeeeF16ToF32(bytes.readUInt16LE(offset));
    case "ieee-bf16-to-f32": return decodeIeeeBF16ToF32(bytes.readUInt16LE(offset));
  }
}

function decodeBase64(payloadBase64: string, name: string): Buffer {
  if (!/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(payloadBase64)) {
    throw new Error(`${name}: payload base64 literal inválido.`);
  }
  return Buffer.from(payloadBase64, "base64");
}

function isSupportedDenseStorageDtype(dtype: string): dtype is LiteralDenseStorageDtype {
  return dtype === "F32" || dtype === "F16" || dtype === "BF16";
}

function storageByteWidth(dtype: string): number {
  if (dtype !== "U32" && !isSupportedDenseStorageDtype(dtype)) throw new Error(`storageDtype literal não suportado: ${dtype}.`);
  return STORAGE_BYTES[dtype as Exclude<LiteralStorageDtype, "GGML_Q8_0">];
}

function literalStorageByteLength(dtype: string, shape: readonly number[]): number {
  const elements = product(shape);
  if (!Number.isSafeInteger(elements) || elements <= 0) throw new Error(`shape literal inválido para ${dtype}.`);
  if (dtype === "GGML_Q8_0") {
    if (elements % 32 !== 0) throw new Error(`GGML_Q8_0 exige número de elementos múltiplo de 32.`);
    return elements / 32 * 34;
  }
  return elements * storageByteWidth(dtype);
}

function allOperations(assignments: LiteralCalculationProgram["assignments"]): Operation[] {
  return [...assignments.prelude, ...assignments.layers.flatMap((layer) => layer.operations), ...assignments.epilogue];
}

function product(shape: readonly number[]): number {
  return shape.reduce((result, dimension) => result * dimension, 1);
}

function validShape(shape: readonly number[]): boolean {
  return shape.length > 0 && shape.every((dimension) => Number.isInteger(dimension) && dimension > 0) && Number.isSafeInteger(product(shape));
}

/** Safetensors scalar clipping bounds have shape []; they still contain one byte sequence. */
function validLiteralStorageShape(shape: readonly number[]): boolean {
  return shape.every((dimension) => Number.isInteger(dimension) && dimension > 0) && Number.isSafeInteger(product(shape));
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((dimension, index) => dimension === right[index]);
}
