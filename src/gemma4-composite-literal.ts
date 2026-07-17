import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, rename, unlink } from "node:fs/promises";
import { once } from "node:events";
import * as path from "node:path";
import {
  buildLiteralStorageBundle,
  decodeLiteralStorageBundleF32,
  validateLiteralStorageBundle,
  validateLiteralStorageReference,
  type LiteralConstant,
  type LiteralDenseStorageDecodeAssignment,
  type LiteralStorageDecodeAssignment,
  type LiteralStorageBundle,
  type LiteralTensorReader,
} from "./literal.js";
import {
  executeGemma4CompositeF32,
  generateGemma4CompositeF32,
  type Gemma4CompositeExecutionRequest,
  type Gemma4CompositeExecutionResult,
  type Gemma4CompositeGenerationRequest,
  type Gemma4CompositeGenerationResult,
  type Gemma4CompositeProgram,
} from "./gemma4-composite.js";
import type { Gemma4AudioAssignment } from "./gemma4-audio.js";
import type { Gemma4VisionAssignment } from "./gemma4-vision.js";
import type { ModelCatalog, Operation, TensorInfo, TensorRef } from "./types.js";

/** Divisible by three so every non-final base64 chunk has no padding. */
const BASE64_CHUNK_BYTES = 12 * 1024 * 1024;

export interface Gemma4CompositeLiteralInput {
  name: string;
  description: string;
  required: boolean;
  dtype: string;
  shape: string;
}

/**
 * A source-independent literal program for the complete registered Gemma 4
 * composite boundary.  The three assignment scopes are kept separate because
 * image and video reuse the vision feature program, while their outer scatter
 * steps remain distinct, named dependencies in the enclosing program.
 */
export interface Gemma4CompositeLiteralCalculationProgram extends LiteralStorageBundle {
  schemaVersion: 1;
  kind: "gemma4-composite-literal-calculation-program";
  sourceFormat: "safetensors";
  numericPolicy: {
    inputDtype: "I32/F32/BOOL";
    computeDtype: "F32";
    /**
     * A single policy applies only when every assignment shares the same
     * reduction boundary. Otherwise every operation owns its declared F32 or
     * F64 reduction contract.
     */
    accumulationDtype: "F32" | "operation-declared";
    outputDtype: "F32" | "operation-declared";
    scalarSemantics:
      | "IEEE-754 binary32; host libm results rounded to F32"
      | "IEEE-754 binary32 reductions; each operation declares its F32 or BF16 result cast"
      /** Legacy headers remain readable; newly written artifacts use the expanded declarations below. */
      | "IEEE-754 binary32 products; each operation declares its ordered-scalar or interleaved-lane F32/F64 reduction and F32 or BF16 result cast"
      | "IEEE-754 binary32 products; each operation declares its ordered-scalar, contiguous blocked-term, or interleaved-lane F32/F64 reduction and F32 or BF16 result cast"
      | "IEEE-754 binary32; each operation declares ordered-scalar, contiguous blocked-term, separately-rounded F32-lane, or fused-multiply-add reduction and its F32 or BF16 result cast"
      | "IEEE-754 binary32 products; each operation declares its ordered-scalar, contiguous blocked-term, blocked tiled-lane, or interleaved-lane F32/F64 reduction and F32 or BF16 result cast"
      | "IEEE-754 binary32; each operation declares ordered-scalar, contiguous blocked-term, blocked tiled-lane, separately-rounded F32-lane, or fused-multiply-add reduction and its F32 or BF16 result cast";
  };
  inputs: Gemma4CompositeLiteralInput[];
  /** No checkpoint path is retained: all tensor bytes are in `constants`. */
  /**
   * Checkpoint-resident bytes intentionally unreachable in the declared
   * forward graph. Gemma 4 shared-KV consumer layers retain their local K/V
   * tensors in storage while the authoritative graph reads the producer cache
   * instead; retaining this list makes that non-use explicit and auditable.
   */
  unreachableConstants: Gemma4CompositeUnreachableConstant[];
  program: Gemma4CompositeProgram;
  assignments: {
    composite: Gemma4CompositeProgram["assignments"];
    vision: Gemma4VisionAssignment[];
    audio: Gemma4AudioAssignment[];
    textLayers: Gemma4CompositeProgram["textProgram"]["layers"];
    textEpilogue: Operation[];
  };
  outputs: { embeddings: "hidden_states_0"; perLayerInputs: "ple_inputs"; logits: "softcapped_logits" | "logits" };
  /** Present on streamed artifacts; binds each embedded payload after source removal. */
  payloadIntegrity?: Gemma4CompositeLiteralPayloadIntegrityEntry[];
}

export interface Gemma4CompositeUnreachableConstant {
  name: string;
  reason: "shared-kv-consumer-local-kv-is-runtime-unreachable";
  producerLayer: number;
}

/**
 * A source-independent integrity commitment for one embedded storage payload.
 * It is written only after the streamed bytes have been read and is therefore
 * usable after the original checkpoint has been removed.
 */
export interface Gemma4CompositeLiteralPayloadIntegrityEntry {
  name: string;
  payloadBytes: number;
  sha256: string;
}

/**
 * Result of the bounded-memory writer used for real dense Gemma 4 packages.
 * The hash covers the exact UTF-8 JSON bytes written to `output`, including
 * its trailing newline, so a checkpoint manifest can bind the artifact
 * without parsing it back into a multi-gigabyte JavaScript object.
 */
export interface Gemma4CompositeLiteralWriteResult {
  output: string;
  artifactSha256: string;
  artifactBytes: number;
  constants: number;
  embeddedPayloadBytes: number;
}

/**
 * Literalizes the entire registered composite, not only Gemma4Text.  Dense
 * Gemma 4 is deliberately required here: a quantized package needs an
 * architecture-specific quantization contract before it can be replayed.
 */
export async function buildGemma4CompositeLiteralCalculationProgram(
  program: Gemma4CompositeProgram,
  catalog: ModelCatalog,
  reader: LiteralTensorReader,
): Promise<Gemma4CompositeLiteralCalculationProgram> {
  if (catalog.format !== "safetensors" || program.sourceFormat !== "safetensors") {
    throw new Error("Programa literal Gemma 4 composite requer Safetensors denso registrado.");
  }
  validateGemma4TextReductionSchedules(program);
  const references = compositeReferences(program);
  const unreachableConstants = sharedKvUnreachableConstants(program, catalog, references);
  const catalogNames = new Set(catalog.tensors.keys());
  const allReferences = new Map(references);
  for (const entry of unreachableConstants) {
    const tensor = catalog.tensors.get(entry.name);
    if (!tensor) throw new Error(`${entry.name}: tensor KV compartilhado ausente do catálogo.`);
    allReferences.set(entry.name, tensorReference(tensor));
  }
  if (allReferences.size !== catalogNames.size || [...catalogNames].some((name) => !allReferences.has(name))) {
    const omitted = [...catalogNames].filter((name) => !allReferences.has(name));
    throw new Error(`Programa literal Gemma 4 composite não possui atribuição semântica para todos os tensores do pacote${omitted.length ? `: ${omitted.slice(0, 4).join(", ")}` : "."}`);
  }
  const storage = await buildLiteralStorageBundle(catalog, reader, allReferences.values());
  const embeddedProgram = embeddedCompositeProgram(program);
  const literal: Gemma4CompositeLiteralCalculationProgram = {
    schemaVersion: 1,
    kind: "gemma4-composite-literal-calculation-program",
    sourceFormat: "safetensors",
    numericPolicy: gemma4CompositeLiteralNumericPolicy(program),
    inputs: literalInputs(),
    ...storage,
    unreachableConstants,
    program: embeddedProgram,
    assignments: {
      composite: structuredClone(embeddedProgram.assignments),
      vision: structuredClone(embeddedProgram.visionProgram.assignments),
      audio: structuredClone(embeddedProgram.audioProgram.assignments),
      textLayers: structuredClone(embeddedProgram.textProgram.layers),
      textEpilogue: structuredClone(embeddedProgram.textProgram.epilogue),
    },
    outputs: structuredClone(embeddedProgram.outputs),
  };
  validateGemma4CompositeLiteralCalculationProgram(literal);
  return literal;
}

/**
 * Atomically writes the same literal calculation-program schema without ever
 * retaining all base64 payloads or a whole JSON string in memory.  The dense
 * E4B package is roughly 15 GiB before base64 expansion, so the object-return
 * builder above remains useful for small callers/tests while this writer is
 * the production boundary for an actual checkpoint.
 */
export async function writeGemma4CompositeLiteralCalculationProgram(
  program: Gemma4CompositeProgram,
  catalog: ModelCatalog,
  reader: LiteralTensorReader,
  output: string,
): Promise<Gemma4CompositeLiteralWriteResult> {
  validateGemma4TextReductionSchedules(program);
  const prepared = prepareStreamedDenseLiteral(program, catalog);
  await mkdir(path.dirname(output), { recursive: true });
  const temporary = `${output}.${process.pid}.${Date.now()}.tmp`;
  const stream = createWriteStream(temporary, { encoding: "utf8", flags: "w" });
  const digest = createHash("sha256");
  let artifactBytes = 0;
  let embeddedPayloadBytes = 0;
  const payloadIntegrity: Gemma4CompositeLiteralPayloadIntegrityEntry[] = [];

  const write = async (chunk: string): Promise<void> => {
    digest.update(chunk, "utf8");
    artifactBytes += Buffer.byteLength(chunk);
    if (!stream.write(chunk, "utf8")) await once(stream, "drain");
  };

  try {
    await once(stream, "open");
    await write(`{"schemaVersion":1,"kind":"gemma4-composite-literal-calculation-program","sourceFormat":"safetensors","numericPolicy":${JSON.stringify(gemma4CompositeLiteralNumericPolicy(program))},"inputs":${JSON.stringify(literalInputs())},"constants":[`);
    for (let index = 0; index < prepared.constants.length; index += 1) {
      const constant = prepared.constants[index]!;
      if (index > 0) await write(",");
      await write(`${JSON.stringify(constant.metadata).slice(0, -1)},"payloadBase64":"`);
      const payload = await writeBase64Payload(constant, reader, write);
      embeddedPayloadBytes += payload.bytes;
      payloadIntegrity.push({ name: constant.name, payloadBytes: payload.bytes, sha256: payload.sha256 });
      await write("\"}");
    }
    await write(`],"unreachableConstants":${JSON.stringify(prepared.unreachableConstants)},"storageDecoders":${JSON.stringify(prepared.storageDecoders)},"program":${JSON.stringify(prepared.embeddedProgram)},"assignments":${JSON.stringify(prepared.assignments)},"outputs":${JSON.stringify(prepared.outputs)},"payloadIntegrity":${JSON.stringify(payloadIntegrity)}}\n`);
    stream.end();
    await once(stream, "finish");
    await rename(temporary, output);
  } catch (error) {
    stream.destroy();
    await unlink(temporary).catch(() => undefined);
    throw error;
  }

  return {
    output,
    artifactSha256: digest.digest("hex"),
    artifactBytes,
    constants: prepared.constants.length,
    embeddedPayloadBytes,
  };
}

async function writeBase64Payload(
  constant: StreamedDenseConstant,
  reader: LiteralTensorReader,
  write: (chunk: string) => Promise<void>,
): Promise<{ bytes: number; sha256: string }> {
  const digest = createHash("sha256");
  if (reader.readTensorBytesRange) {
    let offset = 0;
    while (offset < constant.expectedByteLength) {
      const byteLength = Math.min(BASE64_CHUNK_BYTES, constant.expectedByteLength - offset);
      const bytes = await reader.readTensorBytesRange(constant.tensor, offset, byteLength);
      if (bytes.length !== byteLength) throw new Error(`${constant.name}: leitor literal retornou ${bytes.length} bytes no range ${offset}, esperados ${byteLength}.`);
      // Base64 alphabet is JSON-safe; the quote delimiters are emitted by the
      // caller. Every non-final chunk is 3-byte aligned by the constant above.
      await write(bytes.toString("base64"));
      digest.update(bytes);
      offset += byteLength;
    }
    return { bytes: offset, sha256: digest.digest("hex") };
  }
  if (constant.expectedByteLength > BASE64_CHUNK_BYTES) {
    throw new Error(`${constant.name}: exportação literal de tensor grande requer readTensorBytesRange; o leitor não pode formar um base64 multi-GiB inteiro.`);
  }
  const bytes = await reader.readTensorBytes(constant.tensor);
  if (bytes.length !== constant.expectedByteLength) throw new Error(`${constant.name}: leitor literal retornou ${bytes.length} bytes, esperados ${constant.expectedByteLength}.`);
  await write(bytes.toString("base64"));
  digest.update(bytes);
  return { bytes: bytes.length, sha256: digest.digest("hex") };
}

/** Replays prefill solely from the literal's embedded bytes and assignments. */
export function executeGemma4CompositeLiteralF32(
  literal: Gemma4CompositeLiteralCalculationProgram,
  request: Omit<Gemma4CompositeExecutionRequest, "tensors">,
): Gemma4CompositeExecutionResult {
  validateGemma4CompositeLiteralCalculationProgram(literal);
  assertSynchronousCompositeF32Compatibility(literal.program);
  return executeGemma4CompositeF32(literal.program, { ...request, tensors: decodeLiteralStorageBundleF32(literal) });
}

/** Replays greedy cached decoding solely from the literal's embedded bytes. */
export function generateGemma4CompositeLiteralF32(
  literal: Gemma4CompositeLiteralCalculationProgram,
  request: Omit<Gemma4CompositeGenerationRequest, "tensors">,
): Gemma4CompositeGenerationResult {
  validateGemma4CompositeLiteralCalculationProgram(literal);
  assertSynchronousCompositeF32Compatibility(literal.program);
  return generateGemma4CompositeF32(literal.program, { ...request, tensors: decodeLiteralStorageBundleF32(literal) });
}

/**
 * Validates all three semantic scopes and their storage references before
 * replay.  This rejects a partial tower, an omitted vision-mask/cache step,
 * or a program which tries to retain a source-model path.
 */
export function validateGemma4CompositeLiteralCalculationProgram(literal: Gemma4CompositeLiteralCalculationProgram): void {
  validateGemma4TextReductionSchedules(literal.program);
  if (literal.schemaVersion !== 1 || literal.kind !== "gemma4-composite-literal-calculation-program" || literal.sourceFormat !== "safetensors" ||
    !sameNumericPolicy(literal.numericPolicy, gemma4CompositeLiteralNumericPolicy(literal.program))) {
    throw new Error("Programa literal Gemma 4 composite possui cabeçalho ou política numérica inválida.");
  }
  validateLiteralStorageBundle(literal);
  const constants = new Map<string, LiteralConstant>(literal.constants.map((constant) => [constant.name, constant]));
  const requiredInputs = literalInputs().map((input) => input.name);
  if (literal.inputs.length !== requiredInputs.length || requiredInputs.some((name) => !literal.inputs.some((input) => input.name === name))) {
    throw new Error("Programa literal Gemma 4 composite não declara todos os controles de entrada multimodal.");
  }
  if (literal.program.textProgram.source.path !== "embedded://gemma4-composite-literal" || literal.program.textProgram.source.format !== "safetensors") {
    throw new Error("Programa literal Gemma 4 composite reteve uma referência de source checkpoint.");
  }
  const compositeRequired = ["composite_block_sequence_ids", "composite_full_attention_mask", "composite_sliding_attention_mask", "composite_text_core"];
  if (compositeRequired.some((id) => !literal.assignments.composite.some((assignment) => assignment.id === id))) {
    throw new Error("Programa literal Gemma 4 composite omite uma transição de máscara ou cache obrigatória.");
  }
  if (literal.outputs.logits !== literal.program.outputs.logits || literal.assignments.composite.length !== literal.program.assignments.length ||
    literal.assignments.vision.length !== literal.program.visionProgram.assignments.length || literal.assignments.audio.length !== literal.program.audioProgram.assignments.length) {
    throw new Error("Programa literal Gemma 4 composite diverge das atribuições registradas.");
  }
  validateAssignmentScope(literal.assignments.composite, new Set(requiredInputs), constants, "composite");
  validateAssignmentScope(literal.assignments.vision, new Set(["pixel_values", "pixel_position_ids", "text_embeddings", "input_ids"]), constants, "vision");
  validateAssignmentScope(literal.assignments.audio, new Set(["input_features", "input_features_mask", "text_embeddings", "input_ids"]), constants, "audio");
  validateTextScope(literal.assignments.textLayers, literal.assignments.textEpilogue, constants);

  const registered = compositeReferences(literal.program);
  const expectedUnreachable = sharedKvUnreachableConstantsFromProgram(literal.program, registered);
  if (literal.unreachableConstants.length !== expectedUnreachable.length ||
    expectedUnreachable.some((expected) => !literal.unreachableConstants.some((actual) => actual.name === expected.name && actual.reason === expected.reason && actual.producerLayer === expected.producerLayer))) {
    throw new Error("Programa literal Gemma 4 composite não declara exatamente os tensores locais KV inatingíveis por compartilhamento.");
  }
  const expectedNames = new Set([...registered.keys(), ...expectedUnreachable.map((constant) => constant.name)]);
  const literalNames = new Set(literal.constants.map((constant) => constant.name));
  if (expectedNames.size !== literalNames.size || [...expectedNames].some((name) => !literalNames.has(name))) {
    throw new Error("Programa literal Gemma 4 composite contém constantes omitidas ou não alcançáveis pelas atribuições.");
  }
}

function literalInputs(): Gemma4CompositeLiteralInput[] {
  return [
    { name: "input_ids", description: "Token IDs, including declared modal placeholders.", required: true, dtype: "I32", shape: "[batch, sequence]" },
    { name: "position_ids", description: "Absolute text RoPE positions.", required: false, dtype: "I32", shape: "[batch, sequence]" },
    { name: "attention_mask", description: "Caller additive text attention bias; mutually exclusive with mm_token_type_ids.", required: false, dtype: "F32", shape: "[batch, 1|heads, query, key]" },
    { name: "past_key_values", description: "Post-RoPE text KV cache for incremental decode.", required: false, dtype: "F32", shape: "layer -> {key,value}" },
    { name: "pixel_values", description: "Patchified image pixels.", required: false, dtype: "F32", shape: "[batch, patches, 3*patch_size^2]" },
    { name: "image_position_ids", description: "Image patch [x,y] positions; required with pixel_values.", required: false, dtype: "I32", shape: "[batch][patch][x,y]" },
    { name: "pixel_values_videos", description: "Patchified video frames.", required: false, dtype: "F32", shape: "[videos, frames, patches, 3*patch_size^2]" },
    { name: "video_position_ids", description: "Video frame patch [x,y] positions; required with pixel_values_videos.", required: false, dtype: "I32", shape: "[videos][frames][patch][x,y]" },
    { name: "input_features", description: "Audio features before stride-2 subsampling.", required: false, dtype: "F32", shape: "[batch, frames, features]" },
    { name: "input_features_mask", description: "Boolean validity mask for input_features.", required: false, dtype: "BOOL", shape: "[batch, frames]" },
    { name: "mm_token_type_ids", description: "Image/video block IDs used only during uncached prefill.", required: false, dtype: "I32", shape: "[batch, sequence]" },
  ];
}

/**
 * Derives the artifact-wide numeric declaration from the ordered assignments.
 * This is intentionally stricter than an informational summary: a reader can
 * reject an artifact whose header hides an operation-level reduction boundary.
 */
export function gemma4CompositeLiteralNumericPolicy(program: Gemma4CompositeProgram): Gemma4CompositeLiteralCalculationProgram["numericPolicy"] {
  const textOperations = [...program.textProgram.prelude, ...program.textProgram.layers.flatMap((layer) => layer.operations), ...program.textProgram.epilogue];
  const hasBf16ResultCast = textOperations.some((operation) => operation.dtypePolicy.outputDtype === "BF16");
  const hasDeclaredReduction = textOperations.some((operation) => operation.dtypePolicy.accumulationDtype === "F64" ||
    (operation.dtypePolicy.reduction !== undefined && operation.dtypePolicy.reduction.kind !== "ordered-scalar"));
  const hasFmaBoundary = textOperations.some((operation) => operation.dtypePolicy.reduction?.kind === "ordered-fma" ||
    operation.dtypePolicy.reduction?.kind === "interleaved-fma-lanes" || operation.dtypePolicy.reduction?.kind === "tiled-fma-lanes" ||
    operation.dtypePolicy.reduction?.kind === "arm-neon-bf16-dot-fma" || operation.dtypePolicy.reduction?.kind === "arm-neon-bf16-bfdot-fma" ||
    (operation.dtypePolicy.reduction?.kind === "blocked-tiled-f32-lanes" && operation.dtypePolicy.reduction.productBoundary === "fused-fma") ||
    (operation.dtypePolicy.reduction?.kind === "blocked-f32-terms" && operation.dtypePolicy.reduction.productBoundary === "fused-fma"));
  if (hasDeclaredReduction) {
    return {
      inputDtype: "I32/F32/BOOL",
      computeDtype: "F32",
      accumulationDtype: "operation-declared",
      outputDtype: "operation-declared",
      scalarSemantics: hasFmaBoundary
        ? "IEEE-754 binary32; each operation declares ordered-scalar, contiguous blocked-term, blocked tiled-lane, separately-rounded F32-lane, or fused-multiply-add reduction and its F32 or BF16 result cast"
        : "IEEE-754 binary32 products; each operation declares its ordered-scalar, contiguous blocked-term, blocked tiled-lane, or interleaved-lane F32/F64 reduction and F32 or BF16 result cast",
    };
  }
  return {
    inputDtype: "I32/F32/BOOL",
    computeDtype: "F32",
    accumulationDtype: "F32",
    outputDtype: hasBf16ResultCast ? "operation-declared" : "F32",
    scalarSemantics: hasBf16ResultCast
      ? "IEEE-754 binary32 reductions; each operation declares its F32 or BF16 result cast"
      : "IEEE-754 binary32; host libm results rounded to F32",
  };
}

/** Validates that an artifact header faithfully describes its embedded graph. */
export function validateGemma4CompositeLiteralNumericPolicy(
  policy: Gemma4CompositeLiteralCalculationProgram["numericPolicy"],
  program: Gemma4CompositeProgram,
): void {
  validateGemma4TextReductionSchedules(program);
  if (!sameNumericPolicy(policy, gemma4CompositeLiteralNumericPolicy(program)) && !legacyNumericPolicyMatchesProgram(policy, program)) {
    throw new Error("Programa literal Gemma 4 composite possui política numérica incompatível com as atribuições declaradas.");
  }
}

/**
 * Artifacts written before blocked/tiled reduction declarations used a less
 * specific header string.  It is safe to retain read compatibility only when
 * their embedded graph actually stays within that former grammar; otherwise
 * accepting the header would conceal a material calculation boundary.
 */
function legacyNumericPolicyMatchesProgram(
  policy: Gemma4CompositeLiteralCalculationProgram["numericPolicy"],
  program: Gemma4CompositeProgram,
): boolean {
  if (policy.inputDtype !== "I32/F32/BOOL" || policy.computeDtype !== "F32" || policy.accumulationDtype !== "operation-declared" || policy.outputDtype !== "operation-declared") return false;
  const operations = [...program.textProgram.prelude, ...program.textProgram.layers.flatMap((layer) => layer.operations), ...program.textProgram.epilogue];
  const reductions = operations.filter((operation) => operation.op === "linear" || operation.op === "rms_norm").map((operation) => operation.dtypePolicy.reduction);
  if (policy.scalarSemantics === "IEEE-754 binary32 products; each operation declares its ordered-scalar or interleaved-lane F32/F64 reduction and F32 or BF16 result cast") {
    return reductions.every((reduction) => reduction?.kind === "ordered-scalar" || reduction?.kind === "interleaved-f32-lanes");
  }
  if (policy.scalarSemantics === "IEEE-754 binary32 products; each operation declares its ordered-scalar, contiguous blocked-term, or interleaved-lane F32/F64 reduction and F32 or BF16 result cast") {
    return reductions.every((reduction) => reduction?.kind === "ordered-scalar" || reduction?.kind === "interleaved-f32-lanes" || reduction?.kind === "tiled-f32-lanes" ||
      (reduction?.kind === "blocked-f32-terms" && reduction.productBoundary === "separately-rounded-f32"));
  }
  if (policy.scalarSemantics === "IEEE-754 binary32; each operation declares ordered-scalar, contiguous blocked-term, separately-rounded F32-lane, or fused-multiply-add reduction and its F32 or BF16 result cast") {
    return reductions.every((reduction) => reduction?.kind !== "blocked-tiled-f32-lanes" && reduction?.kind !== "arm-neon-bf16-dot-fma" && reduction?.kind !== "arm-neon-bf16-bfdot-fma");
  }
  return false;
}

/** Every literal linear/RMS reduction must expose its finite index schedule. */
function validateGemma4TextReductionSchedules(program: Gemma4CompositeProgram): void {
  const operations = [...program.textProgram.prelude, ...program.textProgram.layers.flatMap((layer) => layer.operations), ...program.textProgram.epilogue];
  for (const operation of operations) {
    if (operation.op !== "linear" && operation.op !== "rms_norm") continue;
    const reduction = operation.dtypePolicy.reduction;
    if (!reduction) throw new Error(`${operation.id}: programa literal Gemma 4 não declara a agenda de redução.`);
    if (reduction.kind === "ordered-scalar" || reduction.kind === "ordered-fma") {
      if (reduction.indexOrder !== "ascending" || (operation.dtypePolicy.accumulationDtype !== "F32" && operation.dtypePolicy.accumulationDtype !== "F64")) {
        throw new Error(`${operation.id}: agenda escalar de redução Gemma 4 é incompatível com sua política numérica.`);
      }
      if (reduction.kind === "ordered-fma" && operation.dtypePolicy.accumulationDtype !== "F32") throw new Error(`${operation.id}: agenda FMA escalar Gemma 4 requer acumulador F32.`);
      continue;
    }
    if (reduction.kind === "blocked-f32-terms") {
      if (operation.op !== "linear" || operation.dtypePolicy.accumulationDtype !== "F32" ||
        !Number.isSafeInteger(reduction.termsPerBlock) || reduction.termsPerBlock < 2 || reduction.inputBlock !== "contiguous-terms" || reduction.blockOrder !== "ascending" ||
        (reduction.termOrder !== "ascending" && reduction.termOrder !== "descending") ||
        (reduction.productBoundary !== "separately-rounded-f32" && reduction.productBoundary !== "fused-fma")) {
        throw new Error(`${operation.id}: agenda de blocos Gemma 4 inválida.`);
      }
      continue;
    }
    if (reduction.kind === "blocked-tiled-f32-lanes") {
      if (operation.op !== "linear" || operation.dtypePolicy.accumulationDtype !== "F32" ||
        !Number.isSafeInteger(reduction.laneCount) || reduction.laneCount < 2 || !Number.isSafeInteger(reduction.termsPerLane) || reduction.termsPerLane < 2 ||
        reduction.inputBlock !== "tile-contiguous-terms" || reduction.blockOrder !== "ascending" ||
        (reduction.laneReductionOrder !== "ascending" && reduction.laneReductionOrder !== "descending" && reduction.laneReductionOrder !== "balanced-pairwise") ||
        (reduction.productBoundary !== "separately-rounded-f32" && reduction.productBoundary !== "fused-fma")) {
        throw new Error(`${operation.id}: agenda de blocos tiled Gemma 4 inválida.`);
      }
      continue;
    }
    if (reduction.kind === "arm-neon-bf16-dot-fma") {
      if (operation.op !== "linear" || operation.dtypePolicy.accumulationDtype !== "F32" ||
        (reduction.laneCount !== 32 && reduction.laneCount !== 64) || reduction.registerCount !== 8 ||
        (reduction.lanesPerRegister !== 4 && reduction.lanesPerRegister !== 8) || reduction.laneCount !== reduction.registerCount * reduction.lanesPerRegister ||
        reduction.inputLane !== "index-modulo-vector-lane-count" ||
        (reduction.horizontalFold !== "ascending" && reduction.horizontalFold !== "pairwise")) {
        throw new Error(`${operation.id}: agenda ARM NEON BF16 dot Gemma 4 inválida.`);
      }
      continue;
    }
    if (reduction.kind === "arm-neon-bf16-bfdot-fma") {
      if (operation.op !== "linear" || operation.dtypePolicy.accumulationDtype !== "F32" || reduction.registerCount !== 8 ||
        reduction.activeRegisterCount !== 4 || reduction.lanesPerRegister !== 4 || reduction.termsPerLane !== 2 ||
        reduction.termsPerInstruction !== 8 || reduction.inputLane !== "contiguous-bf16-pairs" ||
        (reduction.horizontalFold !== "ascending" && reduction.horizontalFold !== "pairwise")) {
        throw new Error(`${operation.id}: agenda ARM NEON BFDOT BF16 Gemma 4 inválida.`);
      }
      continue;
    }
    if (operation.op !== "linear" || operation.dtypePolicy.accumulationDtype !== "F32" ||
      !Number.isSafeInteger(reduction.laneCount) || reduction.laneCount < 2 ||
      (reduction.laneReductionOrder !== "ascending" && reduction.laneReductionOrder !== "descending" && reduction.laneReductionOrder !== "balanced-pairwise")) {
      throw new Error(`${operation.id}: agenda de lanes Gemma 4 inválida.`);
    }
    const tiled = reduction.kind === "tiled-f32-lanes" || reduction.kind === "tiled-fma-lanes";
    if ((!tiled && reduction.inputLane !== "index-modulo-lane-count") ||
      (tiled && (!Number.isSafeInteger(reduction.termsPerLane) || reduction.termsPerLane < 2 || reduction.inputLane !== "tile-contiguous-terms"))) {
      throw new Error(`${operation.id}: mapeamento de lanes Gemma 4 inválido.`);
    }
  }
}

/**
 * The legacy in-memory composite executor is an explicitly scalar-F32 fixture
 * path. It must reject, rather than erase, a literal program's BF16 result
 * casts, F64 accumulators, or lane schedule. The paged text executor owns the
 * storage-backed mixed-policy replay boundary.
 */
function assertSynchronousCompositeF32Compatibility(program: Gemma4CompositeProgram): void {
  const operations = [...program.textProgram.prelude, ...program.textProgram.layers.flatMap((layer) => layer.operations), ...program.textProgram.epilogue];
  for (const operation of operations) {
    const policy = operation.dtypePolicy;
    if (policy.inputDtype === "BF16" || policy.computeDtype !== "F32" || policy.accumulationDtype !== "F32" || policy.outputDtype !== "F32" ||
      ((operation.op === "linear" || operation.op === "rms_norm") && policy.reduction?.kind !== "ordered-scalar")) {
      throw new Error(`${operation.id}: replay composto síncrono F32 não pode apagar a política numérica declarada; use o executor paginado compatível.`);
    }
  }
}

function sameNumericPolicy(
  actual: Gemma4CompositeLiteralCalculationProgram["numericPolicy"],
  expected: Gemma4CompositeLiteralCalculationProgram["numericPolicy"],
): boolean {
  return actual.inputDtype === expected.inputDtype && actual.computeDtype === expected.computeDtype &&
    actual.accumulationDtype === expected.accumulationDtype && actual.outputDtype === expected.outputDtype &&
    actual.scalarSemantics === expected.scalarSemantics;
}

interface StreamedDenseConstant {
  name: string;
  tensor: TensorInfo;
  expectedByteLength: number;
  metadata: Omit<LiteralConstant, "payloadBase64">;
}

interface PreparedStreamedDenseLiteral {
  constants: StreamedDenseConstant[];
  storageDecoders: LiteralStorageDecodeAssignment[];
  unreachableConstants: Gemma4CompositeUnreachableConstant[];
  embeddedProgram: Gemma4CompositeProgram;
  assignments: Gemma4CompositeLiteralCalculationProgram["assignments"];
  outputs: Gemma4CompositeLiteralCalculationProgram["outputs"];
}

/** Builds and validates all non-payload state before opening the output file. */
function prepareStreamedDenseLiteral(program: Gemma4CompositeProgram, catalog: ModelCatalog): PreparedStreamedDenseLiteral {
  if (catalog.format !== "safetensors" || program.sourceFormat !== "safetensors") {
    throw new Error("Escrita literal Gemma 4 requer Safetensors denso registrado.");
  }
  const references = compositeReferences(program);
  const unreachableConstants = sharedKvUnreachableConstants(program, catalog, references);
  const catalogNames = new Set(catalog.tensors.keys());
  const allReferences = new Map(references);
  for (const entry of unreachableConstants) {
    const tensor = catalog.tensors.get(entry.name);
    if (!tensor) throw new Error(`${entry.name}: tensor KV compartilhado ausente do catálogo.`);
    allReferences.set(entry.name, tensorReference(tensor));
  }
  if (allReferences.size !== catalogNames.size || [...catalogNames].some((name) => !allReferences.has(name))) {
    const omitted = [...catalogNames].filter((name) => !allReferences.has(name));
    throw new Error(`Programa literal Gemma 4 composite não possui atribuição semântica para todos os tensores do pacote${omitted.length ? `: ${omitted.slice(0, 4).join(", ")}` : "."}`);
  }
  const constants = [...allReferences.values()].map((reference) => streamedDenseConstant(reference, catalog.tensors.get(reference.name)));
  const embeddedProgram = embeddedCompositeProgram(program);
  const assignments: Gemma4CompositeLiteralCalculationProgram["assignments"] = {
    composite: structuredClone(embeddedProgram.assignments),
    vision: structuredClone(embeddedProgram.visionProgram.assignments),
    audio: structuredClone(embeddedProgram.audioProgram.assignments),
    textLayers: structuredClone(embeddedProgram.textProgram.layers),
    textEpilogue: structuredClone(embeddedProgram.textProgram.epilogue),
  };
  const outputs = structuredClone(embeddedProgram.outputs);
  const constantMap = new Map<string, LiteralConstant>(constants.map((constant) => [constant.name, { ...constant.metadata, payloadBase64: "" }]));
  validateGemma4CompositeLiteralStructure(embeddedProgram, assignments, outputs, constantMap, unreachableConstants);
  return {
    constants,
    storageDecoders: constants.map((constant) => denseStorageDecoder(constant.metadata)),
    unreachableConstants,
    embeddedProgram,
    assignments,
    outputs,
  };
}

function streamedDenseConstant(reference: TensorRef, tensor: TensorInfo | undefined): StreamedDenseConstant {
  if (!tensor || tensor.quantization || !sameShape(reference.shape, tensor.logicalShape) || reference.storageDtype !== tensor.storageDtype ||
    !sameShape(tensor.storageShape, tensor.logicalShape) ||
    (tensor.storageDtype !== "F32" && tensor.storageDtype !== "F16" && tensor.storageDtype !== "BF16")) {
    throw new Error(`${reference.name}: exportação literal Gemma 4 streaming requer tensor denso F32/F16/BF16 compatível.`);
  }
  const expectedByteLength = tensor.storageShape.reduce((total, dimension) => total * dimension, 1) * denseBytes(tensor.storageDtype);
  if (!Number.isSafeInteger(expectedByteLength) || expectedByteLength <= 0) throw new Error(`${reference.name}: shape denso inválido para exportação literal.`);
  return {
    name: tensor.name,
    tensor,
    expectedByteLength,
    metadata: {
      name: tensor.name,
      storageDtype: tensor.storageDtype,
      storageShape: [...tensor.storageShape],
      logicalShape: [...tensor.logicalShape],
      layout: "row-major",
      byteOrder: "little-endian",
      encoding: "base64",
    },
  };
}

function denseBytes(dtype: "F32" | "F16" | "BF16"): number {
  return dtype === "F32" ? 4 : 2;
}

function denseStorageDecoder(constant: Omit<LiteralConstant, "payloadBase64">): LiteralDenseStorageDecodeAssignment {
  const operation = constant.storageDtype === "F32" ? "ieee-f32-little-endian" : constant.storageDtype === "F16" ? "ieee-f16-to-f32" : "ieee-bf16-to-f32";
  return {
    id: `decode_${constant.name}`,
    operation,
    input: `${constant.name}:storage`,
    output: constant.name,
    storageDtype: constant.storageDtype as "F32" | "F16" | "BF16",
    outputDtype: "F32",
    byteOrder: "little-endian",
    semantics: "exact IEEE-754 storage decode; no arithmetic narrowing",
  };
}

/**
 * Validates the semantic graph against literal constant metadata without
 * requiring base64 payload strings. The streamed-artifact reader uses this
 * boundary after it has indexed byte ranges, keeping 20 GiB payloads outside
 * the JavaScript object graph while preserving the same fail-closed rules.
 */
export function validateGemma4CompositeLiteralStructure(
  program: Gemma4CompositeProgram,
  assignments: Gemma4CompositeLiteralCalculationProgram["assignments"],
  outputs: Gemma4CompositeLiteralCalculationProgram["outputs"],
  constants: ReadonlyMap<string, LiteralConstant>,
  unreachableConstants: readonly Gemma4CompositeUnreachableConstant[],
): void {
  const requiredInputs = literalInputs().map((input) => input.name);
  if (program.textProgram.source.path !== "embedded://gemma4-composite-literal" || program.textProgram.source.format !== "safetensors") {
    throw new Error("Programa literal Gemma 4 composite reteve uma referência de source checkpoint.");
  }
  const compositeRequired = ["composite_block_sequence_ids", "composite_full_attention_mask", "composite_sliding_attention_mask", "composite_text_core"];
  if (compositeRequired.some((id) => !assignments.composite.some((assignment) => assignment.id === id))) {
    throw new Error("Programa literal Gemma 4 composite omite uma transição de máscara ou cache obrigatória.");
  }
  if (outputs.logits !== program.outputs.logits || assignments.composite.length !== program.assignments.length ||
    assignments.vision.length !== program.visionProgram.assignments.length || assignments.audio.length !== program.audioProgram.assignments.length) {
    throw new Error("Programa literal Gemma 4 composite diverge das atribuições registradas.");
  }
  validateAssignmentScope(assignments.composite, new Set(requiredInputs), constants, "composite");
  validateAssignmentScope(assignments.vision, new Set(["pixel_values", "pixel_position_ids", "text_embeddings", "input_ids"]), constants, "vision");
  validateAssignmentScope(assignments.audio, new Set(["input_features", "input_features_mask", "text_embeddings", "input_ids"]), constants, "audio");
  validateTextScope(assignments.textLayers, assignments.textEpilogue, constants);
  const semanticReferences = compositeReferences(program);
  const expectedUnreachable = sharedKvUnreachableConstantsFromProgram(program, semanticReferences);
  if (unreachableConstants.length !== expectedUnreachable.length || expectedUnreachable.some((expected) => !unreachableConstants.some((actual) => actual.name === expected.name && actual.reason === expected.reason && actual.producerLayer === expected.producerLayer))) {
    throw new Error("Programa literal Gemma 4 composite não declara exatamente os tensores locais KV inatingíveis por compartilhamento.");
  }
}

function embeddedCompositeProgram(program: Gemma4CompositeProgram): Gemma4CompositeProgram {
  const embedded = structuredClone(program);
  embedded.textProgram.source = { path: "embedded://gemma4-composite-literal", format: "safetensors" };
  return embedded;
}

function compositeReferences(program: Gemma4CompositeProgram): Map<string, TensorRef> {
  const references = new Map<string, TensorRef>();
  const register = (reference: TensorRef): void => {
    const previous = references.get(reference.name);
    if (previous && (previous.storageDtype !== reference.storageDtype || !sameShape(previous.shape, reference.shape) || !sameQuantization(previous.quantization, reference.quantization))) {
      throw new Error(`Programa Gemma 4 composite possui referências incompatíveis para ${reference.name}.`);
    }
    references.set(reference.name, reference);
  };
  for (const assignment of [...program.assignments, ...program.visionProgram.assignments, ...program.audioProgram.assignments]) assignment.tensors?.forEach(register);
  for (const operation of [...program.textProgram.prelude, ...program.textProgram.layers.flatMap((layer) => layer.operations), ...program.textProgram.epilogue]) {
    if (operation.op === "embedding" || operation.op === "per_layer_embedding") register(operation.weight);
    if (operation.op === "rms_norm" && operation.weight) register(operation.weight);
    if (operation.op === "linear") { register(operation.weight); if (operation.bias) register(operation.bias); }
    if (operation.op === "tensor_scale") register(operation.scalar);
  }
  return references;
}

function sharedKvUnreachableConstants(
  program: Gemma4CompositeProgram,
  catalog: ModelCatalog,
  semanticReferences: ReadonlyMap<string, TensorRef>,
): Gemma4CompositeUnreachableConstant[] {
  const unreachable = sharedKvUnreachableConstantsFromProgram(program, semanticReferences);
  for (const entry of unreachable) {
    if (!catalog.tensors.has(entry.name)) throw new Error(`${entry.name}: consumidor KV compartilhado não possui tensor local no pacote.`);
  }
  return unreachable;
}

/**
 * Gemma 4 shared-KV consumers read their producer cache and do not execute
 * local K/V projections. The dense checkpoint nevertheless stores those
 * local K/V tensors, so the literal program embeds and labels them rather
 * than omitting bytes or pretending they participate in the calculation.
 */
function sharedKvUnreachableConstantsFromProgram(
  program: Gemma4CompositeProgram,
  semanticReferences: ReadonlyMap<string, TensorRef>,
): Gemma4CompositeUnreachableConstant[] {
  const embedding = program.textProgram.prelude.find((operation) => operation.op === "embedding");
  if (!embedding || embedding.op !== "embedding" || !embedding.weight.name.endsWith(".embed_tokens.weight")) {
    throw new Error("Programa Gemma 4 literal não possui prefixo textual verificável para KV compartilhado.");
  }
  const textPrefix = embedding.weight.name.slice(0, -".embed_tokens.weight".length);
  const unreachable: Gemma4CompositeUnreachableConstant[] = [];
  for (const layer of program.textProgram.layers) {
    const attention = layer.operations.find((operation) => operation.op === "scaled_dot_product_attention");
    if (!attention || attention.op !== "scaled_dot_product_attention" || !attention.kvSharing?.enabled) continue;
    if (attention.kvSharing.producerLayer === undefined || attention.kvSharing.producerLayer >= layer.index) {
      throw new Error(`Camada Gemma 4 ${layer.index}: produtor KV compartilhado inválido.`);
    }
    for (const suffix of ["k_norm.weight", "k_proj.weight", "v_proj.weight"] as const) {
      const name = `${textPrefix}.layers.${layer.index}.self_attn.${suffix}`;
      if (semanticReferences.has(name)) throw new Error(`${name}: consumidor KV compartilhado não pode usar K/V local e cache de produtor ao mesmo tempo.`);
      unreachable.push({ name, reason: "shared-kv-consumer-local-kv-is-runtime-unreachable", producerLayer: attention.kvSharing.producerLayer });
    }
  }
  return unreachable;
}

function tensorReference(tensor: TensorInfo): TensorRef {
  return {
    name: tensor.name,
    storageDtype: tensor.storageDtype,
    shape: [...tensor.logicalShape],
    ...(tensor.quantization ? { quantization: structuredClone(tensor.quantization) } : {}),
  };
}

function validateAssignmentScope(
  assignments: ReadonlyArray<{ id: string; inputs: string[]; output: string; tensors?: TensorRef[] }>,
  inputs: Set<string>, constants: ReadonlyMap<string, LiteralConstant>, scope: string,
): void {
  const available = new Set(inputs);
  for (const assignment of assignments) {
    for (const input of assignment.inputs) if (!available.has(input)) throw new Error(`${scope}:${assignment.id} lê predecessor não declarado ${input}.`);
    assignment.tensors?.forEach((reference) => validateLiteralStorageReference(reference, constants));
    if (available.has(assignment.output)) throw new Error(`${scope}:${assignment.id} redeclara saída ${assignment.output}.`);
    available.add(assignment.output);
  }
}

function validateTextScope(layers: Gemma4CompositeProgram["textProgram"]["layers"], epilogue: Operation[], constants: ReadonlyMap<string, LiteralConstant>): void {
  const available = new Set(["input_ids", "position_ids", "attention_mask", "past_key_values", "hidden_states_0", "ple_inputs"]);
  for (const operation of [...layers.flatMap((layer) => layer.operations), ...epilogue]) {
    textOperationInputs(operation).forEach((input) => {
      if (input.startsWith("attention_mask:")) { if (!available.has("attention_mask")) throw new Error(`${operation.id}: máscara literal não declarada.`); }
      else if (!available.has(input)) throw new Error(`${operation.id}: predecessor literal não declarado ${input}.`);
    });
    textOperationTensors(operation).forEach((reference) => validateLiteralStorageReference(reference, constants));
    if (available.has(operation.output)) throw new Error(`${operation.id}: saída textual literal duplicada.`);
    available.add(operation.output);
  }
}

function textOperationInputs(operation: Operation): string[] {
  switch (operation.op) {
    case "embedding": case "per_layer_embedding": return [operation.tokenInput];
    case "rms_norm": case "linear": case "reshape_heads": case "reshape_per_layer": case "select_per_layer": case "activation": case "tensor_scale": return [operation.input];
    case "rotary_embedding": return [operation.input, operation.positionInput];
    case "scaled_dot_product_attention": return [operation.query, operation.key, operation.value, operation.maskInput];
    case "elementwise": return operation.inputs;
  }
}

function textOperationTensors(operation: Operation): TensorRef[] {
  switch (operation.op) {
    case "embedding": case "per_layer_embedding": return [operation.weight];
    case "rms_norm": return operation.weight ? [operation.weight] : [];
    case "linear": return operation.bias ? [operation.weight, operation.bias] : [operation.weight];
    case "tensor_scale": return [operation.scalar];
    default: return [];
  }
}

function sameShape(left: readonly number[], right: readonly number[]): boolean { return left.length === right.length && left.every((value, index) => value === right[index]); }
function sameQuantization(left: TensorRef["quantization"], right: TensorRef["quantization"]): boolean {
  if (!left || !right) return left === right;
  return left.family === right.family && left.mode === right.mode && left.bits === right.bits && left.groupSize === right.groupSize && left.tensorType === right.tensorType && left.scaleTensor === right.scaleTensor && left.biasTensor === right.biasTensor && left.globalScaleTensor === right.globalScaleTensor;
}
