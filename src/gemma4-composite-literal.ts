import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, rename, unlink } from "node:fs/promises";
import { once } from "node:events";
import * as path from "node:path";
import { isDeepStrictEqual } from "node:util";
import {
  buildLiteralDenseDecoderLanguageContract,
  buildLiteralDenseStorageDecodeAssignment,
  buildLiteralStorageBundle,
  decodeLiteralStorageBundleF32,
  validateLiteralStorageBundle,
  validateLiteralDenseDecoderLanguageContract,
  validateLiteralStorageReference,
  type LiteralConstant,
  type LiteralDenseStorageDecodeAssignment,
  type LiteralDenseDecoderLanguageContract,
  type LiteralStorageBundle,
  type LiteralTensorReader,
} from "./literal.js";
import {
  executeGemma4CompositeF32,
  type Gemma4CompositeExecutionRequest,
  type Gemma4CompositeExecutionResult,
  type Gemma4CompositeGenerationRequest,
  type Gemma4CompositeGenerationResult,
  type Gemma4CompositeProgram,
} from "./gemma4-composite.js";
import {
  buildGemma4LiteralGenerationControlProgram,
  executeGemma4LiteralGenerationControlProgram,
  type Gemma4LiteralGenerationControlProgram,
} from "./gemma4-literal-generation-control.js";
import {
  buildGemma4LiteralForwardControlProgram,
  executeGemma4LiteralForwardControlProgram,
  gemma4CompositeRequestInputPresence,
  validateGemma4LiteralForwardControlProgram,
  type Gemma4LiteralForwardControlProgram,
} from "./gemma4-literal-forward-control.js";
import {
  buildGemma4LiteralInputContract,
  executeGemma4LiteralInputContract,
  validateGemma4LiteralInputDeclarationAlignment,
  validateGemma4LiteralInputContract,
  type Gemma4LiteralInputContract,
} from "./gemma4-literal-input-contract.js";
import {
  buildGemma4LiteralOutputContract,
  executeGemma4LiteralForwardOutputContract,
  executeGemma4LiteralGenerationOutputContract,
  validateGemma4LiteralOutputContract,
  type Gemma4LiteralOutputContract,
} from "./gemma4-literal-output-contract.js";
import type { Gemma4AudioAssignment } from "./gemma4-audio.js";
import { GEMMA4_E4B_PYTORCH_BF16_TANH_IMPLEMENTATION, GEMMA4_E4B_PYTORCH_BF16_TRIG_IMPLEMENTATION } from "./gemma4-text.js";
import type { Gemma4VisionAssignment } from "./gemma4-vision.js";
import {
  buildGemma4LiteralCalculationDomains,
  validateGemma4LiteralCalculationDomains,
  type Gemma4LiteralCalculationDomains,
} from "./gemma4-literal-domains.js";
import {
  buildGemma4LiteralLearnedOperandBindings,
  validateGemma4LiteralLearnedOperandBindings,
  type Gemma4LiteralLearnedOperandBindings,
} from "./gemma4-literal-learned-operands.js";
import {
  buildGemma4LiteralScalarCalculations,
  validateGemma4LiteralScalarCalculations,
  type Gemma4LiteralScalarCalculations,
} from "./gemma4-literal-scalar-calculations.js";
import {
  buildGemma4LiteralCalculationGraph,
  validateGemma4LiteralCalculationGraph,
  type Gemma4LiteralCalculationGraph,
} from "./gemma4-literal-calculation-graph.js";
import {
  buildGemma4LiteralNumericLiterals,
  validateGemma4LiteralNumericLiterals,
  type Gemma4LiteralNumericLiterals,
} from "./gemma4-literal-numeric-literals.js";
import {
  buildGemma4LiteralFormulaLanguageContract,
  validateGemma4LiteralFormulaLanguageContract,
  type Gemma4LiteralFormulaLanguageContract,
} from "./gemma4-literal-formula-language.js";
import {
  buildGemma4LiteralGenerationForwardCalculationContract,
  buildGemma4LiteralGenerationScalarCalculations,
  type Gemma4LiteralGenerationOperation,
  type Gemma4LiteralGenerationForwardCalculationContract,
  type Gemma4LiteralGenerationScalarCalculations,
} from "./gemma4-literal-generation-calculations.js";
import type { ModelCatalog, Operation, TensorInfo, TensorRef } from "./types.js";
import {
  validateGemma4LiteralSourceIdentity,
  validateGemma4LiteralSourceWeightMappings,
  type Gemma4LiteralSourceIdentity,
} from "./gemma4-literal-source-identity.js";
import {
  gemma4AuthoritativeExecutionContract,
  loadGemma4RuntimeReductionAdapterProgram,
  validateGemma4AuthoritativeExecutionContract,
  type Gemma4AuthoritativeExecutionContract,
} from "./gemma4-authoritative-runtime.js";
import {
  buildGemma4LiteralTranscendentalPrograms,
  validateGemma4LiteralTranscendentalCoverage,
  type Gemma4LiteralTranscendentalPrograms,
} from "./gemma4-literal-transcendental-programs.js";
import {
  buildGemma4LiteralArtifactIntegrityManifest,
  validateGemma4LiteralArtifactIntegrityManifest,
  type Gemma4LiteralArtifactIntegrityManifest,
  type Gemma4LiteralIntegritySections,
} from "./gemma4-literal-artifact-integrity.js";
import {
  buildGemma4LiteralFidelityGate,
  validateGemma4LiteralFidelityGate,
  type Gemma4LiteralFidelityGate,
} from "./gemma4-literal-fidelity-gate.js";
import {
  buildGemma4LiteralPayloadIntegrity,
  GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES,
  payloadIntegrityEntry,
  type Gemma4CompositeLiteralPayloadIntegrityEntry,
  type Gemma4LiteralPayloadIntegrityChunk,
} from "./gemma4-literal-payload-integrity.js";

export type { Gemma4CompositeLiteralPayloadIntegrityEntry } from "./gemma4-literal-payload-integrity.js";

export interface Gemma4CompositeLiteralInput {
  name: string;
  description: string;
  required: boolean;
  dtype: string;
  shape: string;
  usedBy: Array<"forward" | "generation">;
  requiredFor: Array<"forward" | "generation">;
}

export interface Gemma4LiteralGenerationAssignment {
  id: string;
  operation: Gemma4LiteralGenerationOperation;
  inputs: string[];
  output: string;
  dtype: string;
  shape: string;
  iteration?: "step = 0..max_new_tokens-1 while stop_after_step[step-1] is false";
  semantics: string;
}

/**
 * The generation state machine is data in the artifact, not behavior supplied
 * by this TypeScript module. Both forward invocations refer to the complete
 * declared composite assignment program and therefore cannot substitute an
 * opaque generic decoder for the serialized Gemma 4 calculation.
 */
export interface Gemma4LiteralGreedyGenerationProgram {
  kind: "gemma4-literal-greedy-generation-program";
  forwardProgram: {
    reference: "program";
    expansionOrder: "composite assignments in array order; vision/audio definitions inline at invocation; prepared text layers then epilogue";
    firstAssignment: "composite_block_sequence_ids";
    lastAssignment: "final_logit_softcap" | "lm_head";
  };
  /** Exact instantiated forward order and explicit producer/reuse KV transitions. */
  forwardCalculation: Gemma4LiteralGenerationForwardCalculationContract;
  loop: {
    iterator: "step";
    startInclusive: 0;
    endExclusiveInput: "max_new_tokens";
    earlyStop: "after incremental forward and cache capture when selected_token[step] == eos_token_id";
  };
  assignments: Gemma4LiteralGenerationAssignment[];
  /** Indexed generation-control formulas stored in the artifact, not rebuilt by its reader. */
  scalarCalculations: Gemma4LiteralGenerationScalarCalculations;
  /** Normative structured control flow executed by source-removed replay. */
  controlProgram: Gemma4LiteralGenerationControlProgram;
  outputs: {
    prefillState: "forward_state[0]";
    generatedTokenIds: "generated_token_ids";
    selectionLogits: "selection_logits";
    stepForwardLogits: "step_forward_logits";
    stepPastKeyValues: "step_past_key_values";
    terminalLogits: "terminal_logits";
    terminalPastKeyValues: "terminal_past_key_values";
  };
}

export const GEMMA4_COMPOSITE_LITERAL_SCHEMA_VERSION = 60 as const;

/**
 * A source-independent literal program for the complete registered Gemma 4
 * composite boundary.  The three assignment scopes are kept separate because
 * image and video reuse the vision feature program, while their outer scatter
 * steps remain distinct, named dependencies in the enclosing program.
 */
export interface Gemma4CompositeLiteralCalculationProgram extends LiteralStorageBundle {
  schemaVersion: typeof GEMMA4_COMPOSITE_LITERAL_SCHEMA_VERSION;
  kind: "gemma4-composite-literal-calculation-program";
  sourceFormat: "safetensors";
  /** Gemma 4 checkpoint is dense; packed/quantized decoder variants are forbidden here. */
  storageDecoders: LiteralDenseStorageDecodeAssignment[];
  /** Executable operator semantics shared by every address and IEEE bit-expression program. */
  denseDecoderLanguage: LiteralDenseDecoderLanguageContract;
  /** Immutable package identity and exact metadata bytes used to derive semantics. */
  sourceIdentity: Gemma4LiteralSourceIdentity;
  /** Exact runtime context used as numeric authority, including its unresolved native reduction boundary. */
  authoritativeExecution: Gemma4AuthoritativeExecutionContract;
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
  /** Exact dtype, row-major layout and symbolic coordinate bounds for every declared calculation output. */
  calculationDomains: Gemma4LiteralCalculationDomains;
  /** Explicit learned roles and logical tensor index expressions for every checkpoint-backed assignment. */
  learnedOperands: Gemma4LiteralLearnedOperandBindings;
  /** Indexed formulas, casts and complete reduction domains for every assignment definition. */
  scalarCalculations: Gemma4LiteralScalarCalculations;
  /** Normative, embedded interpretation of the indexed formula notation. */
  formulaLanguage: Gemma4LiteralFormulaLanguageContract;
  /** Complete scalar programs for every SLEEF intrinsic named by the formulas. */
  transcendentalPrograms: Gemma4LiteralTranscendentalPrograms;
  /** Exact F64/F32/BF16 bit patterns for every numeric token used by forward or generation formulas. */
  numericLiterals: Gemma4LiteralNumericLiterals;
  /** Fully instantiated, dependency-ordered forward graph with all subprogram call-site bindings. */
  calculationGraph: Gemma4LiteralCalculationGraph;
  /** Integrity-bound, graph-derived prohibition against overclaiming unresolved native reductions. */
  fidelityGate: Gemma4LiteralFidelityGate;
  /** Normative optional-input routing and identity aliases for the composite forward. */
  forwardControl: Gemma4LiteralForwardControlProgram;
  /** Executable ranks, shapes, value domains, cache ownership and modal cardinality for every input. */
  inputContract: Gemma4LiteralInputContract;
  /** Executable public forward/generation tensor, cache and terminal-state invariants. */
  outputContract: Gemma4LiteralOutputContract;
  outputs: { embeddings: "hidden_states_0"; perLayerInputs: "ple_inputs"; logits: "softcapped_logits" | "logits" };
  generation: Gemma4LiteralGreedyGenerationProgram;
  /** Mandatory per-tensor commitments over every embedded learned payload. */
  payloadIntegrity: Gemma4CompositeLiteralPayloadIntegrityEntry[];
  /** Mandatory commitment over metadata, executable semantics and the payload commitment table. */
  integrityManifest: Gemma4LiteralArtifactIntegrityManifest;
}

export interface Gemma4CompositeUnreachableConstant {
  name: string;
  reason: "shared-kv-consumer-local-kv-is-runtime-unreachable";
  producerLayer: number;
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
  sourceIdentity: Gemma4LiteralSourceIdentity,
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
  const storageDecoders = storage.storageDecoders.map((decoder): LiteralDenseStorageDecodeAssignment => {
    if (decoder.operation === "mlx-affine-u32-to-f32" || decoder.operation === "ggml-q8-0-to-f32") {
      throw new Error(`${decoder.id}: Gemma 4 dense não aceita decoder quantizado.`);
    }
    return decoder;
  });
  const embeddedProgram = embeddedCompositeProgram(program);
  const calculationGraph = buildGemma4LiteralCalculationGraph(embeddedProgram);
  const authoritativeExecution = gemma4AuthoritativeExecutionContract(await loadGemma4RuntimeReductionAdapterProgram());
  const fidelityGate = buildGemma4LiteralFidelityGate(calculationGraph, authoritativeExecution);
  const scalarCalculations = buildGemma4LiteralScalarCalculations(embeddedProgram);
  const generation = gemma4LiteralGreedyGenerationProgram(embeddedProgram, calculationGraph);
  const forwardControl = buildGemma4LiteralForwardControlProgram(embeddedProgram);
  const inputContract = buildGemma4LiteralInputContract(embeddedProgram);
  const outputContract = buildGemma4LiteralOutputContract(embeddedProgram);
  const payloadIntegrity = storage.constants.map((constant) => literalPayloadIntegrity(constant));
  const base: Omit<Gemma4CompositeLiteralCalculationProgram, "integrityManifest"> = {
    schemaVersion: GEMMA4_COMPOSITE_LITERAL_SCHEMA_VERSION,
    kind: "gemma4-composite-literal-calculation-program",
    sourceFormat: "safetensors",
    sourceIdentity: structuredClone(sourceIdentity),
    authoritativeExecution,
    numericPolicy: gemma4CompositeLiteralNumericPolicy(program),
    inputs: literalInputs(),
    constants: storage.constants,
    storageDecoders,
    denseDecoderLanguage: buildLiteralDenseDecoderLanguageContract(),
    unreachableConstants,
    program: embeddedProgram,
    assignments: {
      composite: structuredClone(embeddedProgram.assignments),
      vision: structuredClone(embeddedProgram.visionProgram.assignments),
      audio: structuredClone(embeddedProgram.audioProgram.assignments),
      textLayers: structuredClone(embeddedProgram.textProgram.layers),
      textEpilogue: structuredClone(embeddedProgram.textProgram.epilogue),
    },
    calculationDomains: buildGemma4LiteralCalculationDomains(embeddedProgram),
    learnedOperands: buildGemma4LiteralLearnedOperandBindings(embeddedProgram),
    scalarCalculations,
    formulaLanguage: buildGemma4LiteralFormulaLanguageContract(),
    transcendentalPrograms: buildGemma4LiteralTranscendentalPrograms(),
    numericLiterals: buildGemma4LiteralNumericLiterals(scalarCalculations, generation.scalarCalculations, generation.forwardCalculation),
    calculationGraph,
    fidelityGate,
    forwardControl,
    inputContract,
    outputContract,
    outputs: structuredClone(embeddedProgram.outputs),
    generation,
    payloadIntegrity,
  };
  const literal: Gemma4CompositeLiteralCalculationProgram = {
    ...base,
    integrityManifest: buildGemma4LiteralArtifactIntegrityManifest(gemma4LiteralIntegritySections(base)),
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
  sourceIdentity: Gemma4LiteralSourceIdentity,
): Promise<Gemma4CompositeLiteralWriteResult> {
  validateGemma4TextReductionSchedules(program);
  validateGemma4LiteralSourceIdentity(sourceIdentity);
  const authoritativeExecution = gemma4AuthoritativeExecutionContract(await loadGemma4RuntimeReductionAdapterProgram());
  const prepared = prepareStreamedDenseLiteral(program, catalog, authoritativeExecution);
  validateGemma4LiteralSourceWeightMappings(sourceIdentity, prepared.constants.map((constant) => ({
    name: constant.name,
    storageDtype: constant.metadata.storageDtype,
    storageShape: constant.metadata.storageShape,
    payloadBytes: constant.expectedByteLength,
  })));
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
    await write(`{"schemaVersion":${GEMMA4_COMPOSITE_LITERAL_SCHEMA_VERSION},"kind":"gemma4-composite-literal-calculation-program","sourceFormat":"safetensors","sourceIdentity":${JSON.stringify(sourceIdentity)},"authoritativeExecution":${JSON.stringify(authoritativeExecution)},"numericPolicy":${JSON.stringify(gemma4CompositeLiteralNumericPolicy(program))},"inputs":${JSON.stringify(literalInputs())},"constants":[`);
    for (let index = 0; index < prepared.constants.length; index += 1) {
      const constant = prepared.constants[index]!;
      if (index > 0) await write(",");
      await write(`${JSON.stringify(constant.metadata).slice(0, -1)},"payloadBase64":"`);
      const payload = await writeBase64Payload(constant, reader, write);
      embeddedPayloadBytes += payload.bytes;
      payloadIntegrity.push(payload.integrity);
      await write("\"}");
    }
    const integritySource = {
      sourceIdentity,
      authoritativeExecution,
      numericPolicy: gemma4CompositeLiteralNumericPolicy(program),
      inputs: literalInputs(),
      constants: prepared.constants.map((constant) => constant.metadata),
      unreachableConstants: prepared.unreachableConstants,
      storageDecoders: prepared.storageDecoders,
      denseDecoderLanguage: prepared.denseDecoderLanguage,
      program: prepared.embeddedProgram,
      assignments: prepared.assignments,
      calculationDomains: prepared.calculationDomains,
      learnedOperands: prepared.learnedOperands,
      scalarCalculations: prepared.scalarCalculations,
      formulaLanguage: prepared.formulaLanguage,
      transcendentalPrograms: prepared.transcendentalPrograms,
      numericLiterals: prepared.numericLiterals,
      calculationGraph: prepared.calculationGraph,
      fidelityGate: prepared.fidelityGate,
      forwardControl: prepared.forwardControl,
      inputContract: prepared.inputContract,
      outputContract: prepared.outputContract,
      outputs: prepared.outputs,
      generation: prepared.generation,
      payloadIntegrity,
    };
    const integrityManifest = buildGemma4LiteralArtifactIntegrityManifest(gemma4LiteralIntegritySections(integritySource));
    await write(`],"unreachableConstants":${JSON.stringify(prepared.unreachableConstants)},"storageDecoders":${JSON.stringify(prepared.storageDecoders)},"denseDecoderLanguage":${JSON.stringify(prepared.denseDecoderLanguage)},"program":${JSON.stringify(prepared.embeddedProgram)},"assignments":${JSON.stringify(prepared.assignments)},"calculationDomains":${JSON.stringify(prepared.calculationDomains)},"learnedOperands":${JSON.stringify(prepared.learnedOperands)},"scalarCalculations":${JSON.stringify(prepared.scalarCalculations)},"formulaLanguage":${JSON.stringify(prepared.formulaLanguage)},"transcendentalPrograms":${JSON.stringify(prepared.transcendentalPrograms)},"numericLiterals":${JSON.stringify(prepared.numericLiterals)},"calculationGraph":${JSON.stringify(prepared.calculationGraph)},"fidelityGate":${JSON.stringify(prepared.fidelityGate)},"forwardControl":${JSON.stringify(prepared.forwardControl)},"inputContract":${JSON.stringify(prepared.inputContract)},"outputContract":${JSON.stringify(prepared.outputContract)},"outputs":${JSON.stringify(prepared.outputs)},"generation":${JSON.stringify(prepared.generation)},"payloadIntegrity":${JSON.stringify(payloadIntegrity)},"integrityManifest":${JSON.stringify(integrityManifest)}}\n`);
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
): Promise<{ bytes: number; integrity: Gemma4CompositeLiteralPayloadIntegrityEntry }> {
  const digest = createHash("sha256");
  const chunks: Gemma4LiteralPayloadIntegrityChunk[] = [];
  if (reader.readTensorBytesRange) {
    let offset = 0;
    while (offset < constant.expectedByteLength) {
      const byteLength = Math.min(GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES, constant.expectedByteLength - offset);
      const bytes = await reader.readTensorBytesRange(constant.tensor, offset, byteLength);
      if (bytes.length !== byteLength) throw new Error(`${constant.name}: leitor literal retornou ${bytes.length} bytes no range ${offset}, esperados ${byteLength}.`);
      // Base64 alphabet is JSON-safe; the quote delimiters are emitted by the
      // caller. Every non-final chunk is 3-byte aligned by the constant above.
      await write(bytes.toString("base64"));
      digest.update(bytes);
      chunks.push({
        ordinal: chunks.length,
        byteOffset: offset,
        byteLength,
        sha256: createHash("sha256").update(bytes).digest("hex"),
      });
      offset += byteLength;
    }
    return { bytes: offset, integrity: payloadIntegrityEntry(constant.name, offset, digest.digest("hex"), chunks) };
  }
  if (constant.expectedByteLength > GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES) {
    throw new Error(`${constant.name}: exportação literal de tensor grande requer readTensorBytesRange; o leitor não pode formar um base64 multi-GiB inteiro.`);
  }
  const bytes = await reader.readTensorBytes(constant.tensor);
  if (bytes.length !== constant.expectedByteLength) throw new Error(`${constant.name}: leitor literal retornou ${bytes.length} bytes, esperados ${constant.expectedByteLength}.`);
  await write(bytes.toString("base64"));
  digest.update(bytes);
  chunks.push({ ordinal: 0, byteOffset: 0, byteLength: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") });
  return { bytes: bytes.length, integrity: payloadIntegrityEntry(constant.name, bytes.length, digest.digest("hex"), chunks) };
}

/** Replays prefill solely from the literal's embedded bytes and assignments. */
export function executeGemma4CompositeLiteralF32(
  literal: Gemma4CompositeLiteralCalculationProgram,
  request: Omit<Gemma4CompositeExecutionRequest, "tensors">,
): Gemma4CompositeExecutionResult {
  validateGemma4CompositeLiteralCalculationProgram(literal);
  assertSynchronousCompositeF32Compatibility(literal.program);
  executeGemma4LiteralInputContract(literal.inputContract, literal.program, request);
  const selection = executeGemma4LiteralForwardControlProgram(literal.forwardControl, literal.program, gemma4CompositeRequestInputPresence(request));
  const result = executeGemma4CompositeF32(literal.program, { ...request, tensors: decodeLiteralStorageBundleF32(literal) }, selection);
  executeGemma4LiteralForwardOutputContract(literal.outputContract, literal.program, request, result);
  return result;
}

/** Replays greedy cached decoding solely from the literal's embedded bytes. */
export function generateGemma4CompositeLiteralF32(
  literal: Gemma4CompositeLiteralCalculationProgram,
  request: Omit<Gemma4CompositeGenerationRequest, "tensors">,
): Gemma4CompositeGenerationResult {
  validateGemma4CompositeLiteralCalculationProgram(literal);
  assertSynchronousCompositeF32Compatibility(literal.program);
  const tensors = decodeLiteralStorageBundleF32(literal);
  const result = executeGemma4LiteralGenerationControlProgram(
    literal.generation.controlProgram,
    literal.program,
    request,
    (forwardRequest) => {
      executeGemma4LiteralInputContract(literal.inputContract, literal.program, forwardRequest);
      const selection = executeGemma4LiteralForwardControlProgram(literal.forwardControl, literal.program, gemma4CompositeRequestInputPresence(forwardRequest));
      const forwardResult = executeGemma4CompositeF32(literal.program, { ...forwardRequest, tensors }, selection);
      executeGemma4LiteralForwardOutputContract(literal.outputContract, literal.program, forwardRequest, forwardResult);
      return forwardResult;
    },
  );
  executeGemma4LiteralGenerationOutputContract(literal.outputContract, literal.program, request, {
    prefill: result.prefill.text,
    generatedTokenIds: result.generatedTokenIds,
    selectionLogits: result.selectionLogits,
    stepForwardLogits: result.stepForwardLogits,
    stepPastKeyValues: result.stepPastKeyValues,
    terminal: result.text,
  });
  return result;
}

/**
 * Validates all three semantic scopes and their storage references before
 * replay.  This rejects a partial tower, an omitted vision-mask/cache step,
 * or a program which tries to retain a source-model path.
 */
export function validateGemma4CompositeLiteralCalculationProgram(literal: Gemma4CompositeLiteralCalculationProgram): void {
  validateGemma4TextReductionSchedules(literal.program);
  if (literal.schemaVersion !== GEMMA4_COMPOSITE_LITERAL_SCHEMA_VERSION || literal.kind !== "gemma4-composite-literal-calculation-program" || literal.sourceFormat !== "safetensors" ||
    !sameNumericPolicy(literal.numericPolicy, gemma4CompositeLiteralNumericPolicy(literal.program))) {
    throw new Error("Programa literal Gemma 4 composite possui cabeçalho ou política numérica inválida.");
  }
  validateGemma4LiteralSourceIdentity(literal.sourceIdentity);
  validateGemma4AuthoritativeExecutionContract(literal.authoritativeExecution);
  if (literal.program.audioProgram.attentionMaskContract !== "transformers-eager-additive-mask-logical-not-v1") {
    throw new Error("Programa literal Gemma 4 não declara a conversão eager autoritativa da máscara de áudio.");
  }
  validateLiteralDenseDecoderLanguageContract(literal.denseDecoderLanguage);
  validateLiteralStorageBundle(literal);
  validateLiteralPayloadIntegrity(literal.constants, literal.payloadIntegrity);
  const constants = new Map<string, LiteralConstant>(literal.constants.map((constant) => [constant.name, constant]));
  validateGemma4LiteralSourceWeightMappings(literal.sourceIdentity, literal.constants.map((constant) => ({
    name: constant.name,
    storageDtype: constant.storageDtype,
    storageShape: constant.storageShape,
    payloadBytes: Buffer.from(constant.payloadBase64, "base64").length,
  })));
  const expectedInputs = literalInputs();
  const requiredInputs = expectedInputs.map((input) => input.name);
  validateGemma4CompositeLiteralInputs(literal.inputs);
  validateGemma4LiteralGenerationProgram(literal.generation, literal.program);
  validateGemma4LiteralCalculationDomains(literal.calculationDomains, literal.program);
  validateGemma4LiteralLearnedOperandBindings(literal.learnedOperands, literal.program);
  validateGemma4LiteralScalarCalculations(literal.scalarCalculations, literal.program);
  validateGemma4LiteralFormulaLanguageContract(literal.formulaLanguage, literal.scalarCalculations, literal.generation.scalarCalculations, literal.generation.forwardCalculation);
  validateGemma4LiteralTranscendentalCoverage(literal.transcendentalPrograms, [
    ...literal.scalarCalculations.assignments.flatMap((assignment) => assignment.scalarAssignments),
    ...literal.generation.scalarCalculations.assignments.flatMap((assignment) => assignment.scalarAssignments),
  ]);
  validateGemma4LiteralNumericLiterals(literal.numericLiterals, literal.scalarCalculations, literal.generation.scalarCalculations, literal.generation.forwardCalculation);
  validateGemma4LiteralCalculationGraph(literal.calculationGraph, literal.program);
  validateGemma4LiteralFidelityGate(literal.fidelityGate, literal.calculationGraph, literal.authoritativeExecution);
  validateGemma4LiteralForwardControlProgram(literal.forwardControl, literal.program);
  validateGemma4LiteralInputContract(literal.inputContract, literal.program);
  validateGemma4LiteralInputDeclarationAlignment(literal.inputContract, literal.inputs);
  validateGemma4LiteralOutputContract(literal.outputContract, literal.program);
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
  validateGemma4LiteralArtifactIntegrityManifest(literal.integrityManifest, gemma4LiteralIntegritySections(literal));
}

export function gemma4LiteralIntegritySections(source: {
  sourceIdentity: unknown;
  authoritativeExecution: unknown;
  numericPolicy: unknown;
  inputs: unknown;
  constants: ReadonlyArray<Omit<LiteralConstant, "payloadBase64"> | LiteralConstant>;
  unreachableConstants: unknown;
  storageDecoders: unknown;
  denseDecoderLanguage: unknown;
  program: unknown;
  assignments: unknown;
  calculationDomains: unknown;
  learnedOperands: unknown;
  scalarCalculations: unknown;
  formulaLanguage: unknown;
  transcendentalPrograms: unknown;
  numericLiterals: unknown;
  calculationGraph: unknown;
  fidelityGate: unknown;
  forwardControl: unknown;
  inputContract: unknown;
  outputContract: unknown;
  outputs: unknown;
  generation: unknown;
  payloadIntegrity: unknown;
}): Gemma4LiteralIntegritySections {
  return {
    sourceIdentity: source.sourceIdentity,
    authoritativeExecution: source.authoritativeExecution,
    numericPolicy: source.numericPolicy,
    inputs: source.inputs,
    constantMetadata: source.constants.map((constant) => {
      const { payloadBase64: _payloadBase64, ...metadata } = constant as LiteralConstant;
      return metadata;
    }),
    unreachableConstants: source.unreachableConstants,
    storageDecoders: source.storageDecoders,
    denseDecoderLanguage: source.denseDecoderLanguage,
    program: source.program,
    assignments: source.assignments,
    calculationDomains: source.calculationDomains,
    learnedOperands: source.learnedOperands,
    scalarCalculations: source.scalarCalculations,
    formulaLanguage: source.formulaLanguage,
    transcendentalPrograms: source.transcendentalPrograms,
    numericLiterals: source.numericLiterals,
    calculationGraph: source.calculationGraph,
    fidelityGate: source.fidelityGate,
    forwardControl: source.forwardControl,
    inputContract: source.inputContract,
    outputContract: source.outputContract,
    outputs: source.outputs,
    generation: source.generation,
    payloadIntegrity: source.payloadIntegrity,
  };
}

function literalPayloadIntegrity(constant: LiteralConstant): Gemma4CompositeLiteralPayloadIntegrityEntry {
  const bytes = Buffer.from(constant.payloadBase64, "base64");
  return buildGemma4LiteralPayloadIntegrity(constant.name, bytes);
}

function validateLiteralPayloadIntegrity(
  constants: readonly LiteralConstant[],
  entries: readonly Gemma4CompositeLiteralPayloadIntegrityEntry[],
): void {
  const expected = constants.map((constant) => literalPayloadIntegrity(constant));
  if (!isDeepStrictEqual(entries, expected)) {
    throw new Error("Programa literal Gemma 4 possui compromissos de payload incompletos ou divergentes.");
  }
}

function literalInputs(): Gemma4CompositeLiteralInput[] {
  return [
    { name: "input_ids", description: "Token IDs, including declared modal placeholders.", required: true, dtype: "I32", shape: "[batch, sequence]", usedBy: ["forward", "generation"], requiredFor: ["forward", "generation"] },
    { name: "position_ids", description: "Absolute text RoPE positions.", required: false, dtype: "I32", shape: "[batch, sequence]", usedBy: ["forward", "generation"], requiredFor: [] },
    { name: "attention_mask", description: "Caller additive text attention bias; mutually exclusive with mm_token_type_ids.", required: false, dtype: "F32", shape: "[batch, 1|heads, query, key]", usedBy: ["forward"], requiredFor: [] },
    { name: "past_key_values", description: "Post-RoPE text KV cache for incremental decode.", required: false, dtype: "F32", shape: "layer -> {key,value}", usedBy: ["forward"], requiredFor: [] },
    { name: "pixel_values", description: "Patchified image pixels.", required: false, dtype: "F32", shape: "[batch, patches, 3*patch_size^2]", usedBy: ["forward", "generation"], requiredFor: [] },
    { name: "image_position_ids", description: "Image patch [x,y] positions; required with pixel_values.", required: false, dtype: "I32", shape: "[batch][patch][x,y]", usedBy: ["forward", "generation"], requiredFor: [] },
    { name: "pixel_values_videos", description: "Patchified video frames.", required: false, dtype: "F32", shape: "[videos, frames, patches, 3*patch_size^2]", usedBy: ["forward", "generation"], requiredFor: [] },
    { name: "video_position_ids", description: "Video frame patch [x,y] positions; required with pixel_values_videos.", required: false, dtype: "I32", shape: "[videos][frames][patch][x,y]", usedBy: ["forward", "generation"], requiredFor: [] },
    { name: "input_features", description: "Audio features before stride-2 subsampling.", required: false, dtype: "F32", shape: "[batch, frames, features]", usedBy: ["forward", "generation"], requiredFor: [] },
    { name: "input_features_mask", description: "Boolean validity mask for input_features.", required: false, dtype: "BOOL", shape: "[batch, frames]", usedBy: ["forward", "generation"], requiredFor: [] },
    { name: "mm_token_type_ids", description: "Image/video block IDs used only during uncached prefill.", required: false, dtype: "I32", shape: "[batch, sequence]", usedBy: ["forward", "generation"], requiredFor: [] },
    { name: "max_new_tokens", description: "Maximum greedy tokens; generation requires a non-negative integer.", required: false, dtype: "I32", shape: "[]", usedBy: ["generation"], requiredFor: ["generation"] },
    { name: "eos_token_id", description: "Optional non-negative token ID that stops after its incremental forward/cache transition.", required: false, dtype: "I32", shape: "[]", usedBy: ["generation"], requiredFor: [] },
  ];
}

export function validateGemma4CompositeLiteralInputs(inputs: Gemma4CompositeLiteralInput[]): void {
  if (!isDeepStrictEqual(inputs, literalInputs())) {
    throw new Error("Programa literal Gemma 4 composite não declara todos os controles de forward e geração.");
  }
}

export function gemma4LiteralGreedyGenerationProgram(
  program: Gemma4CompositeProgram,
  calculationGraph: Gemma4LiteralCalculationGraph = buildGemma4LiteralCalculationGraph(program),
): Gemma4LiteralGreedyGenerationProgram {
  const lastAssignment = program.textProgram.epilogue.at(-1)?.id;
  if (lastAssignment !== "final_logit_softcap" && lastAssignment !== "lm_head") {
    throw new Error("Programa literal Gemma 4 requer lm_head ou final_logit_softcap como atribuição final para geração.");
  }
  const iteration = "step = 0..max_new_tokens-1 while stop_after_step[step-1] is false" as const;
  const assignments: Gemma4LiteralGenerationAssignment[] = [
    {
      id: "generation_prefill", operation: "execute-declared-forward",
      inputs: ["input_ids", "position_ids?", "pixel_values?", "image_position_ids?", "pixel_values_videos?", "video_position_ids?", "input_features?", "input_features_mask?", "mm_token_type_ids?"],
      output: "forward_state[0]", dtype: "STRUCT", shape: "{logits:[1,sequence,vocab],past_key_values:layer->{key,value}}",
      semantics: "Execute the artifact's complete declared composite calculation once. Generation requires batch=1, non-empty input_ids and no caller attention_mask; all supplied modalities and mm_token_type_ids participate only in this prefill.",
    },
    {
      id: "generation_initial_position", operation: "initialize-position", inputs: ["position_ids?", "input_ids"], output: "position[-1]", dtype: "I32", shape: "[]",
      semantics: "position[-1] = position_ids[0,last] when supplied, otherwise input_ids.shape[1]-1.",
    },
    {
      id: "generation_selection_logits", operation: "capture-selection-logits", inputs: ["forward_state[step].logits"], output: "selection_logits[step]", dtype: "F32", shape: "[1,current_sequence,vocab]", iteration,
      semantics: "Capture the current forward state's complete declared logits tensor without changing dtype or values; incremental states have current_sequence=1.",
    },
    {
      id: "generation_argmax", operation: "argmax-lowest-token-id", inputs: ["selection_logits[step][0,current_sequence-1,0..vocab-1]"], output: "selected_token[step]", dtype: "I32", shape: "[]", iteration,
      semantics: "From the final sequence row, reject non-finite logits; choose the greatest logit and, for an exact tie, the lowest token ID by ascending token scan.",
    },
    {
      id: "generation_token_append", operation: "append-token", inputs: ["generated_token_ids[0..step-1]", "selected_token[step]"], output: "generated_token_ids[0..step]", dtype: "I32", shape: "[step+1]", iteration,
      semantics: "Append selected_token[step] exactly once, including when it equals eos_token_id.",
    },
    {
      id: "generation_position_advance", operation: "increment-position", inputs: ["position[step-1]"], output: "position[step]", dtype: "I32", shape: "[]", iteration,
      semantics: "position[step] = position[step-1] + 1 using exact integer arithmetic.",
    },
    {
      id: "generation_incremental_inputs", operation: "prepare-incremental-forward-inputs", inputs: ["selected_token[step]", "position[step]", "forward_state[step].past_key_values"], output: "incremental_inputs[step]", dtype: "STRUCT", shape: "{input_ids:[1,1],position_ids:[1,1],past_key_values:layer->{key,value}}", iteration,
      semantics: "Set input_ids=[[selected_token[step]]], position_ids=[[position[step]]] and carry the exact post-RoPE cache. Omit attention_mask, mm_token_type_ids, every image/video/audio input and every multimodal mask after prefill.",
    },
    {
      id: "generation_incremental_forward", operation: "execute-declared-incremental-forward", inputs: ["incremental_inputs[step]", "constants", "program"], output: "forward_state[step+1]", dtype: "STRUCT", shape: "{logits:[1,1,vocab],past_key_values:layer->{key,value}}", iteration,
      semantics: "Execute the same complete declared composite calculation with the incremental inputs. Text attention reads each serialized append-post-rope or reuse-producer cache transition; no generic decoder or source checkpoint is invoked.",
    },
    {
      id: "generation_logits_append", operation: "append-forward-logits-snapshot", inputs: ["step_forward_logits[0..step-1]", "forward_state[step+1].logits"], output: "step_forward_logits[0..step]", dtype: "F32", shape: "[step+1,1,1,vocab]", iteration,
      semantics: "Append the exact complete logits tensor produced after the selected token's incremental forward. This snapshot pairs with the cache snapshot from the same forward state and becomes the next step's selection logits.",
    },
    {
      id: "generation_cache_append", operation: "append-cache-snapshot", inputs: ["step_past_key_values[0..step-1]", "forward_state[step+1].past_key_values"], output: "step_past_key_values[0..step]", dtype: "STRUCT", shape: "[step+1] of layer->{key,value}", iteration,
      semantics: "Append the exact cache produced after the selected token's incremental forward, preserving BHSD layout and producer-layer ownership.",
    },
    {
      id: "generation_eos_stop", operation: "evaluate-eos-stop", inputs: ["eos_token_id?", "selected_token[step]", "forward_state[step+1]", "step_past_key_values[step]"], output: "stop_after_step[step]", dtype: "BOOL", shape: "[]", iteration,
      semantics: "After incremental logits and cache exist, stop iff eos_token_id is supplied and selected_token[step] == eos_token_id; otherwise continue until step+1 == max_new_tokens.",
    },
    {
      id: "generation_terminal_logits", operation: "select-terminal-logits", inputs: ["forward_state[executed_steps].logits", "stop_after_step", "max_new_tokens"], output: "terminal_logits", dtype: "F32", shape: "[1,1|prefill_sequence,vocab]",
      semantics: "Return logits from the final executed forward state; when max_new_tokens=0 this is prefill logits, otherwise it is the incremental logits produced after the last appended token, including EOS.",
    },
    {
      id: "generation_terminal_cache", operation: "select-terminal-cache", inputs: ["forward_state[executed_steps].past_key_values", "stop_after_step", "max_new_tokens"], output: "terminal_past_key_values", dtype: "STRUCT", shape: "layer->{key,value}",
      semantics: "Return the exact post-RoPE cache from the same final forward state as terminal_logits; when max_new_tokens=0 this is the prefill cache.",
    },
  ];
  const forwardCalculation = buildGemma4LiteralGenerationForwardCalculationContract(program, calculationGraph);
  return {
    kind: "gemma4-literal-greedy-generation-program",
    forwardProgram: {
      reference: "program",
      expansionOrder: "composite assignments in array order; vision/audio definitions inline at invocation; prepared text layers then epilogue",
      firstAssignment: "composite_block_sequence_ids",
      lastAssignment,
    },
    forwardCalculation,
    loop: { iterator: "step", startInclusive: 0, endExclusiveInput: "max_new_tokens", earlyStop: "after incremental forward and cache capture when selected_token[step] == eos_token_id" },
    assignments,
    scalarCalculations: buildGemma4LiteralGenerationScalarCalculations(assignments, forwardCalculation, program.contract.text.vocabSize),
    controlProgram: buildGemma4LiteralGenerationControlProgram(program),
    outputs: {
      prefillState: "forward_state[0]",
      generatedTokenIds: "generated_token_ids",
      selectionLogits: "selection_logits",
      stepForwardLogits: "step_forward_logits",
      stepPastKeyValues: "step_past_key_values",
      terminalLogits: "terminal_logits",
      terminalPastKeyValues: "terminal_past_key_values",
    },
  };
}

export function validateGemma4LiteralGenerationProgram(generation: Gemma4LiteralGreedyGenerationProgram, program: Gemma4CompositeProgram): void {
  const expected = gemma4LiteralGreedyGenerationProgram(program);
  if (!isDeepStrictEqual(generation, expected)) {
    throw new Error("Programa literal Gemma 4 possui atribuições ou transições de geração greedy incompletas.");
  }
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
    validateGemma4TextBf16Tanh(operation);
    validateGemma4TextBf16Rotary(operation);
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
    if (reduction.kind === "pytorch-cpu-f32-cascade-sum") {
      if (operation.op !== "rms_norm" || operation.dtypePolicy.accumulationDtype !== "F32" || reduction.vectorLanes !== 4 || reduction.ilpFactor !== 4 ||
        reduction.cascadeLevels !== 4 || reduction.minimumLevelStep !== 16 || reduction.registerFold !== "ascending" || reduction.laneFold !== "ascending" ||
        !Number.isSafeInteger(operation.reductionSize) || operation.reductionSize! <= 0 || operation.reductionSize! % 16 !== 0) {
        throw new Error(`${operation.id}: agenda PyTorch CPU cascade RMS Gemma 4 inválida.`);
      }
      continue;
    }
    if (reduction.kind === "pytorch-cpu-bf16-welford") {
      throw new Error(`${operation.id}: agenda Welford de LayerNorm de canais não é uma redução textual Gemma 4.`);
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

function validateGemma4TextBf16Rotary(operation: Operation): void {
  if (operation.op !== "rotary_embedding" || operation.dtypePolicy.outputDtype !== "BF16") return;
  if (!isDeepStrictEqual(operation.trigImplementation, GEMMA4_E4B_PYTORCH_BF16_TRIG_IMPLEMENTATION)) {
    throw new Error(`${operation.id}: RoPE BF16 Gemma 4 não declara os kernels SLEEF fixados pelo runtime.`);
  }
  if (!isDeepStrictEqual(operation.rotaryCasts, {
    cosine: "BF16", sine: "BF16", directProduct: "BF16", rotatedProduct: "BF16", sum: "BF16",
  })) {
    throw new Error(`${operation.id}: RoPE BF16 Gemma 4 não declara todas as fronteiras de cast do runtime.`);
  }
}

/**
 * PyTorch's BF16 GELU and final softcap do not use the host JavaScript tanh.
 * The artifact must retain both the pinned SLEEF implementation and, for the
 * composite softcap expression, each native BF16 materialization boundary.
 */
function validateGemma4TextBf16Tanh(operation: Operation): void {
  const bf16Gelu = operation.op === "activation" && operation.function === "gelu" && operation.approximation === "tanh" && operation.dtypePolicy.outputDtype === "BF16";
  const bf16Softcap = operation.op === "elementwise" && operation.kind === "tanh_softcap" && operation.dtypePolicy.outputDtype === "BF16";
  if (!bf16Gelu && !bf16Softcap) return;
  if (!isDeepStrictEqual(operation.tanhImplementation, GEMMA4_E4B_PYTORCH_BF16_TANH_IMPLEMENTATION)) {
    throw new Error(`${operation.id}: operação tanh BF16 Gemma 4 não declara o kernel SLEEF fixado pelo runtime.`);
  }
  if (bf16Softcap && !isDeepStrictEqual(operation.tanhSoftcapCasts, {
    afterDivide: "BF16", afterTanh: "BF16", afterMultiply: "BF16",
  })) {
    throw new Error(`${operation.id}: softcap tanh BF16 Gemma 4 não declara todas as fronteiras de cast do runtime.`);
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
  storageDecoders: LiteralDenseStorageDecodeAssignment[];
  denseDecoderLanguage: LiteralDenseDecoderLanguageContract;
  unreachableConstants: Gemma4CompositeUnreachableConstant[];
  embeddedProgram: Gemma4CompositeProgram;
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
}

/** Builds and validates all non-payload state before opening the output file. */
function prepareStreamedDenseLiteral(
  program: Gemma4CompositeProgram,
  catalog: ModelCatalog,
  authoritativeExecution: Gemma4AuthoritativeExecutionContract,
): PreparedStreamedDenseLiteral {
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
  const calculationDomains = buildGemma4LiteralCalculationDomains(embeddedProgram);
  const learnedOperands = buildGemma4LiteralLearnedOperandBindings(embeddedProgram);
  const scalarCalculations = buildGemma4LiteralScalarCalculations(embeddedProgram);
  const calculationGraph = buildGemma4LiteralCalculationGraph(embeddedProgram);
  const fidelityGate = buildGemma4LiteralFidelityGate(calculationGraph, authoritativeExecution);
  const forwardControl = buildGemma4LiteralForwardControlProgram(embeddedProgram);
  const inputContract = buildGemma4LiteralInputContract(embeddedProgram);
  const outputContract = buildGemma4LiteralOutputContract(embeddedProgram);
  const generation = gemma4LiteralGreedyGenerationProgram(embeddedProgram, calculationGraph);
  const numericLiterals = buildGemma4LiteralNumericLiterals(scalarCalculations, generation.scalarCalculations, generation.forwardCalculation);
  const formulaLanguage = buildGemma4LiteralFormulaLanguageContract();
  const transcendentalPrograms = buildGemma4LiteralTranscendentalPrograms();
  validateGemma4LiteralCalculationDomains(calculationDomains, embeddedProgram);
  validateGemma4LiteralLearnedOperandBindings(learnedOperands, embeddedProgram);
  validateGemma4LiteralScalarCalculations(scalarCalculations, embeddedProgram);
  validateGemma4LiteralFormulaLanguageContract(formulaLanguage, scalarCalculations, generation.scalarCalculations, generation.forwardCalculation);
  validateGemma4LiteralTranscendentalCoverage(transcendentalPrograms, [
    ...scalarCalculations.assignments.flatMap((assignment) => assignment.scalarAssignments),
    ...generation.scalarCalculations.assignments.flatMap((assignment) => assignment.scalarAssignments),
  ]);
  validateGemma4LiteralNumericLiterals(numericLiterals, scalarCalculations, generation.scalarCalculations, generation.forwardCalculation);
  validateGemma4LiteralCalculationGraph(calculationGraph, embeddedProgram);
  validateGemma4LiteralFidelityGate(fidelityGate, calculationGraph, authoritativeExecution);
  validateGemma4LiteralForwardControlProgram(forwardControl, embeddedProgram);
  validateGemma4LiteralInputContract(inputContract, embeddedProgram);
  validateGemma4LiteralOutputContract(outputContract, embeddedProgram);
  const constantMap = new Map<string, LiteralConstant>(constants.map((constant) => [constant.name, { ...constant.metadata, payloadBase64: "" }]));
  validateGemma4CompositeLiteralStructure(embeddedProgram, assignments, outputs, generation, constantMap, unreachableConstants);
  return {
    constants,
    storageDecoders: constants.map((constant) => denseStorageDecoder(constant.metadata)),
    denseDecoderLanguage: buildLiteralDenseDecoderLanguageContract(),
    unreachableConstants,
    embeddedProgram,
    assignments,
    calculationDomains,
    learnedOperands,
    scalarCalculations,
    formulaLanguage,
    transcendentalPrograms,
    numericLiterals,
    calculationGraph,
    fidelityGate,
    forwardControl,
    inputContract,
    outputContract,
    outputs,
    generation,
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
  return buildLiteralDenseStorageDecodeAssignment(constant);
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
  generation: Gemma4LiteralGreedyGenerationProgram,
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
  validateGemma4LiteralGenerationProgram(generation, program);
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
