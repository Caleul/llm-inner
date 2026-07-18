import { constants as fsConstants } from "node:fs";
import { access } from "node:fs/promises";
import { compareCapturedOperationCheckpoints } from "./differential.js";
import { assertGemma4AuthoritativeRuntime } from "./gemma4-authoritative-runtime.js";
import {
  executeGemma4AudioBf16CastF32,
  executeGemma4AudioDerivedAttentionStagesF32,
  executeGemma4AudioF32,
  type Gemma4AudioExecutionRequest,
  type Gemma4AudioExecutionResult,
} from "./gemma4-audio.js";
import { openGemma4CompositeLiteralArtifact, type OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { readLiteralDenseF32Tensor } from "./paged-dense.js";
import type { DenseF32Tensor, DifferentialCheckpointComparisonReport, DifferentialOperationSample, TensorInfo, TensorRef } from "./types.js";

export interface Gemma4AudioLiteralExecutionOptions {
  /** Largest single decoded tensor allocation. The complete model is never read as one buffer. */
  maxTensorBytes?: number;
}

export interface Gemma4AudioDifferentialTrace {
  schemaVersion: 1;
  kind: "gemma4-audio-checkpoints";
  source: {
    model: string;
    revisionOrChecksum: string;
    containerFormat: "safetensors";
    quantization: "none; dense BF16 storage";
  };
  reference: {
    runtime: string;
    executionMode: "torch.inference_mode";
    attentionImplementation: "eager";
    executionDevice: "cpu";
    dtypePolicy: string;
    inputFeatures: DenseF32Tensor;
    inputFeaturesMask: boolean[][];
    operations: DifferentialOperationSample[];
  };
}

export interface Gemma4AudioDifferentialComparisonReport extends DifferentialCheckpointComparisonReport {
  /** Exact validation of non-native score stages, seeded only at the two authoritative matmul outputs. */
  sourceAnchoredAttentionStages: DifferentialCheckpointComparisonReport;
  /** Exact validation of the BF16 cast, seeded at each authoritative native F32 value-matmul output. */
  sourceAnchoredAttentionContextCasts: DifferentialCheckpointComparisonReport;
}

/**
 * Execute the real Gemma 4 audio assignment graph from its embedded literal
 * constants. Tensor names are accepted only when the serialized program owns
 * a matching dense constant; there is no checkpoint or catalog fallback.
 */
export async function executeGemma4LiteralAudioF32(
  artifact: OpenGemma4CompositeLiteralArtifact,
  request: Omit<Gemma4AudioExecutionRequest, "tensors">,
  options: Gemma4AudioLiteralExecutionOptions = {},
): Promise<Gemma4AudioExecutionResult> {
  const maxTensorBytes = options.maxTensorBytes ?? 16 * 1024 * 1024;
  if (!Number.isSafeInteger(maxTensorBytes) || maxTensorBytes <= 0) throw new Error("Execução literal de áudio requer maxTensorBytes positivo seguro.");
  const references = new Map<string, TensorRef>();
  for (const assignment of artifact.program.audioProgram.assignments) for (const reference of assignment.tensors ?? []) {
    const existing = references.get(reference.name);
    if (existing && !sameReference(existing, reference)) throw new Error(`${reference.name}: programa de áudio possui referências incompatíveis.`);
    references.set(reference.name, reference);
  }
  const tensors = new Map<string, DenseF32Tensor>();
  for (const reference of references.values()) {
    const info = literalTensorInfo(artifact, reference);
    tensors.set(reference.name, await readLiteralDenseF32Tensor(info, artifact, maxTensorBytes));
  }
  return executeGemma4AudioF32(artifact.program.audioProgram, { ...request, tensors });
}

/** Source-removed comparison at authoritative module boundaries and output. */
export async function compareGemma4LiteralAudioTrace(options: {
  artifact: string;
  trace: Gemma4AudioDifferentialTrace;
  maxTensorBytes: number;
  maxAbsoluteError?: number;
  maxRelativeError?: number;
  topK?: number;
  assertSourceUnavailable?: string;
}): Promise<Gemma4AudioDifferentialComparisonReport> {
  if (options.assertSourceUnavailable) {
    let exists = true;
    try { await access(options.assertSourceUnavailable, fsConstants.F_OK); } catch { exists = false; }
    if (exists) throw new Error(`Comparação de áudio Gemma 4 requer source indisponível, mas '${options.assertSourceUnavailable}' ainda existe.`);
  }
  assertTrace(options.trace);
  const artifact = await openGemma4CompositeLiteralArtifact(options.artifact);
  try {
    const candidate = await executeGemma4LiteralAudioF32(artifact, {
      inputFeatures: options.trace.reference.inputFeatures,
      inputFeaturesMask: options.trace.reference.inputFeaturesMask,
    }, { maxTensorBytes: options.maxTensorBytes });
    const comparison = compareCapturedOperationCheckpoints(candidate.values, { operations: options.trace.reference.operations }, {
      candidateRuntime: "llm-inner embedded-literal Gemma4Audio BF16-policy scalar executor",
      ...(options.topK === undefined ? {} : { topK: options.topK }),
      tolerance: {
        maxAbsoluteError: options.maxAbsoluteError ?? 0,
        maxRelativeError: options.maxRelativeError ?? 0,
      },
    });
    const operations = indexOperations(options.trace.reference.operations);
    const sourceAnchoredAttentionStages = compareSourceAnchoredAttentionStages(
      artifact.program.audioProgram,
      operations,
      options.trace.reference.inputFeaturesMask,
      options.topK,
    );
    const sourceAnchoredAttentionContextCasts = compareSourceAnchoredAttentionContextCasts(
      artifact.program.audioProgram,
      operations,
      options.topK,
    );
    return { ...comparison, sourceAnchoredAttentionStages, sourceAnchoredAttentionContextCasts };
  } finally {
    await artifact.close();
  }
}

function compareSourceAnchoredAttentionStages(
  program: OpenGemma4CompositeLiteralArtifact["program"]["audioProgram"],
  operations: ReadonlyMap<string, DifferentialOperationSample>,
  mask: readonly boolean[][],
  topK: number | undefined,
): DifferentialCheckpointComparisonReport {
  const candidates = new Map<string, DenseF32Tensor>(), references: DifferentialOperationSample[] = [];
  for (let layer = 0; layer < program.tower.layers; layer += 1) {
    const stem = `audio_layer_${layer}_attention`;
    const content = requiredSample(operations, `${stem}_ac`);
    const position = requiredSample(operations, `${stem}_bd_unshifted`);
    const derived = executeGemma4AudioDerivedAttentionStagesF32(program, content.tensor, position.tensor, mask);
    for (const [suffix, tensor] of [
      ["bd", derived.shiftedPositionScores],
      ["logits", derived.logits],
      ["softcapped", derived.softcapped],
      ["scores", derived.maskedScores],
    ] as const) {
      const output = `${stem}_${suffix}`;
      candidates.set(output, tensor);
      references.push(requiredSample(operations, output));
    }
  }
  return compareCapturedOperationCheckpoints(candidates, { operations: references }, {
    candidateRuntime: "llm-inner source-anchored literal relative-shift/add/softcap/mask executor",
    ...(topK === undefined ? {} : { topK }),
    tolerance: { maxAbsoluteError: 0, maxRelativeError: 0 },
  });
}

function compareSourceAnchoredAttentionContextCasts(
  program: OpenGemma4CompositeLiteralArtifact["program"]["audioProgram"],
  operations: ReadonlyMap<string, DifferentialOperationSample>,
  topK: number | undefined,
): DifferentialCheckpointComparisonReport {
  const candidates = new Map<string, DenseF32Tensor>(), references: DifferentialOperationSample[] = [];
  for (let layer = 0; layer < program.tower.layers; layer += 1) {
    const context = requiredSample(operations, `audio_layer_${layer}_attention_context`);
    const output = `audio_layer_${layer}_attention_context_bf16`;
    candidates.set(output, executeGemma4AudioBf16CastF32(context.tensor));
    references.push(requiredSample(operations, output));
  }
  return compareCapturedOperationCheckpoints(candidates, { operations: references }, {
    candidateRuntime: "llm-inner source-anchored literal F32-to-BF16 attention-context cast",
    ...(topK === undefined ? {} : { topK }),
    tolerance: { maxAbsoluteError: 0, maxRelativeError: 0 },
  });
}

function indexOperations(operations: readonly DifferentialOperationSample[]): ReadonlyMap<string, DifferentialOperationSample> {
  const byOutput = new Map<string, DifferentialOperationSample>();
  for (const operation of operations) {
    if (byOutput.has(operation.output)) throw new Error(`Trace de áudio redeclara output '${operation.output}'.`);
    byOutput.set(operation.output, operation);
  }
  return byOutput;
}

function requiredSample(
  samples: ReadonlyMap<string, DifferentialOperationSample>,
  output: string,
): DifferentialOperationSample & { tensor: DenseF32Tensor } {
  const sample = samples.get(output);
  if (!sample) throw new Error(`Trace de áudio não contém fronteira explícita '${output}'.`);
  if (!(sample.tensor.values instanceof Float32Array)) throw new Error(`Trace de áudio '${output}' não possui payload F32.`);
  return sample as DifferentialOperationSample & { tensor: DenseF32Tensor };
}

function literalTensorInfo(artifact: OpenGemma4CompositeLiteralArtifact, reference: TensorRef): TensorInfo {
  const constant = artifact.constants.get(reference.name);
  if (!constant || reference.quantization || constant.quantization || constant.storageDtype !== reference.storageDtype ||
    !sameShape(constant.logicalShape, reference.shape) || !sameShape(constant.storageShape, reference.shape)) {
    throw new Error(`${reference.name}: referência de áudio não corresponde à constante densa literal.`);
  }
  return {
    name: constant.name,
    storageDtype: constant.storageDtype,
    storageShape: [...constant.storageShape],
    logicalShape: [...constant.logicalShape],
  };
}

function assertTrace(trace: Gemma4AudioDifferentialTrace): void {
  assertGemma4AuthoritativeRuntime("audio", trace.reference);
  const input = trace.reference.inputFeatures;
  const elements = input.shape.reduce((total, dimension) => total * dimension, 1);
  if (trace.schemaVersion !== 1 || trace.kind !== "gemma4-audio-checkpoints" || trace.source.containerFormat !== "safetensors" ||
    trace.source.quantization !== "none; dense BF16 storage" || trace.reference.executionDevice !== "cpu" ||
    input.shape.length !== 3 || input.values.length !== elements || trace.reference.inputFeaturesMask.length !== input.shape[0] ||
    trace.reference.inputFeaturesMask.some((row) => row.length !== input.shape[1]) || trace.reference.operations.length === 0) {
    throw new Error("Trace diferencial de áudio Gemma 4 inválido.");
  }
}

function sameReference(left: TensorRef, right: TensorRef): boolean {
  return left.name === right.name && left.storageDtype === right.storageDtype && !left.quantization && !right.quantization && sameShape(left.shape, right.shape);
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((dimension, index) => dimension === right[index]);
}
