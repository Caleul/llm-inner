import { constants as fsConstants } from "node:fs";
import { access } from "node:fs/promises";
import { compareCapturedOperationCheckpoints } from "./differential.js";
import { openGemma4CompositeLiteralArtifact, type OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { executeGemma4VisionF32, type Gemma4VisionExecutionRequest, type Gemma4VisionExecutionResult } from "./gemma4-vision.js";
import { readLiteralDenseF32Tensor } from "./paged-dense.js";
import type { DenseF32Tensor, DifferentialCheckpointComparisonReport, DifferentialOperationSample, TensorInfo, TensorRef } from "./types.js";

export type Gemma4VisionInvocation = "image" | "video";

export interface Gemma4VisionLiteralExecutionOptions {
  /** Largest single decoded tensor allocation. The complete model is never read as one buffer. */
  maxTensorBytes?: number;
}

export interface Gemma4VisionDifferentialTrace {
  schemaVersion: 1;
  kind: "gemma4-vision-checkpoints";
  invocation: Gemma4VisionInvocation;
  source: {
    model: string;
    revisionOrChecksum: string;
    containerFormat: "safetensors";
    quantization: "none; dense BF16 storage";
  };
  reference: {
    runtime: string;
    executionDevice: "cpu";
    dtypePolicy: string;
    pixelValues: DenseF32Tensor;
    pixelPositionIds: number[][][] | number[][][][];
    operations: DifferentialOperationSample[];
  };
}

/** Execute the shared real image/video tower solely from embedded constants. */
export async function executeGemma4LiteralVisionF32(
  artifact: OpenGemma4CompositeLiteralArtifact,
  request: Omit<Gemma4VisionExecutionRequest, "tensors">,
  options: Gemma4VisionLiteralExecutionOptions = {},
): Promise<Gemma4VisionExecutionResult> {
  const maxTensorBytes = options.maxTensorBytes ?? 64 * 1024 * 1024;
  if (!Number.isSafeInteger(maxTensorBytes) || maxTensorBytes <= 0) throw new Error("Execução literal de vision requer maxTensorBytes positivo seguro.");
  const references = new Map<string, TensorRef>();
  for (const assignment of artifact.program.visionProgram.assignments) for (const reference of assignment.tensors ?? []) {
    const existing = references.get(reference.name);
    if (existing && !sameReference(existing, reference)) throw new Error(`${reference.name}: programa vision possui referências incompatíveis.`);
    references.set(reference.name, reference);
  }
  const tensors = new Map<string, DenseF32Tensor>();
  for (const reference of references.values()) {
    const info = literalTensorInfo(artifact, reference);
    tensors.set(reference.name, await readLiteralDenseF32Tensor(info, artifact, maxTensorBytes));
  }
  return executeGemma4VisionF32(artifact.program.visionProgram, { ...request, tensors });
}

/** Source-removed image or video comparison at authoritative tower boundaries. */
export async function compareGemma4LiteralVisionTrace(options: {
  artifact: string;
  trace: Gemma4VisionDifferentialTrace;
  maxTensorBytes: number;
  maxAbsoluteError?: number;
  maxRelativeError?: number;
  topK?: number;
  assertSourceUnavailable?: string;
}): Promise<DifferentialCheckpointComparisonReport> {
  if (options.assertSourceUnavailable) {
    let exists = true;
    try { await access(options.assertSourceUnavailable, fsConstants.F_OK); } catch { exists = false; }
    if (exists) throw new Error(`Comparação vision Gemma 4 requer source indisponível, mas '${options.assertSourceUnavailable}' ainda existe.`);
  }
  assertTrace(options.trace);
  const flattened = flattenInvocation(options.trace);
  const artifact = await openGemma4CompositeLiteralArtifact(options.artifact);
  try {
    const candidate = await executeGemma4LiteralVisionF32(artifact, flattened, { maxTensorBytes: options.maxTensorBytes });
    return compareCapturedOperationCheckpoints(candidate.values, { operations: options.trace.reference.operations }, {
      candidateRuntime: `llm-inner embedded-literal Gemma4Vision ${options.trace.invocation} BF16-policy scalar executor`,
      ...(options.topK === undefined ? {} : { topK: options.topK }),
      tolerance: {
        maxAbsoluteError: options.maxAbsoluteError ?? 0,
        maxRelativeError: options.maxRelativeError ?? 0,
      },
    });
  } finally {
    await artifact.close();
  }
}

function flattenInvocation(trace: Gemma4VisionDifferentialTrace): Omit<Gemma4VisionExecutionRequest, "tensors"> {
  if (trace.invocation === "image") return {
    pixelValues: trace.reference.pixelValues,
    pixelPositionIds: trace.reference.pixelPositionIds as number[][][],
  };
  const [videos, frames, patches, width] = trace.reference.pixelValues.shape;
  return {
    pixelValues: { shape: [videos! * frames!, patches!, width!], values: trace.reference.pixelValues.values },
    pixelPositionIds: (trace.reference.pixelPositionIds as number[][][][]).flatMap((video) => video),
  };
}

function literalTensorInfo(artifact: OpenGemma4CompositeLiteralArtifact, reference: TensorRef): TensorInfo {
  const constant = artifact.constants.get(reference.name);
  if (!constant || reference.quantization || constant.quantization || constant.storageDtype !== reference.storageDtype ||
    !sameShape(constant.logicalShape, reference.shape) || !sameShape(constant.storageShape, reference.shape)) {
    throw new Error(`${reference.name}: referência vision não corresponde à constante densa literal.`);
  }
  return { name: constant.name, storageDtype: constant.storageDtype, storageShape: [...constant.storageShape], logicalShape: [...constant.logicalShape] };
}

function assertTrace(trace: Gemma4VisionDifferentialTrace): void {
  const input = trace.reference.pixelValues;
  const rank = trace.invocation === "image" ? 3 : 4;
  const elements = input.shape.reduce((total, dimension) => total * dimension, 1);
  const positionShape = trace.invocation === "image"
    ? [trace.reference.pixelPositionIds.length, (trace.reference.pixelPositionIds as number[][][])[0]?.length]
    : [trace.reference.pixelPositionIds.length, (trace.reference.pixelPositionIds as number[][][][])[0]?.length, (trace.reference.pixelPositionIds as number[][][][])[0]?.[0]?.length];
  const inputPrefix = input.shape.slice(0, -1);
  if (trace.schemaVersion !== 1 || trace.kind !== "gemma4-vision-checkpoints" || trace.source.containerFormat !== "safetensors" ||
    trace.source.quantization !== "none; dense BF16 storage" || trace.reference.executionDevice !== "cpu" || input.shape.length !== rank ||
    input.shape.at(-1) !== 768 || input.values.length !== elements || inputPrefix.some((dimension, index) => dimension !== positionShape[index]) ||
    trace.reference.operations.length === 0) throw new Error("Trace vision Gemma 4 inválido.");
}

function sameReference(left: TensorRef, right: TensorRef): boolean {
  return left.name === right.name && left.storageDtype === right.storageDtype && !left.quantization && !right.quantization && sameShape(left.shape, right.shape);
}
function sameShape(left: readonly number[], right: readonly number[]): boolean { return left.length === right.length && left.every((dimension, index) => dimension === right[index]); }
