import { constants as fsConstants } from "node:fs";
import { access, readFile } from "node:fs/promises";
import { compareCapturedOperationCheckpoints, compareGenerationTrace } from "./differential.js";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { generateGemma4LiteralCompositeF32 } from "./gemma4-literal-composite.js";
import { gemma4CompositeTraceProfile, type Gemma4CompositeTraceModality } from "./gemma4-composite-trace-profile.js";
import { fingerprintIR, readGenerationTraceBundle } from "./trace.js";
import type {
  DenseF32Tensor,
  DifferentialCheckpointComparisonReport,
  DifferentialGenerationComparisonReport,
  DifferentialOperationSample,
} from "./types.js";

interface CompositeInputs {
  modality: Gemma4CompositeTraceModality;
  inputTokens: number[];
  positionIds: number[];
  mmTokenTypeIds: number[];
  pixelValues?: { shape: number[]; values: number[] };
  imagePositionIds?: number[][][];
  pixelValuesVideos?: { shape: number[]; values: number[] };
  videoPositionIds?: number[][][][];
  inputFeatures?: { shape: number[]; values: number[] };
  inputFeaturesMask?: boolean[][];
  maxNewTokens: number;
}

interface SerializedOperation {
  operationId: string;
  output: string;
  tensor: { dtype: "F32"; shape: number[]; valuesBase64: string };
}

export interface Gemma4LiteralCompositeDifferentialReport {
  prefill: DifferentialCheckpointComparisonReport;
  generation: DifferentialGenerationComparisonReport;
}

/** Compares real image, video, or audio prefill and cached generation with the source absent. */
export async function compareGemma4LiteralCompositeTrace(options: {
  artifact: string;
  trace: string;
  maxReadBytes: number;
  maxTowerTensorBytes: number;
  maxAbsoluteError?: number;
  maxRelativeError?: number;
  topK?: number;
  assertSourceUnavailable?: string;
  allowUnverifiedFidelity?: boolean;
}): Promise<Gemma4LiteralCompositeDifferentialReport> {
  if (![options.maxReadBytes, options.maxTowerTensorBytes].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new Error("Comparação composite requer limites de leitura positivos seguros.");
  }
  if (options.assertSourceUnavailable) {
    let exists = true;
    try { await access(options.assertSourceUnavailable, fsConstants.F_OK); } catch { exists = false; }
    if (exists) throw new Error(`Comparação composite requer source indisponível, mas '${options.assertSourceUnavailable}' ainda existe.`);
  }
  const decoded = await readGenerationTraceBundle(options.trace);
  if (decoded.bundle.candidatePolicy.runtime !== "llm-inner embedded-literal Gemma4 composite BF16-policy executor") {
    throw new Error("Trace não declara o executor composite literal esperado.");
  }
  const raw = JSON.parse(await readFile(options.trace, "utf8")) as { compositeInputs?: unknown; prefillOperations?: unknown };
  const inputs = parseInputs(raw.compositeInputs, decoded.reference.inputTokens, decoded.reference.promptPositionIds, decoded.reference.maxNewTokens);
  const operations = parseOperations(raw.prefillOperations, inputs.modality);
  const artifact = await openGemma4CompositeLiteralArtifact(options.artifact);
  try {
    if (fingerprintIR(artifact.program.textProgram) !== decoded.bundle.irFingerprint) throw new Error("Trace composite não corresponde ao programa textual incorporado.");
    const candidate = await generateGemma4LiteralCompositeF32(artifact, {
      inputIds: [inputs.inputTokens],
      positionIds: [inputs.positionIds],
      mmTokenTypeIds: [inputs.mmTokenTypeIds],
      ...(inputs.pixelValues ? { pixelValues: denseInput(inputs.pixelValues), imagePositionIds: inputs.imagePositionIds! } : {}),
      ...(inputs.pixelValuesVideos ? { pixelValuesVideos: denseInput(inputs.pixelValuesVideos), videoPositionIds: inputs.videoPositionIds! } : {}),
      ...(inputs.inputFeatures ? { inputFeatures: denseInput(inputs.inputFeatures), inputFeaturesMask: inputs.inputFeaturesMask! } : {}),
      maxNewTokens: inputs.maxNewTokens,
    }, {
      maxReadBytes: options.maxReadBytes,
      maxTowerTensorBytes: options.maxTowerTensorBytes,
      ...(options.allowUnverifiedFidelity ? { allowUnverifiedFidelity: true } : {}),
    });
    const tolerance = { maxAbsoluteError: options.maxAbsoluteError ?? 0, maxRelativeError: options.maxRelativeError ?? 0 };
    const candidateRuntime = decoded.bundle.candidatePolicy.runtime;
    return {
      prefill: compareCapturedOperationCheckpoints(candidate.compositePrefill.values, { operations }, {
        candidateRuntime,
        tolerance,
        ...(options.topK === undefined ? {} : { topK: options.topK }),
      }),
      generation: compareGenerationTrace(candidate, decoded.reference, {
        candidateRuntime,
        tolerance,
        ...(options.topK === undefined ? {} : { topK: options.topK }),
      }),
    };
  } finally {
    await artifact.close();
  }
}

function parseInputs(raw: unknown, inputTokens: number[], positionIds: number[], maxNewTokens: number): CompositeInputs {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("Trace composite não declara compositeInputs objeto.");
  const value = raw as Partial<CompositeInputs>;
  const vectors = [value.inputTokens, value.positionIds, value.mmTokenTypeIds];
  const modality = value.modality;
  if (modality !== "image" && modality !== "video" && modality !== "audio") throw new Error("Trace composite não declara modalidade image, video ou audio.");
  const profile = gemma4CompositeTraceProfile(modality);
  if (vectors.some((entry) => !Array.isArray(entry) || entry.some((item) => !Number.isSafeInteger(item) || item < 0)) ||
    !sameVector(value.inputTokens!, inputTokens) || !sameVector(value.positionIds!, positionIds) || value.mmTokenTypeIds!.length !== inputTokens.length ||
    value.maxNewTokens !== maxNewTokens) {
    throw new Error("Trace composite possui inputs multimodais inválidos ou divergentes da geração.");
  }
  const modalities = [value.pixelValues !== undefined || value.imagePositionIds !== undefined,
    value.pixelValuesVideos !== undefined || value.videoPositionIds !== undefined,
    value.inputFeatures !== undefined || value.inputFeaturesMask !== undefined];
  const expected = modality === "image" ? 0 : modality === "video" ? 1 : 2;
  if (modalities.filter(Boolean).length !== 1 || !modalities[expected] ||
    value.mmTokenTypeIds!.filter((entry) => entry === profile.tokenTypeId).length !== (profile.tokenTypeId === 0 ? value.mmTokenTypeIds!.length : 1)) {
    throw new Error(`Trace composite ${modality} possui roteamento de modalidade incompatível.`);
  }
  if (modality === "image") parseVisionInput(value.pixelValues, value.imagePositionIds, false);
  if (modality === "video") parseVisionInput(value.pixelValuesVideos, value.videoPositionIds, true);
  if (modality === "audio") parseAudioInput(value.inputFeatures, value.inputFeaturesMask);
  return value as CompositeInputs;
}

function parseOperations(raw: unknown, expectedModality: Gemma4CompositeTraceModality): DifferentialOperationSample[] {
  if (!Array.isArray(raw) || raw.length !== 5) throw new Error("Trace composite requer cinco fronteiras de prefill.");
  const feature = raw[1] as Partial<SerializedOperation> | undefined;
  const modality = feature?.operationId === "composite_image_features" ? "image" : feature?.operationId === "composite_video_features" ? "video" : feature?.operationId === "composite_audio_features" ? "audio" : undefined;
  if (!modality || modality !== expectedModality) throw new Error("Trace composite não declara fronteira de feature coerente com sua modalidade.");
  const profile = gemma4CompositeTraceProfile(modality);
  const boundaries = [
    ["composite_text_embedding", "composite_text_embeddings"],
    [profile.featureOperationId, profile.featureOutput],
    [profile.scatterOperationId, "hidden_states_0"],
    ["composite_ple_combine_scale", "ple_inputs"],
    ["final_logit_softcap", "softcapped_logits"],
  ] as const;
  return raw.map((entry, index) => {
    if (!entry || typeof entry !== "object") throw new Error(`Fronteira composite ${index} inválida.`);
    const operation = entry as Partial<SerializedOperation>;
    const expected = boundaries[index]!;
    if (operation.operationId !== expected[0] || operation.output !== expected[1] || !operation.tensor) throw new Error(`Fronteira composite ${index} não corresponde a ${expected[0]} -> ${expected[1]}.`);
    return { operationId: operation.operationId, output: operation.output, tensor: decodeTensor(operation.tensor, operation.operationId) };
  });
}

function denseInput(input: { shape: number[]; values: number[] }): DenseF32Tensor {
  return { shape: [...input.shape], values: Float32Array.from(input.values) };
}

function parseVisionInput(tensor: { shape: number[]; values: number[] } | undefined, positions: number[][][] | number[][][][] | undefined, video: boolean): void {
  const rank = video ? 4 : 3;
  if (!tensor || !positions || tensor.shape.length !== rank || tensor.shape.at(-1) !== 768 ||
    tensor.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0) || !Array.isArray(tensor.values) ||
    tensor.shape.reduce((total, dimension) => total * dimension, 1) !== tensor.values.length || tensor.values.some((item) => !Number.isFinite(item))) {
    throw new Error(`Trace composite possui pixels ${video ? "video" : "image"} inválidos.`);
  }
  const flattened = video ? (positions as number[][][][]).flat(1) : positions as number[][][];
  const batches = video ? tensor.shape[0]! * tensor.shape[1]! : tensor.shape[0]!;
  if (flattened.length !== batches || flattened.some((batch) => batch.length !== tensor.shape.at(-2) || batch.some((position) => position.length !== 2 || position.some((item) => !Number.isSafeInteger(item))))) {
    throw new Error(`Trace composite possui posições ${video ? "video" : "image"} inválidas.`);
  }
}

function parseAudioInput(tensor: { shape: number[]; values: number[] } | undefined, mask: boolean[][] | undefined): void {
  if (!tensor || !mask || tensor.shape.length !== 3 || tensor.shape[2] !== 128 ||
    tensor.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0) || !Array.isArray(tensor.values) ||
    tensor.shape.reduce((total, dimension) => total * dimension, 1) !== tensor.values.length || tensor.values.some((item) => !Number.isFinite(item)) ||
    mask.length !== tensor.shape[0] || mask.some((row) => row.length !== tensor.shape[1] || row.some((item) => typeof item !== "boolean"))) {
    throw new Error("Trace composite possui input_features ou mask inválidos.");
  }
}

function decodeTensor(tensor: SerializedOperation["tensor"], label: string): DenseF32Tensor {
  if (tensor.dtype !== "F32" || !Array.isArray(tensor.shape) || tensor.shape.length === 0 || tensor.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0) || typeof tensor.valuesBase64 !== "string") {
    throw new Error(`${label}: tensor F32 serializado inválido.`);
  }
  const bytes = Buffer.from(tensor.valuesBase64, "base64");
  const elements = tensor.shape.reduce((total, dimension) => total * dimension, 1);
  if (bytes.length !== elements * 4) throw new Error(`${label}: payload F32 não corresponde ao shape.`);
  const values = new Float32Array(elements);
  for (let index = 0; index < elements; index += 1) values[index] = bytes.readFloatLE(index * 4);
  return { shape: [...tensor.shape], values };
}

function sameVector(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
