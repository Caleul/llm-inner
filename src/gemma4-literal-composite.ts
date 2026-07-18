import {
  buildGemma4VisionAttentionMasks,
  flattenGemma4VideoInput,
  replaceGemma4MultimodalIdsWithPad,
  type Gemma4CompositeExecutionRequest,
} from "./gemma4-composite.js";
import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { executeGemma4LiteralAudioF32 } from "./gemma4-literal-audio.js";
import {
  executeGemma4LiteralGenerationProgram,
  type Gemma4LiteralGenerationExecutionResult,
} from "./gemma4-literal-generation.js";
import { executeGemma4LiteralVisionF32 } from "./gemma4-literal-vision.js";
import {
  executeGemma4PagedTextInputEmbeddingsLiteralF32,
  executeGemma4PagedTextLiteralF32WithPreparedPrelude,
  executeGemma4PagedTextProjectionPreludeLiteralF32,
  type Gemma4PagedTextOptions,
} from "./gemma4-paged-text.js";
import { scatterGemma4AudioFeaturesF32 } from "./gemma4-audio.js";
import { scatterGemma4ImageFeaturesF32 } from "./gemma4-vision.js";
import type { DenseF32Tensor, ReferenceF32ExecutionResult } from "./types.js";

export interface Gemma4LiteralCompositeExecutionOptions extends Gemma4PagedTextOptions {
  /** Largest fully decoded tensor used by either multimodal tower. */
  maxTowerTensorBytes?: number;
}

export interface Gemma4LiteralCompositeExecutionResult {
  values: ReadonlyMap<string, DenseF32Tensor>;
  llmInputIds: number[][];
  text: ReferenceF32ExecutionResult;
}

export interface Gemma4LiteralCompositeGenerationRequest extends Omit<Gemma4CompositeExecutionRequest, "tensors"> {
  maxNewTokens: number;
  eosTokenId?: number;
}

export interface Gemma4LiteralCompositeGenerationResult extends Gemma4LiteralGenerationExecutionResult {
  compositePrefill: Gemma4LiteralCompositeExecutionResult;
}

/**
 * Executes the complete serialized Gemma 4 composite forward from embedded
 * storage. The only accepted runtime values are inputs declared by the
 * artifact; neither a tensor catalog nor a checkpoint reader is available.
 */
export async function executeGemma4LiteralCompositeF32(
  artifact: OpenGemma4CompositeLiteralArtifact,
  request: Omit<Gemma4CompositeExecutionRequest, "tensors">,
  options: Gemma4LiteralCompositeExecutionOptions = {},
): Promise<Gemma4LiteralCompositeExecutionResult> {
  validateInputIds(request.inputIds);
  if (request.mmTokenTypeIds !== undefined && request.attentionMask !== undefined) {
    throw new Error("Gemma 4 literal composite não combina mm_token_type_ids com attentionMask 4-D fornecida.");
  }
  if (request.mmTokenTypeIds !== undefined && request.pastKeyValues !== undefined) {
    throw new Error("Gemma 4 literal composite aceita mm_token_type_ids somente no prefill sem cache.");
  }
  const towerLimit = options.maxTowerTensorBytes ?? 64 * 1024 * 1024;
  if (!Number.isSafeInteger(towerLimit) || towerLimit <= 0) throw new Error("Gemma 4 literal composite requer maxTowerTensorBytes positivo seguro.");
  const text = artifact.program.textProgram;
  const pad = text.config.pad_token_id;
  if (!Number.isSafeInteger(pad) || (pad as number) < 0) throw new Error("Gemma 4 literal composite requer pad_token_id inteiro não negativo.");
  const llmInputIds = replaceGemma4MultimodalIdsWithPad(request.inputIds, artifact.program.contract, pad as number);
  const values = new Map(await executeGemma4PagedTextInputEmbeddingsLiteralF32(artifact, llmInputIds, options));
  let embeddings = required(values, "hidden_states_0");
  values.set("composite_text_embeddings", embeddings);

  assertPair(request.pixelValues, request.imagePositionIds, "pixel_values", "image_position_ids");
  if (request.pixelValues && request.imagePositionIds) {
    const image = await executeGemma4LiteralVisionF32(artifact, {
      pixelValues: request.pixelValues,
      pixelPositionIds: request.imagePositionIds,
    }, { maxTensorBytes: towerLimit });
    merge(values, image.values);
    values.set("image_features", image.imageFeatures);
    embeddings = scatterGemma4ImageFeaturesF32(embeddings, request.inputIds, artifact.program.contract.modalities.imageTokenId, image.imageFeatures);
  }
  values.set("composite_embeddings_after_image", embeddings);

  assertPair(request.pixelValuesVideos, request.videoPositionIds, "pixel_values_videos", "video_position_ids");
  if (request.pixelValuesVideos && request.videoPositionIds) {
    const videoTokenId = artifact.program.contract.modalities.videoTokenId;
    if (videoTokenId === undefined) throw new Error("Gemma 4 literal composite recebeu vídeo sem video_token_id declarado.");
    const flattened = flattenGemma4VideoInput(request.pixelValuesVideos, request.videoPositionIds);
    values.set("composite_video_pixels", flattened.pixels);
    values.set("composite_video_position_ids", positionsTensor(flattened.positions));
    const video = await executeGemma4LiteralVisionF32(artifact, {
      pixelValues: flattened.pixels,
      pixelPositionIds: flattened.positions,
    }, { maxTensorBytes: towerLimit });
    merge(values, prefixValues(video.values, "video_"));
    values.set("video_features", video.imageFeatures);
    embeddings = scatterGemma4ImageFeaturesF32(embeddings, request.inputIds, videoTokenId, video.imageFeatures);
  }
  values.set("composite_embeddings_after_video", embeddings);

  assertPair(request.inputFeatures, request.inputFeaturesMask, "input_features", "input_features_mask");
  if (request.inputFeatures && request.inputFeaturesMask) {
    const audio = await executeGemma4LiteralAudioF32(artifact, {
      inputFeatures: request.inputFeatures,
      inputFeaturesMask: request.inputFeaturesMask,
    }, { maxTensorBytes: towerLimit });
    merge(values, audio.values);
    values.set("audio_features", audio.audioFeatures);
    embeddings = scatterGemma4AudioFeaturesF32(embeddings, request.inputIds, artifact.program.contract.modalities.audioTokenId, audio.audioFeatures);
  }
  values.set("hidden_states_0", embeddings);

  const prepared = new Map(await executeGemma4PagedTextProjectionPreludeLiteralF32(artifact, llmInputIds, values, options));
  const masks = request.mmTokenTypeIds === undefined
    ? undefined
    : buildGemma4VisionAttentionMasks(request.mmTokenTypeIds, request.inputIds, text);
  if (masks) {
    prepared.set("vision_block_sequence_ids", masks.blockSequenceIds);
    prepared.set("full_attention_mask", masks.full);
    prepared.set("sliding_attention_mask", masks.sliding);
  }
  const executedText = await executeGemma4PagedTextLiteralF32WithPreparedPrelude(artifact, {
    inputIds: request.inputIds,
    ...(request.positionIds ? { positionIds: request.positionIds } : {}),
    ...(request.attentionMask ? { attentionMask: request.attentionMask } : {}),
    ...(masks ? { attentionMasksByLayer: masks.byLayer } : {}),
    ...(request.pastKeyValues ? { pastKeyValues: request.pastKeyValues } : {}),
  }, prepared, options);
  return { values: executedText.values, llmInputIds, text: executedText };
}

/** Executes the serialized greedy state machine with multimodal prefill only. */
export async function generateGemma4LiteralCompositeF32(
  artifact: OpenGemma4CompositeLiteralArtifact,
  request: Gemma4LiteralCompositeGenerationRequest,
  options: Gemma4LiteralCompositeExecutionOptions = {},
): Promise<Gemma4LiteralCompositeGenerationResult> {
  let compositePrefill: Gemma4LiteralCompositeExecutionResult | undefined;
  const generation = await executeGemma4LiteralGenerationProgram(artifact.program, artifact.generation, request, {
    prefill: async (prefill) => {
      compositePrefill = await executeGemma4LiteralCompositeF32(artifact, prefill, options);
      return compositePrefill.text;
    },
    incremental: async (incremental) => (await executeGemma4LiteralCompositeF32(artifact, incremental, options)).text,
  });
  if (!compositePrefill) throw new Error("Programa literal composite não produziu o prefill multimodal.");
  return { ...generation, compositePrefill };
}

function validateInputIds(inputIds: number[][]): void {
  if (inputIds.length === 0 || inputIds.some((row) => row.length === 0 || row.length !== inputIds[0]!.length || row.some((id) => !Number.isSafeInteger(id) || id < 0))) {
    throw new Error("Gemma 4 literal composite requer input_ids não vazio, retangular e inteiro não negativo.");
  }
}

function assertPair(left: unknown, right: unknown, leftName: string, rightName: string): void {
  if ((left === undefined) !== (right === undefined)) throw new Error(`Gemma 4 literal composite requer ${leftName} e ${rightName} juntos.`);
}

function required(values: ReadonlyMap<string, DenseF32Tensor>, name: string): DenseF32Tensor {
  const found = values.get(name);
  if (!found) throw new Error(`Gemma 4 literal composite não encontrou intermediário ${name}.`);
  return found;
}

function positionsTensor(positions: number[][][]): DenseF32Tensor {
  return { shape: [positions.length, positions[0]?.length ?? 0, 2], values: Float32Array.from(positions.flat(2)) };
}

function merge(target: Map<string, DenseF32Tensor>, source: ReadonlyMap<string, DenseF32Tensor>): void {
  for (const [name, value] of source) target.set(name, value);
}

function prefixValues(values: ReadonlyMap<string, DenseF32Tensor>, prefix: string): Map<string, DenseF32Tensor> {
  return new Map([...values].map(([name, value]) => [`${prefix}${name}`, value]));
}
