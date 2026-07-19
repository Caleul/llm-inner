import {
  buildGemma4VisionAttentionMasks,
  flattenGemma4VideoInput,
  replaceGemma4MultimodalIdsWithPad,
  type Gemma4CompositeExecutionRequest,
} from "./gemma4-composite.js";
import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import {
  executeGemma4LiteralForwardControlProgram,
  gemma4CompositeRequestInputPresence,
} from "./gemma4-literal-forward-control.js";
import { executeGemma4LiteralInputContract } from "./gemma4-literal-input-contract.js";
import {
  executeGemma4LiteralForwardOutputContract,
  executeGemma4LiteralGenerationOutputContract,
} from "./gemma4-literal-output-contract.js";
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
  executeGemma4LiteralInputContract(artifact.inputContract, artifact.program, request);
  const selection = executeGemma4LiteralForwardControlProgram(
    artifact.forwardControl,
    artifact.program,
    gemma4CompositeRequestInputPresence(request),
  );
  const towerLimit = options.maxTowerTensorBytes ?? 64 * 1024 * 1024;
  if (!Number.isSafeInteger(towerLimit) || towerLimit <= 0) throw new Error("Gemma 4 literal composite requer maxTowerTensorBytes positivo seguro.");
  const text = artifact.program.textProgram;
  const pad = text.config.pad_token_id;
  if (!Number.isSafeInteger(pad) || (pad as number) < 0) throw new Error("Gemma 4 literal composite requer pad_token_id inteiro não negativo.");
  const llmInputIds = replaceGemma4MultimodalIdsWithPad(request.inputIds, artifact.program.contract, pad as number);
  const values = new Map(await executeGemma4PagedTextInputEmbeddingsLiteralF32(artifact, llmInputIds, options));
  let embeddings = required(values, "hidden_states_0");
  values.set("composite_text_embeddings", embeddings);

  if (selection.modalities.image) {
    const image = await executeGemma4LiteralVisionF32(artifact, {
      pixelValues: request.pixelValues!,
      pixelPositionIds: request.imagePositionIds!,
    }, { maxTensorBytes: towerLimit });
    merge(values, image.values);
    values.set("image_features", image.imageFeatures);
    embeddings = scatterGemma4ImageFeaturesF32(embeddings, request.inputIds, artifact.program.contract.modalities.imageTokenId, image.imageFeatures);
  }
  values.set("composite_embeddings_after_image", embeddings);

  if (selection.modalities.video) {
    const videoTokenId = artifact.program.contract.modalities.videoTokenId;
    if (videoTokenId === undefined) throw new Error("Gemma 4 literal composite recebeu vídeo sem video_token_id declarado.");
    const flattened = flattenGemma4VideoInput(request.pixelValuesVideos!, request.videoPositionIds!);
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

  if (selection.modalities.audio) {
    const audio = await executeGemma4LiteralAudioF32(artifact, {
      inputFeatures: request.inputFeatures!,
      inputFeaturesMask: request.inputFeaturesMask!,
    }, { maxTensorBytes: towerLimit });
    merge(values, audio.values);
    values.set("audio_features", audio.audioFeatures);
    embeddings = scatterGemma4AudioFeaturesF32(embeddings, request.inputIds, artifact.program.contract.modalities.audioTokenId, audio.audioFeatures);
  }
  values.set("hidden_states_0", embeddings);

  const prepared = new Map(await executeGemma4PagedTextProjectionPreludeLiteralF32(artifact, llmInputIds, values, options));
  const masks = selection.visionMasks
    ? buildGemma4VisionAttentionMasks(request.mmTokenTypeIds!, request.inputIds, text)
    : undefined;
  if (masks) {
    prepared.set("vision_block_sequence_ids", masks.blockSequenceIds);
    prepared.set("full_attention_mask", masks.full);
    prepared.set("sliding_attention_mask", masks.sliding);
  }
  const selectedPositionIds = selection.positionIdsMode === "caller-i32-values"
    ? request.positionIds!
    : request.inputIds.map((row) => row.map((_, sequence) => sequence));
  const executedText = await executeGemma4PagedTextLiteralF32WithPreparedPrelude(artifact, {
    inputIds: request.inputIds,
    positionIds: selectedPositionIds,
    ...(request.attentionMask ? { attentionMask: request.attentionMask } : {}),
    ...(masks ? { attentionMasksByLayer: masks.byLayer } : {}),
    ...(request.pastKeyValues ? { pastKeyValues: request.pastKeyValues } : {}),
  }, prepared, options);
  const result = { values: executedText.values, llmInputIds, text: executedText };
  executeGemma4LiteralForwardOutputContract(artifact.outputContract, artifact.program, request, result);
  return result;
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
  executeGemma4LiteralGenerationOutputContract(artifact.outputContract, artifact.program, request, {
    prefill: generation.prefill,
    generatedTokenIds: generation.generatedTokenIds,
    selectionLogits: generation.selectionLogits,
    stepPastKeyValues: generation.stepPastKeyValues,
    terminal: generation,
  });
  return { ...generation, compositePrefill };
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
