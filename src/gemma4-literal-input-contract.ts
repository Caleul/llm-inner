import { isDeepStrictEqual } from "node:util";
import type {
  Gemma4CompositeExecutionRequest,
  Gemma4CompositeProgram,
} from "./gemma4-composite.js";
import type { DenseF32Tensor, ReferenceF32KeyValueCache } from "./types.js";

export type Gemma4LiteralInputName =
  | "input_ids"
  | "position_ids"
  | "attention_mask"
  | "past_key_values"
  | "pixel_values"
  | "image_position_ids"
  | "pixel_values_videos"
  | "video_position_ids"
  | "input_features"
  | "input_features_mask"
  | "mm_token_type_ids"
  | "max_new_tokens"
  | "eos_token_id";

export interface Gemma4LiteralInputDeclaration {
  name: Gemma4LiteralInputName;
  requiredFor: Array<"forward" | "generation">;
  representation:
    | "rectangular-i32-array"
    | "rectangular-bool-array"
    | "nested-i32-coordinate-array"
    | "dense-f32-tensor"
    | "producer-owned-kv-map"
    | "safe-i32-scalar";
  rank?: number;
  axes?: string[];
}

export interface Gemma4LiteralCacheProducerInputContract {
  layer: number;
  keyValueHeads: number;
  headDim: number;
  layout: "BHSD";
  ownership: "producer-only";
}

/**
 * Executable input boundary for the serialized composite forward. Every
 * number here is derived from the registered Gemma 4 program, so source-
 * removed replay does not recover tensor ranks or modality cardinality from
 * TypeScript conventions.
 */
export interface Gemma4LiteralInputContract {
  kind: "gemma4-literal-input-contract";
  schemaVersion: 1;
  declarations: Gemma4LiteralInputDeclaration[];
  text: {
    vocabSize: number;
    attentionHeads: number;
    modalTokenIds: {
      image: number;
      video?: number;
      audio: number;
    };
    inputIds: {
      shape: "non-empty-rectangular-[batch,sequence]";
      values: "safe-non-negative-integer-and-(less-than-vocab-or-modal-token)";
    };
    positionIds: {
      shape: "same-[batch,sequence]-as-input_ids";
      values: "safe-non-negative-integer";
    };
    mmTokenTypeIds: {
      shape: "same-[batch,sequence]-as-input_ids";
      values: "safe-integer";
    };
    attentionMask: {
      layout: "[batch,1|attention_heads,query_sequence,past_sequence+query_sequence]";
      values: "finite-or-negative-infinity";
    };
    cacheProducers: Gemma4LiteralCacheProducerInputContract[];
  };
  vision: {
    pixelFeatures: number;
    positionEmbeddingSize: number;
    poolingKernelSize: number;
    pixelValues: "dense-f32-[items,patches,pixel_features]";
    positions: "i32-[items,patches,2]-with-only-[-1,-1]-padding";
    pooledPlaceholderCardinality: "stable-valid-pool-cells-equals-modal-token-count";
  };
  video: {
    pixelValues: "dense-f32-[videos,frames,patches,pixel_features]";
    positions: "i32-[videos,frames,patches,2]-with-only-[-1,-1]-padding";
    flattenOrder: "video-then-frame-row-major";
    pooledPlaceholderCardinality: "stable-valid-pool-cells-equals-modal-token-count";
  };
  audio: {
    inputFeatures: "dense-f32-[batch,frames,features]";
    inputMask: "bool-[batch,frames]";
    subsamplingStrides: [2, 2];
    finalSubsamplingChannels: number;
    projectionInputFeatures: number;
    featureWidthConstraint: "ceil(ceil(features/2)/2)*final_subsampling_channels==projection_input_features";
    placeholderCardinality: "stable-true-count-after-two-stride-2-slices-equals-audio-token-count";
  };
  presence: {
    paired: Array<readonly [Gemma4LiteralInputName, Gemma4LiteralInputName]>;
    mmTokenTypeIdsForbids: ["attention_mask", "past_key_values"];
  };
}

const DECLARATIONS: Gemma4LiteralInputDeclaration[] = [
  { name: "input_ids", requiredFor: ["forward", "generation"], representation: "rectangular-i32-array", rank: 2, axes: ["batch", "sequence"] },
  { name: "position_ids", requiredFor: [], representation: "rectangular-i32-array", rank: 2, axes: ["batch", "sequence"] },
  { name: "attention_mask", requiredFor: [], representation: "dense-f32-tensor", rank: 4, axes: ["batch", "head", "query_sequence", "key_sequence"] },
  { name: "past_key_values", requiredFor: [], representation: "producer-owned-kv-map" },
  { name: "pixel_values", requiredFor: [], representation: "dense-f32-tensor", rank: 3, axes: ["image", "patch", "pixel_feature"] },
  { name: "image_position_ids", requiredFor: [], representation: "nested-i32-coordinate-array", rank: 3, axes: ["image", "patch", "xy"] },
  { name: "pixel_values_videos", requiredFor: [], representation: "dense-f32-tensor", rank: 4, axes: ["video", "frame", "patch", "pixel_feature"] },
  { name: "video_position_ids", requiredFor: [], representation: "nested-i32-coordinate-array", rank: 4, axes: ["video", "frame", "patch", "xy"] },
  { name: "input_features", requiredFor: [], representation: "dense-f32-tensor", rank: 3, axes: ["batch", "frame", "audio_feature"] },
  { name: "input_features_mask", requiredFor: [], representation: "rectangular-bool-array", rank: 2, axes: ["batch", "frame"] },
  { name: "mm_token_type_ids", requiredFor: [], representation: "rectangular-i32-array", rank: 2, axes: ["batch", "sequence"] },
  { name: "max_new_tokens", requiredFor: ["generation"], representation: "safe-i32-scalar", rank: 0, axes: [] },
  { name: "eos_token_id", requiredFor: [], representation: "safe-i32-scalar", rank: 0, axes: [] },
];

export function buildGemma4LiteralInputContract(program: Gemma4CompositeProgram): Gemma4LiteralInputContract {
  const attention = program.textProgram.layers.flatMap((layer) => layer.operations)
    .filter((operation) => operation.op === "scaled_dot_product_attention");
  const cacheProducers = attention
    .filter((operation) => !operation.kvSharing?.enabled)
    .map((operation): Gemma4LiteralCacheProducerInputContract => {
      if (operation.layer === undefined) throw new Error(`${operation.id}: contrato de input KV requer layer explícita.`);
      return {
        layer: operation.layer,
        keyValueHeads: operation.numKeyValueHeads,
        headDim: operation.headDim,
        layout: "BHSD",
        ownership: "producer-only",
      };
    });
  if (attention.length === 0 || cacheProducers.length === 0 || new Set(cacheProducers.map((entry) => entry.layer)).size !== cacheProducers.length) {
    throw new Error("Contrato de input Gemma 4 requer produtores KV textuais únicos.");
  }
  const inputProjection = program.audioProgram.assignments.find((assignment) => assignment.id === "audio_input_projection")?.tensors?.[0];
  const projectionInputFeatures = inputProjection?.shape[1];
  const finalSubsamplingChannels = program.audioProgram.tower.subsamplingChannels[1];
  if (!Number.isSafeInteger(projectionInputFeatures) || (projectionInputFeatures ?? 0) <= 0 || !Number.isSafeInteger(finalSubsamplingChannels) || finalSubsamplingChannels <= 0) {
    throw new Error("Contrato de input Gemma 4 não derivou a largura pós-subsampling de áudio.");
  }
  const pixelFeatures = 3 * program.visionProgram.tower.patchSize ** 2;
  return {
    kind: "gemma4-literal-input-contract",
    schemaVersion: 1,
    declarations: structuredClone(DECLARATIONS),
    text: {
      vocabSize: program.contract.text.vocabSize,
      attentionHeads: program.contract.text.attentionHeads,
      modalTokenIds: {
        image: program.contract.modalities.imageTokenId,
        ...(program.contract.modalities.videoTokenId === undefined ? {} : { video: program.contract.modalities.videoTokenId }),
        audio: program.contract.modalities.audioTokenId,
      },
      inputIds: {
        shape: "non-empty-rectangular-[batch,sequence]",
        values: "safe-non-negative-integer-and-(less-than-vocab-or-modal-token)",
      },
      positionIds: { shape: "same-[batch,sequence]-as-input_ids", values: "safe-non-negative-integer" },
      mmTokenTypeIds: { shape: "same-[batch,sequence]-as-input_ids", values: "safe-integer" },
      attentionMask: {
        layout: "[batch,1|attention_heads,query_sequence,past_sequence+query_sequence]",
        values: "finite-or-negative-infinity",
      },
      cacheProducers,
    },
    vision: {
      pixelFeatures,
      positionEmbeddingSize: program.visionProgram.tower.positionEmbeddingSize,
      poolingKernelSize: program.visionProgram.tower.poolingKernelSize,
      pixelValues: "dense-f32-[items,patches,pixel_features]",
      positions: "i32-[items,patches,2]-with-only-[-1,-1]-padding",
      pooledPlaceholderCardinality: "stable-valid-pool-cells-equals-modal-token-count",
    },
    video: {
      pixelValues: "dense-f32-[videos,frames,patches,pixel_features]",
      positions: "i32-[videos,frames,patches,2]-with-only-[-1,-1]-padding",
      flattenOrder: "video-then-frame-row-major",
      pooledPlaceholderCardinality: "stable-valid-pool-cells-equals-modal-token-count",
    },
    audio: {
      inputFeatures: "dense-f32-[batch,frames,features]",
      inputMask: "bool-[batch,frames]",
      subsamplingStrides: [2, 2],
      finalSubsamplingChannels,
      projectionInputFeatures: projectionInputFeatures!,
      featureWidthConstraint: "ceil(ceil(features/2)/2)*final_subsampling_channels==projection_input_features",
      placeholderCardinality: "stable-true-count-after-two-stride-2-slices-equals-audio-token-count",
    },
    presence: {
      paired: [
        ["pixel_values", "image_position_ids"],
        ["pixel_values_videos", "video_position_ids"],
        ["input_features", "input_features_mask"],
      ],
      mmTokenTypeIdsForbids: ["attention_mask", "past_key_values"],
    },
  };
}

export function validateGemma4LiteralInputContract(
  contract: Gemma4LiteralInputContract,
  program: Gemma4CompositeProgram,
): void {
  if (!isDeepStrictEqual(contract, buildGemma4LiteralInputContract(program))) {
    throw new Error("Programa literal Gemma 4 possui contrato de inputs, shapes ou cardinalidade divergente.");
  }
}

/** Binds the human-facing input catalog to the executable declarations. */
export function validateGemma4LiteralInputDeclarationAlignment(
  contract: Gemma4LiteralInputContract,
  inputs: ReadonlyArray<{ name: string; requiredFor: Array<"forward" | "generation"> }>,
): void {
  const actual = inputs.map((input) => ({ name: input.name, requiredFor: input.requiredFor }));
  const expected = contract.declarations.map((input) => ({ name: input.name, requiredFor: input.requiredFor }));
  if (!isDeepStrictEqual(actual, expected)) {
    throw new Error("Programa literal Gemma 4 desalinhou o catálogo de inputs do contrato executável.");
  }
}

/** Validates a concrete forward invocation solely against the serialized contract. */
export function executeGemma4LiteralInputContract(
  contract: Gemma4LiteralInputContract,
  program: Gemma4CompositeProgram,
  request: Omit<Gemma4CompositeExecutionRequest, "tensors">,
): void {
  validateGemma4LiteralInputContract(contract, program);
  const [batch, sequence] = validateInputIds(contract, request.inputIds);
  validatePairedPresence(contract, request);
  if (request.positionIds !== undefined) validateI32Matrix(request.positionIds, batch, sequence, "position_ids", true);
  if (request.mmTokenTypeIds !== undefined) validateI32Matrix(request.mmTokenTypeIds, batch, sequence, "mm_token_type_ids", false);
  if (request.mmTokenTypeIds !== undefined && (request.attentionMask !== undefined || request.pastKeyValues !== undefined)) {
    throw new Error(`Contrato de input Gemma 4 não combina mm_token_type_ids com ${request.attentionMask !== undefined ? "attention_mask" : "past_key_values"}.`);
  }
  const pastSequence = validateCache(contract, request.pastKeyValues, batch);
  if (request.attentionMask !== undefined) validateAttentionMask(contract, request.attentionMask, batch, sequence, pastSequence);
  if (request.pixelValues !== undefined && request.imagePositionIds !== undefined) {
    validateVisionInput(contract, request.pixelValues, request.imagePositionIds, "image");
    validatePlaceholderCount(request.inputIds, contract.text.modalTokenIds.image,
      countVisionFeatures(contract, request.imagePositionIds), "image");
  }
  if (request.pixelValuesVideos !== undefined && request.videoPositionIds !== undefined) {
    validateVideoInput(contract, request.pixelValuesVideos, request.videoPositionIds);
    const positions = request.videoPositionIds.flatMap((video) => video);
    validatePlaceholderCount(request.inputIds, requiredVideoToken(contract), countVisionFeatures(contract, positions), "video");
  }
  if (request.inputFeatures !== undefined && request.inputFeaturesMask !== undefined) {
    validateAudioInput(contract, request.inputFeatures, request.inputFeaturesMask);
    validatePlaceholderCount(request.inputIds, contract.text.modalTokenIds.audio,
      countSubsampledAudioFeatures(contract, request.inputFeaturesMask), "audio");
  }
}

function validateInputIds(contract: Gemma4LiteralInputContract, inputIds: number[][]): [number, number] {
  const batch = inputIds.length;
  const sequence = inputIds[0]?.length ?? 0;
  const modal = new Set<number>([
    contract.text.modalTokenIds.image,
    contract.text.modalTokenIds.audio,
    ...(contract.text.modalTokenIds.video === undefined ? [] : [contract.text.modalTokenIds.video]),
  ]);
  const valid = batch > 0 && sequence > 0 && inputIds.every((row) => row.length === sequence && row.every((token) =>
    Number.isSafeInteger(token) && token >= 0 && (token < contract.text.vocabSize || modal.has(token))));
  if (!valid) throw new Error("Contrato de input Gemma 4 requer input_ids retangular, não vazio e dentro do vocabulário/modal tokens.");
  return [batch, sequence];
}

function validatePairedPresence(
  contract: Gemma4LiteralInputContract,
  request: Omit<Gemma4CompositeExecutionRequest, "tensors">,
): void {
  const values: Record<Gemma4LiteralInputName, unknown> = {
    input_ids: request.inputIds,
    position_ids: request.positionIds,
    attention_mask: request.attentionMask,
    past_key_values: request.pastKeyValues,
    pixel_values: request.pixelValues,
    image_position_ids: request.imagePositionIds,
    pixel_values_videos: request.pixelValuesVideos,
    video_position_ids: request.videoPositionIds,
    input_features: request.inputFeatures,
    input_features_mask: request.inputFeaturesMask,
    mm_token_type_ids: request.mmTokenTypeIds,
    max_new_tokens: undefined,
    eos_token_id: undefined,
  };
  for (const [left, right] of contract.presence.paired) {
    if ((values[left] === undefined) !== (values[right] === undefined)) {
      throw new Error(`Contrato de input Gemma 4 requer ${left} e ${right} juntos, ou ambos ausentes.`);
    }
  }
}

function validateI32Matrix(
  values: number[][],
  batch: number,
  sequence: number,
  name: string,
  nonNegative: boolean,
): void {
  if (values.length !== batch || values.some((row) => row.length !== sequence || row.some((value) =>
    !Number.isSafeInteger(value) || nonNegative && value < 0))) {
    throw new Error(`Contrato de input Gemma 4 requer ${name} I32 com shape [${batch},${sequence}].`);
  }
}

function validateDenseTensor(tensor: DenseF32Tensor, rank: number, name: string): void {
  const validShape = tensor.shape.length === rank && tensor.shape.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0);
  const size = validShape ? tensor.shape.reduce((total, dimension) => total * dimension, 1) : -1;
  if (!validShape || !Number.isSafeInteger(size) || size !== tensor.values.length) {
    throw new Error(`Contrato de input Gemma 4 requer ${name} F32 rank ${rank} com buffer completo.`);
  }
}

function validateAttentionMask(
  contract: Gemma4LiteralInputContract,
  mask: DenseF32Tensor,
  batch: number,
  sequence: number,
  pastSequence: number,
): void {
  validateDenseTensor(mask, 4, "attention_mask");
  const [maskBatch, heads, query, key] = mask.shape;
  if (maskBatch !== batch || (heads !== 1 && heads !== contract.text.attentionHeads) || query !== sequence || key !== pastSequence + sequence ||
    [...mask.values].some((value) => Number.isNaN(value) || value === Infinity)) {
    throw new Error(`Contrato de input Gemma 4 requer attention_mask [${batch},1|${contract.text.attentionHeads},${sequence},${pastSequence + sequence}] finita ou -Infinity.`);
  }
}

function validateCache(
  contract: Gemma4LiteralInputContract,
  cache: ReadonlyMap<number, ReferenceF32KeyValueCache> | undefined,
  batch: number,
): number {
  if (cache === undefined) return 0;
  if (!(cache instanceof Map) || cache.size !== contract.text.cacheProducers.length) {
    throw new Error("Contrato de input Gemma 4 requer exatamente um cache por produtor KV.");
  }
  let pastSequence: number | undefined;
  for (const producer of contract.text.cacheProducers) {
    const entry = cache.get(producer.layer);
    if (!entry) throw new Error(`Contrato de input Gemma 4 não encontrou cache do produtor ${producer.layer}.`);
    for (const [kind, tensor] of [["key", entry.key], ["value", entry.value]] as const) {
      validateDenseTensor(tensor, 4, `past_key_values[${producer.layer}].${kind}`);
      const [cacheBatch, heads, sequence, headDim] = tensor.shape;
      if (cacheBatch !== batch || heads !== producer.keyValueHeads || headDim !== producer.headDim) {
        throw new Error(`Contrato de input Gemma 4 encontrou shape KV incompatível no produtor ${producer.layer}.`);
      }
      if (pastSequence === undefined) pastSequence = sequence;
      else if (sequence !== pastSequence) throw new Error("Contrato de input Gemma 4 requer a mesma sequência passada em todos os caches produtores.");
    }
    if (entry.key.shape[2] !== entry.value.shape[2]) throw new Error(`Contrato de input Gemma 4 requer K/V alinhados no produtor ${producer.layer}.`);
  }
  for (const layer of cache.keys()) if (!contract.text.cacheProducers.some((producer) => producer.layer === layer)) {
    throw new Error(`Contrato de input Gemma 4 rejeitou cache sem propriedade no layer ${layer}.`);
  }
  return pastSequence ?? 0;
}

function validateVisionInput(
  contract: Gemma4LiteralInputContract,
  pixels: DenseF32Tensor,
  positions: number[][][],
  label: "image" | "video",
): void {
  validateDenseTensor(pixels, 3, `${label}_pixel_values`);
  if (pixels.shape[2] !== contract.vision.pixelFeatures || positions.length !== pixels.shape[0] ||
    positions.some((row) => row.length !== pixels.shape[1])) {
    throw new Error(`Contrato de input Gemma 4 requer ${label} pixels [items,patches,${contract.vision.pixelFeatures}] e posições correspondentes.`);
  }
  validateVisionPositions(contract, positions, label);
}

function validateVideoInput(
  contract: Gemma4LiteralInputContract,
  pixels: DenseF32Tensor,
  positions: number[][][][],
): void {
  validateDenseTensor(pixels, 4, "pixel_values_videos");
  const [videos, frames, patches, features] = pixels.shape;
  if (features !== contract.vision.pixelFeatures || positions.length !== videos || positions.some((video) =>
    video.length !== frames || video.some((frame) => frame.length !== patches))) {
    throw new Error(`Contrato de input Gemma 4 requer vídeo [videos,frames,patches,${contract.vision.pixelFeatures}] e posições correspondentes.`);
  }
  validateVisionPositions(contract, positions.flatMap((video) => video), "video");
}

function validateVisionPositions(
  contract: Gemma4LiteralInputContract,
  positions: number[][][],
  label: "image" | "video",
): void {
  const kernelArea = contract.vision.poolingKernelSize ** 2;
  for (const row of positions) {
    if (row.length === 0 || row.length % kernelArea !== 0) {
      throw new Error(`Contrato de input Gemma 4 requer patches ${label} divisíveis por pooling_kernel_size^2.`);
    }
    let valid = 0;
    for (const position of row) {
      const [x, y] = position;
      const padding = x === -1 && y === -1;
      if (position.length !== 2 || !Number.isSafeInteger(x) || !Number.isSafeInteger(y) ||
        !padding && (x! < 0 || y! < 0 || x! >= contract.vision.positionEmbeddingSize || y! >= contract.vision.positionEmbeddingSize)) {
        throw new Error(`Contrato de input Gemma 4 requer posições ${label} I32 válidas; padding é somente [-1,-1].`);
      }
      if (!padding) valid += 1;
    }
    if (valid === 0) throw new Error(`Contrato de input Gemma 4 requer ao menos um patch ${label} válido por item.`);
  }
}

function countVisionFeatures(contract: Gemma4LiteralInputContract, positions: number[][][]): number {
  let count = 0;
  const kernel = contract.vision.poolingKernelSize;
  for (const row of positions) {
    const poolCells = row.length / (kernel * kernel);
    const validXs = row.filter(([x, y]) => x !== -1 || y !== -1).map(([x]) => x!);
    const maxX = Math.max(...validXs) + 1;
    const occupied = new Set<number>();
    for (const [x, y] of row) {
      if (x === -1 && y === -1) continue;
      const slot = Math.floor(x! / kernel) + Math.floor(maxX / kernel) * Math.floor(y! / kernel);
      if (!Number.isSafeInteger(slot) || slot < 0 || slot >= poolCells) {
        throw new Error(`Contrato de input Gemma 4 produziu pool slot ${slot} fora de 0..${poolCells - 1}.`);
      }
      occupied.add(slot);
    }
    count += occupied.size;
  }
  return count;
}

function validateAudioInput(
  contract: Gemma4LiteralInputContract,
  features: DenseF32Tensor,
  mask: boolean[][],
): void {
  validateDenseTensor(features, 3, "input_features");
  const [batch, frames, width] = features.shape;
  if (mask.length !== batch || mask.some((row) => row.length !== frames || row.some((value) => typeof value !== "boolean"))) {
    throw new Error("Contrato de input Gemma 4 requer input_features_mask BOOL [batch,frames].");
  }
  const downsampledWidth = Math.ceil(Math.ceil(width! / contract.audio.subsamplingStrides[0]) / contract.audio.subsamplingStrides[1]);
  if (downsampledWidth * contract.audio.finalSubsamplingChannels !== contract.audio.projectionInputFeatures) {
    throw new Error("Contrato de input Gemma 4 encontrou largura de áudio incompatível com os dois subsamplings e input projection.");
  }
}

function countSubsampledAudioFeatures(contract: Gemma4LiteralInputContract, mask: boolean[][]): number {
  let current = mask.map((row) => [...row]);
  for (const stride of contract.audio.subsamplingStrides) current = current.map((row) => row.filter((_, index) => index % stride === 0));
  return current.flat().filter(Boolean).length;
}

function validatePlaceholderCount(inputIds: number[][], tokenId: number, features: number, label: string): void {
  const placeholders = inputIds.flat().filter((token) => token === tokenId).length;
  if (placeholders !== features) {
    throw new Error(`Contrato de input Gemma 4 ${label} placeholder count=${placeholders} não corresponde a features=${features}.`);
  }
}

function requiredVideoToken(contract: Gemma4LiteralInputContract): number {
  const token = contract.text.modalTokenIds.video;
  if (token === undefined) throw new Error("Contrato de input Gemma 4 recebeu vídeo sem video_token_id.");
  return token;
}
