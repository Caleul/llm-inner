import { executeReferenceF32WithPreparedPrelude } from "./executor.js";
import { buildGemma4AudioProgram, executeGemma4AudioF32, scatterGemma4AudioFeaturesF32, type Gemma4AudioProgram } from "./gemma4-audio.js";
import { inspectGemma4PackageContract, type Gemma4PackageContract } from "./gemma4-contract.js";
import { buildGemma4TextIR } from "./gemma4-text.js";
import { buildGemma4VisionProgram, executeGemma4VisionF32, scatterGemma4ImageFeaturesF32, type Gemma4VisionProgram } from "./gemma4-vision.js";
import { selectGreedyToken } from "./generation.js";
import type { DenseF32Tensor, ModelCatalog, ModelIR, PreviewOptions, ReferenceF32ExecutionResult, ReferenceF32KeyValueCache, TensorInfo, TensorRef } from "./types.js";

const F32_POLICY = { inputDtype: "F32", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" } as const;

/**
 * Explicit outer-model boundary for the registered dense Gemma 4 package.
 * The tower programs retain their own assignments; this program records the
 * semantically significant order that connects them to Gemma4Text.
 */
export interface Gemma4CompositeProgram {
  kind: "gemma4-composite-prelude";
  sourceFormat: "safetensors";
  contract: Gemma4PackageContract;
  textProgram: ModelIR;
  visionProgram: Gemma4VisionProgram;
  audioProgram: Gemma4AudioProgram;
  assignments: Gemma4CompositeAssignment[];
  outputs: { embeddings: "hidden_states_0"; perLayerInputs: "ple_inputs"; logits: "softcapped_logits" | "logits" };
}

export interface Gemma4CompositeAssignment {
  id: string;
  operation:
    | "placeholder-masks"
    | "vision-block-sequence-ids"
    | "causal-attention-mask"
    | "vision-sliding-attention-mask"
    | "replace-multimodal-ids-with-pad"
    | "embedding"
    | "per-layer-embedding"
    | "vision-feature-program"
    | "video-frame-flatten"
    | "audio-feature-program"
    | "masked-scatter"
    | "linear"
    | "scale-f32"
    | "reshape-per-layer"
    | "rms-norm"
    | "add"
    | "text-core";
  inputs: string[];
  output: string;
  tensors?: TensorRef[];
  semantics?: string;
}

export interface Gemma4CompositeExecutionRequest {
  inputIds: number[][];
  tensors: ReadonlyMap<string, DenseF32Tensor>;
  positionIds?: number[][];
  attentionMask?: DenseF32Tensor;
  pastKeyValues?: ReadonlyMap<number, ReferenceF32KeyValueCache>;
  pixelValues?: DenseF32Tensor;
  imagePositionIds?: number[][][];
  /** [videos, frames, patches, 3 * patch_size^2]. */
  pixelValuesVideos?: DenseF32Tensor;
  /** [videos][frames][patches][x,y]. */
  videoPositionIds?: number[][][][];
  inputFeatures?: DenseF32Tensor;
  inputFeaturesMask?: boolean[][];
  /** Source mm_token_type_ids: 1=image and 2=video form bidirectional vision blocks. */
  mmTokenTypeIds?: number[][];
}

export interface Gemma4CompositeExecutionResult {
  values: ReadonlyMap<string, DenseF32Tensor>;
  llmInputIds: number[][];
  text: ReferenceF32ExecutionResult;
}

export interface Gemma4CompositeGenerationRequest extends Gemma4CompositeExecutionRequest {
  maxNewTokens: number;
  eosTokenId?: number;
}

export interface Gemma4CompositeGenerationResult {
  prefill: Gemma4CompositeExecutionResult;
  generatedTokenIds: number[];
  selectionLogits: DenseF32Tensor[];
  stepPastKeyValues: Array<ReadonlyMap<number, ReferenceF32KeyValueCache>>;
  text: ReferenceF32ExecutionResult;
}

/**
 * Builds only after the complete package contract has been validated. In
 * particular, this does not select an adapter from tensor-name fragments.
 */
export function buildGemma4CompositeProgram(catalog: ModelCatalog, preview: PreviewOptions): Gemma4CompositeProgram {
  const contract = inspectGemma4PackageContract(catalog);
  const textProgram = asF32ReferenceProgram(buildGemma4TextIR(catalog, textConfig(catalog), preview));
  const visionProgram = buildGemma4VisionProgram(catalog);
  const audioProgram = buildGemma4AudioProgram(catalog);
  const prefix = textPrefix(catalog);
  const refs = (names: readonly string[]): TensorRef[] => names.map((name) => tensorRef(requireDenseTensor(catalog, name)));
  const assignments: Gemma4CompositeAssignment[] = [
    { id: "composite_placeholder_masks", operation: "placeholder-masks", inputs: ["input_ids"], output: "composite_image_video_audio_masks", semantics: "input_ids == image_token_id, video_token_id, audio_token_id; masks remain independent for ordered scatter" },
    { id: "composite_block_sequence_ids", operation: "vision-block-sequence-ids", inputs: ["mm_token_type_ids"], output: "vision_block_sequence_ids", semantics: "source get_block_sequence_ids_for_mask: contiguous types 1=image or 2=video receive incrementing group IDs; every other type is -1" },
    { id: "composite_full_attention_mask", operation: "causal-attention-mask", inputs: ["vision_block_sequence_ids", "past_key_values"], output: "full_attention_mask", semantics: "Gemma 4 full_attention remains causal; vision blocks do not make full-attention layers bidirectional" },
    { id: "composite_sliding_attention_mask", operation: "vision-sliding-attention-mask", inputs: ["vision_block_sequence_ids", "past_key_values"], output: "sliding_attention_mask", semantics: "sliding_window AND (causal OR same non-negative vision block), matching create_masks_for_vision_model" },
    { id: "composite_pad_substitution", operation: "replace-multimodal-ids-with-pad", inputs: ["input_ids", "composite_image_video_audio_masks"], output: "composite_llm_input_ids", semantics: "replace every image/video/audio ID with text_config.pad_token_id before initial text embedding and PLE identity lookup" },
    { id: "composite_text_embedding", operation: "embedding", inputs: ["composite_llm_input_ids"], output: "composite_text_embeddings", tensors: refs([`${prefix}.embed_tokens.weight`]), semantics: "scaled Gemma4Text embedding of PAD-substituted ids" },
    { id: "composite_ple_identity", operation: "per-layer-embedding", inputs: ["composite_llm_input_ids"], output: "ple_token_identity", tensors: refs([`${prefix}.embed_tokens_per_layer.weight`]), semantics: "packed PLE identity uses PAD at all soft-token coordinates" },
    { id: "composite_image_features", operation: "vision-feature-program", inputs: ["pixel_values", "image_position_ids"], output: "image_features", semantics: "registered Gemma4Vision program, including pooling, multimodal RMSNorm and language projection" },
    { id: "composite_image_scatter", operation: "masked-scatter", inputs: ["composite_text_embeddings", "input_ids", "image_features"], output: "composite_embeddings_after_image", semantics: "replace only image_token_id values; cardinality is exact" },
    { id: "composite_video_flatten", operation: "video-frame-flatten", inputs: ["pixel_values_videos", "video_position_ids"], output: "composite_video_frames", semantics: "flatten videos and frames on dimensions 0 and 1 before the same vision tower" },
    { id: "composite_video_features", operation: "vision-feature-program", inputs: ["composite_video_frames"], output: "video_features", semantics: "same registered vision program after exact frame flattening" },
    { id: "composite_video_scatter", operation: "masked-scatter", inputs: ["composite_embeddings_after_image", "input_ids", "video_features"], output: "composite_embeddings_after_video", semantics: "replace only video_token_id values after images and before audio" },
    { id: "composite_audio_features", operation: "audio-feature-program", inputs: ["input_features", "input_features_mask"], output: "audio_features", semantics: "registered Gemma4Audio program, including valid-frame stripping and language projection" },
    { id: "composite_audio_scatter", operation: "masked-scatter", inputs: ["composite_embeddings_after_video", "input_ids", "audio_features"], output: "hidden_states_0", semantics: "replace only audio_token_id values after image/video; exact cardinality" },
    { id: "composite_ple_context_projection", operation: "linear", inputs: ["hidden_states_0"], output: "ple_context_packed", tensors: refs([`${prefix}.per_layer_model_projection.weight`]), semantics: "PLE context is derived after every modal scatter, not from pre-scatter PAD embeddings" },
    { id: "composite_ple_context_scale", operation: "scale-f32", inputs: ["ple_context_packed"], output: "ple_context_scaled", semantics: "multiply by hidden_size^-0.5" },
    { id: "composite_ple_context_reshape", operation: "reshape-per-layer", inputs: ["ple_context_scaled"], output: "ple_context_reshaped", semantics: "[B,S,L*P] -> [B,S,L,P] without reordering" },
    { id: "composite_ple_context_norm", operation: "rms-norm", inputs: ["ple_context_reshaped"], output: "ple_context_normalized", tensors: refs([`${prefix}.per_layer_projection_norm.weight`]), semantics: "Gemma4RMSNorm in F32 scalar reference mode" },
    { id: "composite_ple_combine", operation: "add", inputs: ["ple_context_normalized", "ple_token_identity"], output: "ple_combined" },
    { id: "composite_ple_combine_scale", operation: "scale-f32", inputs: ["ple_combined"], output: "ple_inputs", semantics: "multiply by 2^-0.5" },
    { id: "composite_text_core", operation: "text-core", inputs: ["hidden_states_0", "ple_inputs", "position_ids", "attention_mask", "past_key_values"], output: textProgram.epilogue.at(-1)?.output === "softcapped_logits" ? "softcapped_logits" : "logits", semantics: "prepared named outputs enter every registered Gemma4Text layer; no embedding or PLE prelude is rerun" },
  ];
  return { kind: "gemma4-composite-prelude", sourceFormat: "safetensors", contract, textProgram, visionProgram, audioProgram, assignments, outputs: { embeddings: "hidden_states_0", perLayerInputs: "ple_inputs", logits: textProgram.epilogue.at(-1)?.output === "softcapped_logits" ? "softcapped_logits" : "logits" } };
}

/** Executes the composite prelude then enters the existing Gemma4Text F32 path. */
export function executeGemma4CompositeF32(program: Gemma4CompositeProgram, request: Gemma4CompositeExecutionRequest): Gemma4CompositeExecutionResult {
  validateInputIds(request.inputIds);
  if (request.mmTokenTypeIds !== undefined && request.attentionMask !== undefined) throw new Error("Gemma 4 composite não combina mm_token_type_ids com attentionMask 4-D fornecida pelo chamador; o runtime autoritativo trata essa máscara como uma substituição já preparada.");
  if (request.mmTokenTypeIds !== undefined && request.pastKeyValues !== undefined) throw new Error("Gemma 4 composite requer mm_token_type_ids somente no prefill sem cache; o prepare_inputs_for_generation autoritativo os remove no decode incremental.");
  const llmInputIds = replaceModalIdsWithPad(request.inputIds, program.contract, textPadTokenId(program.textProgram));
  const prefix = textPrefixFromProgram(program.textProgram);
  const tokenEmbedding = embedding(llmInputIds, tensor(request.tensors, `${prefix}.embed_tokens.weight`), Math.sqrt(program.contract.text.hiddenSize));
  const pleIdentity = perLayerEmbedding(llmInputIds, tensor(request.tensors, `${prefix}.embed_tokens_per_layer.weight`), program.contract.text.layers, program.contract.text.perLayerInputSize, Math.sqrt(program.contract.text.perLayerInputSize));
  const values = new Map<string, DenseF32Tensor>();
  values.set("composite_text_embeddings", tokenEmbedding);
  values.set("ple_token_identity", pleIdentity);
  let embeddings = tokenEmbedding;

  if ((request.pixelValues === undefined) !== (request.imagePositionIds === undefined)) throw new Error("Gemma 4 composite requer pixel_values e image_position_ids juntos, ou ambos ausentes.");
  if (request.pixelValues !== undefined) {
    const image = executeGemma4VisionF32(program.visionProgram, { pixelValues: request.pixelValues, pixelPositionIds: request.imagePositionIds!, tensors: request.tensors });
    merge(values, image.values);
    values.set("image_features", image.imageFeatures);
    embeddings = scatterGemma4ImageFeaturesF32(embeddings, request.inputIds, program.contract.modalities.imageTokenId, image.imageFeatures);
  }
  values.set("composite_embeddings_after_image", embeddings);

  if ((request.pixelValuesVideos === undefined) !== (request.videoPositionIds === undefined)) throw new Error("Gemma 4 composite requer pixel_values_videos e video_position_ids juntos, ou ambos ausentes.");
  if (request.pixelValuesVideos !== undefined) {
    const videoTokenId = program.contract.modalities.videoTokenId;
    if (videoTokenId === undefined) throw new Error("Gemma 4 composite recebeu vídeo, mas o contrato não declara video_token_id.");
    const flattened = flattenVideo(request.pixelValuesVideos, request.videoPositionIds!);
    const video = executeGemma4VisionF32(program.visionProgram, { pixelValues: flattened.pixels, pixelPositionIds: flattened.positions, tensors: request.tensors });
    merge(values, prefixValues(video.values, "video_"));
    values.set("video_features", video.imageFeatures);
    embeddings = scatterGemma4ImageFeaturesF32(embeddings, request.inputIds, videoTokenId, video.imageFeatures);
  }
  values.set("composite_embeddings_after_video", embeddings);

  if ((request.inputFeatures === undefined) !== (request.inputFeaturesMask === undefined)) throw new Error("Gemma 4 composite requer input_features e input_features_mask juntos, ou ambos ausentes.");
  if (request.inputFeatures !== undefined) {
    const audio = executeGemma4AudioF32(program.audioProgram, { inputFeatures: request.inputFeatures, inputFeaturesMask: request.inputFeaturesMask!, tensors: request.tensors });
    merge(values, audio.values);
    values.set("audio_features", audio.audioFeatures);
    embeddings = scatterGemma4AudioFeaturesF32(embeddings, request.inputIds, program.contract.modalities.audioTokenId, audio.audioFeatures);
  }
  values.set("hidden_states_0", embeddings);

  const contextPacked = linear(embeddings, tensor(request.tensors, `${prefix}.per_layer_model_projection.weight`));
  const contextScaled = scale(contextPacked, program.contract.text.hiddenSize ** -0.5);
  const contextReshaped = reshapePerLayer(contextScaled, program.contract.text.layers, program.contract.text.perLayerInputSize);
  const contextNormalized = rmsNorm(contextReshaped, tensor(request.tensors, `${prefix}.per_layer_projection_norm.weight`), rmsNormEpsilon(program.textProgram));
  const pleCombined = add(contextNormalized, pleIdentity);
  const pleInputs = scale(pleCombined, 2 ** -0.5);
  values.set("ple_context_packed", contextPacked);
  values.set("ple_context_scaled", contextScaled);
  values.set("ple_context_reshaped", contextReshaped);
  values.set("ple_context_normalized", contextNormalized);
  values.set("ple_combined", pleCombined);
  values.set("ple_inputs", pleInputs);
  const visionMasks = request.mmTokenTypeIds === undefined ? undefined : visionAttentionMasks(request.mmTokenTypeIds, request.inputIds, program.textProgram);
  if (visionMasks !== undefined) {
    values.set("vision_block_sequence_ids", visionMasks.blockSequenceIds);
    values.set("full_attention_mask", visionMasks.full);
    values.set("sliding_attention_mask", visionMasks.sliding);
  }
  const text = executeReferenceF32WithPreparedPrelude(program.textProgram, {
    inputIds: request.inputIds,
    ...(request.positionIds ? { positionIds: request.positionIds } : {}),
    ...(request.attentionMask ? { attentionMask: request.attentionMask } : {}),
    ...(visionMasks ? { attentionMasksByLayer: visionMasks.byLayer } : {}),
    ...(request.pastKeyValues ? { pastKeyValues: request.pastKeyValues } : {}),
    tensors: request.tensors,
  }, values);
  merge(values, text.values);
  return { values, llmInputIds, text };
}

/**
 * Source-equivalent narrow greedy path: the multimodal block mask exists only
 * during prefill, then cached decode deliberately omits mm_token_type_ids as
 * Gemma4ForConditionalGeneration.prepare_inputs_for_generation does.
 */
export function generateGemma4CompositeF32(program: Gemma4CompositeProgram, request: Gemma4CompositeGenerationRequest): Gemma4CompositeGenerationResult {
  if (request.inputIds.length !== 1 || request.inputIds[0]?.length === 0) throw new Error("Geração Gemma 4 composite requer um único prompt não vazio sem padding.");
  if (!Number.isInteger(request.maxNewTokens) || request.maxNewTokens < 0) throw new Error("Geração Gemma 4 composite requer maxNewTokens inteiro não negativo.");
  if (request.eosTokenId !== undefined && (!Number.isInteger(request.eosTokenId) || request.eosTokenId < 0)) throw new Error("Geração Gemma 4 composite requer eosTokenId inteiro não negativo.");
  if (request.attentionMask !== undefined) throw new Error("Geração Gemma 4 composite ainda requer attentionMask ausente; padding/4-D caller masks exigem um contrato de geração próprio.");
  const prefill = executeGemma4CompositeF32(program, request);
  let current = prefill;
  let position = request.positionIds?.[0]?.at(-1) ?? request.inputIds[0]!.length - 1;
  const generatedTokenIds: number[] = [];
  const selectionLogits: DenseF32Tensor[] = [];
  const stepPastKeyValues: Array<ReadonlyMap<number, ReferenceF32KeyValueCache>> = [];
  for (let index = 0; index < request.maxNewTokens; index += 1) {
    selectionLogits.push(current.text.logits);
    const tokenId = selectGreedyToken(current.text.logits);
    generatedTokenIds.push(tokenId);
    position += 1;
    current = executeGemma4CompositeF32(program, { inputIds: [[tokenId]], positionIds: [[position]], pastKeyValues: current.text.pastKeyValues, tensors: request.tensors });
    stepPastKeyValues.push(current.text.pastKeyValues);
    if (tokenId === request.eosTokenId) break;
  }
  return { prefill, generatedTokenIds, selectionLogits, stepPastKeyValues, text: current.text };
}

function asF32ReferenceProgram(program: ModelIR): ModelIR {
  const result = structuredClone(program);
  for (const operation of [...result.prelude, ...result.layers.flatMap((layer) => layer.operations), ...result.epilogue]) {
    operation.dtypePolicy = F32_POLICY;
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
  return result;
}

function textConfig(catalog: ModelCatalog): Record<string, unknown> {
  const config = catalog.config.text_config;
  if (typeof config !== "object" || config === null || Array.isArray(config)) throw new Error("Gemma 4 composite requer text_config objeto.");
  return config as Record<string, unknown>;
}

function textPrefix(catalog: ModelCatalog): string {
  for (const prefix of ["model.language_model", "language_model.model"] as const) if (catalog.tensors.has(`${prefix}.embed_tokens.weight`)) return prefix;
  throw new Error("Gemma 4 composite não encontrou um layout textual registrado.");
}

function textPrefixFromProgram(program: ModelIR): string {
  const embedding = program.prelude.find((operation) => operation.id === "token_embedding");
  if (embedding?.op !== "embedding") throw new Error("Gemma 4 text program não declarou token_embedding.");
  return embedding.weight.name.replace(/\.embed_tokens\.weight$/, "");
}

function textPadTokenId(program: ModelIR): number {
  const config = program.config.pad_token_id;
  if (!Number.isInteger(config) || (config as number) < 0) throw new Error("Gemma 4 composite requer text_config.pad_token_id inteiro não negativo.");
  return config as number;
}

function rmsNormEpsilon(program: ModelIR): number {
  const config = program.config.rms_norm_eps;
  if (typeof config !== "number" || !Number.isFinite(config) || config <= 0) throw new Error("Gemma 4 composite requer text_config.rms_norm_eps positivo.");
  return config;
}

function requireDenseTensor(catalog: ModelCatalog, name: string): TensorInfo {
  const tensor = catalog.tensors.get(name);
  if (!tensor || tensor.quantization) throw new Error(`Gemma 4 composite requer tensor denso ${name}.`);
  return tensor;
}

function tensorRef(tensor: TensorInfo): TensorRef { return { name: tensor.name, shape: [...tensor.logicalShape], storageDtype: tensor.storageDtype }; }
function tensor(tensors: ReadonlyMap<string, DenseF32Tensor>, name: string): DenseF32Tensor { const found = tensors.get(name); if (!found) throw new Error(`Tensor F32 Gemma 4 composite ausente: ${name}`); return found; }
function validateInputIds(inputIds: number[][]): void { if (inputIds.length === 0 || inputIds.some((row) => row.length === 0) || inputIds.some((row) => row.length !== inputIds[0]!.length)) throw new Error("Gemma 4 composite requer input_ids não vazio e retangular."); }

function visionAttentionMasks(mmTokenTypeIds: number[][], inputIds: number[][], program: ModelIR): { blockSequenceIds: DenseF32Tensor; full: DenseF32Tensor; sliding: DenseF32Tensor; byLayer: ReadonlyMap<number, DenseF32Tensor> } {
  const sequence = inputIds[0]!.length;
  if (mmTokenTypeIds.length !== inputIds.length || mmTokenTypeIds.some((row) => row.length !== sequence)) throw new Error("Gemma 4 composite mm_token_type_ids deve acompanhar input_ids no shape [batch,sequence].");
  if (mmTokenTypeIds.some((row) => row.some((value) => !Number.isInteger(value)))) throw new Error("Gemma 4 composite mm_token_type_ids requer IDs inteiros.");
  const batch = mmTokenTypeIds.length;
  const blocks = new Float32Array(batch * sequence);
  for (let b = 0; b < batch; b += 1) {
    let nextGroup = 0;
    let previousVision = false;
    for (let index = 0; index < sequence; index += 1) {
      const type = mmTokenTypeIds[b]![index]!;
      const vision = type === 1 || type === 2;
      if (vision && !previousVision) nextGroup += 1;
      blocks[b * sequence + index] = vision ? nextGroup - 1 : -1;
      previousVision = vision;
    }
  }
  const attention = program.layers.flatMap((layer) => layer.operations).filter((operation): operation is Extract<typeof operation, { op: "scaled_dot_product_attention" }> => operation.op === "scaled_dot_product_attention");
  if (attention.length === 0 || attention.some((operation) => operation.layer === undefined || operation.causal !== true)) throw new Error("Gemma 4 composite requer atenções causais indexadas para compor máscaras multimodais.");
  const slidingWindow = new Set(attention.filter((operation) => operation.slidingWindow !== undefined).map((operation) => operation.slidingWindow!));
  if (slidingWindow.size !== 1) throw new Error("Gemma 4 composite requer exatamente uma janela deslizante registrada para a máscara multimodal.");
  const window = [...slidingWindow][0]!;
  const fullValues = new Float32Array(batch * sequence * sequence);
  const slidingValues = new Float32Array(batch * sequence * sequence);
  for (let b = 0; b < batch; b += 1) for (let query = 0; query < sequence; query += 1) for (let key = 0; key < sequence; key += 1) {
    const offset = (b * sequence + query) * sequence + key;
    const causal = key <= query;
    const sameVisionBlock = blocks[b * sequence + query] === blocks[b * sequence + key] && blocks[b * sequence + query]! >= 0;
    fullValues[offset] = causal ? 0 : -Infinity;
    slidingValues[offset] = key > query - window && (causal || sameVisionBlock) ? 0 : -Infinity;
  }
  const full = dense([batch, 1, sequence, sequence], fullValues);
  const sliding = dense([batch, 1, sequence, sequence], slidingValues);
  const byLayer = new Map<number, DenseF32Tensor>();
  for (const operation of attention) byLayer.set(operation.layer!, operation.slidingWindow === undefined ? full : sliding);
  return { blockSequenceIds: dense([batch, sequence], blocks), full, sliding, byLayer };
}

function replaceModalIdsWithPad(inputIds: number[][], contract: Gemma4PackageContract, pad: number): number[][] {
  const ids = new Set<number>([contract.modalities.imageTokenId, contract.modalities.audioTokenId]);
  if (contract.modalities.videoTokenId !== undefined) ids.add(contract.modalities.videoTokenId);
  return inputIds.map((row) => row.map((id) => ids.has(id) ? pad : id));
}

function embedding(inputIds: number[][], weight: DenseF32Tensor, scaleValue: number): DenseF32Tensor {
  if (weight.shape.length !== 2) throw new Error("Embedding Gemma 4 composite requer tabela [vocab,hidden].");
  const [batch, sequence] = [inputIds.length, inputIds[0]!.length], hidden = weight.shape[1]!;
  const out = new Float32Array(batch * sequence * hidden);
  for (let b = 0; b < batch; b += 1) for (let s = 0; s < sequence; s += 1) { const token = inputIds[b]![s]!; if (!Number.isInteger(token) || token < 0 || token >= weight.shape[0]!) throw new Error(`Token Gemma 4 fora de vocab: ${token}.`); for (let d = 0; d < hidden; d += 1) out[(b * sequence + s) * hidden + d] = f32(weight.values[token * hidden + d]! * f32(scaleValue)); }
  return dense([batch, sequence, hidden], out);
}

function perLayerEmbedding(inputIds: number[][], weight: DenseF32Tensor, layers: number, width: number, scaleValue: number): DenseF32Tensor {
  if (weight.shape.length !== 2 || weight.shape[1] !== layers * width) throw new Error("PLE Gemma 4 composite incompatível com layers/width.");
  const [batch, sequence] = [inputIds.length, inputIds[0]!.length], out = new Float32Array(batch * sequence * layers * width);
  for (let b = 0; b < batch; b += 1) for (let s = 0; s < sequence; s += 1) { const token = inputIds[b]![s]!; if (!Number.isInteger(token) || token < 0 || token >= weight.shape[0]!) throw new Error(`Token PLE Gemma 4 fora de vocab: ${token}.`); for (let d = 0; d < layers * width; d += 1) out[((b * sequence + s) * layers * width) + d] = f32(weight.values[token * layers * width + d]! * f32(scaleValue)); }
  return dense([batch, sequence, layers, width], out);
}

function linear(input: DenseF32Tensor, weight: DenseF32Tensor): DenseF32Tensor {
  if (weight.shape.length !== 2 || input.shape.at(-1) !== weight.shape[1]) throw new Error("Linear Gemma 4 composite incompatível.");
  const rows = input.values.length / weight.shape[1]!, out = new Float32Array(rows * weight.shape[0]!);
  for (let row = 0; row < rows; row += 1) for (let output = 0; output < weight.shape[0]!; output += 1) { let sum = f32(0); for (let feature = 0; feature < weight.shape[1]!; feature += 1) sum = f32(sum + f32(input.values[row * weight.shape[1]! + feature]! * weight.values[output * weight.shape[1]! + feature]!)); out[row * weight.shape[0]! + output] = sum; }
  return dense([...input.shape.slice(0, -1), weight.shape[0]!], out);
}

function reshapePerLayer(input: DenseF32Tensor, layers: number, width: number): DenseF32Tensor { if (input.shape.at(-1) !== layers * width) throw new Error("reshape PLE Gemma 4 incompatível."); return dense([...input.shape.slice(0, -1), layers, width], Float32Array.from(input.values)); }
function rmsNorm(input: DenseF32Tensor, weight: DenseF32Tensor, epsilon: number): DenseF32Tensor { const width = input.shape.at(-1)!; if (weight.shape.length !== 1 || weight.shape[0] !== width) throw new Error("RMSNorm PLE Gemma 4 incompatível."); const out = new Float32Array(input.values.length); for (let offset = 0; offset < out.length; offset += width) { let sum = f32(0); for (let d = 0; d < width; d += 1) sum = f32(sum + f32(input.values[offset + d]! * input.values[offset + d]!)); const inv = f32(Math.pow(f32(f32(sum / f32(width)) + f32(epsilon)), -0.5)); for (let d = 0; d < width; d += 1) out[offset + d] = f32(f32(input.values[offset + d]! * inv) * weight.values[d]!); } return dense([...input.shape], out); }
function scale(input: DenseF32Tensor, scalar: number): DenseF32Tensor { return dense([...input.shape], Float32Array.from(input.values, (value) => f32(value * f32(scalar)))); }
function add(left: DenseF32Tensor, right: DenseF32Tensor): DenseF32Tensor { if (left.shape.length !== right.shape.length || left.shape.some((value, index) => value !== right.shape[index])) throw new Error("Soma PLE Gemma 4 incompatível."); return dense([...left.shape], Float32Array.from(left.values, (value, index) => f32(value + right.values[index]!))); }
function dense(shape: number[], values: Float32Array): DenseF32Tensor { return { shape, values }; }
function f32(value: number): number { return Math.fround(value); }
function merge(target: Map<string, DenseF32Tensor>, source: ReadonlyMap<string, DenseF32Tensor>): void { for (const [name, value] of source) target.set(name, value); }
function prefixValues(values: ReadonlyMap<string, DenseF32Tensor>, prefix: string): Map<string, DenseF32Tensor> { return new Map([...values].map(([name, value]) => [`${prefix}${name}`, value])); }

function flattenVideo(pixels: DenseF32Tensor, positions: number[][][][]): { pixels: DenseF32Tensor; positions: number[][][] } {
  if (pixels.shape.length !== 4) throw new Error("Gemma 4 video requer pixel_values_videos [videos,frames,patches,features].");
  const [videos, frames, patches, features] = pixels.shape as [number, number, number, number];
  if (positions.length !== videos || positions.some((video) => video.length !== frames || video.some((frame) => frame.length !== patches || frame.some((position) => position.length !== 2)))) throw new Error("Gemma 4 video_position_ids não acompanha [videos,frames,patches,2].");
  return { pixels: dense([videos * frames, patches, features], Float32Array.from(pixels.values)), positions: positions.flat() };
}
