import type { JsonObject, ModelCatalog, TensorInfo } from "./types.js";

/**
 * The facts required to lower a Gemma 4 package are deliberately separated
 * from the existing decoder adapter.  This is an evidence/audit boundary, not
 * an implicit fallback: callers must still reject the package until every
 * listed operation family has an executor and differential evidence.
 */
export interface Gemma4LayerContract {
  index: number;
  attentionType: "sliding_attention" | "full_attention";
  headDim: number;
  keyValueHeads: number;
  keyEqualsValue: boolean;
  keyValueProducerLayer?: number;
  storesSharedKeyValue: boolean;
  mlpIntermediateSize: number;
}

export interface Gemma4PackageContract {
  modelType: "gemma4";
  sourceFormat: "safetensors";
  text: {
    hiddenSize: number;
    vocabSize: number;
    layers: number;
    attentionHeads: number;
    keyValueHeads: number;
    globalKeyValueHeads: number;
    headDim: number;
    globalHeadDim: number;
    sharedKeyValueLayers: number;
    perLayerInputSize: number;
    perLayerInputVocabSize: number;
    finalLogitSoftcap?: number;
    layersContract: Gemma4LayerContract[];
  };
  modalities: {
    vision: true;
    audio: true;
    imageTokenId: number;
    audioTokenId: number;
    videoTokenId?: number;
  };
  requiredOperationFamilies: readonly [
    "vision-token-injection",
    "audio-token-injection",
    "per-layer-embeddings",
    "type-specific-rope",
    "shared-kv-state",
    "gemma4-four-rmsnorm-decoder",
    "multimodal-generation-state",
  ];
}

const REQUIRED_OPERATION_FAMILIES: Gemma4PackageContract["requiredOperationFamilies"] = [
  "vision-token-injection",
  "audio-token-injection",
  "per-layer-embeddings",
  "type-specific-rope",
  "shared-kv-state",
  "gemma4-four-rmsnorm-decoder",
  "multimodal-generation-state",
];

/**
 * Validates the package-level Gemma 4 semantic contract against its actual
 * Safetensors roles and shapes. This produces no ModelIR on purpose: returning
 * this contract must never be mistaken for an executable Gemma 4 adapter.
 */
export function inspectGemma4PackageContract(catalog: ModelCatalog): Gemma4PackageContract {
  if (catalog.format !== "safetensors") {
    throw new Error(`Auditoria Gemma 4 requer Safetensors denso; recebeu ${catalog.format}.`);
  }
  const outer = catalog.config;
  if (string(outer, "model_type") !== "gemma4") {
    throw new Error("Auditoria Gemma 4 requer config.model_type='gemma4' do pacote composto.");
  }
  const text = object(outer, "text_config");
  if (string(text, "model_type") !== "gemma4_text") {
    throw new Error("Pacote Gemma 4 não declara text_config.model_type='gemma4_text'.");
  }
  const vision = object(outer, "vision_config");
  const audio = object(outer, "audio_config");
  if (string(vision, "model_type") !== "gemma4_vision" || string(audio, "model_type") !== "gemma4_audio") {
    throw new Error("Gemma 4 requer os subcontratos vision_config=gemma4_vision e audio_config=gemma4_audio; não reduza o pacote a texto.");
  }

  const hiddenSize = positiveInt(text, "hidden_size");
  const vocabSize = positiveInt(text, "vocab_size");
  const layers = positiveInt(text, "num_hidden_layers");
  const attentionHeads = positiveInt(text, "num_attention_heads");
  const keyValueHeads = positiveInt(text, "num_key_value_heads");
  const globalKeyValueHeads = optionalPositiveInt(text, "num_global_key_value_heads") ?? keyValueHeads;
  const headDim = positiveInt(text, "head_dim");
  const globalHeadDim = positiveInt(text, "global_head_dim");
  const intermediateSize = positiveInt(text, "intermediate_size");
  const sharedKeyValueLayers = nonnegativeInt(text, "num_kv_shared_layers");
  const perLayerInputSize = positiveInt(text, "hidden_size_per_layer_input");
  const perLayerInputVocabSize = positiveInt(text, "vocab_size_per_layer_input");
  const layerTypes = layerTypesFrom(text, layers);
  const attentionKeyEqualsValue = boolean(text, "attention_k_eq_v");
  const doubleWideMlp = boolean(text, "use_double_wide_mlp");
  const finalLogitSoftcap = optionalPositiveNumber(text, "final_logit_softcapping");

  if (attentionHeads % keyValueHeads !== 0 || attentionHeads % globalKeyValueHeads !== 0) {
    throw new Error("Gemma 4 num_attention_heads deve ser múltiplo de ambos num_key_value_heads e num_global_key_value_heads.");
  }
  if (sharedKeyValueLayers > layers) {
    throw new Error(`Gemma 4 num_kv_shared_layers=${sharedKeyValueLayers} excede num_hidden_layers=${layers}.`);
  }
  validateRope(text, "sliding_attention", "default", headDim);
  validateRope(text, "full_attention", "proportional", globalHeadDim);

  const prefix = registeredTextPrefix(catalog);
  requireShape(catalog, `${prefix}.embed_tokens.weight`, [vocabSize, hiddenSize]);
  requireShape(catalog, `${prefix}.embed_tokens_per_layer.weight`, [perLayerInputVocabSize, layers * perLayerInputSize]);
  requireShape(catalog, `${prefix}.per_layer_model_projection.weight`, [layers * perLayerInputSize, hiddenSize]);
  requireShape(catalog, `${prefix}.per_layer_projection_norm.weight`, [perLayerInputSize]);
  requireShape(catalog, `${prefix}.norm.weight`, [hiddenSize]);
  requireNamespace(catalog, "model.vision_tower.");
  requireNamespace(catalog, "model.audio_tower.");
  requireNamespace(catalog, "model.embed_vision.");
  requireNamespace(catalog, "model.embed_audio.");

  const firstSharedLayer = layers - sharedKeyValueLayers;
  const ownerByType = new Map<Gemma4LayerContract["attentionType"], number>();
  for (let layer = 0; layer < firstSharedLayer; layer += 1) ownerByType.set(layerTypes[layer]!, layer);
  if (sharedKeyValueLayers > 0 && ownerByType.size !== 2) {
    throw new Error("Gemma 4 KV compartilhado requer um produtor não compartilhado para cada tipo de atenção declarado.");
  }

  const layersContract: Gemma4LayerContract[] = [];
  for (let layer = 0; layer < layers; layer += 1) {
    const attentionType = layerTypes[layer]!;
    const isSliding = attentionType === "sliding_attention";
    const keyEqualsValue = attentionKeyEqualsValue && !isSliding;
    const layerHeadDim = isSliding ? headDim : globalHeadDim;
    const layerKeyValueHeads = keyEqualsValue ? globalKeyValueHeads : keyValueHeads;
    const isShared = sharedKeyValueLayers > 0 && layer >= firstSharedLayer;
    const layerPrefix = `${prefix}.layers.${layer}`;
    requireShape(catalog, `${layerPrefix}.input_layernorm.weight`, [hiddenSize]);
    requireShape(catalog, `${layerPrefix}.self_attn.q_proj.weight`, [attentionHeads * layerHeadDim, hiddenSize]);
    requireShape(catalog, `${layerPrefix}.self_attn.q_norm.weight`, [layerHeadDim]);
    requireShape(catalog, `${layerPrefix}.self_attn.o_proj.weight`, [hiddenSize, attentionHeads * layerHeadDim]);
    requireShape(catalog, `${layerPrefix}.post_attention_layernorm.weight`, [hiddenSize]);
    requireShape(catalog, `${layerPrefix}.pre_feedforward_layernorm.weight`, [hiddenSize]);
    requireShape(catalog, `${layerPrefix}.post_feedforward_layernorm.weight`, [hiddenSize]);
    requireShape(catalog, `${layerPrefix}.per_layer_input_gate.weight`, [perLayerInputSize, hiddenSize]);
    requireShape(catalog, `${layerPrefix}.per_layer_projection.weight`, [hiddenSize, perLayerInputSize]);
    requireShape(catalog, `${layerPrefix}.post_per_layer_input_norm.weight`, [hiddenSize]);
    requireShape(catalog, `${layerPrefix}.layer_scalar`, [1]);
    const mlpIntermediateSize = intermediateSize * (doubleWideMlp && isShared ? 2 : 1);
    requireShape(catalog, `${layerPrefix}.mlp.gate_proj.weight`, [mlpIntermediateSize, hiddenSize]);
    requireShape(catalog, `${layerPrefix}.mlp.up_proj.weight`, [mlpIntermediateSize, hiddenSize]);
    requireShape(catalog, `${layerPrefix}.mlp.down_proj.weight`, [hiddenSize, mlpIntermediateSize]);

    const producer = isShared ? ownerByType.get(attentionType) : undefined;
    if (isShared && producer === undefined) throw new Error(`Camada Gemma 4 ${layer} não possui produtor KV ${attentionType} não compartilhado.`);
    if (!isShared) {
      requireShape(catalog, `${layerPrefix}.self_attn.k_proj.weight`, [layerKeyValueHeads * layerHeadDim, hiddenSize]);
      requireShape(catalog, `${layerPrefix}.self_attn.k_norm.weight`, [layerHeadDim]);
      if (!keyEqualsValue) {
        requireShape(catalog, `${layerPrefix}.self_attn.v_proj.weight`, [layerKeyValueHeads * layerHeadDim, hiddenSize]);
      }
    }
    layersContract.push({
      index: layer,
      attentionType,
      headDim: layerHeadDim,
      keyValueHeads: layerKeyValueHeads,
      keyEqualsValue,
      ...(producer !== undefined ? { keyValueProducerLayer: producer } : {}),
      storesSharedKeyValue: !isShared && ownerByType.get(attentionType) === layer,
      mlpIntermediateSize,
    });
  }

  return {
    modelType: "gemma4",
    sourceFormat: "safetensors",
    text: {
      hiddenSize, vocabSize, layers, attentionHeads, keyValueHeads, globalKeyValueHeads,
      headDim, globalHeadDim, sharedKeyValueLayers, perLayerInputSize, perLayerInputVocabSize,
      ...(finalLogitSoftcap !== undefined ? { finalLogitSoftcap } : {}), layersContract,
    },
    modalities: {
      vision: true,
      audio: true,
      imageTokenId: positiveInt(outer, "image_token_id"),
      audioTokenId: positiveInt(outer, "audio_token_id"),
      ...(optionalPositiveInt(outer, "video_token_id") !== undefined ? { videoTokenId: optionalPositiveInt(outer, "video_token_id")! } : {}),
    },
    requiredOperationFamilies: REQUIRED_OPERATION_FAMILIES,
  };
}

function requireNamespace(catalog: ModelCatalog, prefix: string): void {
  if (![...catalog.tensors.keys()].some((name) => name.startsWith(prefix))) throw new Error(`Gemma 4 exige tensors no namespace ${prefix}.`);
}

/** These complete prefixes are registered package layouts, never suffix guesses. */
function registeredTextPrefix(catalog: ModelCatalog): string {
  for (const prefix of ["model.language_model", "language_model.model"] as const) {
    if (catalog.tensors.has(`${prefix}.embed_tokens.weight`)) return prefix;
  }
  throw new Error("Gemma 4 não usa um layout textual registrado (model.language_model ou language_model.model).");
}

function requireShape(catalog: ModelCatalog, name: string, expected: readonly number[]): void {
  const tensor = catalog.tensors.get(name);
  if (!tensor) throw new Error(`Gemma 4 exige tensor ${name}.`);
  if (tensor.quantization) throw new Error(`${name}: auditoria do checkpoint Gemma 4 requer armazenamento denso, recebeu ${tensor.quantization.family}/${tensor.quantization.mode}.`);
  if (tensor.logicalShape.length !== expected.length || tensor.logicalShape.some((dimension, index) => dimension !== expected[index])) {
    throw new Error(`${name}: shape Gemma 4 incompatível; esperado [${expected.join(", ")}], recebeu [${tensor.logicalShape.join(", ")}].`);
  }
}

function validateRope(text: JsonObject, attentionType: "sliding_attention" | "full_attention", expectedType: string, headDim: number): void {
  const root = object(text, "rope_parameters");
  const parameters = object(root, attentionType);
  if (string(parameters, "rope_type") !== expectedType) throw new Error(`Gemma 4 ${attentionType} requer rope_type='${expectedType}'.`);
  if (!(positiveNumber(parameters, "rope_theta") > 0)) throw new Error(`Gemma 4 ${attentionType} rope_theta deve ser positivo.`);
  if (expectedType === "proportional") {
    const factor = positiveNumber(parameters, "partial_rotary_factor");
    if (factor > 1 || !Number.isInteger(headDim * factor) || (headDim * factor) % 2 !== 0) {
      throw new Error("Gemma 4 full_attention partial_rotary_factor deve produzir uma dimensão RoPE inteira e par.");
    }
  }
}

function layerTypesFrom(text: JsonObject, layers: number): Array<"sliding_attention" | "full_attention"> {
  const value = text.layer_types;
  if (!Array.isArray(value) || value.length !== layers || value.some((type) => type !== "sliding_attention" && type !== "full_attention")) {
    throw new Error("Gemma 4 layer_types deve declarar uma topologia sliding_attention/full_attention para cada camada.");
  }
  return value as Array<"sliding_attention" | "full_attention">;
}

function object(value: JsonObject, key: string): JsonObject {
  const result = value[key];
  if (typeof result !== "object" || result === null || Array.isArray(result)) throw new Error(`${key} deve ser um objeto Gemma 4.`);
  return result as JsonObject;
}

function string(value: JsonObject, key: string): string {
  const result = value[key];
  if (typeof result !== "string" || result.length === 0) throw new Error(`${key} deve ser uma string não vazia.`);
  return result;
}

function boolean(value: JsonObject, key: string): boolean {
  if (typeof value[key] !== "boolean") throw new Error(`${key} deve ser booleano no contrato Gemma 4.`);
  return value[key] as boolean;
}

function positiveInt(value: JsonObject, key: string): number {
  const result = value[key];
  if (!Number.isInteger(result) || (result as number) <= 0) throw new Error(`${key} deve ser inteiro positivo no contrato Gemma 4.`);
  return result as number;
}

function optionalPositiveInt(value: JsonObject, key: string): number | undefined {
  if (value[key] === undefined || value[key] === null) return undefined;
  return positiveInt(value, key);
}

function nonnegativeInt(value: JsonObject, key: string): number {
  const result = value[key];
  if (!Number.isInteger(result) || (result as number) < 0) throw new Error(`${key} deve ser inteiro não negativo no contrato Gemma 4.`);
  return result as number;
}

function positiveNumber(value: JsonObject, key: string): number {
  const result = value[key];
  if (typeof result !== "number" || !Number.isFinite(result) || result <= 0) throw new Error(`${key} deve ser número positivo no contrato Gemma 4.`);
  return result;
}

function optionalPositiveNumber(value: JsonObject, key: string): number | undefined {
  if (value[key] === undefined || value[key] === null) return undefined;
  return positiveNumber(value, key);
}
