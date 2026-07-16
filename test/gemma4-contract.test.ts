import test from "node:test";
import assert from "node:assert/strict";
import { inspectGemma4PackageContract } from "../src/gemma4-contract.js";
import type { ModelCatalog, TensorInfo } from "../src/types.js";

function tensor(name: string, shape: number[]): TensorInfo {
  return { name, storageDtype: "BF16", storageShape: [...shape], logicalShape: [...shape] };
}

function fixture(): ModelCatalog {
  const hidden = 8;
  const layers = 4;
  const perLayer = 2;
  const tensors = new Map<string, TensorInfo>();
  const add = (name: string, shape: number[]) => tensors.set(name, tensor(name, shape));
  const prefix = "model.language_model";
  add(`${prefix}.embed_tokens.weight`, [16, hidden]);
  add(`${prefix}.embed_tokens_per_layer.weight`, [16, layers * perLayer]);
  add(`${prefix}.per_layer_model_projection.weight`, [layers * perLayer, hidden]);
  add(`${prefix}.per_layer_projection_norm.weight`, [perLayer]);
  add(`${prefix}.norm.weight`, [hidden]);
  addVisionTower(add, hidden);
  addAudioTower(add, hidden);
  for (let layer = 0; layer < layers; layer += 1) {
    const attentionType = layer % 2 === 0 ? "sliding_attention" : "full_attention";
    const headDim = attentionType === "sliding_attention" ? 2 : 4;
    const layerPrefix = `${prefix}.layers.${layer}`;
    add(`${layerPrefix}.input_layernorm.weight`, [hidden]);
    add(`${layerPrefix}.self_attn.q_proj.weight`, [4 * headDim, hidden]);
    add(`${layerPrefix}.self_attn.q_norm.weight`, [headDim]);
    add(`${layerPrefix}.self_attn.o_proj.weight`, [hidden, 4 * headDim]);
    add(`${layerPrefix}.post_attention_layernorm.weight`, [hidden]);
    add(`${layerPrefix}.pre_feedforward_layernorm.weight`, [hidden]);
    add(`${layerPrefix}.post_feedforward_layernorm.weight`, [hidden]);
    add(`${layerPrefix}.per_layer_input_gate.weight`, [perLayer, hidden]);
    add(`${layerPrefix}.per_layer_projection.weight`, [hidden, perLayer]);
    add(`${layerPrefix}.post_per_layer_input_norm.weight`, [hidden]);
    add(`${layerPrefix}.layer_scalar`, [1]);
    add(`${layerPrefix}.mlp.gate_proj.weight`, [12, hidden]);
    add(`${layerPrefix}.mlp.up_proj.weight`, [12, hidden]);
    add(`${layerPrefix}.mlp.down_proj.weight`, [hidden, 12]);
    if (layer < 2) {
      add(`${layerPrefix}.self_attn.k_proj.weight`, [2 * headDim, hidden]);
      add(`${layerPrefix}.self_attn.v_proj.weight`, [2 * headDim, hidden]);
      add(`${layerPrefix}.self_attn.k_norm.weight`, [headDim]);
    }
  }
  return {
    source: "/tmp/gemma4-dense", format: "safetensors", rawMetadata: {}, tensors,
    config: {
      model_type: "gemma4", image_token_id: 99, audio_token_id: 98,
      vision_config: {
        model_type: "gemma4_vision", hidden_size: 8, num_hidden_layers: 2, num_attention_heads: 2,
        num_key_value_heads: 2, head_dim: 4, intermediate_size: 16, patch_size: 2,
        position_embedding_size: 8, default_output_length: 4, pooling_kernel_size: 2,
        attention_bias: false, hidden_activation: "gelu_pytorch_tanh", use_clipped_linears: true,
        rope_parameters: { rope_type: "default", rope_theta: 100 },
      },
      audio_config: {
        model_type: "gemma4_audio", hidden_size: 8, num_hidden_layers: 2, num_attention_heads: 2,
        output_proj_dims: 6, attention_chunk_size: 2, attention_context_left: 1, attention_context_right: 0,
        attention_logit_cap: 50, conv_kernel_size: 5, residual_weight: 0.5,
        subsampling_conv_channels: [2, 2], hidden_act: "silu",
      },
      text_config: {
        model_type: "gemma4_text", hidden_size: hidden, vocab_size: 16, num_hidden_layers: layers,
        num_attention_heads: 4, num_key_value_heads: 2, global_head_dim: 4, head_dim: 2,
        intermediate_size: 12, num_kv_shared_layers: 2, hidden_size_per_layer_input: perLayer,
        vocab_size_per_layer_input: 16, attention_k_eq_v: false, use_double_wide_mlp: false,
        layer_types: ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        rope_parameters: {
          sliding_attention: { rope_type: "default", rope_theta: 10_000 },
          full_attention: { rope_type: "proportional", rope_theta: 1_000_000, partial_rotary_factor: 0.5 },
        },
      },
    },
  };
}

test("Gemma 4 package audit makes PLE, per-type heads, multimodal towers, and shared-KV owners explicit", () => {
  const contract = inspectGemma4PackageContract(fixture());
  assert.equal(contract.text.layers, 4);
  assert.equal(contract.text.layersContract[0]!.headDim, 2);
  assert.equal(contract.text.layersContract[1]!.headDim, 4);
  assert.equal(contract.text.layersContract[2]!.keyValueProducerLayer, 0);
  assert.equal(contract.text.layersContract[3]!.keyValueProducerLayer, 1);
  assert.equal(contract.text.layersContract[0]!.storesSharedKeyValue, true);
  assert.equal(contract.text.layersContract[1]!.storesSharedKeyValue, true);
  assert.deepEqual(contract.modalities.visionTower, {
    hiddenSize: 8, layers: 2, attentionHeads: 2, headDim: 4, intermediateSize: 16,
    patchSize: 2, positionEmbeddingSize: 8, defaultOutputLength: 4, poolingKernelSize: 2,
    ropeTheta: 100, clippedLinears: true,
  });
  assert.equal(contract.modalities.audioTower.headDim, 4);
  assert.deepEqual(contract.modalities.audioTower.subsamplingChannels, [2, 2]);
  assert.deepEqual(contract.requiredOperationFamilies, [
    "vision-token-injection", "audio-token-injection", "clipped-linear", "vision-patch-position-pooling",
    "audio-subsample-local-convolution", "per-layer-embeddings", "type-specific-rope",
    "shared-kv-state", "gemma4-four-rmsnorm-decoder", "multimodal-generation-state",
  ]);
});

test("Gemma 4 package audit fails closed when a shared-KV owner has an incompatible K projection", () => {
  const source = fixture();
  source.tensors.set("model.language_model.layers.1.self_attn.k_proj.weight", tensor("model.language_model.layers.1.self_attn.k_proj.weight", [7, 8]));
  assert.throws(() => inspectGemma4PackageContract(source), /layers\.1\.self_attn\.k_proj\.weight: shape Gemma 4 incompatível/);
});

test("Gemma 4 package audit rejects a missing checkpointed clippable-linear bound", () => {
  const source = fixture();
  source.tensors.delete("model.vision_tower.encoder.layers.0.self_attn.q_proj.output_max");
  assert.throws(() => inspectGemma4PackageContract(source), /q_proj\.output_max/);
});

function addVisionTower(add: (name: string, shape: number[]) => void, hidden: number): void {
  const prefix = "model.vision_tower";
  add(`${prefix}.patch_embedder.input_proj.weight`, [hidden, 12]);
  add(`${prefix}.patch_embedder.position_embedding_table`, [2, 8, hidden]);
  for (let layer = 0; layer < 2; layer += 1) {
    const base = `${prefix}.encoder.layers.${layer}`;
    for (const name of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]) add(`${base}.${name}.weight`, [hidden]);
    add(`${base}.self_attn.q_norm.weight`, [4]);
    add(`${base}.self_attn.k_norm.weight`, [4]);
    for (const name of ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]) addClippedLinear(add, `${base}.${name}`, [hidden, hidden]);
    addClippedLinear(add, `${base}.mlp.gate_proj`, [16, hidden]);
    addClippedLinear(add, `${base}.mlp.up_proj`, [16, hidden]);
    addClippedLinear(add, `${base}.mlp.down_proj`, [hidden, 16]);
  }
  add("model.embed_vision.embedding_projection.weight", [hidden, hidden]);
}

function addAudioTower(add: (name: string, shape: number[]) => void, hidden: number): void {
  const prefix = "model.audio_tower";
  add(`${prefix}.subsample_conv_projection.layer0.conv.weight`, [2, 1, 3, 3]);
  add(`${prefix}.subsample_conv_projection.layer0.norm.weight`, [2]);
  add(`${prefix}.subsample_conv_projection.layer1.conv.weight`, [2, 2, 3, 3]);
  add(`${prefix}.subsample_conv_projection.layer1.norm.weight`, [2]);
  add(`${prefix}.subsample_conv_projection.input_proj_linear.weight`, [hidden, hidden]);
  for (let layer = 0; layer < 2; layer += 1) {
    const base = `${prefix}.layers.${layer}`;
    for (const name of ["norm_pre_attn", "norm_post_attn", "norm_out"]) add(`${base}.${name}.weight`, [hidden]);
    add(`${base}.self_attn.per_dim_scale`, [4]);
    add(`${base}.self_attn.relative_k_proj.weight`, [hidden, hidden]);
    for (const name of ["q_proj", "k_proj", "v_proj", "post"]) addClippedLinear(add, `${base}.self_attn.${name}`, [hidden, hidden]);
    for (const feedForward of ["feed_forward1", "feed_forward2"]) {
      add(`${base}.${feedForward}.pre_layer_norm.weight`, [hidden]);
      add(`${base}.${feedForward}.post_layer_norm.weight`, [hidden]);
      addClippedLinear(add, `${base}.${feedForward}.ffw_layer_1`, [hidden * 4, hidden]);
      addClippedLinear(add, `${base}.${feedForward}.ffw_layer_2`, [hidden, hidden * 4]);
    }
    add(`${base}.lconv1d.pre_layer_norm.weight`, [hidden]);
    add(`${base}.lconv1d.conv_norm.weight`, [hidden]);
    add(`${base}.lconv1d.depthwise_conv1d.weight`, [hidden, 1, 5]);
    addClippedLinear(add, `${base}.lconv1d.linear_start`, [hidden * 2, hidden]);
    addClippedLinear(add, `${base}.lconv1d.linear_end`, [hidden, hidden]);
  }
  add(`${prefix}.output_proj.weight`, [6, hidden]);
  add(`${prefix}.output_proj.bias`, [6]);
  add("model.embed_audio.embedding_projection.weight", [hidden, 6]);
}

function addClippedLinear(add: (name: string, shape: number[]) => void, prefix: string, shape: number[]): void {
  add(`${prefix}.linear.weight`, shape);
  for (const suffix of ["input_min", "input_max", "output_min", "output_max"]) add(`${prefix}.${suffix}`, []);
}
