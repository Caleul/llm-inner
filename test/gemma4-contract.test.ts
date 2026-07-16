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
  add("model.vision_tower.patch_embedder.input_proj.weight", [1, 1]);
  add("model.audio_tower.output_proj.weight", [1, 1]);
  add("model.embed_vision.embedding_projection.weight", [1, 1]);
  add("model.embed_audio.embedding_projection.weight", [1, 1]);
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
      vision_config: { model_type: "gemma4_vision" }, audio_config: { model_type: "gemma4_audio" },
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
  assert.deepEqual(contract.requiredOperationFamilies, [
    "vision-token-injection", "audio-token-injection", "per-layer-embeddings", "type-specific-rope",
    "shared-kv-state", "gemma4-four-rmsnorm-decoder", "multimodal-generation-state",
  ]);
});

test("Gemma 4 package audit fails closed when a shared-KV owner has an incompatible K projection", () => {
  const source = fixture();
  source.tensors.set("model.language_model.layers.1.self_attn.k_proj.weight", tensor("model.language_model.layers.1.self_attn.k_proj.weight", [7, 8]));
  assert.throws(() => inspectGemma4PackageContract(source), /layers\.1\.self_attn\.k_proj\.weight: shape Gemma 4 incompatível/);
});
