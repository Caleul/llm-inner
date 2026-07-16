import assert from "node:assert/strict";
import test from "node:test";
import { buildModelIR } from "../src/architecture.js";
import { executeReferenceF32 } from "../src/executor.js";
import type { DenseF32Tensor, ModelCatalog, TensorInfo } from "../src/types.js";

const preview = { outputRows: 1, inputTerms: 1, includeWeights: false } as const;

test("standalone Gemma 4 text lowering makes PLE, proportional RoPE, unscaled V norm, and scalar assignments explicit", async () => {
  const catalog = tinyGemma4Text();
  const ir = await buildModelIR(catalog, preview);
  assert.equal(ir.architecture.modelType, "gemma4_text");
  assert.deepEqual(ir.prelude.map((operation) => operation.id), [
    "token_embedding", "ple_token_identity", "ple_context_projection", "ple_context_scale",
    "ple_context_reshape", "ple_context_norm", "ple_combine", "ple_combine_scale",
  ]);
  const full = ir.layers[1]!.operations;
  const rope = full.find((operation) => operation.id === "layer_1_q_rope");
  assert.equal(rope?.op, "rotary_embedding");
  if (rope?.op === "rotary_embedding") assert.deepEqual({ type: rope.ropeType, dim: rope.rotaryDim, partial: rope.scaling?.partial_rotary_factor }, { type: "proportional", dim: 4, partial: 0.5 });
  const vNorm = full.find((operation) => operation.id === "layer_1_v_norm");
  assert.equal(vNorm?.op, "rms_norm");
  if (vNorm?.op === "rms_norm") assert.deepEqual({ transform: vNorm.weightTransform, weight: vNorm.weight }, { transform: "none", weight: undefined });
  assert.ok(full.some((operation) => operation.op === "select_per_layer" && operation.layerIndex === 1));
  assert.ok(full.some((operation) => operation.op === "tensor_scale" && operation.output === "hidden_states_2"));
});

test("Gemma 4 text F32 reference path executes the complete PLE branch and proportional RoPE", async () => {
  const catalog = tinyGemma4Text();
  const ir = await buildModelIR(catalog, preview);
  for (const operation of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    operation.dtypePolicy = { inputDtype: "F32", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
  const tensors = new Map<string, DenseF32Tensor>();
  for (const tensor of catalog.tensors.values()) tensors.set(tensor.name, denseFor(tensor));
  const result = executeReferenceF32(ir, { inputIds: [[1]], tensors });
  assert.deepEqual(result.logits.shape, [1, 1, 3]);
  assert.equal(result.values.has("layer_0_ple_input"), true);
  assert.equal(result.values.has("layer_1_ple_input"), true);
  assert.equal(result.values.has("layer_1_q_rot"), true);
  assert.equal(result.values.has("hidden_states_2"), true);
  assert.ok([...result.logits.values].every(Number.isFinite));
});

function tinyGemma4Text(): ModelCatalog {
  const tensors = new Map<string, TensorInfo>();
  const add = (name: string, shape: number[]): void => { tensors.set(name, { name, storageDtype: "F32", storageShape: shape, logicalShape: shape }); };
  const hidden = 4, layers = 2, ple = 2, vocab = 3;
  add("model.embed_tokens.weight", [vocab, hidden]);
  add("model.embed_tokens_per_layer.weight", [vocab, layers * ple]);
  add("model.per_layer_model_projection.weight", [layers * ple, hidden]);
  add("model.per_layer_projection_norm.weight", [ple]);
  add("model.norm.weight", [hidden]);
  for (let layer = 0; layer < layers; layer += 1) {
    const prefix = `model.layers.${layer}`;
    const dim = layer === 0 ? 2 : 4;
    add(`${prefix}.input_layernorm.weight`, [hidden]);
    add(`${prefix}.self_attn.q_proj.weight`, [dim, hidden]);
    add(`${prefix}.self_attn.q_norm.weight`, [dim]);
    add(`${prefix}.self_attn.k_proj.weight`, [dim, hidden]);
    add(`${prefix}.self_attn.k_norm.weight`, [dim]);
    add(`${prefix}.self_attn.v_proj.weight`, [dim, hidden]);
    add(`${prefix}.self_attn.o_proj.weight`, [hidden, dim]);
    add(`${prefix}.post_attention_layernorm.weight`, [hidden]);
    add(`${prefix}.pre_feedforward_layernorm.weight`, [hidden]);
    add(`${prefix}.post_feedforward_layernorm.weight`, [hidden]);
    add(`${prefix}.mlp.gate_proj.weight`, [6, hidden]);
    add(`${prefix}.mlp.up_proj.weight`, [6, hidden]);
    add(`${prefix}.mlp.down_proj.weight`, [hidden, 6]);
    add(`${prefix}.per_layer_input_gate.weight`, [ple, hidden]);
    add(`${prefix}.per_layer_projection.weight`, [hidden, ple]);
    add(`${prefix}.post_per_layer_input_norm.weight`, [hidden]);
    add(`${prefix}.layer_scalar`, [1]);
  }
  return { source: "/tmp/tiny-gemma4-text", format: "safetensors", rawMetadata: {}, tensors, config: {
    model_type: "gemma4_text", hidden_size: hidden, vocab_size: vocab, num_hidden_layers: layers,
    num_attention_heads: 1, num_key_value_heads: 1, num_global_key_value_heads: 1,
    head_dim: 2, global_head_dim: 4, intermediate_size: 6, hidden_size_per_layer_input: ple,
    vocab_size_per_layer_input: vocab, num_kv_shared_layers: 0, attention_bias: false,
    attention_k_eq_v: false, enable_moe_block: false, use_double_wide_mlp: false, rms_norm_eps: 1e-6,
    final_logit_softcapping: 30, sliding_window: 4, layer_types: ["sliding_attention", "full_attention"],
    rope_parameters: { sliding_attention: { rope_type: "default", rope_theta: 10_000 }, full_attention: { rope_type: "proportional", rope_theta: 1_000_000, partial_rotary_factor: 0.5 } },
  } };
}

function denseFor(tensor: TensorInfo): DenseF32Tensor {
  const size = tensor.logicalShape.reduce((total, dimension) => total * dimension, 1);
  const values = new Float32Array(size);
  const norm = tensor.name.includes("norm.weight");
  for (let index = 0; index < size; index += 1) values[index] = tensor.name.endsWith("layer_scalar") || norm ? 1 : Math.fround((index % 7 + 1) / 100);
  return { shape: [...tensor.logicalShape], values };
}
