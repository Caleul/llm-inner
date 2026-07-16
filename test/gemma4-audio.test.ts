import assert from "node:assert/strict";
import test from "node:test";
import { buildGemma4AudioProgram, executeGemma4AudioF32, scatterGemma4AudioFeaturesF32 } from "../src/gemma4-audio.js";
import type { DenseF32Tensor, ModelCatalog, TensorInfo } from "../src/types.js";

test("Gemma 4 audio lowering makes subsampling, chunk-relative attention, local convolution, projection, and scatter explicit", () => {
  const catalog = fixture();
  const program = buildGemma4AudioProgram(catalog);
  assert.equal(program.kind, "gemma4-audio-features");
  assert.ok(program.assignments.some((assignment) => assignment.operation === "chunked-relative-attention"));
  assert.ok(program.assignments.some((assignment) => assignment.operation === "causal-depthwise-convolution"));
  assert.ok(program.assignments.some((assignment) => assignment.operation === "clipped-linear" && assignment.tensors?.some((tensor) => tensor.name.endsWith("output_max"))));
  const result = executeGemma4AudioF32(program, { inputFeatures: patterned([1, 4, 16]), inputFeaturesMask: [[true, true, true, false]], tensors: materialize(catalog) });
  assert.deepEqual(result.audioFeatures.shape, [1, 4]);
  assert.deepEqual(result.outputMask, [[true]]);
  assert.ok([...result.audioFeatures.values].every(Number.isFinite));
  const normalizedOutput = result.values.get("audio_output_normalized")!;
  const rms = Math.sqrt([...normalizedOutput.values].reduce((sum, value) => sum + value ** 2, 0) / normalizedOutput.values.length);
  assert.ok(Math.abs(rms - 1) < 1e-3, `expected unscaled multimodal RMSNorm, got RMS=${rms}`);
  for (const assignment of program.assignments.filter((assignment) => assignment.operation !== "masked-scatter-audio-features")) assert.equal(result.values.has(assignment.output), true, `missing named output ${assignment.output}`);
  const injected = scatterGemma4AudioFeaturesF32(dense([1, 3, 4], 0), [[1, 98, 2]], 98, result.audioFeatures);
  assert.deepEqual([...injected.values.slice(4, 8)], [...result.audioFeatures.values]);
});

test("Gemma 4 audio masks invalid frames before subsampling and rejects wrong placeholder cardinality", () => {
  const catalog = fixture(), program = buildGemma4AudioProgram(catalog), tensors = materialize(catalog);
  const baseline = patterned([1, 4, 16]);
  const paddedNoise = patterned([1, 4, 16]); paddedNoise.values.fill(99, 3 * 16);
  const mask = [[true, true, true, false]];
  const expected = executeGemma4AudioF32(program, { inputFeatures: baseline, inputFeaturesMask: mask, tensors });
  const actual = executeGemma4AudioF32(program, { inputFeatures: paddedNoise, inputFeaturesMask: mask, tensors });
  assert.deepEqual([...actual.audioFeatures.values], [...expected.audioFeatures.values]);
  assert.throws(() => scatterGemma4AudioFeaturesF32(dense([1, 2, 4], 0), [[98, 98]], 98, dense([1, 4], 1)), /placeholder count=2/);
  assert.throws(() => executeGemma4AudioF32(program, { inputFeatures: dense([1, 4, 16], 0), inputFeaturesMask: [[true]], tensors }), /input_features \[B,T,F\].*input_features_mask/);
});

test("Gemma 4 audio refuses a missing runtime numeric contract instead of applying a local default", () => {
  const catalog = fixture();
  delete (catalog.config.audio_config as Record<string, unknown>).gradient_clipping;
  assert.throws(() => buildGemma4AudioProgram(catalog), /gradient_clipping positivo/);
});

function fixture(): ModelCatalog {
  const tensors = new Map<string, TensorInfo>();
  const add = (name: string, shape: number[]): void => { tensors.set(name, { name, storageDtype: "F32", storageShape: shape, logicalShape: shape }); };
  const clipped = (prefix: string, shape: number[]): void => { add(`${prefix}.linear.weight`, shape); for (const suffix of ["input_min", "input_max", "output_min", "output_max"]) add(`${prefix}.${suffix}`, []); };
  const hidden = 4, layers = 1, ple = 1, text = "model.language_model";
  add(`${text}.embed_tokens.weight`, [4, hidden]); add(`${text}.embed_tokens_per_layer.weight`, [4, layers * ple]); add(`${text}.per_layer_model_projection.weight`, [layers * ple, hidden]); add(`${text}.per_layer_projection_norm.weight`, [ple]); add(`${text}.norm.weight`, [hidden]);
  const textLayer = `${text}.layers.0`; for (const norm of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"]) add(`${textLayer}.${norm}.weight`, [hidden]); for (const projection of ["q_proj", "k_proj", "v_proj"]) add(`${textLayer}.self_attn.${projection}.weight`, [hidden, hidden]); add(`${textLayer}.self_attn.q_norm.weight`, [hidden]); add(`${textLayer}.self_attn.k_norm.weight`, [hidden]); add(`${textLayer}.self_attn.o_proj.weight`, [hidden, hidden]); add(`${textLayer}.per_layer_input_gate.weight`, [ple, hidden]); add(`${textLayer}.per_layer_projection.weight`, [hidden, ple]); add(`${textLayer}.layer_scalar`, [1]); for (const projection of ["gate_proj", "up_proj"]) add(`${textLayer}.mlp.${projection}.weight`, [8, hidden]); add(`${textLayer}.mlp.down_proj.weight`, [hidden, 8]);
  const vision = "model.vision_tower"; add(`${vision}.patch_embedder.input_proj.weight`, [hidden, 12]); add(`${vision}.patch_embedder.position_embedding_table`, [2, 4, hidden]); const visionLayer = `${vision}.encoder.layers.0`; for (const norm of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]) add(`${visionLayer}.${norm}.weight`, [hidden]); add(`${visionLayer}.self_attn.q_norm.weight`, [hidden]); add(`${visionLayer}.self_attn.k_norm.weight`, [hidden]); for (const projection of ["q_proj", "k_proj", "v_proj", "o_proj"]) clipped(`${visionLayer}.self_attn.${projection}`, [hidden, hidden]); clipped(`${visionLayer}.mlp.gate_proj`, [8, hidden]); clipped(`${visionLayer}.mlp.up_proj`, [8, hidden]); clipped(`${visionLayer}.mlp.down_proj`, [hidden, 8]); add("model.embed_vision.embedding_projection.weight", [hidden, hidden]);
  const audio = "model.audio_tower"; add(`${audio}.subsample_conv_projection.layer0.conv.weight`, [1, 1, 3, 3]); add(`${audio}.subsample_conv_projection.layer0.norm.weight`, [1]); add(`${audio}.subsample_conv_projection.layer1.conv.weight`, [1, 1, 3, 3]); add(`${audio}.subsample_conv_projection.layer1.norm.weight`, [1]); add(`${audio}.subsample_conv_projection.input_proj_linear.weight`, [hidden, hidden]); const audioLayer = `${audio}.layers.0`; for (const norm of ["norm_pre_attn", "norm_post_attn", "norm_out"]) add(`${audioLayer}.${norm}.weight`, [hidden]); add(`${audioLayer}.self_attn.per_dim_scale`, [hidden]); add(`${audioLayer}.self_attn.relative_k_proj.weight`, [hidden, hidden]); for (const projection of ["q_proj", "k_proj", "v_proj", "post"]) clipped(`${audioLayer}.self_attn.${projection}`, [hidden, hidden]); for (const ffn of ["feed_forward1", "feed_forward2"]) { add(`${audioLayer}.${ffn}.pre_layer_norm.weight`, [hidden]); add(`${audioLayer}.${ffn}.post_layer_norm.weight`, [hidden]); clipped(`${audioLayer}.${ffn}.ffw_layer_1`, [hidden * 4, hidden]); clipped(`${audioLayer}.${ffn}.ffw_layer_2`, [hidden, hidden * 4]); } add(`${audioLayer}.lconv1d.pre_layer_norm.weight`, [hidden]); add(`${audioLayer}.lconv1d.conv_norm.weight`, [hidden]); add(`${audioLayer}.lconv1d.depthwise_conv1d.weight`, [hidden, 1, 5]); clipped(`${audioLayer}.lconv1d.linear_start`, [hidden * 2, hidden]); clipped(`${audioLayer}.lconv1d.linear_end`, [hidden, hidden]); add(`${audio}.output_proj.weight`, [hidden, hidden]); add(`${audio}.output_proj.bias`, [hidden]); add("model.embed_audio.embedding_projection.weight", [hidden, hidden]);
  return { source: "/tmp/tiny-gemma4-audio", format: "safetensors", rawMetadata: {}, tensors, config: { model_type: "gemma4", image_token_id: 99, audio_token_id: 98, vision_config: { model_type: "gemma4_vision", hidden_size: hidden, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1, head_dim: hidden, intermediate_size: 8, patch_size: 2, position_embedding_size: 4, default_output_length: 1, pooling_kernel_size: 2, attention_bias: false, hidden_activation: "gelu_pytorch_tanh", use_clipped_linears: true, rms_norm_eps: 1e-6, rope_parameters: { rope_type: "default", rope_theta: 100 } }, audio_config: { model_type: "gemma4_audio", hidden_size: hidden, num_hidden_layers: 1, num_attention_heads: 1, output_proj_dims: hidden, attention_chunk_size: 2, attention_context_left: 1, attention_context_right: 0, attention_logit_cap: 50, attention_invalid_logits_value: -1e9, gradient_clipping: 1e10, conv_kernel_size: 5, residual_weight: 0.5, subsampling_conv_channels: [1, 1], hidden_act: "silu", use_clipped_linears: true, rms_norm_eps: 1e-6 }, text_config: { model_type: "gemma4_text", hidden_size: hidden, vocab_size: 4, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1, global_head_dim: hidden, head_dim: hidden, intermediate_size: 8, num_kv_shared_layers: 0, hidden_size_per_layer_input: ple, vocab_size_per_layer_input: 4, attention_k_eq_v: false, use_double_wide_mlp: false, layer_types: ["sliding_attention"], rope_parameters: { sliding_attention: { rope_type: "default", rope_theta: 10_000 }, full_attention: { rope_type: "proportional", rope_theta: 1_000_000, partial_rotary_factor: 1 } } } } };
}
function dense(shape: number[], value: number): DenseF32Tensor { return { shape, values: Float32Array.from({ length: shape.reduce((total, dimension) => total * dimension, 1) }, () => value) }; }
function patterned(shape: number[]): DenseF32Tensor { return { shape, values: Float32Array.from({ length: shape.reduce((total, dimension) => total * dimension, 1) }, (_, index) => Math.fround((index % 9 + 1) / 25)) }; }
function materialize(catalog: ModelCatalog): Map<string, DenseF32Tensor> { const result = new Map<string, DenseF32Tensor>(); for (const entry of catalog.tensors.values()) { const size = entry.logicalShape.reduce((total, dimension) => total * dimension, 1) || 1; const values = new Float32Array(size); if (entry.name.endsWith("input_min") || entry.name.endsWith("output_min")) values[0] = -100; else if (entry.name.endsWith("input_max") || entry.name.endsWith("output_max")) values[0] = 100; else if (entry.name.endsWith("norm.weight")) values.fill(1); else for (let index = 0; index < size; index += 1) values[index] = Math.fround((index % 5 + 1) / 50); result.set(entry.name, { shape: [...entry.logicalShape], values }); } return result; }
