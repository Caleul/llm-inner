import assert from "node:assert/strict";
import test from "node:test";
import { buildGemma4VisionProgram, executeGemma4VisionF32, scatterGemma4ImageFeaturesF32 } from "../src/gemma4-vision.js";
import type { DenseF32Tensor, ModelCatalog, TensorInfo } from "../src/types.js";

test("Gemma 4 vision lowerer makes 2-D RoPE, clipping, pooling, projection, and image scatter explicit", () => {
  const catalog = fixture();
  const program = buildGemma4VisionProgram(catalog);
  assert.equal(program.kind, "gemma4-vision-features");
  assert.deepEqual(program.assignments.slice(0, 4).map((assignment) => assignment.operation), ["pixel-affine", "linear", "position-embedding-2d", "add"]);
  assert.ok(program.assignments.some((assignment) => assignment.operation === "multidimensional-rope"));
  assert.ok(program.assignments.some((assignment) => assignment.operation === "clipped-linear" && assignment.tensors?.some((tensor) => tensor.name.endsWith("input_min"))));
  assert.equal(program.assignments.at(-1)?.operation, "masked-scatter-image-features");

  const result = executeGemma4VisionF32(program, { pixelValues: dense([1, 4, 12], 0.25), pixelPositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]], tensors: materialize(catalog) });
  assert.deepEqual(result.imageFeatures.shape, [1, 4]);
  assert.ok([...result.imageFeatures.values].every(Number.isFinite));
  assert.equal(result.values.has("vision_layer_0_q_rotated"), true);
  assert.equal(result.values.has("vision_soft_tokens"), true);
  const normalizedSoftTokens = result.values.get("vision_soft_tokens_normalized")!;
  const rms = Math.sqrt([...normalizedSoftTokens.values].reduce((sum, value) => sum + value ** 2, 0) / normalizedSoftTokens.values.length);
  assert.ok(Math.abs(rms - 1) < 1e-4, `expected unscaled multimodal RMSNorm, got RMS=${rms}`);
  for (const assignment of program.assignments.filter((assignment) => assignment.operation !== "masked-scatter-image-features")) {
    assert.equal(result.values.has(assignment.output), true, `missing named output ${assignment.output}`);
  }

  const injected = scatterGemma4ImageFeaturesF32(dense([1, 3, 4], 0), [[1, 99, 2]], 99, result.imageFeatures);
  assert.deepEqual([...injected.values.slice(4, 8)], [...result.imageFeatures.values]);
});

test("Gemma 4 vision rejects non-pair padding coordinates and mismatched image placeholders", () => {
  const catalog = fixture();
  const program = buildGemma4VisionProgram(catalog);
  assert.throws(() => executeGemma4VisionF32(program, { pixelValues: dense([1, 4, 12], 0), pixelPositionIds: [[[0, 0], [-1, 0], [0, 1], [1, 1]]], tensors: materialize(catalog) }), /padding é somente/);
  assert.throws(() => scatterGemma4ImageFeaturesF32(dense([1, 2, 4], 0), [[99, 99]], 99, dense([1, 4], 1)), /placeholder count=2/);
});

test("Gemma 4 vision zeros padding patches before spatial pooling", () => {
  const catalog = fixture();
  const program = buildGemma4VisionProgram(catalog);
  const positions = [[[0, 0], [1, 0], [0, 1], [-1, -1]]];
  const baseline = dense([1, 4, 12], 0.25);
  const paddedNoise = dense([1, 4, 12], 0.25);
  paddedNoise.values.fill(100, 3 * 12);
  const tensors = materialize(catalog);
  const expected = executeGemma4VisionF32(program, { pixelValues: baseline, pixelPositionIds: positions, tensors });
  const actual = executeGemma4VisionF32(program, { pixelValues: paddedNoise, pixelPositionIds: positions, tensors });
  assert.deepEqual([...actual.imageFeatures.values], [...expected.imageFeatures.values]);
});

test("Gemma 4 vision dispatches BF16 numeric policy across complete operation classes", () => {
  const catalog = fixture();
  (catalog.config.vision_config as Record<string, unknown>).dtype = "bfloat16";
  const program = buildGemma4VisionProgram(catalog);
  assert.equal(program.runtimeDtype, "BF16");
  const linears = program.assignments.filter((assignment) => assignment.operation === "linear" || assignment.operation === "clipped-linear");
  assert.ok(linears.length > 0);
  for (const assignment of linears) {
    assert.equal(assignment.dtypePolicy?.reduction?.kind, "arm-neon-bf16-dot-fma", assignment.id);
    assert.equal(assignment.dtypePolicy?.outputDtype, "BF16", assignment.id);
  }
  for (const assignment of program.assignments.filter((entry) => entry.operation === "rms-norm")) {
    assert.equal(assignment.dtypePolicy?.reduction?.kind, "pytorch-cpu-f32-cascade-sum", assignment.id);
  }
  assert.deepEqual(
    program.assignments.filter((assignment) => assignment.dtypePolicy?.accumulationDtype === "runtime-defined").map((assignment) => assignment.operation),
    ["attention-score-matmul", "attention-value-matmul"],
  );
  assert.equal(program.assignments.find((assignment) => assignment.operation === "pool-by-position")?.dtypePolicy?.reduction?.kind, "ordered-fma");
});

function fixture(): ModelCatalog {
  const tensors = new Map<string, TensorInfo>();
  const add = (name: string, shape: number[]): void => { tensors.set(name, { name, storageDtype: "F32", storageShape: shape, logicalShape: shape }); };
  const clipped = (prefix: string, shape: number[]): void => { add(`${prefix}.linear.weight`, shape); for (const suffix of ["input_min", "input_max", "output_min", "output_max"]) add(`${prefix}.${suffix}`, []); };
  const hidden = 4, layers = 1, ple = 1;
  const text = "model.language_model";
  add(`${text}.embed_tokens.weight`, [4, hidden]); add(`${text}.embed_tokens_per_layer.weight`, [4, layers * ple]); add(`${text}.per_layer_model_projection.weight`, [layers * ple, hidden]); add(`${text}.per_layer_projection_norm.weight`, [ple]); add(`${text}.norm.weight`, [hidden]);
  const textLayer = `${text}.layers.0`;
  for (const norm of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"]) add(`${textLayer}.${norm}.weight`, [hidden]);
  for (const projection of ["q_proj", "k_proj", "v_proj"]) add(`${textLayer}.self_attn.${projection}.weight`, [hidden, hidden]);
  add(`${textLayer}.self_attn.q_norm.weight`, [hidden]); add(`${textLayer}.self_attn.k_norm.weight`, [hidden]); add(`${textLayer}.self_attn.o_proj.weight`, [hidden, hidden]);
  add(`${textLayer}.per_layer_input_gate.weight`, [ple, hidden]); add(`${textLayer}.per_layer_projection.weight`, [hidden, ple]); add(`${textLayer}.layer_scalar`, [1]);
  for (const projection of ["gate_proj", "up_proj"]) add(`${textLayer}.mlp.${projection}.weight`, [8, hidden]); add(`${textLayer}.mlp.down_proj.weight`, [hidden, 8]);
  const vision = "model.vision_tower";
  add(`${vision}.patch_embedder.input_proj.weight`, [hidden, 12]); add(`${vision}.patch_embedder.position_embedding_table`, [2, 4, hidden]);
  const visionLayer = `${vision}.encoder.layers.0`;
  for (const norm of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]) add(`${visionLayer}.${norm}.weight`, [hidden]);
  add(`${visionLayer}.self_attn.q_norm.weight`, [hidden]); add(`${visionLayer}.self_attn.k_norm.weight`, [hidden]);
  for (const projection of ["q_proj", "k_proj", "v_proj", "o_proj"]) clipped(`${visionLayer}.self_attn.${projection}`, [hidden, hidden]);
  clipped(`${visionLayer}.mlp.gate_proj`, [8, hidden]); clipped(`${visionLayer}.mlp.up_proj`, [8, hidden]); clipped(`${visionLayer}.mlp.down_proj`, [hidden, 8]);
  add("model.embed_vision.embedding_projection.weight", [hidden, hidden]);
  const audio = "model.audio_tower";
  add(`${audio}.subsample_conv_projection.layer0.conv.weight`, [1, 1, 3, 3]); add(`${audio}.subsample_conv_projection.layer0.norm.weight`, [1]); add(`${audio}.subsample_conv_projection.layer1.conv.weight`, [1, 1, 3, 3]); add(`${audio}.subsample_conv_projection.layer1.norm.weight`, [1]); add(`${audio}.subsample_conv_projection.input_proj_linear.weight`, [hidden, hidden]);
  const audioLayer = `${audio}.layers.0`;
  for (const norm of ["norm_pre_attn", "norm_post_attn", "norm_out"]) add(`${audioLayer}.${norm}.weight`, [hidden]); add(`${audioLayer}.self_attn.per_dim_scale`, [hidden]); add(`${audioLayer}.self_attn.relative_k_proj.weight`, [hidden, hidden]);
  for (const projection of ["q_proj", "k_proj", "v_proj", "post"]) clipped(`${audioLayer}.self_attn.${projection}`, [hidden, hidden]);
  for (const ffn of ["feed_forward1", "feed_forward2"]) { add(`${audioLayer}.${ffn}.pre_layer_norm.weight`, [hidden]); add(`${audioLayer}.${ffn}.post_layer_norm.weight`, [hidden]); clipped(`${audioLayer}.${ffn}.ffw_layer_1`, [hidden * 4, hidden]); clipped(`${audioLayer}.${ffn}.ffw_layer_2`, [hidden, hidden * 4]); }
  add(`${audioLayer}.lconv1d.pre_layer_norm.weight`, [hidden]); add(`${audioLayer}.lconv1d.conv_norm.weight`, [hidden]); add(`${audioLayer}.lconv1d.depthwise_conv1d.weight`, [hidden, 1, 5]); clipped(`${audioLayer}.lconv1d.linear_start`, [hidden * 2, hidden]); clipped(`${audioLayer}.lconv1d.linear_end`, [hidden, hidden]); add(`${audio}.output_proj.weight`, [hidden, hidden]); add(`${audio}.output_proj.bias`, [hidden]); add("model.embed_audio.embedding_projection.weight", [hidden, hidden]);
  return { source: "/tmp/tiny-gemma4-vision", format: "safetensors", rawMetadata: {}, tensors, config: {
    model_type: "gemma4", image_token_id: 99, audio_token_id: 98,
    vision_config: { model_type: "gemma4_vision", hidden_size: hidden, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1, head_dim: hidden, intermediate_size: 8, patch_size: 2, position_embedding_size: 4, default_output_length: 1, pooling_kernel_size: 2, attention_bias: false, hidden_activation: "gelu_pytorch_tanh", use_clipped_linears: true, rms_norm_eps: 1e-6, rope_parameters: { rope_type: "default", rope_theta: 100 } },
    audio_config: { model_type: "gemma4_audio", hidden_size: hidden, num_hidden_layers: 1, num_attention_heads: 1, output_proj_dims: hidden, attention_chunk_size: 2, attention_context_left: 1, attention_context_right: 0, attention_logit_cap: 50, conv_kernel_size: 5, residual_weight: 0.5, subsampling_conv_channels: [1, 1], hidden_act: "silu" },
    text_config: { model_type: "gemma4_text", hidden_size: hidden, vocab_size: 4, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1, global_head_dim: hidden, head_dim: hidden, intermediate_size: 8, num_kv_shared_layers: 0, hidden_size_per_layer_input: ple, vocab_size_per_layer_input: 4, attention_k_eq_v: false, use_double_wide_mlp: false, layer_types: ["sliding_attention"], rope_parameters: { sliding_attention: { rope_type: "default", rope_theta: 10_000 }, full_attention: { rope_type: "proportional", rope_theta: 1_000_000, partial_rotary_factor: 1 } } },
  } };
}

function dense(shape: number[], value: number): DenseF32Tensor { return { shape, values: Float32Array.from({ length: shape.reduce((total, dimension) => total * dimension, 1) }, () => value) }; }
function materialize(catalog: ModelCatalog): Map<string, DenseF32Tensor> { const result = new Map<string, DenseF32Tensor>(); for (const entry of catalog.tensors.values()) { const size = entry.logicalShape.reduce((total, dimension) => total * dimension, 1) || 1; const values = new Float32Array(size); if (entry.name.endsWith("input_min") || entry.name.endsWith("output_min")) values[0] = -100; else if (entry.name.endsWith("input_max") || entry.name.endsWith("output_max")) values[0] = 100; else if (entry.name.endsWith("norm.weight")) values.fill(1); else for (let index = 0; index < size; index += 1) values[index] = Math.fround((index % 5 + 1) / 50); result.set(entry.name, { shape: [...entry.logicalShape], values }); } return result; }
