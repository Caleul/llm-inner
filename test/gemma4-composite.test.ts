import assert from "node:assert/strict";
import test from "node:test";
import { executeReferenceF32WithPreparedPrelude } from "../src/executor.js";
import { buildGemma4CompositeProgram, executeGemma4CompositeF32, generateGemma4CompositeF32 } from "../src/gemma4-composite.js";
import {
  buildGemma4CompositeLiteralCalculationProgram,
  executeGemma4CompositeLiteralF32,
  generateGemma4CompositeLiteralF32,
  validateGemma4CompositeLiteralCalculationProgram,
} from "../src/gemma4-composite-literal.js";
import type { DenseF32Tensor, ModelCatalog, TensorInfo } from "../src/types.js";

const preview = { outputRows: 1, inputTerms: 1, includeWeights: false } as const;

test("Gemma 4 composite prelude replaces PAD-backed image/video/audio slots before context PLE and enters text core", () => {
  const catalog = fixture();
  const program = buildGemma4CompositeProgram(catalog, preview);
  assert.equal(program.kind, "gemma4-composite-prelude");
  assert.deepEqual(program.assignments.map((assignment) => assignment.id).slice(0, 8), [
    "composite_placeholder_masks", "composite_block_sequence_ids", "composite_full_attention_mask", "composite_sliding_attention_mask",
    "composite_pad_substitution", "composite_text_embedding", "composite_ple_identity", "composite_image_features",
  ]);
  const ids = [[1, 99, 97, 98, 2]];
  const tensors = materialize(catalog);
  const noFeatures = executeGemma4CompositeF32(program, { inputIds: ids, tensors });
  const result = executeGemma4CompositeF32(program, {
    inputIds: ids,
    tensors,
    pixelValues: patterned([1, 4, 12]),
    imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]],
    pixelValuesVideos: patterned([1, 1, 4, 12]),
    videoPositionIds: [[[[0, 0], [1, 0], [0, 1], [1, 1]]]],
    inputFeatures: patterned([1, 4, 16]),
    inputFeaturesMask: [[true, true, true, true]],
  });
  assert.deepEqual(result.llmInputIds, [[1, 0, 0, 0, 2]]);
  assert.deepEqual(result.text.logits.shape, [1, 5, 6]);
  assert.ok([...result.text.logits.values].every(Number.isFinite));
  assert.equal(result.values.has("image_features"), true);
  assert.equal(result.values.has("video_features"), true);
  assert.equal(result.values.has("audio_features"), true);
  assert.equal(result.values.has("layer_0_ple_input"), true);
  assert.notDeepEqual([...result.values.get("ple_context_packed")!.values], [...noFeatures.values.get("ple_context_packed")!.values], "PLE context must observe the post-scatter embeddings");
});

test("Gemma 4 composite lowers vision blocks into distinct full/sliding masks and rejects ambiguous caller masks", () => {
  const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), tensors = materialize(catalog);
  const result = executeGemma4CompositeF32(program, { inputIds: [[1, 99, 99, 2, 3]], mmTokenTypeIds: [[0, 1, 1, 0, 0]], tensors });
  assert.deepEqual([...result.values.get("vision_block_sequence_ids")!.values], [-1, 0, 0, -1, -1]);
  const full = result.values.get("full_attention_mask")!, sliding = result.values.get("sliding_attention_mask")!;
  assert.equal(full.shape.join(","), "1,1,5,5");
  assert.equal(full.values[1 * 5 + 2], -Infinity, "full attention stays causal across a vision block");
  assert.equal(sliding.values[1 * 5 + 2], 0, "sliding attention permits a future token in the same vision block");
  assert.equal(sliding.values[0 * 5 + 4], -Infinity, "sliding attention rejects unrelated future text");
  assert.ok([...result.text.logits.values].every(Number.isFinite));
  assert.throws(
    () => executeGemma4CompositeF32(program, { inputIds: [[1]], mmTokenTypeIds: [[0]], attentionMask: patterned([1, 1, 1, 1]), tensors }),
    /attentionMask 4-D fornecida pelo chamador/,
  );
  assert.throws(
    () => executeGemma4CompositeF32(program, { inputIds: [[1]], mmTokenTypeIds: [[0]], pastKeyValues: result.text.pastKeyValues, tensors }),
    /somente no prefill sem cache/,
  );
});

test("Gemma 4 composite reuses producer-owned KV after vision-aware prefill without reapplying block masks", () => {
  const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), tensors = materialize(catalog);
  const generated = generateGemma4CompositeF32(program, { inputIds: [[1, 99, 99, 2]], mmTokenTypeIds: [[0, 1, 1, 0]], tensors, maxNewTokens: 2 });
  assert.equal(generated.generatedTokenIds.length, 2);
  assert.equal(generated.stepPastKeyValues.length, 2);
  assert.equal(generated.prefill.text.pastKeyValues.size, 2);
  assert.equal(generated.text.pastKeyValues.size, 2);
  for (const cache of generated.text.pastKeyValues.values()) assert.equal(cache.key.shape[2], 6);
});

test("Gemma 4 composite literal embeds every tower weight and replays multimodal prefill plus cached decode after source bytes are removed", async () => {
  const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), sourceTensors = materialize(catalog);
  const expected = executeGemma4CompositeF32(program, {
    inputIds: [[1, 99, 2]], mmTokenTypeIds: [[0, 1, 0]], tensors: sourceTensors,
    pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]],
  });
  const expectedGeneration = generateGemma4CompositeF32(program, {
    inputIds: [[1, 99, 2]], mmTokenTypeIds: [[0, 1, 0]], tensors: sourceTensors,
    pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]], maxNewTokens: 2,
  });
  const literal = await buildGemma4CompositeLiteralCalculationProgram(program, catalog, {
    async readTensorBytes(info) {
      const tensor = sourceTensors.get(info.name);
      if (!tensor) throw new Error(`source tensor missing: ${info.name}`);
      const bytes = Buffer.alloc(tensor.values.length * 4);
      tensor.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
      return bytes;
    },
  });
  assert.equal(literal.constants.length, catalog.tensors.size);
  assert.equal(literal.storageDecoders.length, catalog.tensors.size);
  assert.equal(JSON.stringify(literal).includes(catalog.source), false);
  assert.equal(literal.program.textProgram.source.path, "embedded://gemma4-composite-literal");

  sourceTensors.clear();
  const replay = executeGemma4CompositeLiteralF32(literal, {
    inputIds: [[1, 99, 2]], mmTokenTypeIds: [[0, 1, 0]],
    pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]],
  });
  const generation = generateGemma4CompositeLiteralF32(literal, {
    inputIds: [[1, 99, 2]], mmTokenTypeIds: [[0, 1, 0]],
    pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]], maxNewTokens: 2,
  });
  assert.deepEqual([...replay.text.logits.values], [...expected.text.logits.values]);
  assert.deepEqual(generation.generatedTokenIds, expectedGeneration.generatedTokenIds);
  assert.deepEqual([...generation.text.logits.values], [...expectedGeneration.text.logits.values]);

  const missingMask = structuredClone(literal);
  missingMask.assignments.composite = missingMask.assignments.composite.filter((assignment) => assignment.id !== "composite_sliding_attention_mask");
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(missingMask), /omite uma transição de máscara ou cache obrigatória/);
  const external = structuredClone(literal);
  external.program.textProgram.source.path = "/checkpoint/model.safetensors";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(external), /reteve uma referência de source checkpoint/);
});

test("Gemma 4 composite fails closed for cardinality, partial modality inputs, and malformed vision blocks", () => {
  const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), tensors = materialize(catalog);
  assert.throws(
    () => executeGemma4CompositeF32(program, { inputIds: [[1, 99, 99, 2]], tensors, pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]] }),
    /placeholder count=2/,
  );
  assert.throws(
    () => executeGemma4CompositeF32(program, { inputIds: [[1]], tensors, inputFeatures: patterned([1, 4, 16]) }),
    /input_features e input_features_mask juntos/,
  );
  assert.throws(
    () => executeGemma4CompositeF32(program, { inputIds: [[1]], tensors, mmTokenTypeIds: [[0, 1]] }),
    /deve acompanhar input_ids/,
  );
  assert.throws(
    () => executeReferenceF32WithPreparedPrelude(program.textProgram, { inputIds: [[1]], tensors }, new Map([["hidden_states_0", patterned([1, 1, 4])]])),
    /saída obrigatória ple_token_identity/,
  );
});

function fixture(): ModelCatalog {
  const tensors = new Map<string, TensorInfo>();
  const add = (name: string, shape: number[]): void => { tensors.set(name, { name, storageDtype: "F32", storageShape: shape, logicalShape: shape }); };
  const clipped = (prefix: string, shape: number[]): void => { add(`${prefix}.linear.weight`, shape); for (const suffix of ["input_min", "input_max", "output_min", "output_max"]) add(`${prefix}.${suffix}`, []); };
  const hidden = 4, layers = 2, ple = 1, vocab = 6, text = "model.language_model";
  add(`${text}.embed_tokens.weight`, [vocab, hidden]); add(`${text}.embed_tokens_per_layer.weight`, [vocab, layers * ple]); add(`${text}.per_layer_model_projection.weight`, [layers * ple, hidden]); add(`${text}.per_layer_projection_norm.weight`, [ple]); add(`${text}.norm.weight`, [hidden]);
  for (let layer = 0; layer < layers; layer += 1) {
    const textLayer = `${text}.layers.${layer}`;
    for (const norm of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"]) add(`${textLayer}.${norm}.weight`, [hidden]);
    for (const projection of ["q_proj", "k_proj", "v_proj"]) add(`${textLayer}.self_attn.${projection}.weight`, [hidden, hidden]);
    add(`${textLayer}.self_attn.q_norm.weight`, [hidden]); add(`${textLayer}.self_attn.k_norm.weight`, [hidden]); add(`${textLayer}.self_attn.o_proj.weight`, [hidden, hidden]); add(`${textLayer}.per_layer_input_gate.weight`, [ple, hidden]); add(`${textLayer}.per_layer_projection.weight`, [hidden, ple]); add(`${textLayer}.layer_scalar`, [1]);
    for (const projection of ["gate_proj", "up_proj"]) add(`${textLayer}.mlp.${projection}.weight`, [8, hidden]); add(`${textLayer}.mlp.down_proj.weight`, [hidden, 8]);
  }
  const vision = "model.vision_tower";
  add(`${vision}.patch_embedder.input_proj.weight`, [hidden, 12]); add(`${vision}.patch_embedder.position_embedding_table`, [2, 4, hidden]);
  const visionLayer = `${vision}.encoder.layers.0`;
  for (const norm of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]) add(`${visionLayer}.${norm}.weight`, [hidden]);
  add(`${visionLayer}.self_attn.q_norm.weight`, [hidden]); add(`${visionLayer}.self_attn.k_norm.weight`, [hidden]);
  for (const projection of ["q_proj", "k_proj", "v_proj", "o_proj"]) clipped(`${visionLayer}.self_attn.${projection}`, [hidden, hidden]);
  clipped(`${visionLayer}.mlp.gate_proj`, [8, hidden]); clipped(`${visionLayer}.mlp.up_proj`, [8, hidden]); clipped(`${visionLayer}.mlp.down_proj`, [hidden, 8]); add("model.embed_vision.embedding_projection.weight", [hidden, hidden]);
  const audio = "model.audio_tower";
  add(`${audio}.subsample_conv_projection.layer0.conv.weight`, [1, 1, 3, 3]); add(`${audio}.subsample_conv_projection.layer0.norm.weight`, [1]); add(`${audio}.subsample_conv_projection.layer1.conv.weight`, [1, 1, 3, 3]); add(`${audio}.subsample_conv_projection.layer1.norm.weight`, [1]); add(`${audio}.subsample_conv_projection.input_proj_linear.weight`, [hidden, hidden]);
  const audioLayer = `${audio}.layers.0`;
  for (const norm of ["norm_pre_attn", "norm_post_attn", "norm_out"]) add(`${audioLayer}.${norm}.weight`, [hidden]); add(`${audioLayer}.self_attn.per_dim_scale`, [hidden]); add(`${audioLayer}.self_attn.relative_k_proj.weight`, [hidden, hidden]);
  for (const projection of ["q_proj", "k_proj", "v_proj", "post"]) clipped(`${audioLayer}.self_attn.${projection}`, [hidden, hidden]);
  for (const ffn of ["feed_forward1", "feed_forward2"]) { add(`${audioLayer}.${ffn}.pre_layer_norm.weight`, [hidden]); add(`${audioLayer}.${ffn}.post_layer_norm.weight`, [hidden]); clipped(`${audioLayer}.${ffn}.ffw_layer_1`, [hidden * 4, hidden]); clipped(`${audioLayer}.${ffn}.ffw_layer_2`, [hidden, hidden * 4]); }
  add(`${audioLayer}.lconv1d.pre_layer_norm.weight`, [hidden]); add(`${audioLayer}.lconv1d.conv_norm.weight`, [hidden]); add(`${audioLayer}.lconv1d.depthwise_conv1d.weight`, [hidden, 1, 5]); clipped(`${audioLayer}.lconv1d.linear_start`, [hidden * 2, hidden]); clipped(`${audioLayer}.lconv1d.linear_end`, [hidden, hidden]); add(`${audio}.output_proj.weight`, [hidden, hidden]); add(`${audio}.output_proj.bias`, [hidden]); add("model.embed_audio.embedding_projection.weight", [hidden, hidden]);
  return { source: "/tmp/tiny-gemma4-composite", format: "safetensors", rawMetadata: {}, tensors, config: {
    model_type: "gemma4", image_token_id: 99, video_token_id: 97, audio_token_id: 98,
    vision_config: { model_type: "gemma4_vision", hidden_size: hidden, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1, head_dim: hidden, intermediate_size: 8, patch_size: 2, position_embedding_size: 4, default_output_length: 1, pooling_kernel_size: 2, attention_bias: false, hidden_activation: "gelu_pytorch_tanh", use_clipped_linears: true, rms_norm_eps: 1e-6, rope_parameters: { rope_type: "default", rope_theta: 100 } },
    audio_config: { model_type: "gemma4_audio", hidden_size: hidden, num_hidden_layers: 1, num_attention_heads: 1, output_proj_dims: hidden, attention_chunk_size: 2, attention_context_left: 1, attention_context_right: 0, attention_logit_cap: 50, attention_invalid_logits_value: -1e9, gradient_clipping: 1e10, conv_kernel_size: 5, residual_weight: 0.5, subsampling_conv_channels: [1, 1], hidden_act: "silu", use_clipped_linears: true, rms_norm_eps: 1e-6 },
    text_config: { model_type: "gemma4_text", hidden_size: hidden, vocab_size: vocab, pad_token_id: 0, num_hidden_layers: layers, num_attention_heads: 1, num_key_value_heads: 1, global_head_dim: hidden, head_dim: hidden, intermediate_size: 8, num_kv_shared_layers: 0, hidden_size_per_layer_input: ple, vocab_size_per_layer_input: vocab, attention_bias: false, attention_k_eq_v: false, enable_moe_block: false, use_double_wide_mlp: false, rms_norm_eps: 1e-6, sliding_window: 4, layer_types: ["full_attention", "sliding_attention"], rope_parameters: { sliding_attention: { rope_type: "default", rope_theta: 10_000 }, full_attention: { rope_type: "proportional", rope_theta: 1_000_000, partial_rotary_factor: 1 } } },
  } };
}

function patterned(shape: number[]): DenseF32Tensor { return { shape, values: Float32Array.from({ length: shape.reduce((total, dimension) => total * dimension, 1) }, (_, index) => Math.fround((index % 9 + 1) / 25)) }; }
function materialize(catalog: ModelCatalog): Map<string, DenseF32Tensor> { const result = new Map<string, DenseF32Tensor>(); for (const entry of catalog.tensors.values()) { const size = entry.logicalShape.reduce((total, dimension) => total * dimension, 1) || 1; const values = new Float32Array(size); if (entry.name.endsWith("input_min") || entry.name.endsWith("output_min")) values[0] = -100; else if (entry.name.endsWith("input_max") || entry.name.endsWith("output_max")) values[0] = 100; else if (entry.name.endsWith("norm.weight")) values.fill(1); else if (entry.name.endsWith("layer_scalar")) values.fill(1); else for (let index = 0; index < size; index += 1) values[index] = Math.fround((index % 5 + 1) / 50); result.set(entry.name, { shape: [...entry.logicalShape], values }); } return result; }
