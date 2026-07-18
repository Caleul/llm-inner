import { inspectGemma4PackageContract, type Gemma4AudioTowerContract } from "./gemma4-contract.js";
import { pytorchCpuCascadeSquareSumF32, pytorchPowNegativeHalfF32 } from "./executor.js";
import { armNeonBf16DotF32, pytorchCpuBf16GemmIlp4F32, pytorchCpuBf16WelfordMomentsF32 } from "./native-reductions.js";
import { roundDenseF32ToBF16 } from "./paged-dense.js";
import { sleefCosF32, sleefExpF32, sleefSinF32, sleefTanhF32 } from "./sleef-f32.js";
import type { DenseF32Tensor, DtypePolicy, ModelCatalog, TensorInfo, TensorRef } from "./types.js";
import { roundF32ToBF16 } from "./utils.js";

const ORDERED_F32_REDUCTION = { kind: "ordered-scalar", indexOrder: "ascending" } as const;
const BF16_ARM_DOT_REDUCTION = {
  kind: "arm-neon-bf16-dot-fma", laneCount: 32, registerCount: 8, lanesPerRegister: 4,
  inputLane: "index-modulo-vector-lane-count", horizontalFold: "pairwise",
} as const;
const BF16_GEMM_ILP4_REDUCTION = {
  kind: "interleaved-f32-lanes", laneCount: 4,
  inputLane: "index-modulo-lane-count", laneReductionOrder: "ascending",
} as const;
const BF16_WELFORD_REDUCTION = {
  kind: "pytorch-cpu-bf16-welford", inputVectorLanes: 8, accumulatorVectorLanes: 4,
  chunkVectors: 16, vectorMergeOrder: "low-then-high", laneFold: "ascending",
  secondPass: "x-times-scale-plus-bias-times-gamma",
} as const;
const F32_POLICY: DtypePolicy = { inputDtype: "F32", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" };
const F32_TO_BF16_POLICY: DtypePolicy = { inputDtype: "F32", computeDtype: "F32", accumulationDtype: "none", outputDtype: "BF16" };
const BF16_POLICY: DtypePolicy = { inputDtype: "BF16", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "BF16" };
const BF16_LINEAR_POLICY: DtypePolicy = { ...BF16_POLICY, reduction: BF16_ARM_DOT_REDUCTION };
const BF16_RMS_POLICY = {
  ...BF16_POLICY,
  reduction: {
    kind: "pytorch-cpu-f32-cascade-sum", vectorLanes: 4, ilpFactor: 4, cascadeLevels: 4,
    minimumLevelStep: 16, registerFold: "ascending", laneFold: "ascending",
  },
} as const;

/**
 * Literal, named execution boundary for the registered Gemma 4 audio encoder.
 * This is deliberately an audio-feature program, not a generic Conformer:
 * its blocked relative attention, clips, mask subsampling, causal depthwise
 * convolution and residual ordering are the pinned Gemma 4 runtime contract.
 */
export interface Gemma4AudioProgram {
  kind: "gemma4-audio-features";
  sourceFormat: "safetensors";
  attentionMaskContract: "transformers-eager-additive-mask-logical-not-v1";
  tower: Gemma4AudioTowerContract;
  textHiddenSize: number;
  rmsNormEpsilon: number;
  gradientClipping: number;
  invalidAttentionLogit: number;
  runtimeDtype: "F32" | "BF16";
  assignments: Gemma4AudioAssignment[];
  output: "audio_features";
}

export interface Gemma4AudioAssignment {
  id: string;
  operation:
    | "mask-input-features" | "conv2d-stride2" | "layer-norm-channels" | "relu"
    | "reshape-conv-features" | "linear" | "relative-position-encoding"
    | "rms-norm" | "clip" | "clipped-linear" | "silu" | "scale-f32" | "add" | "per-dim-softplus-scale"
    | "split-gated-linear-unit" | "causal-depthwise-convolution"
    | "chunked-attention-content-matmul" | "relative-attention-position-matmul" | "relative-attention-shift"
    | "attention-logit-add" | "attention-softcap" | "chunked-attention-mask"
    | "chunked-relative-attention-softmax" | "chunked-relative-attention-values"
    | "cast-bf16" | "subsample-mask" | "strip-padding"
    | "masked-scatter-audio-features";
  inputs: string[];
  output: string;
  tensors?: TensorRef[];
  semantics?: string;
  /** Authoritative modal token selected by stable row-major scatter. */
  placeholderTokenId?: number | undefined;
  dtypePolicy?: DtypePolicy;
}

export interface Gemma4AudioExecutionRequest {
  /** [batch, frames, features], before the two stride-2 subsampling layers. */
  inputFeatures: DenseF32Tensor;
  /** [batch][frames], true only for real audio frames. */
  inputFeaturesMask: readonly boolean[][];
  tensors: ReadonlyMap<string, DenseF32Tensor>;
}

export interface Gemma4AudioExecutionResult {
  values: ReadonlyMap<string, DenseF32Tensor>;
  outputMask: readonly boolean[][];
  /** Flattened, valid, language-projected audio soft tokens [tokens, text_hidden]. */
  audioFeatures: DenseF32Tensor;
}

export interface Gemma4AudioDerivedAttentionStages {
  shiftedPositionScores: DenseF32Tensor;
  logits: DenseF32Tensor;
  softcapped: DenseF32Tensor;
  maskedScores: DenseF32Tensor;
}

/** Declared F32 -> BF16 assignment used by every narrowed audio value. */
export function executeGemma4AudioBf16CastF32(input: DenseF32Tensor): DenseF32Tensor {
  return roundDenseF32ToBF16(input);
}

export function buildGemma4AudioProgram(catalog: ModelCatalog): Gemma4AudioProgram {
  const contract = inspectGemma4PackageContract(catalog);
  const config = object(catalog.config, "audio_config");
  const epsilon = positive(config, "rms_norm_eps");
  const gradientClipping = positive(config, "gradient_clipping");
  const invalidAttentionLogit = finite(config, "attention_invalid_logits_value");
  const runtimeDtype = declaredAudioRuntimeDtype(config);
  if (config.use_clipped_linears !== true || config.hidden_act !== "silu") throw new Error("Gemma 4 audio requer use_clipped_linears=true e hidden_act='silu'.");
  const tensors = (names: string[]): TensorRef[] => names.map((name) => tensorRef(requireTensor(catalog, name)));
  const prefix = "model.audio_tower";
  const assignments: Gemma4AudioAssignment[] = [
    { id: "audio_input_mask", operation: "mask-input-features", inputs: ["input_features", "input_features_mask"], output: "audio_masked_features", semantics: "zero invalid frames before each stride-2 convolution" },
    { id: "audio_input_unsqueeze", operation: "reshape-conv-features", inputs: ["audio_masked_features"], output: "audio_masked_features_4d", semantics: "[B,T,F] -> [B,1,T,F] for Conv2d" },
    ...subsampleAssignment(0, prefix, tensors),
    { id: "audio_subsample_mask_0", operation: "subsample-mask", inputs: ["input_features_mask"], output: "audio_mask_1", semantics: "mask[:, ::2] after the first Conv2d" },
    { id: "audio_subsample_1_input_mask", operation: "mask-input-features", inputs: ["audio_subsample_0", "audio_mask_1"], output: "audio_subsample_0_masked", semantics: "zero invalid first-stage frames before the second Conv2d" },
    ...subsampleAssignment(1, prefix, tensors),
    { id: "audio_subsample_mask_1", operation: "subsample-mask", inputs: ["audio_mask_1"], output: "audio_output_mask", semantics: "mask[:, ::2] after the second Conv2d" },
    { id: "audio_subsample_flatten", operation: "reshape-conv-features", inputs: ["audio_subsample_1"], output: "audio_subsample_flat", semantics: "[B,C,T,F] -> [B,T,F*C] without reordering" },
    { id: "audio_input_projection", operation: "linear", inputs: ["audio_subsample_flat"], output: "audio_hidden_0", tensors: tensors([`${prefix}.subsample_conv_projection.input_proj_linear.weight`]), semantics: "bias-free input projection" },
    { id: "audio_relative_positions", operation: "relative-position-encoding", inputs: ["audio_hidden_0"], output: "audio_relative_positions", semantics: "BF16(exp(F32 arange * scalar increment)) -> BF16(position * inverse_timescale) -> BF16 SLEEF sin/cos" },
  ];
  for (let layer = 0; layer < contract.modalities.audioTower.layers; layer += 1) assignments.push(...layerAssignments(layer, prefix, tensors));
  assignments.push(
    { id: "audio_output_projection", operation: "linear", inputs: [`audio_hidden_${contract.modalities.audioTower.layers}`], output: "audio_output_projected", tensors: tensors([`${prefix}.output_proj.weight`, `${prefix}.output_proj.bias`]), semantics: "output projection with checkpointed bias" },
    { id: "audio_language_projection_norm", operation: "rms-norm", inputs: ["audio_output_projected"], output: "audio_output_normalized", semantics: "unscaled Gemma4MultimodalEmbedder RMSNorm before language projection" },
    { id: "audio_language_projection", operation: "linear", inputs: ["audio_output_normalized"], output: "audio_projected_features", tensors: tensors(["model.embed_audio.embedding_projection.weight"]), semantics: "Gemma4MultimodalEmbedder bias-free language projection" },
    { id: "audio_strip_padding", operation: "strip-padding", inputs: ["audio_projected_features", "audio_output_mask"], output: "audio_features", semantics: "keep only true output-mask rows in batch-major order" },
    { id: "audio_placeholder_scatter", operation: "masked-scatter-audio-features", inputs: ["text_embeddings", "input_ids", "audio_features"], output: "text_embeddings_with_audio", placeholderTokenId: contract.modalities.audioTokenId, semantics: "replace only audio_token_id coordinates; feature rows and placeholders must agree exactly" },
  );
  for (const assignment of assignments) assignment.dtypePolicy = audioAssignmentDtypePolicy(assignment, runtimeDtype);
  return {
    kind: "gemma4-audio-features",
    sourceFormat: "safetensors",
    attentionMaskContract: "transformers-eager-additive-mask-logical-not-v1",
    tower: contract.modalities.audioTower,
    textHiddenSize: contract.text.hiddenSize,
    rmsNormEpsilon: epsilon,
    gradientClipping,
    invalidAttentionLogit,
    runtimeDtype,
    assignments,
    output: "audio_features",
  };
}

/** Executes every declared audio assignment with scalar IEEE binary32 boundaries. */
export function executeGemma4AudioF32(program: Gemma4AudioProgram, request: Gemma4AudioExecutionRequest): Gemma4AudioExecutionResult {
  validateInput(request);
  const values = new Map<string, DenseF32Tensor>();
  const prefix = "model.audio_tower";
  const get = (name: string): DenseF32Tensor => tensor(request.tensors, name);
  const policyByOutput = new Map(program.assignments.map((assignment) => [assignment.output, assignment.dtypePolicy]));
  const put = (name: string, value: DenseF32Tensor): DenseF32Tensor => {
    const stored = policyByOutput.get(name)?.outputDtype === "BF16" ? executeGemma4AudioBf16CastF32(value) : value;
    values.set(name, stored);
    return stored;
  };
  const project = (input: DenseF32Tensor, weight: DenseF32Tensor, bias?: DenseF32Tensor): DenseF32Tensor =>
    linear(input, weight, bias, program.runtimeDtype === "BF16");
  const normalize = (input: DenseF32Tensor, weight?: DenseF32Tensor): DenseF32Tensor =>
    rmsNorm(input, weight, program.rmsNormEpsilon, program.runtimeDtype === "BF16");
  const mask0 = request.inputFeaturesMask.map((row) => [...row]);
  const masked = put("audio_masked_features", maskFeatures(request.inputFeatures, mask0));
  const masked4d = put("audio_masked_features_4d", unsqueezeChannel(masked));
  const first = put("audio_subsample_0_conv", conv2d(masked4d, get(`${prefix}.subsample_conv_projection.layer0.conv.weight`), program.runtimeDtype === "BF16"));
  const firstNorm = put("audio_subsample_0_norm", layerNormChannels(first, get(`${prefix}.subsample_conv_projection.layer0.norm.weight`), program.rmsNormEpsilon, program.runtimeDtype === "BF16"));
  const firstActivated = put("audio_subsample_0", relu(firstNorm));
  const mask1 = subsampleMask(mask0); put("audio_mask_1", boolMask(mask1));
  const secondInput = put("audio_subsample_0_masked", maskConv(firstActivated, mask1));
  const second = put("audio_subsample_1_conv", conv2d(secondInput, get(`${prefix}.subsample_conv_projection.layer1.conv.weight`), program.runtimeDtype === "BF16"));
  const secondNorm = put("audio_subsample_1_norm", layerNormChannels(second, get(`${prefix}.subsample_conv_projection.layer1.norm.weight`), program.rmsNormEpsilon, program.runtimeDtype === "BF16"));
  const secondActivated = put("audio_subsample_1", relu(secondNorm));
  const outputMask = subsampleMask(mask1); put("audio_output_mask", boolMask(outputMask));
  const flat = put("audio_subsample_flat", flattenConv(secondActivated));
  put("audio_hidden_0", project(flat, get(`${prefix}.subsample_conv_projection.input_proj_linear.weight`)));
  const positions = put("audio_relative_positions", relativePositions(program.tower));
  for (let layer = 0; layer < program.tower.layers; layer += 1) executeLayer(values, layer, program, outputMask, positions, request.tensors, put, project, normalize);
  const output = put("audio_output_projected", project(values.get(`audio_hidden_${program.tower.layers}`)!, get(`${prefix}.output_proj.weight`), get(`${prefix}.output_proj.bias`)));
  const normalizedOutput = put("audio_output_normalized", normalize(output));
  const projected = put("audio_projected_features", project(normalizedOutput, get("model.embed_audio.embedding_projection.weight")));
  const audioFeatures = put("audio_features", stripPadding(projected, outputMask));
  return { values, outputMask, audioFeatures };
}

/** Exact audio masked_scatter contract for the future composite adapter. */
export function scatterGemma4AudioFeaturesF32(textEmbeddings: DenseF32Tensor, inputIds: readonly number[][], audioTokenId: number, audioFeatures: DenseF32Tensor): DenseF32Tensor {
  if (textEmbeddings.shape.length !== 3 || inputIds.length !== textEmbeddings.shape[0] || inputIds.some((row) => row.length !== textEmbeddings.shape[1])) throw new Error("Gemma 4 audio scatter requer input_ids e embeddings [B,S,D] compatíveis.");
  const width = textEmbeddings.shape[2]!;
  if (audioFeatures.shape.length !== 2 || audioFeatures.shape[1] !== width) throw new Error("Gemma 4 audio features possuem hidden_size incompatível.");
  const slots = inputIds.flat().filter((id) => id === audioTokenId).length;
  if (slots !== audioFeatures.shape[0]) throw new Error(`Gemma 4 audio placeholder count=${slots} não corresponde a audio_features=${audioFeatures.shape[0]}.`);
  const output = Float32Array.from(textEmbeddings.values); let feature = 0;
  for (let batch = 0; batch < inputIds.length; batch += 1) for (let sequence = 0; sequence < inputIds[batch]!.length; sequence += 1) if (inputIds[batch]![sequence] === audioTokenId) {
    output.set(audioFeatures.values.subarray(feature * width, (feature + 1) * width), (batch * inputIds[batch]!.length + sequence) * width); feature += 1;
  }
  return dense([...textEmbeddings.shape], output);
}

function subsampleAssignment(layer: number, prefix: string, refs: (names: string[]) => TensorRef[]): Gemma4AudioAssignment[] { const base = `${prefix}.subsample_conv_projection.layer${layer}`; const input = layer === 0 ? "audio_masked_features_4d" : "audio_subsample_0_masked"; return [
  { id: `audio_subsample_${layer}_conv`, operation: "conv2d-stride2", inputs: [input], output: `audio_subsample_${layer}_conv`, tensors: refs([`${base}.conv.weight`]), semantics: "Conv2d kernel=3 stride=2 padding=1 bias=false" },
  { id: `audio_subsample_${layer}_norm`, operation: "layer-norm-channels", inputs: [`audio_subsample_${layer}_conv`], output: `audio_subsample_${layer}_norm`, tensors: refs([`${base}.norm.weight`]), semantics: "LayerNorm over output channels only; no bias" },
  { id: `audio_subsample_${layer}_relu`, operation: "relu", inputs: [`audio_subsample_${layer}_norm`], output: `audio_subsample_${layer}`, semantics: "ReLU after channel LayerNorm" },
]; }

function layerAssignments(layer: number, prefix: string, refs: (names: string[]) => TensorRef[]): Gemma4AudioAssignment[] {
  const base = `${prefix}.layers.${layer}`; const clipped = (name: string, input: string, output: string): Gemma4AudioAssignment => ({ id: `audio_layer_${layer}_${name}`, operation: "clipped-linear", inputs: [input], output, tensors: refs([`${base}.${name}.linear.weight`, `${base}.${name}.input_min`, `${base}.${name}.input_max`, `${base}.${name}.output_min`, `${base}.${name}.output_max`]), semantics: "clamp(input,min,max) -> linear(no bias) -> clamp(output,min,max)" });
  const ffn = (index: 1 | 2, input: string): Gemma4AudioAssignment[] => { const p = `${base}.feed_forward${index}`; const stem = `audio_layer_${layer}_ffn${index}`; return [
    { id: `${stem}_clip_in`, operation: "clip", inputs: [input], output: `${stem}_clipped_in`, semantics: "clamp to ±gradient_clipping before RMSNorm" },
    { id: `${stem}_pre_norm`, operation: "rms-norm", inputs: [`${stem}_clipped_in`], output: `${stem}_norm`, tensors: refs([`${p}.pre_layer_norm.weight`]) },
    { id: `${stem}_linear_1`, operation: "clipped-linear", inputs: [`${stem}_norm`], output: `${stem}_linear_1`, tensors: refs([`${p}.ffw_layer_1.linear.weight`, `${p}.ffw_layer_1.input_min`, `${p}.ffw_layer_1.input_max`, `${p}.ffw_layer_1.output_min`, `${p}.ffw_layer_1.output_max`]) },
    { id: `${stem}_silu`, operation: "silu", inputs: [`${stem}_linear_1`], output: `${stem}_activated` },
    { id: `${stem}_linear_2`, operation: "clipped-linear", inputs: [`${stem}_activated`], output: `${stem}_linear_2`, tensors: refs([`${p}.ffw_layer_2.linear.weight`, `${p}.ffw_layer_2.input_min`, `${p}.ffw_layer_2.input_max`, `${p}.ffw_layer_2.output_min`, `${p}.ffw_layer_2.output_max`]) },
    { id: `${stem}_clip_out`, operation: "clip", inputs: [`${stem}_linear_2`], output: `${stem}_clipped_out`, semantics: "clamp to ±gradient_clipping before post RMSNorm" },
    { id: `${stem}_post_norm`, operation: "rms-norm", inputs: [`${stem}_clipped_out`], output: `${stem}_normalized`, tensors: refs([`${p}.post_layer_norm.weight`]) },
    { id: `${stem}_residual_scale`, operation: "scale-f32", inputs: [`${stem}_normalized`], output: `${stem}_scaled`, semantics: "multiply by residual_weight" },
    { id: `${stem}_residual`, operation: "add", inputs: [input, `${stem}_scaled`], output: `${stem}_output` },
  ]; };
  const input = `audio_hidden_${layer}`;
  return [
    ...ffn(1, input),
    { id: `audio_layer_${layer}_attn_clip_in`, operation: "clip", inputs: [`audio_layer_${layer}_ffn1_output`], output: `audio_layer_${layer}_attn_clipped_in` },
    { id: `audio_layer_${layer}_attn_pre_norm`, operation: "rms-norm", inputs: [`audio_layer_${layer}_attn_clipped_in`], output: `audio_layer_${layer}_attn_norm`, tensors: refs([`${base}.norm_pre_attn.weight`]) },
    clipped("self_attn.q_proj", `audio_layer_${layer}_attn_norm`, `audio_layer_${layer}_q`),
    clipped("self_attn.k_proj", `audio_layer_${layer}_attn_norm`, `audio_layer_${layer}_k`),
    clipped("self_attn.v_proj", `audio_layer_${layer}_attn_norm`, `audio_layer_${layer}_v`),
    { id: `audio_layer_${layer}_relative_k_projection`, operation: "linear", inputs: ["audio_relative_positions"], output: `audio_layer_${layer}_relative_keys`, tensors: refs([`${base}.self_attn.relative_k_proj.weight`]), semantics: "bias-free relative-position projection with ordered F32 products and accumulation" },
    { id: `audio_layer_${layer}_q_scale`, operation: "per-dim-softplus-scale", inputs: [`audio_layer_${layer}_q`], output: `audio_layer_${layer}_q_scaled`, tensors: refs([`${base}.self_attn.per_dim_scale`]), semantics: "F32(F32(q * F32(head_dim^-0.5/log(2))) * BF16(softplus(BF16(per_dim_scale[d]))))" },
    { id: `audio_layer_${layer}_k_scale`, operation: "scale-f32", inputs: [`audio_layer_${layer}_k`], output: `audio_layer_${layer}_k_scaled`, semantics: "F32(k * F32(log1p(exp(1))/log(2)))" },
    { id: `audio_layer_${layer}_attention_content_scores`, operation: "chunked-attention-content-matmul", inputs: [`audio_layer_${layer}_q_scaled`, `audio_layer_${layer}_k_scaled`], output: `audio_layer_${layer}_attention_ac`, semantics: "AC[b,h,block,q,k] = native F32 batched matmul over head_dim after query blocking and key context extraction" },
    { id: `audio_layer_${layer}_attention_position_scores`, operation: "relative-attention-position-matmul", inputs: [`audio_layer_${layer}_q_scaled`, `audio_layer_${layer}_relative_keys`], output: `audio_layer_${layer}_attention_bd_unshifted`, semantics: "BD_unshifted[b,h,block,q,r] = native F32 batched matmul over head_dim" },
    { id: `audio_layer_${layer}_attention_relative_shift`, operation: "relative-attention-shift", inputs: [`audio_layer_${layer}_attention_bd_unshifted`], output: `audio_layer_${layer}_attention_bd`, semantics: "source pad(context+1-relative_length), flatten, prefix slice and reshape into context slots" },
    { id: `audio_layer_${layer}_attention_logit_add`, operation: "attention-logit-add", inputs: [`audio_layer_${layer}_attention_ac`, `audio_layer_${layer}_attention_bd`], output: `audio_layer_${layer}_attention_logits`, semantics: "F32 AC + relative-shifted BD" },
    { id: `audio_layer_${layer}_attention_softcap`, operation: "attention-softcap", inputs: [`audio_layer_${layer}_attention_logits`], output: `audio_layer_${layer}_attention_softcapped`, semantics: "F32(SLEEF_TANH_F32(F32(logit / cap)) * cap) before masking" },
    { id: `audio_layer_${layer}_attention_mask`, operation: "chunked-attention-mask", inputs: [`audio_layer_${layer}_attention_softcapped`, "audio_output_mask"], output: `audio_layer_${layer}_attention_scores`, semantics: "pinned eager source: create additive 4-D local mask, pad/gather it to blocked 5-D, then masked_fill(mask.logical_not()); zero allowed/padding entries are filled and nonzero rejected source entries retain scores" },
    { id: `audio_layer_${layer}_attention_softmax`, operation: "chunked-relative-attention-softmax", inputs: [`audio_layer_${layer}_attention_scores`], output: `audio_layer_${layer}_attention_weights`, semantics: "F32 maximum, exponential, denominator and probability over context slots" },
    { id: `audio_layer_${layer}_attention`, operation: "chunked-relative-attention-values", inputs: [`audio_layer_${layer}_attention_weights`, `audio_layer_${layer}_v`], output: `audio_layer_${layer}_attention_context`, semantics: "context-slot value reduction, head merge and sequence trim" },
    { id: `audio_layer_${layer}_attention_context_cast`, operation: "cast-bf16", inputs: [`audio_layer_${layer}_attention_context`], output: `audio_layer_${layer}_attention_context_bf16`, semantics: "explicit source-visible to(dtype=post.linear.weight.dtype) before attention post projection" },
    clipped("self_attn.post", `audio_layer_${layer}_attention_context_bf16`, `audio_layer_${layer}_attention_projected`),
    { id: `audio_layer_${layer}_attn_clip_out`, operation: "clip", inputs: [`audio_layer_${layer}_attention_projected`], output: `audio_layer_${layer}_attn_clipped_out` },
    { id: `audio_layer_${layer}_attn_post_norm`, operation: "rms-norm", inputs: [`audio_layer_${layer}_attn_clipped_out`], output: `audio_layer_${layer}_after_attention_norm`, tensors: refs([`${base}.norm_post_attn.weight`]) },
    { id: `audio_layer_${layer}_attn_residual`, operation: "add", inputs: [`audio_layer_${layer}_ffn1_output`, `audio_layer_${layer}_after_attention_norm`], output: `audio_layer_${layer}_after_attention` },
    { id: `audio_layer_${layer}_conv_pre_norm`, operation: "rms-norm", inputs: [`audio_layer_${layer}_after_attention`], output: `audio_layer_${layer}_conv_norm_in`, tensors: refs([`${base}.lconv1d.pre_layer_norm.weight`]) },
    clipped("lconv1d.linear_start", `audio_layer_${layer}_conv_norm_in`, `audio_layer_${layer}_conv_glu_linear`),
    { id: `audio_layer_${layer}_conv_glu`, operation: "split-gated-linear-unit", inputs: [`audio_layer_${layer}_conv_glu_linear`], output: `audio_layer_${layer}_conv_glu` },
    { id: `audio_layer_${layer}_conv_depthwise`, operation: "causal-depthwise-convolution", inputs: [`audio_layer_${layer}_conv_glu`], output: `audio_layer_${layer}_conv_depthwise`, tensors: refs([`${base}.lconv1d.depthwise_conv1d.weight`]), semantics: "left-pad kernel_size-1, groups=hidden_size, bias=false" },
    { id: `audio_layer_${layer}_conv_clip`, operation: "clip", inputs: [`audio_layer_${layer}_conv_depthwise`], output: `audio_layer_${layer}_conv_clipped` },
    { id: `audio_layer_${layer}_conv_post_norm`, operation: "rms-norm", inputs: [`audio_layer_${layer}_conv_clipped`], output: `audio_layer_${layer}_conv_normalized`, tensors: refs([`${base}.lconv1d.conv_norm.weight`]) },
    { id: `audio_layer_${layer}_conv_silu`, operation: "silu", inputs: [`audio_layer_${layer}_conv_normalized`], output: `audio_layer_${layer}_conv_activated` },
    clipped("lconv1d.linear_end", `audio_layer_${layer}_conv_activated`, `audio_layer_${layer}_conv_projected`),
    { id: `audio_layer_${layer}_conv_residual`, operation: "add", inputs: [`audio_layer_${layer}_after_attention`, `audio_layer_${layer}_conv_projected`], output: `audio_layer_${layer}_after_conv` },
    ...ffn(2, `audio_layer_${layer}_after_conv`),
    { id: `audio_layer_${layer}_out_clip`, operation: "clip", inputs: [`audio_layer_${layer}_ffn2_output`], output: `audio_layer_${layer}_out_clipped` },
    { id: `audio_layer_${layer}_out_norm`, operation: "rms-norm", inputs: [`audio_layer_${layer}_out_clipped`], output: `audio_hidden_${layer + 1}`, tensors: refs([`${base}.norm_out.weight`]) },
  ];
}

function executeLayer(
  values: Map<string, DenseF32Tensor>, layer: number, program: Gemma4AudioProgram,
  outputMask: readonly boolean[][], positions: DenseF32Tensor, tensors: ReadonlyMap<string, DenseF32Tensor>,
  put: (name: string, value: DenseF32Tensor) => DenseF32Tensor,
  project: (input: DenseF32Tensor, weight: DenseF32Tensor, bias?: DenseF32Tensor) => DenseF32Tensor,
  normalize: (input: DenseF32Tensor, weight?: DenseF32Tensor) => DenseF32Tensor,
): void {
  const get = (name: string): DenseF32Tensor => tensor(tensors, name);
  const base = `model.audio_tower.layers.${layer}`;
  const input = values.get(`audio_hidden_${layer}`)!;
  const clippedProject = (name: string, inputTensor: DenseF32Tensor): DenseF32Tensor => {
    const clipping = bounds(tensors, `${base}.${name}`);
    return clipRange(project(clipRange(inputTensor, clipping[0], clipping[1]), get(`${base}.${name}.linear.weight`)), clipping[2], clipping[3]);
  };
  const ffn = (index: 1 | 2, inputTensor: DenseF32Tensor): DenseF32Tensor => {
    const p = `${base}.feed_forward${index}`, stem = `audio_layer_${layer}_ffn${index}`;
    const clippedIn = put(`${stem}_clipped_in`, clip(inputTensor, program.gradientClipping));
    const norm = put(`${stem}_norm`, normalize(clippedIn, get(`${p}.pre_layer_norm.weight`)));
    const oneBounds = bounds(tensors, `${p}.ffw_layer_1`);
    const one = put(`${stem}_linear_1`, clipRange(project(clipRange(norm, oneBounds[0], oneBounds[1]), get(`${p}.ffw_layer_1.linear.weight`)), oneBounds[2], oneBounds[3]));
    const activated = put(`${stem}_activated`, silu(one));
    const twoBounds = bounds(tensors, `${p}.ffw_layer_2`);
    const two = put(`${stem}_linear_2`, clipRange(project(clipRange(activated, twoBounds[0], twoBounds[1]), get(`${p}.ffw_layer_2.linear.weight`)), twoBounds[2], twoBounds[3]));
    const clippedOut = put(`${stem}_clipped_out`, clip(two, program.gradientClipping));
    const normalized = put(`${stem}_normalized`, normalize(clippedOut, get(`${p}.post_layer_norm.weight`)));
    const scaled = put(`${stem}_scaled`, scale(normalized, program.tower.residualWeight));
    return put(`${stem}_output`, add(inputTensor, scaled));
  };
  const afterFfn1 = ffn(1, input);
  const attnInput = put(`audio_layer_${layer}_attn_clipped_in`, clip(afterFfn1, program.gradientClipping));
  const attnNorm = put(`audio_layer_${layer}_attn_norm`, normalize(attnInput, get(`${base}.norm_pre_attn.weight`)));
  const q = put(`audio_layer_${layer}_q`, clippedProject("self_attn.q_proj", attnNorm));
  const k = put(`audio_layer_${layer}_k`, clippedProject("self_attn.k_proj", attnNorm));
  const v = put(`audio_layer_${layer}_v`, clippedProject("self_attn.v_proj", attnNorm));
  const relativeKeys = put(`audio_layer_${layer}_relative_keys`, project(positions, get(`${base}.self_attn.relative_k_proj.weight`)));
  const qScaled = put(`audio_layer_${layer}_q_scaled`, scaleAttentionQuery(q, get(`${base}.self_attn.per_dim_scale`), program.tower.headDim));
  const kScaled = put(`audio_layer_${layer}_k_scaled`, scale(k, f32(Math.log1p(Math.exp(1)) / Math.log(2))));
  const contentScores = put(`audio_layer_${layer}_attention_ac`, chunkedAttentionContentScores(qScaled, kScaled, program));
  const positionScores = put(`audio_layer_${layer}_attention_bd_unshifted`, relativeAttentionPositionScores(qScaled, relativeKeys, program));
  const derivedAttention = executeGemma4AudioDerivedAttentionStagesF32(program, contentScores, positionScores, outputMask);
  put(`audio_layer_${layer}_attention_bd`, derivedAttention.shiftedPositionScores);
  put(`audio_layer_${layer}_attention_logits`, derivedAttention.logits);
  put(`audio_layer_${layer}_attention_softcapped`, derivedAttention.softcapped);
  const scores = put(`audio_layer_${layer}_attention_scores`, derivedAttention.maskedScores);
  const weights = put(`audio_layer_${layer}_attention_weights`, chunkedAttentionSoftmax(scores));
  const context = put(`audio_layer_${layer}_attention_context`, chunkedAttentionValues(weights, v, program));
  const contextBf16 = put(`audio_layer_${layer}_attention_context_bf16`, context);
  const projected = put(`audio_layer_${layer}_attention_projected`, clippedProject("self_attn.post", contextBf16));
  const attnClip = put(`audio_layer_${layer}_attn_clipped_out`, clip(projected, program.gradientClipping));
  const afterAttnNorm = put(`audio_layer_${layer}_after_attention_norm`, normalize(attnClip, get(`${base}.norm_post_attn.weight`)));
  const afterAttention = put(`audio_layer_${layer}_after_attention`, add(afterFfn1, afterAttnNorm));
  const convNormIn = put(`audio_layer_${layer}_conv_norm_in`, normalize(afterAttention, get(`${base}.lconv1d.pre_layer_norm.weight`)));
  const gluLinear = put(`audio_layer_${layer}_conv_glu_linear`, clippedProject("lconv1d.linear_start", convNormIn));
  const glu = put(`audio_layer_${layer}_conv_glu`, gatedLinearUnit(gluLinear));
  const depthwise = put(`audio_layer_${layer}_conv_depthwise`, causalDepthwise(glu, get(`${base}.lconv1d.depthwise_conv1d.weight`), program.runtimeDtype === "BF16"));
  const convClip = put(`audio_layer_${layer}_conv_clipped`, clip(depthwise, program.gradientClipping));
  const convNorm = put(`audio_layer_${layer}_conv_normalized`, normalize(convClip, get(`${base}.lconv1d.conv_norm.weight`)));
  const convActivated = put(`audio_layer_${layer}_conv_activated`, silu(convNorm));
  const convProjected = put(`audio_layer_${layer}_conv_projected`, clippedProject("lconv1d.linear_end", convActivated));
  const afterConv = put(`audio_layer_${layer}_after_conv`, add(afterAttention, convProjected));
  const afterFfn2 = ffn(2, afterConv);
  const outClip = put(`audio_layer_${layer}_out_clipped`, clip(afterFfn2, program.gradientClipping));
  put(`audio_hidden_${layer + 1}`, normalize(outClip, get(`${base}.norm_out.weight`)));
}

interface AudioAttentionLayout {
  batch: number;
  sequence: number;
  width: number;
  heads: number;
  dim: number;
  chunk: number;
  left: number;
  blocks: number;
  context: number;
  relativeLength: number;
}

function audioAttentionLayout(input: DenseF32Tensor, program: Gemma4AudioProgram): AudioAttentionLayout {
  const [batch, sequence, width] = input.shape as [number, number, number];
  const { attentionHeads: heads, headDim: dim, attentionChunkSize: chunk, attentionContextLeft: left, attentionContextRight: right } = program.tower;
  if (input.shape.length !== 3 || width !== heads * dim) throw new Error("Gemma 4 audio attention possui hidden shape incompatível.");
  const blocks = Math.ceil(sequence / chunk), context = chunk + left - 1 + right;
  return { batch, sequence, width, heads, dim, chunk, left, blocks, context, relativeLength: Math.floor(context / 2) + 1 };
}

function chunkedAttentionContentScores(q: DenseF32Tensor, k: DenseF32Tensor, program: Gemma4AudioProgram): DenseF32Tensor {
  const layout = audioAttentionLayout(q, program);
  if (k.shape.some((value, index) => value !== q.shape[index])) throw new Error("Gemma 4 audio attention possui Q/K incompatíveis.");
  const { batch, sequence, heads, dim, chunk, left, blocks, context } = layout;
  const out = new Float32Array(batch * heads * blocks * chunk * context);
  const at = (values: Float32Array, b: number, s: number, h: number, d: number): number => values[((b * sequence + s) * heads + h) * dim + d]!;
  for (let b = 0; b < batch; b += 1) for (let h = 0; h < heads; h += 1) for (let block = 0; block < blocks; block += 1) for (let query = 0; query < chunk; query += 1) for (let keySlot = 0; keySlot < context; keySlot += 1) {
    const queryIndex = block * chunk + query, keyIndex = block * chunk - (left - 1) + keySlot;
    let sum = f32(0);
    if (queryIndex < sequence && keyIndex >= 0 && keyIndex < sequence) for (let d = 0; d < dim; d += 1) {
      sum = f32(sum + f32(at(q.values, b, queryIndex, h, d) * at(k.values, b, keyIndex, h, d)));
    }
    out[((((b * heads + h) * blocks + block) * chunk + query) * context) + keySlot] = sum;
  }
  return dense([batch, heads, blocks, chunk, context], out);
}

function relativeAttentionPositionScores(q: DenseF32Tensor, relative: DenseF32Tensor, program: Gemma4AudioProgram): DenseF32Tensor {
  const layout = audioAttentionLayout(q, program);
  const { batch, sequence, width, heads, dim, chunk, blocks, relativeLength } = layout;
  if (relative.shape.length !== 3 || relative.shape[0] !== 1 || relative.shape[1] !== relativeLength || relative.shape[2] !== width) {
    throw new Error("Gemma 4 audio relative position shape incompatível.");
  }
  const out = new Float32Array(batch * heads * blocks * chunk * relativeLength);
  const at = (b: number, s: number, h: number, d: number): number => q.values[((b * sequence + s) * heads + h) * dim + d]!;
  for (let b = 0; b < batch; b += 1) for (let h = 0; h < heads; h += 1) for (let block = 0; block < blocks; block += 1) for (let query = 0; query < chunk; query += 1) for (let position = 0; position < relativeLength; position += 1) {
    const queryIndex = block * chunk + query;
    let sum = f32(0);
    if (queryIndex < sequence) for (let d = 0; d < dim; d += 1) {
      sum = f32(sum + f32(at(b, queryIndex, h, d) * relative.values[(position * heads + h) * dim + d]!));
    }
    out[((((b * heads + h) * blocks + block) * chunk + query) * relativeLength) + position] = sum;
  }
  return dense([batch, heads, blocks, chunk, relativeLength], out);
}

function shiftRelativeAttentionScores(unshifted: DenseF32Tensor, program: Gemma4AudioProgram): DenseF32Tensor {
  const { attentionChunkSize: chunk, attentionContextLeft: left, attentionContextRight: right } = program.tower;
  const context = chunk + left - 1 + right, relativeLength = Math.floor(context / 2) + 1;
  if (unshifted.shape.length !== 5 || unshifted.shape[3] !== chunk || unshifted.shape[4] !== relativeLength) {
    throw new Error("Gemma 4 audio relative shift possui shape incompatível.");
  }
  const [batch, heads, blocks] = unshifted.shape, out = new Float32Array(batch! * heads! * blocks! * chunk * context);
  for (let b = 0; b < batch!; b += 1) for (let h = 0; h < heads!; h += 1) for (let block = 0; block < blocks!; block += 1) for (let query = 0; query < chunk; query += 1) for (let keySlot = 0; keySlot < context; keySlot += 1) {
    const source = relativeShiftSource(query, keySlot, context, relativeLength);
    if (source.relativeIndex === undefined) continue;
    const sourceIndex = ((((b * heads! + h) * blocks! + block) * chunk + source.queryInBlock) * relativeLength) + source.relativeIndex;
    out[((((b * heads! + h) * blocks! + block) * chunk + query) * context) + keySlot] = unshifted.values[sourceIndex]!;
  }
  return dense([batch!, heads!, blocks!, chunk, context], out);
}

/**
 * Executes the source-visible, non-native stages which follow the two F32
 * attention matmuls. Differential validation can seed this boundary with
 * authoritative AC/BD tensors without replacing either unknown matmul by a
 * guessed scalar schedule.
 */
export function executeGemma4AudioDerivedAttentionStagesF32(
  program: Gemma4AudioProgram,
  contentScores: DenseF32Tensor,
  unshiftedPositionScores: DenseF32Tensor,
  mask: readonly boolean[][],
): Gemma4AudioDerivedAttentionStages {
  const shiftedPositionScores = shiftRelativeAttentionScores(unshiftedPositionScores, program);
  const logits = add(contentScores, shiftedPositionScores);
  const softcapped = softcapAttentionScores(logits, program.tower.attentionLogitCap);
  const maskedScores = executeGemma4AudioEagerAttentionMaskF32(softcapped, mask, program);
  return { shiftedPositionScores, logits, softcapped, maskedScores };
}

function softcapAttentionScores(logits: DenseF32Tensor, cap: number): DenseF32Tensor {
  return dense([...logits.shape], Float32Array.from(logits.values, (value) => f32(sleefTanhF32(f32(value / cap)) * cap)));
}

/**
 * Reproduces the pinned Transformers eager path, including its conversion of
 * the additive 4-D mask to a blocked tensor followed by `logical_not()` in
 * Gemma4AudioAttention. Consequently zero (allowed) and padded entries are
 * filled, while nonzero additive-mask entries retain their score. This is an
 * observed source contract, not the conventional logical mask one might infer.
 */
export function executeGemma4AudioEagerAttentionMaskF32(
  scores: DenseF32Tensor,
  mask: readonly boolean[][],
  program: Gemma4AudioProgram,
): DenseF32Tensor {
  const [batch, heads, blocks, chunk, context] = scores.shape;
  const { attentionContextLeft: left } = program.tower;
  const sequence = mask[0]?.length ?? 0;
  if (program.attentionMaskContract !== "transformers-eager-additive-mask-logical-not-v1" || scores.shape.length !== 5 ||
    mask.length !== batch || mask.some((row) => row.length !== sequence)) {
    throw new Error("Gemma 4 audio mask de atenção eager incompatível.");
  }
  const out = Float32Array.from(scores.values);
  for (let b = 0; b < batch!; b += 1) for (let h = 0; h < heads!; h += 1) for (let block = 0; block < blocks!; block += 1) for (let query = 0; query < chunk!; query += 1) for (let keySlot = 0; keySlot < context!; keySlot += 1) {
    const queryIndex = block * chunk! + query, keyIndex = block * chunk! - (left - 1) + keySlot;
    const sourceEntryExists = queryIndex < sequence && keyIndex >= 0 && keyIndex < sequence;
    const additiveMaskIsZero = sourceEntryExists && mask[b]![keyIndex] === true &&
      queryIndex >= keyIndex && queryIndex - keyIndex < left;
    const paddedBlockedEntryIsZero = !sourceEntryExists;
    if (additiveMaskIsZero || paddedBlockedEntryIsZero) {
      out[((((b * heads! + h) * blocks! + block) * chunk! + query) * context!) + keySlot] = program.invalidAttentionLogit;
    }
  }
  return dense([...scores.shape], out);
}

function chunkedAttentionSoftmax(scores: DenseF32Tensor): DenseF32Tensor { const context = scores.shape.at(-1)!; if (scores.shape.length !== 5 || context <= 0) throw new Error("Gemma 4 audio softmax requer scores [B,H,blocks,chunk,context]."); const out = new Float32Array(scores.values.length); for (let row = 0; row < scores.values.length / context; row += 1) { const base = row * context; let maximum = -Infinity; for (let k = 0; k < context; k += 1) maximum = Math.max(maximum, scores.values[base + k]!); let total = f32(0); for (let k = 0; k < context; k += 1) { out[base + k] = sleefExpF32(f32(scores.values[base + k]! - maximum)); total = f32(total + out[base + k]!); } for (let k = 0; k < context; k += 1) out[base + k] = f32(out[base + k]! / total); } return dense([...scores.shape], out); }

function chunkedAttentionValues(weights: DenseF32Tensor, values: DenseF32Tensor, program: Gemma4AudioProgram): DenseF32Tensor { const [batch, sequence, width] = values.shape as [number, number, number], { attentionHeads: heads, headDim: dim, attentionChunkSize: chunk, attentionContextLeft: left, attentionContextRight: right } = program.tower, blocks = Math.ceil(sequence / chunk), context = chunk + left - 1 + right; if (width !== heads * dim || weights.shape.length !== 5 || weights.shape[0] !== batch || weights.shape[1] !== heads || weights.shape[2] !== blocks || weights.shape[3] !== chunk || weights.shape[4] !== context) throw new Error("Gemma 4 audio pesos/V incompatíveis."); const out = new Float32Array(batch * sequence * width), at = (b: number, s: number, h: number, d: number): number => values.values[((b * sequence + s) * heads + h) * dim + d]!; for (let b = 0; b < batch; b += 1) for (let h = 0; h < heads; h += 1) for (let block = 0; block < blocks; block += 1) for (let qi = 0; qi < chunk; qi += 1) { const queryIndex = block * chunk + qi; if (queryIndex >= sequence) continue; for (let d = 0; d < dim; d += 1) { let sum = f32(0); for (let keySlot = 0; keySlot < context; keySlot += 1) { const keyIndex = block * chunk - (left - 1) + keySlot; if (keyIndex >= 0 && keyIndex < sequence) sum = f32(sum + f32(weights.values[((((b * heads + h) * blocks + block) * chunk + qi) * context) + keySlot]! * at(b, keyIndex, h, d))); } out[((b * sequence + queryIndex) * heads + h) * dim + d] = sum; } } return dense([batch, sequence, width], out); }

function scaleAttentionQuery(input: DenseF32Tensor, perDimScale: DenseF32Tensor, headDim: number): DenseF32Tensor {
  if (perDimScale.shape.length !== 1 || perDimScale.shape[0] !== headDim || input.shape.at(-1)! % headDim !== 0) throw new Error("Gemma 4 audio per_dim_scale incompatível.");
  const qScale = f32(headDim ** -0.5 / Math.log(2));
  return dense([...input.shape], Float32Array.from(input.values, (value, index) => {
    const softplusBf16 = roundF32ToBF16(f32(Math.log1p(Math.exp(perDimScale.values[index % headDim]!))));
    return f32(f32(value * qScale) * softplusBf16);
  }));
}

/** Mirrors the source pad/view/slice/view relative-shift sequence exactly. */
function relativeShiftSource(queryInBlock: number, keySlot: number, context: number, relativeLength: number): { queryInBlock: number; relativeIndex?: number } { const paddedLength = context + 1; const flattened = queryInBlock * context + keySlot; const sourceQuery = Math.floor(flattened / paddedLength); const sourceRelative = flattened % paddedLength; return sourceRelative < relativeLength ? { queryInBlock: sourceQuery, relativeIndex: sourceRelative } : { queryInBlock: sourceQuery }; }

function relativePositions(tower: Gemma4AudioTowerContract): DenseF32Tensor { const context = tower.attentionChunkSize + tower.attentionContextLeft - 1 + tower.attentionContextRight, length = Math.floor(context / 2) + 1, out = new Float32Array(length * tower.hiddenSize); const times = tower.hiddenSize / 2; const increment = f32(Math.log(10_000) / Math.max(times - 1, 1)); for (let row = 0; row < length; row += 1) for (let i = 0; i < times; i += 1) { const inverseTimescale = roundF32ToBF16(sleefExpF32(f32(-i * increment))); const scaled = roundF32ToBF16(f32((length - 1 - row) * inverseTimescale)); out[row * tower.hiddenSize + i] = roundF32ToBF16(sleefSinF32(scaled)); out[row * tower.hiddenSize + times + i] = roundF32ToBF16(sleefCosF32(scaled)); } return dense([1, length, tower.hiddenSize], out); }
function bounds(tensors: ReadonlyMap<string, DenseF32Tensor>, prefix: string): readonly [number, number, number, number] { return [tensor(tensors, `${prefix}.input_min`).values[0]!, tensor(tensors, `${prefix}.input_max`).values[0]!, tensor(tensors, `${prefix}.output_min`).values[0]!, tensor(tensors, `${prefix}.output_max`).values[0]!]; }
function clippedLinear(input: DenseF32Tensor, weight: DenseF32Tensor, bound: readonly [number, number, number, number]): DenseF32Tensor { return clipRange(linear(clipRange(input, bound[0], bound[1]), weight), bound[2], bound[3]); }
function clipRange(input: DenseF32Tensor, low: number, high: number): DenseF32Tensor { const out = Float32Array.from(input.values, (x) => f32(Math.min(high, Math.max(low, x)))); return dense([...input.shape], out); }
function clip(input: DenseF32Tensor, cap: number): DenseF32Tensor { return clipRange(input, -cap, cap); }
function linear(input: DenseF32Tensor, weight: DenseF32Tensor, bias?: DenseF32Tensor, nativeBf16 = false): DenseF32Tensor { const inFeatures = input.shape.at(-1)!; if (weight.shape.length !== 2 || weight.shape[1] !== inFeatures || (bias !== undefined && (bias.shape.length !== 1 || bias.shape[0] !== weight.shape[0]))) throw new Error("Gemma 4 audio linear incompatível."); const rows = input.values.length / inFeatures, outFeatures = weight.shape[0]!, out = new Float32Array(rows * outFeatures); for (let row = 0; row < rows; row += 1) for (let output = 0; output < outFeatures; output += 1) { let sum = bias?.values[output] ?? 0; if (nativeBf16) { const inputBase = row * inFeatures, weightBase = output * inFeatures; sum = armNeonBf16DotF32(inFeatures, (feature) => roundF32ToBF16(input.values[inputBase + feature]!), (feature) => weight.values[weightBase + feature]!, BF16_ARM_DOT_REDUCTION); if (bias) sum = f32(bias.values[output]! + sum); } else for (let feature = 0; feature < inFeatures; feature += 1) sum = f32(sum + f32(input.values[row * inFeatures + feature]! * weight.values[output * inFeatures + feature]!)); out[row * outFeatures + output] = sum; } return dense([...input.shape.slice(0, -1), outFeatures], out); }
function rmsNorm(input: DenseF32Tensor, weight: DenseF32Tensor | undefined, epsilon: number, nativeBf16 = false): DenseF32Tensor { const width = input.shape.at(-1)!; if (weight !== undefined && (weight.shape.length !== 1 || weight.shape[0] !== width)) throw new Error("Gemma 4 audio RMSNorm incompatível."); const out = new Float32Array(input.values.length); for (let row = 0; row < input.values.length / width; row += 1) { let sum = f32(0); if (nativeBf16) sum = pytorchCpuCascadeSquareSumF32(input, row * width, width); else for (let d = 0; d < width; d += 1) sum = f32(sum + f32(input.values[row * width + d]! * input.values[row * width + d]!)); const mean = f32(f32(sum / width) + epsilon); const scale = pytorchPowNegativeHalfF32(mean); for (let d = 0; d < width; d += 1) out[row * width + d] = f32(f32(input.values[row * width + d]! * scale) * (weight?.values[d] ?? 1)); } return dense([...input.shape], out); }
function conv2d(input: DenseF32Tensor, weight: DenseF32Tensor, nativeBf16 = false): DenseF32Tensor { const [batch, channels, time, features] = input.shape as [number, number, number, number]; const [outChannels, expectedChannels, kernelTime, kernelFeatures] = weight.shape as [number, number, number, number]; if (expectedChannels !== channels || kernelTime !== 3 || kernelFeatures !== 3) throw new Error("Gemma 4 audio Conv2d incompatível."); const outTime = Math.ceil(time / 2), outFeatures = Math.ceil(features / 2), out = new Float32Array(batch * outChannels * outTime * outFeatures), terms = channels * 9; for (let b = 0; b < batch; b += 1) for (let oc = 0; oc < outChannels; oc += 1) for (let t = 0; t < outTime; t += 1) for (let f = 0; f < outFeatures; f += 1) { const product = (index: number): number => { const ic = Math.floor(index / 9), local = index % 9, kt = Math.floor(local / 3), kf = local % 3; const sourceT = t * 2 + kt - 1, sourceF = f * 2 + kf - 1; const value = sourceT >= 0 && sourceT < time && sourceF >= 0 && sourceF < features ? input.values[((b * channels + ic) * time + sourceT) * features + sourceF]! : 0; return nativeBf16 ? roundF32ToBF16(value) : value; }; const sum = nativeBf16 ? pytorchCpuBf16GemmIlp4F32(terms, product, (index) => weight.values[oc * terms + index]!) : orderedConvDot(terms, product, (index) => weight.values[oc * terms + index]!); out[((b * outChannels + oc) * outTime + t) * outFeatures + f] = sum; } return dense([batch, outChannels, outTime,outFeatures], out); }
function layerNormChannels(input: DenseF32Tensor, weight: DenseF32Tensor, epsilon: number, nativeBf16 = false): DenseF32Tensor { const [batch, channels, time, features] = input.shape as [number, number, number, number]; if (weight.shape.length !== 1 || weight.shape[0] !== channels) throw new Error("Gemma 4 audio LayerNorm incompatível."); const out = new Float32Array(input.values.length); for (let b = 0; b < batch; b += 1) for (let t = 0; t < time; t += 1) for (let f = 0; f < features; f += 1) { const at = (c: number): number => input.values[((b * channels + c) * time + t) * features + f]!; let mean: number, variance: number; if (nativeBf16) ({ mean, variance } = pytorchCpuBf16WelfordMomentsF32(channels, at, BF16_WELFORD_REDUCTION)); else { mean = f32(0); for (let c = 0; c < channels; c += 1) mean = f32(mean + at(c)); mean = f32(mean / channels); variance = f32(0); for (let c = 0; c < channels; c += 1) variance = f32(variance + f32((at(c) - mean) ** 2)); variance = f32(variance / channels); } const inv = f32(1 / f32(Math.sqrt(f32(variance + epsilon)))); const bias = f32(-inv * mean); for (let c = 0; c < channels; c += 1) out[((b * channels + c) * time + t) * features + f] = nativeBf16 ? f32(f32(f32(at(c) * inv) + bias) * weight.values[c]!) : f32(f32(at(c) - mean) * inv * weight.values[c]!); } return dense([...input.shape], out); }
function orderedConvDot(length: number, left: (index: number) => number, right: (index: number) => number): number { let sum = f32(0); for (let index = 0; index < length; index += 1) sum = f32(sum + f32(left(index) * right(index))); return sum; }
function flattenConv(input: DenseF32Tensor): DenseF32Tensor { const [batch, channels, time, features] = input.shape as [number, number, number, number]; const out = new Float32Array(batch * time * features * channels); for (let b = 0; b < batch; b += 1) for (let t = 0; t < time; t += 1) for (let f = 0; f < features; f += 1) for (let c = 0; c < channels; c += 1) out[(b * time + t) * features * channels + f * channels + c] = input.values[((b * channels + c) * time + t) * features + f]!; return dense([batch, time, features * channels], out); }
function maskFeatures(input: DenseF32Tensor, mask: readonly boolean[][]): DenseF32Tensor { const [batch, time, features] = input.shape as [number, number, number]; const out = Float32Array.from(input.values); for (let b = 0; b < batch; b += 1) for (let t = 0; t < time; t += 1) if (!mask[b]![t]) out.fill(0, (b * time + t) * features, (b * time + t + 1) * features); return dense([...input.shape], out); }
function unsqueezeChannel(input: DenseF32Tensor): DenseF32Tensor { const [batch, time, features] = input.shape as [number, number, number]; return dense([batch, 1, time, features], Float32Array.from(input.values)); }
function maskConv(input: DenseF32Tensor, mask: readonly boolean[][]): DenseF32Tensor { const [batch, channels, time, features] = input.shape as [number, number, number, number]; const out = Float32Array.from(input.values); for (let b = 0; b < batch; b += 1) for (let t = 0; t < time; t += 1) if (!mask[b]![t]) for (let c = 0; c < channels; c += 1) out.fill(0, ((b * channels + c) * time + t) * features, ((b * channels + c) * time + t + 1) * features); return dense([...input.shape], out); }
function subsampleMask(mask: readonly boolean[][]): boolean[][] { return mask.map((row) => row.filter((_, index) => index % 2 === 0)); }
function relu(input: DenseF32Tensor): DenseF32Tensor { return dense([...input.shape], Float32Array.from(input.values, (x) => f32(Math.max(0, x)))); }
function silu(input: DenseF32Tensor): DenseF32Tensor { return dense([...input.shape], Float32Array.from(input.values, (x) => f32(x / f32(1 + sleefExpF32(f32(-x)))))); }
function add(left: DenseF32Tensor, right: DenseF32Tensor): DenseF32Tensor { if (left.values.length !== right.values.length || left.shape.some((v, i) => v !== right.shape[i])) throw new Error("Gemma 4 audio add incompatível."); return dense([...left.shape], Float32Array.from(left.values, (x, i) => f32(x + right.values[i]!))); }
function scale(input: DenseF32Tensor, scalar: number): DenseF32Tensor { return dense([...input.shape], Float32Array.from(input.values, (x) => f32(x * scalar))); }
function gatedLinearUnit(input: DenseF32Tensor): DenseF32Tensor { const width = input.shape.at(-1)!; if (width % 2 !== 0) throw new Error("Gemma 4 audio GLU requer width par."); const half = width / 2, rows = input.values.length / width, out = new Float32Array(rows * half); for (let row = 0; row < rows; row += 1) for (let d = 0; d < half; d += 1) out[row * half + d] = f32(input.values[row * width + d]! / f32(1 + sleefExpF32(f32(-input.values[row * width + half + d]!)))); return dense([...input.shape.slice(0, -1), half], out); }
function causalDepthwise(input: DenseF32Tensor, weight: DenseF32Tensor, nativeBf16 = false): DenseF32Tensor { const [batch, time, channels] = input.shape as [number, number, number]; const [outChannels, groups, kernel] = weight.shape as [number, number, number]; if (outChannels !== channels || groups !== 1) throw new Error("Gemma 4 audio depthwise convolution incompatível."); const out = new Float32Array(input.values.length); for (let b = 0; b < batch; b += 1) for (let t = 0; t < time; t += 1) for (let c = 0; c < channels; c += 1) { const source = (k: number): number => { const sourceTime = t - kernel + 1 + k; return sourceTime < 0 ? 0 : input.values[(b * time + sourceTime) * channels + c]!; }; const sum = nativeBf16 ? pytorchCpuBf16GemmIlp4F32(kernel, source, (k) => weight.values[c * kernel + k]!) : orderedConvDot(kernel, source, (k) => weight.values[c * kernel + k]!); out[(b * time + t) * channels + c] = sum; } return dense([...input.shape], out); }
function stripPadding(input: DenseF32Tensor, mask: readonly boolean[][]): DenseF32Tensor { const [batch, time, width] = input.shape as [number, number, number]; const rows: number[] = []; for (let b = 0; b < batch; b += 1) for (let t = 0; t < time; t += 1) if (mask[b]![t]) for (let d = 0; d < width; d += 1) rows.push(input.values[(b * time + t) * width + d]!); return dense([rows.length / width, width], Float32Array.from(rows)); }
function boolMask(mask: readonly boolean[][]): DenseF32Tensor { return dense([mask.length, mask[0]!.length], Float32Array.from(mask.flat().map((value) => value ? 1 : 0))); }

function declaredAudioRuntimeDtype(config: Record<string, unknown>): "F32" | "BF16" {
  const dtype = config.dtype;
  if (dtype === undefined || dtype === "float32") return "F32";
  if (dtype === "bfloat16") return "BF16";
  throw new Error(`Gemma 4 audio dtype '${String(dtype)}' não possui contrato registrado.`);
}

/** Source- and dtype-dispatched policy applied to the complete audio operation class. */
function audioAssignmentDtypePolicy(assignment: Gemma4AudioAssignment, runtimeDtype: "F32" | "BF16"): DtypePolicy {
  if (assignment.operation === "cast-bf16") return { ...F32_TO_BF16_POLICY };
  if (assignment.operation === "chunked-attention-content-matmul" || assignment.operation === "relative-attention-position-matmul" || assignment.operation === "chunked-relative-attention-values") {
    return { ...F32_POLICY, computeDtype: "pytorch-cpu-f32-matmul", accumulationDtype: "runtime-defined" };
  }
  if (runtimeDtype === "F32") {
    const reduction = assignment.operation === "linear" || assignment.operation === "clipped-linear" || assignment.operation === "rms-norm" ||
      assignment.operation === "conv2d-stride2" || assignment.operation === "layer-norm-channels" || assignment.operation === "causal-depthwise-convolution";
    return reduction ? { ...F32_POLICY, reduction: ORDERED_F32_REDUCTION } : { ...F32_POLICY };
  }
  if (assignment.operation === "subsample-mask") return { inputDtype: "BOOL", computeDtype: "BOOL", accumulationDtype: "none", outputDtype: "BOOL" };
  if (assignment.id === "audio_input_mask" || assignment.id === "audio_input_unsqueeze") return { ...F32_POLICY };
  if (assignment.id.endsWith("_q_scale") || assignment.id.endsWith("_k_scale") || assignment.operation === "relative-attention-shift" ||
    assignment.operation === "attention-logit-add" || assignment.operation === "attention-softcap" ||
    assignment.operation === "chunked-attention-mask" || assignment.operation === "chunked-relative-attention-softmax") return { ...F32_POLICY };
  if (assignment.operation === "linear" || assignment.operation === "clipped-linear") {
    const hasBias = assignment.tensors?.some((tensor) => tensor.shape.length === 1 && !tensor.name.endsWith("input_min") && !tensor.name.endsWith("input_max") && !tensor.name.endsWith("output_min") && !tensor.name.endsWith("output_max"));
    return hasBias
      ? { ...BF16_LINEAR_POLICY, computeDtype: "pytorch-cpu-bf16-addmm", reduction: structuredClone(BF16_ARM_DOT_REDUCTION) }
      : { ...BF16_LINEAR_POLICY, reduction: structuredClone(BF16_ARM_DOT_REDUCTION) };
  }
  if (assignment.operation === "rms-norm") return { ...BF16_RMS_POLICY, reduction: structuredClone(BF16_RMS_POLICY.reduction) };
  if (assignment.operation === "layer-norm-channels") return { ...BF16_POLICY, computeDtype: "pytorch-cpu-bf16-welford", reduction: structuredClone(BF16_WELFORD_REDUCTION) };
  if (assignment.operation === "conv2d-stride2" || assignment.operation === "causal-depthwise-convolution") return {
    ...BF16_POLICY, computeDtype: "pytorch-cpu-bf16-gemm-ilp4", reduction: structuredClone(BF16_GEMM_ILP4_REDUCTION),
  };
  return { ...BF16_POLICY };
}

function validateInput(request: Gemma4AudioExecutionRequest): void { const [batch, frames] = request.inputFeatures.shape; if (request.inputFeatures.shape.length !== 3 || batch === undefined || frames === undefined || request.inputFeaturesMask.length !== batch || request.inputFeaturesMask.some((row) => row.length !== frames)) throw new Error("Gemma 4 audio requer input_features [B,T,F] e input_features_mask [B,T] compatíveis."); if (request.inputFeatures.values.length !== request.inputFeatures.shape.reduce((total, dimension) => total * dimension, 1)) throw new Error("Gemma 4 audio input_features possui buffer incompatível."); }
function tensor(tensors: ReadonlyMap<string, DenseF32Tensor>, name: string): DenseF32Tensor { const found = tensors.get(name); if (!found) throw new Error(`Tensor F32 Gemma 4 audio ausente: ${name}`); return found; }
function dense(shape: number[], values: Float32Array): DenseF32Tensor { return { shape, values }; }
function tensorRef(tensor: TensorInfo): TensorRef { return { name: tensor.name, shape: [...tensor.logicalShape], storageDtype: tensor.storageDtype }; }
function requireTensor(catalog: ModelCatalog, name: string): TensorInfo { const found = catalog.tensors.get(name); if (!found || found.quantization) throw new Error(`Gemma 4 audio requer tensor denso ${name}.`); return found; }
function object(input: Record<string, unknown>, key: string): Record<string, unknown> { const value = input[key]; if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`Gemma 4 audio requer ${key} objeto.`); return value as Record<string, unknown>; }
function positive(config: Record<string, unknown>, key: string): number { const value = config[key]; if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) throw new Error(`Gemma 4 audio requer ${key} positivo.`); return value; }
function finite(config: Record<string, unknown>, key: string): number { const value = config[key]; if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`Gemma 4 audio requer ${key} finito.`); return value; }
const f32 = Math.fround;
