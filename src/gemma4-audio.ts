import { inspectGemma4PackageContract, type Gemma4AudioTowerContract } from "./gemma4-contract.js";
import type { DenseF32Tensor, ModelCatalog, TensorInfo, TensorRef } from "./types.js";

/**
 * Literal, named execution boundary for the registered Gemma 4 audio encoder.
 * This is deliberately an audio-feature program, not a generic Conformer:
 * its blocked relative attention, clips, mask subsampling, causal depthwise
 * convolution and residual ordering are the pinned Gemma 4 runtime contract.
 */
export interface Gemma4AudioProgram {
  kind: "gemma4-audio-features";
  sourceFormat: "safetensors";
  tower: Gemma4AudioTowerContract;
  textHiddenSize: number;
  rmsNormEpsilon: number;
  gradientClipping: number;
  invalidAttentionLogit: number;
  assignments: Gemma4AudioAssignment[];
  output: "audio_features";
}

export interface Gemma4AudioAssignment {
  id: string;
  operation:
    | "mask-input-features" | "conv2d-stride2" | "layer-norm-channels" | "relu"
    | "reshape-conv-features" | "linear" | "relative-position-encoding"
    | "rms-norm" | "clip" | "clipped-linear" | "silu" | "scale-f32" | "add"
    | "split-gated-linear-unit" | "causal-depthwise-convolution"
    | "chunked-relative-attention" | "subsample-mask" | "strip-padding"
    | "masked-scatter-audio-features";
  inputs: string[];
  output: string;
  tensors?: TensorRef[];
  semantics?: string;
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

export function buildGemma4AudioProgram(catalog: ModelCatalog): Gemma4AudioProgram {
  const contract = inspectGemma4PackageContract(catalog);
  const config = object(catalog.config, "audio_config");
  const epsilon = positive(config, "rms_norm_eps");
  const gradientClipping = positive(config, "gradient_clipping");
  const invalidAttentionLogit = finite(config, "attention_invalid_logits_value");
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
    { id: "audio_relative_positions", operation: "relative-position-encoding", inputs: ["audio_hidden_0"], output: "audio_relative_positions", semantics: "sin/cos positions context_size//2 down to zero, projected per layer" },
  ];
  for (let layer = 0; layer < contract.modalities.audioTower.layers; layer += 1) assignments.push(...layerAssignments(layer, prefix, tensors));
  assignments.push(
    { id: "audio_output_projection", operation: "linear", inputs: [`audio_hidden_${contract.modalities.audioTower.layers}`], output: "audio_output_projected", tensors: tensors([`${prefix}.output_proj.weight`, `${prefix}.output_proj.bias`]), semantics: "output projection with checkpointed bias" },
    { id: "audio_language_projection_norm", operation: "rms-norm", inputs: ["audio_output_projected"], output: "audio_output_normalized", semantics: "unscaled Gemma4MultimodalEmbedder RMSNorm before language projection" },
    { id: "audio_language_projection", operation: "linear", inputs: ["audio_output_normalized"], output: "audio_projected_features", tensors: tensors(["model.embed_audio.embedding_projection.weight"]), semantics: "Gemma4MultimodalEmbedder bias-free language projection" },
    { id: "audio_strip_padding", operation: "strip-padding", inputs: ["audio_projected_features", "audio_output_mask"], output: "audio_features", semantics: "keep only true output-mask rows in batch-major order" },
    { id: "audio_placeholder_scatter", operation: "masked-scatter-audio-features", inputs: ["text_embeddings", "input_ids", "audio_features"], output: "text_embeddings_with_audio", semantics: "replace only audio_token_id coordinates; feature rows and placeholders must agree exactly" },
  );
  return { kind: "gemma4-audio-features", sourceFormat: "safetensors", tower: contract.modalities.audioTower, textHiddenSize: contract.text.hiddenSize, rmsNormEpsilon: epsilon, gradientClipping, invalidAttentionLogit, assignments, output: "audio_features" };
}

/** Executes every declared audio assignment with scalar IEEE binary32 boundaries. */
export function executeGemma4AudioF32(program: Gemma4AudioProgram, request: Gemma4AudioExecutionRequest): Gemma4AudioExecutionResult {
  validateInput(request);
  const values = new Map<string, DenseF32Tensor>();
  const prefix = "model.audio_tower";
  const get = (name: string): DenseF32Tensor => tensor(request.tensors, name);
  const mask0 = request.inputFeaturesMask.map((row) => [...row]);
  const masked = maskFeatures(request.inputFeatures, mask0); values.set("audio_masked_features", masked);
  const masked4d = unsqueezeChannel(masked); values.set("audio_masked_features_4d", masked4d);
  const first = conv2d(masked4d, get(`${prefix}.subsample_conv_projection.layer0.conv.weight`));
  const firstNorm = layerNormChannels(first, get(`${prefix}.subsample_conv_projection.layer0.norm.weight`), program.rmsNormEpsilon);
  const firstActivated = relu(firstNorm); values.set("audio_subsample_0_conv", first); values.set("audio_subsample_0_norm", firstNorm); values.set("audio_subsample_0", firstActivated);
  const mask1 = subsampleMask(mask0); values.set("audio_mask_1", boolMask(mask1));
  const secondInput = maskConv(firstActivated, mask1); values.set("audio_subsample_0_masked", secondInput);
  const second = conv2d(secondInput, get(`${prefix}.subsample_conv_projection.layer1.conv.weight`));
  const secondNorm = layerNormChannels(second, get(`${prefix}.subsample_conv_projection.layer1.norm.weight`), program.rmsNormEpsilon);
  const secondActivated = relu(secondNorm); values.set("audio_subsample_1_conv", second); values.set("audio_subsample_1_norm", secondNorm); values.set("audio_subsample_1", secondActivated);
  const outputMask = subsampleMask(mask1); values.set("audio_output_mask", boolMask(outputMask));
  const flat = flattenConv(secondActivated); values.set("audio_subsample_flat", flat);
  const initial = linear(flat, get(`${prefix}.subsample_conv_projection.input_proj_linear.weight`)); values.set("audio_hidden_0", initial);
  const positions = relativePositions(program.tower); values.set("audio_relative_positions", positions);
  for (let layer = 0; layer < program.tower.layers; layer += 1) executeLayer(values, layer, program, outputMask, positions, request.tensors);
  const output = linear(values.get(`audio_hidden_${program.tower.layers}`)!, get(`${prefix}.output_proj.weight`), get(`${prefix}.output_proj.bias`)); values.set("audio_output_projected", output);
  const normalizedOutput = rmsNorm(output, undefined, program.rmsNormEpsilon); values.set("audio_output_normalized", normalizedOutput);
  const projected = linear(normalizedOutput, get("model.embed_audio.embedding_projection.weight")); values.set("audio_projected_features", projected);
  const audioFeatures = stripPadding(projected, outputMask); values.set("audio_features", audioFeatures);
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
    { id: `audio_layer_${layer}_attention`, operation: "chunked-relative-attention", inputs: [`audio_layer_${layer}_q`, `audio_layer_${layer}_k`, `audio_layer_${layer}_v`, "audio_relative_positions", "audio_output_mask"], output: `audio_layer_${layer}_attention_context`, tensors: refs([`${base}.self_attn.per_dim_scale`, `${base}.self_attn.relative_k_proj.weight`]), semantics: "q softplus scale, blocked local AC+relative-BD, tanh softcap, F32 softmax" },
    clipped("self_attn.post", `audio_layer_${layer}_attention_context`, `audio_layer_${layer}_attention_projected`),
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

function executeLayer(values: Map<string, DenseF32Tensor>, layer: number, program: Gemma4AudioProgram, outputMask: readonly boolean[][], positions: DenseF32Tensor, tensors: ReadonlyMap<string, DenseF32Tensor>): void {
  const get = (name: string): DenseF32Tensor => tensor(tensors, name); const base = `model.audio_tower.layers.${layer}`; const input = values.get(`audio_hidden_${layer}`)!;
  const ffn = (index: 1 | 2, inputTensor: DenseF32Tensor): DenseF32Tensor => { const p = `${base}.feed_forward${index}`, stem = `audio_layer_${layer}_ffn${index}`; const clippedIn = clip(inputTensor, program.gradientClipping); const norm = rmsNorm(clippedIn, get(`${p}.pre_layer_norm.weight`), program.rmsNormEpsilon); const one = clippedLinear(norm, get(`${p}.ffw_layer_1.linear.weight`), bounds(tensors, `${p}.ffw_layer_1`)); const activated = silu(one); const two = clippedLinear(activated, get(`${p}.ffw_layer_2.linear.weight`), bounds(tensors, `${p}.ffw_layer_2`)); const clippedOut = clip(two, program.gradientClipping); const normalized = rmsNorm(clippedOut, get(`${p}.post_layer_norm.weight`), program.rmsNormEpsilon); const scaled = scale(normalized, program.tower.residualWeight); const result = add(inputTensor, scaled); for (const [name, value] of [[`${stem}_clipped_in`, clippedIn], [`${stem}_norm`, norm], [`${stem}_linear_1`, one], [`${stem}_activated`, activated], [`${stem}_linear_2`, two], [`${stem}_clipped_out`, clippedOut], [`${stem}_normalized`, normalized], [`${stem}_scaled`, scaled], [`${stem}_output`, result]] as const) values.set(name, value); return result; };
  const afterFfn1 = ffn(1, input); const attnInput = clip(afterFfn1, program.gradientClipping); const attnNorm = rmsNorm(attnInput, get(`${base}.norm_pre_attn.weight`), program.rmsNormEpsilon); const q = clippedLinear(attnNorm, get(`${base}.self_attn.q_proj.linear.weight`), bounds(tensors, `${base}.self_attn.q_proj`)); const k = clippedLinear(attnNorm, get(`${base}.self_attn.k_proj.linear.weight`), bounds(tensors, `${base}.self_attn.k_proj`)); const v = clippedLinear(attnNorm, get(`${base}.self_attn.v_proj.linear.weight`), bounds(tensors, `${base}.self_attn.v_proj`)); const context = chunkedAttention(q, k, v, positions, outputMask, get(`${base}.self_attn.per_dim_scale`), get(`${base}.self_attn.relative_k_proj.weight`), program); const projected = clippedLinear(context, get(`${base}.self_attn.post.linear.weight`), bounds(tensors, `${base}.self_attn.post`)); const attnClip = clip(projected, program.gradientClipping); const afterAttnNorm = rmsNorm(attnClip, get(`${base}.norm_post_attn.weight`), program.rmsNormEpsilon); const afterAttention = add(afterFfn1, afterAttnNorm);
  const convNormIn = rmsNorm(afterAttention, get(`${base}.lconv1d.pre_layer_norm.weight`), program.rmsNormEpsilon); const gluLinear = clippedLinear(convNormIn, get(`${base}.lconv1d.linear_start.linear.weight`), bounds(tensors, `${base}.lconv1d.linear_start`)); const glu = gatedLinearUnit(gluLinear); const depthwise = causalDepthwise(glu, get(`${base}.lconv1d.depthwise_conv1d.weight`)); const convClip = clip(depthwise, program.gradientClipping); const convNorm = rmsNorm(convClip, get(`${base}.lconv1d.conv_norm.weight`), program.rmsNormEpsilon); const convActivated = silu(convNorm); const convProjected = clippedLinear(convActivated, get(`${base}.lconv1d.linear_end.linear.weight`), bounds(tensors, `${base}.lconv1d.linear_end`)); const afterConv = add(afterAttention, convProjected); const afterFfn2 = ffn(2, afterConv); const outClip = clip(afterFfn2, program.gradientClipping); const output = rmsNorm(outClip, get(`${base}.norm_out.weight`), program.rmsNormEpsilon);
  for (const [name, value] of [[`audio_layer_${layer}_attn_clipped_in`, attnInput], [`audio_layer_${layer}_attn_norm`, attnNorm], [`audio_layer_${layer}_q`, q], [`audio_layer_${layer}_k`, k], [`audio_layer_${layer}_v`, v], [`audio_layer_${layer}_attention_context`, context], [`audio_layer_${layer}_attention_projected`, projected], [`audio_layer_${layer}_attn_clipped_out`, attnClip], [`audio_layer_${layer}_after_attention_norm`, afterAttnNorm], [`audio_layer_${layer}_after_attention`, afterAttention], [`audio_layer_${layer}_conv_norm_in`, convNormIn], [`audio_layer_${layer}_conv_glu_linear`, gluLinear], [`audio_layer_${layer}_conv_glu`, glu], [`audio_layer_${layer}_conv_depthwise`, depthwise], [`audio_layer_${layer}_conv_clipped`, convClip], [`audio_layer_${layer}_conv_normalized`, convNorm], [`audio_layer_${layer}_conv_activated`, convActivated], [`audio_layer_${layer}_conv_projected`, convProjected], [`audio_layer_${layer}_after_conv`, afterConv], [`audio_layer_${layer}_out_clipped`, outClip], [`audio_hidden_${layer + 1}`, output]] as const) values.set(name, value);
}

function chunkedAttention(q0: DenseF32Tensor, k0: DenseF32Tensor, v0: DenseF32Tensor, relativePositions: DenseF32Tensor, mask: readonly boolean[][], perDimScale: DenseF32Tensor, relativeWeight: DenseF32Tensor, program: Gemma4AudioProgram): DenseF32Tensor {
  const [batch, sequence, width] = q0.shape as [number, number, number]; const { attentionHeads: heads, headDim: dim, attentionChunkSize: chunk, attentionContextLeft: left, attentionContextRight: right } = program.tower; if (width !== heads * dim) throw new Error("Gemma 4 audio attention possui width incompatível."); const blocks = Math.ceil(sequence / chunk), context = chunk + left - 1 + right, relative = linear(relativePositions, relativeWeight); const q = Float32Array.from(q0.values), k = Float32Array.from(k0.values); const qScale = f32(dim ** -0.5 / Math.log(2)), kScale = f32(Math.log(1 + Math.E) / Math.log(2)); for (let i = 0; i < q.length; i += 1) { const d = i % dim; q[i] = f32(q[i]! * f32(qScale * f32(Math.log1p(Math.exp(perDimScale.values[d]!))))); k[i] = f32(k[i]! * kScale); }
  const out = new Float32Array(batch * sequence * width); const relLength = relative.shape[0]!; if (relLength !== Math.floor(context / 2) + 1) throw new Error("Gemma 4 audio relative position length incompatível."); const at = (values: Float32Array, b: number, s: number, h: number, d: number): number => values[((b * sequence + s) * heads + h) * dim + d]!;
  for (let b = 0; b < batch; b += 1) for (let h = 0; h < heads; h += 1) for (let block = 0; block < blocks; block += 1) for (let qi = 0; qi < chunk; qi += 1) { const queryIndex = block * chunk + qi; const scores = new Float32Array(context); let max = -Infinity; for (let keySlot = 0; keySlot < context; keySlot += 1) { const keyIndex = block * chunk - (left - 1) + keySlot; let ac = f32(0); if (queryIndex < sequence && keyIndex >= 0 && keyIndex < sequence) for (let d = 0; d < dim; d += 1) ac = f32(ac + f32(at(q, b, queryIndex, h, d) * at(k, b, keyIndex, h, d))); const shifted = relativeShiftSource(qi, keySlot, context, relLength), shiftedQuery = block * chunk + shifted.queryInBlock; let bd = f32(0); if (shifted.relativeIndex !== undefined && shiftedQuery < sequence) for (let d = 0; d < dim; d += 1) bd = f32(bd + f32(at(q, b, shiftedQuery, h, d) * relative.values[(shifted.relativeIndex * heads + h) * dim + d]!)); const allowed = queryIndex < sequence && keyIndex >= 0 && keyIndex < sequence && mask[b]![queryIndex] === true && mask[b]![keyIndex] === true && queryIndex >= keyIndex && queryIndex - keyIndex < left; const raw = allowed ? f32(ac + bd) : program.invalidAttentionLogit; const softcapped = f32(f32(Math.tanh(f32(raw / program.tower.attentionLogitCap))) * program.tower.attentionLogitCap); scores[keySlot] = softcapped; if (softcapped > max) max = softcapped; }
    let total = f32(0); for (let keySlot = 0; keySlot < context; keySlot += 1) { scores[keySlot] = f32(Math.exp(f32(scores[keySlot]! - max))); total = f32(total + scores[keySlot]!); } if (queryIndex < sequence) for (let d = 0; d < dim; d += 1) { let sum = f32(0); for (let keySlot = 0; keySlot < context; keySlot += 1) { const keyIndex = block * chunk - (left - 1) + keySlot; if (keyIndex >= 0 && keyIndex < sequence) sum = f32(sum + f32(f32(scores[keySlot]! / total) * at(v0.values, b, keyIndex, h, d))); } out[((b * sequence + queryIndex) * heads + h) * dim + d] = sum; }
  }
  return dense([batch, sequence, width], out);
}

/** Mirrors the source pad/view/slice/view relative-shift sequence exactly. */
function relativeShiftSource(queryInBlock: number, keySlot: number, context: number, relativeLength: number): { queryInBlock: number; relativeIndex?: number } { const paddedLength = context + 1; const flattened = queryInBlock * context + keySlot; const sourceQuery = Math.floor(flattened / paddedLength); const sourceRelative = flattened % paddedLength; return sourceRelative < relativeLength ? { queryInBlock: sourceQuery, relativeIndex: sourceRelative } : { queryInBlock: sourceQuery }; }

function relativePositions(tower: Gemma4AudioTowerContract): DenseF32Tensor { const context = tower.attentionChunkSize + tower.attentionContextLeft - 1 + tower.attentionContextRight, length = Math.floor(context / 2) + 1, out = new Float32Array(length * tower.hiddenSize); const times = tower.hiddenSize / 2; const increment = Math.log(10_000) / Math.max(times - 1, 1); for (let row = 0; row < length; row += 1) for (let i = 0; i < times; i += 1) { const scaled = f32((length - 1 - row) * Math.exp(-i * increment)); out[row * tower.hiddenSize + i] = f32(Math.sin(scaled)); out[row * tower.hiddenSize + times + i] = f32(Math.cos(scaled)); } return dense([length, tower.hiddenSize], out); }
function bounds(tensors: ReadonlyMap<string, DenseF32Tensor>, prefix: string): readonly [number, number, number, number] { return [tensor(tensors, `${prefix}.input_min`).values[0]!, tensor(tensors, `${prefix}.input_max`).values[0]!, tensor(tensors, `${prefix}.output_min`).values[0]!, tensor(tensors, `${prefix}.output_max`).values[0]!]; }
function clippedLinear(input: DenseF32Tensor, weight: DenseF32Tensor, bound: readonly [number, number, number, number]): DenseF32Tensor { return clipRange(linear(clipRange(input, bound[0], bound[1]), weight), bound[2], bound[3]); }
function clipRange(input: DenseF32Tensor, low: number, high: number): DenseF32Tensor { const out = Float32Array.from(input.values, (x) => f32(Math.min(high, Math.max(low, x)))); return dense([...input.shape], out); }
function clip(input: DenseF32Tensor, cap: number): DenseF32Tensor { return clipRange(input, -cap, cap); }
function linear(input: DenseF32Tensor, weight: DenseF32Tensor, bias?: DenseF32Tensor): DenseF32Tensor { const inFeatures = input.shape.at(-1)!; if (weight.shape.length !== 2 || weight.shape[1] !== inFeatures || (bias !== undefined && (bias.shape.length !== 1 || bias.shape[0] !== weight.shape[0]))) throw new Error("Gemma 4 audio linear incompatível."); const rows = input.values.length / inFeatures, outFeatures = weight.shape[0]!, out = new Float32Array(rows * outFeatures); for (let row = 0; row < rows; row += 1) for (let output = 0; output < outFeatures; output += 1) { let sum = bias?.values[output] ?? 0; for (let feature = 0; feature < inFeatures; feature += 1) sum = f32(sum + f32(input.values[row * inFeatures + feature]! * weight.values[output * inFeatures + feature]!)); out[row * outFeatures + output] = sum; } return dense([...input.shape.slice(0, -1), outFeatures], out); }
function rmsNorm(input: DenseF32Tensor, weight: DenseF32Tensor | undefined, epsilon: number): DenseF32Tensor { const width = input.shape.at(-1)!; if (weight !== undefined && (weight.shape.length !== 1 || weight.shape[0] !== width)) throw new Error("Gemma 4 audio RMSNorm incompatível."); const out = new Float32Array(input.values.length); for (let row = 0; row < input.values.length / width; row += 1) { let mean = f32(0); for (let d = 0; d < width; d += 1) mean = f32(mean + f32(input.values[row * width + d]! * input.values[row * width + d]!)); mean = f32(f32(mean / width) + epsilon); const scale = f32(mean ** -0.5); for (let d = 0; d < width; d += 1) out[row * width + d] = f32(f32(input.values[row * width + d]! * scale) * (weight?.values[d] ?? 1)); } return dense([...input.shape], out); }
function conv2d(input: DenseF32Tensor, weight: DenseF32Tensor): DenseF32Tensor { const [batch, channels, time, features] = input.shape as [number, number, number, number]; const [outChannels, expectedChannels, kernelTime, kernelFeatures] = weight.shape as [number, number, number, number]; if (expectedChannels !== channels || kernelTime !== 3 || kernelFeatures !== 3) throw new Error("Gemma 4 audio Conv2d incompatível."); const outTime = Math.ceil(time / 2), outFeatures = Math.ceil(features / 2), out = new Float32Array(batch * outChannels * outTime * outFeatures); for (let b = 0; b < batch; b += 1) for (let oc = 0; oc < outChannels; oc += 1) for (let t = 0; t < outTime; t += 1) for (let f = 0; f < outFeatures; f += 1) { let sum = f32(0); for (let ic = 0; ic < channels; ic += 1) for (let kt = 0; kt < 3; kt += 1) for (let kf = 0; kf < 3; kf += 1) { const sourceT = t * 2 + kt - 1, sourceF = f * 2 + kf - 1; if (sourceT >= 0 && sourceT < time && sourceF >= 0 && sourceF < features) sum = f32(sum + f32(input.values[((b * channels + ic) * time + sourceT) * features + sourceF]! * weight.values[((oc * channels + ic) * 3 + kt) * 3 + kf]!)); } out[((b * outChannels + oc) * outTime + t) * outFeatures + f] = sum; } return dense([batch, outChannels, outTime, outFeatures], out); }
function layerNormChannels(input: DenseF32Tensor, weight: DenseF32Tensor, epsilon: number): DenseF32Tensor { const [batch, channels, time, features] = input.shape as [number, number, number, number]; if (weight.shape.length !== 1 || weight.shape[0] !== channels) throw new Error("Gemma 4 audio LayerNorm incompatível."); const out = new Float32Array(input.values.length); for (let b = 0; b < batch; b += 1) for (let t = 0; t < time; t += 1) for (let f = 0; f < features; f += 1) { let mean = f32(0); for (let c = 0; c < channels; c += 1) mean = f32(mean + input.values[((b * channels + c) * time + t) * features + f]!); mean = f32(mean / channels); let variance = f32(0); for (let c = 0; c < channels; c += 1) variance = f32(variance + f32((input.values[((b * channels + c) * time + t) * features + f]! - mean) ** 2)); const inv = f32(f32(variance / channels + epsilon) ** -0.5); for (let c = 0; c < channels; c += 1) out[((b * channels + c) * time + t) * features + f] = f32(f32(input.values[((b * channels + c) * time + t) * features + f]! - mean) * inv * weight.values[c]!); } return dense([...input.shape], out); }
function flattenConv(input: DenseF32Tensor): DenseF32Tensor { const [batch, channels, time, features] = input.shape as [number, number, number, number]; const out = new Float32Array(batch * time * features * channels); for (let b = 0; b < batch; b += 1) for (let t = 0; t < time; t += 1) for (let f = 0; f < features; f += 1) for (let c = 0; c < channels; c += 1) out[(b * time + t) * features * channels + f * channels + c] = input.values[((b * channels + c) * time + t) * features + f]!; return dense([batch, time, features * channels], out); }
function maskFeatures(input: DenseF32Tensor, mask: readonly boolean[][]): DenseF32Tensor { const [batch, time, features] = input.shape as [number, number, number]; const out = Float32Array.from(input.values); for (let b = 0; b < batch; b += 1) for (let t = 0; t < time; t += 1) if (!mask[b]![t]) out.fill(0, (b * time + t) * features, (b * time + t + 1) * features); return dense([...input.shape], out); }
function unsqueezeChannel(input: DenseF32Tensor): DenseF32Tensor { const [batch, time, features] = input.shape as [number, number, number]; return dense([batch, 1, time, features], Float32Array.from(input.values)); }
function maskConv(input: DenseF32Tensor, mask: readonly boolean[][]): DenseF32Tensor { const [batch, channels, time, features] = input.shape as [number, number, number, number]; const out = Float32Array.from(input.values); for (let b = 0; b < batch; b += 1) for (let t = 0; t < time; t += 1) if (!mask[b]![t]) for (let c = 0; c < channels; c += 1) out.fill(0, ((b * channels + c) * time + t) * features, ((b * channels + c) * time + t + 1) * features); return dense([...input.shape], out); }
function subsampleMask(mask: readonly boolean[][]): boolean[][] { return mask.map((row) => row.filter((_, index) => index % 2 === 0)); }
function relu(input: DenseF32Tensor): DenseF32Tensor { return dense([...input.shape], Float32Array.from(input.values, (x) => f32(Math.max(0, x)))); }
function silu(input: DenseF32Tensor): DenseF32Tensor { return dense([...input.shape], Float32Array.from(input.values, (x) => f32(x / (1 + Math.exp(-x))))); }
function add(left: DenseF32Tensor, right: DenseF32Tensor): DenseF32Tensor { if (left.values.length !== right.values.length || left.shape.some((v, i) => v !== right.shape[i])) throw new Error("Gemma 4 audio add incompatível."); return dense([...left.shape], Float32Array.from(left.values, (x, i) => f32(x + right.values[i]!))); }
function scale(input: DenseF32Tensor, scalar: number): DenseF32Tensor { return dense([...input.shape], Float32Array.from(input.values, (x) => f32(x * scalar))); }
function gatedLinearUnit(input: DenseF32Tensor): DenseF32Tensor { const width = input.shape.at(-1)!; if (width % 2 !== 0) throw new Error("Gemma 4 audio GLU requer width par."); const half = width / 2, rows = input.values.length / width, out = new Float32Array(rows * half); for (let row = 0; row < rows; row += 1) for (let d = 0; d < half; d += 1) out[row * half + d] = f32(input.values[row * width + d]! / (1 + Math.exp(-input.values[row * width + half + d]!))); return dense([...input.shape.slice(0, -1), half], out); }
function causalDepthwise(input: DenseF32Tensor, weight: DenseF32Tensor): DenseF32Tensor { const [batch, time, channels] = input.shape as [number, number, number]; const [outChannels, groups, kernel] = weight.shape as [number, number, number]; if (outChannels !== channels || groups !== 1) throw new Error("Gemma 4 audio depthwise convolution incompatível."); const out = new Float32Array(input.values.length); for (let b = 0; b < batch; b += 1) for (let t = 0; t < time; t += 1) for (let c = 0; c < channels; c += 1) { let sum = f32(0); for (let k = 0; k < kernel; k += 1) { const source = t - kernel + 1 + k; if (source >= 0) sum = f32(sum + f32(input.values[(b * time + source) * channels + c]! * weight.values[(c * kernel) + k]!)); } out[(b * time + t) * channels + c] = sum; } return dense([...input.shape], out); }
function stripPadding(input: DenseF32Tensor, mask: readonly boolean[][]): DenseF32Tensor { const [batch, time, width] = input.shape as [number, number, number]; const rows: number[] = []; for (let b = 0; b < batch; b += 1) for (let t = 0; t < time; t += 1) if (mask[b]![t]) for (let d = 0; d < width; d += 1) rows.push(input.values[(b * time + t) * width + d]!); return dense([rows.length / width, width], Float32Array.from(rows)); }
function boolMask(mask: readonly boolean[][]): DenseF32Tensor { return dense([mask.length, mask[0]!.length], Float32Array.from(mask.flat().map((value) => value ? 1 : 0))); }
function validateInput(request: Gemma4AudioExecutionRequest): void { const [batch, frames] = request.inputFeatures.shape; if (request.inputFeatures.shape.length !== 3 || batch === undefined || frames === undefined || request.inputFeaturesMask.length !== batch || request.inputFeaturesMask.some((row) => row.length !== frames)) throw new Error("Gemma 4 audio requer input_features [B,T,F] e input_features_mask [B,T] compatíveis."); if (request.inputFeatures.values.length !== request.inputFeatures.shape.reduce((total, dimension) => total * dimension, 1)) throw new Error("Gemma 4 audio input_features possui buffer incompatível."); }
function tensor(tensors: ReadonlyMap<string, DenseF32Tensor>, name: string): DenseF32Tensor { const found = tensors.get(name); if (!found) throw new Error(`Tensor F32 Gemma 4 audio ausente: ${name}`); return found; }
function dense(shape: number[], values: Float32Array): DenseF32Tensor { return { shape, values }; }
function tensorRef(tensor: TensorInfo): TensorRef { return { name: tensor.name, shape: [...tensor.logicalShape], storageDtype: tensor.storageDtype }; }
function requireTensor(catalog: ModelCatalog, name: string): TensorInfo { const found = catalog.tensors.get(name); if (!found || found.quantization) throw new Error(`Gemma 4 audio requer tensor denso ${name}.`); return found; }
function object(input: Record<string, unknown>, key: string): Record<string, unknown> { const value = input[key]; if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`Gemma 4 audio requer ${key} objeto.`); return value as Record<string, unknown>; }
function positive(config: Record<string, unknown>, key: string): number { const value = config[key]; if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) throw new Error(`Gemma 4 audio requer ${key} positivo.`); return value; }
function finite(config: Record<string, unknown>, key: string): number { const value = config[key]; if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`Gemma 4 audio requer ${key} finito.`); return value; }
const f32 = Math.fround;
