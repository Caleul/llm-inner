import { inspectGemma4PackageContract, type Gemma4VisionTowerContract } from "./gemma4-contract.js";
import { pytorchCpuCascadeSquareSumF32, pytorchPowNegativeHalfF32 } from "./executor.js";
import { armNeonBf16DotF32 } from "./native-reductions.js";
import { roundDenseF32ToBF16 } from "./paged-dense.js";
import { sleefCosF32, sleefExpF32, sleefSinF32, sleefTanhF32 } from "./sleef-f32.js";
import { roundF32ToBF16 } from "./utils.js";
import type { DenseF32Tensor, DtypePolicy, ModelCatalog, TensorInfo, TensorRef } from "./types.js";

const ORDERED_F32_REDUCTION = { kind: "ordered-scalar", indexOrder: "ascending" } as const;
const ORDERED_F32_FMA_REDUCTION = { kind: "ordered-fma", indexOrder: "ascending" } as const;
const BF16_ARM_DOT_REDUCTION = {
  kind: "arm-neon-bf16-dot-fma", laneCount: 32, registerCount: 8, lanesPerRegister: 4,
  inputLane: "index-modulo-vector-lane-count", horizontalFold: "pairwise",
} as const;
const F32_POLICY: DtypePolicy = { inputDtype: "F32", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" };
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
 * Executable, dependency-ordered semantic boundary for the Gemma 4 vision
 * tower.  It intentionally is not a generic ViT: every operation below is
 * tied to the registered Gemma4Vision runtime contract, including its
 * checkpointed clipping scalars and two-dimensional RoPE.
 */
export interface Gemma4VisionProgram {
  kind: "gemma4-vision-features";
  sourceFormat: "safetensors";
  tower: Gemma4VisionTowerContract;
  textHiddenSize: number;
  rmsNormEpsilon: number;
  runtimeDtype: "F32" | "BF16";
  assignments: Gemma4VisionAssignment[];
  output: "image_features";
}

export interface Gemma4VisionAssignment {
  id: string;
  operation:
    | "pixel-affine"
    | "linear"
    | "position-embedding-2d"
    | "add"
    | "rms-norm"
    | "clipped-linear"
    | "reshape-heads"
    | "multidimensional-rope"
    | "attention-score-matmul"
    | "masked-softmax"
    | "attention-value-matmul"
    | "gelu-tanh"
    | "multiply"
    | "pool-by-position"
    | "pool-valid-mask"
    | "scale-f32"
    | "strip-padding"
    | "masked-scatter-image-features";
  inputs: string[];
  output: string;
  tensors?: TensorRef[];
  semantics?: string;
  /** Authoritative modal token selected by stable row-major scatter. */
  placeholderTokenId?: number | undefined;
  dtypePolicy?: DtypePolicy;
}

export interface Gemma4VisionExecutionRequest {
  /** [batch, patches, 3 * patch_size^2], already patchified by the processor. */
  pixelValues: DenseF32Tensor;
  /** [batch][patch][x,y]; padding is represented only by the pair [-1,-1]. */
  pixelPositionIds: number[][][];
  /** Exact F32 materialization of every tensor referenced by the program. */
  tensors: ReadonlyMap<string, DenseF32Tensor>;
}

export interface Gemma4VisionExecutionResult {
  values: ReadonlyMap<string, DenseF32Tensor>;
  imageFeatures: DenseF32Tensor;
}

/**
 * Explicitly lower the established vision branch of a composite dense Gemma 4
 * package.  Audio is deliberately not accepted here: callers cannot confuse
 * this feature extractor with a complete Gemma4ForConditionalGeneration IR.
 */
export function buildGemma4VisionProgram(catalog: ModelCatalog): Gemma4VisionProgram {
  const contract = inspectGemma4PackageContract(catalog);
  const config = object(catalog.config, "vision_config");
  const epsilon = finitePositive(config, "rms_norm_eps");
  const runtimeDtype = declaredVisionRuntimeDtype(config);
  const tensors = (names: string[]): TensorRef[] => names.map((name) => tensorRef(requireTensor(catalog, name)));
  const prefix = "model.vision_tower";
  const assignments: Gemma4VisionAssignment[] = [
    { id: "vision_pixels_affine", operation: "pixel-affine", inputs: ["pixel_values"], output: "vision_pixels_standardized", semantics: "F32(2 * (pixel_values - 0.5)) before input_proj" },
    { id: "vision_patch_projection", operation: "linear", inputs: ["vision_pixels_standardized"], output: "vision_patch_projected", tensors: tensors([`${prefix}.patch_embedder.input_proj.weight`]) },
    { id: "vision_position_embedding", operation: "position-embedding-2d", inputs: ["pixel_position_ids"], output: "vision_position_embeddings", tensors: tensors([`${prefix}.patch_embedder.position_embedding_table`]), semantics: "lookup x+y; negative padding coordinates map through zero then are zeroed" },
    { id: "vision_patch_embeddings", operation: "add", inputs: ["vision_patch_projected", "vision_position_embeddings"], output: "vision_hidden_0" },
  ];
  for (let layer = 0; layer < contract.modalities.visionTower.layers; layer += 1) {
    const base = `${prefix}.encoder.layers.${layer}`;
    const input = `vision_hidden_${layer}`;
    const attnNorm = `vision_layer_${layer}_attn_norm`;
    assignments.push(
      { id: `vision_layer_${layer}_input_norm`, operation: "rms-norm", inputs: [input], output: attnNorm, tensors: tensors([`${base}.input_layernorm.weight`]) },
      clippedAssignment(`vision_layer_${layer}_q`, attnNorm, `vision_layer_${layer}_q_linear`, `${base}.self_attn.q_proj`, tensors),
      { id: `vision_layer_${layer}_q_heads`, operation: "reshape-heads", inputs: [`vision_layer_${layer}_q_linear`], output: `vision_layer_${layer}_q_heads`, semantics: "[B,S,H*D] -> [B,H,S,D]" },
      { id: `vision_layer_${layer}_q_norm`, operation: "rms-norm", inputs: [`vision_layer_${layer}_q_heads`], output: `vision_layer_${layer}_q_normalized`, tensors: tensors([`${base}.self_attn.q_norm.weight`]) },
      { id: `vision_layer_${layer}_q_rope`, operation: "multidimensional-rope", inputs: [`vision_layer_${layer}_q_normalized`, "pixel_position_ids"], output: `vision_layer_${layer}_q_rotated`, semantics: "x/y partitions of each head use independent default RoPE frequency ranges" },
      clippedAssignment(`vision_layer_${layer}_k`, attnNorm, `vision_layer_${layer}_k_linear`, `${base}.self_attn.k_proj`, tensors),
      { id: `vision_layer_${layer}_k_heads`, operation: "reshape-heads", inputs: [`vision_layer_${layer}_k_linear`], output: `vision_layer_${layer}_k_heads`, semantics: "[B,S,H*D] -> [B,H,S,D]" },
      { id: `vision_layer_${layer}_k_norm`, operation: "rms-norm", inputs: [`vision_layer_${layer}_k_heads`], output: `vision_layer_${layer}_k_normalized`, tensors: tensors([`${base}.self_attn.k_norm.weight`]) },
      { id: `vision_layer_${layer}_k_rope`, operation: "multidimensional-rope", inputs: [`vision_layer_${layer}_k_normalized`, "pixel_position_ids"], output: `vision_layer_${layer}_k_rotated`, semantics: "x/y partitions of each head use independent default RoPE frequency ranges" },
      clippedAssignment(`vision_layer_${layer}_v`, attnNorm, `vision_layer_${layer}_v_linear`, `${base}.self_attn.v_proj`, tensors),
      { id: `vision_layer_${layer}_v_heads`, operation: "reshape-heads", inputs: [`vision_layer_${layer}_v_linear`], output: `vision_layer_${layer}_v_heads`, semantics: "[B,S,H*D] -> [B,H,S,D]" },
      { id: `vision_layer_${layer}_v_norm`, operation: "rms-norm", inputs: [`vision_layer_${layer}_v_heads`], output: `vision_layer_${layer}_v_normalized`, semantics: "unscaled Gemma4RMSNorm" },
      { id: `vision_layer_${layer}_attention_scores`, operation: "attention-score-matmul", inputs: [`vision_layer_${layer}_q_rotated`, `vision_layer_${layer}_k_rotated`], output: `vision_layer_${layer}_attention_scores`, semantics: "score[b,h,q,k] = BF16(native_batched_matmul(sum_d(q[b,h,q,d] * k[b,h,k,d]))); scale=1" },
      { id: `vision_layer_${layer}_attention_weights`, operation: "masked-softmax", inputs: [`vision_layer_${layer}_attention_scores`, "pixel_position_ids"], output: `vision_layer_${layer}_attention_weights`, semantics: "invalid patch keys become -Infinity; softmax is evaluated in F32 along ascending key and narrowed to BF16" },
      { id: `vision_layer_${layer}_attention`, operation: "attention-value-matmul", inputs: [`vision_layer_${layer}_attention_weights`, `vision_layer_${layer}_v_normalized`], output: `vision_layer_${layer}_attention_context`, semantics: "context[b,q,h,d] = BF16(native_batched_matmul(sum_k(probability[b,h,q,k] * value[b,h,k,d])))" },
      clippedAssignment(`vision_layer_${layer}_o`, `vision_layer_${layer}_attention_context`, `vision_layer_${layer}_attention_projected`, `${base}.self_attn.o_proj`, tensors),
      { id: `vision_layer_${layer}_post_attention_norm`, operation: "rms-norm", inputs: [`vision_layer_${layer}_attention_projected`], output: `vision_layer_${layer}_post_attention_normalized`, tensors: tensors([`${base}.post_attention_layernorm.weight`]) },
      { id: `vision_layer_${layer}_attention_residual`, operation: "add", inputs: [input, `vision_layer_${layer}_post_attention_normalized`], output: `vision_layer_${layer}_after_attention` },
      { id: `vision_layer_${layer}_pre_ffn_norm`, operation: "rms-norm", inputs: [`vision_layer_${layer}_after_attention`], output: `vision_layer_${layer}_ffn_norm`, tensors: tensors([`${base}.pre_feedforward_layernorm.weight`]) },
      clippedAssignment(`vision_layer_${layer}_gate`, `vision_layer_${layer}_ffn_norm`, `vision_layer_${layer}_gate_linear`, `${base}.mlp.gate_proj`, tensors),
      { id: `vision_layer_${layer}_gate_activation`, operation: "gelu-tanh", inputs: [`vision_layer_${layer}_gate_linear`], output: `vision_layer_${layer}_gate_activated` },
      clippedAssignment(`vision_layer_${layer}_up`, `vision_layer_${layer}_ffn_norm`, `vision_layer_${layer}_up_linear`, `${base}.mlp.up_proj`, tensors),
      { id: `vision_layer_${layer}_mlp_product`, operation: "multiply", inputs: [`vision_layer_${layer}_gate_activated`, `vision_layer_${layer}_up_linear`], output: `vision_layer_${layer}_mlp_product` },
      clippedAssignment(`vision_layer_${layer}_down`, `vision_layer_${layer}_mlp_product`, `vision_layer_${layer}_mlp_output`, `${base}.mlp.down_proj`, tensors),
      { id: `vision_layer_${layer}_post_ffn_norm`, operation: "rms-norm", inputs: [`vision_layer_${layer}_mlp_output`], output: `vision_layer_${layer}_post_ffn_normalized`, tensors: tensors([`${base}.post_feedforward_layernorm.weight`]) },
      { id: `vision_layer_${layer}_ffn_residual`, operation: "add", inputs: [`vision_layer_${layer}_after_attention`, `vision_layer_${layer}_post_ffn_normalized`], output: `vision_hidden_${layer + 1}` },
    );
  }
  const finalHidden = `vision_hidden_${contract.modalities.visionTower.layers}`;
  assignments.push(
    { id: "vision_pool", operation: "pool-by-position", inputs: [finalHidden, "pixel_position_ids"], output: "vision_pooled", semantics: "zero padding; spatial average in F32 by exact position-derived kernel; requested length=patches/pooling_kernel_size^2" },
    { id: "vision_pool_mask", operation: "pool-valid-mask", inputs: ["pixel_position_ids"], output: "vision_pool_mask", semantics: "True exactly for spatial pooling cells containing at least one non-padding patch" },
    { id: "vision_pool_scale", operation: "scale-f32", inputs: ["vision_pooled"], output: "vision_pooled_scaled", semantics: "F32(pool * sqrt(hidden_size))" },
    { id: "vision_strip_padding", operation: "strip-padding", inputs: ["vision_pooled_scaled", "vision_pool_mask"], output: "vision_soft_tokens" },
    { id: "vision_language_projection_norm", operation: "rms-norm", inputs: ["vision_soft_tokens"], output: "vision_soft_tokens_normalized", semantics: "unscaled Gemma4MultimodalEmbedder RMSNorm before language projection" },
    { id: "vision_language_projection", operation: "linear", inputs: ["vision_soft_tokens_normalized"], output: "image_features", tensors: tensors(["model.embed_vision.embedding_projection.weight"]) },
    { id: "vision_placeholder_scatter", operation: "masked-scatter-image-features", inputs: ["text_embeddings", "input_ids", "image_features"], output: "text_embeddings_with_images", placeholderTokenId: contract.modalities.imageTokenId, semantics: "replace only image_token_id coordinates; feature rows and placeholder count must agree exactly" },
  );
  for (const assignment of assignments) assignment.dtypePolicy = visionAssignmentDtypePolicy(assignment, runtimeDtype);
  return { kind: "gemma4-vision-features", sourceFormat: "safetensors", tower: contract.modalities.visionTower, textHiddenSize: contract.text.hiddenSize, rmsNormEpsilon: epsilon, runtimeDtype, assignments, output: "image_features" };
}

/** Executes the program's image branch with scalar binary32 boundaries. */
export function executeGemma4VisionF32(program: Gemma4VisionProgram, request: Gemma4VisionExecutionRequest): Gemma4VisionExecutionResult {
  validatePixels(program, request.pixelValues, request.pixelPositionIds);
  const values = new Map<string, DenseF32Tensor>();
  const prefix = "model.vision_tower";
  const get = (name: string): DenseF32Tensor => tensor(request.tensors, name);
  const policyByOutput = new Map(program.assignments.map((assignment) => [assignment.output, assignment.dtypePolicy]));
  const put = (name: string, value: DenseF32Tensor): DenseF32Tensor => {
    const stored = policyByOutput.get(name)?.outputDtype === "BF16" ? roundDenseF32ToBF16(value) : value;
    values.set(name, stored);
    return stored;
  };
  const project = (input: DenseF32Tensor, weight: DenseF32Tensor): DenseF32Tensor =>
    linear(input, weight, program.runtimeDtype === "BF16");
  const normalize = (input: DenseF32Tensor, weight?: DenseF32Tensor): DenseF32Tensor =>
    rmsNorm(input, weight, program.rmsNormEpsilon, program.runtimeDtype === "BF16");
  const positions = request.pixelPositionIds;
  const padding = positions.map((row) => row.map((position) => position[0] === -1 && position[1] === -1));
  const standardized = put("vision_pixels_standardized", affinePixels(request.pixelValues));
  const patchProjected = put("vision_patch_projected", project(standardized, get(`${prefix}.patch_embedder.input_proj.weight`)));
  const positionEmbeddings = put("vision_position_embeddings", positionEmbedding(positions, padding, get(`${prefix}.patch_embedder.position_embedding_table`)));
  put("vision_hidden_0", add(patchProjected, positionEmbeddings));
  for (let layer = 0; layer < program.tower.layers; layer += 1) {
    const base = `${prefix}.encoder.layers.${layer}`;
    const hidden = values.get(`vision_hidden_${layer}`)!;
    const clippedProject = (name: string, input: DenseF32Tensor): DenseF32Tensor => clippedLinear(input, get(`${base}.${name}.linear.weight`), clipBounds(request.tensors, `${base}.${name}`), program.runtimeDtype === "BF16");
    const attnNorm = put(`vision_layer_${layer}_attn_norm`, normalize(hidden, get(`${base}.input_layernorm.weight`)));
    const qLinear = put(`vision_layer_${layer}_q_linear`, clippedProject("self_attn.q_proj", attnNorm));
    const qHeads = put(`vision_layer_${layer}_q_heads`, reshapeHeads(qLinear, program.tower.attentionHeads, program.tower.headDim));
    const qNormalized = put(`vision_layer_${layer}_q_normalized`, normalize(qHeads, get(`${base}.self_attn.q_norm.weight`)));
    const q = put(`vision_layer_${layer}_q_rotated`, multidimensionalRope(qNormalized, positions, program.tower.ropeTheta));
    const kLinear = put(`vision_layer_${layer}_k_linear`, clippedProject("self_attn.k_proj", attnNorm));
    const kHeads = put(`vision_layer_${layer}_k_heads`, reshapeHeads(kLinear, program.tower.attentionHeads, program.tower.headDim));
    const kNormalized = put(`vision_layer_${layer}_k_normalized`, normalize(kHeads, get(`${base}.self_attn.k_norm.weight`)));
    const k = put(`vision_layer_${layer}_k_rotated`, multidimensionalRope(kNormalized, positions, program.tower.ropeTheta));
    const vLinear = put(`vision_layer_${layer}_v_linear`, clippedProject("self_attn.v_proj", attnNorm));
    const vHeads = put(`vision_layer_${layer}_v_heads`, reshapeHeads(vLinear, program.tower.attentionHeads, program.tower.headDim));
    const v = put(`vision_layer_${layer}_v_normalized`, normalize(vHeads));
    const nativeBf16 = program.runtimeDtype === "BF16";
    const scores = put(`vision_layer_${layer}_attention_scores`, attentionScores(q, k, nativeBf16));
    const weights = put(`vision_layer_${layer}_attention_weights`, maskedSoftmax(scores, padding, nativeBf16));
    const context = put(`vision_layer_${layer}_attention_context`, attentionValues(weights, v, nativeBf16));
    const attnProjected = put(`vision_layer_${layer}_attention_projected`, clippedProject("self_attn.o_proj", context));
    const postAttention = put(`vision_layer_${layer}_post_attention_normalized`, normalize(attnProjected, get(`${base}.post_attention_layernorm.weight`)));
    const afterAttention = put(`vision_layer_${layer}_after_attention`, add(hidden, postAttention));
    const ffnNorm = put(`vision_layer_${layer}_ffn_norm`, normalize(afterAttention, get(`${base}.pre_feedforward_layernorm.weight`)));
    const gateLinear = put(`vision_layer_${layer}_gate_linear`, clippedProject("mlp.gate_proj", ffnNorm));
    const gate = put(`vision_layer_${layer}_gate_activated`, geluTanh(gateLinear));
    const up = put(`vision_layer_${layer}_up_linear`, clippedProject("mlp.up_proj", ffnNorm));
    const product = put(`vision_layer_${layer}_mlp_product`, multiply(gate, up));
    const ffn = put(`vision_layer_${layer}_mlp_output`, clippedProject("mlp.down_proj", product));
    const postFfn = put(`vision_layer_${layer}_post_ffn_normalized`, normalize(ffn, get(`${base}.post_feedforward_layernorm.weight`)));
    put(`vision_hidden_${layer + 1}`, add(afterAttention, postFfn));
  }
  const pooled = poolByPosition(values.get(`vision_hidden_${program.tower.layers}`)!, positions, padding, program.tower.poolingKernelSize);
  put("vision_pooled", pooled.tensor); put("vision_pool_mask", boolMask(pooled.valid));
  const pooledScaled = put("vision_pooled_scaled", scale(values.get("vision_pooled")!, Math.sqrt(program.tower.hiddenSize)));
  const softTokens = put("vision_soft_tokens", stripPadding(pooledScaled, pooled.valid));
  const normalizedSoftTokens = put("vision_soft_tokens_normalized", normalize(softTokens));
  const imageFeatures = put("image_features", project(normalizedSoftTokens, get("model.embed_vision.embedding_projection.weight")));
  return { values, imageFeatures };
}

/** Exact Gemma 4 masked_scatter contract, exposed for the composite adapter. */
export function scatterGemma4ImageFeaturesF32(
  textEmbeddings: DenseF32Tensor,
  inputIds: readonly number[][],
  imageTokenId: number,
  imageFeatures: DenseF32Tensor,
): DenseF32Tensor {
  if (textEmbeddings.shape.length !== 3 || inputIds.length !== textEmbeddings.shape[0] || inputIds.some((row) => row.length !== textEmbeddings.shape[1])) throw new Error("Gemma 4 image scatter requer input_ids e embeddings [B,S,D] compatíveis.");
  const width = textEmbeddings.shape[2]!;
  if (imageFeatures.shape.length !== 2 || imageFeatures.shape[1] !== width) throw new Error("Gemma 4 image features possuem hidden_size incompatível.");
  const slots = inputIds.flat().filter((id) => id === imageTokenId).length;
  if (slots !== imageFeatures.shape[0]) throw new Error(`Gemma 4 image placeholder count=${slots} não corresponde a image_features=${imageFeatures.shape[0]}.`);
  const output = Float32Array.from(textEmbeddings.values); let feature = 0;
  for (let batch = 0; batch < inputIds.length; batch += 1) for (let sequence = 0; sequence < inputIds[batch]!.length; sequence += 1) if (inputIds[batch]![sequence] === imageTokenId) {
    output.set(imageFeatures.values.subarray(feature * width, (feature + 1) * width), (batch * inputIds[batch]!.length + sequence) * width); feature += 1;
  }
  return dense([textEmbeddings.shape[0]!, textEmbeddings.shape[1]!, width], output);
}

function clippedAssignment(id: string, input: string, output: string, prefix: string, refs: (names: string[]) => TensorRef[]): Gemma4VisionAssignment {
  return { id, operation: "clipped-linear", inputs: [input], output, tensors: refs([`${prefix}.linear.weight`, `${prefix}.input_min`, `${prefix}.input_max`, `${prefix}.output_min`, `${prefix}.output_max`]), semantics: "clamp(input,input_min,input_max) -> linear(no bias) -> clamp(output,output_min,output_max)" };
}

function tensorRef(tensor: TensorInfo): TensorRef { return { name: tensor.name, shape: [...tensor.logicalShape], storageDtype: tensor.storageDtype, ...(tensor.quantization ? { quantization: tensor.quantization } : {}) }; }
function requireTensor(catalog: ModelCatalog, name: string): TensorInfo { const found = catalog.tensors.get(name); if (!found || found.quantization) throw new Error(`Gemma 4 vision requer tensor denso ${name}.`); return found; }
function object(input: Record<string, unknown>, key: string): Record<string, unknown> { const value = input[key]; if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`Gemma 4 vision requer ${key} objeto.`); return value as Record<string, unknown>; }
function finitePositive(value: unknown, key: string): number { if (typeof value !== "object" || value === null || Array.isArray(value) || typeof (value as Record<string, unknown>)[key] !== "number" || !Number.isFinite((value as Record<string, unknown>)[key]) || (value as Record<string, unknown>)[key] as number <= 0) throw new Error(`Gemma 4 vision requer ${key} positivo.`); return (value as Record<string, number>)[key]!; }

function declaredVisionRuntimeDtype(config: Record<string, unknown>): "F32" | "BF16" {
  const dtype = config.dtype;
  if (dtype === undefined || dtype === "float32") return "F32";
  if (dtype === "bfloat16") return "BF16";
  throw new Error(`Gemma 4 vision dtype '${String(dtype)}' não possui contrato registrado.`);
}

/** Source- and dtype-dispatched policy applied to every compatible vision assignment. */
function visionAssignmentDtypePolicy(assignment: Gemma4VisionAssignment, runtimeDtype: "F32" | "BF16"): DtypePolicy {
  if (assignment.operation === "pool-valid-mask") return { inputDtype: "I64", computeDtype: "BOOL", accumulationDtype: "none", outputDtype: "BOOL" };
  if (assignment.operation === "pixel-affine") return { ...F32_POLICY };
  if (runtimeDtype === "F32") {
    const reduction = assignment.operation === "linear" || assignment.operation === "clipped-linear" || assignment.operation === "rms-norm" ||
      assignment.operation === "attention-score-matmul" || assignment.operation === "masked-softmax" ||
      assignment.operation === "attention-value-matmul" || assignment.operation === "pool-by-position";
    const fma = assignment.operation === "pool-by-position" || assignment.operation === "attention-score-matmul" || assignment.operation === "attention-value-matmul";
    return reduction ? { ...F32_POLICY, ...(fma ? { computeDtype: "F32_FMA", reduction: ORDERED_F32_FMA_REDUCTION } : { reduction: ORDERED_F32_REDUCTION }) } : { ...F32_POLICY };
  }
  if (assignment.operation === "linear" || assignment.operation === "clipped-linear") {
    return { ...BF16_LINEAR_POLICY, reduction: structuredClone(BF16_ARM_DOT_REDUCTION) };
  }
  if (assignment.operation === "rms-norm") return { ...BF16_RMS_POLICY, reduction: structuredClone(BF16_RMS_POLICY.reduction) };
  if (assignment.operation === "attention-score-matmul" || assignment.operation === "attention-value-matmul") {
    return { inputDtype: "BF16", computeDtype: "pytorch-native-batched-matmul", accumulationDtype: "runtime-defined", outputDtype: "BF16" };
  }
  if (assignment.operation === "masked-softmax") {
    return { ...BF16_POLICY, computeDtype: "pytorch-cpu-softmax-f32", reduction: ORDERED_F32_REDUCTION };
  }
  if (assignment.operation === "pool-by-position") {
    return { ...BF16_POLICY, computeDtype: "F32_FMA", reduction: ORDERED_F32_FMA_REDUCTION };
  }
  return { ...BF16_POLICY };
}
const f32 = Math.fround;
function dense(shape: number[], values: Float32Array): DenseF32Tensor { return { shape, values }; }
function tensor(tensors: ReadonlyMap<string, DenseF32Tensor>, name: string): DenseF32Tensor { const found = tensors.get(name); if (!found) throw new Error(`Tensor F32 Gemma 4 vision ausente: ${name}`); return found; }
function validatePixels(program: Gemma4VisionProgram, pixels: DenseF32Tensor, positions: number[][][]): void {
  const expected = 3 * program.tower.patchSize ** 2;
  const validPositions = positions.every((row) => row.every((position) => {
    const x = position[0]; const y = position[1];
    return position.length === 2 && Number.isInteger(x) && Number.isInteger(y) && x !== undefined && y !== undefined &&
      !((x < 0 || y < 0) && !(x === -1 && y === -1)) && x < program.tower.positionEmbeddingSize && y < program.tower.positionEmbeddingSize;
  }));
  if (pixels.shape.length !== 3 || pixels.shape[2] !== expected || positions.length !== pixels.shape[0] || positions.some((row) => row.length !== pixels.shape[1]) || !validPositions) throw new Error(`Gemma 4 vision requer pixel_values [B,P,${expected}] e posições [B,P,2] válidas; padding é somente [-1,-1].`);
  if (pixels.values.length !== pixels.shape.reduce((total, dim) => total * dim, 1)) throw new Error("pixel_values possui buffer incompatível com seu shape.");
}
function affinePixels(input: DenseF32Tensor): DenseF32Tensor { const values = new Float32Array(input.values.length); for (let i = 0; i < values.length; i += 1) values[i] = f32(2 * f32(input.values[i]! - 0.5)); return dense([...input.shape], values); }
function positionEmbedding(positions: number[][][], padding: boolean[][], table: DenseF32Tensor): DenseF32Tensor { if (table.shape.length !== 3 || table.shape[0] !== 2) throw new Error("Tabela de posição Gemma 4 vision inválida."); const [batch, sequence] = [positions.length, positions[0]!.length]; const hidden = table.shape[2]!; const out = new Float32Array(batch * sequence * hidden); for (let b = 0; b < batch; b += 1) for (let s = 0; s < sequence; s += 1) for (let d = 0; d < hidden; d += 1) out[(b * sequence + s) * hidden + d] = padding[b]![s] ? 0 : f32(table.values[positions[b]![s]![0]! * hidden + d]! + table.values[table.shape[1]! * hidden + positions[b]![s]![1]! * hidden + d]!); return dense([batch, sequence, hidden], out); }
function linear(input: DenseF32Tensor, weight: DenseF32Tensor, nativeBf16 = false): DenseF32Tensor { if (weight.shape.length !== 2 || input.shape.at(-1) !== weight.shape[1]) throw new Error("Linear Gemma 4 vision incompatível."); const rows = input.values.length / weight.shape[1]!, outFeatures = weight.shape[0]!, out = new Float32Array(rows * outFeatures); for (let r = 0; r < rows; r += 1) for (let o = 0; o < outFeatures; o += 1) { let sum = f32(0); if (nativeBf16) { const inputBase = r * weight.shape[1]!, weightBase = o * weight.shape[1]!; sum = armNeonBf16DotF32(weight.shape[1]!, (i) => roundF32ToBF16(input.values[inputBase + i]!), (i) => weight.values[weightBase + i]!, BF16_ARM_DOT_REDUCTION); } else for (let i = 0; i < weight.shape[1]!; i += 1) sum = f32(sum + f32(input.values[r * weight.shape[1]! + i]! * weight.values[o * weight.shape[1]! + i]!)); out[r * outFeatures + o] = sum; } return dense([...input.shape.slice(0, -1), outFeatures], out); }
function clipBounds(tensors: ReadonlyMap<string, DenseF32Tensor>, prefix: string): [number, number, number, number] { const names = ["input_min", "input_max", "output_min", "output_max"].map((suffix) => tensor(tensors, `${prefix}.${suffix}`)); if (names.some((entry) => entry.shape.length !== 0 || entry.values.length !== 1)) throw new Error(`${prefix}: limites de clipping devem ser escalares.`); const bounds = names.map((entry) => entry.values[0]!); if (bounds[0]! > bounds[1]! || bounds[2]! > bounds[3]!) throw new Error(`${prefix}: limites de clipping invertidos.`); return bounds as [number, number, number, number]; }
function clippedLinear(input: DenseF32Tensor, weight: DenseF32Tensor, bounds: [number, number, number, number], nativeBf16 = false): DenseF32Tensor { const clamped = new Float32Array(input.values.length); for (let i = 0; i < clamped.length; i += 1) clamped[i] = f32(Math.min(bounds[1], Math.max(bounds[0], input.values[i]!))); const output = linear(dense([...input.shape], clamped), weight, nativeBf16); for (let i = 0; i < output.values.length; i += 1) output.values[i] = f32(Math.min(bounds[3], Math.max(bounds[2], output.values[i]!))); return output; }
function rmsNorm(input: DenseF32Tensor, weight: DenseF32Tensor | undefined, epsilon: number, nativeBf16 = false): DenseF32Tensor { const width = input.shape.at(-1)!; if (weight && (weight.shape.length !== 1 || weight.shape[0] !== width)) throw new Error("RMSNorm Gemma 4 vision incompatível."); const out = new Float32Array(input.values.length); for (let offset = 0; offset < out.length; offset += width) { let sum = f32(0); if (nativeBf16) sum = pytorchCpuCascadeSquareSumF32(input, offset, width); else for (let d = 0; d < width; d += 1) sum = f32(sum + f32(input.values[offset + d]! * input.values[offset + d]!)); const scale = pytorchPowNegativeHalfF32(f32(f32(sum / f32(width)) + f32(epsilon))); for (let d = 0; d < width; d += 1) out[offset + d] = f32(f32(input.values[offset + d]! * scale) * (weight ? weight.values[d]! : 1)); } return dense([...input.shape], out); }
function add(left: DenseF32Tensor, right: DenseF32Tensor): DenseF32Tensor { if (left.shape.some((dim, i) => dim !== right.shape[i])) throw new Error("Soma Gemma 4 vision incompatível."); const out = new Float32Array(left.values.length); for (let i = 0; i < out.length; i += 1) out[i] = f32(left.values[i]! + right.values[i]!); return dense([...left.shape], out); }
function multiply(left: DenseF32Tensor, right: DenseF32Tensor): DenseF32Tensor { if (left.shape.some((dim, i) => dim !== right.shape[i])) throw new Error("Produto Gemma 4 vision incompatível."); const out = new Float32Array(left.values.length); for (let i = 0; i < out.length; i += 1) out[i] = f32(left.values[i]! * right.values[i]!); return dense([...left.shape], out); }
function reshapeHeads(input: DenseF32Tensor, heads: number, dim: number): DenseF32Tensor { if (input.shape.length !== 3 || input.shape[2] !== heads * dim) throw new Error("Heads Gemma 4 vision incompatíveis."); const [batch, sequence] = input.shape as [number, number, number]; const out = new Float32Array(input.values.length); for (let b = 0; b < batch; b += 1) for (let s = 0; s < sequence; s += 1) for (let h = 0; h < heads; h += 1) for (let d = 0; d < dim; d += 1) out[((b * heads + h) * sequence + s) * dim + d] = input.values[(b * sequence + s) * heads * dim + h * dim + d]!; return dense([batch, heads, sequence, dim], out); }
function multidimensionalRope(input: DenseF32Tensor, positions: number[][][], theta: number): DenseF32Tensor { const [batch, heads, sequence, dim] = input.shape as [number, number, number, number]; if (dim % 4 !== 0) throw new Error("RoPE 2-D Gemma 4 vision requer head_dim múltiplo de quatro."); const part = dim / 2, half = part / 2; const out = Float32Array.from(input.values); for (let b = 0; b < batch; b += 1) for (let h = 0; h < heads; h += 1) for (let s = 0; s < sequence; s += 1) for (let axis = 0; axis < 2; axis += 1) for (let pair = 0; pair < half; pair += 1) { const angle = f32(positions[b]![s]![axis]! / f32(theta ** ((2 * pair) / part))); const cosine = roundF32ToBF16(sleefCosF32(angle)), sine = roundF32ToBF16(sleefSinF32(angle)), base = ((b * heads + h) * sequence + s) * dim + axis * part; const first = input.values[base + pair]!, second = input.values[base + pair + half]!; out[base + pair] = f32(roundF32ToBF16(first * cosine) - roundF32ToBF16(second * sine)); out[base + pair + half] = f32(roundF32ToBF16(second * cosine) + roundF32ToBF16(first * sine)); } return dense([...input.shape], out); }
function attentionScores(q: DenseF32Tensor, k: DenseF32Tensor, nativeBf16 = false): DenseF32Tensor { const [batch, heads, sequence, dim] = q.shape as [number, number, number, number]; if (k.shape.length !== 4 || k.shape.some((value, index) => value !== q.shape[index])) throw new Error("Gemma 4 vision Q/K incompatíveis."); const out = new Float32Array(batch * heads * sequence * sequence); for (let b = 0; b < batch; b += 1) for (let h = 0; h < heads; h += 1) for (let query = 0; query < sequence; query += 1) for (let key = 0; key < sequence; key += 1) { let dot = f32(0); for (let d = 0; d < dim; d += 1) dot = f32(dot + q.values[((b * heads + h) * sequence + query) * dim + d]! * k.values[((b * heads + h) * sequence + key) * dim + d]!); out[((b * heads + h) * sequence + query) * sequence + key] = nativeBf16 ? roundF32ToBF16(dot) : dot; } return dense([batch, heads, sequence, sequence], out); }
function maskedSoftmax(scores: DenseF32Tensor, padding: boolean[][], nativeBf16 = false): DenseF32Tensor { const [batch, heads, queries, keys] = scores.shape as [number, number, number, number]; if (scores.shape.length !== 4 || queries !== keys || padding.length !== batch || padding.some((row) => row.length !== keys)) throw new Error("Gemma 4 vision softmax/máscara incompatíveis."); const out = new Float32Array(scores.values.length); for (let b = 0; b < batch; b += 1) for (let h = 0; h < heads; h += 1) for (let query = 0; query < queries; query += 1) { const base = ((b * heads + h) * queries + query) * keys; let max = -Infinity; for (let key = 0; key < keys; key += 1) if (!padding[b]![key]) max = Math.max(max, scores.values[base + key]!); if (max === -Infinity) throw new Error("Gemma 4 vision attention não possui chave válida."); let total = f32(0); for (let key = 0; key < keys; key += 1) if (!padding[b]![key]) { out[base + key] = sleefExpF32(f32(scores.values[base + key]! - f32(max))); total = f32(total + out[base + key]!); } for (let key = 0; key < keys; key += 1) { const probability = padding[b]![key] ? 0 : f32(out[base + key]! / total); out[base + key] = nativeBf16 ? roundF32ToBF16(probability) : probability; } } return dense([...scores.shape], out); }
function attentionValues(weights: DenseF32Tensor, values: DenseF32Tensor, nativeBf16 = false): DenseF32Tensor { const [batch, heads, queries, keys] = weights.shape as [number, number, number, number]; const [valueBatch, valueHeads, valueKeys, dim] = values.shape as [number, number, number, number]; if (weights.shape.length !== 4 || values.shape.length !== 4 || batch !== valueBatch || heads !== valueHeads || keys !== valueKeys) throw new Error("Gemma 4 vision probabilidades/V incompatíveis."); const out = new Float32Array(batch * queries * heads * dim); for (let b = 0; b < batch; b += 1) for (let h = 0; h < heads; h += 1) for (let query = 0; query < queries; query += 1) for (let d = 0; d < dim; d += 1) { let sum = f32(0); for (let key = 0; key < keys; key += 1) sum = f32(sum + weights.values[((b * heads + h) * queries + query) * keys + key]! * values.values[((b * heads + h) * keys + key) * dim + d]!); out[((b * queries + query) * heads + h) * dim + d] = nativeBf16 ? roundF32ToBF16(sum) : sum; } return dense([batch, queries, heads * dim], out); }
function geluTanh(input: DenseF32Tensor): DenseF32Tensor { const out = new Float32Array(input.values.length); for (let i = 0; i < out.length; i += 1) { const x = input.values[i]!; out[i] = f32(f32(0.5 * x) * f32(1 + sleefTanhF32(f32(Math.sqrt(2 / Math.PI) * f32(x + f32(0.044715 * f32(x * f32(x * x)))))))); } return dense([...input.shape], out); }
function poolByPosition(input: DenseF32Tensor, positions: number[][][], padding: boolean[][], kernel: number): { tensor: DenseF32Tensor; valid: boolean[][] } { const [batch, sequence, hidden] = input.shape as [number, number, number]; const length = sequence / (kernel * kernel); if (!Number.isInteger(length)) throw new Error(`Gemma 4 vision não pode pool ${sequence} patches com kernel ${kernel}.`); const out = new Float32Array(batch * length * hidden), weight = f32(1 / f32(kernel * kernel)); const valid = Array.from({ length: batch }, () => Array<boolean>(length).fill(false)); for (let b = 0; b < batch; b += 1) { const xs = positions[b]!.filter((_, index) => !padding[b]![index]).map((position) => position[0]!); if (xs.length === 0) throw new Error("Gemma 4 vision não pode pool uma imagem sem patches válidos."); const maxX = Math.max(...xs) + 1; for (let s = 0; s < sequence; s += 1) { if (padding[b]![s]) continue; const x = positions[b]![s]![0]!, y = positions[b]![s]![1]!; const slot = Math.floor(x / kernel) + Math.floor(maxX / kernel) * Math.floor(y / kernel); if (slot < 0 || slot >= length) throw new Error(`Gemma 4 vision pool position produz slot ${slot} fora de ${length}.`); valid[b]![slot] = true; for (let d = 0; d < hidden; d += 1) out[(b * length + slot) * hidden + d] = f32(out[(b * length + slot) * hidden + d]! + input.values[(b * sequence + s) * hidden + d]! * weight); } } return { tensor: dense([batch, length, hidden], out), valid }; }
function scale(input: DenseF32Tensor, scalar: number): DenseF32Tensor { const out = new Float32Array(input.values.length); for (let i = 0; i < out.length; i += 1) out[i] = f32(input.values[i]! * f32(scalar)); return dense([...input.shape], out); }
function boolMask(valid: boolean[][]): DenseF32Tensor { const values = new Float32Array(valid.flat().map((entry) => entry ? 1 : 0)); return dense([valid.length, valid[0]!.length], values); }
function stripPadding(input: DenseF32Tensor, valid: boolean[][]): DenseF32Tensor { const width = input.shape[2]!; const rows: number[] = []; for (let b = 0; b < valid.length; b += 1) for (let s = 0; s < valid[b]!.length; s += 1) if (valid[b]![s]) for (let d = 0; d < width; d += 1) rows.push(input.values[(b * valid[b]!.length + s) * width + d]!); return dense([rows.length / width, width], Float32Array.from(rows)); }
