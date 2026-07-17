import type { JsonObject, LayerIR, ModelCatalog, ModelIR, Operation, PreviewOptions, TensorInfo, TensorRef } from "./types.js";
import { roundF32ToBF16 } from "./utils.js";

const F32_POLICY = { computeDtype: "model-configured", accumulationDtype: "runtime-defined", outputDtype: "model-configured" } as const;
const F32_RUNTIME_POLICY = { inputDtype: "F32", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" } as const;
const ORDERED_SCALAR_REDUCTION = { kind: "ordered-scalar", indexOrder: "ascending" } as const;
const BF16_NATIVE_REDUCTION_POLICY = { inputDtype: "BF16", computeDtype: "F32", accumulationDtype: "F64", outputDtype: "BF16", reduction: ORDERED_SCALAR_REDUCTION } as const;
/**
 * Trace-bound E4B CPU policy for registered MLP assignments only. It encodes
 * the complete finite ARM register tree and is never inferred by the executor.
 * Identical matrix dimensions alone are insufficient evidence for another
 * assignment to inherit it.
 */
const BF16_TRACE_BOUND_ARM_32_MLP_PROJECTION_POLICY = {
  inputDtype: "BF16", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "BF16",
  reduction: {
    kind: "arm-neon-bf16-dot-fma", laneCount: 32, registerCount: 8, lanesPerRegister: 4,
    inputLane: "index-modulo-vector-lane-count", horizontalFold: "pairwise",
  },
} as const;

/**
 * Lowers the authoritative Gemma4Text model only. The outer Gemma 4 package
 * remains a separate composite adapter because image/audio replacement changes
 * the main and PLE inputs before this graph begins.
 */
export function buildGemma4TextIR(catalog: ModelCatalog, config: JsonObject, preview: PreviewOptions): ModelIR {
  const hidden = integer(config, "hidden_size");
  const layers = integer(config, "num_hidden_layers");
  const heads = integer(config, "num_attention_heads");
  const kvHeads = integer(config, "num_key_value_heads");
  const globalKvHeads = optionalInteger(config, "num_global_key_value_heads") ?? kvHeads;
  const headDim = integer(config, "head_dim");
  const globalHeadDim = integer(config, "global_head_dim");
  const intermediate = integer(config, "intermediate_size");
  const vocab = integer(config, "vocab_size");
  const pleWidth = integer(config, "hidden_size_per_layer_input");
  const pleVocab = integer(config, "vocab_size_per_layer_input");
  const epsilon = positive(config, "rms_norm_eps");
  const runtimeDtype = declaredTextRuntimeDtype(config);
  const sharedCount = nonnegative(config, "num_kv_shared_layers");
  const layerTypes = layerTypesOf(config, layers);
  if (heads % kvHeads !== 0 || heads % globalKvHeads !== 0 || sharedCount > layers) throw new Error("Gemma 4 text declara topologia de heads/KV inválida.");
  if (config.attention_bias !== false || config.enable_moe_block !== false) throw new Error("Adaptador Gemma 4 text requer attention_bias=false e enable_moe_block=false explicitamente.");
  if (typeof config.attention_k_eq_v !== "boolean" || typeof config.use_double_wide_mlp !== "boolean") throw new Error("Adaptador Gemma 4 text requer attention_k_eq_v e use_double_wide_mlp booleanos.");
  const prefix = textPrefix(catalog);
  const ref = (name: string): TensorRef => tensorRef(requireTensor(catalog, name));
  const shape = (name: string, expected: readonly number[]): TensorRef => {
    const tensor = requireTensor(catalog, name);
    if (tensor.quantization || tensor.storageDtype !== runtimeDtype || tensor.logicalShape.length !== expected.length || tensor.logicalShape.some((value, index) => value !== expected[index])) throw new Error(`${name}: tensor Gemma 4 text exige storage ${runtimeDtype} denso e shape [${expected.join(", ")}]; recebeu ${tensor.storageDtype} [${tensor.logicalShape.join(", ")}].`);
    return tensorRef(tensor);
  };
  const normalEmbedding = shape(`${prefix}.embed_tokens.weight`, [vocab, hidden]);
  const pleEmbedding = shape(`${prefix}.embed_tokens_per_layer.weight`, [pleVocab, layers * pleWidth]);
  const pleProjection = shape(`${prefix}.per_layer_model_projection.weight`, [layers * pleWidth, hidden]);
  const pleNorm = shape(`${prefix}.per_layer_projection_norm.weight`, [pleWidth]);
  const prelude: Operation[] = [
    // Gemma4TextScaledWordEmbedding converts this scalar to the weight dtype
    // before multiplication.  For the mandatory BF16 package that is an
    // observable cast boundary: sqrt(2560) becomes BF16 50.5, not F32
    // 50.596442... .  Store the widened exact BF16 value in the literal IR so
    // a source-independent F32 reader does not silently invent a scale.
    { id: "token_embedding", op: "embedding", tokenInput: "input_ids", output: "hidden_states_0", weight: normalEmbedding, scale: gemma4TextEmbeddingScale(hidden, normalEmbedding.storageDtype), dtypePolicy: F32_POLICY },
    { id: "ple_token_identity", op: "per_layer_embedding", tokenInput: "input_ids", output: "ple_token_identity", weight: pleEmbedding, numLayers: layers, layerWidth: pleWidth, scale: Math.sqrt(pleWidth), dtypePolicy: F32_POLICY },
    linear("ple_context_projection", "hidden_states_0", "ple_context_packed", pleProjection),
    { id: "ple_context_scale", op: "elementwise", kind: "scale", inputs: ["ple_context_packed"], scalar: hidden ** -0.5, output: "ple_context_scaled", dtypePolicy: F32_POLICY },
    { id: "ple_context_reshape", op: "reshape_per_layer", input: "ple_context_scaled", output: "ple_context_reshaped", numLayers: layers, layerWidth: pleWidth, dtypePolicy: F32_POLICY },
    { id: "ple_context_norm", op: "rms_norm", input: "ple_context_reshaped", output: "ple_context_normalized", weight: pleNorm, epsilon, weightTransform: "direct", axis: -1, dtypePolicy: F32_POLICY },
    { id: "ple_combine", op: "elementwise", kind: "add", inputs: ["ple_context_normalized", "ple_token_identity"], output: "ple_combined", dtypePolicy: F32_POLICY },
    { id: "ple_combine_scale", op: "elementwise", kind: "scale", inputs: ["ple_combined"], scalar: 2 ** -0.5, output: "ple_inputs", dtypePolicy: F32_POLICY },
  ];
  const firstShared = layers - sharedCount;
  const owner = new Map<string, number>();
  for (let layer = 0; layer < firstShared; layer += 1) owner.set(layerTypes[layer]!, layer);
  if (sharedCount > 0 && owner.size !== 2) throw new Error("Gemma 4 text compartilhado requer produtores sliding/full não compartilhados.");
  const lowered: LayerIR[] = [];
  for (let layer = 0; layer < layers; layer += 1) {
    const type = layerTypes[layer]!;
    const sliding = type === "sliding_attention";
    const dim = sliding ? headDim : globalHeadDim;
    const alternative = config.attention_k_eq_v === true && !sliding;
    const layerKvHeads = alternative ? globalKvHeads : kvHeads;
    const shared = layer >= firstShared;
    const layerPrefix = `${prefix}.layers.${layer}`;
    const mlpWidth = intermediate * (config.use_double_wide_mlp === true && shared ? 2 : 1);
    const inputNorm = shape(`${layerPrefix}.input_layernorm.weight`, [hidden]);
    const q = shape(`${layerPrefix}.self_attn.q_proj.weight`, [heads * dim, hidden]);
    const qNorm = shape(`${layerPrefix}.self_attn.q_norm.weight`, [dim]);
    const o = shape(`${layerPrefix}.self_attn.o_proj.weight`, [hidden, heads * dim]);
    const postAttn = shape(`${layerPrefix}.post_attention_layernorm.weight`, [hidden]);
    const preFfn = shape(`${layerPrefix}.pre_feedforward_layernorm.weight`, [hidden]);
    const gate = shape(`${layerPrefix}.mlp.gate_proj.weight`, [mlpWidth, hidden]);
    const up = shape(`${layerPrefix}.mlp.up_proj.weight`, [mlpWidth, hidden]);
    const down = shape(`${layerPrefix}.mlp.down_proj.weight`, [hidden, mlpWidth]);
    const postFfn = shape(`${layerPrefix}.post_feedforward_layernorm.weight`, [hidden]);
    const pleGate = shape(`${layerPrefix}.per_layer_input_gate.weight`, [pleWidth, hidden]);
    const pleProject = shape(`${layerPrefix}.per_layer_projection.weight`, [hidden, pleWidth]);
    const postPle = shape(`${layerPrefix}.post_per_layer_input_norm.weight`, [hidden]);
    const scalar = shape(`${layerPrefix}.layer_scalar`, [1]);
    const ops: Operation[] = [
      { id: `layer_${layer}_input_norm`, layer, op: "rms_norm", input: `hidden_states_${layer}`, output: `layer_${layer}_attn_norm`, weight: inputNorm, epsilon, weightTransform: "direct", axis: -1, dtypePolicy: F32_POLICY },
      linear(`layer_${layer}_q_proj`, `layer_${layer}_attn_norm`, `layer_${layer}_q_linear`, q, layer),
      { id: `layer_${layer}_q_heads`, layer, op: "reshape_heads", input: `layer_${layer}_q_linear`, output: `layer_${layer}_q_heads`, numHeads: heads, headDim: dim, layout: "BHSD", dtypePolicy: F32_POLICY },
      { id: `layer_${layer}_q_norm`, layer, op: "rms_norm", input: `layer_${layer}_q_heads`, output: `layer_${layer}_q_normalized`, weight: qNorm, epsilon, weightTransform: "direct", axis: -1, dtypePolicy: F32_POLICY },
      rope(layer, "q", `layer_${layer}_q_normalized`, dim, type, config),
    ];
    let key = "";
    let value = "";
    const sharedProducer = shared ? owner.get(type) : undefined;
    if (shared) {
      if (sharedProducer === undefined) throw new Error(`Gemma 4 text layer ${layer} lacks a ${type} KV owner.`);
      key = `layer_${sharedProducer}_k_rot`;
      value = `layer_${sharedProducer}_v_normalized`;
    } else {
      const k = shape(`${layerPrefix}.self_attn.k_proj.weight`, [layerKvHeads * dim, hidden]);
      const kNorm = shape(`${layerPrefix}.self_attn.k_norm.weight`, [dim]);
      ops.push(
        linear(`layer_${layer}_k_proj`, `layer_${layer}_attn_norm`, `layer_${layer}_k_linear`, k, layer),
        { id: `layer_${layer}_k_heads`, layer, op: "reshape_heads", input: `layer_${layer}_k_linear`, output: `layer_${layer}_k_heads`, numHeads: layerKvHeads, headDim: dim, layout: "BHSD", dtypePolicy: F32_POLICY },
        { id: `layer_${layer}_k_norm`, layer, op: "rms_norm", input: `layer_${layer}_k_heads`, output: `layer_${layer}_k_normalized`, weight: kNorm, epsilon, weightTransform: "direct", axis: -1, dtypePolicy: F32_POLICY },
        rope(layer, "k", `layer_${layer}_k_normalized`, dim, type, config),
      );
      key = `layer_${layer}_k_rot`;
      if (alternative) {
        ops.push({ id: `layer_${layer}_v_norm`, layer, op: "rms_norm", input: `layer_${layer}_k_heads`, output: `layer_${layer}_v_normalized`, epsilon, weightTransform: "none", axis: -1, dtypePolicy: F32_POLICY });
      } else {
        const v = shape(`${layerPrefix}.self_attn.v_proj.weight`, [layerKvHeads * dim, hidden]);
        ops.push(linear(`layer_${layer}_v_proj`, `layer_${layer}_attn_norm`, `layer_${layer}_v_linear`, v, layer), { id: `layer_${layer}_v_heads`, layer, op: "reshape_heads", input: `layer_${layer}_v_linear`, output: `layer_${layer}_v_heads`, numHeads: layerKvHeads, headDim: dim, layout: "BHSD", dtypePolicy: F32_POLICY }, { id: `layer_${layer}_v_norm`, layer, op: "rms_norm", input: `layer_${layer}_v_heads`, output: `layer_${layer}_v_normalized`, epsilon, weightTransform: "none", axis: -1, dtypePolicy: F32_POLICY });
      }
      value = `layer_${layer}_v_normalized`;
    }
    ops.push(
      { id: `layer_${layer}_attention`, layer, op: "scaled_dot_product_attention", query: `layer_${layer}_q_rot`, key, value, maskInput: `attention_mask:${type}`, output: `layer_${layer}_attention_context`, numAttentionHeads: heads, numKeyValueHeads: layerKvHeads, headDim: dim, scale: 1, softmaxComputeDtype: "float32", causal: true, ...(sliding ? { slidingWindow: integer(config, "sliding_window") } : {}), ...(sharedProducer === undefined ? {} : { kvSharing: { enabled: true, producerLayer: sharedProducer, group: type } }), dtypePolicy: F32_POLICY },
      linear(`layer_${layer}_o_proj`, `layer_${layer}_attention_context`, `layer_${layer}_attention_projected`, o, layer),
      { id: `layer_${layer}_post_attention_norm`, layer, op: "rms_norm", input: `layer_${layer}_attention_projected`, output: `layer_${layer}_post_attention_normalized`, weight: postAttn, epsilon, weightTransform: "direct", axis: -1, dtypePolicy: F32_POLICY },
      elementwise(layer, "attention_residual", "add", [`hidden_states_${layer}`, `layer_${layer}_post_attention_normalized`], `layer_${layer}_after_attention`),
      { id: `layer_${layer}_pre_ffn_norm`, layer, op: "rms_norm", input: `layer_${layer}_after_attention`, output: `layer_${layer}_ffn_norm`, weight: preFfn, epsilon, weightTransform: "direct", axis: -1, dtypePolicy: F32_POLICY },
      linear(`layer_${layer}_gate_proj`, `layer_${layer}_ffn_norm`, `layer_${layer}_gate`, gate, layer),
      linear(`layer_${layer}_up_proj`, `layer_${layer}_ffn_norm`, `layer_${layer}_up`, up, layer),
      { id: `layer_${layer}_activation`, layer, op: "activation", input: `layer_${layer}_gate`, output: `layer_${layer}_gate_activated`, function: "gelu", approximation: "tanh", dtypePolicy: F32_POLICY },
      elementwise(layer, "gated_mlp", "multiply", [`layer_${layer}_gate_activated`, `layer_${layer}_up`], `layer_${layer}_gated_mlp`),
      linear(`layer_${layer}_down_proj`, `layer_${layer}_gated_mlp`, `layer_${layer}_mlp_output`, down, layer),
      { id: `layer_${layer}_post_ffn_norm`, layer, op: "rms_norm", input: `layer_${layer}_mlp_output`, output: `layer_${layer}_post_ffn_normalized`, weight: postFfn, epsilon, weightTransform: "direct", axis: -1, dtypePolicy: F32_POLICY },
      elementwise(layer, "mlp_residual", "add", [`layer_${layer}_after_attention`, `layer_${layer}_post_ffn_normalized`], `layer_${layer}_after_mlp`),
      { id: `layer_${layer}_ple_select`, layer, op: "select_per_layer", input: "ple_inputs", output: `layer_${layer}_ple_input`, layerIndex: layer, numLayers: layers, layerWidth: pleWidth, dtypePolicy: F32_POLICY },
      linear(`layer_${layer}_ple_gate`, `layer_${layer}_after_mlp`, `layer_${layer}_ple_gate_linear`, pleGate, layer),
      { id: `layer_${layer}_ple_activation`, layer, op: "activation", input: `layer_${layer}_ple_gate_linear`, output: `layer_${layer}_ple_gate_activated`, function: "gelu", approximation: "tanh", dtypePolicy: F32_POLICY },
      elementwise(layer, "ple_gated_multiply", "multiply", [`layer_${layer}_ple_gate_activated`, `layer_${layer}_ple_input`], `layer_${layer}_ple_gated`),
      linear(`layer_${layer}_ple_project`, `layer_${layer}_ple_gated`, `layer_${layer}_ple_projected`, pleProject, layer),
      { id: `layer_${layer}_post_ple_norm`, layer, op: "rms_norm", input: `layer_${layer}_ple_projected`, output: `layer_${layer}_post_ple_normalized`, weight: postPle, epsilon, weightTransform: "direct", axis: -1, dtypePolicy: F32_POLICY },
      elementwise(layer, "ple_residual", "add", [`layer_${layer}_after_mlp`, `layer_${layer}_post_ple_normalized`], `layer_${layer}_before_scalar`),
      { id: `layer_${layer}_scalar`, layer, op: "tensor_scale", input: `layer_${layer}_before_scalar`, scalar, output: `hidden_states_${layer + 1}`, dtypePolicy: F32_POLICY },
    );
    lowered.push({ index: layer, layerType: type, operations: ops });
  }
  const finalNorm = shape(`${prefix}.norm.weight`, [hidden]);
  const finalSoftcap = optionalPositive(config, "final_logit_softcapping");
  const epilogue: Operation[] = [
    { id: "final_norm", op: "rms_norm", input: `hidden_states_${layers}`, output: "final_hidden_states", weight: finalNorm, epsilon, weightTransform: "direct", axis: -1, dtypePolicy: F32_POLICY },
    linear("lm_head", "final_hidden_states", "logits", normalEmbedding),
  ];
  if (finalSoftcap !== undefined) epilogue.push({ id: "final_logit_softcap", op: "elementwise", kind: "tanh_softcap", inputs: ["logits"], scalar: finalSoftcap, output: "softcapped_logits", dtypePolicy: F32_POLICY });
  const dtypePolicy = textRuntimeDtypePolicy(runtimeDtype);
  // This is a pinned E4B trace-compatible candidate, not a replay-time shape
  // heuristic. The complete registered topology narrows the declaration to
  // the only E4B operations measured against independent native captures; the
  // executor receives the schedule from the serialized assignment and never
  // selects it from a shape at replay time.
  for (const operation of [...prelude, ...lowered.flatMap((layer) => layer.operations), ...epilogue]) {
    operation.dtypePolicy = runtimeDtype === "BF16" && isTraceBoundGemma4E4bArm32MlpProjection(
      { hidden, intermediate, layers, pleWidth, vocab }, operation.id,
    )
      ? BF16_TRACE_BOUND_ARM_32_MLP_PROJECTION_POLICY
      : runtimeDtype === "BF16" && (operation.op === "linear" || operation.op === "rms_norm")
        ? BF16_NATIVE_REDUCTION_POLICY
        : operation.op === "linear" || operation.op === "rms_norm"
          ? { ...dtypePolicy, reduction: ORDERED_SCALAR_REDUCTION }
          : dtypePolicy;
  }
  return { schemaVersion: 2, source: { path: catalog.source, format: catalog.format }, architecture: { modelType: "gemma4_text", hiddenSize: hidden, intermediateSize: intermediate, numLayers: layers, numAttentionHeads: heads, numKeyValueHeads: kvHeads, headDim, vocabSize: vocab }, config: structuredClone(config), preview, inputs: [{ name: "input_ids", description: "IDs dos tokens de entrada." }, { name: "attention_mask", description: "Máscaras causais por tipo de atenção." }, { name: "position_ids", description: "Posições absolutas RoPE." }, { name: "past_key_values", description: "Estado KV pós-RoPE por camada produtora." }], prelude, layers: lowered, epilogue, fidelity: { exactByConstruction: false, assumptions: ["A política declarada preserva fronteiras BF16 de saída quando a configuração autoritativa as exige; a ordem exata de redução do kernel nativo continua sujeita à validação diferencial."], unsupported: [], warnings: ["Este adaptador cobre Gemma4Text isolado. O pacote multimodal Gemma4 continua rejeitado até vision/audio replacement ser lowered."] } };
}

/** Native Gemma4TextScaledWordEmbedding casts its scalar scale to weight dtype. */
export function gemma4TextEmbeddingScale(hiddenSize: number, storageDtype: string): number {
  const scale = Math.sqrt(hiddenSize);
  return storageDtype === "BF16" ? roundF32ToBF16(scale) : Math.fround(scale);
}

/**
 * The registered reduction contract is bound to the complete E4B text topology,
 * rather than a projection name or a single matrix dimension.
 * Runtime replay never calls this: it receives the explicit schedule already
 * serialized into the selected assignment.
 */
export function isTraceBoundGemma4E4bArm32Topology(topology: {
  hidden: number;
  intermediate: number;
  layers: number;
  pleWidth: number;
  vocab: number;
}): boolean {
  return topology.hidden === 2560 && topology.intermediate === 10240 && topology.layers === 42 &&
    topology.pleWidth === 256 && topology.vocab === 262144;
}

/**
 * Only MLP assignments with independently replayed E4B CPU evidence may carry
 * this native reduction declaration. `layer_0_down_proj` additionally has the
 * pinned PyTorch GEMV source-dispatch evidence. This binding is evaluated
 * while compiling the IR; replay only receives the serialized schedule.
 */
export function isTraceBoundGemma4E4bArm32MlpProjection(
  topology: { hidden: number; intermediate: number; layers: number; pleWidth: number; vocab: number },
  operationId: string,
): boolean {
  return isTraceBoundGemma4E4bArm32Topology(topology) &&
    (operationId === "layer_0_gate_proj" || operationId === "layer_0_up_proj" || operationId === "layer_0_down_proj");
}

/**
 * Gemma4Text modules return tensors in the configured model dtype. The
 * registered eager-BF16 linear/RMSNorm compatibility profiles keep products
 * in F32, reduce them in a declared ordered F64 scalar accumulator, then
 * narrow the result to BF16. `layer_0_gate_proj`, `layer_0_up_proj`, and
 * `layer_0_down_proj` are bound to separately replayed, pinned-runtime
 * captures. Their literal declarations record the complete 32-lane ARM BF16
 * FMA register tree; source and wheel disassembly independently establish the
 * pairwise horizontal fold for `down_proj`. The executor never selects any
 * profile from shape.
 * Other operations retain their source-visible F32 policy.
 * F32 fixtures may omit `dtype`, but an unfamiliar declared runtime dtype is
 * not safe to approximate.
 */
function declaredTextRuntimeDtype(config: JsonObject): "F32" | "BF16" {
  const dtype = config.dtype;
  if (dtype === undefined || dtype === "float32") return "F32";
  if (dtype === "bfloat16") return "BF16";
  throw new Error(`Gemma 4 text dtype '${String(dtype)}' não possui contrato de execução registrado.`);
}

function textRuntimeDtypePolicy(dtype: "F32" | "BF16") {
  if (dtype === "F32") return F32_RUNTIME_POLICY;
  if (dtype === "BF16") {
    return { inputDtype: "BF16", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "BF16" } as const;
  }
  throw new Error(`Gemma 4 text dtype interno não suportado: ${dtype}.`);
}

function rope(layer: number, role: "q" | "k", input: string, dim: number, type: string, config: JsonObject): Operation {
  const parameters = object(object(config, "rope_parameters"), type);
  const ropeType = string(parameters, "rope_type");
  const theta = positive(parameters, "rope_theta");
  if (ropeType !== "default" && ropeType !== "proportional") throw new Error(`Gemma 4 text ${type} RoPE '${ropeType}' não é suportado.`);
  if (ropeType === "proportional" && (typeof parameters.partial_rotary_factor !== "number" || parameters.partial_rotary_factor <= 0 || parameters.partial_rotary_factor > 1)) throw new Error("Gemma 4 proportional RoPE requer partial_rotary_factor em (0,1].");
  return { id: `layer_${layer}_${role}_rope`, layer, op: "rotary_embedding", input, positionInput: "position_ids", output: `layer_${layer}_${role}_rot`, ropeType, theta, rotaryDim: dim, layout: "rotate_half", ...(ropeType === "proportional" ? { scaling: parameters } : {}), dtypePolicy: F32_POLICY };
}

function linear(id: string, input: string, output: string, weight: TensorRef, layer?: number): Operation { return { id, ...(layer === undefined ? {} : { layer }), op: "linear", input, output, weight, inFeatures: weight.shape[1]!, outFeatures: weight.shape[0]!, transposeWeight: true, dtypePolicy: F32_POLICY }; }
function elementwise(layer: number, suffix: string, kind: "add" | "multiply", inputs: string[], output: string): Operation { return { id: `layer_${layer}_${suffix}`, layer, op: "elementwise", kind, inputs, output, dtypePolicy: F32_POLICY }; }
function textPrefix(catalog: ModelCatalog): string { for (const prefix of ["model", "model.language_model", "language_model.model"]) if (catalog.tensors.has(`${prefix}.embed_tokens.weight`)) return prefix; throw new Error("Gemma 4 text não usa um layout textual registrado."); }
function requireTensor(catalog: ModelCatalog, name: string): TensorInfo { const tensor = catalog.tensors.get(name); if (!tensor) throw new Error(`Gemma 4 text exige tensor ${name}.`); return tensor; }
function tensorRef(tensor: TensorInfo): TensorRef { return { name: tensor.name, shape: [...tensor.logicalShape], storageDtype: tensor.storageDtype, ...(tensor.quantization ? { quantization: tensor.quantization } : {}) }; }
function integer(config: JsonObject, key: string): number { const value = config[key]; if (!Number.isInteger(value) || (value as number) <= 0) throw new Error(`Gemma 4 text ${key} deve ser inteiro positivo.`); return value as number; }
function optionalInteger(config: JsonObject, key: string): number | undefined { if (config[key] === undefined || config[key] === null) return undefined; return integer(config, key); }
function nonnegative(config: JsonObject, key: string): number { const value = config[key]; if (!Number.isInteger(value) || (value as number) < 0) throw new Error(`Gemma 4 text ${key} deve ser inteiro não negativo.`); return value as number; }
function positive(config: JsonObject, key: string): number { const value = config[key]; if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) throw new Error(`Gemma 4 text ${key} deve ser número positivo.`); return value; }
function optionalPositive(config: JsonObject, key: string): number | undefined { if (config[key] === undefined || config[key] === null) return undefined; return positive(config, key); }
function object(config: JsonObject, key: string): JsonObject { const value = config[key]; if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`Gemma 4 text ${key} deve ser objeto.`); return value as JsonObject; }
function string(config: JsonObject, key: string): string { const value = config[key]; if (typeof value !== "string" || value.length === 0) throw new Error(`Gemma 4 text ${key} deve ser string.`); return value; }
function layerTypesOf(config: JsonObject, count: number): Array<"sliding_attention" | "full_attention"> { const value = config.layer_types; if (!Array.isArray(value) || value.length !== count || value.some((type) => type !== "sliding_attention" && type !== "full_attention")) throw new Error("Gemma 4 text layer_types inválido."); return value as Array<"sliding_attention" | "full_attention">; }
