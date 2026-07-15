import type {
  ActivationOp,
  AttentionOp,
  DtypePolicy,
  ElementwiseOp,
  JsonObject,
  LayerIR,
  LinearOp,
  ModelCatalog,
  ModelIR,
  Operation,
  PreviewOptions,
  RmsNormOp,
  TensorInfo,
  TensorRef,
} from "./types.js";
import type { TensorBridge } from "./bridge.js";
import {
  arrayOfStrings,
  numberFrom,
  optionalNumber,
  optionalString,
  stringFrom,
} from "./utils.js";

interface ArchitectureContext {
  catalog: ModelCatalog;
  config: JsonObject;
  modelType: string;
  architectureClass?: string;
  hiddenSize: number;
  intermediateSize?: number | number[];
  numLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  vocabSize?: number;
  preview: PreviewOptions;
  bridge?: TensorBridge;
  warnings: string[];
  assumptions: string[];
  unsupported: string[];
}

interface LayerTensors {
  inputNorm: string;
  qProj: string;
  kProj?: string;
  vProj?: string;
  qkvProj?: string;
  oProj: string;
  postAttentionNorm?: string;
  preFeedForwardNorm?: string;
  postFeedForwardNorm?: string;
  qNorm?: string;
  kNorm?: string;
  vNorm?: string;
  gateProj?: string;
  upProj?: string;
  downProj?: string;
  fc1?: string;
  fc2?: string;
}

const SUPPORTED_MODEL_TYPES = new Set([
  "llama",
  "mistral",
  "qwen2",
  "qwen3",
  "gemma",
  "gemma2",
  "gemma3",
  "gemma3_text",
]);

const DEFAULT_DTYPE_POLICY: DtypePolicy = {
  computeDtype: "model-configured",
  accumulationDtype: "runtime-defined",
  outputDtype: "model-configured",
};

export async function buildModelIR(
  catalog: ModelCatalog,
  preview: PreviewOptions,
  bridge?: TensorBridge,
): Promise<ModelIR> {
  const config = selectTextConfig(catalog.config);
  const architectureClass = arrayOfStrings(catalog.config.architectures)?.[0];
  const modelType = normalizeModelType(
    optionalString(config, ["model_type"]) ??
      optionalString(catalog.config, ["model_type"]) ??
      optionalString(catalog.rawMetadata, ["general.architecture"]) ??
      "",
  );
  if (!SUPPORTED_MODEL_TYPES.has(modelType)) {
    throw new Error(
      `Arquitetura '${modelType || "desconhecida"}' não possui adaptador exato. ` +
        "O compilador falha deliberadamente em vez de assumir um bloco Transformer genérico.",
    );
  }

  const hiddenSize = numberFrom(config, ["hidden_size", "embedding_length"], "hidden_size");
  const numLayers = numberFrom(config, ["num_hidden_layers", "block_count"], "num_hidden_layers");
  const numAttentionHeads = numberFrom(
    config,
    ["num_attention_heads", "attention_head_count"],
    "num_attention_heads",
  );
  const numKeyValueHeads = numberFrom(
    config,
    ["num_key_value_heads", "attention_head_count_kv"],
    "num_key_value_heads",
    numAttentionHeads,
  );
  const headDim = numberFrom(
    config,
    ["head_dim", "attention_key_length"],
    "head_dim",
    hiddenSize / numAttentionHeads,
  );
  if (!Number.isInteger(headDim)) {
    throw new Error(`head_dim não inteiro: hidden=${hiddenSize}, heads=${numAttentionHeads}.`);
  }

  const intermediateRaw = config.intermediate_size ?? config.feed_forward_length;
  const intermediateSize =
    typeof intermediateRaw === "number"
      ? intermediateRaw
      : Array.isArray(intermediateRaw) && intermediateRaw.every((v) => typeof v === "number")
        ? (intermediateRaw as number[])
        : undefined;

  const context: ArchitectureContext = {
    catalog,
    config,
    modelType,
    ...(architectureClass ? { architectureClass } : {}),
    hiddenSize,
    ...(intermediateSize !== undefined ? { intermediateSize } : {}),
    numLayers,
    numAttentionHeads,
    numKeyValueHeads,
    headDim,
    ...(typeof config.vocab_size === "number" ? { vocabSize: config.vocab_size } : {}),
    preview,
    ...(bridge ? { bridge } : {}),
    warnings: [],
    assumptions: [],
    unsupported: [],
  };

  context.assumptions.push(
    "O IR preserva a semântica declarada, mas equivalência numérica/bitwise exige validação diferencial contra o runtime de referência com os mesmos dtypes, ordem de redução e política de cache.",
  );
  validateArchitectureConfig(context);
  const prelude = await buildPrelude(context);
  const layers: LayerIR[] = [];
  for (let layer = 0; layer < numLayers; layer += 1) {
    layers.push(await buildDecoderLayer(context, layer));
  }
  const epilogue = await buildEpilogue(context);

  if (context.unsupported.length > 0) {
    throw new Error(`Modelo contém semânticas não implementadas:\n- ${context.unsupported.join("\n- ")}`);
  }

  return {
    schemaVersion: 2,
    source: { path: catalog.source, format: catalog.format },
    architecture: {
      modelType,
      ...(architectureClass ? { architectureClass } : {}),
      hiddenSize,
      ...(intermediateSize !== undefined ? { intermediateSize } : {}),
      numLayers,
      numAttentionHeads,
      numKeyValueHeads,
      headDim,
      ...(context.vocabSize !== undefined ? { vocabSize: context.vocabSize } : {}),
    },
    config: catalog.config,
    preview,
    inputs: [
      { name: "input_ids", description: "IDs dos tokens de entrada." },
      { name: "attention_mask", description: "Máscara causal/padding completa." },
      { name: "position_ids", description: "Posições absolutas usadas pelo RoPE." },
      { name: "past_key_values", description: "Estado KV anterior, quando decode incremental." },
    ],
    prelude,
    layers,
    epilogue,
    fidelity: {
      exactByConstruction: context.assumptions.length === 0 && context.warnings.length === 0,
      assumptions: context.assumptions,
      unsupported: context.unsupported,
      warnings: context.warnings,
    },
  };
}

function selectTextConfig(config: JsonObject): JsonObject {
  const text = config.text_config;
  if (typeof text === "object" && text !== null && !Array.isArray(text)) return text as JsonObject;
  return config;
}

function normalizeModelType(value: string): string {
  return value.toLowerCase().replaceAll("-", "_");
}

function validateArchitectureConfig(ctx: ArchitectureContext): void {
  if (ctx.hiddenSize % ctx.numAttentionHeads !== 0 && !ctx.config.head_dim) {
    throw new Error("hidden_size não é divisível por num_attention_heads e head_dim não foi informado.");
  }
  if (ctx.numAttentionHeads % ctx.numKeyValueHeads !== 0) {
    throw new Error("num_attention_heads deve ser múltiplo de num_key_value_heads para GQA/MQA.");
  }
  const quantMethod = optionalString(ctx.config, ["quant_method"]);
  if (quantMethod && !["mlx", "affine", "mxfp4", "mxfp8", "nvfp4"].includes(quantMethod)) {
    ctx.warnings.push(`quant_method '${quantMethod}' será delegado ao backend do contêiner.`);
  }
  if (ctx.modelType === "mixtral") {
    ctx.unsupported.push("MoE Mixtral exige operações de router, top-k e experts; não pode ser reduzido a MLP densa.");
  }
  if (ctx.modelType === "gemma3n_text") {
    const perLayerInput = optionalNumber(ctx.config, ["hidden_size_per_layer_input"]);
    if (perLayerInput) {
      ctx.unsupported.push("Gemma 3n usa per-layer input embeddings/AltUp/LAuReL; adaptador específico ainda necessário.");
    }
  }
}

async function buildPrelude(ctx: ArchitectureContext): Promise<Operation[]> {
  const embed = findTensor(ctx.catalog, [
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "transformer.wte.weight",
    "token_embd.weight",
  ]);
  if (!embed) throw new Error("Tensor de embedding não encontrado.");
  assertMatrixShape(embed, "embedding", undefined, ctx.hiddenSize);
  if (ctx.vocabSize !== undefined && embed.logicalShape[0] !== ctx.vocabSize) {
    throw new Error(
      `Embedding ${embed.name} possui vocab=${embed.logicalShape[0]}, mas config declara vocab_size=${ctx.vocabSize}.`,
    );
  }
  const scale = ctx.modelType.startsWith("gemma") ? Math.sqrt(ctx.hiddenSize) : undefined;
  return [
    {
      id: "token_embedding",
      op: "embedding",
      tokenInput: "input_ids",
      output: "hidden_states_0",
      weight: tensorRef(embed),
      ...(scale !== undefined ? { scale } : {}),
      dtypePolicy: DEFAULT_DTYPE_POLICY,
    },
  ];
}

async function buildEpilogue(ctx: ArchitectureContext): Promise<Operation[]> {
  const operations: Operation[] = [];
  const norm = findTensor(ctx.catalog, [
    "model.norm.weight",
    "language_model.model.norm.weight",
    "transformer.ln_f.weight",
    "output_norm.weight",
  ]);
  let input = `hidden_states_${ctx.numLayers}`;
  if (norm) {
    assertVectorShape(norm, "norma final", ctx.hiddenSize);
    operations.push({
      id: "final_norm",
      op: "rms_norm",
      input,
      output: "final_hidden_states",
      weight: tensorRef(norm),
      epsilon: normEpsilon(ctx.config),
      weightTransform: usesUnitOffsetRmsNorm(ctx.modelType) ? "one_plus_weight" : "direct",
      axis: -1,
      dtypePolicy: DEFAULT_DTYPE_POLICY,
    });
    input = "final_hidden_states";
  }
  const lmHead = findTensor(ctx.catalog, [
    "lm_head.weight",
    "language_model.lm_head.weight",
    "output.weight",
  ]);
  const embedding = findTensor(ctx.catalog, [
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "token_embd.weight",
  ]);
  const tied = ctx.config.tie_word_embeddings === true;
  const head = lmHead ?? (tied ? embedding : undefined);
  if (!head) throw new Error("lm_head não encontrado e embeddings não estão declarados como tied.");
  assertMatrixShape(head, "lm_head", ctx.vocabSize, ctx.hiddenSize);
  operations.push(await linearOp(ctx, "lm_head", undefined, input, "logits", head, findBias(ctx.catalog, head.name)));
  const finalSoftcap = optionalNumber(ctx.config, ["final_logit_softcapping"]);
  if (finalSoftcap !== undefined) {
    operations.push({
      id: "final_logit_softcap",
      op: "elementwise",
      kind: "tanh_softcap",
      inputs: ["logits"],
      scalar: finalSoftcap,
      output: "softcapped_logits",
      dtypePolicy: DEFAULT_DTYPE_POLICY,
    });
  }
  return operations;
}

async function buildDecoderLayer(ctx: ArchitectureContext, layer: number): Promise<LayerIR> {
  const tensors = resolveLayerTensors(ctx, layer);
  validateLayerTensorShapes(ctx, layer, tensors);
  const operations: Operation[] = [];
  const layerInput = `hidden_states_${layer}`;
  const inputNorm = `layer_${layer}_attn_norm`;
  operations.push(rmsNormOp(ctx, layer, "input_norm", layerInput, inputNorm, tensors.inputNorm));

  const qLinear = `layer_${layer}_q_linear`;
  const kLinear = `layer_${layer}_k_linear`;
  const vLinear = `layer_${layer}_v_linear`;

  if (tensors.qkvProj) {
    ctx.unsupported.push(`Camada ${layer}: projeção QKV fundida requer split segundo layout específico da arquitetura.`);
  } else {
    operations.push(
      await linearOp(ctx, "q_proj", layer, inputNorm, qLinear, requireTensor(ctx.catalog, tensors.qProj)),
    );
    if (!tensors.kProj || !tensors.vProj) {
      const shared = resolveSharedKvProducer(ctx, layer);
      if (shared === undefined) {
        throw new Error(`Camada ${layer} não possui K/V e config não declara KV compartilhado.`);
      }
    } else {
      operations.push(
        await linearOp(ctx, "k_proj", layer, inputNorm, kLinear, requireTensor(ctx.catalog, tensors.kProj)),
        await linearOp(ctx, "v_proj", layer, inputNorm, vLinear, requireTensor(ctx.catalog, tensors.vProj)),
      );
    }
  }

  operations.push(
    {
      id: `layer_${layer}_q_heads`,
      layer,
      op: "reshape_heads",
      input: qLinear,
      output: `layer_${layer}_q_heads`,
      numHeads: ctx.numAttentionHeads,
      headDim: ctx.headDim,
      layout: "BHSD",
      dtypePolicy: DEFAULT_DTYPE_POLICY,
    },
  );

  const hasOwnKv = Boolean(tensors.kProj && tensors.vProj);
  if (hasOwnKv) {
    operations.push(
      {
        id: `layer_${layer}_k_heads`,
        layer,
        op: "reshape_heads",
        input: kLinear,
        output: `layer_${layer}_k_heads`,
        numHeads: ctx.numKeyValueHeads,
        headDim: ctx.headDim,
        layout: "BHSD",
        dtypePolicy: DEFAULT_DTYPE_POLICY,
      },
      {
        id: `layer_${layer}_v_heads`,
        layer,
        op: "reshape_heads",
        input: vLinear,
        output: `layer_${layer}_v_heads`,
        numHeads: ctx.numKeyValueHeads,
        headDim: ctx.headDim,
        layout: "BHSD",
        dtypePolicy: DEFAULT_DTYPE_POLICY,
      },
    );
  }

  // Q/K/V norm weights are head_dim vectors. Applying them to the flattened
  // [batch, sequence, num_heads * head_dim] projection would normalize across
  // heads and therefore describe a different function. The reference decoder
  // implementations reshape first and normalize the final head_dim axis.
  let qHeads = `layer_${layer}_q_heads`;
  let kHeads = `layer_${layer}_k_heads`;
  let vHeads = `layer_${layer}_v_heads`;
  if (tensors.qNorm) {
    const output = `layer_${layer}_q_norm`;
    operations.push(rmsNormOp(ctx, layer, "q_norm", qHeads, output, tensors.qNorm, -1, ctx.headDim));
    qHeads = output;
  }
  if (tensors.kNorm) {
    if (!hasOwnKv) throw new Error(`Camada ${layer}: k_norm requer projeção K própria.`);
    const output = `layer_${layer}_k_norm`;
    operations.push(rmsNormOp(ctx, layer, "k_norm", kHeads, output, tensors.kNorm, -1, ctx.headDim));
    kHeads = output;
  }
  if (tensors.vNorm) {
    if (!hasOwnKv) throw new Error(`Camada ${layer}: v_norm requer projeção V própria.`);
    const output = `layer_${layer}_v_norm`;
    operations.push(rmsNormOp(ctx, layer, "v_norm", vHeads, output, tensors.vNorm, -1, ctx.headDim));
    vHeads = output;
  }

  const rope = ropeConfig(ctx.config, ctx.headDim, ctx.modelType, layer);
  operations.push({
    id: `layer_${layer}_q_rope`,
    layer,
    op: "rotary_embedding",
    input: qHeads,
    positionInput: "position_ids",
    output: `layer_${layer}_q_rot`,
    ropeType: rope.type,
    theta: rope.theta,
    rotaryDim: rope.dim,
    layout: rope.layout,
    ...(rope.scaling ? { scaling: rope.scaling } : {}),
    dtypePolicy: DEFAULT_DTYPE_POLICY,
  });
  if (hasOwnKv) {
    operations.push({
      id: `layer_${layer}_k_rope`,
      layer,
      op: "rotary_embedding",
      input: kHeads,
      positionInput: "position_ids",
      output: `layer_${layer}_k_rot`,
      ropeType: rope.type,
      theta: rope.theta,
      rotaryDim: rope.dim,
      layout: rope.layout,
      ...(rope.scaling ? { scaling: rope.scaling } : {}),
      dtypePolicy: DEFAULT_DTYPE_POLICY,
    });
  }

  const sharedProducer = hasOwnKv ? undefined : resolveSharedKvProducer(ctx, layer);
  const attention: AttentionOp = {
    id: `layer_${layer}_attention`,
    layer,
    op: "scaled_dot_product_attention",
    query: `layer_${layer}_q_rot`,
    key: hasOwnKv ? `layer_${layer}_k_rot` : `layer_${sharedProducer}_k_rot`,
    value: hasOwnKv ? vHeads : `layer_${sharedProducer}_v_heads`,
    maskInput: `attention_mask:${layerType(ctx.config, layer)}`,
    output: `layer_${layer}_attention_context`,
    numAttentionHeads: ctx.numAttentionHeads,
    numKeyValueHeads: ctx.numKeyValueHeads,
    headDim: ctx.headDim,
    scale: attentionScale(ctx.config, ctx.headDim),
    ...(optionalNumber(ctx.config, ["attn_logit_softcapping", "attention_logit_softcapping"]) !== undefined
      ? {
          scoreSoftcap: optionalNumber(ctx.config, ["attn_logit_softcapping", "attention_logit_softcapping"])!,
        }
      : {}),
    softmaxComputeDtype: "float32",
    causal: true,
    ...(slidingWindow(ctx.config, layer) !== undefined
      ? { slidingWindow: slidingWindow(ctx.config, layer)! }
      : {}),
    ...(sharedProducer !== undefined
      ? { kvSharing: { enabled: true, producerLayer: sharedProducer, group: layerType(ctx.config, layer) } }
      : {}),
    dtypePolicy: DEFAULT_DTYPE_POLICY,
  };
  operations.push(attention);

  const attnProjected = `layer_${layer}_attention_projected`;
  operations.push(
    await linearOp(
      ctx,
      "o_proj",
      layer,
      attention.output,
      attnProjected,
      requireTensor(ctx.catalog, tensors.oProj),
    ),
  );

  let attentionBranch = attnProjected;
  if (tensors.postAttentionNorm) {
    const output = `layer_${layer}_post_attention_norm`;
    operations.push(rmsNormOp(ctx, layer, "post_attention_norm", attentionBranch, output, tensors.postAttentionNorm));
    attentionBranch = output;
  }
  const afterAttention = `layer_${layer}_after_attention`;
  operations.push(elementwise(layer, "attention_residual", "add", [layerInput, attentionBranch], afterAttention));

  let mlpInput = afterAttention;
  if (tensors.preFeedForwardNorm) {
    const output = `layer_${layer}_ffn_norm`;
    operations.push(rmsNormOp(ctx, layer, "pre_ffn_norm", mlpInput, output, tensors.preFeedForwardNorm));
    mlpInput = output;
  }

  let mlpOutput: string;
  if (tensors.gateProj && tensors.upProj && tensors.downProj) {
    const gate = `layer_${layer}_gate`;
    const up = `layer_${layer}_up`;
    operations.push(
      await linearOp(ctx, "gate_proj", layer, mlpInput, gate, requireTensor(ctx.catalog, tensors.gateProj)),
      await linearOp(ctx, "up_proj", layer, mlpInput, up, requireTensor(ctx.catalog, tensors.upProj)),
    );
    const activated = `layer_${layer}_gate_activated`;
    const activation = hiddenActivation(ctx.config, ctx.modelType);
    const activationOp: ActivationOp = {
      id: `layer_${layer}_activation`,
      layer,
      op: "activation",
      input: gate,
      output: activated,
      function: activation.function,
      ...(activation.approximation ? { approximation: activation.approximation } : {}),
      dtypePolicy: DEFAULT_DTYPE_POLICY,
    };
    operations.push(activationOp);
    const gated = `layer_${layer}_gated_mlp`;
    operations.push(elementwise(layer, "gated_multiply", "multiply", [activated, up], gated));
    mlpOutput = `layer_${layer}_mlp_output`;
    operations.push(
      await linearOp(ctx, "down_proj", layer, gated, mlpOutput, requireTensor(ctx.catalog, tensors.downProj)),
    );
  } else if (tensors.fc1 && tensors.fc2) {
    const fc1 = `layer_${layer}_fc1`;
    operations.push(await linearOp(ctx, "fc1", layer, mlpInput, fc1, requireTensor(ctx.catalog, tensors.fc1)));
    const activated = `layer_${layer}_fc1_activated`;
    const activation = hiddenActivation(ctx.config, ctx.modelType);
    operations.push({
      id: `layer_${layer}_activation`,
      layer,
      op: "activation",
      input: fc1,
      output: activated,
      function: activation.function,
      ...(activation.approximation ? { approximation: activation.approximation } : {}),
      dtypePolicy: DEFAULT_DTYPE_POLICY,
    });
    mlpOutput = `layer_${layer}_mlp_output`;
    operations.push(await linearOp(ctx, "fc2", layer, activated, mlpOutput, requireTensor(ctx.catalog, tensors.fc2)));
  } else {
    throw new Error(`Camada ${layer}: MLP não reconhecida.`);
  }

  let mlpBranch = mlpOutput;
  if (tensors.postFeedForwardNorm) {
    const output = `layer_${layer}_post_ffn_norm`;
    operations.push(rmsNormOp(ctx, layer, "post_ffn_norm", mlpBranch, output, tensors.postFeedForwardNorm));
    mlpBranch = output;
  }
  const layerOutput = `hidden_states_${layer + 1}`;
  operations.push(elementwise(layer, "mlp_residual", "add", [afterAttention, mlpBranch], layerOutput));

  return { index: layer, layerType: layerType(ctx.config, layer), operations };
}

function resolveLayerTensors(ctx: ArchitectureContext, layer: number): LayerTensors {
  const inputNorm = requireLayerTensor(ctx.catalog, layer, [
    "input_layernorm.weight",
    "attention_norm.weight",
    "attn_norm.weight",
    "ln_1.weight",
  ]);
  const qProj = findLayerTensor(ctx.catalog, layer, ["self_attn.q_proj.weight", "attention.wq.weight", "attn_q.weight"]);
  const kProj = findLayerTensor(ctx.catalog, layer, ["self_attn.k_proj.weight", "attention.wk.weight", "attn_k.weight"]);
  const vProj = findLayerTensor(ctx.catalog, layer, ["self_attn.v_proj.weight", "attention.wv.weight", "attn_v.weight"]);
  const qkvProj = findLayerTensor(ctx.catalog, layer, ["self_attn.qkv_proj.weight", "attn.c_attn.weight", "attn_qkv.weight"]);
  const oProj = requireLayerTensor(ctx.catalog, layer, [
    "self_attn.o_proj.weight",
    "attention.wo.weight",
    "attn_output.weight",
  ]);

  if (!qProj && !qkvProj) throw new Error(`Camada ${layer}: q_proj/qkv_proj ausente.`);

  const postAttentionCandidate = findLayerTensor(ctx.catalog, layer, [
    "post_attention_layernorm.weight",
    "post_attention_norm.weight",
  ]);
  const explicitPreFeedForward = findLayerTensor(ctx.catalog, layer, [
    "pre_feedforward_layernorm.weight",
    "ffn_norm.weight",
  ]);
  const postFeedForward = findLayerTensor(ctx.catalog, layer, ["post_feedforward_layernorm.weight"]);

  // Gemma 2/3/4 têm normalizações de saída do ramo de atenção e uma norma
  // separada antes do MLP. Em Llama/Mistral/Qwen/Gemma 1,
  // post_attention_layernorm é a norma *pré-MLP*, não uma pós-norma do ramo.
  const hasFourNormBlock =
    ["gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text"].includes(ctx.modelType) &&
    explicitPreFeedForward !== undefined;

  const preFeedForwardNorm = hasFourNormBlock
    ? explicitPreFeedForward
    : explicitPreFeedForward ?? postAttentionCandidate;
  const postAttentionNorm = hasFourNormBlock ? postAttentionCandidate : undefined;

  const qNorm = findLayerTensor(ctx.catalog, layer, ["self_attn.q_norm.weight", "attention.q_norm.weight"]);
  const kNorm = findLayerTensor(ctx.catalog, layer, ["self_attn.k_norm.weight", "attention.k_norm.weight"]);
  const vNorm = findLayerTensor(ctx.catalog, layer, ["self_attn.v_norm.weight", "attention.v_norm.weight"]);
  const gateProj = findLayerTensor(ctx.catalog, layer, ["mlp.gate_proj.weight", "feed_forward.w1.weight", "ffn_gate.weight"]);
  const upProj = findLayerTensor(ctx.catalog, layer, ["mlp.up_proj.weight", "feed_forward.w3.weight", "ffn_up.weight"]);
  const downProj = findLayerTensor(ctx.catalog, layer, ["mlp.down_proj.weight", "feed_forward.w2.weight", "ffn_down.weight"]);
  const fc1 = findLayerTensor(ctx.catalog, layer, ["mlp.fc1.weight", "mlp.dense_h_to_4h.weight"]);
  const fc2 = findLayerTensor(ctx.catalog, layer, ["mlp.fc2.weight", "mlp.dense_4h_to_h.weight"]);

  return {
    inputNorm,
    qProj: qProj ?? qkvProj!,
    ...(kProj ? { kProj } : {}),
    ...(vProj ? { vProj } : {}),
    ...(qkvProj ? { qkvProj } : {}),
    oProj,
    ...(postAttentionNorm ? { postAttentionNorm } : {}),
    ...(preFeedForwardNorm ? { preFeedForwardNorm } : {}),
    ...(postFeedForward ? { postFeedForwardNorm: postFeedForward } : {}),
    ...(qNorm ? { qNorm } : {}),
    ...(kNorm ? { kNorm } : {}),
    ...(vNorm ? { vNorm } : {}),
    ...(gateProj ? { gateProj } : {}),
    ...(upProj ? { upProj } : {}),
    ...(downProj ? { downProj } : {}),
    ...(fc1 ? { fc1 } : {}),
    ...(fc2 ? { fc2 } : {}),
  };
}

function rmsNormOp(
  ctx: ArchitectureContext,
  layer: number,
  suffix: string,
  input: string,
  output: string,
  tensorName: string,
  axis = -1,
  expectedWeightLength?: number,
): RmsNormOp {
  const tensor = requireTensor(ctx.catalog, tensorName);
  assertVectorShape(
    tensor,
    expectedWeightLength === undefined
      ? `RMSNorm ${tensorName}`
      : `${tensorName} como vetor RMSNorm de head_dim=${expectedWeightLength}`,
    expectedWeightLength ?? ctx.hiddenSize,
  );
  return {
    id: `layer_${layer}_${suffix}`,
    layer,
    op: "rms_norm",
    input,
    output,
    weight: tensorRef(tensor),
    epsilon: normEpsilon(ctx.config),
    weightTransform: usesUnitOffsetRmsNorm(ctx.modelType) ? "one_plus_weight" : "direct",
    axis,
    dtypePolicy: DEFAULT_DTYPE_POLICY,
  };
}

async function linearOp(
  ctx: ArchitectureContext,
  suffix: string,
  layer: number | undefined,
  input: string,
  output: string,
  weight: TensorInfo,
  bias?: TensorInfo,
): Promise<LinearOp> {
  if (weight.logicalShape.length !== 2) {
    throw new Error(`${weight.name} deveria ser matriz 2D; shape=${weight.logicalShape.join("x")}.`);
  }
  const [outFeatures, inFeatures] = weight.logicalShape;
  if (outFeatures === undefined || inFeatures === undefined) throw new Error(`Shape incompleto em ${weight.name}.`);
  const resolvedBias = bias ?? findBias(ctx.catalog, weight.name);
  if (resolvedBias) assertVectorShape(resolvedBias, `bias de ${weight.name}`, outFeatures);
  const preview =
    ctx.preview.includeWeights && ctx.bridge
      ? await ctx.bridge.readLinearPreview(
          ctx.catalog,
          weight.name,
          ctx.preview.outputRows,
          ctx.preview.inputTerms,
        )
      : undefined;
  return {
    id: layer === undefined ? suffix : `layer_${layer}_${suffix}`,
    ...(layer !== undefined ? { layer } : {}),
    op: "linear",
    input,
    output,
    weight: tensorRef(weight),
    ...(resolvedBias ? { bias: tensorRef(resolvedBias) } : {}),
    inFeatures,
    outFeatures,
    transposeWeight: true,
    ...(preview ? { preview } : {}),
    dtypePolicy: DEFAULT_DTYPE_POLICY,
  };
}

function validateLayerTensorShapes(ctx: ArchitectureContext, layer: number, tensors: LayerTensors): void {
  const hidden = ctx.hiddenSize;
  const qOut = ctx.numAttentionHeads * ctx.headDim;
  const kvOut = ctx.numKeyValueHeads * ctx.headDim;
  assertVectorShape(requireTensor(ctx.catalog, tensors.inputNorm), `input norm da camada ${layer}`, hidden);
  assertMatrixShape(requireTensor(ctx.catalog, tensors.qProj), `q_proj da camada ${layer}`, qOut, hidden);

  if (tensors.kProj || tensors.vProj) {
    if (!tensors.kProj || !tensors.vProj) {
      throw new Error(`Camada ${layer}: K e V devem estar ambos presentes ou ambos serem compartilhados.`);
    }
    assertMatrixShape(requireTensor(ctx.catalog, tensors.kProj), `k_proj da camada ${layer}`, kvOut, hidden);
    assertMatrixShape(requireTensor(ctx.catalog, tensors.vProj), `v_proj da camada ${layer}`, kvOut, hidden);
  }
  assertMatrixShape(requireTensor(ctx.catalog, tensors.oProj), `o_proj da camada ${layer}`, hidden, qOut);

  for (const [label, name] of [
    ["post-attention norm", tensors.postAttentionNorm],
    ["pre-feedforward norm", tensors.preFeedForwardNorm],
    ["post-feedforward norm", tensors.postFeedForwardNorm],
  ] as const) {
    if (name) assertVectorShape(requireTensor(ctx.catalog, name), `${label} da camada ${layer}`, hidden);
  }

  const intermediate = intermediateSizeForLayer(ctx, layer);
  if (tensors.gateProj || tensors.upProj || tensors.downProj) {
    if (!tensors.gateProj || !tensors.upProj || !tensors.downProj) {
      throw new Error(`Camada ${layer}: MLP gated exige gate_proj, up_proj e down_proj.`);
    }
    assertMatrixShape(requireTensor(ctx.catalog, tensors.gateProj), `gate_proj da camada ${layer}`, intermediate, hidden);
    assertMatrixShape(requireTensor(ctx.catalog, tensors.upProj), `up_proj da camada ${layer}`, intermediate, hidden);
    assertMatrixShape(requireTensor(ctx.catalog, tensors.downProj), `down_proj da camada ${layer}`, hidden, intermediate);
  } else if (tensors.fc1 || tensors.fc2) {
    if (!tensors.fc1 || !tensors.fc2) throw new Error(`Camada ${layer}: MLP densa exige fc1 e fc2.`);
    assertMatrixShape(requireTensor(ctx.catalog, tensors.fc1), `fc1 da camada ${layer}`, intermediate, hidden);
    assertMatrixShape(requireTensor(ctx.catalog, tensors.fc2), `fc2 da camada ${layer}`, hidden, intermediate);
  }
}

function intermediateSizeForLayer(ctx: ArchitectureContext, layer: number): number {
  if (typeof ctx.intermediateSize === "number") return ctx.intermediateSize;
  if (Array.isArray(ctx.intermediateSize) && ctx.intermediateSize[layer] !== undefined) {
    return ctx.intermediateSize[layer]!;
  }
  throw new Error(`Camada ${layer}: intermediate_size obrigatório para validar a topologia da MLP.`);
}

function assertVectorShape(tensor: TensorInfo, label: string, length: number): void {
  if (tensor.logicalShape.length !== 1 || tensor.logicalShape[0] !== length) {
    throw new Error(`${label} deve ter shape ${length}; recebido ${tensor.logicalShape.join("x")}.`);
  }
}

function assertMatrixShape(
  tensor: TensorInfo,
  label: string,
  rows: number | undefined,
  columns: number,
): void {
  if (
    tensor.logicalShape.length !== 2 ||
    tensor.logicalShape[1] !== columns ||
    (rows !== undefined && tensor.logicalShape[0] !== rows)
  ) {
    const expected = rows === undefined ? `*x${columns}` : `${rows}x${columns}`;
    throw new Error(`${label} deve ter shape ${expected}; recebido ${tensor.logicalShape.join("x")}.`);
  }
}

function elementwise(
  layer: number,
  suffix: string,
  kind: ElementwiseOp["kind"],
  inputs: string[],
  output: string,
): ElementwiseOp {
  return {
    id: `layer_${layer}_${suffix}`,
    layer,
    op: "elementwise",
    kind,
    inputs,
    output,
    dtypePolicy: DEFAULT_DTYPE_POLICY,
  };
}

function tensorRef(tensor: TensorInfo): TensorRef {
  return {
    name: tensor.name,
    shape: [...tensor.logicalShape],
    storageDtype: tensor.storageDtype,
    ...(tensor.quantization ? { quantization: tensor.quantization } : {}),
  };
}

function requireTensor(catalog: ModelCatalog, name: string): TensorInfo {
  const tensor = catalog.tensors.get(name);
  if (!tensor) throw new Error(`Tensor obrigatório ausente: ${name}`);
  return tensor;
}

function findTensor(catalog: ModelCatalog, exactNames: string[]): TensorInfo | undefined {
  for (const name of exactNames) {
    const tensor = catalog.tensors.get(name);
    if (tensor) return tensor;
  }
  return undefined;
}

function findBias(catalog: ModelCatalog, weightName: string): TensorInfo | undefined {
  const biasName = weightName.endsWith(".weight")
    ? `${weightName.slice(0, -".weight".length)}.bias`
    : `${weightName}.bias`;
  return catalog.tensors.get(biasName);
}

function layerPrefixes(layer: number): string[] {
  return [
    `model.layers.${layer}.`,
    `language_model.model.layers.${layer}.`,
    `transformer.h.${layer}.`,
    `layers.${layer}.`,
    `blk.${layer}.`,
    `blocks.${layer}.`,
  ];
}

function findLayerTensor(catalog: ModelCatalog, layer: number, suffixes: string[]): string | undefined {
  for (const prefix of layerPrefixes(layer)) {
    for (const suffix of suffixes) {
      const name = `${prefix}${suffix}`;
      if (catalog.tensors.has(name)) return name;
    }
  }
  // GGUF usa blk.N.<nome> e já está contemplado; fallback controlado por final de nome.
  const layerMarkers = [`.${layer}.`, `blk.${layer}.`];
  const candidates = [...catalog.tensors.keys()].filter(
    (name) => layerMarkers.some((marker) => name.includes(marker)) && suffixes.some((suffix) => name.endsWith(suffix)),
  );
  if (candidates.length > 1) {
    throw new Error(`Tensor ambíguo na camada ${layer}: ${candidates.join(", ")}`);
  }
  return candidates[0];
}

function requireLayerTensor(catalog: ModelCatalog, layer: number, suffixes: string[]): string {
  const found = findLayerTensor(catalog, layer, suffixes);
  if (!found) throw new Error(`Camada ${layer}: tensor não encontrado para ${suffixes.join(" | ")}`);
  return found;
}

function usesUnitOffsetRmsNorm(modelType: string): boolean {
  return ["gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n_text"].includes(modelType);
}

function normEpsilon(config: JsonObject): number {
  return numberFrom(config, ["rms_norm_eps", "layer_norm_epsilon", "attention_layer_norm_rms_epsilon"], "norm epsilon", 1e-6);
}

function hiddenActivation(config: JsonObject, modelType: string): { function: string; approximation?: string } {
  const value = stringFrom(config, ["hidden_activation", "hidden_act", "activation_function"], "activation", modelType.startsWith("gemma") ? "gelu_pytorch_tanh" : "silu");
  if (value === "gelu_pytorch_tanh" || value === "gelu_new" || value === "gelu_fast") {
    return { function: "gelu", approximation: "tanh" };
  }
  if (value === "gelu") return { function: "gelu", approximation: "erf" };
  return { function: value };
}

function attentionScale(config: JsonObject, headDim: number): number {
  const scalar = optionalNumber(config, ["query_pre_attn_scalar"]);
  return scalar !== undefined ? scalar ** -0.5 : headDim ** -0.5;
}

function layerType(config: JsonObject, layer: number): string {
  const layerTypes = arrayOfStrings(config.layer_types);
  if (layerTypes?.[layer]) return layerTypes[layer]!;
  const pattern = optionalNumber(config, ["sliding_window_pattern"]);
  if (pattern && pattern > 0) return (layer + 1) % pattern === 0 ? "full_attention" : "sliding_attention";
  return optionalNumber(config, ["sliding_window"]) ? "sliding_attention" : "full_attention";
}

function slidingWindow(config: JsonObject, layer: number): number | undefined {
  if (layerType(config, layer) !== "sliding_attention") return undefined;
  return optionalNumber(config, ["sliding_window"]);
}

function resolveSharedKvProducer(ctx: ArchitectureContext, layer: number): number | undefined {
  const numShared = optionalNumber(ctx.config, ["num_kv_shared_layers"]);
  if (!numShared || layer < ctx.numLayers - numShared) return undefined;
  const type = layerType(ctx.config, layer);
  for (let candidate = layer - 1; candidate >= 0; candidate -= 1) {
    if (layerType(ctx.config, candidate) !== type) continue;
    const k = findLayerTensor(ctx.catalog, candidate, ["self_attn.k_proj.weight", "attention.wk.weight", "attn_k.weight"]);
    const v = findLayerTensor(ctx.catalog, candidate, ["self_attn.v_proj.weight", "attention.wv.weight", "attn_v.weight"]);
    if (k && v) return candidate;
  }
  return undefined;
}

function ropeConfig(
  config: JsonObject,
  headDim: number,
  modelType: string,
  layer: number,
): {
  type: string;
  theta: number;
  dim: number;
  layout: "rotate_half" | "interleaved_pairs" | "multidimensional";
  scaling?: JsonObject;
} {
  const rootRaw = config.rope_parameters ?? config.rope_scaling;
  const root =
    typeof rootRaw === "object" && rootRaw !== null && !Array.isArray(rootRaw)
      ? (rootRaw as JsonObject)
      : undefined;
  const currentLayerType = layerType(config, layer);
  const nestedRaw = root?.[currentLayerType];
  const parameters =
    typeof nestedRaw === "object" && nestedRaw !== null && !Array.isArray(nestedRaw)
      ? (nestedRaw as JsonObject)
      : root;

  const defaultGemma3Theta =
    modelType === "gemma3" || modelType === "gemma3_text"
      ? currentLayerType === "sliding_attention"
        ? 10_000
        : 1_000_000
      : undefined;
  const theta =
    (parameters ? optionalNumber(parameters, ["rope_theta"]) : undefined) ??
    optionalNumber(config, ["rope_theta", "rope_freq_base"]) ??
    defaultGemma3Theta ??
    10_000;
  const type =
    (parameters ? optionalString(parameters, ["rope_type", "type"]) : undefined) ?? "default";
  const partial =
    (parameters ? optionalNumber(parameters, ["partial_rotary_factor"]) : undefined) ??
    optionalNumber(config, ["partial_rotary_factor"]);
  const explicitDim = optionalNumber(config, ["rope_dimension_count"]);
  const dim = explicitDim ?? (partial !== undefined ? Math.floor(headDim * partial) : headDim);
  const layout = modelType === "gemma4" || modelType === "gemma4_text" ? "multidimensional" : "rotate_half";
  return {
    type,
    theta,
    dim,
    layout,
    ...(parameters ? { scaling: parameters } : {}),
  };
}
