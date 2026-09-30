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
import { adaptGgufDecoderCatalog } from "./gguf-llama.js";
import { buildGemma4TextIR } from "./gemma4-text.js";
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

/**
 * A bias is part of the forward equation, not an optional display detail.
 * When the architecture config declares whether a projection family has one,
 * the checkpoint must agree before we lower the linear operation.
 */
type BiasRequirement = "optional" | "required" | "forbidden";

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

/** Context defaults belong to architecture discovery, not scalar lowering. */
export function resolveModelContextLimit(config: JsonObject, modelType: string): number | undefined {
  const configured = optionalNumber(config, ["max_position_embeddings"]);
  if (configured !== undefined) return configured;
  if (modelType === "llama") return 2048;
  return undefined;
}

export async function buildModelIR(
  catalog: ModelCatalog,
  preview: PreviewOptions,
  bridge?: TensorBridge,
): Promise<ModelIR> {
  const adaptedCatalog = adaptGgufDecoderCatalog(catalog);
  const config = selectTextConfig(adaptedCatalog.config);
  const architectureClass = arrayOfStrings(adaptedCatalog.config.architectures)?.[0];
  const modelType = normalizeModelType(
    optionalString(config, ["model_type"]) ??
      optionalString(adaptedCatalog.config, ["model_type"]) ??
      optionalString(adaptedCatalog.rawMetadata, ["general.architecture"]) ??
      "",
  );
  rejectCompositeMultimodalPackage(adaptedCatalog.config, config, modelType);
  if (modelType === "gemma4_text") return buildGemma4TextIR(adaptedCatalog, config, preview);
  rejectKnownUnsupportedArchitecture(adaptedCatalog, config, modelType);
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
    catalog: adaptedCatalog,
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
    config: adaptedCatalog.config,
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

/**
 * A text sub-config describes only one component of a composite checkpoint.
 * It is not authority to discard image/audio token injection, modality towers,
 * or their ordering in the forward pass.  Keep this check before adapter
 * selection so a future text adapter cannot accidentally accept the text
 * weights from a multimodal package as if they were a standalone model.
 */
function rejectCompositeMultimodalPackage(
  packageConfig: JsonObject,
  textConfig: JsonObject,
  textModelType: string,
): void {
  if (packageConfig === textConfig) return;
  const outerModelType = optionalString(packageConfig, ["model_type"]);
  const modalityConfigs = ["vision_config", "audio_config"].filter((key) =>
    typeof packageConfig[key] === "object" && packageConfig[key] !== null && !Array.isArray(packageConfig[key]),
  );
  const modalityTokenIds = ["image_token_id", "audio_token_id", "boi_token_id", "eoi_token_id", "boa_token_id", "eoa_token_id"]
    .filter((key) => Object.hasOwn(packageConfig, key));
  if (modalityConfigs.length === 0 && modalityTokenIds.length === 0) return;

  throw new Error(
    `Pacote composto '${outerModelType ?? "desconhecido"}' contém submodelo de texto '${textModelType || "desconhecido"}' ` +
      `e semântica multimodal declarada (${[...modalityConfigs, ...modalityTokenIds].join(", ")}). ` +
      "Não é seguro compilar somente text_config: um adaptador do pacote composto deve declarar a injeção e a ordem dos tokens/modos antes de reutilizar o adaptador textual.",
  );
}

/**
 * These are named architecture families, not tensor-name guesses.  Reporting
 * the declared semantic boundary makes unsupported packages actionable while
 * preserving the fail-closed contract.  In particular, Gemma 4 text cannot be
 * lowered through the Gemma 2/3 decoder path merely because several projection
 * names coincide.
 */
function rejectKnownUnsupportedArchitecture(
  catalog: ModelCatalog,
  config: JsonObject,
  modelType: string,
): void {
  if (modelType !== "gemma4_text") return;

  const reasons: string[] = [];
  const perLayerWidth = declaredNumber(config, ["hidden_size_per_layer_input"], "hidden_size_per_layer_input");
  if (perLayerWidth !== undefined) {
    reasons.push(`hidden_size_per_layer_input=${perLayerWidth} exige PLE/AltUp explícito`);
  }
  const globalHeadDim = declaredNumber(config, ["global_head_dim"], "global_head_dim");
  if (globalHeadDim !== undefined) {
    reasons.push(`global_head_dim=${globalHeadDim} exige shapes e RoPE por tipo de camada`);
  }
  const ropeParameters = config.rope_parameters;
  if (typeof ropeParameters === "object" && ropeParameters !== null && !Array.isArray(ropeParameters)) {
    const full = (ropeParameters as JsonObject).full_attention;
    if (typeof full === "object" && full !== null && !Array.isArray(full) &&
      optionalString(full as JsonObject, ["rope_type", "type"]) === "proportional") {
      reasons.push("RoPE proporcional para full_attention exige fórmula registrada e executor compatível");
    }
  }
  const shared = declaredNumber(config, ["num_kv_shared_layers"], "num_kv_shared_layers");
  if (shared !== undefined) {
    reasons.push(`num_kv_shared_layers=${shared} exige contrato de ownership/layout específico do runtime Gemma 4`);
  }
  const pleTensors = [
    "language_model.model.embed_tokens_per_layer.weight",
    "model.embed_tokens_per_layer.weight",
  ].filter((name) => catalog.tensors.has(name));
  const layerPleTensors = [
    "per_layer_input_gate.weight",
    "per_layer_projection.weight",
    "post_per_layer_input_norm.weight",
    "layer_scalar",
  ].filter((suffix) =>
    [...catalog.tensors.keys()].some((name) => name.endsWith(`.${suffix}`)),
  );
  if (pleTensors.length > 0 || layerPleTensors.length > 0) {
    reasons.push(`tensores PLE encontrados (${[...pleTensors, ...layerPleTensors].join(", ")})`);
  }
  throw new Error(
    `Gemma 4 text não possui adaptador matemático exato: ${reasons.length > 0 ? reasons.join("; ") : "topologia Gemma 4 não especificada para este compilador"}. ` +
      "Não será rebaixada ao decoder Gemma genérico; implemente operações, política de dtype/cache e validação diferencial do runtime antes de registrá-la.",
  );
}

function normalizeModelType(value: string): string {
  return value.toLowerCase().replaceAll("-", "_");
}

function validateArchitectureConfig(ctx: ArchitectureContext): void {
  for (const [label, value] of [
    ["hidden_size", ctx.hiddenSize],
    ["num_hidden_layers", ctx.numLayers],
    ["num_attention_heads", ctx.numAttentionHeads],
    ["num_key_value_heads", ctx.numKeyValueHeads],
    ["head_dim", ctx.headDim],
  ] as const) {
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`${label} deve ser inteiro positivo; recebido ${value}.`);
    }
  }
  if (ctx.hiddenSize % ctx.numAttentionHeads !== 0 && !ctx.config.head_dim) {
    throw new Error("hidden_size não é divisível por num_attention_heads e head_dim não foi informado.");
  }
  if (ctx.numAttentionHeads % ctx.numKeyValueHeads !== 0) {
    throw new Error("num_attention_heads deve ser múltiplo de num_key_value_heads para GQA/MQA.");
  }
  const epsilon = declaredNumber(ctx.config, ["rms_norm_eps", "layer_norm_epsilon", "attention_layer_norm_rms_epsilon"], "norm epsilon");
  if (epsilon !== undefined && epsilon <= 0) {
    throw new Error(`norm epsilon deve ser positivo; recebido ${epsilon}.`);
  }
  const attentionScalar = declaredNumber(ctx.config, ["query_pre_attn_scalar"], "query_pre_attn_scalar");
  if (attentionScalar !== undefined && attentionScalar <= 0) {
    throw new Error(`query_pre_attn_scalar deve ser positivo; recebido ${attentionScalar}.`);
  }
  for (const label of ["attn_logit_softcapping", "attention_logit_softcapping", "final_logit_softcapping"] as const) {
    const softcap = declaredNumber(ctx.config, [label], label);
    if (softcap !== undefined && softcap <= 0) {
      throw new Error(`${label} deve ser positivo; recebido ${softcap}.`);
    }
  }
  validateAttentionTopologyConfig(ctx);
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

/**
 * Attention layer selection and its window are executable semantics.  A
 * malformed declaration must not quietly become the generic full-attention
 * fallback used when the fields are absent.
 */
function validateAttentionTopologyConfig(ctx: ArchitectureContext): void {
  if (Object.hasOwn(ctx.config, "layer_types")) {
    const layerTypes = ctx.config.layer_types;
    if (!Array.isArray(layerTypes) || layerTypes.length !== ctx.numLayers || !layerTypes.every((value) => value === "full_attention" || value === "sliding_attention")) {
      throw new Error(
        `layer_types deve listar exatamente ${ctx.numLayers} valores 'full_attention' ou 'sliding_attention'.`,
      );
    }
  }
  const window = declaredNumber(ctx.config, ["sliding_window"], "sliding_window");
  if (window !== undefined && (!Number.isInteger(window) || window <= 0)) {
    throw new Error(`sliding_window deve ser inteiro positivo; recebido ${window}.`);
  }
  const pattern = declaredNumber(ctx.config, ["sliding_window_pattern"], "sliding_window_pattern");
  if (pattern !== undefined && (!Number.isInteger(pattern) || pattern <= 0)) {
    throw new Error(`sliding_window_pattern deve ser inteiro positivo; recebido ${pattern}.`);
  }
  if (Object.hasOwn(ctx.config, "num_kv_shared_layers")) {
    const shared = declaredNumber(ctx.config, ["num_kv_shared_layers"], "num_kv_shared_layers");
    if (!Number.isInteger(shared) || shared! <= 0 || shared! > ctx.numLayers) {
      throw new Error(`num_kv_shared_layers deve ser inteiro entre 1 e ${ctx.numLayers}; recebido ${shared}.`);
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
  const tied = tiedWordEmbeddings(ctx.config, ctx.modelType);
  // `tie_word_embeddings` declares the parameter used by the output
  // projection. A serialized duplicate lm_head.weight must not override that
  // contract, because it might be stale or otherwise not the shared parameter.
  const head = tied ? embedding : lmHead;
  if (!head) {
    throw new Error(
      "lm_head não encontrado e tie_word_embeddings não declara um output head amarrado ao embedding.",
    );
  }
  assertMatrixShape(head, "lm_head", ctx.vocabSize, ctx.hiddenSize);
  operations.push(
    await linearOp(
      ctx,
      "lm_head",
      undefined,
      input,
      "logits",
      head,
      "optional",
      findOutputHeadBias(ctx.catalog, lmHead),
    ),
  );
  const finalSoftcap = declaredNumber(ctx.config, ["final_logit_softcapping"], "final_logit_softcapping");
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
  const attentionBias = biasRequirement(ctx.config, ["attention_bias"], "attention_bias");
  const mlpBias = biasRequirement(ctx.config, ["mlp_bias"], "mlp_bias");

  if (tensors.qkvProj) {
    ctx.unsupported.push(`Camada ${layer}: projeção QKV fundida requer split segundo layout específico da arquitetura.`);
  } else {
    operations.push(
      await linearOp(ctx, "q_proj", layer, inputNorm, qLinear, requireTensor(ctx.catalog, tensors.qProj), attentionProjectionBiasRequirement(ctx, "q_proj", attentionBias)),
    );
    if (!tensors.kProj || !tensors.vProj) {
      const shared = resolveSharedKvProducer(ctx, layer);
      if (shared === undefined) {
        throw new Error(`Camada ${layer} não possui K/V e config não declara KV compartilhado.`);
      }
    } else {
      operations.push(
        await linearOp(ctx, "k_proj", layer, inputNorm, kLinear, requireTensor(ctx.catalog, tensors.kProj), attentionProjectionBiasRequirement(ctx, "k_proj", attentionBias)),
        await linearOp(ctx, "v_proj", layer, inputNorm, vLinear, requireTensor(ctx.catalog, tensors.vProj), attentionProjectionBiasRequirement(ctx, "v_proj", attentionBias)),
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
    ...(declaredNumber(ctx.config, ["attn_logit_softcapping", "attention_logit_softcapping"], "attention logit softcap") !== undefined
      ? {
          scoreSoftcap: declaredNumber(ctx.config, ["attn_logit_softcapping", "attention_logit_softcapping"], "attention logit softcap")!,
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
      attentionProjectionBiasRequirement(ctx, "o_proj", attentionBias),
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
      await linearOp(ctx, "gate_proj", layer, mlpInput, gate, requireTensor(ctx.catalog, tensors.gateProj), mlpBias),
      await linearOp(ctx, "up_proj", layer, mlpInput, up, requireTensor(ctx.catalog, tensors.upProj), mlpBias),
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
      await linearOp(ctx, "down_proj", layer, gated, mlpOutput, requireTensor(ctx.catalog, tensors.downProj), mlpBias),
    );
  } else if (tensors.fc1 && tensors.fc2) {
    const fc1 = `layer_${layer}_fc1`;
    operations.push(await linearOp(ctx, "fc1", layer, mlpInput, fc1, requireTensor(ctx.catalog, tensors.fc1), mlpBias));
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
    operations.push(await linearOp(ctx, "fc2", layer, activated, mlpOutput, requireTensor(ctx.catalog, tensors.fc2), mlpBias));
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

/**
 * The pinned Qwen2 eager reference constructs Q/K/V with bias and `o_proj`
 * with `bias=False`, even where a public config omits `attention_bias`.
 * Treating a generic flag as a family-wide switch would accept a weight that
 * native Qwen 2 cannot consume, so the output projection is explicitly
 * forbidden rather than left optional.
 */
function attentionProjectionBiasRequirement(
  ctx: ArchitectureContext,
  projection: "q_proj" | "k_proj" | "v_proj" | "o_proj",
  declared: BiasRequirement,
): BiasRequirement {
  if (ctx.modelType === "qwen2") {
    if (projection === "o_proj") return "forbidden";
    if (declared === "forbidden") {
      throw new Error(
        "Qwen 2 declara attention_bias=false, mas o adaptador Qwen2 registrado exige bias em q_proj, k_proj e v_proj; " +
          "não é seguro omitir esses termos com base em uma flag incompatível.",
      );
    }
    return "required";
  }
  return declared;
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

  const qNorm = findLayerTensor(ctx.catalog, layer, ["self_attn.q_norm.weight", "attention.q_norm.weight", "attn_q_norm.weight"]);
  const kNorm = findLayerTensor(ctx.catalog, layer, ["self_attn.k_norm.weight", "attention.k_norm.weight", "attn_k_norm.weight"]);
  const vNorm = findLayerTensor(ctx.catalog, layer, ["self_attn.v_norm.weight", "attention.v_norm.weight", "attn_v_norm.weight"]);
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
  biasRequirement: BiasRequirement = "optional",
  bias?: TensorInfo,
): Promise<LinearOp> {
  if (weight.logicalShape.length !== 2) {
    throw new Error(`${weight.name} deveria ser matriz 2D; shape=${weight.logicalShape.join("x")}.`);
  }
  const [outFeatures, inFeatures] = weight.logicalShape;
  if (outFeatures === undefined || inFeatures === undefined) throw new Error(`Shape incompleto em ${weight.name}.`);
  const resolvedBias = bias ?? findBias(ctx.catalog, weight.name);
  if (biasRequirement === "required" && !resolvedBias) {
    throw new Error(`${weight.name}: config declara bias obrigatório, mas o tensor ${biasNameForWeight(weight.name)} está ausente.`);
  }
  if (biasRequirement === "forbidden" && resolvedBias) {
    throw new Error(`${weight.name}: config declara ausência de bias, mas o tensor ${resolvedBias.name} está presente.`);
  }
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
  return catalog.tensors.get(biasNameForWeight(weightName));
}

/**
 * An output bias belongs to the output module even when its weight is tied to
 * the input embedding and the serialized lm_head.weight is absent.
 */
function findOutputHeadBias(catalog: ModelCatalog, serializedHead: TensorInfo | undefined): TensorInfo | undefined {
  return serializedHead
    ? findBias(catalog, serializedHead.name)
    : findTensor(catalog, ["lm_head.bias", "language_model.lm_head.bias", "output.bias"]);
}

function biasNameForWeight(weightName: string): string {
  return weightName.endsWith(".weight")
    ? `${weightName.slice(0, -".weight".length)}.bias`
    : `${weightName}.bias`;
}

function biasRequirement(config: JsonObject, keys: string[], label: string): BiasRequirement {
  const declared = keys.filter((key) => Object.hasOwn(config, key)).map((key) => ({ key, value: config[key] }));
  if (declared.length === 0) return "optional";
  if (declared.some(({ value }) => typeof value !== "boolean")) {
    throw new Error(`${label} deve ser booleano quando declarado; não é seguro inferir a semântica de bias.`);
  }
  const value = declared[0]!.value as boolean;
  if (declared.some((entry) => entry.value !== value)) {
    throw new Error(`${label} possui declarações conflitantes; não é seguro escolher a semântica de bias.`);
  }
  return value ? "required" : "forbidden";
}

/**
 * Unlike an optional convenience setting, a present numerical architecture
 * field is a declared part of the model contract.  Do not make a malformed
 * value disappear and then lower the fallback equation instead.
 */
function declaredNumber(config: JsonObject, keys: string[], label: string): number | undefined {
  const declarations = keys.filter((key) => Object.hasOwn(config, key)).map((key) => ({ key, value: config[key] }));
  if (declarations.length === 0) return undefined;
  if (declarations.some(({ value }) => typeof value !== "number" || !Number.isFinite(value))) {
    throw new Error(`${label} deve ser número finito quando declarado; não é seguro usar o fallback.`);
  }
  const value = declarations[0]!.value as number;
  if (declarations.some((entry) => entry.value !== value)) {
    throw new Error(`${label} possui declarações conflitantes; não é seguro escolher uma semântica numérica.`);
  }
  return value;
}

function tiedWordEmbeddings(config: JsonObject, modelType: string): boolean {
  // Gemma 2's registered configuration contract defaults this field to true.
  // A minimal config may omit the serialised default and its Safetensors then
  // legitimately has no lm_head payload. This is an adapter-owned semantic
  // default, not an inference from tensor naming or checkpoint absence.
  if (!Object.hasOwn(config, "tie_word_embeddings")) return modelType === "gemma2";
  if (typeof config.tie_word_embeddings !== "boolean") {
    throw new Error("tie_word_embeddings deve ser booleano quando declarado; não é seguro inferir o peso do output head.");
  }
  return config.tie_word_embeddings;
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
  // Role resolution is deliberately restricted to the adapter's registered
  // complete names. A suffix/substring fallback would make an arbitrary
  // checkpoint naming convention executable semantics: for example,
  // `vendor.layers.0.self_attn.q_proj.weight` could be mistaken for the
  // Llama role merely because it happens to end in `q_proj.weight`. GGUF's
  // registered `blk.N.` convention is already represented in layerPrefixes.
  // New layouts must be added as explicit, reviewed adapter conventions.
  return undefined;
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
  return declaredNumber(config, ["rms_norm_eps", "layer_norm_epsilon", "attention_layer_norm_rms_epsilon"], "norm epsilon") ?? 1e-6;
}

function hiddenActivation(config: JsonObject, modelType: string): { function: string; approximation?: string } {
  const value = stringFrom(config, ["hidden_activation", "hidden_act", "activation_function"], "activation", modelType.startsWith("gemma") ? "gelu_pytorch_tanh" : "silu");
  if (value === "gelu_pytorch_tanh" || value === "gelu_new" || value === "gelu_fast") {
    return { function: "gelu", approximation: "tanh" };
  }
  if (value === "gelu") return { function: "gelu", approximation: "erf" };
  // An activation label is executable mathematical semantics, not a display
  // string.  Passing an unfamiliar config value through would create an IR
  // whose function is unspecified (and which the reference executor may
  // interpret differently or reject only much later).  Keep aliases only
  // where they are definitionally the same formula we model.
  if (value === "silu" || value === "swish") return { function: "silu" };
  throw new Error(
    `Ativação '${value}' não possui semântica exata registrada neste adaptador. ` +
      "Adicione uma operação/implementação validada ou rejeite a arquitetura; nunca propague um nome desconhecido para o IR.",
  );
}

function attentionScale(config: JsonObject, headDim: number): number {
  const scalar = declaredNumber(config, ["query_pre_attn_scalar"], "query_pre_attn_scalar");
  return scalar !== undefined ? scalar ** -0.5 : headDim ** -0.5;
}

function layerType(config: JsonObject, layer: number): string {
  if (Object.hasOwn(config, "layer_types")) {
    const layerTypes = config.layer_types;
    if (!Array.isArray(layerTypes) || layerTypes.length <= layer || (layerTypes[layer] !== "full_attention" && layerTypes[layer] !== "sliding_attention")) {
      throw new Error(`layer_types não possui semântica de atenção válida para a camada ${layer}.`);
    }
    return layerTypes[layer] as string;
  }
  const pattern = declaredNumber(config, ["sliding_window_pattern"], "sliding_window_pattern");
  if (pattern && pattern > 0) return (layer + 1) % pattern === 0 ? "full_attention" : "sliding_attention";
  return declaredNumber(config, ["sliding_window"], "sliding_window") ? "sliding_attention" : "full_attention";
}

function slidingWindow(config: JsonObject, layer: number): number | undefined {
  if (layerType(config, layer) !== "sliding_attention") return undefined;
  return declaredNumber(config, ["sliding_window"], "sliding_window");
}

function resolveSharedKvProducer(ctx: ArchitectureContext, layer: number): number | undefined {
  const numShared = declaredNumber(ctx.config, ["num_kv_shared_layers"], "num_kv_shared_layers");
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
  // Some HF configs retain `rope_parameters: null` while declaring the older
  // `rope_scaling` field.  Null means no value here, not an instruction to
  // discard the alternate explicit declaration.
  const rootKey = config.rope_parameters !== undefined && config.rope_parameters !== null
    ? "rope_parameters"
    : config.rope_scaling !== undefined && config.rope_scaling !== null
      ? "rope_scaling"
      : undefined;
  const rootRaw = rootKey ? config[rootKey] : undefined;
  if (rootRaw !== undefined && (typeof rootRaw !== "object" || rootRaw === null || Array.isArray(rootRaw))) {
    throw new Error(`${rootKey} deve ser um objeto para declarar a semântica RoPE.`);
  }
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
    (parameters ? declaredNumber(parameters, ["rope_theta"], "rope_theta") : undefined) ??
    declaredNumber(config, ["rope_theta", "rope_freq_base"], "rope_theta") ??
    defaultGemma3Theta ??
    10_000;
  if (theta <= 0) throw new Error(`rope_theta deve ser positivo; recebido ${theta}.`);
  const type = parameters ? ropeType(parameters) : "default";
  if (type !== "default") {
    throw new Error(`RoPE '${type}' não possui adaptador matemático registrado; o compilador não pode reduzi-lo a RoPE padrão.`);
  }
  const partial =
    (parameters ? declaredNumber(parameters, ["partial_rotary_factor"], "partial_rotary_factor") : undefined) ??
    declaredNumber(config, ["partial_rotary_factor"], "partial_rotary_factor");
  if (partial !== undefined && (partial <= 0 || partial > 1)) {
    throw new Error(`partial_rotary_factor deve estar no intervalo (0, 1]; recebido ${partial}.`);
  }
  const explicitDim = declaredNumber(config, ["rope_dimension_count"], "rope_dimension_count");
  const dim = explicitDim ?? (partial !== undefined ? Math.floor(headDim * partial) : headDim);
  if (!Number.isInteger(dim) || dim <= 0 || dim > headDim || dim % 2 !== 0) {
    throw new Error(`dimensão RoPE deve ser inteira, positiva, par e não maior que head_dim=${headDim}; recebeu ${dim}.`);
  }
  const layout = "rotate_half";
  return {
    type,
    theta,
    dim,
    layout,
    ...(parameters ? { scaling: parameters } : {}),
  };
}

function ropeType(parameters: JsonObject): string {
  const declarations = ["rope_type", "type"]
    .filter((key) => Object.hasOwn(parameters, key))
    .map((key) => ({ key, value: parameters[key] }));
  if (declarations.length === 0) return "default";
  if (declarations.some(({ value }) => typeof value !== "string" || value.length === 0)) {
    throw new Error("rope_type deve ser string não vazia quando declarado.");
  }
  const value = declarations[0]!.value as string;
  if (declarations.some((entry) => entry.value !== value)) {
    throw new Error("rope_type possui declarações conflitantes; não é seguro escolher uma fórmula.");
  }
  return value;
}
