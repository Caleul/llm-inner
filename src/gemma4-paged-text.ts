import {
  activationF32,
  attentionF32,
  concatSequenceF32,
  elementwiseF32,
  reshapeHeadsF32,
  reshapePerLayerF32,
  rmsNormF32,
  rotaryF32,
  selectPerLayerF32,
  tensorScaleF32,
} from "./executor.js";
import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import {
  executeGemma4LiteralGenerationProgram,
  type Gemma4LiteralGenerationExecutionResult,
} from "./gemma4-literal-generation.js";
import { createPagedDenseF32Matrix, pagedEmbeddingF32, pagedLinearF32, readPagedDenseF32Vector, roundDenseF32ToBF16 } from "./paged-dense.js";
import type {
  DenseF32Tensor,
  Operation,
  ReferenceF32ExecutionResult,
  ReferenceF32KeyValueCache,
  TensorInfo,
  TensorRef,
} from "./types.js";

export interface Gemma4PagedTextExecutionRequest {
  inputIds: number[][];
  positionIds?: number[][];
  attentionMask?: DenseF32Tensor;
  attentionMasksByLayer?: ReadonlyMap<number, DenseF32Tensor>;
  pastKeyValues?: ReadonlyMap<number, ReferenceF32KeyValueCache>;
}

export interface Gemma4PagedTextGenerationRequest extends Gemma4PagedTextExecutionRequest {
  maxNewTokens: number;
  eosTokenId?: number;
}

export interface Gemma4PagedTextOptions {
  /** Maximum decoded literal-storage range held by one matrix/vector read. */
  maxReadBytes?: number;
  /**
   * Required only when the embedded program declares unresolved numerical
   * fidelity. This keeps a diagnostic candidate replay from looking like an
   * established exact execution contract.
   */
  allowUnverifiedFidelity?: boolean;
}

/**
 * Executes the declared Gemma4Text graph directly from an indexed literal
 * artifact. Matrix bytes are decoded a bounded output-row page at a time;
 * vector constants are bounded separately. This deliberately accepts no
 * image, video, or audio values: those tower paths remain fail-closed until
 * their own storage-backed kernels exist.
 */
export async function executeGemma4PagedTextLiteralF32(
  artifact: OpenGemma4CompositeLiteralArtifact,
  request: Gemma4PagedTextExecutionRequest,
  options: Gemma4PagedTextOptions = {},
): Promise<ReferenceF32ExecutionResult> {
  const maxReadBytes = options.maxReadBytes ?? 16 * 1024 * 1024;
  assertExecutionFidelityAcknowledged(artifact, options);
  const inputIds = request.inputIds;
  if (inputIds.length === 0 || inputIds.some((row) => row.length === 0 || row.length !== inputIds[0]!.length)) {
    throw new Error("Gemma 4 paginado requer input_ids não vazio e retangular.");
  }
  const sequence = inputIds[0]!.length;
  const positions = request.positionIds ?? inputIds.map((row) => row.map((_, index) => index));
  if (positions.length !== inputIds.length || positions.some((row) => row.length !== sequence)) {
    throw new Error("Gemma 4 paginado position_ids deve acompanhar input_ids.");
  }
  const vectors = new Map<string, Promise<DenseF32Tensor>>();
  const matrix = (reference: TensorRef) => createPagedDenseF32Matrix(tensorInfo(artifact, reference), artifact, maxReadBytes);
  const vector = (reference: TensorRef): Promise<DenseF32Tensor> => {
    let result = vectors.get(reference.name);
    if (!result) {
      result = readPagedDenseF32Vector(tensorInfo(artifact, reference), artifact, maxReadBytes);
      vectors.set(reference.name, result);
    }
    return result;
  };
  const values = new Map<string, DenseF32Tensor>();
  const store = (operation: Operation, tensor: DenseF32Tensor): void => {
    values.set(operation.output, operation.dtypePolicy.outputDtype === "BF16" ? roundDenseF32ToBF16(tensor) : tensor);
  };
  const producedCache = new Map<number, ReferenceF32KeyValueCache>();
  const operations = [...artifact.program.textProgram.prelude, ...artifact.program.textProgram.layers.flatMap((layer) => layer.operations), ...artifact.program.textProgram.epilogue];
  for (const operation of operations) {
    assertPagedF32Policy(operation);
    switch (operation.op) {
      case "embedding":
        store(operation, await pagedEmbeddingF32(inputIds, matrix(operation.weight), operation.scale, {
          roundOutputToBf16: operation.weight.storageDtype === "BF16",
        }));
        break;
      case "per_layer_embedding":
        store(operation, reshapePerLayerF32(await pagedEmbeddingF32(inputIds, matrix(operation.weight), operation.scale, {
          roundOutputToBf16: operation.weight.storageDtype === "BF16",
        }), operation.numLayers, operation.layerWidth));
        break;
      case "rms_norm":
        store(operation, rmsNormF32(value(values, operation.input), operation.weight ? await vector(operation.weight) : undefined, operation));
        break;
      case "reshape_per_layer":
        store(operation, reshapePerLayerF32(value(values, operation.input), operation.numLayers, operation.layerWidth));
        break;
      case "select_per_layer":
        store(operation, selectPerLayerF32(value(values, operation.input), operation));
        break;
      case "tensor_scale":
        store(operation, tensorScaleF32(value(values, operation.input), await vector(operation.scalar), operation.id));
        break;
      case "linear":
        if (!operation.transposeWeight || operation.bias) throw new Error(`${operation.id}: executor Gemma 4 paginado requer linear [out,in] sem bias.`);
        store(operation, await pagedLinearF32(value(values, operation.input), matrix(operation.weight), {
          outputDtype: operation.dtypePolicy.outputDtype === "BF16" ? "BF16" : "F32",
          accumulationDtype: operation.dtypePolicy.accumulationDtype === "F64" ? "F64" : "F32",
          ...(operation.dtypePolicy.reduction ? { reduction: operation.dtypePolicy.reduction } : {}),
        }));
        break;
      case "reshape_heads":
        if (operation.layout !== "BHSD") throw new Error(`${operation.id}: executor Gemma 4 paginado requer layout BHSD.`);
        store(operation, reshapeHeadsF32(value(values, operation.input), operation.numHeads, operation.headDim));
        break;
      case "rotary_embedding":
        store(operation, rotaryF32(value(values, operation.input), positions, operation));
        break;
      case "scaled_dot_product_attention": {
        if (operation.layer === undefined) throw new Error(`${operation.id}: attention sem índice não pode possuir cache KV.`);
        const topologyMask = request.attentionMasksByLayer?.get(operation.layer);
        if (operation.kvSharing) {
          const producer = operation.kvSharing.producerLayer;
          if (producer === undefined) throw new Error(`${operation.id}: KV compartilhado sem produtor declarado.`);
          const shared = producedCache.get(producer);
          if (!shared) throw new Error(`${operation.id}: KV compartilhado não encontrou cache do produtor ${producer}.`);
          const query = value(values, operation.query);
          store(operation, attentionF32(query, shared.key, shared.value, operation, topologyMask ?? request.attentionMask, sharedPastLength(operation.id, shared, query), topologyMask !== undefined));
          break;
        }
        const currentKey = value(values, operation.key);
        const currentValue = value(values, operation.value);
        const previous = request.pastKeyValues?.get(operation.layer);
        if (request.pastKeyValues && !previous) throw new Error(`${operation.id}: pastKeyValues não contém a camada ${operation.layer}.`);
        if (previous) assertCompatibleCache(previous, currentKey, currentValue, operation.id);
        const key = previous ? concatSequenceF32(previous.key, currentKey) : currentKey;
        const valueTensor = previous ? concatSequenceF32(previous.value, currentValue) : currentValue;
        store(operation, attentionF32(value(values, operation.query), key, valueTensor, operation, topologyMask ?? request.attentionMask, previous?.key.shape[2] ?? 0, topologyMask !== undefined));
        producedCache.set(operation.layer, { key, value: valueTensor });
        break;
      }
      case "activation":
        store(operation, activationF32(value(values, operation.input), operation.function, operation.approximation));
        break;
      case "elementwise":
        store(operation, elementwiseF32(operation.inputs.map((name) => value(values, name)), operation.kind, operation.scalar));
        break;
      default: {
        const unsupported: never = operation;
        throw new Error(`Operação Gemma 4 paginada não suportada: ${JSON.stringify(unsupported)}.`);
      }
    }
  }
  const logits = values.get("softcapped_logits") ?? values.get("logits");
  if (!logits) throw new Error("Gemma 4 paginado não produziu logits.");
  return { values, logits, pastKeyValues: producedCache };
}

/**
 * A literal artifact can be storage-complete while its declared operation
 * policy remains only a measured candidate. Do not silently turn that state
 * into a replay claim just because the embedded program is executable.
 */
function assertExecutionFidelityAcknowledged(
  artifact: OpenGemma4CompositeLiteralArtifact,
  options: Gemma4PagedTextOptions,
): void {
  if (artifact.program.textProgram.fidelity.exactByConstruction || options.allowUnverifiedFidelity) return;
  throw new Error(
    "O programa literal Gemma 4 declara fidelidade numérica não verificada; passe allowUnverifiedFidelity: true somente para replay diagnóstico aproximado.",
  );
}

/** Greedy cached decode through the same source-independent text-only path. */
export async function generateGemma4PagedTextLiteralF32(
  artifact: OpenGemma4CompositeLiteralArtifact,
  request: Gemma4PagedTextGenerationRequest,
  options: Gemma4PagedTextOptions = {},
): Promise<Gemma4LiteralGenerationExecutionResult> {
  return executeGemma4LiteralGenerationProgram(artifact.program, artifact.generation, request, {
    prefill: (prefill) => executeGemma4PagedTextLiteralF32(artifact, prefill, options),
    incremental: (incremental) => executeGemma4PagedTextLiteralF32(artifact, incremental, options),
  });
}

function tensorInfo(artifact: OpenGemma4CompositeLiteralArtifact, reference: TensorRef): TensorInfo {
  const constant = artifact.constants.get(reference.name);
  if (!constant || reference.quantization || constant.storageDtype !== reference.storageDtype ||
    constant.logicalShape.length !== reference.shape.length || constant.logicalShape.some((value, index) => value !== reference.shape[index])) {
    throw new Error(`${reference.name}: referência textual não corresponde à constante densa literal.`);
  }
  return { name: constant.name, storageDtype: constant.storageDtype, storageShape: [...constant.storageShape], logicalShape: [...constant.logicalShape] };
}

function value(values: ReadonlyMap<string, DenseF32Tensor>, name: string): DenseF32Tensor {
  const found = values.get(name);
  if (!found) throw new Error(`Gemma 4 paginado não encontrou intermediário ${name}.`);
  return found;
}

function assertPagedF32Policy(operation: Operation): void {
  const policy = operation.dtypePolicy;
  const f32 = policy.computeDtype === "F32" && policy.accumulationDtype === "F32" && policy.outputDtype === "F32";
  const bf16 = policy.inputDtype === "BF16" && policy.computeDtype === "F32" && policy.accumulationDtype === "F32" && policy.outputDtype === "BF16";
  const bf16F64Reduction = (operation.op === "linear" || operation.op === "rms_norm") && policy.inputDtype === "BF16" && policy.computeDtype === "F32" && policy.accumulationDtype === "F64" && policy.outputDtype === "BF16";
  const reduction = policy.reduction;
  const tiled = reduction?.kind === "tiled-f32-lanes" || reduction?.kind === "tiled-fma-lanes";
  const laneProfile = operation.op === "linear" && policy.inputDtype === "BF16" && policy.computeDtype === "F32" && policy.accumulationDtype === "F32" && policy.outputDtype === "BF16" &&
    reduction && (reduction.kind === "interleaved-f32-lanes" || reduction.kind === "interleaved-fma-lanes" || reduction.kind === "tiled-f32-lanes" || reduction.kind === "tiled-fma-lanes") && Number.isSafeInteger(reduction.laneCount) && reduction.laneCount >= 2 &&
    (reduction.laneReductionOrder === "ascending" || reduction.laneReductionOrder === "descending" || reduction.laneReductionOrder === "balanced-pairwise") &&
    ((!tiled && reduction.inputLane === "index-modulo-lane-count") ||
      (tiled && Number.isSafeInteger(reduction.termsPerLane) && reduction.termsPerLane >= 2 && reduction.inputLane === "tile-contiguous-terms"));
  const orderedFma = operation.op === "linear" && policy.inputDtype === "BF16" && policy.computeDtype === "F32" && policy.accumulationDtype === "F32" && policy.outputDtype === "BF16" &&
    reduction?.kind === "ordered-fma" && reduction.indexOrder === "ascending";
  const blockedTerms = operation.op === "linear" && policy.inputDtype === "BF16" && policy.computeDtype === "F32" && policy.accumulationDtype === "F32" && policy.outputDtype === "BF16" &&
    reduction?.kind === "blocked-f32-terms" && Number.isSafeInteger(reduction.termsPerBlock) && reduction.termsPerBlock >= 2 && reduction.inputBlock === "contiguous-terms" && reduction.blockOrder === "ascending" &&
    (reduction.termOrder === "ascending" || reduction.termOrder === "descending") &&
    (reduction.productBoundary === "separately-rounded-f32" || reduction.productBoundary === "fused-fma");
  const blockedTiled = operation.op === "linear" && policy.inputDtype === "BF16" && policy.computeDtype === "F32" && policy.accumulationDtype === "F32" && policy.outputDtype === "BF16" &&
    reduction?.kind === "blocked-tiled-f32-lanes" && Number.isSafeInteger(reduction.laneCount) && reduction.laneCount >= 2 && Number.isSafeInteger(reduction.termsPerLane) && reduction.termsPerLane >= 2 &&
    reduction.inputBlock === "tile-contiguous-terms" && reduction.blockOrder === "ascending" &&
    (reduction.laneReductionOrder === "ascending" || reduction.laneReductionOrder === "descending" || reduction.laneReductionOrder === "balanced-pairwise") &&
    (reduction.productBoundary === "separately-rounded-f32" || reduction.productBoundary === "fused-fma");
  const armNeonBf16Dot = operation.op === "linear" && policy.inputDtype === "BF16" && policy.computeDtype === "F32" && policy.accumulationDtype === "F32" && policy.outputDtype === "BF16" &&
    reduction?.kind === "arm-neon-bf16-dot-fma" && (reduction.laneCount === 32 || reduction.laneCount === 64) && reduction.registerCount === 8 &&
    (reduction.lanesPerRegister === 4 || reduction.lanesPerRegister === 8) && reduction.laneCount === reduction.registerCount * reduction.lanesPerRegister &&
    reduction.inputLane === "index-modulo-vector-lane-count" && (reduction.horizontalFold === "ascending" || reduction.horizontalFold === "pairwise");
  const armNeonBf16Bfdot = operation.op === "linear" && policy.inputDtype === "BF16" && policy.computeDtype === "F32" && policy.accumulationDtype === "F32" && policy.outputDtype === "BF16" &&
    reduction?.kind === "arm-neon-bf16-bfdot-fma" && reduction.registerCount === 8 && reduction.activeRegisterCount === 4 &&
    reduction.lanesPerRegister === 4 && reduction.termsPerLane === 2 && reduction.termsPerInstruction === 8 &&
    reduction.inputLane === "contiguous-bf16-pairs" && (reduction.horizontalFold === "ascending" || reduction.horizontalFold === "pairwise");
  if (!f32 && !bf16 && !bf16F64Reduction && !laneProfile && !orderedFma && !blockedTerms && !blockedTiled && !armNeonBf16Dot && !armNeonBf16Bfdot) {
    throw new Error(`${operation.id}: executor Gemma 4 paginado requer política F32, BF16 explícita, ou redução linear/RMSNorm BF16 F64 declarada.`);
  }
}

function assertCompatibleCache(cache: ReferenceF32KeyValueCache, currentKey: DenseF32Tensor, currentValue: DenseF32Tensor, operationId: string): void {
  const entries = [cache.key, cache.value, currentKey, currentValue];
  if (entries.some((entry) => entry.shape.length !== 4) ||
    cache.key.shape[0] !== currentKey.shape[0] || cache.key.shape[1] !== currentKey.shape[1] || cache.key.shape[3] !== currentKey.shape[3] ||
    cache.value.shape[0] !== currentValue.shape[0] || cache.value.shape[1] !== currentValue.shape[1] || cache.value.shape[3] !== currentValue.shape[3] ||
    cache.key.shape.some((dimension, index) => dimension !== cache.value.shape[index])) {
    throw new Error(`${operationId}: cache KV incompatível com as projeções atuais.`);
  }
}

function sharedPastLength(operationId: string, cache: ReferenceF32KeyValueCache, query: DenseF32Tensor): number {
  if (cache.key.shape.length !== 4 || cache.value.shape.length !== 4 || query.shape.length !== 4 ||
    cache.key.shape.some((dimension, index) => dimension !== cache.value.shape[index]) || cache.key.shape[0] !== query.shape[0]) {
    throw new Error(`${operationId}: cache KV compartilhado incompatível.`);
  }
  const pastLength = cache.key.shape[2]! - query.shape[2]!;
  if (pastLength < 0) throw new Error(`${operationId}: cache KV compartilhado menor que a consulta atual.`);
  return pastLength;
}
