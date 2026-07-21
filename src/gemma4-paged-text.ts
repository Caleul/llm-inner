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
import { createPagedDenseF32Matrix, pagedEmbeddingF32, pagedLinearBatchF32, pagedLinearF32, readPagedDenseF32Vector, roundDenseF32ToBF16, type PagedLinearTileKernel } from "./paged-dense.js";
import type {
  DenseF32Tensor,
  Operation,
  ReferenceF32ExecutionResult,
  ReferenceF32KeyValueCache,
  TensorInfo,
  TensorRef,
} from "./types.js";
import type { LiteralTensorReader } from "./literal.js";

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
  /** Optional approximate native tile kernel; the scalar schedule remains the fidelity reference. */
  linearTileKernel?: PagedLinearTileKernel;
  /** Optional compiled binary constant-pool reader replacing base64 payload reads. */
  tensorReader?: Pick<LiteralTensorReader, "readTensorBytesRange">;
  /** Opt-in whole-MLP native subgraph; real removes the internal BF16 boundaries. */
  fusedMlpRounding?: "bf16" | "real";
  /** Final vocabulary projection compute path; BF16 is the allowed final rounding boundary. */
  finalHeadCompute?: "f32" | "native-bf16";
  /** Optional native QK/softmax/PV kernel; real removes its internal BF16 boundaries. */
  nativeAttentionRounding?: "bf16" | "real";
}

/**
 * Executes the two token-indexed assignments that precede multimodal scatter.
 * The returned map contains `hidden_states_0` and `ple_token_identity`; callers
 * may replace only `hidden_states_0` with the declared image/video/audio
 * scatter result before continuing through the projection prelude.
 */
export async function executeGemma4PagedTextInputEmbeddingsLiteralF32(
  artifact: OpenGemma4CompositeLiteralArtifact,
  inputIds: number[][],
  options: Gemma4PagedTextOptions = {},
): Promise<ReadonlyMap<string, DenseF32Tensor>> {
  assertExecutionFidelityAcknowledged(artifact, options);
  validateInputIds(inputIds);
  const operations = artifact.program.textProgram.prelude.slice(0, 2);
  if (operations[0]?.id !== "token_embedding" || operations[1]?.id !== "ple_token_identity") {
    throw new Error("Gemma 4 paginado requer embedding textual e identidade PLE como as duas primeiras atribuições.");
  }
  return (await executePagedOperations(artifact, operations, inputIds, undefined, new Map(), options)).values;
}

/**
 * Executes the post-scatter PLE projection assignments from a caller-supplied
 * `hidden_states_0` and the artifact-produced `ple_token_identity`.
 */
export async function executeGemma4PagedTextProjectionPreludeLiteralF32(
  artifact: OpenGemma4CompositeLiteralArtifact,
  inputIds: number[][],
  prepared: ReadonlyMap<string, DenseF32Tensor>,
  options: Gemma4PagedTextOptions = {},
): Promise<ReadonlyMap<string, DenseF32Tensor>> {
  assertExecutionFidelityAcknowledged(artifact, options);
  validateInputIds(inputIds);
  requirePreparedPreludeValue(prepared, "hidden_states_0", inputIds);
  requirePreparedPreludeValue(prepared, "ple_token_identity", inputIds);
  const values = new Map(prepared);
  await executePagedOperations(artifact, artifact.program.textProgram.prelude.slice(2), inputIds, undefined, values, options);
  requirePreparedPreludeValue(values, "ple_inputs", inputIds);
  return values;
}

/**
 * Executes the declared Gemma4Text graph directly from an indexed literal
 * artifact. Matrix bytes are decoded a bounded output-row page at a time;
 * vector constants are bounded separately. This entry point owns standalone
 * token text; the prepared-prelude entry point below is the only route by
 * which the composite executor can enter the same layers after modal scatter.
 */
export async function executeGemma4PagedTextLiteralF32(
  artifact: OpenGemma4CompositeLiteralArtifact,
  request: Gemma4PagedTextExecutionRequest,
  options: Gemma4PagedTextOptions = {},
): Promise<ReferenceF32ExecutionResult> {
  assertExecutionFidelityAcknowledged(artifact, options);
  const inputIds = request.inputIds;
  validateInputIds(inputIds);
  const sequence = inputIds[0]!.length;
  const positions = request.positionIds ?? inputIds.map((row) => row.map((_, index) => index));
  if (positions.length !== inputIds.length || positions.some((row) => row.length !== sequence)) {
    throw new Error("Gemma 4 paginado position_ids deve acompanhar input_ids.");
  }
  const values = new Map<string, DenseF32Tensor>();
  return requireCompletedTextExecution(await executePagedOperations(artifact, allTextOperations(artifact), inputIds, positions, values, options, request));
}

/** Executes text layers and logits from the exact composite prelude values. */
export async function executeGemma4PagedTextLiteralF32WithPreparedPrelude(
  artifact: OpenGemma4CompositeLiteralArtifact,
  request: Gemma4PagedTextExecutionRequest,
  prepared: ReadonlyMap<string, DenseF32Tensor>,
  options: Gemma4PagedTextOptions = {},
): Promise<ReferenceF32ExecutionResult> {
  assertExecutionFidelityAcknowledged(artifact, options);
  validateInputIds(request.inputIds);
  requirePreparedPreludeValue(prepared, "hidden_states_0", request.inputIds);
  requirePreparedPreludeValue(prepared, "ple_inputs", request.inputIds);
  const sequence = request.inputIds[0]!.length;
  const positions = request.positionIds ?? request.inputIds.map((row) => row.map((_, index) => index));
  if (positions.length !== request.inputIds.length || positions.some((row) => row.length !== sequence)) {
    throw new Error("Gemma 4 paginado position_ids deve acompanhar input_ids.");
  }
  return requireCompletedTextExecution(await executePagedOperations(
    artifact,
    [...artifact.program.textProgram.layers.flatMap((layer) => layer.operations), ...artifact.program.textProgram.epilogue],
    request.inputIds,
    positions,
    new Map(prepared),
    options,
    request,
  ));
}

interface PagedOperationsResult {
  values: Map<string, DenseF32Tensor>;
  logits?: DenseF32Tensor;
  pastKeyValues: ReadonlyMap<number, ReferenceF32KeyValueCache>;
}

async function executePagedOperations(
  artifact: OpenGemma4CompositeLiteralArtifact,
  operations: readonly Operation[],
  inputIds: number[][],
  positions: number[][] | undefined,
  values: Map<string, DenseF32Tensor>,
  options: Gemma4PagedTextOptions,
  request: Gemma4PagedTextExecutionRequest = { inputIds },
): Promise<PagedOperationsResult> {
  const maxReadBytes = options.maxReadBytes ?? 16 * 1024 * 1024;
  const tensorReader = options.tensorReader ?? artifact;
  const vectors = new Map<string, Promise<DenseF32Tensor>>();
  const matrix = (reference: TensorRef) => createPagedDenseF32Matrix(tensorInfo(artifact, reference), tensorReader, maxReadBytes);
  const vector = (reference: TensorRef): Promise<DenseF32Tensor> => {
    let result = vectors.get(reference.name);
    if (!result) {
      result = readPagedDenseF32Vector(tensorInfo(artifact, reference), tensorReader, maxReadBytes);
      vectors.set(reference.name, result);
    }
    return result;
  };
  const store = (operation: Operation, tensor: DenseF32Tensor): void => {
    values.set(operation.output, operation.dtypePolicy.outputDtype === "BF16" ? roundDenseF32ToBF16(tensor) : tensor);
  };
  const executeAndStoreAttention = async (operation: Extract<Operation, { op: "scaled_dot_product_attention" }>, query: DenseF32Tensor, key: DenseF32Tensor, valueTensor: DenseF32Tensor, mask: DenseF32Tensor | undefined, pastLength: number, maskDefinesTopology: boolean): Promise<void> => {
    const result = await attentionWithOptionalNative(query, key, valueTensor, operation, mask, pastLength, maskDefinesTopology, options.linearTileKernel, options.nativeAttentionRounding);
    if (result.native && options.nativeAttentionRounding === "real") values.set(operation.output, result.tensor); else store(operation, result.tensor);
  };
  const producedCache = new Map<number, ReferenceF32KeyValueCache>();
  for (let operationIndex = 0; operationIndex < operations.length; operationIndex += 1) {
    const operation = operations[operationIndex]!;
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
        if (options.fusedMlpRounding && options.linearTileKernel?.fusedGatedMlpStorageReference) {
          const fused = matchFusedGatedMlp(operations, operationIndex, operation);
          if (fused) {
            for (const fusedOperation of fused.operations) assertPagedF32Policy(fusedOperation);
            const input = value(values, operation.input), rows = input.values.length / operation.inFeatures;
            const outputValues = await options.linearTileKernel.fusedGatedMlpStorageReference(input.values, tensorInfo(artifact, operation.weight), tensorInfo(artifact, fused.up.weight), tensorInfo(artifact, fused.down.weight), rows, options.fusedMlpRounding);
            if (outputValues.length !== rows * fused.down.outFeatures || outputValues.some((entry) => !Number.isFinite(entry))) throw new Error(`${options.linearTileKernel.backend}: subgrafo MLP retornou saída inválida.`);
            const output = { shape: [...input.shape.slice(0, -1), fused.down.outFeatures], values: outputValues };
            if (options.fusedMlpRounding === "bf16") store(fused.down, output); else values.set(fused.down.output, output);
            operationIndex += 4;
            break;
          }
        }
        if (options.linearTileKernel?.multiplyStorageReferences) {
          const next = operations[operationIndex + 1];
          if (isFusibleSharedInputLinear(operation, next)) {
            assertPagedF32Policy(next);
            const outputs = await pagedLinearBatchF32(value(values, operation.input), [matrix(operation.weight), matrix(next.weight)], [
              { outputDtype: operation.dtypePolicy.outputDtype === "BF16" ? "BF16" : "F32", tileKernel: options.linearTileKernel },
              { outputDtype: next.dtypePolicy.outputDtype === "BF16" ? "BF16" : "F32", tileKernel: options.linearTileKernel },
            ]);
            store(operation, outputs[0]!); store(next, outputs[1]!); operationIndex += 1;
            break;
          }
        }
        store(operation, await pagedLinearF32(value(values, operation.input), matrix(operation.weight), {
          outputDtype: operation.dtypePolicy.outputDtype === "BF16" ? "BF16" : "F32",
          accumulationDtype: operation.dtypePolicy.accumulationDtype === "F64" ? "F64" : "F32",
          ...(operation.dtypePolicy.reduction ? { reduction: operation.dtypePolicy.reduction } : {}),
          ...(options.linearTileKernel ? { tileKernel: options.linearTileKernel } : {}),
          ...(operation.id === "lm_head" && options.finalHeadCompute === "native-bf16" ? { nativeBf16: true } : {}),
        }));
        break;
      case "reshape_heads":
        if (operation.layout !== "BHSD") throw new Error(`${operation.id}: executor Gemma 4 paginado requer layout BHSD.`);
        store(operation, reshapeHeadsF32(value(values, operation.input), operation.numHeads, operation.headDim));
        break;
      case "rotary_embedding":
        if (!positions) throw new Error(`${operation.id}: position_ids ausentes na fase textual preparada.`);
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
          await executeAndStoreAttention(operation, query, shared.key, shared.value, topologyMask ?? request.attentionMask, sharedPastLength(operation.id, shared, query), topologyMask !== undefined);
          break;
        }
        const currentKey = value(values, operation.key);
        const currentValue = value(values, operation.value);
        const previous = request.pastKeyValues?.get(operation.layer);
        if (request.pastKeyValues && !previous) throw new Error(`${operation.id}: pastKeyValues não contém a camada ${operation.layer}.`);
        if (previous) assertCompatibleCache(previous, currentKey, currentValue, operation.id);
        const key = previous ? concatSequenceF32(previous.key, currentKey) : currentKey;
        const valueTensor = previous ? concatSequenceF32(previous.value, currentValue) : currentValue;
        await executeAndStoreAttention(operation, value(values, operation.query), key, valueTensor, topologyMask ?? request.attentionMask, previous?.key.shape[2] ?? 0, topologyMask !== undefined);
        producedCache.set(operation.layer, { key, value: valueTensor });
        break;
      }
      case "activation":
        store(operation, activationF32(value(values, operation.input), operation.function, operation.approximation, operation.tanhImplementation));
        break;
      case "elementwise":
        store(operation, elementwiseF32(operation.inputs.map((name) => value(values, name)), operation.kind, operation.scalar, operation.tanhImplementation, operation.tanhSoftcapCasts));
        break;
      default: {
        const unsupported: never = operation;
        throw new Error(`Operação Gemma 4 paginada não suportada: ${JSON.stringify(unsupported)}.`);
      }
    }
  }
  const logits = values.get("softcapped_logits") ?? values.get("logits");
  return { values, ...(logits ? { logits } : {}), pastKeyValues: producedCache };
}

function isFusibleSharedInputLinear(left: Extract<Operation, { op: "linear" }>, right: Operation | undefined): right is Extract<Operation, { op: "linear" }> {
  return right?.op === "linear" && right.input === left.input && right.transposeWeight && !right.bias;
}

type LinearOperation = Extract<Operation, { op: "linear" }>;
function matchFusedGatedMlp(operations: readonly Operation[], index: number, gate: LinearOperation): { up: LinearOperation; down: LinearOperation; operations: readonly Operation[] } | undefined {
  const up = operations[index + 1], activation = operations[index + 2], multiply = operations[index + 3], down = operations[index + 4];
  if (up?.op !== "linear" || activation?.op !== "activation" || multiply?.op !== "elementwise" || down?.op !== "linear") return undefined;
  if (up.input !== gate.input || !up.transposeWeight || up.bias || activation.input !== gate.output || activation.function !== "gelu" || activation.approximation !== "tanh" || multiply.kind !== "multiply" || multiply.inputs.length !== 2 || multiply.inputs[0] !== activation.output || multiply.inputs[1] !== up.output || down.input !== multiply.output || !down.transposeWeight || down.bias) return undefined;
  if ([gate, up, activation, multiply, down].some((operation) => operation.dtypePolicy.outputDtype !== "BF16")) return undefined;
  return { up, down, operations: [up, activation, multiply, down] };
}

async function attentionWithOptionalNative(
  query: DenseF32Tensor,
  key: DenseF32Tensor,
  valueTensor: DenseF32Tensor,
  operation: Extract<Operation, { op: "scaled_dot_product_attention" }>,
  mask: DenseF32Tensor | undefined,
  pastLength: number,
  maskDefinesTopology: boolean,
  kernel: PagedLinearTileKernel | undefined,
  rounding: "bf16" | "real" | undefined,
): Promise<{ tensor: DenseF32Tensor; native: boolean }> {
  if (!rounding || !kernel?.attention) return { tensor: attentionF32(query, key, valueTensor, operation, mask, pastLength, maskDefinesTopology), native: false };
  if (query.shape.length !== 4 || key.shape.length !== 4 || valueTensor.shape.length !== 4) throw new Error(`${operation.id}: attention nativa requer tensores BHSD.`);
  const [batch, queryHeads, querySequence, headDim] = query.shape as [number, number, number, number];
  const [keyBatch, keyValueHeads, keySequence, keyDim] = key.shape as [number, number, number, number];
  if (batch !== keyBatch || queryHeads !== operation.numAttentionHeads || keyValueHeads !== operation.numKeyValueHeads || headDim !== operation.headDim || keyDim !== headDim || valueTensor.shape.length !== 4 || valueTensor.shape.some((dimension, index) => dimension !== key.shape[index])) throw new Error(`${operation.id}: topologia da attention nativa é incompatível.`);
  if (mask) {
    const [maskBatch, maskHeads, maskQuery, maskKey] = mask.shape;
    if (mask.shape.length !== 4 || maskBatch !== batch || (maskHeads !== 1 && maskHeads !== queryHeads) || maskQuery !== querySequence || maskKey !== keySequence || mask.values.some((entry) => Number.isNaN(entry) || entry === Infinity)) throw new Error(`${operation.id}: máscara da attention nativa é incompatível.`);
  }
  const nativeMask = maskDefinesTopology && mask ? mask : materializeAttentionTopologyMask(mask, operation, batch, querySequence, keySequence, pastLength);
  const values = await kernel.attention({ query: query.values, key: key.values, value: valueTensor.values, mask: nativeMask.values, batch, queryHeads, keyValueHeads, querySequence, keySequence, headDim, maskHeads: nativeMask.shape[1]!, scale: operation.scale, rounding });
  if (values.length !== batch * querySequence * queryHeads * headDim || values.some((entry) => !Number.isFinite(entry))) throw new Error(`${operation.id}: attention nativa retornou saída inválida.`);
  return { tensor: { shape: [batch, querySequence, queryHeads * headDim], values }, native: true };
}

function materializeAttentionTopologyMask(mask: DenseF32Tensor | undefined, operation: Extract<Operation, { op: "scaled_dot_product_attention" }>, batch: number, querySequence: number, keySequence: number, pastLength: number): DenseF32Tensor {
  const maskHeads = mask?.shape[1] ?? 1, values = new Float32Array(batch * maskHeads * querySequence * keySequence);
  for (let b = 0; b < batch; b += 1) for (let h = 0; h < maskHeads; h += 1) for (let q = 0; q < querySequence; q += 1) for (let k = 0; k < keySequence; k += 1) {
    const absoluteQuery = pastLength + q;
    const firstKey = operation.slidingWindow === undefined ? 0 : Math.max(0, absoluteQuery - operation.slidingWindow + 1);
    const lastKey = operation.causal ? Math.min(absoluteQuery, keySequence - 1) : keySequence - 1;
    const topology = k >= firstKey && k <= lastKey ? 0 : -Infinity;
    const declared = mask ? mask.values[((b * maskHeads + h) * querySequence + q) * keySequence + k]! : 0;
    values[((b * maskHeads + h) * querySequence + q) * keySequence + k] = Math.fround(topology + declared);
  }
  return { shape: [batch, maskHeads, querySequence, keySequence], values };
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

function allTextOperations(artifact: OpenGemma4CompositeLiteralArtifact): Operation[] {
  return [...artifact.program.textProgram.prelude, ...artifact.program.textProgram.layers.flatMap((layer) => layer.operations), ...artifact.program.textProgram.epilogue];
}

function validateInputIds(inputIds: number[][]): void {
  if (inputIds.length === 0 || inputIds.some((row) => row.length === 0 || row.length !== inputIds[0]!.length || row.some((token) => !Number.isSafeInteger(token) || token < 0))) {
    throw new Error("Gemma 4 paginado requer input_ids não vazio, retangular e inteiro não negativo.");
  }
}

function requirePreparedPreludeValue(values: ReadonlyMap<string, DenseF32Tensor>, name: string, inputIds: number[][]): DenseF32Tensor {
  const tensor = value(values, name);
  if (tensor.shape[0] !== inputIds.length || tensor.shape[1] !== inputIds[0]!.length) {
    throw new Error(`${name}: intermediário preparado não acompanha input_ids em batch/sequence.`);
  }
  return tensor;
}

function requireCompletedTextExecution(result: PagedOperationsResult): ReferenceF32ExecutionResult {
  if (!result.logits) throw new Error("Gemma 4 paginado não produziu logits.");
  return { values: result.values, logits: result.logits, pastKeyValues: result.pastKeyValues };
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
