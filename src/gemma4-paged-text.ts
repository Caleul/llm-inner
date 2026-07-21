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
  /** Independent page size for the terminal vocabulary projection. */
  finalHeadMaxReadBytes?: number;
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
  fusedMlpRounding?: "bf16" | "real" | "native-bf16";
  /** Pre-FFN norm, native BF16 gated MLP, post-FFN norm and residual in one worker dispatch. */
  fusedFfnRounding?: "native-bf16";
  /** Complete decoder layer from input norm through PLE scalar in one native dispatch. */
  fusedDecoderLayerRounding?: "native-bf16";
  /** Complete ordered decoder stack in one native dispatch. */
  fusedDecoderStackRounding?: "native-bf16";
  /** Final vocabulary projection compute path; BF16 is the allowed final rounding boundary. */
  finalHeadCompute?: "f32" | "native-bf16" | "native-bf16-stream" | "native-bf16-whole";
  /** Optional native QK/softmax/PV kernel; real removes its internal BF16 boundaries. */
  nativeAttentionRounding?: "bf16" | "real";
  /** Whole Q/K/V -> norm/RoPE -> attention -> O subgraph with explicit boundary policy. */
  fusedAttentionRounding?: "bf16" | "real" | "native-bf16";
  /** Whole per-layer-input gate -> projection -> norm -> residual subgraph. */
  fusedPleRounding?: "bf16" | "real";
  /** Experimental projection -> norm -> token combine fusion before decoder layers. */
  fusedPlePreludeRounding?: "bf16" | "real";
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
  const matrix = (reference: TensorRef, readBytes = maxReadBytes) => createPagedDenseF32Matrix(tensorInfo(artifact, reference), tensorReader, readBytes);
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
        if (options.fusedDecoderStackRounding && options.linearTileKernel?.fusedDecoderStackStorageReferences) {
          const stack = matchFusedDecoderStack(operations, operationIndex, operation);
          if (stack) {
            if (!positions) throw new Error(`${operation.id}: pilha decoder fundida requer posições declaradas.`);
            for (const stackOperation of stack.operations) assertPagedF32Policy(stackOperation);
            const input = value(values, operation.input), perLayerInputs = value(values, stack.perLayerInput);
            if (input.shape.length !== 3 || perLayerInputs.shape.length !== 4 || input.shape[0] !== perLayerInputs.shape[0] || input.shape[1] !== perLayerInputs.shape[1] || perLayerInputs.shape[2] !== stack.layers.length || perLayerInputs.shape[3] !== stack.perLayerWidth) throw new Error(`${operation.id}: pilha decoder requer tensores [B,S,H] e [B,S,L,P].`);
            const [batch, querySequence, hiddenSize] = input.shape as [number, number, number];
            if (positions.length !== batch || positions.some((row) => row.length !== querySequence)) throw new Error(`${operation.id}: posições incompatíveis com a pilha decoder.`);
            const positionValues = new Int32Array(batch * querySequence);
            for (let b = 0; b < batch; b += 1) for (let s = 0; s < querySequence; s += 1) {
              const position = positions[b]![s]!;
              if (!Number.isSafeInteger(position) || position < -2_147_483_648 || position > 2_147_483_647) throw new Error(`${operation.id}: posição ${position} excede o protocolo Int32 da pilha.`);
              positionValues[b * querySequence + s] = position;
            }
            const layers = stack.layers.map((fused, layerIndex) => {
              const attention = fused.attention.attention;
              if (attention.layer !== layerIndex) throw new Error(`${attention.id}: índice incompatível com a pilha decoder.`);
              const topologyMask = request.attentionMasksByLayer?.get(layerIndex);
              let sourceKey: DenseF32Tensor | undefined, sourceValue: DenseF32Tensor | undefined, sourceSequence = 0, sharedProducerLayer: number | undefined;
              if (attention.kvSharing) {
                sharedProducerLayer = attention.kvSharing.producerLayer;
                if (sharedProducerLayer === undefined || sharedProducerLayer >= layerIndex) throw new Error(`${attention.id}: produtor KV compartilhado inválido.`);
                const previous = request.pastKeyValues?.get(sharedProducerLayer);
                if (request.pastKeyValues && !previous) throw new Error(`${attention.id}: pastKeyValues não contém o produtor ${sharedProducerLayer}.`);
                if (previous) {
                  assertFusedSourceCache(previous.key, previous.value, batch, attention.numKeyValueHeads, attention.headDim, attention.id);
                  sourceSequence = previous.key.shape[2]!;
                }
                sourceSequence += querySequence;
              } else {
                const previous = request.pastKeyValues?.get(layerIndex);
                if (request.pastKeyValues && !previous) throw new Error(`${attention.id}: pastKeyValues não contém a camada ${layerIndex}.`);
                if (previous) {
                  assertFusedSourceCache(previous.key, previous.value, batch, attention.numKeyValueHeads, attention.headDim, attention.id);
                  sourceKey = previous.key; sourceValue = previous.value; sourceSequence = previous.key.shape[2]!;
                }
              }
              const producesKeyValue = sharedProducerLayer === undefined;
              const keySequence = sourceSequence + (producesKeyValue ? querySequence : 0), pastLength = producesKeyValue ? sourceSequence : sourceSequence - querySequence;
              const declaredMask = topologyMask ?? request.attentionMask;
              const nativeMask = topologyMask ? topologyMask : materializeAttentionTopologyMask(declaredMask, attention, batch, querySequence, keySequence, pastLength);
              assertNativeAttentionMask(nativeMask, batch, attention.numAttentionHeads, querySequence, keySequence, attention.id);
              const half = fused.attention.queryRope.rotaryDim / 2;
              const proportionalPairs = fused.attention.queryRope.ropeType === "proportional" ? Math.floor(Number(fused.attention.queryRope.scaling?.partial_rotary_factor) * attention.headDim / 2) : half;
              const proportionalFactor = fused.attention.queryRope.ropeType === "proportional" ? Number(fused.attention.queryRope.scaling?.factor ?? 1) : 1;
              return {
                layerIndex, ...(sharedProducerLayer === undefined ? {} : { sharedProducerLayer }), mask: nativeMask.values,
                sourceKey: sourceKey?.values ?? new Float32Array(), sourceValue: sourceValue?.values ?? new Float32Array(),
                inputNormWeight: tensorInfo(artifact, fused.inputNorm.weight!), queryWeight: tensorInfo(artifact, fused.query.weight), queryNorm: tensorInfo(artifact, fused.attention.queryNorm.weight!), outputWeight: tensorInfo(artifact, fused.attention.output.weight),
                ...(fused.attention.key ? { keyWeight: tensorInfo(artifact, fused.attention.key.weight), keyNorm: tensorInfo(artifact, fused.attention.keyNorm!.weight!) } : {}),
                ...(fused.attention.value ? { valueWeight: tensorInfo(artifact, fused.attention.value.weight) } : {}),
                postAttentionNormWeight: tensorInfo(artifact, fused.postAttentionNorm.weight!), preFfnNormWeight: tensorInfo(artifact, fused.ffn.preNorm.weight!), gateWeight: tensorInfo(artifact, fused.ffn.gate.weight), upWeight: tensorInfo(artifact, fused.ffn.up.weight), downWeight: tensorInfo(artifact, fused.ffn.down.weight), postFfnNormWeight: tensorInfo(artifact, fused.ffn.postNorm.weight!),
                pleGateWeight: tensorInfo(artifact, fused.ple.gate.weight), pleProjectionWeight: tensorInfo(artifact, fused.ple.projection.weight), pleNormWeight: tensorInfo(artifact, fused.ple.norm.weight!), layerScalar: tensorInfo(artifact, fused.ple.scalar.scalar),
                batch, querySequence, sourceSequence, hiddenSize, queryHeads: attention.numAttentionHeads, keyValueHeads: attention.numKeyValueHeads, headDim: attention.headDim, maskHeads: nativeMask.shape[1]!, producesKeyValue, valueFromKey: fused.attention.valueFromKey,
                epsilon: fused.attention.queryNorm.epsilon, scale: attention.scale, ropeType: fused.attention.queryRope.ropeType, theta: fused.attention.queryRope.theta, rotaryDim: fused.attention.queryRope.rotaryDim, proportionalPairs, proportionalFactor,
                intermediateSize: fused.ffn.gate.outFeatures, perLayerWidth: stack.perLayerWidth, inputNormEpsilon: fused.inputNorm.epsilon, postAttentionNormEpsilon: fused.postAttentionNorm.epsilon, preFfnNormEpsilon: fused.ffn.preNorm.epsilon, postFfnNormEpsilon: fused.ffn.postNorm.epsilon, pleNormEpsilon: fused.ple.norm.epsilon,
              };
            });
            const result = await options.linearTileKernel.fusedDecoderStackStorageReferences({ input: input.values, perLayerInputs: perLayerInputs.values, positions: positionValues, batch, querySequence, hiddenSize, numLayers: stack.layers.length, perLayerWidth: stack.perLayerWidth, layers, rounding: options.fusedDecoderStackRounding });
            if (result.hidden.length !== input.values.length || result.hidden.some((entry) => !Number.isFinite(entry))) throw new Error(`${operation.id}: pilha decoder retornou vetor inválido.`);
            store(stack.layers.at(-1)!.ple.scalar, { shape: [...input.shape], values: result.hidden });
            const expectedProducers = layers.filter((layer) => layer.producesKeyValue);
            if (result.caches.length !== expectedProducers.length) throw new Error(`${operation.id}: pilha decoder retornou quantidade de caches inválida.`);
            for (let index = 0; index < expectedProducers.length; index += 1) {
              const expected = expectedProducers[index]!, actual = result.caches[index]!, keySequence = expected.sourceSequence + querySequence;
              const cacheElements = batch * expected.keyValueHeads * keySequence * expected.headDim;
              if (actual.layerIndex !== expected.layerIndex || actual.key.length !== cacheElements || actual.value.length !== cacheElements || actual.key.some((entry) => !Number.isFinite(entry)) || actual.value.some((entry) => !Number.isFinite(entry))) throw new Error(`${operation.id}: pilha decoder retornou cache ${expected.layerIndex} inválido.`);
              producedCache.set(expected.layerIndex, { key: { shape: [batch, expected.keyValueHeads, keySequence, expected.headDim], values: actual.key }, value: { shape: [batch, expected.keyValueHeads, keySequence, expected.headDim], values: actual.value } });
            }
            operationIndex += stack.operations.length - 1;
            break;
          }
        }
        if (options.fusedDecoderLayerRounding && options.linearTileKernel?.fusedDecoderLayerStorageReferences) {
          const fused = matchFusedDecoderLayer(operations, operationIndex, operation);
          if (fused) {
            const attention = fused.attention.attention;
            if (!positions || attention.layer === undefined) throw new Error(`${operation.id}: decoder layer fundida requer posições e camada declaradas.`);
            for (const fusedOperation of fused.operations) assertPagedF32Policy(fusedOperation);
            const input = value(values, operation.input), perLayerInput = selectPerLayerF32(value(values, fused.pleSelect.input), fused.pleSelect);
            if (input.shape.length !== 3 || perLayerInput.shape.length !== 3 || input.shape[0] !== perLayerInput.shape[0] || input.shape[1] !== perLayerInput.shape[1]) throw new Error(`${operation.id}: decoder layer fundida requer tensores [B,S,H] e [B,S,P].`);
            const [batch, querySequence, hiddenSize] = input.shape as [number, number, number];
            if (hiddenSize !== fused.query.inFeatures || positions.length !== batch || positions.some((row) => row.length !== querySequence)) throw new Error(`${operation.id}: shapes incompatíveis com a decoder layer fundida.`);
            const topologyMask = request.attentionMasksByLayer?.get(attention.layer);
            let sourceKey: DenseF32Tensor | undefined, sourceValue: DenseF32Tensor | undefined, sourceSequence = 0;
            if (attention.kvSharing) {
              const producer = attention.kvSharing.producerLayer;
              if (producer === undefined) throw new Error(`${attention.id}: KV compartilhado sem produtor declarado.`);
              const shared = producedCache.get(producer);
              if (!shared) throw new Error(`${attention.id}: KV compartilhado não encontrou cache do produtor ${producer}.`);
              assertFusedSourceCache(shared.key, shared.value, batch, attention.numKeyValueHeads, attention.headDim, attention.id);
              sourceKey = shared.key; sourceValue = shared.value; sourceSequence = shared.key.shape[2]!;
            } else {
              const previous = request.pastKeyValues?.get(attention.layer);
              if (request.pastKeyValues && !previous) throw new Error(`${attention.id}: pastKeyValues não contém a camada ${attention.layer}.`);
              if (previous) {
                assertFusedSourceCache(previous.key, previous.value, batch, attention.numKeyValueHeads, attention.headDim, attention.id);
                sourceKey = previous.key; sourceValue = previous.value; sourceSequence = previous.key.shape[2]!;
              }
            }
            const producesKeyValue = !attention.kvSharing;
            const keySequence = sourceSequence + (producesKeyValue ? querySequence : 0), pastLength = producesKeyValue ? sourceSequence : sourceSequence - querySequence;
            if (pastLength < 0) throw new Error(`${attention.id}: cache compartilhado é menor que a consulta atual.`);
            const declaredMask = topologyMask ?? request.attentionMask;
            const nativeMask = topologyMask ? topologyMask : materializeAttentionTopologyMask(declaredMask, attention, batch, querySequence, keySequence, pastLength);
            assertNativeAttentionMask(nativeMask, batch, attention.numAttentionHeads, querySequence, keySequence, attention.id);
            const positionValues = new Int32Array(batch * querySequence);
            for (let b = 0; b < batch; b += 1) for (let s = 0; s < querySequence; s += 1) {
              const position = positions[b]![s]!;
              if (!Number.isSafeInteger(position) || position < -2_147_483_648 || position > 2_147_483_647) throw new Error(`${operation.id}: posição ${position} excede o protocolo Int32 do subgrafo.`);
              positionValues[b * querySequence + s] = position;
            }
            const half = fused.attention.queryRope.rotaryDim / 2;
            const proportionalPairs = fused.attention.queryRope.ropeType === "proportional" ? Math.floor(Number(fused.attention.queryRope.scaling?.partial_rotary_factor) * attention.headDim / 2) : half;
            const proportionalFactor = fused.attention.queryRope.ropeType === "proportional" ? Number(fused.attention.queryRope.scaling?.factor ?? 1) : 1;
            const result = await options.linearTileKernel.fusedDecoderLayerStorageReferences({
              input: input.values, perLayerInput: perLayerInput.values, positions: positionValues, mask: nativeMask.values,
              sourceKey: sourceKey?.values ?? new Float32Array(), sourceValue: sourceValue?.values ?? new Float32Array(),
              inputNormWeight: tensorInfo(artifact, operation.weight!), queryWeight: tensorInfo(artifact, fused.query.weight), queryNorm: tensorInfo(artifact, fused.attention.queryNorm.weight!), outputWeight: tensorInfo(artifact, fused.attention.output.weight),
              ...(fused.attention.key ? { keyWeight: tensorInfo(artifact, fused.attention.key.weight), keyNorm: tensorInfo(artifact, fused.attention.keyNorm!.weight!) } : {}),
              ...(fused.attention.value ? { valueWeight: tensorInfo(artifact, fused.attention.value.weight) } : {}),
              postAttentionNormWeight: tensorInfo(artifact, fused.postAttentionNorm.weight!), preFfnNormWeight: tensorInfo(artifact, fused.ffn.preNorm.weight!), gateWeight: tensorInfo(artifact, fused.ffn.gate.weight), upWeight: tensorInfo(artifact, fused.ffn.up.weight), downWeight: tensorInfo(artifact, fused.ffn.down.weight), postFfnNormWeight: tensorInfo(artifact, fused.ffn.postNorm.weight!),
              pleGateWeight: tensorInfo(artifact, fused.ple.gate.weight), pleProjectionWeight: tensorInfo(artifact, fused.ple.projection.weight), pleNormWeight: tensorInfo(artifact, fused.ple.norm.weight!), layerScalar: tensorInfo(artifact, fused.ple.scalar.scalar),
              batch, querySequence, sourceSequence, hiddenSize, queryHeads: attention.numAttentionHeads, keyValueHeads: attention.numKeyValueHeads, headDim: attention.headDim, maskHeads: nativeMask.shape[1]!, producesKeyValue, valueFromKey: fused.attention.valueFromKey,
              epsilon: fused.attention.queryNorm.epsilon, scale: attention.scale, ropeType: fused.attention.queryRope.ropeType, theta: fused.attention.queryRope.theta, rotaryDim: fused.attention.queryRope.rotaryDim, proportionalPairs, proportionalFactor,
              intermediateSize: fused.ffn.gate.outFeatures, perLayerWidth: fused.pleSelect.layerWidth, inputNormEpsilon: operation.epsilon, postAttentionNormEpsilon: fused.postAttentionNorm.epsilon, preFfnNormEpsilon: fused.ffn.preNorm.epsilon, postFfnNormEpsilon: fused.ffn.postNorm.epsilon, pleNormEpsilon: fused.ple.norm.epsilon, rounding: options.fusedDecoderLayerRounding,
            });
            if (result.hidden.length !== input.values.length || result.hidden.some((entry) => !Number.isFinite(entry))) throw new Error(`${operation.id}: decoder layer fundida retornou vetor inválido.`);
            store(fused.ple.scalar, { shape: [...input.shape], values: result.hidden });
            if (producesKeyValue) {
              const cacheElements = batch * attention.numKeyValueHeads * keySequence * attention.headDim;
              if (!result.key || !result.value || result.key.length !== cacheElements || result.value.length !== cacheElements || result.key.some((entry) => !Number.isFinite(entry)) || result.value.some((entry) => !Number.isFinite(entry))) throw new Error(`${operation.id}: decoder layer fundida retornou cache KV inválido.`);
              producedCache.set(attention.layer, { key: { shape: [batch, attention.numKeyValueHeads, keySequence, attention.headDim], values: result.key }, value: { shape: [batch, attention.numKeyValueHeads, keySequence, attention.headDim], values: result.value } });
            }
            operationIndex += fused.operations.length - 1;
            break;
          }
        }
        if (options.fusedFfnRounding && options.linearTileKernel?.fusedFfnStorageReferences) {
          const fused = matchFusedFfn(operations, operationIndex, operation);
          if (fused) {
            for (const fusedOperation of fused.operations) assertPagedF32Policy(fusedOperation);
            const input = value(values, operation.input);
            if (input.shape.length !== 3 || input.shape[2] !== fused.gate.inFeatures) throw new Error(`${operation.id}: subgrafo FFN requer entrada [B,S,H].`);
            const rows = input.shape[0]! * input.shape[1]!;
            const outputValues = await options.linearTileKernel.fusedFfnStorageReferences({
              input: input.values,
              preNormWeight: tensorInfo(artifact, operation.weight!),
              gateWeight: tensorInfo(artifact, fused.gate.weight),
              upWeight: tensorInfo(artifact, fused.up.weight),
              downWeight: tensorInfo(artifact, fused.down.weight),
              postNormWeight: tensorInfo(artifact, fused.postNorm.weight!),
              rows,
              hiddenSize: fused.gate.inFeatures,
              intermediateSize: fused.gate.outFeatures,
              preNormEpsilon: operation.epsilon,
              postNormEpsilon: fused.postNorm.epsilon,
              rounding: options.fusedFfnRounding,
            });
            if (outputValues.length !== input.values.length || outputValues.some((entry) => !Number.isFinite(entry))) throw new Error(`${options.linearTileKernel.backend}: subgrafo FFN retornou saída inválida.`);
            store(fused.residual, { shape: [...input.shape], values: outputValues });
            operationIndex += fused.operations.length - 1;
            break;
          }
        }
        store(operation, rmsNormF32(value(values, operation.input), operation.weight ? await vector(operation.weight) : undefined, operation));
        break;
      case "reshape_per_layer":
        store(operation, reshapePerLayerF32(value(values, operation.input), operation.numLayers, operation.layerWidth));
        break;
      case "select_per_layer":
        if (options.fusedPleRounding && options.linearTileKernel?.fusedPleStorageReferences) {
          const fused = matchFusedPle(operations, operationIndex, operation, options.fusedPleRounding);
          if (fused) {
            for (const fusedOperation of fused.operations) assertPagedF32Policy(fusedOperation);
            const input = value(values, fused.gate.input), perLayerInput = selectPerLayerF32(value(values, operation.input), operation);
            if (input.shape.length !== 3 || perLayerInput.shape.length !== 3 || input.shape[0] !== perLayerInput.shape[0] || input.shape[1] !== perLayerInput.shape[1]) throw new Error(`${operation.id}: subgrafo PLE requer tensores [B,S,H] e [B,S,P].`);
            const rows = input.shape[0]! * input.shape[1]!, hiddenSize = input.shape[2]!, perLayerWidth = perLayerInput.shape[2]!;
            const outputValues = await options.linearTileKernel.fusedPleStorageReferences({ input: input.values, perLayerInput: perLayerInput.values, gateWeight: tensorInfo(artifact, fused.gate.weight), projectionWeight: tensorInfo(artifact, fused.projection.weight), normWeight: tensorInfo(artifact, fused.norm.weight!), layerScalar: tensorInfo(artifact, fused.scalar.scalar), rows, hiddenSize, perLayerWidth, epsilon: fused.norm.epsilon, rounding: options.fusedPleRounding });
            if (outputValues.length !== input.values.length || outputValues.some((entry) => !Number.isFinite(entry))) throw new Error(`${options.linearTileKernel.backend}: subgrafo PLE retornou saída inválida.`);
            const output = { shape: [...input.shape], values: outputValues };
            if (options.fusedPleRounding === "bf16") store(fused.scalar, output); else values.set(fused.scalar.output, output);
            operationIndex += fused.operations.length - 1;
            break;
          }
        }
        store(operation, selectPerLayerF32(value(values, operation.input), operation));
        break;
      case "tensor_scale":
        store(operation, tensorScaleF32(value(values, operation.input), await vector(operation.scalar), operation.id));
        break;
      case "linear":
        if (!operation.transposeWeight || operation.bias) throw new Error(`${operation.id}: executor Gemma 4 paginado requer linear [out,in] sem bias.`);
        if (options.fusedPlePreludeRounding && options.linearTileKernel?.fusedPlePreludeStorageReference) {
          const fused = matchFusedPlePrelude(operations, operationIndex, operation, options.fusedPlePreludeRounding);
          if (fused) {
            for (const fusedOperation of fused.operations) assertPagedF32Policy(fusedOperation);
            const input = value(values, operation.input), tokenIdentity = value(values, fused.tokenIdentityInput);
            if (input.shape.length !== 3 || tokenIdentity.shape.length !== 4 || input.shape[0] !== tokenIdentity.shape[0] || input.shape[1] !== tokenIdentity.shape[1]) throw new Error(`${operation.id}: prelude PLE requer entrada [B,S,H] e identidade [B,S,L,P].`);
            const rows = input.shape[0]! * input.shape[1]!, hiddenSize = input.shape[2]!;
            const outputValues = await options.linearTileKernel.fusedPlePreludeStorageReference({ input: input.values, tokenIdentity: tokenIdentity.values, projectionWeight: tensorInfo(artifact, operation.weight), normWeight: tensorInfo(artifact, fused.norm.weight!), rows, hiddenSize, numLayers: fused.reshape.numLayers, perLayerWidth: fused.reshape.layerWidth, contextScale: fused.contextScale.scalar!, combineScale: fused.combineScale.scalar!, epsilon: fused.norm.epsilon, maxReadBytes, rounding: options.fusedPlePreludeRounding });
            if (outputValues.length !== tokenIdentity.values.length || outputValues.some((entry) => !Number.isFinite(entry))) throw new Error(`${options.linearTileKernel.backend}: prelude PLE fundido retornou saída inválida.`);
            const output = { shape: [...tokenIdentity.shape], values: outputValues };
            if (options.fusedPlePreludeRounding === "bf16") store(fused.combineScale, output); else values.set(fused.combineScale.output, output);
            operationIndex += fused.operations.length - 1;
            break;
          }
        }
        if (options.fusedAttentionRounding && options.linearTileKernel?.fusedAttentionStorageReferences) {
          const fused = matchFusedAttention(operations, operationIndex, operation);
          if (fused) {
            if (!positions || fused.attention.layer === undefined) throw new Error(`${operation.id}: subgrafo de attention requer posições e camada declaradas.`);
            for (const fusedOperation of fused.operations) assertPagedF32Policy(fusedOperation);
            const input = value(values, operation.input);
            if (input.shape.length !== 3) throw new Error(`${operation.id}: subgrafo de attention requer entrada [B,S,H].`);
            const [batch, querySequence, hiddenSize] = input.shape as [number, number, number];
            if (positions.length !== batch || positions.some((row) => row.length !== querySequence)) throw new Error(`${operation.id}: posições incompatíveis com o subgrafo de attention.`);
            const topologyMask = request.attentionMasksByLayer?.get(fused.attention.layer);
            let sourceKey: DenseF32Tensor | undefined, sourceValue: DenseF32Tensor | undefined, sourceSequence = 0;
            if (fused.attention.kvSharing) {
              const producer = fused.attention.kvSharing.producerLayer;
              if (producer === undefined) throw new Error(`${fused.attention.id}: KV compartilhado sem produtor declarado.`);
              const shared = producedCache.get(producer);
              if (!shared) throw new Error(`${fused.attention.id}: KV compartilhado não encontrou cache do produtor ${producer}.`);
              assertFusedSourceCache(shared.key, shared.value, batch, fused.attention.numKeyValueHeads, fused.attention.headDim, fused.attention.id);
              sourceKey = shared.key; sourceValue = shared.value; sourceSequence = shared.key.shape[2]!;
            } else {
              const previous = request.pastKeyValues?.get(fused.attention.layer);
              if (request.pastKeyValues && !previous) throw new Error(`${fused.attention.id}: pastKeyValues não contém a camada ${fused.attention.layer}.`);
              if (previous) {
                assertFusedSourceCache(previous.key, previous.value, batch, fused.attention.numKeyValueHeads, fused.attention.headDim, fused.attention.id);
                sourceKey = previous.key; sourceValue = previous.value; sourceSequence = previous.key.shape[2]!;
              }
            }
            const producesKeyValue = !fused.attention.kvSharing;
            const keySequence = sourceSequence + (producesKeyValue ? querySequence : 0);
            const pastLength = producesKeyValue ? sourceSequence : sourceSequence - querySequence;
            if (pastLength < 0) throw new Error(`${fused.attention.id}: cache compartilhado é menor que a consulta atual.`);
            const declaredMask = topologyMask ?? request.attentionMask;
            const nativeMask = topologyMask ? topologyMask : materializeAttentionTopologyMask(declaredMask, fused.attention, batch, querySequence, keySequence, pastLength);
            assertNativeAttentionMask(nativeMask, batch, fused.attention.numAttentionHeads, querySequence, keySequence, fused.attention.id);
            const positionValues = new Int32Array(batch * querySequence);
            for (let b = 0; b < batch; b += 1) for (let s = 0; s < querySequence; s += 1) {
              const position = positions[b]![s]!;
              if (!Number.isSafeInteger(position) || position < -2_147_483_648 || position > 2_147_483_647) throw new Error(`${operation.id}: posição ${position} excede o protocolo Int32 do subgrafo.`);
              positionValues[b * querySequence + s] = position;
            }
            const half = fused.queryRope.rotaryDim / 2;
            const proportionalPairs = fused.queryRope.ropeType === "proportional" ? Math.floor(Number(fused.queryRope.scaling?.partial_rotary_factor) * fused.attention.headDim / 2) : half;
            const proportionalFactor = fused.queryRope.ropeType === "proportional" ? Number(fused.queryRope.scaling?.factor ?? 1) : 1;
            const result = await options.linearTileKernel.fusedAttentionStorageReferences({
              input: input.values, positions: positionValues, mask: nativeMask.values,
              sourceKey: sourceKey?.values ?? new Float32Array(), sourceValue: sourceValue?.values ?? new Float32Array(),
              queryWeight: tensorInfo(artifact, operation.weight), queryNorm: tensorInfo(artifact, fused.queryNorm.weight!), outputWeight: tensorInfo(artifact, fused.output.weight),
              ...(fused.key ? { keyWeight: tensorInfo(artifact, fused.key.weight), keyNorm: tensorInfo(artifact, fused.keyNorm!.weight!) } : {}),
              ...(fused.value ? { valueWeight: tensorInfo(artifact, fused.value.weight) } : {}),
              batch, querySequence, sourceSequence, hiddenSize, queryHeads: fused.attention.numAttentionHeads, keyValueHeads: fused.attention.numKeyValueHeads, headDim: fused.attention.headDim, maskHeads: nativeMask.shape[1]!,
              producesKeyValue, valueFromKey: fused.valueFromKey, epsilon: fused.queryNorm.epsilon, scale: fused.attention.scale,
              ropeType: fused.queryRope.ropeType, theta: fused.queryRope.theta, rotaryDim: fused.queryRope.rotaryDim, proportionalPairs, proportionalFactor, rounding: options.fusedAttentionRounding,
            });
            if (result.projected.length !== batch * querySequence * fused.output.outFeatures || result.projected.some((entry) => !Number.isFinite(entry))) throw new Error(`${operation.id}: subgrafo de attention retornou projeção inválida.`);
            values.set(fused.output.output, { shape: [batch, querySequence, fused.output.outFeatures], values: result.projected });
            if (producesKeyValue) {
              const cacheElements = batch * fused.attention.numKeyValueHeads * keySequence * fused.attention.headDim;
              if (!result.key || !result.value || result.key.length !== cacheElements || result.value.length !== cacheElements || result.key.some((entry) => !Number.isFinite(entry)) || result.value.some((entry) => !Number.isFinite(entry))) throw new Error(`${operation.id}: subgrafo de attention retornou cache KV inválido.`);
              producedCache.set(fused.attention.layer, {
                key: { shape: [batch, fused.attention.numKeyValueHeads, keySequence, fused.attention.headDim], values: result.key },
                value: { shape: [batch, fused.attention.numKeyValueHeads, keySequence, fused.attention.headDim], values: result.value },
              });
            }
            operationIndex += fused.operations.length - 1;
            break;
          }
        }
        if (options.fusedMlpRounding && options.linearTileKernel?.fusedGatedMlpStorageReference) {
          const fused = matchFusedGatedMlp(operations, operationIndex, operation);
          if (fused) {
            for (const fusedOperation of fused.operations) assertPagedF32Policy(fusedOperation);
            const input = value(values, operation.input), rows = input.values.length / operation.inFeatures;
            const outputValues = await options.linearTileKernel.fusedGatedMlpStorageReference(input.values, tensorInfo(artifact, operation.weight), tensorInfo(artifact, fused.up.weight), tensorInfo(artifact, fused.down.weight), rows, options.fusedMlpRounding);
            if (outputValues.length !== rows * fused.down.outFeatures || outputValues.some((entry) => !Number.isFinite(entry))) throw new Error(`${options.linearTileKernel.backend}: subgrafo MLP retornou saída inválida.`);
            const output = { shape: [...input.shape.slice(0, -1), fused.down.outFeatures], values: outputValues };
            if (options.fusedMlpRounding === "real") values.set(fused.down.output, output); else store(fused.down, output);
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
        store(operation, await pagedLinearF32(value(values, operation.input), matrix(operation.weight, operation.id === "lm_head" ? options.finalHeadMaxReadBytes ?? maxReadBytes : maxReadBytes), {
          outputDtype: operation.dtypePolicy.outputDtype === "BF16" ? "BF16" : "F32",
          accumulationDtype: operation.dtypePolicy.accumulationDtype === "F64" ? "F64" : "F32",
          ...(operation.dtypePolicy.reduction ? { reduction: operation.dtypePolicy.reduction } : {}),
          ...(options.linearTileKernel ? { tileKernel: options.linearTileKernel } : {}),
          ...(operation.id === "lm_head" && options.finalHeadCompute !== undefined && options.finalHeadCompute !== "f32" ? { nativeBf16: true } : {}),
          ...(operation.id === "lm_head" && options.finalHeadCompute === "native-bf16-stream" ? { streamedNativeBf16: true } : {}),
          ...(operation.id === "lm_head" && options.finalHeadCompute === "native-bf16-whole" ? { wholeNativeBf16: true } : {}),
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
type ReshapeHeadsOperation = Extract<Operation, { op: "reshape_heads" }>;
type RmsNormOperation = Extract<Operation, { op: "rms_norm" }>;
type RotaryOperation = Extract<Operation, { op: "rotary_embedding" }>;
type AttentionOperation = Extract<Operation, { op: "scaled_dot_product_attention" }>;
type SelectPerLayerOperation = Extract<Operation, { op: "select_per_layer" }>;
type TensorScaleOperation = Extract<Operation, { op: "tensor_scale" }>;
type ReshapePerLayerOperation = Extract<Operation, { op: "reshape_per_layer" }>;
type ElementwiseOperation = Extract<Operation, { op: "elementwise" }>;

interface FusedAttentionMatch {
  operations: readonly Operation[];
  queryNorm: RmsNormOperation;
  queryRope: RotaryOperation & { ropeType: "default" | "proportional" };
  key?: LinearOperation;
  keyNorm?: RmsNormOperation;
  value?: LinearOperation;
  valueFromKey: boolean;
  attention: AttentionOperation;
  output: LinearOperation;
}

interface FusedDecoderLayerMatch {
  operations: readonly Operation[];
  inputNorm: RmsNormOperation;
  query: LinearOperation;
  attention: FusedAttentionMatch;
  postAttentionNorm: RmsNormOperation;
  attentionResidual: ElementwiseOperation;
  ffn: ReturnType<typeof matchFusedFfn> & {};
  pleSelect: SelectPerLayerOperation;
  ple: ReturnType<typeof matchFusedPle> & {};
}

interface FusedDecoderStackMatch {
  operations: readonly Operation[];
  layers: readonly FusedDecoderLayerMatch[];
  perLayerInput: string;
  perLayerWidth: number;
}

function matchFusedAttention(operations: readonly Operation[], index: number, query: LinearOperation): FusedAttentionMatch | undefined {
  const queryHeads = operations[index + 1], queryNorm = operations[index + 2], queryRope = operations[index + 3];
  if (!isHeadsAfter(queryHeads, query) || !isWeightedNormAfter(queryNorm, queryHeads) || !isRopeAfter(queryRope, queryNorm)) return undefined;
  if (queryRope.ropeType !== "default" && queryRope.ropeType !== "proportional") return undefined;
  let cursor = index + 4;
  let key: LinearOperation | undefined, keyHeads: ReshapeHeadsOperation | undefined, keyNorm: RmsNormOperation | undefined, keyRope: RotaryOperation | undefined;
  let value: LinearOperation | undefined, valueNorm: RmsNormOperation | undefined, valueFromKey = false;
  if (operations[cursor]?.op !== "scaled_dot_product_attention") {
    const possibleKey = operations[cursor], possibleKeyHeads = operations[cursor + 1], possibleKeyNorm = operations[cursor + 2], possibleKeyRope = operations[cursor + 3];
    if (possibleKey?.op !== "linear" || possibleKey.input !== query.input || !possibleKey.transposeWeight || possibleKey.bias || !isHeadsAfter(possibleKeyHeads, possibleKey) || !isWeightedNormAfter(possibleKeyNorm, possibleKeyHeads) || !isRopeAfter(possibleKeyRope, possibleKeyNorm)) return undefined;
    if (possibleKeyRope.ropeType !== queryRope.ropeType || possibleKeyRope.theta !== queryRope.theta || possibleKeyRope.rotaryDim !== queryRope.rotaryDim || JSON.stringify(possibleKeyRope.scaling) !== JSON.stringify(queryRope.scaling)) return undefined;
    key = possibleKey; keyHeads = possibleKeyHeads; keyNorm = possibleKeyNorm; keyRope = possibleKeyRope; cursor += 4;
    const possibleValue = operations[cursor];
    if (possibleValue?.op === "linear") {
      const possibleValueHeads = operations[cursor + 1], possibleValueNorm = operations[cursor + 2];
      if (possibleValue.input !== query.input || !possibleValue.transposeWeight || possibleValue.bias || !isHeadsAfter(possibleValueHeads, possibleValue) || possibleValueNorm?.op !== "rms_norm" || possibleValueNorm.input !== possibleValueHeads.output || possibleValueNorm.weight || possibleValueNorm.weightTransform !== "none") return undefined;
      value = possibleValue; valueNorm = possibleValueNorm; cursor += 3;
    } else {
      if (possibleValue?.op !== "rms_norm" || possibleValue.input !== keyHeads.output || possibleValue.weight || possibleValue.weightTransform !== "none") return undefined;
      valueNorm = possibleValue; valueFromKey = true; cursor += 1;
    }
  }
  const attention = operations[cursor], output = operations[cursor + 1];
  if (attention?.op !== "scaled_dot_product_attention" || output?.op !== "linear" || !output.transposeWeight || output.bias || output.input !== attention.output || attention.query !== queryRope.output) return undefined;
  if (queryHeads.numHeads !== attention.numAttentionHeads || queryHeads.headDim !== attention.headDim || query.outFeatures !== attention.numAttentionHeads * attention.headDim || output.inFeatures !== query.outFeatures || query.inFeatures !== output.outFeatures) return undefined;
  if (key) {
    if (attention.kvSharing || !keyHeads || !keyNorm || !keyRope || !valueNorm || attention.key !== keyRope.output || attention.value !== valueNorm.output || keyHeads.numHeads !== attention.numKeyValueHeads || keyHeads.headDim !== attention.headDim || key.outFeatures !== attention.numKeyValueHeads * attention.headDim || value && value.outFeatures !== key.outFeatures || queryNorm.epsilon !== keyNorm.epsilon || queryNorm.epsilon !== valueNorm.epsilon) return undefined;
  } else if (!attention.kvSharing) return undefined;
  const matched = operations.slice(index, cursor + 2);
  return { operations: matched, queryNorm, queryRope: queryRope as FusedAttentionMatch["queryRope"], ...(key && keyNorm ? { key, keyNorm } : {}), ...(value ? { value } : {}), valueFromKey, attention, output };
}

function isHeadsAfter(operation: Operation | undefined, linear: LinearOperation): operation is ReshapeHeadsOperation {
  return operation?.op === "reshape_heads" && operation.layout === "BHSD" && operation.input === linear.output && operation.numHeads * operation.headDim === linear.outFeatures;
}

function isWeightedNormAfter(operation: Operation | undefined, input: ReshapeHeadsOperation): operation is RmsNormOperation {
  return operation?.op === "rms_norm" && operation.input === input.output && operation.weight !== undefined && operation.weightTransform === "direct" && operation.weight.shape.length === 1 && operation.weight.shape[0] === input.headDim;
}

function isRopeAfter(operation: Operation | undefined, input: RmsNormOperation): operation is RotaryOperation {
  return operation?.op === "rotary_embedding" && operation.input === input.output && operation.layout === "rotate_half";
}

function matchFusedGatedMlp(operations: readonly Operation[], index: number, gate: LinearOperation): { up: LinearOperation; down: LinearOperation; operations: readonly Operation[] } | undefined {
  const up = operations[index + 1], activation = operations[index + 2], multiply = operations[index + 3], down = operations[index + 4];
  if (up?.op !== "linear" || activation?.op !== "activation" || multiply?.op !== "elementwise" || down?.op !== "linear") return undefined;
  if (up.input !== gate.input || !up.transposeWeight || up.bias || activation.input !== gate.output || activation.function !== "gelu" || activation.approximation !== "tanh" || multiply.kind !== "multiply" || multiply.inputs.length !== 2 || multiply.inputs[0] !== activation.output || multiply.inputs[1] !== up.output || down.input !== multiply.output || !down.transposeWeight || down.bias) return undefined;
  if ([gate, up, activation, multiply, down].some((operation) => operation.dtypePolicy.outputDtype !== "BF16")) return undefined;
  return { up, down, operations: [up, activation, multiply, down] };
}

function matchFusedFfn(operations: readonly Operation[], index: number, preNorm: RmsNormOperation): { operations: readonly Operation[]; preNorm: RmsNormOperation; gate: LinearOperation; up: LinearOperation; down: LinearOperation; postNorm: RmsNormOperation; residual: ElementwiseOperation } | undefined {
  const gate = operations[index + 1], up = operations[index + 2], activation = operations[index + 3], multiply = operations[index + 4], down = operations[index + 5], postNorm = operations[index + 6], residual = operations[index + 7];
  if (!preNorm.weight || preNorm.weightTransform !== "direct" || preNorm.weight.shape.length !== 1 || gate?.op !== "linear" || up?.op !== "linear" || activation?.op !== "activation" || multiply?.op !== "elementwise" || down?.op !== "linear" || postNorm?.op !== "rms_norm" || residual?.op !== "elementwise") return undefined;
  if (gate.input !== preNorm.output || up.input !== preNorm.output || !gate.transposeWeight || gate.bias || !up.transposeWeight || up.bias || gate.inFeatures !== up.inFeatures || gate.outFeatures !== up.outFeatures || preNorm.weight.shape[0] !== gate.inFeatures) return undefined;
  if (activation.input !== gate.output || activation.function !== "gelu" || activation.approximation !== "tanh" || multiply.kind !== "multiply" || multiply.inputs.length !== 2 || multiply.inputs[0] !== activation.output || multiply.inputs[1] !== up.output || down.input !== multiply.output || !down.transposeWeight || down.bias || down.inFeatures !== gate.outFeatures || down.outFeatures !== gate.inFeatures) return undefined;
  if (postNorm.input !== down.output || !postNorm.weight || postNorm.weightTransform !== "direct" || postNorm.weight.shape.length !== 1 || postNorm.weight.shape[0] !== gate.inFeatures || residual.kind !== "add" || residual.inputs.length !== 2 || residual.inputs[0] !== preNorm.input || residual.inputs[1] !== postNorm.output) return undefined;
  const matched = operations.slice(index, index + 8);
  if (matched.some((entry) => entry.dtypePolicy.outputDtype !== "BF16")) return undefined;
  return { operations: matched, preNorm, gate, up, down, postNorm, residual };
}

function matchFusedDecoderLayer(operations: readonly Operation[], index: number, inputNorm: RmsNormOperation): FusedDecoderLayerMatch | undefined {
  if (!inputNorm.weight || inputNorm.weightTransform !== "direct" || inputNorm.weight.shape.length !== 1) return undefined;
  const query = operations[index + 1];
  if (query?.op !== "linear" || query.input !== inputNorm.output) return undefined;
  const attention = matchFusedAttention(operations, index + 1, query);
  if (!attention) return undefined;
  const attentionEnd = index + 1 + attention.operations.length;
  const postAttentionNorm = operations[attentionEnd], attentionResidual = operations[attentionEnd + 1], preFfnNorm = operations[attentionEnd + 2];
  if (postAttentionNorm?.op !== "rms_norm" || postAttentionNorm.input !== attention.output.output || !postAttentionNorm.weight || postAttentionNorm.weightTransform !== "direct" || postAttentionNorm.weight.shape.length !== 1 || postAttentionNorm.weight.shape[0] !== query.inFeatures || attentionResidual?.op !== "elementwise" || attentionResidual.kind !== "add" || attentionResidual.inputs.length !== 2 || attentionResidual.inputs[0] !== inputNorm.input || attentionResidual.inputs[1] !== postAttentionNorm.output || preFfnNorm?.op !== "rms_norm" || preFfnNorm.input !== attentionResidual.output) return undefined;
  const ffn = matchFusedFfn(operations, attentionEnd + 2, preFfnNorm);
  if (!ffn) return undefined;
  const pleStart = attentionEnd + 2 + ffn.operations.length;
  const select = operations[pleStart];
  if (select?.op !== "select_per_layer") return undefined;
  const ple = matchFusedPle(operations, pleStart, select, "bf16");
  if (!ple || ple.gate.input !== ffn.residual.output || ple.scalar.output === undefined) return undefined;
  const matched = operations.slice(index, pleStart + ple.operations.length);
  return { operations: matched, inputNorm, query, attention, postAttentionNorm, attentionResidual, ffn, pleSelect: select, ple };
}

function matchFusedDecoderStack(operations: readonly Operation[], index: number, inputNorm: RmsNormOperation): FusedDecoderStackMatch | undefined {
  const layers: FusedDecoderLayerMatch[] = [];
  let cursor = index, currentNorm: RmsNormOperation | undefined = inputNorm, expectedInput = inputNorm.input, perLayerInput: string | undefined, perLayerWidth: number | undefined, declaredLayers: number | undefined;
  while (currentNorm) {
    const layer = matchFusedDecoderLayer(operations, cursor, currentNorm);
    if (!layer || layer.inputNorm.input !== expectedInput || layer.attention.attention.layer !== layers.length || layer.pleSelect.layerIndex !== layers.length) break;
    perLayerInput ??= layer.pleSelect.input; perLayerWidth ??= layer.pleSelect.layerWidth; declaredLayers ??= layer.pleSelect.numLayers;
    if (layer.pleSelect.input !== perLayerInput || layer.pleSelect.layerWidth !== perLayerWidth || layer.pleSelect.numLayers !== declaredLayers) return undefined;
    layers.push(layer); cursor += layer.operations.length; expectedInput = layer.ple.scalar.output;
    const next = operations[cursor]; currentNorm = next?.op === "rms_norm" ? next : undefined;
  }
  if (!perLayerInput || perLayerWidth === undefined || declaredLayers === undefined || layers.length !== declaredLayers || layers.length < 2) return undefined;
  return { operations: operations.slice(index, cursor), layers, perLayerInput, perLayerWidth };
}

function matchFusedPle(operations: readonly Operation[], index: number, select: SelectPerLayerOperation, rounding: "bf16" | "real"): { operations: readonly Operation[]; gate: LinearOperation; projection: LinearOperation; norm: RmsNormOperation; scalar: TensorScaleOperation } | undefined {
  const gate = operations[index + 1], activation = operations[index + 2], multiply = operations[index + 3], projection = operations[index + 4], norm = operations[index + 5], residual = operations[index + 6], scalar = operations[index + 7];
  if (gate?.op !== "linear" || activation?.op !== "activation" || multiply?.op !== "elementwise" || projection?.op !== "linear" || norm?.op !== "rms_norm" || residual?.op !== "elementwise" || scalar?.op !== "tensor_scale") return undefined;
  if (!gate.transposeWeight || gate.bias || gate.outFeatures !== select.layerWidth || activation.input !== gate.output || activation.function !== "gelu" || activation.approximation !== "tanh" || multiply.kind !== "multiply" || multiply.inputs.length !== 2 || multiply.inputs[0] !== activation.output || multiply.inputs[1] !== select.output || projection.input !== multiply.output || !projection.transposeWeight || projection.bias || projection.inFeatures !== select.layerWidth || projection.outFeatures !== gate.inFeatures || norm.input !== projection.output || !norm.weight || norm.weightTransform !== "direct" || norm.weight.shape.length !== 1 || norm.weight.shape[0] !== projection.outFeatures || residual.kind !== "add" || residual.inputs.length !== 2 || residual.inputs[0] !== gate.input || residual.inputs[1] !== norm.output || scalar.input !== residual.output || scalar.scalar.shape.length !== 1 || scalar.scalar.shape[0] !== 1) return undefined;
  const matched = operations.slice(index, index + 8);
  if (rounding === "bf16" && matched.some((operation) => operation.dtypePolicy.outputDtype !== "BF16")) return undefined;
  return { operations: matched, gate, projection, norm, scalar };
}

function matchFusedPlePrelude(operations: readonly Operation[], index: number, projection: LinearOperation, rounding: "bf16" | "real"): { operations: readonly Operation[]; contextScale: ElementwiseOperation; reshape: ReshapePerLayerOperation; norm: RmsNormOperation; combineScale: ElementwiseOperation; tokenIdentityInput: string } | undefined {
  const contextScale = operations[index + 1], reshape = operations[index + 2], norm = operations[index + 3], combine = operations[index + 4], combineScale = operations[index + 5];
  if (contextScale?.op !== "elementwise" || reshape?.op !== "reshape_per_layer" || norm?.op !== "rms_norm" || combine?.op !== "elementwise" || combineScale?.op !== "elementwise") return undefined;
  if (!projection.transposeWeight || projection.bias || contextScale.kind !== "scale" || contextScale.inputs.length !== 1 || contextScale.inputs[0] !== projection.output || contextScale.scalar === undefined || !Number.isFinite(contextScale.scalar) || reshape.input !== contextScale.output || reshape.numLayers * reshape.layerWidth !== projection.outFeatures || norm.input !== reshape.output || !norm.weight || norm.weightTransform !== "direct" || norm.weight.shape.length !== 1 || norm.weight.shape[0] !== reshape.layerWidth || combine.kind !== "add" || combine.inputs.length !== 2 || combine.inputs[0] !== norm.output || combineScale.kind !== "scale" || combineScale.inputs.length !== 1 || combineScale.inputs[0] !== combine.output || combineScale.scalar === undefined || !Number.isFinite(combineScale.scalar)) return undefined;
  const matched = operations.slice(index, index + 6);
  if (rounding === "bf16" && matched.some((operation) => operation.dtypePolicy.outputDtype !== "BF16")) return undefined;
  return { operations: matched, contextScale, reshape, norm, combineScale, tokenIdentityInput: combine.inputs[1]! };
}

function assertFusedSourceCache(key: DenseF32Tensor, valueTensor: DenseF32Tensor, batch: number, heads: number, headDim: number, operationId: string): void {
  if (key.shape.length !== 4 || valueTensor.shape.length !== 4 || key.shape[0] !== batch || key.shape[1] !== heads || key.shape[3] !== headDim || valueTensor.shape.some((dimension, index) => dimension !== key.shape[index]) || key.shape[2]! < 0) throw new Error(`${operationId}: cache KV é incompatível com o subgrafo de attention.`);
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
  if (mask) assertNativeAttentionMask(mask, batch, queryHeads, querySequence, keySequence, operation.id);
  const nativeMask = maskDefinesTopology && mask ? mask : materializeAttentionTopologyMask(mask, operation, batch, querySequence, keySequence, pastLength);
  const values = await kernel.attention({ query: query.values, key: key.values, value: valueTensor.values, mask: nativeMask.values, batch, queryHeads, keyValueHeads, querySequence, keySequence, headDim, maskHeads: nativeMask.shape[1]!, scale: operation.scale, rounding });
  if (values.length !== batch * querySequence * queryHeads * headDim || values.some((entry) => !Number.isFinite(entry))) throw new Error(`${operation.id}: attention nativa retornou saída inválida.`);
  return { tensor: { shape: [batch, querySequence, queryHeads * headDim], values }, native: true };
}

function assertNativeAttentionMask(mask: DenseF32Tensor, batch: number, queryHeads: number, querySequence: number, keySequence: number, operationId: string): void {
  const [maskBatch, maskHeads, maskQuery, maskKey] = mask.shape;
  if (mask.shape.length !== 4 || maskBatch !== batch || (maskHeads !== 1 && maskHeads !== queryHeads) || maskQuery !== querySequence || maskKey !== keySequence || mask.values.length !== batch * maskHeads! * querySequence * keySequence || mask.values.some((entry) => Number.isNaN(entry) || entry === Infinity)) throw new Error(`${operationId}: máscara da attention nativa é incompatível.`);
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
