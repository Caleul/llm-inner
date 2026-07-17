export type JsonObject = Record<string, unknown>;

export type SourceFormat = "safetensors" | "mlx-safetensors" | "gguf";

export interface TensorInfo {
  name: string;
  storageDtype: string;
  storageShape: number[];
  logicalShape: number[];
  byteOffset?: number;
  byteLength?: number;
  shard?: string;
  quantization?: QuantizationSpec;
}

export interface QuantizationSpec {
  family: "mlx" | "gguf" | "other";
  mode: string;
  bits?: number;
  groupSize?: number;
  tensorType?: string;
  scaleTensor?: string;
  biasTensor?: string;
  globalScaleTensor?: string;
}

export interface ModelCatalog {
  source: string;
  format: SourceFormat;
  config: JsonObject;
  rawMetadata: JsonObject;
  tensors: Map<string, TensorInfo>;
}

export interface TensorRef {
  name: string;
  shape: number[];
  storageDtype: string;
  quantization?: QuantizationSpec;
}

export interface PreviewOptions {
  outputRows: number;
  inputTerms: number;
  includeWeights: boolean;
}

export interface LinearPreviewRow {
  outputIndex: number;
  terms: Array<{ inputIndex: number; weight: number }>;
  omittedInputTerms: number;
}

export interface LinearPreview {
  rows: LinearPreviewRow[];
  omittedOutputRows: number;
}

export interface BaseOp {
  id: string;
  layer?: number;
  output: string;
  dtypePolicy: DtypePolicy;
}

export type ReductionSchedule =
  | { kind: "ordered-scalar"; indexOrder: "ascending" }
  /**
   * An ordered scalar reduction whose F32 accumulator receives the exact
   * product before that single F32 rounding boundary.  This is distinct from
   * `ordered-scalar`, which rounds every product before adding it.
   */
  | { kind: "ordered-fma"; indexOrder: "ascending" }
  /**
   * Consecutive terms first form an explicitly rounded F32 partial sum.  The
   * partial sums then feed one ordered F32 accumulator.  This captures a
   * dot-product instruction's adjacent-term boundary without pretending it
   * is either a SIMD lane fold or an ordinary scalar reduction.
   */
  | {
    kind: "blocked-f32-terms";
    termsPerBlock: number;
    inputBlock: "contiguous-terms";
    termOrder: "ascending" | "descending";
    productBoundary: "separately-rounded-f32" | "fused-fma";
    blockOrder: "ascending";
  }
  /**
   * Product terms are accumulated into lane `i mod laneCount`, then the
   * lanes are folded with the declared F32 order.  The fold is semantic: an
   * eager kernel may expose a different answer at a cancellation boundary
   * even when its lane count is unchanged.
   */
  | {
    kind: "interleaved-f32-lanes";
    laneCount: number;
    inputLane: "index-modulo-lane-count";
    laneReductionOrder: "ascending" | "descending" | "balanced-pairwise";
  }
  /**
   * Like `interleaved-f32-lanes`, except each lane addition performs one F32
   * rounding over `lane + exact(product)` rather than rounding the product
   * first.  It models an explicitly declared fused multiply-add boundary.
   */
  | {
    kind: "interleaved-fma-lanes";
    laneCount: number;
    inputLane: "index-modulo-lane-count";
    laneReductionOrder: "ascending" | "descending" | "balanced-pairwise";
  }
  /**
   * A tiled dot-product reduction.  For every tile of
   * `laneCount * termsPerLane` input coordinates, contiguous groups of
   * `termsPerLane` products feed one F32 lane.  This models BF16 dot-product
   * instructions which reduce adjacent BF16 pairs before their horizontal F32
   * fold; it is intentionally distinct from `i mod laneCount`.
   */
  | {
    kind: "tiled-f32-lanes";
    laneCount: number;
    termsPerLane: number;
    inputLane: "tile-contiguous-terms";
    laneReductionOrder: "ascending" | "descending" | "balanced-pairwise";
  }
  /** Like tiled F32 lanes, but each lane add has a fused product boundary. */
  | {
    kind: "tiled-fma-lanes";
    laneCount: number;
    termsPerLane: number;
    inputLane: "tile-contiguous-terms";
    laneReductionOrder: "ascending" | "descending" | "balanced-pairwise";
  }
  /**
   * A finite tiled dot-product partial.  Unlike `tiled-*-lanes`, its lanes
   * are reset for each tile; the folded F32 tile result is then added to one
   * ordered F32 accumulator.  This makes the instruction/tile boundary
   * explicit instead of silently carrying SIMD registers through the whole
   * feature dimension.
   */
  | {
    kind: "blocked-tiled-f32-lanes";
    laneCount: number;
    termsPerLane: number;
    inputBlock: "tile-contiguous-terms";
    laneReductionOrder: "ascending" | "descending" | "balanced-pairwise";
    productBoundary: "separately-rounded-f32" | "fused-fma";
    blockOrder: "ascending";
  };

export interface DtypePolicy {
  inputDtype?: string;
  computeDtype?: string;
  accumulationDtype?: string;
  outputDtype?: string;
  /**
   * The reduction is part of the mathematical program, rather than an
   * executor implementation detail.  A lane schedule is deliberately
   * explicit: it is never selected from a tensor shape at replay time.
   */
  reduction?: ReductionSchedule;
}

export interface RmsNormOp extends BaseOp {
  op: "rms_norm";
  input: string;
  /** Absent only for an explicitly declared unscaled RMS normalization. */
  weight?: TensorRef;
  epsilon: number;
  weightTransform: "direct" | "one_plus_weight" | "none";
  axis: number;
}

export interface LinearOp extends BaseOp {
  op: "linear";
  input: string;
  weight: TensorRef;
  bias?: TensorRef;
  inFeatures: number;
  outFeatures: number;
  transposeWeight: boolean;
  preview?: LinearPreview;
}

export interface ReshapeHeadsOp extends BaseOp {
  op: "reshape_heads";
  input: string;
  numHeads: number;
  headDim: number;
  layout: "BSHD" | "BHSD";
}

export interface RotaryOp extends BaseOp {
  op: "rotary_embedding";
  input: string;
  positionInput: string;
  ropeType: string;
  theta: number;
  rotaryDim: number;
  layout: "rotate_half" | "interleaved_pairs" | "multidimensional";
  scaling?: JsonObject;
}

export interface AttentionOp extends BaseOp {
  op: "scaled_dot_product_attention";
  query: string;
  key: string;
  value: string;
  maskInput: string;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  scale: number;
  scoreSoftcap?: number;
  softmaxComputeDtype: string;
  causal: boolean;
  slidingWindow?: number;
  kvSharing?: {
    enabled: boolean;
    producerLayer?: number;
    group?: string;
  };
}

export interface ActivationOp extends BaseOp {
  op: "activation";
  input: string;
  function: string;
  approximation?: string;
}

export interface ElementwiseOp extends BaseOp {
  op: "elementwise";
  kind: "add" | "multiply" | "scale" | "tanh_softcap";
  inputs: string[];
  scalar?: number;
}

export interface EmbeddingOp extends BaseOp {
  op: "embedding";
  tokenInput: string;
  weight: TensorRef;
  scale?: number;
}

/**
 * A packed auxiliary embedding that becomes one independent vector per
 * decoder layer. This is deliberately distinct from a normal token embedding:
 * its output rank and layer ordering are part of the model equation.
 */
export interface PerLayerEmbeddingOp extends BaseOp {
  op: "per_layer_embedding";
  tokenInput: string;
  weight: TensorRef;
  numLayers: number;
  layerWidth: number;
  scale?: number;
}

/** Reshapes a packed final dimension into [layer, feature] without reordering. */
export interface ReshapePerLayerOp extends BaseOp {
  op: "reshape_per_layer";
  input: string;
  numLayers: number;
  layerWidth: number;
}

/** Selects one declared decoder-layer slice from [B,S,L,D] in stable order. */
export interface SelectPerLayerOp extends BaseOp {
  op: "select_per_layer";
  input: string;
  layerIndex: number;
  numLayers: number;
  layerWidth: number;
}

/** Multiplies every value by a scalar tensor whose [1] shape is explicit. */
export interface TensorScaleOp extends BaseOp {
  op: "tensor_scale";
  input: string;
  scalar: TensorRef;
}

export type Operation =
  | RmsNormOp
  | LinearOp
  | ReshapeHeadsOp
  | RotaryOp
  | AttentionOp
  | ActivationOp
  | ElementwiseOp
  | EmbeddingOp
  | PerLayerEmbeddingOp
  | ReshapePerLayerOp
  | SelectPerLayerOp
  | TensorScaleOp;

export interface LayerIR {
  index: number;
  layerType: string;
  operations: Operation[];
}

export interface ModelIR {
  schemaVersion: 2;
  source: {
    path: string;
    format: SourceFormat;
  };
  architecture: {
    modelType: string;
    architectureClass?: string;
    hiddenSize: number;
    intermediateSize?: number | number[];
    numLayers: number;
    numAttentionHeads: number;
    numKeyValueHeads: number;
    headDim: number;
    vocabSize?: number;
  };
  config: JsonObject;
  preview: PreviewOptions;
  inputs: Array<{ name: string; description: string }>;
  prelude: Operation[];
  layers: LayerIR[];
  epilogue: Operation[];
  fidelity: {
    exactByConstruction: boolean;
    assumptions: string[];
    unsupported: string[];
    warnings: string[];
  };
}

/** A dense, row-major tensor used by the deterministic reference executor. */
export interface DenseTensor {
  shape: number[];
  values: Float64Array;
}

/** A dense, row-major F32 tensor. Values stay in their declared storage dtype. */
export interface DenseF32Tensor {
  shape: number[];
  values: Float32Array;
  /**
   * Present only when this F32 buffer was materialized by the declared
   * container backend from a specific quantized source tensor.  This is
   * provenance, not a claim about the original runtime's compute dtype.
   */
  sourceQuantization?: QuantizationSpec;
}

export interface ReferenceExecutionRequest {
  inputIds: number[][];
  /** Optional absolute positions. When omitted, each sequence starts at zero. */
  positionIds?: number[][];
  /** Canonical additive bias [batch, 1|heads, query, key]; -Infinity excludes a key. */
  attentionMask?: DenseTensor;
  /**
   * Canonical post-RoPE KV state keyed by decoder layer. Every entry uses
   * [batch, num_key_value_heads, cached_sequence, head_dim]. When supplied,
   * it must cover every KV-producing attention layer in the IR. An attention
   * operation with kvSharing consumes its producer's entry and never owns a
   * duplicate cache entry of its own.
   */
  pastKeyValues?: ReadonlyMap<number, ReferenceKeyValueCache>;
  /** Dense F64 constants keyed by the tensor reference name in the IR. */
  tensors: ReadonlyMap<string, DenseTensor>;
}

export interface ReferenceKeyValueCache {
  key: DenseTensor;
  value: DenseTensor;
}

export interface ReferenceExecutionResult {
  values: ReadonlyMap<string, DenseTensor>;
  logits: DenseTensor;
  pastKeyValues: ReadonlyMap<number, ReferenceKeyValueCache>;
}

/**
 * Scalar IEEE-754 binary32 reference execution. This is deliberately a
 * separate contract from the F64 interpreter: callers must provide F32
 * constants and every IR operation must declare an explicit F32 policy.
 */
export interface ReferenceF32ExecutionRequest {
  inputIds: number[][];
  positionIds?: number[][];
  /** Canonical additive bias [batch, 1|heads, query, key]; -Infinity excludes a key. */
  attentionMask?: DenseF32Tensor;
  /**
   * Optional per-layer complete attention topologies. A mask here replaces the
   * operation's built-in causal/sliding bounds for that layer: every excluded
   * key must therefore be encoded as -Infinity in the supplied additive bias.
   * This is required for architectures whose authoritative runtime assigns
   * different mask families to different decoder layers.
   */
  attentionMasksByLayer?: ReadonlyMap<number, DenseF32Tensor>;
  /** Canonical post-RoPE KV state; see ReferenceExecutionRequest.pastKeyValues. */
  pastKeyValues?: ReadonlyMap<number, ReferenceF32KeyValueCache>;
  tensors: ReadonlyMap<string, DenseF32Tensor>;
}

export interface ReferenceF32KeyValueCache {
  key: DenseF32Tensor;
  value: DenseF32Tensor;
}

export interface ReferenceF32ExecutionResult {
  values: ReadonlyMap<string, DenseF32Tensor>;
  logits: DenseF32Tensor;
  pastKeyValues: ReadonlyMap<number, ReferenceF32KeyValueCache>;
}

/**
 * Greedy decoding is intentionally a separate, narrow contract from a
 * forward request. The current reference path accepts one unpadded prompt so
 * that every sampled token has one unambiguous absolute position and one KV
 * ownership chain. Padding/batched stop policies require their own declared
 * mask and cache semantics rather than being guessed here.
 */
export interface ReferenceGenerationRequest {
  inputIds: number[][];
  positionIds?: number[][];
  tensors: ReadonlyMap<string, DenseTensor>;
  maxNewTokens: number;
  eosTokenId?: number;
}

export interface ReferenceF32GenerationRequest {
  inputIds: number[][];
  positionIds?: number[][];
  tensors: ReadonlyMap<string, DenseF32Tensor>;
  maxNewTokens: number;
  eosTokenId?: number;
}

export interface ReferenceGenerationStep {
  /** Token chosen from the preceding terminal logits. */
  tokenId: number;
  /** Absolute position at which that chosen token was subsequently evaluated. */
  positionId: number;
}

export interface ReferenceGenerationResult {
  inputIds: number[];
  generatedTokenIds: number[];
  steps: ReferenceGenerationStep[];
  /** Logits used to select each corresponding generated token, before it is evaluated into the cache. */
  selectionLogits: DenseTensor[];
  /** Post-RoPE KV state immediately after evaluating each emitted token. */
  stepPastKeyValues: ReadonlyArray<ReadonlyMap<number, ReferenceKeyValueCache>>;
  /** Terminal logits after evaluating inputIds plus every generated token. */
  logits: DenseTensor;
  /** Post-RoPE KV cache for inputIds plus every generated token. */
  pastKeyValues: ReadonlyMap<number, ReferenceKeyValueCache>;
}

export interface ReferenceF32GenerationResult {
  inputIds: number[];
  generatedTokenIds: number[];
  steps: ReferenceGenerationStep[];
  /** Logits used to select each corresponding generated token, under the F32 policy. */
  selectionLogits: DenseF32Tensor[];
  /** Post-RoPE KV state immediately after evaluating each emitted token. */
  stepPastKeyValues: ReadonlyArray<ReadonlyMap<number, ReferenceF32KeyValueCache>>;
  logits: DenseF32Tensor;
  pastKeyValues: ReadonlyMap<number, ReferenceF32KeyValueCache>;
}

/** Runtime-captured output for one IR operation, keyed by the stable IR id. */
export interface DifferentialOperationSample {
  operationId: string;
  output: string;
  tensor: DenseTensor | DenseF32Tensor;
}

/** Authoritative post-RoPE cache in the canonical BHSD layout for one layer. */
export interface DifferentialKeyValueCacheSample {
  layer: number;
  key: DenseTensor | DenseF32Tensor;
  value: DenseTensor | DenseF32Tensor;
}

/**
 * Authoritative-runtime trace required for an operation-level comparison. The
 * caller, not the decompiler, is responsible for capturing values with a
 * verified runtime hook and recording its immutable model identity.
 */
export interface DifferentialReferenceTrace {
  runtime: string;
  /**
   * Concrete device used by the authoritative runtime, when the capture
   * contract establishes one.  CPU and MPS linear kernels are distinct
   * numerical contracts and must never be merged into one fidelity claim.
   */
  executionDevice?: string;
  model: string;
  revisionOrChecksum: string;
  containerFormat: SourceFormat | string;
  quantization: string;
  inputTokens: number[][];
  /**
   * Absolute positions used by the authoritative execution capture. Omitted
   * only for legacy traces whose sequences all start at zero.
   */
  positionIds?: number[][];
  dtypePolicy: string;
  operations: readonly DifferentialOperationSample[];
  pastKeyValues: readonly DifferentialKeyValueCacheSample[];
}

export interface DifferentialTolerance {
  /** A coordinate passes when either absolute or relative error is within tolerance. */
  maxAbsoluteError: number;
  maxRelativeError: number;
}

export interface DifferentialTensorMetrics {
  shape: number[];
  elementCount: number;
  maxAbsoluteError: number;
  maxRelativeError: number;
  cosineSimilarity: number | null;
  topKOverlap: number | null;
  argmaxAgreement: boolean | null;
  nonFiniteMismatchCount: number;
}

export interface DifferentialOperationComparison {
  operationId: string;
  output: string;
  status: "pass" | "diverged" | "missing-reference" | "shape-mismatch";
  metrics?: DifferentialTensorMetrics;
}

/**
 * A deliberately partial, named set of runtime checkpoints.  Unlike a full
 * operation trace this is used to localize an already-known mismatch at
 * native module boundaries; it must never be promoted to an end-to-end
 * fidelity claim.
 */
export interface DifferentialCheckpointComparisonReport {
  candidateRuntime: string;
  tolerance: DifferentialTolerance;
  operations: DifferentialOperationComparison[];
  missingCandidateOperationIds: string[];
  firstDivergentOperation: string | null;
  fidelityClass: "lossless-within-dtype" | "approximate" | "incomplete";
}

export interface DifferentialKeyValueCacheComparison {
  layer: number;
  status: "pass" | "diverged" | "missing-reference" | "missing-candidate" | "shape-mismatch";
  key?: DifferentialTensorMetrics;
  value?: DifferentialTensorMetrics;
}

/** Serializable evidence object; it never upgrades a numerical comparison to bitwise equivalence. */
export interface DifferentialComparisonReport {
  reference: Omit<DifferentialReferenceTrace, "operations" | "pastKeyValues">;
  candidateRuntime: string;
  tolerance: DifferentialTolerance;
  operations: DifferentialOperationComparison[];
  kvCache: DifferentialKeyValueCacheComparison[];
  missingReferenceOperationIds: string[];
  unexpectedReferenceOperationIds: string[];
  firstDivergentOperation: string | null;
  logits: DifferentialTensorMetrics | null;
  fidelityClass: "bitwise" | "lossless-within-dtype" | "numerically-equivalent" | "approximate" | "incomplete";
}

/**
 * Authoritative greedy-decoding capture. This deliberately records the
 * prompt positions and every emitted-token position: token agreement without
 * cache/position agreement is not evidence of equivalent generation.
 */
export interface DifferentialGenerationReferenceTrace {
  runtime: string;
  /** See DifferentialReferenceTrace.executionDevice. */
  executionDevice?: string;
  model: string;
  revisionOrChecksum: string;
  containerFormat: SourceFormat | string;
  quantization: string;
  inputTokens: number[];
  promptPositionIds: number[];
  dtypePolicy: string;
  maxNewTokens: number;
  eosTokenId?: number;
  generatedTokenIds: number[];
  steps: readonly ReferenceGenerationStep[];
  /** One pre-selection logits tensor per emitted token. */
  selectionLogits: readonly (DenseTensor | DenseF32Tensor)[];
  /** One canonical post-RoPE KV snapshot per emitted token, in decode order. */
  stepPastKeyValues: readonly (readonly DifferentialKeyValueCacheSample[])[];
  logits: DenseTensor | DenseF32Tensor;
  pastKeyValues: readonly DifferentialKeyValueCacheSample[];
}

export interface DifferentialGenerationStepComparison {
  index: number;
  status: "pass" | "diverged" | "missing-reference" | "missing-candidate";
  candidate?: ReferenceGenerationStep;
  reference?: ReferenceGenerationStep;
  selectionLogits?: DifferentialTensorMetrics | null;
  /** KV state after the corresponding selected token was evaluated. */
  kvCache?: DifferentialKeyValueCacheComparison[];
}

/** Serializable end-to-end greedy-generation evidence. */
export interface DifferentialGenerationComparisonReport {
  reference: Omit<DifferentialGenerationReferenceTrace, "generatedTokenIds" | "steps" | "selectionLogits" | "stepPastKeyValues" | "logits" | "pastKeyValues">;
  candidateRuntime: string;
  tolerance: DifferentialTolerance;
  promptMatches: boolean;
  generatedTokenIds: DifferentialGenerationStepComparison[];
  terminalLogits: DifferentialTensorMetrics | null;
  kvCache: DifferentialKeyValueCacheComparison[];
  firstDivergence: string | null;
  fidelityClass: "bitwise" | "lossless-within-dtype" | "numerically-equivalent" | "approximate" | "incomplete";
}
