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

export interface DtypePolicy {
  inputDtype?: string;
  computeDtype?: string;
  accumulationDtype?: string;
  outputDtype?: string;
}

export interface RmsNormOp extends BaseOp {
  op: "rms_norm";
  input: string;
  weight: TensorRef;
  epsilon: number;
  weightTransform: "direct" | "one_plus_weight";
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

export type Operation =
  | RmsNormOp
  | LinearOp
  | ReshapeHeadsOp
  | RotaryOp
  | AttentionOp
  | ActivationOp
  | ElementwiseOp
  | EmbeddingOp;

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
   * it must cover every non-shared decoder layer in the IR.
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
  /** Terminal logits after evaluating inputIds plus every generated token. */
  logits: DenseTensor;
  /** Post-RoPE KV cache for inputIds plus every generated token. */
  pastKeyValues: ReadonlyMap<number, ReferenceKeyValueCache>;
}

export interface ReferenceF32GenerationResult {
  inputIds: number[];
  generatedTokenIds: number[];
  steps: ReferenceGenerationStep[];
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
  model: string;
  revisionOrChecksum: string;
  containerFormat: SourceFormat | string;
  quantization: string;
  inputTokens: number[][];
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
  logits: DenseTensor | DenseF32Tensor;
  pastKeyValues: readonly DifferentialKeyValueCacheSample[];
}

export interface DifferentialGenerationStepComparison {
  index: number;
  status: "pass" | "diverged" | "missing-reference" | "missing-candidate";
  candidate?: ReferenceGenerationStep;
  reference?: ReferenceGenerationStep;
}

/** Serializable end-to-end greedy-generation evidence. */
export interface DifferentialGenerationComparisonReport {
  reference: Omit<DifferentialGenerationReferenceTrace, "generatedTokenIds" | "steps" | "logits" | "pastKeyValues">;
  candidateRuntime: string;
  tolerance: DifferentialTolerance;
  promptMatches: boolean;
  generatedTokenIds: DifferentialGenerationStepComparison[];
  terminalLogits: DifferentialTensorMetrics | null;
  kvCache: DifferentialKeyValueCacheComparison[];
  firstDivergence: string | null;
  fidelityClass: "bitwise" | "lossless-within-dtype" | "numerically-equivalent" | "approximate" | "incomplete";
}
