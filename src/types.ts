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

export interface ReferenceExecutionRequest {
  inputIds: number[][];
  /** Optional absolute positions. When omitted, each sequence starts at zero. */
  positionIds?: number[][];
  /** Dense F64 constants keyed by the tensor reference name in the IR. */
  tensors: ReadonlyMap<string, DenseTensor>;
}

export interface ReferenceExecutionResult {
  values: ReadonlyMap<string, DenseTensor>;
  logits: DenseTensor;
}
