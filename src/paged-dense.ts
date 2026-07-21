import { decodeIeeeBF16ToF32, decodeIeeeF16ToF32, roundF32ToBF16 } from "./utils.js";
import type { LiteralTensorReader } from "./literal.js";
import type { DenseF32Tensor, ReductionSchedule, TensorInfo } from "./types.js";
import { armNeonBf16DotF32 } from "./native-reductions.js";

/**
 * A deliberately narrow, source-independent dense storage boundary for
 * literal artifacts.  It only accepts row-major F32/F16/BF16 matrices: packed
 * storage needs its own declared paging/decoder contract rather than being
 * treated as dense bytes.
 */
export interface PagedDenseF32Matrix {
  readonly tensor: TensorInfo;
  readonly shape: readonly [number, number];
  readonly maxReadBytes: number;
  readRows(startRow: number, rowCount: number): Promise<DenseF32Tensor>;
  readStorageRows?(startRow: number, rowCount: number): Promise<Buffer>;
}

export interface PagedLinearTileKernel {
  readonly backend: string;
  multiply(input: Float32Array, weight: Float32Array, rows: number, outputCount: number, inFeatures: number): Promise<Float32Array>;
  multiplyStorage?(input: Float32Array, weight: Buffer, storageDtype: "F32" | "F16" | "BF16", rows: number, outputCount: number, inFeatures: number): Promise<Float32Array>;
  multiplyStorageReference?: ((input: Float32Array, tensor: TensorInfo, startOutput: number, outputCount: number, rows: number) => Promise<Float32Array>) | undefined;
  multiplyStorageReferenceNativeBf16?: ((input: Float32Array, tensor: TensorInfo, startOutput: number, outputCount: number, rows: number) => Promise<Float32Array>) | undefined;
  multiplyWholeStorageReferenceNativeBf16?: ((input: Float32Array, tensor: TensorInfo, rows: number) => Promise<Float32Array>) | undefined;
  multiplyStorageReferences?: ((input: Float32Array, requests: readonly PagedLinearStorageReference[], rows: number) => Promise<readonly Float32Array[]>) | undefined;
  fusedGatedMlpStorageReference?: ((input: Float32Array, gate: TensorInfo, up: TensorInfo, down: TensorInfo, rows: number, rounding: "bf16" | "real" | "native-bf16") => Promise<Float32Array>) | undefined;
  fusedFfnStorageReferences?: ((request: PagedFusedFfnRequest) => Promise<Float32Array>) | undefined;
  fusedPleStorageReferences?: ((request: PagedFusedPleRequest) => Promise<Float32Array>) | undefined;
  fusedPlePreludeStorageReference?: ((request: PagedFusedPlePreludeRequest) => Promise<Float32Array>) | undefined;
  attention?: ((request: PagedNativeAttentionRequest) => Promise<Float32Array>) | undefined;
  fusedAttentionStorageReferences?: ((request: PagedFusedAttentionRequest) => Promise<PagedFusedAttentionResult>) | undefined;
  fusedDecoderLayerStorageReferences?: ((request: PagedFusedDecoderLayerRequest) => Promise<PagedFusedDecoderLayerResult>) | undefined;
}

export interface PagedFusedFfnRequest {
  input: Float32Array;
  preNormWeight: TensorInfo;
  gateWeight: TensorInfo;
  upWeight: TensorInfo;
  downWeight: TensorInfo;
  postNormWeight: TensorInfo;
  rows: number;
  hiddenSize: number;
  intermediateSize: number;
  preNormEpsilon: number;
  postNormEpsilon: number;
  rounding: "native-bf16";
}

export interface PagedFusedPleRequest {
  input: Float32Array;
  perLayerInput: Float32Array;
  gateWeight: TensorInfo;
  projectionWeight: TensorInfo;
  normWeight: TensorInfo;
  layerScalar: TensorInfo;
  rows: number;
  hiddenSize: number;
  perLayerWidth: number;
  epsilon: number;
  rounding: "bf16" | "real";
}

export interface PagedFusedPlePreludeRequest {
  input: Float32Array;
  tokenIdentity: Float32Array;
  projectionWeight: TensorInfo;
  normWeight: TensorInfo;
  rows: number;
  hiddenSize: number;
  numLayers: number;
  perLayerWidth: number;
  contextScale: number;
  combineScale: number;
  epsilon: number;
  maxReadBytes: number;
  rounding: "bf16" | "real";
}

export interface PagedNativeAttentionRequest {
  query: Float32Array;
  key: Float32Array;
  value: Float32Array;
  mask: Float32Array;
  batch: number;
  queryHeads: number;
  keyValueHeads: number;
  querySequence: number;
  keySequence: number;
  headDim: number;
  maskHeads: number;
  scale: number;
  rounding: "bf16" | "real";
}

export interface PagedFusedAttentionRequest {
  input: Float32Array;
  positions: Int32Array;
  mask: Float32Array;
  sourceKey: Float32Array;
  sourceValue: Float32Array;
  queryWeight: TensorInfo;
  queryNorm: TensorInfo;
  outputWeight: TensorInfo;
  keyWeight?: TensorInfo;
  keyNorm?: TensorInfo;
  valueWeight?: TensorInfo;
  batch: number;
  querySequence: number;
  sourceSequence: number;
  hiddenSize: number;
  queryHeads: number;
  keyValueHeads: number;
  headDim: number;
  maskHeads: number;
  producesKeyValue: boolean;
  valueFromKey: boolean;
  epsilon: number;
  scale: number;
  ropeType: "default" | "proportional";
  theta: number;
  rotaryDim: number;
  proportionalPairs: number;
  proportionalFactor: number;
  rounding: "bf16" | "real" | "native-bf16";
}

export interface PagedFusedAttentionResult {
  projected: Float32Array;
  key?: Float32Array;
  value?: Float32Array;
}

export interface PagedFusedDecoderLayerRequest extends PagedFusedAttentionRequest {
  perLayerInput: Float32Array;
  inputNormWeight: TensorInfo;
  postAttentionNormWeight: TensorInfo;
  preFfnNormWeight: TensorInfo;
  gateWeight: TensorInfo;
  upWeight: TensorInfo;
  downWeight: TensorInfo;
  postFfnNormWeight: TensorInfo;
  pleGateWeight: TensorInfo;
  pleProjectionWeight: TensorInfo;
  pleNormWeight: TensorInfo;
  layerScalar: TensorInfo;
  intermediateSize: number;
  perLayerWidth: number;
  inputNormEpsilon: number;
  postAttentionNormEpsilon: number;
  preFfnNormEpsilon: number;
  postFfnNormEpsilon: number;
  pleNormEpsilon: number;
  rounding: "native-bf16";
}

export interface PagedFusedDecoderLayerResult {
  hidden: Float32Array;
  key?: Float32Array;
  value?: Float32Array;
}

export interface PagedLinearStorageReference {
  tensor: TensorInfo;
  startOutput: number;
  outputCount: number;
}

/**
 * Materialize one complete dense literal tensor without consulting its source
 * container. This is intentionally bounded per tensor rather than per model:
 * multimodal feature programs can decode hundreds of modest parameters while
 * still rejecting a single unexpectedly large allocation.
 */
export async function readLiteralDenseF32Tensor(
  tensor: TensorInfo,
  reader: Pick<LiteralTensorReader, "readTensorBytesRange">,
  maxTensorBytes = 16 * 1024 * 1024,
): Promise<DenseF32Tensor> {
  if (!reader.readTensorBytesRange) throw new Error(`${tensor.name}: materialização literal requer readTensorBytesRange.`);
  if (tensor.quantization || (tensor.storageDtype !== "F32" && tensor.storageDtype !== "F16" && tensor.storageDtype !== "BF16") ||
    !sameShape(tensor.storageShape, tensor.logicalShape) || tensor.storageShape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 0)) {
    throw new Error(`${tensor.name}: materialização literal requer storage denso F32/F16/BF16 sem quantização.`);
  }
  const elements = tensor.storageShape.reduce((total, dimension) => total * dimension, 1);
  const byteLength = elements * (tensor.storageDtype === "F32" ? 4 : 2);
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || byteLength > maxTensorBytes) {
    throw new Error(`${tensor.name}: tensor de ${byteLength} bytes excede maxTensorBytes=${maxTensorBytes}.`);
  }
  const bytes = await reader.readTensorBytesRange(tensor, 0, byteLength);
  if (bytes.length !== byteLength) throw new Error(`${tensor.name}: leitor literal retornou ${bytes.length} bytes; esperados ${byteLength}.`);
  return { shape: [...tensor.logicalShape], values: decodeDenseRows(bytes, tensor.storageDtype) };
}

/**
 * Widens one bounded dense vector from declared literal storage. Vectors are
 * intentionally separate from matrices: norms and scalar tensors do not gain
 * an invented row-major matrix layout merely to reuse a projection kernel.
 */
export async function readPagedDenseF32Vector(
  tensor: TensorInfo,
  reader: Pick<LiteralTensorReader, "readTensorBytesRange">,
  maxReadBytes = 16 * 1024 * 1024,
): Promise<DenseF32Tensor> {
  if (!reader.readTensorBytesRange) throw new Error(`${tensor.name}: execução paginada requer readTensorBytesRange.`);
  if (tensor.quantization || (tensor.storageDtype !== "F32" && tensor.storageDtype !== "F16" && tensor.storageDtype !== "BF16") ||
    tensor.storageShape.length !== 1 || !sameShape(tensor.storageShape, tensor.logicalShape)) {
    throw new Error(`${tensor.name}: vetor paginado requer storage denso F32/F16/BF16 1-D sem quantização.`);
  }
  const elements = tensor.storageShape[0]!;
  const bytesPerElement = tensor.storageDtype === "F32" ? 4 : 2;
  const byteLength = elements * bytesPerElement;
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || byteLength > maxReadBytes) {
    throw new Error(`${tensor.name}: vetor de ${byteLength} bytes excede maxReadBytes=${maxReadBytes}.`);
  }
  const bytes = await reader.readTensorBytesRange(tensor, 0, byteLength);
  if (bytes.length !== byteLength) throw new Error(`${tensor.name}: leitor paginado retornou ${bytes.length} bytes; esperados ${byteLength}.`);
  return { shape: [elements], values: decodeDenseRows(bytes, tensor.storageDtype) };
}

export function createPagedDenseF32Matrix(
  tensor: TensorInfo,
  reader: Pick<LiteralTensorReader, "readTensorBytesRange">,
  maxReadBytes = 16 * 1024 * 1024,
): PagedDenseF32Matrix {
  if (!reader.readTensorBytesRange) throw new Error(`${tensor.name}: execução paginada requer readTensorBytesRange.`);
  if (tensor.quantization || (tensor.storageDtype !== "F32" && tensor.storageDtype !== "F16" && tensor.storageDtype !== "BF16") ||
    tensor.storageShape.length !== 2 || !sameShape(tensor.storageShape, tensor.logicalShape)) {
    throw new Error(`${tensor.name}: matriz paginada requer storage denso F32/F16/BF16 row-major 2-D sem quantização.`);
  }
  const storageDtype = tensor.storageDtype;
  const [rows, columns] = tensor.storageShape as [number, number];
  const elementBytes = tensor.storageDtype === "F32" ? 4 : 2;
  const rowBytes = columns * elementBytes;
  if (!Number.isSafeInteger(rowBytes) || rowBytes <= 0 || !Number.isSafeInteger(maxReadBytes) || maxReadBytes < rowBytes) {
    throw new Error(`${tensor.name}: maxReadBytes deve acomodar ao menos uma linha de ${rowBytes} bytes.`);
  }
  return {
    tensor,
    shape: [rows, columns],
    maxReadBytes,
    async readStorageRows(startRow, rowCount) {
      if (!Number.isInteger(startRow) || !Number.isInteger(rowCount) || startRow < 0 || rowCount <= 0 || startRow + rowCount > rows) {
        throw new Error(`${tensor.name}: intervalo de linhas inválido ${startRow}+${rowCount}.`);
      }
      const byteLength = rowCount * rowBytes;
      if (byteLength > maxReadBytes) throw new Error(`${tensor.name}: leitura de ${byteLength} bytes excede maxReadBytes=${maxReadBytes}.`);
      const bytes = await reader.readTensorBytesRange!(tensor, startRow * rowBytes, byteLength);
      if (bytes.length !== byteLength) throw new Error(`${tensor.name}: leitor paginado retornou ${bytes.length} bytes; esperados ${byteLength}.`);
      return bytes;
    },
    async readRows(startRow, rowCount) {
      const bytes = await this.readStorageRows!(startRow, rowCount);
      return { shape: [rowCount, columns], values: decodeDenseRows(bytes, storageDtype) };
    },
  };
}

/** Reads only the token rows needed by an embedding lookup. */
export async function pagedEmbeddingF32(
  inputIds: readonly number[][],
  weight: PagedDenseF32Matrix,
  scale = 1,
  options: { roundOutputToBf16?: boolean } = {},
): Promise<DenseF32Tensor> {
  const [vocab, hidden] = weight.shape;
  if (inputIds.length === 0 || inputIds.some((row) => row.length === 0 || row.length !== inputIds[0]!.length)) throw new Error("Embedding paginado requer batch não vazio e sequências iguais.");
  const result = new Float32Array(inputIds.length * inputIds[0]!.length * hidden);
  const rows = new Map<number, Float32Array>();
  const f32Scale = Math.fround(scale);
  for (let batch = 0; batch < inputIds.length; batch += 1) for (let sequence = 0; sequence < inputIds[batch]!.length; sequence += 1) {
    const token = inputIds[batch]![sequence]!;
    if (!Number.isInteger(token) || token < 0 || token >= vocab) throw new Error(`Token fora do vocabulário: ${token}.`);
    let values = rows.get(token);
    if (!values) { values = (await weight.readRows(token, 1)).values; rows.set(token, values); }
    const offset = (batch * inputIds[batch]!.length + sequence) * hidden;
    for (let column = 0; column < hidden; column += 1) {
      const product = Math.fround(Math.fround(values[column]!) * f32Scale);
      result[offset + column] = options.roundOutputToBf16 ? roundF32ToBF16(product) : product;
    }
  }
  return { shape: [inputIds.length, inputIds[0]!.length, hidden], values: result };
}

/**
 * F32 scalar linear algebra over bounded literal-artifact row ranges. Its
 * accumulation order is identical to linearF32: output row, then input
 * column; only storage acquisition is paged.
 */
export async function pagedLinearF32(
  input: DenseF32Tensor,
  weight: PagedDenseF32Matrix,
  options: {
    outputDtype?: "F32" | "BF16";
    accumulationDtype?: "F32" | "F64";
    reduction?: ReductionSchedule;
    tileKernel?: PagedLinearTileKernel;
    nativeBf16?: boolean;
    wholeNativeBf16?: boolean;
  } = {},
): Promise<DenseF32Tensor> {
  if (input.shape.length < 1) throw new Error("Linear paginado requer entrada com dimensão de features.");
  const [outFeatures, inFeatures] = weight.shape;
  if (input.shape.at(-1) !== inFeatures) throw new Error(`Linear paginado: entrada ${input.shape.at(-1)} incompatível com weight ${outFeatures}x${inFeatures}.`);
  const rows = input.values.length / inFeatures;
  const result = new Float32Array(rows * outFeatures);
  if (options.wholeNativeBf16) {
    if (!options.nativeBf16) throw new Error("GEMM BF16 integral requer nativeBf16.");
    if (!options.tileKernel?.multiplyWholeStorageReferenceNativeBf16) throw new Error(`${options.tileKernel?.backend ?? "kernel ausente"}: kernel não oferece GEMM BF16 integral.`);
    const output = await options.tileKernel.multiplyWholeStorageReferenceNativeBf16(input.values, weight.tensor, rows);
    if (output.length !== result.length || output.some((value) => !Number.isFinite(value))) throw new Error(`${options.tileKernel.backend}: kernel linear integral retornou saída inválida.`);
    for (let index = 0; index < output.length; index += 1) result[index] = options.outputDtype === "BF16" ? roundF32ToBF16(output[index]!) : Math.fround(output[index]!);
    return { shape: [...input.shape.slice(0, -1), outFeatures], values: result };
  }
  const rowBytes = inFeatures * (weight.tensor.storageDtype === "F32" ? 4 : 2);
  const chunkRows = Math.max(1, Math.floor(weight.maxReadBytes / rowBytes));
  for (let firstOutput = 0; firstOutput < outFeatures; firstOutput += chunkRows) {
    const outputCount = Math.min(chunkRows, outFeatures - firstOutput);
    if (options.tileKernel) {
      if (options.nativeBf16 && !options.tileKernel.multiplyStorageReferenceNativeBf16) throw new Error(`${options.tileKernel.backend}: kernel não oferece GEMM BF16 nativo.`);
      const tile = options.nativeBf16
        ? await options.tileKernel.multiplyStorageReferenceNativeBf16!(input.values, weight.tensor, firstOutput, outputCount, rows)
        : options.tileKernel.multiplyStorageReference
        ? await options.tileKernel.multiplyStorageReference(input.values, weight.tensor, firstOutput, outputCount, rows)
        : options.tileKernel.multiplyStorage && weight.readStorageRows
        ? await options.tileKernel.multiplyStorage(input.values, await weight.readStorageRows(firstOutput, outputCount), weight.tensor.storageDtype as "F32" | "F16" | "BF16", rows, outputCount, inFeatures)
        : await options.tileKernel.multiply(input.values, (await weight.readRows(firstOutput, outputCount)).values, rows, outputCount, inFeatures);
      if (tile.length !== rows * outputCount || tile.some((value) => !Number.isFinite(value))) throw new Error(`${options.tileKernel.backend}: kernel linear retornou tile inválido.`);
      for (let row = 0; row < rows; row += 1) for (let output = 0; output < outputCount; output += 1) {
        const value = tile[row * outputCount + output]!;
        result[row * outFeatures + firstOutput + output] = options.outputDtype === "BF16" ? roundF32ToBF16(value) : Math.fround(value);
      }
      continue;
    }
    const stored = await weight.readRows(firstOutput, outputCount);
    for (let row = 0; row < rows; row += 1) for (let output = 0; output < outputCount; output += 1) {
      const sum = options.reduction?.kind === "ordered-fma"
        ? linearF32ProductsOrderedFma(input, stored.values, row, output, inFeatures)
        : options.reduction?.kind === "blocked-f32-terms"
          ? linearF32ProductsBlockedTerms(input, stored.values, row, output, inFeatures, options.reduction)
          : options.reduction?.kind === "blocked-tiled-f32-lanes"
            ? linearF32ProductsBlockedTiledLanes(input, stored.values, row, output, inFeatures, options.reduction)
            : options.reduction?.kind === "arm-neon-bf16-dot-fma"
              ? linearF32ProductsArmNeonBf16DotFma(input, stored.values, row, output, inFeatures, options.reduction)
              : options.reduction?.kind === "arm-neon-bf16-bfdot-fma"
                ? linearF32ProductsArmNeonBf16BfdotFma(input, stored.values, row, output, inFeatures, options.reduction)
          : options.reduction?.kind === "interleaved-f32-lanes" || options.reduction?.kind === "interleaved-fma-lanes" ||
        options.reduction?.kind === "tiled-f32-lanes" || options.reduction?.kind === "tiled-fma-lanes"
            ? linearF32ProductsInterleavedF32Lanes(input, stored.values, row, output, inFeatures, options.reduction)
            : options.accumulationDtype === "F64"
              ? linearF32ProductsF64Accumulation(input, stored.values, row, output, inFeatures)
              : linearF32ProductsF32Accumulation(input, stored.values, row, output, inFeatures);
      result[row * outFeatures + firstOutput + output] = options.outputDtype === "BF16" ? roundF32ToBF16(sum) : Math.fround(sum);
    }
  }
  return { shape: [...input.shape.slice(0, -1), outFeatures], values: result };
}

/**
 * Executes equal-shaped projections that consume the same input through one
 * referenced native dispatch per output tile. The outputs stay independent;
 * only transport and kernel scheduling are fused.
 */
export async function pagedLinearBatchF32(
  input: DenseF32Tensor,
  weights: readonly PagedDenseF32Matrix[],
  options: readonly { outputDtype?: "F32" | "BF16"; tileKernel: PagedLinearTileKernel }[],
): Promise<readonly DenseF32Tensor[]> {
  if (weights.length < 2 || options.length !== weights.length) throw new Error("Linear paginado em lote requer ao menos duas matrizes e uma política por matriz.");
  const kernel = options[0]!.tileKernel;
  if (!kernel.multiplyStorageReferences || options.some((entry) => entry.tileKernel !== kernel)) throw new Error("Linear paginado em lote requer um único kernel referenciado com suporte a lote.");
  const [outFeatures, inFeatures] = weights[0]!.shape;
  if (input.shape.length < 1 || input.shape.at(-1) !== inFeatures) throw new Error("Linear paginado em lote recebeu entrada incompatível.");
  if (weights.some((weight) => weight.shape[0] !== outFeatures || weight.shape[1] !== inFeatures || weight.maxReadBytes !== weights[0]!.maxReadBytes)) throw new Error("Linear paginado em lote requer matrizes com shape e paginação idênticos.");
  const rows = input.values.length / inFeatures;
  const results = weights.map(() => new Float32Array(rows * outFeatures));
  const storageBytes = (weight: PagedDenseF32Matrix) => weight.tensor.storageDtype === "F32" ? 4 : 2;
  const elementBytes = storageBytes(weights[0]!);
  if (weights.some((weight) => storageBytes(weight) !== elementBytes)) throw new Error("Linear paginado em lote requer largura de armazenamento idêntica.");
  const chunkRows = Math.max(1, Math.floor(weights[0]!.maxReadBytes / (inFeatures * elementBytes)));
  for (let firstOutput = 0; firstOutput < outFeatures; firstOutput += chunkRows) {
    const outputCount = Math.min(chunkRows, outFeatures - firstOutput);
    const tiles = await kernel.multiplyStorageReferences(input.values, weights.map((weight) => ({ tensor: weight.tensor, startOutput: firstOutput, outputCount })), rows);
    if (tiles.length !== weights.length) throw new Error(`${kernel.backend}: kernel linear em lote retornou ${tiles.length} tiles; esperados ${weights.length}.`);
    for (let projection = 0; projection < tiles.length; projection += 1) {
      const tile = tiles[projection]!;
      if (tile.length !== rows * outputCount || tile.some((value) => !Number.isFinite(value))) throw new Error(`${kernel.backend}: kernel linear em lote retornou tile inválido.`);
      const result = results[projection]!, outputDtype = options[projection]!.outputDtype;
      for (let row = 0; row < rows; row += 1) for (let output = 0; output < outputCount; output += 1) {
        const value = tile[row * outputCount + output]!;
        result[row * outFeatures + firstOutput + output] = outputDtype === "BF16" ? roundF32ToBF16(value) : Math.fround(value);
      }
    }
  }
  return results.map((values) => ({ shape: [...input.shape.slice(0, -1), outFeatures], values }));
}

/** Ordered F32 products and F32 additions: the generic scalar reference contract. */
function linearF32ProductsF32Accumulation(input: DenseF32Tensor, weight: Float32Array, row: number, output: number, inFeatures: number): number {
  let sum = Math.fround(0);
  for (let column = 0; column < inFeatures; column += 1) {
    sum = Math.fround(sum + Math.fround(input.values[row * inFeatures + column]! * weight[output * inFeatures + column]!));
  }
  return sum;
}

/** Ordered F32 FMA boundaries: products stay exact until each accumulator add. */
function linearF32ProductsOrderedFma(input: DenseF32Tensor, weight: Float32Array, row: number, output: number, inFeatures: number): number {
  let sum = Math.fround(0);
  for (let column = 0; column < inFeatures; column += 1) {
    sum = Math.fround(sum + input.values[row * inFeatures + column]! * weight[output * inFeatures + column]!);
  }
  return sum;
}

/**
 * Keeps adjacent-term dot-product partials explicit.  A block is not a lane:
 * the completed partial is rounded, then added to the one scalar accumulator.
 */
function linearF32ProductsBlockedTerms(
  input: DenseF32Tensor,
  weight: Float32Array,
  row: number,
  output: number,
  inFeatures: number,
  reduction: Extract<ReductionSchedule, { kind: "blocked-f32-terms" }>,
): number {
  if (!Number.isSafeInteger(reduction.termsPerBlock) || reduction.termsPerBlock < 2 ||
    reduction.inputBlock !== "contiguous-terms" || reduction.blockOrder !== "ascending" ||
    (reduction.termOrder !== "ascending" && reduction.termOrder !== "descending") ||
    (reduction.productBoundary !== "separately-rounded-f32" && reduction.productBoundary !== "fused-fma")) {
    throw new Error("Linear paginado recebeu agenda de blocos F32 inválida.");
  }
  let sum = Math.fround(0);
  for (let first = 0; first < inFeatures; first += reduction.termsPerBlock) {
    const last = Math.min(first + reduction.termsPerBlock, inFeatures);
    let partial = Math.fround(0);
    for (let offset = 0; offset < last - first; offset += 1) {
      const column = reduction.termOrder === "ascending" ? first + offset : last - 1 - offset;
      const product = input.values[row * inFeatures + column]! * weight[output * inFeatures + column]!;
      partial = reduction.productBoundary === "fused-fma"
        ? Math.fround(partial + product)
        : Math.fround(partial + Math.fround(product));
    }
    sum = Math.fround(sum + partial);
  }
  return sum;
}

/**
 * Reduces each finite contiguous SIMD-like tile before advancing the scalar
 * accumulator.  Keeping tile lanes local is materially distinct from the
 * persistent-lane tiled schedules: the latter can carry cancellation across
 * tiles, while this schedule cannot.
 */
function linearF32ProductsBlockedTiledLanes(
  input: DenseF32Tensor,
  weight: Float32Array,
  row: number,
  output: number,
  inFeatures: number,
  reduction: Extract<ReductionSchedule, { kind: "blocked-tiled-f32-lanes" }>,
): number {
  if (!Number.isSafeInteger(reduction.laneCount) || reduction.laneCount < 2 ||
    !Number.isSafeInteger(reduction.termsPerLane) || reduction.termsPerLane < 2 ||
    reduction.inputBlock !== "tile-contiguous-terms" || reduction.blockOrder !== "ascending" ||
    (reduction.laneReductionOrder !== "ascending" && reduction.laneReductionOrder !== "descending" && reduction.laneReductionOrder !== "balanced-pairwise") ||
    (reduction.productBoundary !== "separately-rounded-f32" && reduction.productBoundary !== "fused-fma")) {
    throw new Error("Linear paginado recebeu agenda de blocos tiled F32 inválida.");
  }
  const tileWidth = reduction.laneCount * reduction.termsPerLane;
  let sum = Math.fround(0);
  for (let first = 0; first < inFeatures; first += tileWidth) {
    const lanes = new Float32Array(reduction.laneCount);
    const last = Math.min(first + tileWidth, inFeatures);
    for (let column = first; column < last; column += 1) {
      const lane = Math.floor((column - first) / reduction.termsPerLane);
      const product = input.values[row * inFeatures + column]! * weight[output * inFeatures + column]!;
      lanes[lane] = reduction.productBoundary === "fused-fma"
        ? Math.fround(lanes[lane]! + product)
        : Math.fround(lanes[lane]! + Math.fround(product));
    }
    sum = Math.fround(sum + foldF32Lanes(lanes, reduction.laneReductionOrder));
  }
  return sum;
}

/**
 * Ordered F32 products with an F64 scalar accumulator. This is deliberately
 * separate from storage widening: callers must declare it in the operation
 * dtype policy, because it changes an observable BF16 result at cancellation
 * boundaries.
 */
function linearF32ProductsF64Accumulation(input: DenseF32Tensor, weight: Float32Array, row: number, output: number, inFeatures: number): number {
  let sum = 0;
  for (let column = 0; column < inFeatures; column += 1) {
    sum += Math.fround(input.values[row * inFeatures + column]! * weight[output * inFeatures + column]!);
  }
  return sum;
}

/**
 * Replays the finite scalar schedule used by the ARM BF16 GEMV source path:
 * eight F32 vector registers of the declared width, FMA accumulation by
 * coordinate modulo the declared lane count, followed by its non-adjacent
 * register tree. `horizontalFold`
 * remains explicit because the final vector horizontal reduction is a
 * separate observable rounding boundary.
 */
function linearF32ProductsArmNeonBf16DotFma(
  input: DenseF32Tensor,
  weight: Float32Array,
  row: number,
  output: number,
  inFeatures: number,
  reduction: Extract<ReductionSchedule, { kind: "arm-neon-bf16-dot-fma" }>,
): number {
  const inputBase = row * inFeatures;
  const weightBase = output * inFeatures;
  return armNeonBf16DotF32(
    inFeatures,
    (column) => input.values[inputBase + column]!,
    (column) => weight[weightBase + column]!,
    reduction,
  );
}

/**
 * Replays PyTorch's BFDOT-specific branch rather than treating it as a
 * generic modulo-lane reduction. A BFDOT instruction performs two adjacent
 * BF16 products into one F32 lane; the source uses four active registers per
 * 32 input terms and reduces all eight registers afterwards.
 */
function linearF32ProductsArmNeonBf16BfdotFma(
  input: DenseF32Tensor,
  weight: Float32Array,
  row: number,
  output: number,
  inFeatures: number,
  reduction: Extract<ReductionSchedule, { kind: "arm-neon-bf16-bfdot-fma" }>,
): number {
  if (reduction.registerCount !== 8 || reduction.activeRegisterCount !== 4 || reduction.lanesPerRegister !== 4 ||
    reduction.termsPerLane !== 2 || reduction.termsPerInstruction !== 8 || reduction.inputLane !== "contiguous-bf16-pairs" ||
    (reduction.horizontalFold !== "ascending" && reduction.horizontalFold !== "pairwise")) {
    throw new Error("Linear paginado recebeu agenda ARM NEON BFDOT BF16 inválida.");
  }
  const registers = new Float32Array(reduction.registerCount * reduction.lanesPerRegister);
  const base = row * inFeatures;
  const weightBase = output * inFeatures;
  for (let column = 0; column < inFeatures; column += reduction.termsPerLane) {
    const withinIteration = column % (reduction.activeRegisterCount * reduction.termsPerInstruction);
    const register = Math.floor(withinIteration / reduction.termsPerInstruction);
    const lane = Math.floor((withinIteration % reduction.termsPerInstruction) / reduction.termsPerLane);
    const index = register * reduction.lanesPerRegister + lane;
    let accumulator = registers[index]!;
    for (let term = 0; term < reduction.termsPerLane && column + term < inFeatures; term += 1) {
      accumulator = Math.fround(accumulator + input.values[base + column + term]! * weight[weightBase + column + term]!);
    }
    registers[index] = accumulator;
  }
  // VectorizedN<float, 8>: x[0..3] += x[4..7], x[0..1] += x[2..3], then x[0] += x[1].
  for (let register = 0; register < 4; register += 1) for (let lane = 0; lane < reduction.lanesPerRegister; lane += 1) {
    const index = register * reduction.lanesPerRegister + lane;
    registers[index] = Math.fround(registers[index]! + registers[(register + 4) * reduction.lanesPerRegister + lane]!);
  }
  for (let register = 0; register < 2; register += 1) for (let lane = 0; lane < reduction.lanesPerRegister; lane += 1) {
    const index = register * reduction.lanesPerRegister + lane;
    registers[index] = Math.fround(registers[index]! + registers[(register + 2) * reduction.lanesPerRegister + lane]!);
  }
  for (let lane = 0; lane < reduction.lanesPerRegister; lane += 1) {
    registers[lane] = Math.fround(registers[lane]! + registers[reduction.lanesPerRegister + lane]!);
  }
  return reduction.horizontalFold === "ascending"
    ? foldF32Lanes(registers.subarray(0, reduction.lanesPerRegister), "ascending")
    : foldF32Lanes(registers.subarray(0, reduction.lanesPerRegister), "balanced-pairwise");
}

/** Declared F32 lane reductions are explicit runtime profiles, never shape heuristics. */
function linearF32ProductsInterleavedF32Lanes(
  input: DenseF32Tensor,
  weight: Float32Array,
  row: number,
  output: number,
  inFeatures: number,
  reduction: Extract<ReductionSchedule, { kind: "interleaved-f32-lanes" | "interleaved-fma-lanes" | "tiled-f32-lanes" | "tiled-fma-lanes" }>,
): number {
  if (!Number.isSafeInteger(reduction.laneCount) || reduction.laneCount < 2 ||
    (reduction.laneReductionOrder !== "ascending" && reduction.laneReductionOrder !== "descending" && reduction.laneReductionOrder !== "balanced-pairwise")) {
    throw new Error("Linear paginado recebeu agenda de lanes F32 inválida.");
  }
  const tiled = reduction.kind === "tiled-f32-lanes" || reduction.kind === "tiled-fma-lanes";
  if ((!tiled && reduction.inputLane !== "index-modulo-lane-count") ||
    (tiled && (!Number.isSafeInteger(reduction.termsPerLane) || reduction.termsPerLane < 2 || reduction.inputLane !== "tile-contiguous-terms"))) {
    throw new Error("Linear paginado recebeu mapeamento de lanes F32 inválido.");
  }
  const lanes = new Float32Array(reduction.laneCount);
  for (let column = 0; column < inFeatures; column += 1) {
    const lane = tiled
      ? Math.floor((column % (reduction.laneCount * reduction.termsPerLane)) / reduction.termsPerLane)
      : column % reduction.laneCount;
    const product = input.values[row * inFeatures + column]! * weight[output * inFeatures + column]!;
    lanes[lane] = reduction.kind === "interleaved-fma-lanes" || reduction.kind === "tiled-fma-lanes"
      ? Math.fround(lanes[lane]! + product)
      : Math.fround(lanes[lane]! + Math.fround(product));
  }
  return foldF32Lanes(lanes, reduction.laneReductionOrder);
}

/**
 * Keep the horizontal SIMD fold explicit.  This is deliberately not a host
 * reduction: every F32 addition and its tree/order are part of the literal
 * calculation program and can therefore be audited or probed independently.
 */
function foldF32Lanes(lanes: Float32Array, order: Extract<ReductionSchedule, { kind: "interleaved-f32-lanes" | "interleaved-fma-lanes" | "tiled-f32-lanes" | "tiled-fma-lanes" }>['laneReductionOrder']): number {
  if (order === "ascending" || order === "descending") {
    let sum = Math.fround(0);
    const start = order === "ascending" ? 0 : lanes.length - 1;
    const end = order === "ascending" ? lanes.length : -1;
    const step = order === "ascending" ? 1 : -1;
    for (let lane = start; lane !== end; lane += step) sum = Math.fround(sum + lanes[lane]!);
    return sum;
  }
  let current = lanes;
  while (current.length > 1) {
    const next = new Float32Array(Math.ceil(current.length / 2));
    for (let index = 0; index < current.length; index += 2) {
      next[index / 2] = index + 1 < current.length ? Math.fround(current[index]! + current[index + 1]!) : current[index]!;
    }
    current = next;
  }
  return current[0]!;
}

/** Applies an explicit tensor-result BF16 cast after a declared operation. */
export function roundDenseF32ToBF16(tensor: DenseF32Tensor): DenseF32Tensor {
  const values = new Float32Array(tensor.values.length);
  for (let index = 0; index < values.length; index += 1) values[index] = roundF32ToBF16(tensor.values[index]!);
  return { shape: [...tensor.shape], values };
}

function decodeDenseRows(bytes: Buffer, dtype: "F32" | "F16" | "BF16"): Float32Array {
  const values = new Float32Array(bytes.length / (dtype === "F32" ? 4 : 2));
  for (let index = 0; index < values.length; index += 1) {
    values[index] = dtype === "F32" ? bytes.readFloatLE(index * 4) :
      dtype === "F16" ? decodeIeeeF16ToF32(bytes.readUInt16LE(index * 2)) : decodeIeeeBF16ToF32(bytes.readUInt16LE(index * 2));
  }
  return values;
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
