import { decodeIeeeBF16ToF32, decodeIeeeF16ToF32 } from "./utils.js";
import type { LiteralTensorReader } from "./literal.js";
import type { DenseF32Tensor, TensorInfo } from "./types.js";

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
    async readRows(startRow, rowCount) {
      if (!Number.isInteger(startRow) || !Number.isInteger(rowCount) || startRow < 0 || rowCount <= 0 || startRow + rowCount > rows) {
        throw new Error(`${tensor.name}: intervalo de linhas inválido ${startRow}+${rowCount}.`);
      }
      const byteLength = rowCount * rowBytes;
      if (byteLength > maxReadBytes) throw new Error(`${tensor.name}: leitura de ${byteLength} bytes excede maxReadBytes=${maxReadBytes}.`);
      const bytes = await reader.readTensorBytesRange!(tensor, startRow * rowBytes, byteLength);
      if (bytes.length !== byteLength) throw new Error(`${tensor.name}: leitor paginado retornou ${bytes.length} bytes; esperados ${byteLength}.`);
      return { shape: [rowCount, columns], values: decodeDenseRows(bytes, storageDtype) };
    },
  };
}

/** Reads only the token rows needed by an embedding lookup. */
export async function pagedEmbeddingF32(inputIds: readonly number[][], weight: PagedDenseF32Matrix, scale = 1): Promise<DenseF32Tensor> {
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
    for (let column = 0; column < hidden; column += 1) result[offset + column] = Math.fround(Math.fround(values[column]!) * f32Scale);
  }
  return { shape: [inputIds.length, inputIds[0]!.length, hidden], values: result };
}

/**
 * F32 scalar linear algebra over bounded literal-artifact row ranges. Its
 * accumulation order is identical to linearF32: output row, then input
 * column; only storage acquisition is paged.
 */
export async function pagedLinearF32(input: DenseF32Tensor, weight: PagedDenseF32Matrix): Promise<DenseF32Tensor> {
  if (input.shape.length < 1) throw new Error("Linear paginado requer entrada com dimensão de features.");
  const [outFeatures, inFeatures] = weight.shape;
  if (input.shape.at(-1) !== inFeatures) throw new Error(`Linear paginado: entrada ${input.shape.at(-1)} incompatível com weight ${outFeatures}x${inFeatures}.`);
  const rows = input.values.length / inFeatures;
  const result = new Float32Array(rows * outFeatures);
  const rowBytes = inFeatures * (weight.tensor.storageDtype === "F32" ? 4 : 2);
  const chunkRows = Math.max(1, Math.floor(weight.maxReadBytes / rowBytes));
  for (let firstOutput = 0; firstOutput < outFeatures; firstOutput += chunkRows) {
    const outputCount = Math.min(chunkRows, outFeatures - firstOutput);
    const stored = await weight.readRows(firstOutput, outputCount);
    for (let row = 0; row < rows; row += 1) for (let output = 0; output < outputCount; output += 1) {
      let sum = Math.fround(0);
      for (let column = 0; column < inFeatures; column += 1) sum = Math.fround(sum + Math.fround(input.values[row * inFeatures + column]! * stored.values[output * inFeatures + column]!));
      result[row * outFeatures + firstOutput + output] = sum;
    }
  }
  return { shape: [...input.shape.slice(0, -1), outFeatures], values: result };
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
