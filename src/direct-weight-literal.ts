import { SafetensorsCatalogReader } from "./safetensors.js";
import type { TensorInfo } from "./types.js";
import { f16BitsToDyadic } from "./fixed-f16-projection.js";

/** Read exactly one reached weight. No matrix or weight cache is materialized. */
export async function readDirectF16Literal(reader: SafetensorsCatalogReader, tensor: TensorInfo,
  index: number): Promise<string> {
  if (tensor.storageDtype !== "F16" || !Number.isSafeInteger(index) || index < 0) {
    throw new RangeError("Direct literal requires a valid F16 coordinate");
  }
  const bytes = await reader.readTensorBytesRange(tensor, index * 2, 2);
  const bits = bytes.readUInt16LE();
  if ((bits & 0x7c00) === 0x7c00) throw new RangeError("Nonfinite checkpoint weight has no declared semantics");
  if (bits === 32768) return "-0";
  const value = f16BitsToDyadic(bits);
  return String(Number(value.coefficient) * 2 ** value.exponent);
}
