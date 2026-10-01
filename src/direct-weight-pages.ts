import { SafetensorsCatalogReader } from './safetensors.js';
import type { TensorInfo } from './types.js';
import { decodeIeeeF16ToF32 } from './utils.js';

/** Bounded LRU of original checkpoint bytes, only during compilation. No
 * decoded matrix, scalar result or activation is retained. Zero disables it.
 * The caller must keep the checkpoint immutable for the entire compilation.
 */
export class DirectWeightPages {
  private readonly pages = new Map<string, Buffer>();
  retainedBytes = 0;
  reads = 0;
  hits = 0;
  constructor(private readonly reader: Pick<SafetensorsCatalogReader, "readTensorBytesRange">,
    readonly budgetBytes = 16 * 1024 * 1024, readonly pageBytes = 65536) {
    if (!Number.isSafeInteger(budgetBytes) || budgetBytes < 0 ||
        !Number.isSafeInteger(pageBytes) || pageBytes < 2 || pageBytes % 2 !== 0)
      throw new RangeError('Invalid weight page budget');
  }
  clear(): void { this.pages.clear(); this.retainedBytes = 0; }
  async read(tensor: TensorInfo, index: number): Promise<number> {
    if (tensor.storageDtype !== 'F16' || !Number.isSafeInteger(index) || index < 0 ||
        tensor.byteLength === undefined || index * 2 + 2 > tensor.byteLength)
      throw new RangeError('Invalid finite F16 weight coordinate');
    const offset = index * 2;
    const start = this.budgetBytes < this.pageBytes ? offset : Math.floor(offset / this.pageBytes) * this.pageBytes;
    const length = this.budgetBytes < this.pageBytes ? 2 : Math.min(this.pageBytes, tensor.byteLength - start);
    const key = `${tensor.shard}:${tensor.byteOffset}:${tensor.byteLength}:${start}`;
    let bytes = this.pages.get(key);
    if (bytes) { this.hits++; this.pages.delete(key); this.pages.set(key, bytes); }
    else {
      this.reads++;
      bytes = await this.reader.readTensorBytesRange(tensor, start, length);
      if (this.budgetBytes >= this.pageBytes) {
        while (this.retainedBytes + bytes.length > this.budgetBytes) {
          const oldest = this.pages.keys().next().value as string | undefined;
          if (oldest === undefined) break;
          this.retainedBytes -= this.pages.get(oldest)!.length; this.pages.delete(oldest);
        }
        this.pages.set(key, bytes); this.retainedBytes += bytes.length;
      }
    }
    const bits = bytes.readUInt16LE(offset - start);
    if ((bits & 0x7c00) === 0x7c00) throw new RangeError('Nonfinite checkpoint weight');
    return decodeIeeeF16ToF32(bits);
  }
}
