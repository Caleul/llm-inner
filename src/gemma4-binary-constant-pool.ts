import { SafetensorsCatalogReader } from "./safetensors.js";
import type { LiteralTensorReader } from "./literal.js";
import type { ModelCatalog, TensorInfo } from "./types.js";

/** Indexed binary view of the compiled constants; tensor identity is checked on every read. */
export class Gemma4BinaryConstantPool implements Pick<LiteralTensorReader, "readTensorBytesRange"> {
  readonly backend = "indexed-safetensors-binary-pool";
  readonly reader: SafetensorsCatalogReader;
  readonly catalog: ModelCatalog;

  private constructor(reader: SafetensorsCatalogReader, catalog: ModelCatalog) { this.reader = reader; this.catalog = catalog; }

  static async open(directory: string): Promise<Gemma4BinaryConstantPool> {
    const reader = new SafetensorsCatalogReader(directory);
    try { return new Gemma4BinaryConstantPool(reader, await reader.inspect()); }
    catch (error) { await reader.close(); throw error; }
  }

  async readTensorBytesRange(tensor: TensorInfo, offset: number, byteLength: number): Promise<Buffer> {
    const stored = this.catalog.tensors.get(tensor.name);
    if (!stored || stored.storageDtype !== tensor.storageDtype || stored.logicalShape.length !== tensor.logicalShape.length || stored.logicalShape.some((value, index) => value !== tensor.logicalShape[index])) {
      throw new Error(`${tensor.name}: constant pool binário diverge do programa literal.`);
    }
    return this.reader.readTensorBytesRange(stored, offset, byteLength);
  }

  close(): Promise<void> { return this.reader.close(); }
}
