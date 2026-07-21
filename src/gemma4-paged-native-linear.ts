import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { once } from "node:events";
import { endianness } from "node:os";
import { resolve } from "node:path";
import type { PagedLinearStorageReference, PagedLinearTileKernel } from "./paged-dense.js";
import type { TensorInfo } from "./types.js";

const STORAGE_REFERENCE_FLAG = 0x8000_0000;
const STORAGE_REFERENCE_BATCH_FLAG = 0x4000_0000;

export class Gemma4PagedNativeLinearWorker implements PagedLinearTileKernel {
  readonly backend: string;
  readonly child: ChildProcessWithoutNullStreams;
  readonly multiplyStorageReference?: PagedLinearTileKernel["multiplyStorageReference"];
  readonly multiplyStorageReferences?: PagedLinearTileKernel["multiplyStorageReferences"];
  readonly #storageTensors: ReadonlyMap<string, TensorInfo> | undefined;
  #buffer = Buffer.alloc(0);
  #waiting: Array<() => void> = [];
  #closedError?: Error;
  #active = false;
  #referenceDispatches = 0;
  #batchDispatches = 0;
  #batchedProjectionTiles = 0;

  constructor(options: { python: string; helper: string; threads: number; binaryPool?: string; storageTensors?: ReadonlyMap<string, TensorInfo>; backend?: "pytorch" | "mlx"; mlxHelper?: string }) {
    if (endianness() !== "LE") throw new Error("Kernel linear binário requer host little-endian.");
    if (!Number.isSafeInteger(options.threads) || options.threads < 1) throw new Error("threads do kernel linear deve ser positivo.");
    if ((options.binaryPool === undefined) !== (options.storageTensors === undefined)) throw new Error("Kernel linear mmap requer pool binário e catálogo juntos.");
    this.#storageTensors = options.storageTensors;
    const backend = options.backend ?? "pytorch";
    if (backend === "mlx" && !options.binaryPool) throw new Error("Kernel MLX requer pool binário referenciado.");
    const helper = backend === "mlx" ? options.mlxHelper : options.helper;
    if (!helper) throw new Error(`Kernel ${backend} requer helper executável.`);
    this.backend = backend === "mlx" ? "persistent-mlx-metal-mmap-f32-tile" : options.binaryPool ? "persistent-pytorch-mmap-f32-tile" : "persistent-pytorch-f32-tile";
    const arguments_ = [helper, "--threads", String(options.threads), ...(options.binaryPool ? ["--binary-pool", resolve(options.binaryPool)] : [])];
    this.child = spawn(options.python, arguments_, { stdio: ["pipe", "pipe", "pipe"] });
    if (options.binaryPool) {
      this.multiplyStorageReference = (input, tensor, startOutput, outputCount, rows) => this.#requestReference(input, tensor, startOutput, outputCount, rows);
      this.multiplyStorageReferences = (input, requests, rows) => this.#requestReferences(input, requests, rows);
    }
    this.child.stdout.on("data", (chunk: Buffer) => { this.#buffer = Buffer.concat([this.#buffer, chunk]); this.#wake(); });
    const errors: Buffer[] = []; this.child.stderr.on("data", (chunk: Buffer) => { if (Buffer.concat(errors).length < 1024 * 1024) errors.push(chunk); });
    this.child.once("error", (error) => this.#fail(error));
    this.child.once("close", (code) => this.#fail(new Error(`Worker linear encerrou com código ${code}: ${Buffer.concat(errors).toString("utf8").trim()}`)));
  }

  async multiply(input: Float32Array, weight: Float32Array, rows: number, outputCount: number, inFeatures: number): Promise<Float32Array> {
    return this.#request(input, Buffer.from(weight.buffer, weight.byteOffset, weight.byteLength), 0, rows, outputCount, inFeatures);
  }

  async multiplyStorage(input: Float32Array, weight: Buffer, storageDtype: "F32" | "F16" | "BF16", rows: number, outputCount: number, inFeatures: number): Promise<Float32Array> {
    const dtype = storageDtype === "F32" ? 0 : storageDtype === "BF16" ? 1 : 2;
    const expected = outputCount * inFeatures * (storageDtype === "F32" ? 4 : 2);
    if (weight.length !== expected) throw new Error(`Tile ${storageDtype} possui ${weight.length} bytes; esperados ${expected}.`);
    return this.#request(input, weight, dtype, rows, outputCount, inFeatures);
  }

  async #request(input: Float32Array, weight: Buffer, dtype: number, rows: number, outputCount: number, inFeatures: number): Promise<Float32Array> {
    if (this.#active) throw new Error("Worker linear persistente não aceita tiles concorrentes no mesmo canal.");
    if (input.length !== rows * inFeatures) throw new Error("Tile linear possui shape incompatível.");
    this.#active = true;
    try {
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(rows, 0); header.writeUInt32LE(outputCount, 4); header.writeUInt32LE(inFeatures, 8); header.writeUInt32LE(dtype, 12);
      await this.#write(header); await this.#write(Buffer.from(input.buffer, input.byteOffset, input.byteLength)); await this.#write(weight);
      return await this.#readResult(rows, outputCount);
    } finally { this.#active = false; }
  }

  async #requestReference(input: Float32Array, tensor: TensorInfo, startOutput: number, outputCount: number, rows: number): Promise<Float32Array> {
    if (this.#active) throw new Error("Worker linear persistente não aceita tiles concorrentes no mesmo canal.");
    const stored = this.#storageTensors?.get(tensor.name);
    if (!stored || stored.storageDtype !== tensor.storageDtype || stored.storageShape.length !== tensor.storageShape.length || stored.storageShape.some((value, index) => value !== tensor.storageShape[index]) || stored.logicalShape.length !== tensor.logicalShape.length || stored.logicalShape.some((value, index) => value !== tensor.logicalShape[index])) throw new Error(`${tensor.name}: catálogo mmap diverge do operando literal.`);
    if (stored.quantization || (stored.storageDtype !== "F32" && stored.storageDtype !== "F16" && stored.storageDtype !== "BF16") || stored.storageShape.length !== 2 || !stored.shard || stored.byteOffset === undefined) throw new Error(`${tensor.name}: referência nativa requer matriz densa F32/F16/BF16 com shard e offset.`);
    const [totalOutputs, inFeatures] = stored.storageShape;
    if (!Number.isSafeInteger(startOutput) || !Number.isSafeInteger(outputCount) || startOutput < 0 || outputCount < 1 || startOutput + outputCount > totalOutputs! || input.length !== rows * inFeatures!) throw new Error(`${tensor.name}: tile referenciado possui shape incompatível.`);
    const dtype = stored.storageDtype === "F32" ? 0 : stored.storageDtype === "BF16" ? 1 : 2;
    const elementBytes = stored.storageDtype === "F32" ? 4 : 2, byteLength = outputCount * inFeatures! * elementBytes;
    const byteOffset = stored.byteOffset + startOutput * inFeatures! * elementBytes;
    if (!Number.isSafeInteger(byteOffset) || !Number.isSafeInteger(byteLength) || byteLength < 1) throw new Error(`${tensor.name}: intervalo binário referenciado inválido.`);
    const shard = Buffer.from(stored.shard, "utf8"); if (shard.length === 0 || shard.length > 4096) throw new Error(`${tensor.name}: nome de shard inválido.`);
    this.#active = true;
    try {
      this.#referenceDispatches += 1;
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(rows, 0); header.writeUInt32LE(outputCount, 4); header.writeUInt32LE(inFeatures!, 8); header.writeUInt32LE((STORAGE_REFERENCE_FLAG + dtype) >>> 0, 12);
      const name = Buffer.from(stored.name, "utf8"); if (name.length === 0 || name.length > 4096) throw new Error(`${tensor.name}: nome de tensor inválido.`);
      const metadata = Buffer.allocUnsafe(24); metadata.writeBigUInt64LE(BigInt(byteOffset), 0); metadata.writeUInt32LE(byteLength, 8); metadata.writeUInt32LE(startOutput, 12); metadata.writeUInt32LE(shard.length, 16); metadata.writeUInt32LE(name.length, 20);
      await this.#write(header); await this.#write(Buffer.from(input.buffer, input.byteOffset, input.byteLength)); await this.#write(metadata); await this.#write(shard); await this.#write(name);
      return await this.#readResult(rows, outputCount);
    } finally { this.#active = false; }
  }

  async #requestReferences(input: Float32Array, requests: readonly PagedLinearStorageReference[], rows: number): Promise<readonly Float32Array[]> {
    if (this.#active) throw new Error("Worker linear persistente não aceita tiles concorrentes no mesmo canal.");
    if (requests.length < 2 || requests.length > 256) throw new Error("Worker linear em lote requer entre 2 e 256 referências.");
    const prepared = requests.map((request) => this.#prepareReference(request, rows, input.length));
    const inFeatures = prepared[0]!.inFeatures;
    if (prepared.some((entry) => entry.inFeatures !== inFeatures)) throw new Error("Worker linear em lote requer o mesmo número de features.");
    this.#active = true;
    try {
      this.#batchDispatches += 1; this.#batchedProjectionTiles += prepared.length;
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(rows, 0); header.writeUInt32LE(prepared.length, 4); header.writeUInt32LE(inFeatures, 8); header.writeUInt32LE((STORAGE_REFERENCE_FLAG + STORAGE_REFERENCE_BATCH_FLAG) >>> 0, 12);
      await this.#write(header); await this.#write(Buffer.from(input.buffer, input.byteOffset, input.byteLength));
      for (const entry of prepared) {
        const metadata = Buffer.allocUnsafe(32); metadata.writeUInt32LE(entry.dtype, 0); metadata.writeUInt32LE(entry.outputCount, 4); metadata.writeBigUInt64LE(BigInt(entry.byteOffset), 8); metadata.writeUInt32LE(entry.byteLength, 16); metadata.writeUInt32LE(entry.startOutput, 20); metadata.writeUInt32LE(entry.shard.length, 24); metadata.writeUInt32LE(entry.name.length, 28);
        await this.#write(metadata); await this.#write(entry.shard); await this.#write(entry.name);
      }
      const results: Float32Array[] = [];
      for (const entry of prepared) results.push(await this.#readResult(rows, entry.outputCount));
      return results;
    } finally { this.#active = false; }
  }

  #prepareReference(request: PagedLinearStorageReference, rows: number, inputLength: number) {
    const { tensor, startOutput, outputCount } = request;
    const stored = this.#storageTensors?.get(tensor.name);
    if (!stored || stored.storageDtype !== tensor.storageDtype || stored.storageShape.length !== tensor.storageShape.length || stored.storageShape.some((value, index) => value !== tensor.storageShape[index]) || stored.logicalShape.length !== tensor.logicalShape.length || stored.logicalShape.some((value, index) => value !== tensor.logicalShape[index])) throw new Error(`${tensor.name}: catálogo mmap diverge do operando literal.`);
    if (stored.quantization || (stored.storageDtype !== "F32" && stored.storageDtype !== "F16" && stored.storageDtype !== "BF16") || stored.storageShape.length !== 2 || !stored.shard || stored.byteOffset === undefined) throw new Error(`${tensor.name}: referência nativa requer matriz densa F32/F16/BF16 com shard e offset.`);
    const [totalOutputs, inFeatures] = stored.storageShape;
    if (!Number.isSafeInteger(startOutput) || !Number.isSafeInteger(outputCount) || startOutput < 0 || outputCount < 1 || startOutput + outputCount > totalOutputs! || inputLength !== rows * inFeatures!) throw new Error(`${tensor.name}: tile referenciado possui shape incompatível.`);
    const dtype = stored.storageDtype === "F32" ? 0 : stored.storageDtype === "BF16" ? 1 : 2;
    const elementBytes = stored.storageDtype === "F32" ? 4 : 2, byteLength = outputCount * inFeatures! * elementBytes;
    const byteOffset = stored.byteOffset + startOutput * inFeatures! * elementBytes;
    const shard = Buffer.from(stored.shard, "utf8"), name = Buffer.from(stored.name, "utf8");
    if (!Number.isSafeInteger(byteOffset) || !Number.isSafeInteger(byteLength) || byteLength < 1 || shard.length < 1 || shard.length > 4096 || name.length < 1 || name.length > 4096) throw new Error(`${tensor.name}: identidade binária referenciada inválida.`);
    return { dtype, outputCount, inFeatures: inFeatures!, byteOffset, byteLength, startOutput, shard, name };
  }

  dispatchMetrics(): { referenceDispatches: number; batchDispatches: number; batchedProjectionTiles: number } {
    return { referenceDispatches: this.#referenceDispatches, batchDispatches: this.#batchDispatches, batchedProjectionTiles: this.#batchedProjectionTiles };
  }

  async #readResult(rows: number, outputCount: number): Promise<Float32Array> {
    const size = (await this.#read(4)).readUInt32LE(0), expected = rows * outputCount * 4;
    if (size !== expected) throw new Error(`Worker linear retornou ${size} bytes; esperados ${expected}.`);
    const bytes = await this.#read(size), copy = new Uint8Array(size); copy.set(bytes);
    return new Float32Array(copy.buffer);
  }

  async close(): Promise<void> { if (!this.child.stdin.destroyed) this.child.stdin.end(); if (this.child.exitCode === null) await once(this.child, "close"); }

  async #write(bytes: Buffer): Promise<void> { if (this.#closedError) throw this.#closedError; if (!this.child.stdin.write(bytes)) await once(this.child.stdin, "drain"); }
  async #read(size: number): Promise<Buffer> {
    while (this.#buffer.length < size) { if (this.#closedError) throw this.#closedError; await new Promise<void>((accept) => this.#waiting.push(accept)); }
    const result = this.#buffer.subarray(0, size); this.#buffer = this.#buffer.subarray(size); return result;
  }
  #wake(): void { for (const accept of this.#waiting.splice(0)) accept(); }
  #fail(error: Error): void { this.#closedError ??= error; this.#wake(); }
}
