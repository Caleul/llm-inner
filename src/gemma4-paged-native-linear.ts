import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { once } from "node:events";
import { endianness } from "node:os";
import type { PagedLinearTileKernel } from "./paged-dense.js";

export class Gemma4PagedNativeLinearWorker implements PagedLinearTileKernel {
  readonly backend = "persistent-pytorch-f32-tile";
  readonly child: ChildProcessWithoutNullStreams;
  #buffer = Buffer.alloc(0);
  #waiting: Array<() => void> = [];
  #closedError?: Error;
  #active = false;

  constructor(options: { python: string; helper: string; threads: number }) {
    if (endianness() !== "LE") throw new Error("Kernel linear binário requer host little-endian.");
    if (!Number.isSafeInteger(options.threads) || options.threads < 1) throw new Error("threads do kernel linear deve ser positivo.");
    this.child = spawn(options.python, [options.helper, "--threads", String(options.threads)], { stdio: ["pipe", "pipe", "pipe"] });
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
      const size = (await this.#read(4)).readUInt32LE(0), expected = rows * outputCount * 4;
      if (size !== expected) throw new Error(`Worker linear retornou ${size} bytes; esperados ${expected}.`);
      const bytes = await this.#read(size), copy = new Uint8Array(size); copy.set(bytes);
      return new Float32Array(copy.buffer);
    } finally { this.#active = false; }
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
