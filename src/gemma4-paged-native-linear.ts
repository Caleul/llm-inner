import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { once } from "node:events";
import { endianness } from "node:os";
import { resolve } from "node:path";
import type { PagedFusedAttentionRequest, PagedFusedAttentionResult, PagedFusedDecoderLayerRequest, PagedFusedDecoderLayerResult, PagedFusedDecoderStackRequest, PagedFusedDecoderStackResult, PagedFusedFfnRequest, PagedFusedPlePreludeRequest, PagedFusedPleRequest, PagedLinearStorageReference, PagedLinearTileKernel, PagedNativeAttentionRequest } from "./paged-dense.js";
import type { TensorInfo } from "./types.js";

const STORAGE_REFERENCE_FLAG = 0x8000_0000;
const STORAGE_REFERENCE_BATCH_FLAG = 0x4000_0000;
const STORAGE_REFERENCE_MLP_FLAG = 0x2000_0000;
const STORAGE_NATIVE_BF16_FLAG = 0x1000_0000;
const NATIVE_ATTENTION_FLAG = 0x0800_0000;
const FUSED_ATTENTION_FLAG = 0x0400_0000;
const FUSED_PLE_FLAG = 0x0200_0000;
const FUSED_PLE_PRELUDE_FLAG = 0x0100_0000;
const FUSED_FFN_FLAG = 0x0080_0000;
const FUSED_DECODER_LAYER_FLAG = 0x0040_0000;
const FUSED_DECODER_STACK_FLAG = 0x0020_0000;
const STORAGE_NATIVE_BF16_STREAM_FLAG = 0x0010_0000;

export class Gemma4PagedNativeLinearWorker implements PagedLinearTileKernel {
  readonly backend: string;
  readonly child: ChildProcessWithoutNullStreams;
  readonly multiplyStorageReference?: PagedLinearTileKernel["multiplyStorageReference"];
  readonly multiplyStorageReferenceNativeBf16?: PagedLinearTileKernel["multiplyStorageReferenceNativeBf16"];
  readonly multiplyWholeStorageReferenceNativeBf16?: PagedLinearTileKernel["multiplyWholeStorageReferenceNativeBf16"];
  readonly multiplyWholeStorageReferenceNativeBf16Tiled?: PagedLinearTileKernel["multiplyWholeStorageReferenceNativeBf16Tiled"];
  readonly multiplyStorageReferences?: PagedLinearTileKernel["multiplyStorageReferences"];
  readonly fusedGatedMlpStorageReference?: PagedLinearTileKernel["fusedGatedMlpStorageReference"];
  readonly fusedFfnStorageReferences?: PagedLinearTileKernel["fusedFfnStorageReferences"];
  readonly fusedDecoderLayerStorageReferences?: PagedLinearTileKernel["fusedDecoderLayerStorageReferences"];
  readonly fusedDecoderStackStorageReferences?: PagedLinearTileKernel["fusedDecoderStackStorageReferences"];
  readonly fusedPleStorageReferences?: PagedLinearTileKernel["fusedPleStorageReferences"];
  readonly fusedPlePreludeStorageReference?: PagedLinearTileKernel["fusedPlePreludeStorageReference"];
  readonly attention?: PagedLinearTileKernel["attention"];
  readonly fusedAttentionStorageReferences?: PagedLinearTileKernel["fusedAttentionStorageReferences"];
  readonly #storageTensors: ReadonlyMap<string, TensorInfo> | undefined;
  #buffer = Buffer.alloc(0);
  #waiting: Array<() => void> = [];
  #closedError?: Error;
  #active = false;
  #referenceDispatches = 0;
  #wholeNativeBf16Dispatches = 0;
  #streamedNativeBf16Dispatches = 0;
  #batchDispatches = 0;
  #batchedProjectionTiles = 0;
  #fusedMlpDispatches = 0;
  #fusedFfnDispatches = 0;
  #fusedDecoderLayerDispatches = 0;
  #fusedDecoderStackDispatches = 0;
  #fusedPleDispatches = 0;
  #fusedPlePreludeDispatches = 0;
  #nativeAttentionDispatches = 0;
  #fusedAttentionDispatches = 0;
  #referenceSeconds = 0;
  #streamedNativeBf16Seconds = 0;
  #batchSeconds = 0;
  #fusedMlpSeconds = 0;
  #fusedFfnSeconds = 0;
  #fusedDecoderLayerSeconds = 0;
  #fusedDecoderStackSeconds = 0;
  #fusedPleSeconds = 0;
  #fusedPlePreludeSeconds = 0;
  #nativeAttentionSeconds = 0;
  #fusedAttentionSeconds = 0;

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
      if (backend === "pytorch") {
        this.multiplyStorageReferenceNativeBf16 = (input, tensor, startOutput, outputCount, rows) => this.#requestReference(input, tensor, startOutput, outputCount, rows, true);
        this.multiplyWholeStorageReferenceNativeBf16 = (input, tensor, rows) => {
          this.#wholeNativeBf16Dispatches += 1;
          return this.#requestReference(input, tensor, 0, tensor.storageShape[0]!, rows, true);
        };
        this.multiplyWholeStorageReferenceNativeBf16Tiled = (input, tensor, rows, maxReadBytes) => this.#requestStreamedNativeBf16(input, tensor, rows, maxReadBytes);
      }
      this.multiplyStorageReferences = (input, requests, rows) => this.#requestReferences(input, requests, rows);
      this.fusedGatedMlpStorageReference = (input, gate, up, down, rows, rounding) => this.#requestGatedMlp(input, gate, up, down, rows, rounding);
      if (backend === "pytorch") {
        this.attention = (request) => this.#requestAttention(request);
        this.fusedAttentionStorageReferences = (request) => this.#requestFusedAttention(request);
        this.fusedFfnStorageReferences = (request) => this.#requestFfn(request);
        this.fusedDecoderLayerStorageReferences = (request) => this.#requestDecoderLayer(request);
        this.fusedDecoderStackStorageReferences = (request) => this.#requestDecoderStack(request);
        this.fusedPleStorageReferences = (request) => this.#requestFusedPle(request);
        this.fusedPlePreludeStorageReference = (request) => this.#requestFusedPlePrelude(request);
      }
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
    const started = performance.now();
    try {
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(rows, 0); header.writeUInt32LE(outputCount, 4); header.writeUInt32LE(inFeatures, 8); header.writeUInt32LE(dtype, 12);
      await this.#write(header); await this.#write(Buffer.from(input.buffer, input.byteOffset, input.byteLength)); await this.#write(weight);
      return await this.#readResult(rows, outputCount);
    } finally { this.#referenceSeconds += (performance.now() - started) / 1000; this.#active = false; }
  }

  async #requestReference(input: Float32Array, tensor: TensorInfo, startOutput: number, outputCount: number, rows: number, nativeBf16 = false): Promise<Float32Array> {
    if (this.#active) throw new Error("Worker linear persistente não aceita tiles concorrentes no mesmo canal.");
    const stored = this.#storageTensors?.get(tensor.name);
    if (!stored || stored.storageDtype !== tensor.storageDtype || stored.storageShape.length !== tensor.storageShape.length || stored.storageShape.some((value, index) => value !== tensor.storageShape[index]) || stored.logicalShape.length !== tensor.logicalShape.length || stored.logicalShape.some((value, index) => value !== tensor.logicalShape[index])) throw new Error(`${tensor.name}: catálogo mmap diverge do operando literal.`);
    if (stored.quantization || (stored.storageDtype !== "F32" && stored.storageDtype !== "F16" && stored.storageDtype !== "BF16") || stored.storageShape.length !== 2 || !stored.shard || stored.byteOffset === undefined) throw new Error(`${tensor.name}: referência nativa requer matriz densa F32/F16/BF16 com shard e offset.`);
    if (nativeBf16 && stored.storageDtype !== "BF16") throw new Error(`${tensor.name}: GEMM BF16 nativo requer storage BF16.`);
    const [totalOutputs, inFeatures] = stored.storageShape;
    if (!Number.isSafeInteger(startOutput) || !Number.isSafeInteger(outputCount) || startOutput < 0 || outputCount < 1 || startOutput + outputCount > totalOutputs! || input.length !== rows * inFeatures!) throw new Error(`${tensor.name}: tile referenciado possui shape incompatível.`);
    const dtype = stored.storageDtype === "F32" ? 0 : stored.storageDtype === "BF16" ? 1 : 2;
    const elementBytes = stored.storageDtype === "F32" ? 4 : 2, byteLength = outputCount * inFeatures! * elementBytes;
    const byteOffset = stored.byteOffset + startOutput * inFeatures! * elementBytes;
    if (!Number.isSafeInteger(byteOffset) || !Number.isSafeInteger(byteLength) || byteLength < 1) throw new Error(`${tensor.name}: intervalo binário referenciado inválido.`);
    const shard = Buffer.from(stored.shard, "utf8"); if (shard.length === 0 || shard.length > 4096) throw new Error(`${tensor.name}: nome de shard inválido.`);
    this.#active = true;
    const started = performance.now();
    try {
      this.#referenceDispatches += 1;
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(rows, 0); header.writeUInt32LE(outputCount, 4); header.writeUInt32LE(inFeatures!, 8); header.writeUInt32LE((STORAGE_REFERENCE_FLAG + (nativeBf16 ? STORAGE_NATIVE_BF16_FLAG : 0) + dtype) >>> 0, 12);
      const name = Buffer.from(stored.name, "utf8"); if (name.length === 0 || name.length > 4096) throw new Error(`${tensor.name}: nome de tensor inválido.`);
      const metadata = Buffer.allocUnsafe(24); metadata.writeBigUInt64LE(BigInt(byteOffset), 0); metadata.writeUInt32LE(byteLength, 8); metadata.writeUInt32LE(startOutput, 12); metadata.writeUInt32LE(shard.length, 16); metadata.writeUInt32LE(name.length, 20);
      await this.#write(header); await this.#write(Buffer.from(input.buffer, input.byteOffset, input.byteLength)); await this.#write(metadata); await this.#write(shard); await this.#write(name);
      return await this.#readResult(rows, outputCount);
    } finally { this.#referenceSeconds += (performance.now() - started) / 1000; this.#active = false; }
  }

  async #requestStreamedNativeBf16(input: Float32Array, tensor: TensorInfo, rows: number, maxReadBytes: number): Promise<Float32Array> {
    if (this.#active) throw new Error("Worker linear persistente não aceita heads concorrentes no mesmo canal.");
    const stored = this.#storageTensors?.get(tensor.name);
    if (!stored || stored.storageDtype !== "BF16" || stored.quantization || stored.storageShape.length !== 2 || stored.logicalShape.length !== 2 || stored.storageShape.some((value, index) => value !== tensor.storageShape[index]) || stored.logicalShape.some((value, index) => value !== tensor.logicalShape[index]) || !stored.shard || stored.byteOffset === undefined) throw new Error(`${tensor.name}: head paginado integral requer matriz BF16 mmap idêntica ao literal.`);
    const [outputCount, inFeatures] = stored.storageShape, rowBytes = inFeatures! * 2, byteLength = outputCount! * rowBytes;
    if (!Number.isSafeInteger(rows) || rows < 1 || input.length !== rows * inFeatures! || !Number.isSafeInteger(maxReadBytes) || maxReadBytes < rowBytes || maxReadBytes > 0xffff_ffff || !Number.isSafeInteger(byteLength) || byteLength < 1) throw new Error(`${tensor.name}: head paginado integral recebeu shape ou tile inválido.`);
    const shard = Buffer.from(stored.shard, "utf8"), name = Buffer.from(stored.name, "utf8");
    if (shard.length < 1 || shard.length > 4096 || name.length < 1 || name.length > 4096) throw new Error(`${tensor.name}: identidade do head paginado integral inválida.`);
    this.#active = true;
    const started = performance.now();
    try {
      this.#streamedNativeBf16Dispatches += 1;
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(rows, 0); header.writeUInt32LE(outputCount!, 4); header.writeUInt32LE(inFeatures!, 8); header.writeUInt32LE((STORAGE_REFERENCE_FLAG + STORAGE_NATIVE_BF16_FLAG + STORAGE_NATIVE_BF16_STREAM_FLAG + 1) >>> 0, 12);
      const metadata = Buffer.allocUnsafe(28); metadata.writeBigUInt64LE(BigInt(stored.byteOffset), 0); metadata.writeUInt32LE(byteLength, 8); metadata.writeUInt32LE(0, 12); metadata.writeUInt32LE(maxReadBytes, 16); metadata.writeUInt32LE(shard.length, 20); metadata.writeUInt32LE(name.length, 24);
      await this.#write(header); await this.#write(Buffer.from(input.buffer, input.byteOffset, input.byteLength)); await this.#write(metadata); await this.#write(shard); await this.#write(name);
      const chunkRows = Math.max(1, Math.floor(maxReadBytes / rowBytes)), result = new Float32Array(rows * outputCount!);
      for (let firstOutput = 0; firstOutput < outputCount!; firstOutput += chunkRows) {
        const tileOutputs = Math.min(chunkRows, outputCount! - firstOutput), tile = await this.#readResult(rows, tileOutputs);
        for (let row = 0; row < rows; row += 1) result.set(tile.subarray(row * tileOutputs, (row + 1) * tileOutputs), row * outputCount! + firstOutput);
      }
      return result;
    } finally { this.#streamedNativeBf16Seconds += (performance.now() - started) / 1000; this.#active = false; }
  }

  async #requestReferences(input: Float32Array, requests: readonly PagedLinearStorageReference[], rows: number): Promise<readonly Float32Array[]> {
    if (this.#active) throw new Error("Worker linear persistente não aceita tiles concorrentes no mesmo canal.");
    if (requests.length < 2 || requests.length > 256) throw new Error("Worker linear em lote requer entre 2 e 256 referências.");
    const prepared = requests.map((request) => this.#prepareReference(request, rows, input.length));
    const inFeatures = prepared[0]!.inFeatures;
    if (prepared.some((entry) => entry.inFeatures !== inFeatures)) throw new Error("Worker linear em lote requer o mesmo número de features.");
    this.#active = true;
    const started = performance.now();
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
    } finally { this.#batchSeconds += (performance.now() - started) / 1000; this.#active = false; }
  }

  async #requestGatedMlp(input: Float32Array, gateTensor: TensorInfo, upTensor: TensorInfo, downTensor: TensorInfo, rows: number, rounding: "bf16" | "real" | "native-bf16"): Promise<Float32Array> {
    if (this.#active) throw new Error("Worker linear persistente não aceita subgrafos concorrentes no mesmo canal.");
    const gate = this.#prepareWholeMatrix(gateTensor), up = this.#prepareWholeMatrix(upTensor), down = this.#prepareWholeMatrix(downTensor);
    if (gate.inFeatures !== up.inFeatures || gate.outputCount !== up.outputCount || down.inFeatures !== gate.outputCount || input.length !== rows * gate.inFeatures) throw new Error("Subgrafo MLP requer gate/up paralelos e down_proj compatível.");
    this.#active = true;
    const started = performance.now();
    try {
      this.#fusedMlpDispatches += 1;
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(rows, 0); header.writeUInt32LE(down.outputCount, 4); header.writeUInt32LE(gate.inFeatures, 8); header.writeUInt32LE((STORAGE_REFERENCE_FLAG + STORAGE_REFERENCE_MLP_FLAG) >>> 0, 12);
      const policy = Buffer.allocUnsafe(4); policy.writeUInt32LE(rounding === "bf16" ? 0 : rounding === "real" ? 1 : 2, 0);
      await this.#write(header); await this.#write(Buffer.from(input.buffer, input.byteOffset, input.byteLength)); await this.#write(policy);
      for (const entry of [gate, up, down]) {
        const metadata = Buffer.allocUnsafe(36); metadata.writeUInt32LE(entry.dtype, 0); metadata.writeUInt32LE(entry.outputCount, 4); metadata.writeUInt32LE(entry.inFeatures, 8); metadata.writeBigUInt64LE(BigInt(entry.byteOffset), 12); metadata.writeUInt32LE(entry.byteLength, 20); metadata.writeUInt32LE(0, 24); metadata.writeUInt32LE(entry.shard.length, 28); metadata.writeUInt32LE(entry.name.length, 32);
        await this.#write(metadata); await this.#write(entry.shard); await this.#write(entry.name);
      }
      return await this.#readResult(rows, down.outputCount);
    } finally { this.#fusedMlpSeconds += (performance.now() - started) / 1000; this.#active = false; }
  }

  async #requestFfn(request: PagedFusedFfnRequest): Promise<Float32Array> {
    if (this.#active) throw new Error("Worker linear persistente não aceita subgrafos concorrentes no mesmo canal.");
    const descriptors = [request.preNormWeight, request.gateWeight, request.upWeight, request.downWeight, request.postNormWeight].map((tensor) => this.#prepareWholeTensor(tensor));
    const [preNorm, gate, up, down, postNorm] = descriptors;
    if (!Number.isSafeInteger(request.rows) || request.rows < 1 || !Number.isSafeInteger(request.hiddenSize) || request.hiddenSize < 1 || !Number.isSafeInteger(request.intermediateSize) || request.intermediateSize < 1 || request.input.length !== request.rows * request.hiddenSize || !Number.isFinite(request.preNormEpsilon) || request.preNormEpsilon <= 0 || !Number.isFinite(request.postNormEpsilon) || request.postNormEpsilon <= 0 || request.rounding !== "native-bf16") throw new Error("Subgrafo FFN recebeu topologia inválida.");
    if (preNorm!.dimensions[0] !== request.hiddenSize || preNorm!.dimensions[1] !== 1 || gate!.dimensions[0] !== request.intermediateSize || gate!.dimensions[1] !== request.hiddenSize || up!.dimensions[0] !== request.intermediateSize || up!.dimensions[1] !== request.hiddenSize || down!.dimensions[0] !== request.hiddenSize || down!.dimensions[1] !== request.intermediateSize || postNorm!.dimensions[0] !== request.hiddenSize || postNorm!.dimensions[1] !== 1 || gate!.dtype !== 1 || up!.dtype !== 1 || down!.dtype !== 1) throw new Error("Subgrafo FFN requer normas e matrizes BF16 compatíveis.");
    this.#active = true;
    const started = performance.now();
    try {
      this.#fusedFfnDispatches += 1;
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(request.rows, 0); header.writeUInt32LE(request.hiddenSize, 4); header.writeUInt32LE(request.hiddenSize, 8); header.writeUInt32LE((STORAGE_REFERENCE_FLAG + FUSED_FFN_FLAG) >>> 0, 12);
      const metadata = Buffer.allocUnsafe(12); metadata.writeUInt32LE(request.intermediateSize, 0); metadata.writeFloatLE(request.preNormEpsilon, 4); metadata.writeFloatLE(request.postNormEpsilon, 8);
      await this.#write(header); await this.#write(Buffer.from(request.input.buffer, request.input.byteOffset, request.input.byteLength)); await this.#write(metadata);
      for (const entry of descriptors) {
        const descriptor = Buffer.allocUnsafe(36); descriptor.writeUInt32LE(entry.dtype, 0); descriptor.writeUInt32LE(entry.dimensions[0], 4); descriptor.writeUInt32LE(entry.dimensions[1], 8); descriptor.writeBigUInt64LE(BigInt(entry.byteOffset), 12); descriptor.writeUInt32LE(entry.byteLength, 20); descriptor.writeUInt32LE(0, 24); descriptor.writeUInt32LE(entry.shard.length, 28); descriptor.writeUInt32LE(entry.name.length, 32);
        await this.#write(descriptor); await this.#write(entry.shard); await this.#write(entry.name);
      }
      return await this.#readResult(request.rows, request.hiddenSize);
    } finally { this.#fusedFfnSeconds += (performance.now() - started) / 1000; this.#active = false; }
  }

  async #requestDecoderStack(request: PagedFusedDecoderStackRequest): Promise<PagedFusedDecoderStackResult> {
    if (this.#active) throw new Error("Worker linear persistente não aceita pilhas decoder concorrentes no mesmo canal.");
    const dimensions = [request.batch, request.querySequence, request.hiddenSize, request.numLayers, request.perLayerWidth];
    if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1) || request.layers.length !== request.numLayers || request.input.length !== request.batch * request.querySequence * request.hiddenSize || request.perLayerInputs.length !== request.batch * request.querySequence * request.numLayers * request.perLayerWidth || request.positions.length !== request.batch * request.querySequence || request.rounding !== "native-bf16") throw new Error("Pilha decoder fundida recebeu topologia global inválida.");
    const prepared = request.layers.map((layer, index) => {
      const shared = layer.sharedProducerLayer !== undefined;
      const totalKeySequence = layer.sourceSequence + (layer.producesKeyValue ? request.querySequence : 0);
      if (layer.layerIndex !== index || layer.batch !== request.batch || layer.querySequence !== request.querySequence || layer.hiddenSize !== request.hiddenSize || layer.perLayerWidth !== request.perLayerWidth || layer.producesKeyValue === shared || (shared && (!Number.isSafeInteger(layer.sharedProducerLayer) || layer.sharedProducerLayer! < 0 || layer.sharedProducerLayer! >= index)) || layer.queryHeads % layer.keyValueHeads !== 0 || (layer.maskHeads !== 1 && layer.maskHeads !== layer.queryHeads) || totalKeySequence < 1 || layer.mask.length !== request.batch * layer.maskHeads * request.querySequence * totalKeySequence || layer.sourceKey.length !== (shared ? 0 : request.batch * layer.keyValueHeads * layer.sourceSequence * layer.headDim) || layer.sourceValue.length !== layer.sourceKey.length) throw new Error(`Pilha decoder recebeu camada ${index} incompatível.`);
      const descriptors = [layer.inputNormWeight, layer.queryWeight, layer.queryNorm, layer.outputWeight, ...(layer.producesKeyValue ? [layer.keyWeight!, layer.keyNorm!, ...(layer.valueWeight ? [layer.valueWeight] : [])] : []), layer.postAttentionNormWeight, layer.preFfnNormWeight, layer.gateWeight, layer.upWeight, layer.downWeight, layer.postFfnNormWeight, layer.pleGateWeight, layer.pleProjectionWeight, layer.pleNormWeight, layer.layerScalar].map((tensor) => this.#prepareWholeTensor(tensor));
      return { layer, descriptors, totalKeySequence };
    });
    this.#active = true;
    const started = performance.now();
    try {
      this.#fusedDecoderStackDispatches += 1;
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(request.batch, 0); header.writeUInt32LE(request.numLayers, 4); header.writeUInt32LE(request.hiddenSize, 8); header.writeUInt32LE(FUSED_DECODER_STACK_FLAG, 12);
      const globalMetadata = Buffer.allocUnsafe(8); globalMetadata.writeUInt32LE(request.querySequence, 0); globalMetadata.writeUInt32LE(request.perLayerWidth, 4);
      await this.#write(header); await this.#write(globalMetadata);
      await this.#write(Buffer.from(request.input.buffer, request.input.byteOffset, request.input.byteLength));
      await this.#write(Buffer.from(request.perLayerInputs.buffer, request.perLayerInputs.byteOffset, request.perLayerInputs.byteLength));
      await this.#write(Buffer.from(request.positions.buffer, request.positions.byteOffset, request.positions.byteLength));
      for (const { layer, descriptors } of prepared) {
        const metadata = Buffer.alloc(96);
        [layer.layerIndex, layer.sharedProducerLayer === undefined ? 0 : layer.sharedProducerLayer + 1, layer.queryHeads, layer.keyValueHeads, layer.sourceSequence, layer.headDim, layer.maskHeads, layer.valueFromKey ? 1 : 0, layer.ropeType === "default" ? 0 : 1, layer.rotaryDim, layer.proportionalPairs, layer.intermediateSize, descriptors.length].forEach((value, index) => metadata.writeUInt32LE(value, index * 4));
        metadata.writeFloatLE(layer.epsilon, 52); metadata.writeFloatLE(layer.scale, 56); metadata.writeDoubleLE(layer.theta, 60); metadata.writeFloatLE(layer.proportionalFactor, 68); metadata.writeFloatLE(layer.inputNormEpsilon, 72); metadata.writeFloatLE(layer.postAttentionNormEpsilon, 76); metadata.writeFloatLE(layer.preFfnNormEpsilon, 80); metadata.writeFloatLE(layer.postFfnNormEpsilon, 84); metadata.writeFloatLE(layer.pleNormEpsilon, 88);
        await this.#write(metadata);
        await this.#write(Buffer.from(layer.mask.buffer, layer.mask.byteOffset, layer.mask.byteLength));
        if (layer.producesKeyValue) {
          await this.#write(Buffer.from(layer.sourceKey.buffer, layer.sourceKey.byteOffset, layer.sourceKey.byteLength));
          await this.#write(Buffer.from(layer.sourceValue.buffer, layer.sourceValue.byteOffset, layer.sourceValue.byteLength));
        }
        for (const entry of descriptors) {
          const descriptor = Buffer.allocUnsafe(36); descriptor.writeUInt32LE(entry.dtype, 0); descriptor.writeUInt32LE(entry.dimensions[0], 4); descriptor.writeUInt32LE(entry.dimensions[1], 8); descriptor.writeBigUInt64LE(BigInt(entry.byteOffset), 12); descriptor.writeUInt32LE(entry.byteLength, 20); descriptor.writeUInt32LE(0, 24); descriptor.writeUInt32LE(entry.shard.length, 28); descriptor.writeUInt32LE(entry.name.length, 32);
          await this.#write(descriptor); await this.#write(entry.shard); await this.#write(entry.name);
        }
      }
      const hidden = await this.#readResult(request.batch * request.querySequence, request.hiddenSize);
      const caches = [];
      for (const { layer, totalKeySequence } of prepared) if (layer.producesKeyValue) {
        const key = await this.#readResult(request.batch * layer.keyValueHeads * totalKeySequence, layer.headDim);
        const value = await this.#readResult(request.batch * layer.keyValueHeads * totalKeySequence, layer.headDim);
        caches.push({ layerIndex: layer.layerIndex, key, value });
      }
      return { hidden, caches };
    } finally { this.#fusedDecoderStackSeconds += (performance.now() - started) / 1000; this.#active = false; }
  }

  async #requestDecoderLayer(request: PagedFusedDecoderLayerRequest): Promise<PagedFusedDecoderLayerResult> {
    if (this.#active) throw new Error("Worker linear persistente não aceita subgrafos concorrentes no mesmo canal.");
    const dimensions = [request.batch, request.queryHeads, request.keyValueHeads, request.querySequence, request.hiddenSize, request.headDim, request.maskHeads, request.intermediateSize, request.perLayerWidth];
    const totalKeySequence = request.sourceSequence + (request.producesKeyValue ? request.querySequence : 0);
    if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1) || request.queryHeads % request.keyValueHeads !== 0 || (request.maskHeads !== 1 && request.maskHeads !== request.queryHeads) || totalKeySequence < 1 || request.input.length !== request.batch * request.querySequence * request.hiddenSize || request.perLayerInput.length !== request.batch * request.querySequence * request.perLayerWidth || request.positions.length !== request.batch * request.querySequence || request.mask.length !== request.batch * request.maskHeads * request.querySequence * totalKeySequence || request.sourceKey.length !== request.batch * request.keyValueHeads * request.sourceSequence * request.headDim || request.sourceValue.length !== request.sourceKey.length || request.rounding !== "native-bf16") throw new Error("Decoder layer fundida recebeu topologia inválida.");
    const descriptors = [request.inputNormWeight, request.queryWeight, request.queryNorm, request.outputWeight, ...(request.producesKeyValue ? [request.keyWeight!, request.keyNorm!, ...(request.valueWeight ? [request.valueWeight] : [])] : []), request.postAttentionNormWeight, request.preFfnNormWeight, request.gateWeight, request.upWeight, request.downWeight, request.postFfnNormWeight, request.pleGateWeight, request.pleProjectionWeight, request.pleNormWeight, request.layerScalar].map((tensor) => this.#prepareWholeTensor(tensor));
    this.#active = true;
    const started = performance.now();
    try {
      this.#fusedDecoderLayerDispatches += 1;
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(request.batch, 0); header.writeUInt32LE(request.queryHeads, 4); header.writeUInt32LE(request.keyValueHeads, 8); header.writeUInt32LE(FUSED_DECODER_LAYER_FLAG, 12);
      const metadata = Buffer.alloc(96);
      [request.querySequence, request.sourceSequence, request.hiddenSize, request.headDim, request.maskHeads, request.producesKeyValue ? 1 : 0, request.valueFromKey ? 1 : 0, request.ropeType === "default" ? 0 : 1, request.rotaryDim, request.proportionalPairs, request.intermediateSize, request.perLayerWidth, descriptors.length].forEach((value, index) => metadata.writeUInt32LE(value, index * 4));
      metadata.writeFloatLE(request.epsilon, 52); metadata.writeFloatLE(request.scale, 56); metadata.writeDoubleLE(request.theta, 60); metadata.writeFloatLE(request.proportionalFactor, 68); metadata.writeFloatLE(request.inputNormEpsilon, 72); metadata.writeFloatLE(request.postAttentionNormEpsilon, 76); metadata.writeFloatLE(request.preFfnNormEpsilon, 80); metadata.writeFloatLE(request.postFfnNormEpsilon, 84); metadata.writeFloatLE(request.pleNormEpsilon, 88);
      await this.#write(header); await this.#write(metadata);
      await this.#write(Buffer.from(request.input.buffer, request.input.byteOffset, request.input.byteLength));
      await this.#write(Buffer.from(request.perLayerInput.buffer, request.perLayerInput.byteOffset, request.perLayerInput.byteLength));
      await this.#write(Buffer.from(request.positions.buffer, request.positions.byteOffset, request.positions.byteLength));
      await this.#write(Buffer.from(request.mask.buffer, request.mask.byteOffset, request.mask.byteLength));
      await this.#write(Buffer.from(request.sourceKey.buffer, request.sourceKey.byteOffset, request.sourceKey.byteLength));
      await this.#write(Buffer.from(request.sourceValue.buffer, request.sourceValue.byteOffset, request.sourceValue.byteLength));
      for (const entry of descriptors) {
        const descriptor = Buffer.allocUnsafe(36); descriptor.writeUInt32LE(entry.dtype, 0); descriptor.writeUInt32LE(entry.dimensions[0], 4); descriptor.writeUInt32LE(entry.dimensions[1], 8); descriptor.writeBigUInt64LE(BigInt(entry.byteOffset), 12); descriptor.writeUInt32LE(entry.byteLength, 20); descriptor.writeUInt32LE(0, 24); descriptor.writeUInt32LE(entry.shard.length, 28); descriptor.writeUInt32LE(entry.name.length, 32);
        await this.#write(descriptor); await this.#write(entry.shard); await this.#write(entry.name);
      }
      const hidden = await this.#readResult(request.batch * request.querySequence, request.hiddenSize);
      if (!request.producesKeyValue) return { hidden };
      const key = await this.#readResult(request.batch * request.keyValueHeads * totalKeySequence, request.headDim);
      const value = await this.#readResult(request.batch * request.keyValueHeads * totalKeySequence, request.headDim);
      return { hidden, key, value };
    } finally { this.#fusedDecoderLayerSeconds += (performance.now() - started) / 1000; this.#active = false; }
  }

  async #requestAttention(request: PagedNativeAttentionRequest): Promise<Float32Array> {
    if (this.#active) throw new Error("Worker nativo persistente não aceita attention concorrente no mesmo canal.");
    const dimensions = [request.batch, request.queryHeads, request.keyValueHeads, request.querySequence, request.keySequence, request.headDim, request.maskHeads];
    if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1) || request.queryHeads % request.keyValueHeads !== 0 || (request.maskHeads !== 1 && request.maskHeads !== request.queryHeads) || !Number.isFinite(request.scale)) throw new Error("Attention nativa recebeu topologia inválida.");
    if (request.query.length !== request.batch * request.queryHeads * request.querySequence * request.headDim || request.key.length !== request.batch * request.keyValueHeads * request.keySequence * request.headDim || request.value.length !== request.key.length || request.mask.length !== request.batch * request.maskHeads * request.querySequence * request.keySequence) throw new Error("Attention nativa recebeu payload incompatível com a topologia.");
    this.#active = true;
    const started = performance.now();
    try {
      this.#nativeAttentionDispatches += 1;
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(request.batch, 0); header.writeUInt32LE(request.queryHeads, 4); header.writeUInt32LE(request.keyValueHeads, 8); header.writeUInt32LE(NATIVE_ATTENTION_FLAG, 12);
      const metadata = Buffer.alloc(32); metadata.writeUInt32LE(request.querySequence, 0); metadata.writeUInt32LE(request.keySequence, 4); metadata.writeUInt32LE(request.headDim, 8); metadata.writeUInt32LE(request.maskHeads, 12); metadata.writeUInt32LE(request.rounding === "bf16" ? 0 : 1, 16); metadata.writeFloatLE(request.scale, 20);
      await this.#write(header); await this.#write(metadata);
      for (const values of [request.query, request.key, request.value, request.mask]) await this.#write(Buffer.from(values.buffer, values.byteOffset, values.byteLength));
      return await this.#readResult(request.batch * request.querySequence, request.queryHeads * request.headDim);
    } finally { this.#nativeAttentionSeconds += (performance.now() - started) / 1000; this.#active = false; }
  }

  async #requestFusedPle(request: PagedFusedPleRequest): Promise<Float32Array> {
    if (this.#active) throw new Error("Worker nativo persistente não aceita subgrafo PLE concorrente no mesmo canal.");
    if (![request.rows, request.hiddenSize, request.perLayerWidth].every((value) => Number.isSafeInteger(value) && value > 0) || request.input.length !== request.rows * request.hiddenSize || request.perLayerInput.length !== request.rows * request.perLayerWidth || !Number.isFinite(request.epsilon) || request.epsilon <= 0 || (request.rounding !== "bf16" && request.rounding !== "real")) throw new Error("Subgrafo PLE recebeu payload ou dimensões inválidas.");
    const descriptors = [request.gateWeight, request.projectionWeight, request.normWeight, request.layerScalar].map((tensor) => this.#prepareWholeTensor(tensor));
    if (descriptors[0]!.dimensions[0] !== request.perLayerWidth || descriptors[0]!.dimensions[1] !== request.hiddenSize || descriptors[1]!.dimensions[0] !== request.hiddenSize || descriptors[1]!.dimensions[1] !== request.perLayerWidth || descriptors[2]!.dimensions[0] !== request.hiddenSize || descriptors[2]!.dimensions[1] !== 1 || descriptors[3]!.dimensions[0] !== 1 || descriptors[3]!.dimensions[1] !== 1) throw new Error("Subgrafo PLE recebeu pesos incompatíveis.");
    this.#active = true;
    const started = performance.now();
    try {
      this.#fusedPleDispatches += 1;
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(request.rows, 0); header.writeUInt32LE(request.hiddenSize, 4); header.writeUInt32LE(request.hiddenSize, 8); header.writeUInt32LE(FUSED_PLE_FLAG, 12);
      const metadata = Buffer.allocUnsafe(16); metadata.writeUInt32LE(request.perLayerWidth, 0); metadata.writeUInt32LE(descriptors.length, 4); metadata.writeUInt32LE(request.rounding === "bf16" ? 0 : 1, 8); metadata.writeFloatLE(request.epsilon, 12);
      await this.#write(header); await this.#write(metadata);
      for (const values of [request.input, request.perLayerInput]) await this.#write(Buffer.from(values.buffer, values.byteOffset, values.byteLength));
      for (const entry of descriptors) {
        const descriptor = Buffer.allocUnsafe(36); descriptor.writeUInt32LE(entry.dtype, 0); descriptor.writeUInt32LE(entry.dimensions[0]!, 4); descriptor.writeUInt32LE(entry.dimensions[1]!, 8); descriptor.writeBigUInt64LE(BigInt(entry.byteOffset), 12); descriptor.writeUInt32LE(entry.byteLength, 20); descriptor.writeUInt32LE(0, 24); descriptor.writeUInt32LE(entry.shard.length, 28); descriptor.writeUInt32LE(entry.name.length, 32);
        await this.#write(descriptor); await this.#write(entry.shard); await this.#write(entry.name);
      }
      return await this.#readResult(request.rows, request.hiddenSize);
    } finally { this.#fusedPleSeconds += (performance.now() - started) / 1000; this.#active = false; }
  }

  async #requestFusedPlePrelude(request: PagedFusedPlePreludeRequest): Promise<Float32Array> {
    if (this.#active) throw new Error("Worker nativo persistente não aceita prelude PLE concorrente no mesmo canal.");
    const packedWidth = request.numLayers * request.perLayerWidth;
    if (![request.rows, request.hiddenSize, request.numLayers, request.perLayerWidth, request.maxReadBytes].every((value) => Number.isSafeInteger(value) && value > 0) || !Number.isSafeInteger(packedWidth) || request.input.length !== request.rows * request.hiddenSize || request.tokenIdentity.length !== request.rows * packedWidth || !Number.isFinite(request.contextScale) || !Number.isFinite(request.combineScale) || !Number.isFinite(request.epsilon) || request.epsilon <= 0 || (request.rounding !== "bf16" && request.rounding !== "real")) throw new Error("Prelude PLE recebeu payload ou dimensões inválidas.");
    const descriptors = [request.projectionWeight, request.normWeight].map((tensor) => this.#prepareWholeTensor(tensor));
    if (descriptors[0]!.dimensions[0] !== packedWidth || descriptors[0]!.dimensions[1] !== request.hiddenSize || descriptors[1]!.dimensions[0] !== request.perLayerWidth || descriptors[1]!.dimensions[1] !== 1) throw new Error("Prelude PLE recebeu pesos incompatíveis.");
    this.#active = true;
    const started = performance.now();
    try {
      this.#fusedPlePreludeDispatches += 1;
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(request.rows, 0); header.writeUInt32LE(packedWidth, 4); header.writeUInt32LE(request.hiddenSize, 8); header.writeUInt32LE(FUSED_PLE_PRELUDE_FLAG, 12);
      const storageBytes = request.projectionWeight.storageDtype === "F32" ? 4 : 2;
      const tileOutputRows = Math.max(1, Math.floor(request.maxReadBytes / (request.hiddenSize * storageBytes)));
      const metadata = Buffer.alloc(32); metadata.writeUInt32LE(request.numLayers, 0); metadata.writeUInt32LE(request.perLayerWidth, 4); metadata.writeUInt32LE(descriptors.length, 8); metadata.writeUInt32LE(request.rounding === "bf16" ? 0 : 1, 12); metadata.writeFloatLE(request.contextScale, 16); metadata.writeFloatLE(request.combineScale, 20); metadata.writeFloatLE(request.epsilon, 24); metadata.writeUInt32LE(tileOutputRows, 28);
      await this.#write(header); await this.#write(metadata);
      for (const values of [request.input, request.tokenIdentity]) await this.#write(Buffer.from(values.buffer, values.byteOffset, values.byteLength));
      for (const entry of descriptors) {
        const descriptor = Buffer.allocUnsafe(36); descriptor.writeUInt32LE(entry.dtype, 0); descriptor.writeUInt32LE(entry.dimensions[0]!, 4); descriptor.writeUInt32LE(entry.dimensions[1]!, 8); descriptor.writeBigUInt64LE(BigInt(entry.byteOffset), 12); descriptor.writeUInt32LE(entry.byteLength, 20); descriptor.writeUInt32LE(0, 24); descriptor.writeUInt32LE(entry.shard.length, 28); descriptor.writeUInt32LE(entry.name.length, 32);
        await this.#write(descriptor); await this.#write(entry.shard); await this.#write(entry.name);
      }
      return await this.#readResult(request.rows, packedWidth);
    } finally { this.#fusedPlePreludeSeconds += (performance.now() - started) / 1000; this.#active = false; }
  }

  async #requestFusedAttention(request: PagedFusedAttentionRequest): Promise<PagedFusedAttentionResult> {
    if (this.#active) throw new Error("Worker nativo persistente não aceita subgrafo de attention concorrente no mesmo canal.");
    const dimensions = [request.batch, request.querySequence, request.hiddenSize, request.queryHeads, request.keyValueHeads, request.headDim, request.maskHeads, request.rotaryDim];
    const totalKeySequence = request.sourceSequence + (request.producesKeyValue ? request.querySequence : 0);
    if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1) || !Number.isSafeInteger(request.sourceSequence) || request.sourceSequence < 0 || totalKeySequence < 1 || request.queryHeads % request.keyValueHeads !== 0 || request.hiddenSize !== request.outputWeight.logicalShape[0] || request.queryHeads * request.headDim !== request.queryWeight.logicalShape[0] || request.rotaryDim > request.headDim || request.rotaryDim % 2 !== 0 || (request.maskHeads !== 1 && request.maskHeads !== request.queryHeads)) throw new Error("Subgrafo de attention recebeu topologia inválida.");
    if (!request.producesKeyValue && request.sourceSequence < request.querySequence) throw new Error("Subgrafo de attention compartilhada recebeu cache menor que a consulta.");
    if (request.producesKeyValue !== (request.keyWeight !== undefined && request.keyNorm !== undefined) || (request.producesKeyValue && request.valueFromKey === (request.valueWeight !== undefined)) || (!request.producesKeyValue && (request.keyWeight || request.keyNorm || request.valueWeight || request.valueFromKey))) throw new Error("Subgrafo de attention recebeu operandos KV incompatíveis.");
    const inputElements = request.batch * request.querySequence * request.hiddenSize;
    const sourceElements = request.batch * request.keyValueHeads * request.sourceSequence * request.headDim;
    const maskElements = request.batch * request.maskHeads * request.querySequence * totalKeySequence;
    if (request.input.length !== inputElements || request.positions.length !== request.batch * request.querySequence || request.mask.length !== maskElements || request.sourceKey.length !== sourceElements || request.sourceValue.length !== sourceElements || request.mask.some((value) => Number.isNaN(value) || value === Infinity) || !Number.isFinite(request.epsilon) || request.epsilon <= 0 || !Number.isFinite(request.scale) || !Number.isFinite(request.theta) || request.theta <= 0 || !Number.isFinite(request.proportionalFactor) || request.proportionalFactor <= 0 || (request.rounding !== "bf16" && request.rounding !== "real" && request.rounding !== "native-bf16")) throw new Error("Subgrafo de attention recebeu payload inválido.");
    const descriptors = [request.queryWeight, request.queryNorm, request.outputWeight, ...(request.producesKeyValue ? [request.keyWeight!, request.keyNorm!, ...(request.valueWeight ? [request.valueWeight] : [])] : [])].map((tensor) => this.#prepareWholeTensor(tensor));
    this.#active = true;
    const started = performance.now();
    try {
      this.#fusedAttentionDispatches += 1;
      const header = Buffer.allocUnsafe(16); header.writeUInt32LE(request.batch, 0); header.writeUInt32LE(request.queryHeads, 4); header.writeUInt32LE(request.keyValueHeads, 8); header.writeUInt32LE(FUSED_ATTENTION_FLAG, 12);
      const metadata = Buffer.alloc(80);
      [request.querySequence, request.sourceSequence, request.hiddenSize, request.headDim, request.maskHeads, request.producesKeyValue ? 1 : 0, request.valueFromKey ? 1 : 0, request.ropeType === "default" ? 0 : 1, request.rotaryDim, request.proportionalPairs, descriptors.length].forEach((value, index) => metadata.writeUInt32LE(value, index * 4));
      metadata.writeFloatLE(request.epsilon, 44); metadata.writeFloatLE(request.scale, 48); metadata.writeDoubleLE(request.theta, 52); metadata.writeFloatLE(request.proportionalFactor, 60);
      metadata.writeUInt32LE(request.rounding === "bf16" ? 0 : request.rounding === "real" ? 1 : 2, 64);
      await this.#write(header); await this.#write(metadata);
      await this.#write(Buffer.from(request.input.buffer, request.input.byteOffset, request.input.byteLength));
      await this.#write(Buffer.from(request.positions.buffer, request.positions.byteOffset, request.positions.byteLength));
      await this.#write(Buffer.from(request.mask.buffer, request.mask.byteOffset, request.mask.byteLength));
      for (const values of [request.sourceKey, request.sourceValue]) await this.#write(Buffer.from(values.buffer, values.byteOffset, values.byteLength));
      for (const entry of descriptors) {
        const descriptor = Buffer.allocUnsafe(36); descriptor.writeUInt32LE(entry.dtype, 0); descriptor.writeUInt32LE(entry.dimensions[0]!, 4); descriptor.writeUInt32LE(entry.dimensions[1]!, 8); descriptor.writeBigUInt64LE(BigInt(entry.byteOffset), 12); descriptor.writeUInt32LE(entry.byteLength, 20); descriptor.writeUInt32LE(0, 24); descriptor.writeUInt32LE(entry.shard.length, 28); descriptor.writeUInt32LE(entry.name.length, 32);
        await this.#write(descriptor); await this.#write(entry.shard); await this.#write(entry.name);
      }
      const projected = await this.#readResult(request.batch * request.querySequence, request.hiddenSize);
      if (!request.producesKeyValue) return { projected };
      const key = await this.#readResult(request.batch * request.keyValueHeads * totalKeySequence, request.headDim);
      const value = await this.#readResult(request.batch * request.keyValueHeads * totalKeySequence, request.headDim);
      return { projected, key, value };
    } finally { this.#fusedAttentionSeconds += (performance.now() - started) / 1000; this.#active = false; }
  }

  #prepareWholeTensor(tensor: TensorInfo) {
    const stored = this.#storageTensors?.get(tensor.name);
    if (!stored || stored.storageDtype !== tensor.storageDtype || stored.storageShape.length !== tensor.storageShape.length || stored.storageShape.some((value, index) => value !== tensor.storageShape[index]) || stored.logicalShape.length !== tensor.logicalShape.length || stored.logicalShape.some((value, index) => value !== tensor.logicalShape[index])) throw new Error(`${tensor.name}: catálogo mmap diverge do tensor do subgrafo.`);
    if (stored.quantization || (stored.storageDtype !== "F32" && stored.storageDtype !== "F16" && stored.storageDtype !== "BF16") || (stored.storageShape.length !== 1 && stored.storageShape.length !== 2) || !stored.shard || stored.byteOffset === undefined) throw new Error(`${tensor.name}: subgrafo nativo requer tensor denso F32/F16/BF16 1-D ou 2-D.`);
    const dimensions: [number, number] = stored.storageShape.length === 1 ? [stored.storageShape[0]!, 1] : [stored.storageShape[0]!, stored.storageShape[1]!];
    const dtype = stored.storageDtype === "F32" ? 0 : stored.storageDtype === "BF16" ? 1 : 2, elementBytes = stored.storageDtype === "F32" ? 4 : 2;
    const byteLength = dimensions[0] * dimensions[1] * elementBytes, shard = Buffer.from(stored.shard, "utf8"), name = Buffer.from(stored.name, "utf8");
    if (!Number.isSafeInteger(byteLength) || byteLength < 1 || shard.length < 1 || shard.length > 4096 || name.length < 1 || name.length > 4096) throw new Error(`${tensor.name}: identidade do tensor do subgrafo inválida.`);
    return { dtype, dimensions, byteOffset: stored.byteOffset, byteLength, shard, name };
  }

  #prepareWholeMatrix(tensor: TensorInfo) {
    const stored = this.#storageTensors?.get(tensor.name);
    if (!stored || stored.storageDtype !== tensor.storageDtype || stored.storageShape.length !== tensor.storageShape.length || stored.storageShape.some((value, index) => value !== tensor.storageShape[index]) || stored.logicalShape.length !== tensor.logicalShape.length || stored.logicalShape.some((value, index) => value !== tensor.logicalShape[index])) throw new Error(`${tensor.name}: catálogo mmap diverge do subgrafo literal.`);
    if (stored.quantization || (stored.storageDtype !== "F32" && stored.storageDtype !== "F16" && stored.storageDtype !== "BF16") || stored.storageShape.length !== 2 || !stored.shard || stored.byteOffset === undefined) throw new Error(`${tensor.name}: subgrafo nativo requer matriz densa F32/F16/BF16.`);
    const [outputCount, inFeatures] = stored.storageShape, dtype = stored.storageDtype === "F32" ? 0 : stored.storageDtype === "BF16" ? 1 : 2;
    const byteLength = outputCount! * inFeatures! * (stored.storageDtype === "F32" ? 4 : 2), shard = Buffer.from(stored.shard, "utf8"), name = Buffer.from(stored.name, "utf8");
    if (!Number.isSafeInteger(byteLength) || byteLength < 1 || shard.length < 1 || shard.length > 4096 || name.length < 1 || name.length > 4096) throw new Error(`${tensor.name}: identidade do subgrafo nativo inválida.`);
    return { dtype, outputCount: outputCount!, inFeatures: inFeatures!, byteOffset: stored.byteOffset, byteLength, shard, name };
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

  dispatchMetrics(): { referenceDispatches: number; wholeNativeBf16Dispatches: number; streamedNativeBf16Dispatches: number; batchDispatches: number; batchedProjectionTiles: number; fusedMlpDispatches: number; fusedFfnDispatches: number; fusedDecoderLayerDispatches: number; fusedDecoderStackDispatches: number; fusedPleDispatches: number; fusedPlePreludeDispatches: number; nativeAttentionDispatches: number; fusedAttentionDispatches: number; referenceSeconds: number; streamedNativeBf16Seconds: number; batchSeconds: number; fusedMlpSeconds: number; fusedFfnSeconds: number; fusedDecoderLayerSeconds: number; fusedDecoderStackSeconds: number; fusedPleSeconds: number; fusedPlePreludeSeconds: number; nativeAttentionSeconds: number; fusedAttentionSeconds: number } {
    return { referenceDispatches: this.#referenceDispatches, wholeNativeBf16Dispatches: this.#wholeNativeBf16Dispatches, streamedNativeBf16Dispatches: this.#streamedNativeBf16Dispatches, batchDispatches: this.#batchDispatches, batchedProjectionTiles: this.#batchedProjectionTiles, fusedMlpDispatches: this.#fusedMlpDispatches, fusedFfnDispatches: this.#fusedFfnDispatches, fusedDecoderLayerDispatches: this.#fusedDecoderLayerDispatches, fusedDecoderStackDispatches: this.#fusedDecoderStackDispatches, fusedPleDispatches: this.#fusedPleDispatches, fusedPlePreludeDispatches: this.#fusedPlePreludeDispatches, nativeAttentionDispatches: this.#nativeAttentionDispatches, fusedAttentionDispatches: this.#fusedAttentionDispatches, referenceSeconds: this.#referenceSeconds, streamedNativeBf16Seconds: this.#streamedNativeBf16Seconds, batchSeconds: this.#batchSeconds, fusedMlpSeconds: this.#fusedMlpSeconds, fusedFfnSeconds: this.#fusedFfnSeconds, fusedDecoderLayerSeconds: this.#fusedDecoderLayerSeconds, fusedDecoderStackSeconds: this.#fusedDecoderStackSeconds, fusedPleSeconds: this.#fusedPleSeconds, fusedPlePreludeSeconds: this.#fusedPlePreludeSeconds, nativeAttentionSeconds: this.#nativeAttentionSeconds, fusedAttentionSeconds: this.#fusedAttentionSeconds };
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
