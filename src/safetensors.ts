import { open, readFile, readdir, stat } from "node:fs/promises";
import * as path from "node:path";
import type { FileHandle } from "node:fs/promises";
import type {
  DenseTensor,
  DenseF32Tensor,
  JsonObject,
  ModelCatalog,
  QuantizationSpec,
  TensorInfo,
} from "./types.js";
import { asObject, product } from "./utils.js";

interface SafeTensorHeaderEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

interface ParsedShard {
  file: string;
  headerLength: number;
  payloadLength: number;
  tensors: Map<string, SafeTensorHeaderEntry>;
}

interface WeightIndex {
  weight_map: Record<string, string>;
}

const DTYPE_BYTES: Record<string, number> = {
  BOOL: 1,
  U8: 1,
  I8: 1,
  U16: 2,
  I16: 2,
  F16: 2,
  BF16: 2,
  U32: 4,
  I32: 4,
  F32: 4,
  U64: 8,
  I64: 8,
  F64: 8,
};
const F64_BYTES = 8;
const F32_BYTES = 4;
const F16_BYTES = 2;
/** Safetensors defines a 100 MiB maximum header to prevent hostile allocations. */
const MAX_HEADER_BYTES = 100 * 1024 * 1024;

export class SafetensorsCatalogReader {
  readonly #sourceDir: string;
  readonly #handles = new Map<string, FileHandle>();

  constructor(sourceDir: string) {
    this.#sourceDir = sourceDir;
  }

  async close(): Promise<void> {
    await Promise.all([...this.#handles.values()].map((handle) => handle.close()));
    this.#handles.clear();
  }

  async inspect(): Promise<ModelCatalog> {
    const sourceStat = await stat(this.#sourceDir);
    if (!sourceStat.isDirectory()) {
      throw new Error(`Esperado diretório de modelo, recebido: ${this.#sourceDir}`);
    }

    const config = await this.#readConfig();
    const files = await readdir(this.#sourceDir);
    const indexName = files.find((file) => file.endsWith(".safetensors.index.json"));

    let weightMap: Record<string, string>;
    if (indexName) {
      const raw = JSON.parse(await readFile(path.join(this.#sourceDir, indexName), "utf8")) as unknown;
      const index = asObject(raw, indexName) as unknown as WeightIndex;
      if (!index.weight_map || typeof index.weight_map !== "object") {
        throw new Error(`${indexName} não possui weight_map válido.`);
      }
      weightMap = index.weight_map;
      for (const [name, shard] of Object.entries(weightMap)) {
        if (typeof shard !== "string") {
          throw new Error(`${indexName}: shard de ${name} deve ser uma string.`);
        }
        this.#validateShardName(shard);
      }
    } else {
      const shards = files.filter((file) => file.endsWith(".safetensors"));
      if (shards.length === 0) throw new Error("Nenhum .safetensors encontrado.");
      weightMap = {};
      for (const shard of shards) {
        const parsed = await this.#parseShard(shard);
        for (const name of parsed.tensors.keys()) {
          if (weightMap[name]) {
            throw new Error(`Tensor duplicado ${name} em shards sem index.`);
          }
          weightMap[name] = shard;
        }
      }
    }

    const parsedByShard = new Map<string, ParsedShard>();
    for (const shard of new Set(Object.values(weightMap))) {
      parsedByShard.set(shard, await this.#parseShard(shard));
    }

    const tensors = new Map<string, TensorInfo>();
    const quantizationRoot = this.#quantizationRoot(config);

    for (const [name, shard] of Object.entries(weightMap)) {
      const parsed = parsedByShard.get(shard);
      const entry = parsed?.tensors.get(name);
      if (!parsed || !entry) {
        throw new Error(`Index aponta ${name} para ${shard}, mas o cabeçalho não contém o tensor.`);
      }

      this.#validateDenseStorage(name, entry, quantizationRoot, parsed.payloadLength);
      const quantization = this.#quantizationForTensor(name, config, tensors, entry);
      const logicalShape = this.#logicalShape(entry, quantization);
      const [start, end] = entry.data_offsets;

      tensors.set(name, {
        name,
        storageDtype: entry.dtype,
        storageShape: [...entry.shape],
        logicalShape,
        byteOffset: 8 + parsed.headerLength + start,
        byteLength: end - start,
        shard,
        ...(quantization ? { quantization } : {}),
      });
    }

    // Segunda passagem: agora scales/biases já estão catalogados. Além de ligar
    // a especificação, derivamos a dimensão lógica a partir de
    // groups * group_size, não apenas da capacidade dos U32 empacotados.
    // Isso evita expor colunas de padding como pesos reais.
    let hasMlxQuantization = false;
    for (const tensor of tensors.values()) {
      if (!tensor.name.endsWith(".weight")) continue;
      const q = this.#quantizationForTensor(tensor.name, config, tensors);
      if (!q) continue;
      tensor.quantization = q;
      hasMlxQuantization = true;

      const scale = q.scaleTensor ? tensors.get(q.scaleTensor) : undefined;
      if (
        tensor.storageShape.length === 2 &&
        scale &&
        scale.logicalShape.length >= 2 &&
        q.groupSize !== undefined &&
        q.bits !== undefined
      ) {
        const rows = tensor.storageShape[0];
        const groups = scale.logicalShape.at(-1);
        const packedWords = tensor.storageShape[1];
        if (rows === undefined || groups === undefined || packedWords === undefined) {
          throw new Error(`Shape quantizado incompleto em ${tensor.name}.`);
        }
        const logicalColumns = groups * q.groupSize;
        const packedCapacity = Math.floor((packedWords * 32) / q.bits);
        if (logicalColumns > packedCapacity) {
          throw new Error(
            `${tensor.name}: scales indicam ${logicalColumns} colunas, ` +
              `mas o armazenamento comporta apenas ${packedCapacity}.`,
          );
        }
        tensor.logicalShape = [rows, logicalColumns];
      }
    }

    const format = hasMlxQuantization ? "mlx-safetensors" : "safetensors";
    return {
      source: this.#sourceDir,
      format,
      config,
      rawMetadata: {},
      tensors,
    };
  }

  /**
   * Reads one unquantized F64 tensor directly from its Safetensors byte range.
   * The reference executor intentionally accepts no implicit conversion here:
   * widening F32/BF16/F16 would not reproduce their original accumulation and
   * rounding policy.
   */
  async readDenseF64(tensor: TensorInfo): Promise<DenseTensor> {
    if (tensor.quantization) {
      throw new Error(`${tensor.name}: leitura F64 não dequantiza ${tensor.quantization.family}/${tensor.quantization.mode}.`);
    }
    if (tensor.storageDtype !== "F64") {
      throw new Error(`${tensor.name}: leitura de referência requer storageDtype=F64, recebeu ${tensor.storageDtype}.`);
    }
    if (
      !tensor.shard ||
      tensor.byteOffset === undefined ||
      tensor.byteLength === undefined ||
      tensor.storageShape.length !== tensor.logicalShape.length ||
      tensor.storageShape.some((dimension, index) => dimension !== tensor.logicalShape[index])
    ) {
      throw new Error(`${tensor.name}: metadados Safetensors densos incompletos ou shape lógico diferente do storage.`);
    }
    const elements = product(tensor.storageShape);
    const expectedBytes = elements * F64_BYTES;
    if (tensor.byteLength !== expectedBytes) {
      throw new Error(`${tensor.name}: intervalo de ${tensor.byteLength} bytes não corresponde a ${elements} valores F64.`);
    }

    const bytes = Buffer.allocUnsafe(tensor.byteLength);
    await this.#readExactly(await this.#getHandle(tensor.shard), bytes, tensor.byteOffset);
    const values = new Float64Array(elements);
    for (let index = 0; index < elements; index += 1) values[index] = bytes.readDoubleLE(index * F64_BYTES);
    return { shape: [...tensor.logicalShape], values };
  }

  /**
   * Reads one unquantized F32 tensor directly from its Safetensors byte range.
   * It intentionally preserves binary32 values in a Float32Array: callers
   * select a separate explicit F32 execution policy instead of widening these
   * constants into the F64 interpreter.
   */
  async readDenseF32(tensor: TensorInfo): Promise<DenseF32Tensor> {
    if (tensor.quantization) {
      throw new Error(`${tensor.name}: leitura F32 não dequantiza ${tensor.quantization.family}/${tensor.quantization.mode}.`);
    }
    if (tensor.storageDtype !== "F32") {
      throw new Error(`${tensor.name}: leitura de referência requer storageDtype=F32, recebeu ${tensor.storageDtype}.`);
    }
    if (
      !tensor.shard ||
      tensor.byteOffset === undefined ||
      tensor.byteLength === undefined ||
      tensor.storageShape.length !== tensor.logicalShape.length ||
      tensor.storageShape.some((dimension, index) => dimension !== tensor.logicalShape[index])
    ) {
      throw new Error(`${tensor.name}: metadados Safetensors densos incompletos ou shape lógico diferente do storage.`);
    }
    const elements = product(tensor.storageShape);
    const expectedBytes = elements * F32_BYTES;
    if (tensor.byteLength !== expectedBytes) {
      throw new Error(`${tensor.name}: intervalo de ${tensor.byteLength} bytes não corresponde a ${elements} valores F32.`);
    }

    const bytes = Buffer.allocUnsafe(tensor.byteLength);
    await this.#readExactly(await this.#getHandle(tensor.shard), bytes, tensor.byteOffset);
    const values = new Float32Array(elements);
    for (let index = 0; index < elements; index += 1) values[index] = bytes.readFloatLE(index * F32_BYTES);
    return { shape: [...tensor.logicalShape], values };
  }

  /**
   * Loads a supported dense floating-point Safetensors tensor into the
   * binary32 reference representation. The storage dtype remains an explicit
   * dispatch boundary: this method does not dequantize integers or choose the
   * IR execution policy. F16 and BF16 are widened only after their stored
   * values have been decoded exactly into binary32.
   */
  async readDenseAsF32(tensor: TensorInfo): Promise<DenseF32Tensor> {
    switch (tensor.storageDtype) {
      case "F32":
        return this.readDenseF32(tensor);
      case "F16":
        return this.readDenseF16AsF32(tensor);
      case "BF16":
        return this.readDenseBF16AsF32(tensor);
      default:
        throw new Error(
          `${tensor.name}: execução de referência F32 não suporta storageDtype=${tensor.storageDtype}; ` +
            "use um decodificador explícito para este formato, nunca uma conversão implícita.",
        );
    }
  }

  /**
   * Reads IEEE-754 binary16 storage and widens every finite value and infinity
   * exactly to binary32. This conversion says nothing about the
   * source runtime's compute or accumulation dtype; callers must still supply
   * the explicit F32 executor policy.
   */
  async readDenseF16AsF32(tensor: TensorInfo): Promise<DenseF32Tensor> {
    const { bytes, elements } = await this.#readDenseBytes(tensor, "F16", F16_BYTES);
    const values = new Float32Array(elements);
    for (let index = 0; index < elements; index += 1) {
      values[index] = decodeF16(bytes.readUInt16LE(index * F16_BYTES));
    }
    return { shape: [...tensor.logicalShape], values };
  }

  /**
   * Reads bfloat16 storage and widens every finite value and infinity exactly
   * to binary32 by preserving its 16 most-significant IEEE-754 bits. As with F16, this is storage
   * decoding only, never an implicit choice of runtime arithmetic policy.
   */
  async readDenseBF16AsF32(tensor: TensorInfo): Promise<DenseF32Tensor> {
    const { bytes, elements } = await this.#readDenseBytes(tensor, "BF16", F16_BYTES);
    const values = new Float32Array(elements);
    const scratch = new DataView(new ArrayBuffer(F32_BYTES));
    for (let index = 0; index < elements; index += 1) {
      scratch.setUint32(0, bytes.readUInt16LE(index * F16_BYTES) << 16, true);
      values[index] = scratch.getFloat32(0, true);
    }
    return { shape: [...tensor.logicalShape], values };
  }

  async #readDenseBytes(
    tensor: TensorInfo,
    storageDtype: "F16" | "BF16",
    elementBytes: number,
  ): Promise<{ bytes: Buffer; elements: number }> {
    if (tensor.quantization) {
      throw new Error(`${tensor.name}: leitura ${storageDtype} não dequantiza ${tensor.quantization.family}/${tensor.quantization.mode}.`);
    }
    if (tensor.storageDtype !== storageDtype) {
      throw new Error(`${tensor.name}: leitura de referência requer storageDtype=${storageDtype}, recebeu ${tensor.storageDtype}.`);
    }
    if (
      !tensor.shard ||
      tensor.byteOffset === undefined ||
      tensor.byteLength === undefined ||
      tensor.storageShape.length !== tensor.logicalShape.length ||
      tensor.storageShape.some((dimension, index) => dimension !== tensor.logicalShape[index])
    ) {
      throw new Error(`${tensor.name}: metadados Safetensors densos incompletos ou shape lógico diferente do storage.`);
    }
    const elements = product(tensor.storageShape);
    const expectedBytes = elements * elementBytes;
    if (tensor.byteLength !== expectedBytes) {
      throw new Error(`${tensor.name}: intervalo de ${tensor.byteLength} bytes não corresponde a ${elements} valores ${storageDtype}.`);
    }
    const bytes = Buffer.allocUnsafe(tensor.byteLength);
    await this.#readExactly(await this.#getHandle(tensor.shard), bytes, tensor.byteOffset);
    return { bytes, elements };
  }

  async #readConfig(): Promise<JsonObject> {
    const configPath = path.join(this.#sourceDir, "config.json");
    try {
      return asObject(JSON.parse(await readFile(configPath, "utf8")) as unknown, "config.json");
    } catch (error) {
      throw new Error(
        `config.json é obrigatório para inferir a semântica do modelo: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async #getHandle(file: string): Promise<FileHandle> {
    const existing = this.#handles.get(file);
    if (existing) return existing;
    const handle = await open(path.join(this.#sourceDir, file), "r");
    this.#handles.set(file, handle);
    return handle;
  }

  async #readExactly(handle: FileHandle, buffer: Buffer, position: number): Promise<void> {
    let offset = 0;
    while (offset < buffer.length) {
      const result = await handle.read(buffer, offset, buffer.length - offset, position + offset);
      if (result.bytesRead === 0) throw new Error("EOF inesperado lendo Safetensors.");
      offset += result.bytesRead;
    }
  }

  async #parseShard(file: string): Promise<ParsedShard> {
    this.#validateShardName(file);
    const handle = await this.#getHandle(file);
    const fileInfo = await handle.stat();
    if (!fileInfo.isFile()) throw new Error(`${file} não é um arquivo Safetensors regular.`);
    if (fileInfo.size < 8) throw new Error(`${file} é menor que o prefixo Safetensors de 8 bytes.`);
    const lengthBuffer = Buffer.allocUnsafe(8);
    await this.#readExactly(handle, lengthBuffer, 0);
    const headerLengthBig = lengthBuffer.readBigUInt64LE(0);
    if (headerLengthBig > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(`Cabeçalho de ${file} excede Number.MAX_SAFE_INTEGER.`);
    }
    const headerLength = Number(headerLengthBig);
    if (headerLength > MAX_HEADER_BYTES) {
      throw new Error(`Cabeçalho de ${file} excede o limite Safetensors de ${MAX_HEADER_BYTES} bytes.`);
    }
    if (headerLength > fileInfo.size - 8) {
      throw new Error(`Cabeçalho de ${file} excede o tamanho do arquivo.`);
    }
    const headerBuffer = Buffer.allocUnsafe(headerLength);
    await this.#readExactly(handle, headerBuffer, 8);
    const rawHeader = asObject(JSON.parse(headerBuffer.toString("utf8")) as unknown, `header ${file}`);
    const tensors = new Map<string, SafeTensorHeaderEntry>();

    for (const [name, value] of Object.entries(rawHeader)) {
      if (name === "__metadata__") continue;
      const entry = asObject(value, `tensor ${name}`);
      const dtype = entry.dtype;
      const shape = entry.shape;
      const offsets = entry.data_offsets;
      if (
        typeof dtype !== "string" ||
        !Array.isArray(shape) ||
        !shape.every((dim) => Number.isInteger(dim) && dim >= 0) ||
        !Array.isArray(offsets) ||
        offsets.length !== 2 ||
        !offsets.every((offset) => Number.isInteger(offset) && offset >= 0)
      ) {
        throw new Error(`Metadados inválidos para ${name} em ${file}.`);
      }
      const [start, end] = offsets as [number, number];
      if (end < start) {
        throw new Error(`Tensor ${name} em ${file} possui data_offsets invertidos.`);
      }
      if (end > fileInfo.size - 8 - headerLength) {
        throw new Error(`Tensor ${name} em ${file} ultrapassa o payload declarado.`);
      }
      tensors.set(name, {
        dtype,
        shape: shape as number[],
        data_offsets: offsets as [number, number],
      });
    }

    const intervals = [...tensors.entries()]
      .map(([name, entry]) => ({ name, start: entry.data_offsets[0], end: entry.data_offsets[1] }))
      .sort((left, right) => left.start - right.start || left.end - right.end);
    let previousEnd = 0;
    for (const interval of intervals) {
      if (interval.start < previousEnd) {
        throw new Error(`Tensor ${interval.name} em ${file} sobrepõe outro intervalo de dados.`);
      }
      previousEnd = Math.max(previousEnd, interval.end);
    }

    return { file, headerLength, payloadLength: fileInfo.size - 8 - headerLength, tensors };
  }

  #validateShardName(file: string): void {
    if (
      file.length === 0 ||
      path.isAbsolute(file) ||
      path.extname(file).toLowerCase() !== ".safetensors" ||
      file.split(/[\\/]+/).some((part) => part === "..")
    ) {
      throw new Error(`Nome de shard Safetensors inválido: ${file}`);
    }
    const root = path.resolve(this.#sourceDir);
    const candidate = path.resolve(root, file);
    if (candidate !== root && !candidate.startsWith(`${root}${path.sep}`)) {
      throw new Error(`Shard Safetensors fora do diretório de origem: ${file}`);
    }
  }

  #quantizationRoot(config: JsonObject): JsonObject | undefined {
    const value = config.quantization ?? config.quantization_config;
    if (typeof value !== "object" || value === null || Array.isArray(value)) return undefined;
    return value as JsonObject;
  }

  #quantizationForTensor(
    tensorName: string,
    config: JsonObject,
    knownTensors?: Map<string, TensorInfo>,
    entry?: SafeTensorHeaderEntry,
  ): QuantizationSpec | undefined {
    if (!tensorName.endsWith(".weight")) return undefined;
    const root = this.#quantizationRoot(config);
    if (!root) return undefined;
    const moduleName = tensorName.slice(0, -".weight".length);
    const overrideRaw = root[moduleName];
    if (overrideRaw === false) return undefined;
    const override =
      typeof overrideRaw === "object" && overrideRaw !== null && !Array.isArray(overrideRaw)
        ? (overrideRaw as JsonObject)
        : {};

    const bitsValue = override.bits ?? root.bits;
    const groupSizeValue = override.group_size ?? root.group_size;
    const modeValue = override.mode ?? root.mode ?? "affine";
    if (typeof bitsValue !== "number" || typeof groupSizeValue !== "number" || typeof modeValue !== "string") {
      return undefined;
    }

    const scaleTensor = `${moduleName}.scales`;
    const biasTensor = `${moduleName}.biases`;
    const globalScaleTensor = `${moduleName}.global_scale`;
    const hasKnown = (name: string): boolean => knownTensors?.has(name) ?? false;
    const storageDtype = entry?.dtype ?? knownTensors?.get(tensorName)?.storageDtype;

    // Pesos MLX quantizados são normalmente U32. Não inferimos quantização apenas pelo dtype;
    // o config é a fonte de verdade e os tensores auxiliares confirmam o layout.
    if (storageDtype && storageDtype !== "U32" && !hasKnown(scaleTensor)) return undefined;

    return {
      family: "mlx",
      mode: modeValue,
      bits: bitsValue,
      groupSize: groupSizeValue,
      ...(hasKnown(scaleTensor) ? { scaleTensor } : {}),
      ...(hasKnown(biasTensor) ? { biasTensor } : {}),
      ...(hasKnown(globalScaleTensor) ? { globalScaleTensor } : {}),
    };
  }

  #logicalShape(entry: SafeTensorHeaderEntry, quantization?: QuantizationSpec): number[] {
    if (!quantization || entry.shape.length !== 2 || !quantization.bits) return [...entry.shape];
    const [rows, packedWords] = entry.shape;
    if (rows === undefined || packedWords === undefined) return [...entry.shape];
    const logicalColumns = Math.floor((packedWords * 32) / quantization.bits);
    return [rows, logicalColumns];
  }

  #validateDenseStorage(
    name: string,
    entry: SafeTensorHeaderEntry,
    quantizationRoot: JsonObject | undefined,
    payloadLength: number,
  ): void {
    const bytes = entry.data_offsets[1] - entry.data_offsets[0];
    if (bytes < 0 || entry.data_offsets[1] > payloadLength) {
      throw new Error(`Tensor ${name}: intervalo de dados fora do payload Safetensors.`);
    }
    if (quantizationRoot && name.endsWith(".weight") && entry.dtype === "U32") return;
    const elementBytes = DTYPE_BYTES[entry.dtype];
    if (!elementBytes) return; // Dtypes novos são preservados e tratados pelo backend de referência.
    const expected = product(entry.shape) * elementBytes;
    if (expected !== bytes) {
      throw new Error(`Tensor ${name}: ${bytes} bytes, mas shape/dtype indicam ${expected}.`);
    }
  }
}

function decodeF16(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}
