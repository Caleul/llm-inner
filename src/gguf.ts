import { open } from "node:fs/promises";
import type { FileHandle } from "node:fs/promises";
import type { DenseF32Tensor, JsonObject, ModelCatalog, TensorInfo } from "./types.js";
import { product } from "./utils.js";

const GGUF_MAGIC = "GGUF";
const MAX_STRING_BYTES = 16 * 1024 * 1024;
const MAX_METADATA_ENTRIES = 1_000_000;
const MAX_TENSORS = 1_000_000;

enum GgufValueType {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12,
}

interface DirectoryTensor {
  name: string;
  dimensions: number[];
  ggmlType: number;
  offset: number;
}

/**
 * Native, read-only GGUF v2/v3 catalog reader. It deliberately recognizes
 * only dense F32/F16 and the explicitly specified Q4_0/Q4_1/Q8_0 block payloads. A
 * GGML type number is not enough to safely dequantize a packed tensor:
 * unsupported encodings fail while the directory is still being validated
 * rather than being mislabeled as dense.
 */
export class GgufCatalogReader {
  readonly #source: string;
  #handle: FileHandle | undefined;
  #fileSize = 0;
  #position = 0;

  constructor(source: string) {
    this.#source = source;
  }

  async close(): Promise<void> {
    await this.#handle?.close();
    this.#handle = undefined;
  }

  async inspect(): Promise<ModelCatalog> {
    this.#handle = await open(this.#source, "r");
    const info = await this.#handle.stat();
    if (!info.isFile()) throw new Error(`GGUF deve ser arquivo regular: ${this.#source}`);
    this.#fileSize = info.size;
    this.#position = 0;

    if (await this.#readText(4) !== GGUF_MAGIC) {
      throw new Error("Arquivo não possui magic GGUF.");
    }
    const version = await this.#readU32();
    if (version !== 2 && version !== 3) {
      throw new Error(`Versão GGUF ${version} não suportada; este leitor aceita somente v2/v3.`);
    }
    const tensorCount = await this.#readCount("tensor_count", MAX_TENSORS);
    const metadataCount = await this.#readCount("metadata_kv_count", MAX_METADATA_ENTRIES);
    const rawMetadata: JsonObject = {};
    for (let index = 0; index < metadataCount; index += 1) {
      const key = await this.#readString(`chave de metadata ${index}`);
      if (key.length === 0 || Object.hasOwn(rawMetadata, key)) {
        throw new Error(`Metadata GGUF possui chave vazia ou duplicada: '${key}'.`);
      }
      rawMetadata[key] = await this.#readMetadataValue(false);
    }

    const directory: DirectoryTensor[] = [];
    const names = new Set<string>();
    for (let index = 0; index < tensorCount; index += 1) {
      const name = await this.#readString(`nome do tensor ${index}`);
      if (name.length === 0 || names.has(name)) throw new Error(`Tensor GGUF vazio ou duplicado: '${name}'.`);
      names.add(name);
      const dimensionsCount = await this.#readU32();
      if (dimensionsCount === 0 || dimensionsCount > 4) {
        throw new Error(`${name}: GGUF declara ${dimensionsCount} dimensões; esperado 1..4.`);
      }
      const dimensions: number[] = [];
      for (let dimension = 0; dimension < dimensionsCount; dimension += 1) {
        const value = await this.#readSafeU64(`${name}.dimensions[${dimension}]`);
        if (value <= 0) throw new Error(`${name}: dimensão GGUF deve ser positiva.`);
        dimensions.push(value);
      }
      const ggmlType = await this.#readU32();
      const offset = await this.#readSafeU64(`${name}.offset`);
      directory.push({ name, dimensions, ggmlType, offset });
    }

    const alignmentValue = rawMetadata["general.alignment"];
    const alignment: unknown = alignmentValue === undefined ? 32 : alignmentValue;
    if (typeof alignment !== "number" || !Number.isSafeInteger(alignment) || alignment <= 0 || alignment > 1024 * 1024 || (alignment & (alignment - 1)) !== 0) {
      throw new Error(`general.alignment GGUF inválido: ${String(alignment)}; esperado potência de dois positiva até 1048576.`);
    }
    const dataStart = alignUp(this.#position, alignment);
    if (dataStart > this.#fileSize) throw new Error("Diretório GGUF termina além do arquivo antes do alinhamento de dados.");
    const payloadLength = this.#fileSize - dataStart;
    const tensors = new Map<string, TensorInfo>();
    const intervals: Array<{ name: string; start: number; end: number }> = [];
    for (const tensor of directory) {
      if (tensor.offset % alignment !== 0) {
        throw new Error(`${tensor.name}: offset GGUF ${tensor.offset} não é alinhado a ${alignment}.`);
      }
      const storage = storageForGgmlType(tensor.ggmlType, tensor.name);
      const elements = product(tensor.dimensions);
      if (!Number.isSafeInteger(elements) || elements <= 0) throw new Error(`${tensor.name}: produto de dimensões GGUF inválido.`);
      const byteLength = storage.byteLength(tensor.dimensions, tensor.name);
      if (!Number.isSafeInteger(byteLength) || tensor.offset > payloadLength || byteLength > payloadLength - tensor.offset) {
        throw new Error(`${tensor.name}: intervalo GGUF ultrapassa o payload declarado.`);
      }
      const start = dataStart + tensor.offset;
      const end = start + byteLength;
      intervals.push({ name: tensor.name, start, end });
      // GGUF dimensions are retained in their declared GGML order. This reader
      // does not claim a row-major logical layout or architecture tensor role.
      tensors.set(tensor.name, {
        name: tensor.name,
        storageDtype: storage.dtype,
        storageShape: [...tensor.dimensions],
        logicalShape: [...tensor.dimensions],
        byteOffset: start,
        byteLength,
        shard: this.#source,
        ...(storage.quantization ? { quantization: storage.quantization } : {}),
      });
    }
    intervals.sort((left, right) => left.start - right.start || left.end - right.end);
    for (let index = 1; index < intervals.length; index += 1) {
      const previous = intervals[index - 1]!;
      const current = intervals[index]!;
      if (current.start < previous.end) throw new Error(`${current.name}: intervalo GGUF sobrepõe ${previous.name}.`);
    }

    return {
      source: this.#source,
      format: "gguf",
      config: metadataToConfig(rawMetadata),
      rawMetadata,
      tensors,
    };
  }

  /** Materializes only verified GGML dense F32/F16 or Q4_0/Q4_1/Q8_0 intervals into F32 values. */
  async readDenseAsF32(tensor: TensorInfo): Promise<DenseF32Tensor> {
    const q4 = isGgufQ4_0(tensor);
    const q41 = isGgufQ4_1(tensor);
    const q8 = isGgufQ8_0(tensor);
    if (!q4 && !q41 && !q8 && (tensor.quantization || (tensor.storageDtype !== "F32" && tensor.storageDtype !== "F16"))) {
      throw new Error(`${tensor.name}: materialização GGUF F32 requer storage GGML F32/F16 denso ou Q4_0/Q4_1/Q8_0 verificado.`);
    }
    if (tensor.shard !== this.#source || tensor.byteOffset === undefined || tensor.byteLength === undefined) {
      throw new Error(`${tensor.name}: referência de intervalo GGUF não pertence a este leitor.`);
    }
    const elements = product(tensor.storageShape);
    const expectedBytes = q4 ? q4ByteLength(tensor.storageShape, tensor.name) : q41 ? q41ByteLength(tensor.storageShape, tensor.name) : q8 ? q8ByteLength(tensor.storageShape, tensor.name) : elements * (tensor.storageDtype === "F32" ? 4 : 2);
    if (!Number.isSafeInteger(elements) || tensor.byteLength !== expectedBytes || tensor.byteOffset < 0 || tensor.byteOffset > this.#fileSize - tensor.byteLength) {
      throw new Error(`${tensor.name}: intervalo denso GGUF não coincide com shape e dtype catalogados.`);
    }
    if (!this.#handle) throw new Error("Leitor GGUF está fechado; mantenha o catálogo aberto durante a materialização.");
    const bytes = Buffer.allocUnsafe(tensor.byteLength);
    let read = 0;
    while (read < bytes.length) {
      const result = await this.#handle.read(bytes, read, bytes.length - read, tensor.byteOffset + read);
      if (result.bytesRead === 0) throw new Error("EOF inesperado materializando tensor GGUF.");
      read += result.bytesRead;
    }
    const values = new Float32Array(elements);
    if (q4) {
      for (let block = 0; block < elements / GGML_Q4_0_BLOCK_SIZE; block += 1) {
        const offset = block * GGML_Q4_0_BLOCK_BYTES;
        const scale = decodeF16(bytes.readUInt16LE(offset));
        // block_q4_0 stores q[0..15] in low nibbles then q[16..31] in high nibbles.
        for (let index = 0; index < GGML_Q4_0_BLOCK_SIZE / 2; index += 1) {
          const packed = bytes[offset + 2 + index]!;
          values[block * GGML_Q4_0_BLOCK_SIZE + index] = scale * ((packed & 0x0f) - 8);
          values[block * GGML_Q4_0_BLOCK_SIZE + 16 + index] = scale * ((packed >>> 4) - 8);
        }
      }
    } else if (q41) {
      for (let block = 0; block < elements / GGML_Q4_1_BLOCK_SIZE; block += 1) {
        const offset = block * GGML_Q4_1_BLOCK_BYTES;
        const scale = decodeF16(bytes.readUInt16LE(offset));
        const minimum = decodeF16(bytes.readUInt16LE(offset + 2));
        // block_q4_1 stores q[0..15] in low nibbles then q[16..31] in high nibbles.
        for (let index = 0; index < GGML_Q4_1_BLOCK_SIZE / 2; index += 1) {
          const packed = bytes[offset + 4 + index]!;
          values[block * GGML_Q4_1_BLOCK_SIZE + index] = scale * (packed & 0x0f) + minimum;
          values[block * GGML_Q4_1_BLOCK_SIZE + 16 + index] = scale * (packed >>> 4) + minimum;
        }
      }
    } else if (q8) {
      for (let block = 0; block < elements / GGML_Q8_0_BLOCK_SIZE; block += 1) {
        const offset = block * GGML_Q8_0_BLOCK_BYTES;
        const scale = decodeF16(bytes.readUInt16LE(offset));
        for (let index = 0; index < GGML_Q8_0_BLOCK_SIZE; index += 1) {
          values[block * GGML_Q8_0_BLOCK_SIZE + index] = scale * bytes.readInt8(offset + 2 + index);
        }
      }
    } else if (tensor.storageDtype === "F32") {
      for (let index = 0; index < elements; index += 1) values[index] = bytes.readFloatLE(index * 4);
    } else {
      for (let index = 0; index < elements; index += 1) values[index] = decodeF16(bytes.readUInt16LE(index * 2));
    }
    return { shape: [...tensor.logicalShape], values, ...((q4 || q41 || q8) ? { sourceQuantization: { ...tensor.quantization! } } : {}) };
  }

  async #readMetadataValue(inArray: boolean): Promise<unknown> {
    const type = await this.#readU32();
    if (type === GgufValueType.Array) {
      if (inArray) throw new Error("GGUF não permite arrays aninhados em metadata.");
      const elementType = await this.#readU32();
      if (elementType === GgufValueType.Array) throw new Error("GGUF metadata array não pode conter arrays.");
      const count = await this.#readCount("metadata array", MAX_METADATA_ENTRIES);
      const values: unknown[] = [];
      for (let index = 0; index < count; index += 1) values.push(await this.#readValue(elementType, true));
      return values;
    }
    return this.#readValue(type, inArray);
  }

  async #readValue(type: number, _inArray: boolean): Promise<unknown> {
    switch (type) {
      case GgufValueType.Uint8: return (await this.#readBytes(1))[0]!;
      case GgufValueType.Int8: return (await this.#readBytes(1)).readInt8(0);
      case GgufValueType.Uint16: return (await this.#readBytes(2)).readUInt16LE(0);
      case GgufValueType.Int16: return (await this.#readBytes(2)).readInt16LE(0);
      case GgufValueType.Uint32: return await this.#readU32();
      case GgufValueType.Int32: return (await this.#readBytes(4)).readInt32LE(0);
      case GgufValueType.Float32: return (await this.#readBytes(4)).readFloatLE(0);
      case GgufValueType.Bool: {
        const value = (await this.#readBytes(1))[0]!;
        if (value !== 0 && value !== 1) throw new Error(`Booleano GGUF inválido: ${value}.`);
        return value === 1;
      }
      case GgufValueType.String: return this.#readString("valor string de metadata");
      case GgufValueType.Uint64: return this.#readIntegerMetadata(false);
      case GgufValueType.Int64: return this.#readIntegerMetadata(true);
      case GgufValueType.Float64: return (await this.#readBytes(8)).readDoubleLE(0);
      default: throw new Error(`Tipo de metadata GGUF não suportado: ${type}.`);
    }
  }

  async #readIntegerMetadata(signed: boolean): Promise<number | string> {
    const value = signed ? (await this.#readBytes(8)).readBigInt64LE(0) : (await this.#readBytes(8)).readBigUInt64LE(0);
    return value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value.toString();
  }

  async #readCount(label: string, maximum: number): Promise<number> {
    const count = await this.#readSafeU64(label);
    if (count > maximum) throw new Error(`${label} GGUF excede o limite local de ${maximum}.`);
    return count;
  }

  async #readSafeU64(label: string): Promise<number> {
    const value = (await this.#readBytes(8)).readBigUInt64LE(0);
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} GGUF excede Number.MAX_SAFE_INTEGER.`);
    return Number(value);
  }

  async #readU32(): Promise<number> { return (await this.#readBytes(4)).readUInt32LE(0); }

  async #readString(label: string): Promise<string> {
    const length = await this.#readSafeU64(`${label}.length`);
    if (length > MAX_STRING_BYTES) throw new Error(`${label} GGUF excede ${MAX_STRING_BYTES} bytes.`);
    return this.#readText(length);
  }

  async #readText(length: number): Promise<string> {
    const bytes = await this.#readBytes(length);
    try { return new TextDecoder("utf-8", { fatal: true }).decode(bytes); } catch { throw new Error("String GGUF não é UTF-8 válido."); }
  }

  async #readBytes(length: number): Promise<Buffer> {
    if (!Number.isSafeInteger(length) || length < 0 || length > this.#fileSize - this.#position) throw new Error("EOF inesperado lendo GGUF.");
    const bytes = Buffer.allocUnsafe(length);
    let read = 0;
    while (read < length) {
      const result = await this.#handle!.read(bytes, read, length - read, this.#position + read);
      if (result.bytesRead === 0) throw new Error("EOF inesperado lendo GGUF.");
      read += result.bytesRead;
    }
    this.#position += length;
    return bytes;
  }
}

interface GgmlStorage {
  dtype: string;
  quantization?: import("./types.js").QuantizationSpec;
  byteLength(dimensions: readonly number[], name: string): number;
}

const GGML_Q8_0_BLOCK_SIZE = 32;
const GGML_Q8_0_BLOCK_BYTES = 34; // ggml_half d followed by 32 signed int8 quants.
const GGML_Q4_0_BLOCK_SIZE = 32;
const GGML_Q4_0_BLOCK_BYTES = 18; // ggml_half d followed by 16 packed unsigned nibbles.
const GGML_Q4_1_BLOCK_SIZE = 32;
const GGML_Q4_1_BLOCK_BYTES = 20; // ggml_half d, ggml_half m, then 16 packed unsigned nibbles.

function storageForGgmlType(type: number, name: string): GgmlStorage {
  if (type === 0) return { dtype: "F32", byteLength: (dimensions) => product([...dimensions]) * 4 }; // GGML_TYPE_F32
  if (type === 1) return { dtype: "F16", byteLength: (dimensions) => product([...dimensions]) * 2 }; // GGML_TYPE_F16
  if (type === 2) {
    return {
      dtype: "GGML_Q4_0",
      quantization: { family: "gguf", mode: "q4_0", bits: 4, groupSize: GGML_Q4_0_BLOCK_SIZE, tensorType: "GGML_TYPE_Q4_0" },
      byteLength: q4ByteLength,
    };
  }
  if (type === 3) {
    return {
      dtype: "GGML_Q4_1",
      quantization: { family: "gguf", mode: "q4_1", bits: 4, groupSize: GGML_Q4_1_BLOCK_SIZE, tensorType: "GGML_TYPE_Q4_1" },
      byteLength: q41ByteLength,
    };
  }
  if (type === 8) {
    return {
      dtype: "GGML_Q8_0",
      quantization: { family: "gguf", mode: "q8_0", bits: 8, groupSize: GGML_Q8_0_BLOCK_SIZE, tensorType: "GGML_TYPE_Q8_0" },
      byteLength: q8ByteLength,
    };
  }
  const label = GGML_TYPE_NAMES[type] ?? `GGML_TYPE_${type}`;
  throw new Error(`${name}: ${label} é um encoding GGML sem decodificador/layout verificado; catálogo rejeitado.`);
}

function q4ByteLength(dimensions: readonly number[], name: string): number {
  const elements = product([...dimensions]);
  if (!Number.isSafeInteger(elements) || elements <= 0 || dimensions[0] === undefined || dimensions[0] % GGML_Q4_0_BLOCK_SIZE !== 0) {
    throw new Error(`${name}: GGML_TYPE_Q4_0 exige a primeira dimensão GGML positiva e múltipla de ${GGML_Q4_0_BLOCK_SIZE}.`);
  }
  return (elements / GGML_Q4_0_BLOCK_SIZE) * GGML_Q4_0_BLOCK_BYTES;
}

function q8ByteLength(dimensions: readonly number[], name: string): number {
  const elements = product([...dimensions]);
  if (!Number.isSafeInteger(elements) || elements <= 0 || dimensions[0] === undefined || dimensions[0] % GGML_Q8_0_BLOCK_SIZE !== 0) {
    throw new Error(`${name}: GGML_TYPE_Q8_0 exige a primeira dimensão GGML positiva e múltipla de ${GGML_Q8_0_BLOCK_SIZE}.`);
  }
  return (elements / GGML_Q8_0_BLOCK_SIZE) * GGML_Q8_0_BLOCK_BYTES;
}

function q41ByteLength(dimensions: readonly number[], name: string): number {
  const elements = product([...dimensions]);
  if (!Number.isSafeInteger(elements) || elements <= 0 || dimensions[0] === undefined || dimensions[0] % GGML_Q4_1_BLOCK_SIZE !== 0) {
    throw new Error(`${name}: GGML_TYPE_Q4_1 exige a primeira dimensão GGML positiva e múltipla de ${GGML_Q4_1_BLOCK_SIZE}.`);
  }
  return (elements / GGML_Q4_1_BLOCK_SIZE) * GGML_Q4_1_BLOCK_BYTES;
}

function isGgufQ8_0(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q8_0" && quantization?.family === "gguf" && quantization.mode === "q8_0" &&
    quantization.bits === 8 && quantization.groupSize === GGML_Q8_0_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q8_0";
}

function isGgufQ4_0(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q4_0" && quantization?.family === "gguf" && quantization.mode === "q4_0" &&
    quantization.bits === 4 && quantization.groupSize === GGML_Q4_0_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q4_0";
}

function isGgufQ4_1(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q4_1" && quantization?.family === "gguf" && quantization.mode === "q4_1" &&
    quantization.bits === 4 && quantization.groupSize === GGML_Q4_1_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q4_1";
}

const GGML_TYPE_NAMES: Record<number, string> = {
  2: "GGML_TYPE_Q4_0", 3: "GGML_TYPE_Q4_1", 6: "GGML_TYPE_Q5_0", 7: "GGML_TYPE_Q5_1",
  8: "GGML_TYPE_Q8_0", 9: "GGML_TYPE_Q8_1", 10: "GGML_TYPE_Q2_K", 11: "GGML_TYPE_Q3_K",
  12: "GGML_TYPE_Q4_K", 13: "GGML_TYPE_Q5_K", 14: "GGML_TYPE_Q6_K", 15: "GGML_TYPE_Q8_K",
  16: "GGML_TYPE_IQ2_XXS", 17: "GGML_TYPE_IQ2_XS", 18: "GGML_TYPE_IQ3_XXS", 19: "GGML_TYPE_IQ1_S",
  20: "GGML_TYPE_IQ4_NL", 21: "GGML_TYPE_IQ3_S", 22: "GGML_TYPE_IQ2_S", 23: "GGML_TYPE_IQ4_XS",
  24: "GGML_TYPE_IQ1_M", 25: "GGML_TYPE_BF16", 26: "GGML_TYPE_Q4_0_4_4", 27: "GGML_TYPE_Q4_0_4_8",
  28: "GGML_TYPE_Q4_0_8_8", 29: "GGML_TYPE_TQ1_0", 30: "GGML_TYPE_TQ2_0",
};

function metadataToConfig(metadata: JsonObject): JsonObject {
  const architecture = metadata["general.architecture"];
  if (typeof architecture !== "string" || architecture.length === 0) return {};
  const config: JsonObject = { model_type: architecture };
  const prefix = `${architecture}.`;
  for (const [key, value] of Object.entries(metadata)) {
    if (!key.startsWith(prefix)) continue;
    const shortName = key.slice(prefix.length);
    config[shortName] = value;
    config[shortName.replaceAll(".", "_")] = value;
  }
  return config;
}

function alignUp(value: number, alignment: number): number {
  const remainder = value % alignment;
  const aligned = remainder === 0 ? value : value + alignment - remainder;
  if (!Number.isSafeInteger(aligned)) throw new Error("Alinhamento GGUF excede Number.MAX_SAFE_INTEGER.");
  return aligned;
}

function decodeF16(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}
