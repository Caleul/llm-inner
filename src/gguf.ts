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
 * only dense F32/F16/BF16 and the explicitly specified Q3_K/Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0/Q8_1/Q8_K block payloads. A
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

  /** Materializes only verified GGML dense F32/F16/BF16 or Q3_K/Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0/Q8_1/Q8_K intervals into F32 values. */
  async readDenseAsF32(tensor: TensorInfo): Promise<DenseF32Tensor> {
    const q3k = isGgufQ3_K(tensor);
    const q4 = isGgufQ4_0(tensor);
    const q41 = isGgufQ4_1(tensor);
    const q4k = isGgufQ4_K(tensor);
    const q5 = isGgufQ5_0(tensor);
    const q51 = isGgufQ5_1(tensor);
    const q5k = isGgufQ5_K(tensor);
    const q6k = isGgufQ6_K(tensor);
    const q8 = isGgufQ8_0(tensor);
    const q81 = isGgufQ8_1(tensor);
    const q8k = isGgufQ8_K(tensor);
    if (!q3k && !q4 && !q41 && !q4k && !q5 && !q51 && !q5k && !q6k && !q8 && !q81 && !q8k && (tensor.quantization || (tensor.storageDtype !== "F32" && tensor.storageDtype !== "F16" && tensor.storageDtype !== "BF16"))) {
      throw new Error(`${tensor.name}: materialização GGUF F32 requer storage GGML F32/F16/BF16 denso ou Q3_K/Q4_0/Q4_1/Q4_K/Q5_0/Q5_1/Q5_K/Q6_K/Q8_0/Q8_1/Q8_K verificado.`);
    }
    if (tensor.shard !== this.#source || tensor.byteOffset === undefined || tensor.byteLength === undefined) {
      throw new Error(`${tensor.name}: referência de intervalo GGUF não pertence a este leitor.`);
    }
    const elements = product(tensor.storageShape);
    const expectedBytes = q3k ? q3kByteLength(tensor.storageShape, tensor.name) : q4 ? q4ByteLength(tensor.storageShape, tensor.name) : q41 ? q41ByteLength(tensor.storageShape, tensor.name) : q4k ? q4kByteLength(tensor.storageShape, tensor.name) : q5 ? q5ByteLength(tensor.storageShape, tensor.name) : q51 ? q51ByteLength(tensor.storageShape, tensor.name) : q5k ? q5kByteLength(tensor.storageShape, tensor.name) : q6k ? q6kByteLength(tensor.storageShape, tensor.name) : q8 ? q8ByteLength(tensor.storageShape, tensor.name) : q81 ? q81ByteLength(tensor.storageShape, tensor.name) : q8k ? q8kByteLength(tensor.storageShape, tensor.name) : elements * (tensor.storageDtype === "F32" ? 4 : 2);
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
    if (q3k) {
      for (let block = 0; block < elements / GGML_Q3_K_BLOCK_SIZE; block += 1) {
        const offset = block * GGML_Q3_K_BLOCK_BYTES;
        // block_q3_K stores a base F16 scale, 16 signed six-bit scales packed
        // in 12 bytes, a 32-byte high-bit mask, and four 2-bit code planes.
        // The high-bit mask maps logical index i to hmask[i % 32] bit i / 32.
        const baseScale = decodeF16(bytes.readUInt16LE(offset));
        for (let index = 0; index < GGML_Q3_K_BLOCK_SIZE; index += 1) {
          const scaleCode = getSignedScaleQ3K(bytes, offset + 98, Math.floor(index / 16));
          const plane = Math.floor((index % 128) / 32);
          const packed = bytes[offset + 34 + Math.floor(index / 128) * 32 + (index % 32)]!;
          const lowCode = (packed >>> (plane * 2)) & 0x03;
          const highBit = (bytes[offset + 2 + (index % 32)]! >>> Math.floor(index / 32)) & 0x01;
          values[block * GGML_Q3_K_BLOCK_SIZE + index] = baseScale * scaleCode * (lowCode - (highBit === 1 ? 0 : 4));
        }
      }
    } else if (q4) {
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
    } else if (q4k) {
      for (let block = 0; block < elements / GGML_Q4_K_BLOCK_SIZE; block += 1) {
        const offset = block * GGML_Q4_K_BLOCK_BYTES;
        // block_q4_K has F16 d, F16 dmin, 12 packed scale/minimum bytes and
        // 128 code bytes. Eight 32-value groups use d*scale*q - dmin*minimum.
        // getScaleMinimumQ4K mirrors GGML's six-bit fields: groups 0..3 use
        // low six bits directly; groups 4..7 join their high two bits from the
        // first eight bytes with their low four-bit fields in bytes 8..11.
        const scaleBase = decodeF16(bytes.readUInt16LE(offset));
        const minimumBase = decodeF16(bytes.readUInt16LE(offset + 2));
        for (let group = 0; group < 8; group += 1) {
          const { scaleCode, minimumCode } = getScaleMinimumQ4K(bytes, offset + 4, group);
          const scale = scaleBase * scaleCode;
          const minimum = minimumBase * minimumCode;
          const codeOffset = offset + 16 + Math.floor(group / 2) * 32;
          const highNibble = group % 2 === 1;
          for (let index = 0; index < 32; index += 1) {
            const packed = bytes[codeOffset + index]!;
            const code = highNibble ? packed >>> 4 : packed & 0x0f;
            values[block * GGML_Q4_K_BLOCK_SIZE + group * 32 + index] = scale * code - minimum;
          }
        }
      }
    } else if (q5) {
      for (let block = 0; block < elements / GGML_Q5_0_BLOCK_SIZE; block += 1) {
        const offset = block * GGML_Q5_0_BLOCK_BYTES;
        const scale = decodeF16(bytes.readUInt16LE(offset));
        // block_q5_0 stores qh bit i as the fifth bit of q[i], followed by
        // qs low nibbles for q[0..15] and high nibbles for q[16..31].
        for (let index = 0; index < GGML_Q5_0_BLOCK_SIZE / 2; index += 1) {
          const packed = bytes[offset + 6 + index]!;
          const lowHighBit = (bytes[offset + 2 + Math.floor(index / 8)]! >>> (index % 8)) & 1;
          const highIndex = 16 + index;
          const highHighBit = (bytes[offset + 2 + Math.floor(highIndex / 8)]! >>> (highIndex % 8)) & 1;
          values[block * GGML_Q5_0_BLOCK_SIZE + index] = scale * (((packed & 0x0f) | (lowHighBit << 4)) - 16);
          values[block * GGML_Q5_0_BLOCK_SIZE + highIndex] = scale * (((packed >>> 4) | (highHighBit << 4)) - 16);
        }
      }
    } else if (q51) {
      for (let block = 0; block < elements / GGML_Q5_1_BLOCK_SIZE; block += 1) {
        const offset = block * GGML_Q5_1_BLOCK_BYTES;
        const scale = decodeF16(bytes.readUInt16LE(offset));
        const minimum = decodeF16(bytes.readUInt16LE(offset + 2));
        // block_q5_1 stores qh bit i as the fifth bit of q[i], then qs low
        // nibbles for q[0..15] and high nibbles for q[16..31]. Unlike Q5_0,
        // the reconstructed five-bit code is unsigned and affine.
        for (let index = 0; index < GGML_Q5_1_BLOCK_SIZE / 2; index += 1) {
          const packed = bytes[offset + 8 + index]!;
          const lowHighBit = (bytes[offset + 4 + Math.floor(index / 8)]! >>> (index % 8)) & 1;
          const highIndex = 16 + index;
          const highHighBit = (bytes[offset + 4 + Math.floor(highIndex / 8)]! >>> (highIndex % 8)) & 1;
          values[block * GGML_Q5_1_BLOCK_SIZE + index] = scale * ((packed & 0x0f) | (lowHighBit << 4)) + minimum;
          values[block * GGML_Q5_1_BLOCK_SIZE + highIndex] = scale * ((packed >>> 4) | (highHighBit << 4)) + minimum;
        }
      }
    } else if (q5k) {
      for (let block = 0; block < elements / GGML_Q5_K_BLOCK_SIZE; block += 1) {
        const offset = block * GGML_Q5_K_BLOCK_BYTES;
        // block_q5_K extends Q4_K's scale/minimum hierarchy with qh[32],
        // where bit i is code q[i]'s fifth bit. qs[128] keeps Q4_K's paired
        // low-nibble planes, so each 32-value group remains independently
        // affine: d*scale*(low4 | high1<<4) - dmin*minimum.
        const scaleBase = decodeF16(bytes.readUInt16LE(offset));
        const minimumBase = decodeF16(bytes.readUInt16LE(offset + 2));
        for (let group = 0; group < 8; group += 1) {
          const { scaleCode, minimumCode } = getScaleMinimumQ4K(bytes, offset + 4, group);
          const scale = scaleBase * scaleCode;
          const minimum = minimumBase * minimumCode;
          const codeOffset = offset + 48 + Math.floor(group / 2) * 32;
          const highNibble = group % 2 === 1;
          for (let index = 0; index < 32; index += 1) {
            const blockIndex = group * 32 + index;
            const packed = bytes[codeOffset + index]!;
            const lowCode = highNibble ? packed >>> 4 : packed & 0x0f;
            const highBit = (bytes[offset + 16 + Math.floor(blockIndex / 8)]! >>> (blockIndex % 8)) & 1;
            values[block * GGML_Q5_K_BLOCK_SIZE + blockIndex] = scale * (lowCode | (highBit << 4)) - minimum;
          }
        }
      }
    } else if (q6k) {
      for (let block = 0; block < elements / GGML_Q6_K_BLOCK_SIZE; block += 1) {
        const offset = block * GGML_Q6_K_BLOCK_BYTES;
        // block_q6_K has ql[128], qh[64], signed scales[16], then half d.
        // ql supplies four low bits and qh supplies two high bits in eight
        // 32-value planes. Each signed scale applies to the next 16 values.
        const scale = decodeF16(bytes.readUInt16LE(offset + 208));
        for (let plane = 0; plane < 8; plane += 1) {
          const lowByteBase = offset + Math.floor(plane / 4) * 64 + (plane % 2) * 32;
          const lowShift = plane % 4 >= 2 ? 4 : 0;
          const highByteBase = offset + 128 + Math.floor(plane / 4) * 32;
          const highShift = (plane % 4) * 2;
          for (let index = 0; index < 32; index += 1) {
            const low = (bytes[lowByteBase + index]! >>> lowShift) & 0x0f;
            const high = (bytes[highByteBase + index]! >>> highShift) & 0x03;
            const signedCode = (low | (high << 4)) - 32;
            const blockIndex = plane * 32 + index;
            const groupScale = bytes.readInt8(offset + 192 + Math.floor(blockIndex / 16));
            values[block * GGML_Q6_K_BLOCK_SIZE + blockIndex] = scale * groupScale * signedCode;
          }
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
    } else if (q81) {
      for (let block = 0; block < elements / GGML_Q8_1_BLOCK_SIZE; block += 1) {
        const offset = block * GGML_Q8_1_BLOCK_BYTES;
        // block_q8_1 has binary32 d and s=d*sum(qs), then 32 signed codes.
        // `s` is an auxiliary dot-product field, not part of per-element
        // reconstruction; it must not be mistaken for Q8_0's F16 scale.
        const scale = bytes.readFloatLE(offset);
        for (let index = 0; index < GGML_Q8_1_BLOCK_SIZE; index += 1) {
          values[block * GGML_Q8_1_BLOCK_SIZE + index] = scale * bytes.readInt8(offset + 8 + index);
        }
      }
    } else if (q8k) {
      for (let block = 0; block < elements / GGML_Q8_K_BLOCK_SIZE; block += 1) {
        const offset = block * GGML_Q8_K_BLOCK_BYTES;
        // block_q8_K is deliberately not Q8_0/Q8_1: its 256 signed codes
        // have one IEEE-754 binary32 scale, with no F16 conversion or
        // auxiliary sum field in the block contract.
        const scale = bytes.readFloatLE(offset);
        for (let index = 0; index < GGML_Q8_K_BLOCK_SIZE; index += 1) {
          values[block * GGML_Q8_K_BLOCK_SIZE + index] = scale * bytes.readInt8(offset + 4 + index);
        }
      }
    } else if (tensor.storageDtype === "F32") {
      for (let index = 0; index < elements; index += 1) values[index] = bytes.readFloatLE(index * 4);
    } else if (tensor.storageDtype === "F16") {
      for (let index = 0; index < elements; index += 1) values[index] = decodeF16(bytes.readUInt16LE(index * 2));
    } else {
      for (let index = 0; index < elements; index += 1) values[index] = decodeBF16(bytes.readUInt16LE(index * 2));
    }
    return { shape: [...tensor.logicalShape], values, ...((q3k || q4 || q41 || q4k || q5 || q51 || q5k || q6k || q8 || q81 || q8k) ? { sourceQuantization: { ...tensor.quantization! } } : {}) };
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
const GGML_Q8_1_BLOCK_SIZE = 32;
const GGML_Q8_1_BLOCK_BYTES = 40; // float d, float s=d*sum(qs), then 32 signed int8 quants.
const GGML_Q4_0_BLOCK_SIZE = 32;
const GGML_Q4_0_BLOCK_BYTES = 18; // ggml_half d followed by 16 packed unsigned nibbles.
const GGML_Q4_1_BLOCK_SIZE = 32;
const GGML_Q4_1_BLOCK_BYTES = 20; // ggml_half d, ggml_half m, then 16 packed unsigned nibbles.
const GGML_Q3_K_BLOCK_SIZE = 256;
const GGML_Q3_K_BLOCK_BYTES = 110; // F16 d, hmask[32], qs[64] packed 2-bit planes, scales[12] packed signed six-bit fields.
const GGML_Q4_K_BLOCK_SIZE = 256;
const GGML_Q4_K_BLOCK_BYTES = 144; // F16 d, F16 dmin, 12 packed scale/minimum fields, then 128 packed codes.
const GGML_Q5_0_BLOCK_SIZE = 32;
const GGML_Q5_0_BLOCK_BYTES = 22; // ggml_half d, 4-byte high-bit plane, then 16 packed low nibbles.
const GGML_Q5_1_BLOCK_SIZE = 32;
const GGML_Q5_1_BLOCK_BYTES = 24; // ggml_half d, ggml_half m, 4-byte high-bit plane, then 16 packed low nibbles.
const GGML_Q5_K_BLOCK_SIZE = 256;
const GGML_Q5_K_BLOCK_BYTES = 176; // F16 d, F16 dmin, 12 packed scale/minimum fields, qh[32], then qs[128].
const GGML_Q6_K_BLOCK_SIZE = 256;
const GGML_Q6_K_BLOCK_BYTES = 210; // ql[128], qh[64], signed scales[16], then ggml_half d.
const GGML_Q8_K_BLOCK_SIZE = 256;
const GGML_Q8_K_BLOCK_BYTES = 260; // float d followed by 256 signed int8 quants.

function storageForGgmlType(type: number, name: string): GgmlStorage {
  if (type === 0) return { dtype: "F32", byteLength: (dimensions) => product([...dimensions]) * 4 }; // GGML_TYPE_F32
  if (type === 1) return { dtype: "F16", byteLength: (dimensions) => product([...dimensions]) * 2 }; // GGML_TYPE_F16
  if (type === 25) return { dtype: "BF16", byteLength: (dimensions) => product([...dimensions]) * 2 }; // GGML_TYPE_BF16
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
  if (type === 11) {
    return {
      dtype: "GGML_Q3_K",
      quantization: { family: "gguf", mode: "q3_k", bits: 3, groupSize: GGML_Q3_K_BLOCK_SIZE, tensorType: "GGML_TYPE_Q3_K" },
      byteLength: q3kByteLength,
    };
  }
  if (type === 12) {
    return {
      dtype: "GGML_Q4_K",
      quantization: { family: "gguf", mode: "q4_k", bits: 4, groupSize: GGML_Q4_K_BLOCK_SIZE, tensorType: "GGML_TYPE_Q4_K" },
      byteLength: q4kByteLength,
    };
  }
  if (type === 6) {
    return {
      dtype: "GGML_Q5_0",
      quantization: { family: "gguf", mode: "q5_0", bits: 5, groupSize: GGML_Q5_0_BLOCK_SIZE, tensorType: "GGML_TYPE_Q5_0" },
      byteLength: q5ByteLength,
    };
  }
  if (type === 7) {
    return {
      dtype: "GGML_Q5_1",
      quantization: { family: "gguf", mode: "q5_1", bits: 5, groupSize: GGML_Q5_1_BLOCK_SIZE, tensorType: "GGML_TYPE_Q5_1" },
      byteLength: q51ByteLength,
    };
  }
  if (type === 13) {
    return {
      dtype: "GGML_Q5_K",
      quantization: { family: "gguf", mode: "q5_k", bits: 5, groupSize: GGML_Q5_K_BLOCK_SIZE, tensorType: "GGML_TYPE_Q5_K" },
      byteLength: q5kByteLength,
    };
  }
  if (type === 14) {
    return {
      dtype: "GGML_Q6_K",
      quantization: { family: "gguf", mode: "q6_k", bits: 6, groupSize: GGML_Q6_K_BLOCK_SIZE, tensorType: "GGML_TYPE_Q6_K" },
      byteLength: q6kByteLength,
    };
  }
  if (type === 15) {
    return {
      dtype: "GGML_Q8_K",
      quantization: { family: "gguf", mode: "q8_k", bits: 8, groupSize: GGML_Q8_K_BLOCK_SIZE, tensorType: "GGML_TYPE_Q8_K" },
      byteLength: q8kByteLength,
    };
  }
  if (type === 8) {
    return {
      dtype: "GGML_Q8_0",
      quantization: { family: "gguf", mode: "q8_0", bits: 8, groupSize: GGML_Q8_0_BLOCK_SIZE, tensorType: "GGML_TYPE_Q8_0" },
      byteLength: q8ByteLength,
    };
  }
  if (type === 9) {
    return {
      dtype: "GGML_Q8_1",
      quantization: { family: "gguf", mode: "q8_1", bits: 8, groupSize: GGML_Q8_1_BLOCK_SIZE, tensorType: "GGML_TYPE_Q8_1" },
      byteLength: q81ByteLength,
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

function q81ByteLength(dimensions: readonly number[], name: string): number {
  const elements = product([...dimensions]);
  if (!Number.isSafeInteger(elements) || elements <= 0 || dimensions[0] === undefined || dimensions[0] % GGML_Q8_1_BLOCK_SIZE !== 0) {
    throw new Error(`${name}: GGML_TYPE_Q8_1 exige a primeira dimensão GGML positiva e múltipla de ${GGML_Q8_1_BLOCK_SIZE}.`);
  }
  return (elements / GGML_Q8_1_BLOCK_SIZE) * GGML_Q8_1_BLOCK_BYTES;
}

function q41ByteLength(dimensions: readonly number[], name: string): number {
  const elements = product([...dimensions]);
  if (!Number.isSafeInteger(elements) || elements <= 0 || dimensions[0] === undefined || dimensions[0] % GGML_Q4_1_BLOCK_SIZE !== 0) {
    throw new Error(`${name}: GGML_TYPE_Q4_1 exige a primeira dimensão GGML positiva e múltipla de ${GGML_Q4_1_BLOCK_SIZE}.`);
  }
  return (elements / GGML_Q4_1_BLOCK_SIZE) * GGML_Q4_1_BLOCK_BYTES;
}

function q3kByteLength(dimensions: readonly number[], name: string): number {
  const elements = product([...dimensions]);
  if (!Number.isSafeInteger(elements) || elements <= 0 || dimensions[0] === undefined || dimensions[0] % GGML_Q3_K_BLOCK_SIZE !== 0) {
    throw new Error(`${name}: GGML_TYPE_Q3_K exige a primeira dimensão GGML positiva e múltipla de ${GGML_Q3_K_BLOCK_SIZE}.`);
  }
  return (elements / GGML_Q3_K_BLOCK_SIZE) * GGML_Q3_K_BLOCK_BYTES;
}

function q4kByteLength(dimensions: readonly number[], name: string): number {
  const elements = product([...dimensions]);
  if (!Number.isSafeInteger(elements) || elements <= 0 || dimensions[0] === undefined || dimensions[0] % GGML_Q4_K_BLOCK_SIZE !== 0) {
    throw new Error(`${name}: GGML_TYPE_Q4_K exige a primeira dimensão GGML positiva e múltipla de ${GGML_Q4_K_BLOCK_SIZE}.`);
  }
  return (elements / GGML_Q4_K_BLOCK_SIZE) * GGML_Q4_K_BLOCK_BYTES;
}

function q5ByteLength(dimensions: readonly number[], name: string): number {
  const elements = product([...dimensions]);
  if (!Number.isSafeInteger(elements) || elements <= 0 || dimensions[0] === undefined || dimensions[0] % GGML_Q5_0_BLOCK_SIZE !== 0) {
    throw new Error(`${name}: GGML_TYPE_Q5_0 exige a primeira dimensão GGML positiva e múltipla de ${GGML_Q5_0_BLOCK_SIZE}.`);
  }
  return (elements / GGML_Q5_0_BLOCK_SIZE) * GGML_Q5_0_BLOCK_BYTES;
}

function q51ByteLength(dimensions: readonly number[], name: string): number {
  const elements = product([...dimensions]);
  if (!Number.isSafeInteger(elements) || elements <= 0 || dimensions[0] === undefined || dimensions[0] % GGML_Q5_1_BLOCK_SIZE !== 0) {
    throw new Error(`${name}: GGML_TYPE_Q5_1 exige a primeira dimensão GGML positiva e múltipla de ${GGML_Q5_1_BLOCK_SIZE}.`);
  }
  return (elements / GGML_Q5_1_BLOCK_SIZE) * GGML_Q5_1_BLOCK_BYTES;
}

function q5kByteLength(dimensions: readonly number[], name: string): number {
  const elements = product([...dimensions]);
  if (!Number.isSafeInteger(elements) || elements <= 0 || dimensions[0] === undefined || dimensions[0] % GGML_Q5_K_BLOCK_SIZE !== 0) {
    throw new Error(`${name}: GGML_TYPE_Q5_K exige a primeira dimensão GGML positiva e múltipla de ${GGML_Q5_K_BLOCK_SIZE}.`);
  }
  return (elements / GGML_Q5_K_BLOCK_SIZE) * GGML_Q5_K_BLOCK_BYTES;
}

function q6kByteLength(dimensions: readonly number[], name: string): number {
  const elements = product([...dimensions]);
  if (!Number.isSafeInteger(elements) || elements <= 0 || dimensions[0] === undefined || dimensions[0] % GGML_Q6_K_BLOCK_SIZE !== 0) {
    throw new Error(`${name}: GGML_TYPE_Q6_K exige a primeira dimensão GGML positiva e múltipla de ${GGML_Q6_K_BLOCK_SIZE}.`);
  }
  return (elements / GGML_Q6_K_BLOCK_SIZE) * GGML_Q6_K_BLOCK_BYTES;
}

function q8kByteLength(dimensions: readonly number[], name: string): number {
  const elements = product([...dimensions]);
  if (!Number.isSafeInteger(elements) || elements <= 0 || dimensions[0] === undefined || dimensions[0] % GGML_Q8_K_BLOCK_SIZE !== 0) {
    throw new Error(`${name}: GGML_TYPE_Q8_K exige a primeira dimensão GGML positiva e múltipla de ${GGML_Q8_K_BLOCK_SIZE}.`);
  }
  return (elements / GGML_Q8_K_BLOCK_SIZE) * GGML_Q8_K_BLOCK_BYTES;
}

function isGgufQ8_0(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q8_0" && quantization?.family === "gguf" && quantization.mode === "q8_0" &&
    quantization.bits === 8 && quantization.groupSize === GGML_Q8_0_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q8_0";
}

function isGgufQ8_1(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q8_1" && quantization?.family === "gguf" && quantization.mode === "q8_1" &&
    quantization.bits === 8 && quantization.groupSize === GGML_Q8_1_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q8_1";
}

function isGgufQ4_0(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q4_0" && quantization?.family === "gguf" && quantization.mode === "q4_0" &&
    quantization.bits === 4 && quantization.groupSize === GGML_Q4_0_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q4_0";
}

function isGgufQ3_K(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q3_K" && quantization?.family === "gguf" && quantization.mode === "q3_k" &&
    quantization.bits === 3 && quantization.groupSize === GGML_Q3_K_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q3_K";
}

/** Decodes the signed six-bit scale for one of Q3_K's sixteen 16-value groups. */
function getSignedScaleQ3K(bytes: Uint8Array, scalesOffset: number, group: number): number {
  if (!Number.isInteger(group) || group < 0 || group >= 16) throw new Error(`Q3_K group inválido: ${group}.`);
  const low = group < 8 ? bytes[scalesOffset + group]! & 0x0f : bytes[scalesOffset + group - 8]! >>> 4;
  const high = (bytes[scalesOffset + 8 + (group % 4)]! >>> (2 * Math.floor(group / 4))) & 0x03;
  return (low | (high << 4)) - 32;
}

function isGgufQ4_1(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q4_1" && quantization?.family === "gguf" && quantization.mode === "q4_1" &&
    quantization.bits === 4 && quantization.groupSize === GGML_Q4_1_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q4_1";
}

function isGgufQ4_K(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q4_K" && quantization?.family === "gguf" && quantization.mode === "q4_k" &&
    quantization.bits === 4 && quantization.groupSize === GGML_Q4_K_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q4_K";
}

function getScaleMinimumQ4K(bytes: Uint8Array, scalesOffset: number, group: number): { scaleCode: number; minimumCode: number } {
  if (!Number.isInteger(group) || group < 0 || group >= 8) throw new Error(`Q4_K group inválido: ${group}.`);
  if (group < 4) {
    return { scaleCode: bytes[scalesOffset + group]! & 0x3f, minimumCode: bytes[scalesOffset + 4 + group]! & 0x3f };
  }
  return {
    scaleCode: (bytes[scalesOffset + 4 + group]! & 0x0f) | ((bytes[scalesOffset + group - 4]! >>> 6) << 4),
    minimumCode: (bytes[scalesOffset + 4 + group]! >>> 4) | ((bytes[scalesOffset + group]! >>> 6) << 4),
  };
}

function isGgufQ5_0(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q5_0" && quantization?.family === "gguf" && quantization.mode === "q5_0" &&
    quantization.bits === 5 && quantization.groupSize === GGML_Q5_0_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q5_0";
}

function isGgufQ5_1(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q5_1" && quantization?.family === "gguf" && quantization.mode === "q5_1" &&
    quantization.bits === 5 && quantization.groupSize === GGML_Q5_1_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q5_1";
}

function isGgufQ5_K(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q5_K" && quantization?.family === "gguf" && quantization.mode === "q5_k" &&
    quantization.bits === 5 && quantization.groupSize === GGML_Q5_K_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q5_K";
}

function isGgufQ6_K(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q6_K" && quantization?.family === "gguf" && quantization.mode === "q6_k" &&
    quantization.bits === 6 && quantization.groupSize === GGML_Q6_K_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q6_K";
}

function isGgufQ8_K(tensor: TensorInfo): boolean {
  const quantization = tensor.quantization;
  return tensor.storageDtype === "GGML_Q8_K" && quantization?.family === "gguf" && quantization.mode === "q8_k" &&
    quantization.bits === 8 && quantization.groupSize === GGML_Q8_K_BLOCK_SIZE && quantization.tensorType === "GGML_TYPE_Q8_K";
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

const BF16_SCRATCH = new DataView(new ArrayBuffer(4));

/** GGML BF16 stores the most-significant 16 IEEE-754 binary32 bits little-endian. */
function decodeBF16(bits: number): number {
  BF16_SCRATCH.setUint32(0, bits << 16, true);
  return BF16_SCRATCH.getFloat32(0, true);
}
