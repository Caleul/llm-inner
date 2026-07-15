import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { GgufCatalogReader } from "../src/gguf.js";
import { buildModelIR } from "../src/architecture.js";
import { materializeReferenceF32Constants } from "../src/materialize.js";

const encoder = new TextEncoder();

function u32(value: number): Buffer { const bytes = Buffer.alloc(4); bytes.writeUInt32LE(value); return bytes; }
function u64(value: number): Buffer { const bytes = Buffer.alloc(8); bytes.writeBigUInt64LE(BigInt(value)); return bytes; }
function text(value: string): Buffer { const bytes = Buffer.from(encoder.encode(value)); return Buffer.concat([u64(bytes.length), bytes]); }
function metadata(key: string, type: number, value: Buffer): Buffer { return Buffer.concat([text(key), u32(type), value]); }
function metadataString(key: string, value: string): Buffer { return metadata(key, 8, text(value)); }
function metadataU32(key: string, value: number): Buffer { return metadata(key, 4, u32(value)); }
function metadataStringArray(key: string, values: string[]): Buffer {
  return metadata(key, 9, Buffer.concat([u32(8), u64(values.length), ...values.map(text)]));
}
function tensor(name: string, dimensions: number[], ggmlType: number, offset: number): Buffer {
  return Buffer.concat([text(name), u32(dimensions.length), ...dimensions.map(u64), u32(ggmlType), u64(offset)]);
}
function q8_0(values: number[], scaleBits = 0x3800): Buffer {
  assert.equal(values.length % 32, 0, "Q8_0 fixture must contain complete 32-value blocks");
  const output = Buffer.alloc((values.length / 32) * 34);
  for (let block = 0; block < values.length / 32; block += 1) {
    const offset = block * 34;
    output.writeUInt16LE(scaleBits, offset);
    for (let index = 0; index < 32; index += 1) output.writeInt8(values[block * 32 + index]!, offset + 2 + index);
  }
  return output;
}
function q8_1(values: number[], scale = 0.5, auxiliarySum = 123.25): Buffer {
  assert.equal(values.length % 32, 0, "Q8_1 fixture must contain complete 32-value blocks");
  const output = Buffer.alloc((values.length / 32) * 40);
  for (let block = 0; block < values.length / 32; block += 1) {
    const offset = block * 40;
    output.writeFloatLE(scale, offset);
    output.writeFloatLE(auxiliarySum, offset + 4);
    for (let index = 0; index < 32; index += 1) output.writeInt8(values[block * 32 + index]!, offset + 8 + index);
  }
  return output;
}
function q8_k(values: number[], scale = 0.5): Buffer {
  assert.equal(values.length % 256, 0, "Q8_K fixture must contain complete 256-value blocks");
  const output = Buffer.alloc((values.length / 256) * 260);
  for (let block = 0; block < values.length / 256; block += 1) {
    const offset = block * 260;
    output.writeFloatLE(scale, offset);
    for (let index = 0; index < 256; index += 1) output.writeInt8(values[block * 256 + index]!, offset + 4 + index);
  }
  return output;
}
function q2_k(values: number[], scales: number[], minimums: number[], scaleBits = 0x3800, minimumBits = 0x3800): Buffer {
  assert.equal(values.length, 256, "Q2_K fixture must contain exactly one 256-value block");
  assert.equal(scales.length, 16, "Q2_K fixture must contain one scale per 16 values");
  assert.equal(minimums.length, 16, "Q2_K fixture must contain one minimum per 16 values");
  const output = Buffer.alloc(84);
  for (let group = 0; group < 16; group += 1) {
    const scale = scales[group]!;
    const minimum = minimums[group]!;
    assert.ok(scale >= 0 && scale <= 15 && minimum >= 0 && minimum <= 15, "Q2_K packed fields must be unsigned nibbles");
    output[group] = scale | (minimum << 4);
    for (let index = 0; index < 16; index += 1) {
      const code = values[group * 16 + index]!;
      assert.ok(code >= 0 && code <= 3, "Q2_K fixture codes must be unsigned two-bit values");
      output[16 + Math.floor(group / 4) * 16 + index]! |= code << ((group % 4) * 2);
    }
  }
  output.writeUInt16LE(scaleBits, 80);
  output.writeUInt16LE(minimumBits, 82);
  return output;
}
function q3_k(values: number[], scales: number[], scaleBits = 0x3800): Buffer {
  assert.equal(values.length, 256, "Q3_K fixture must contain exactly one 256-value block");
  assert.equal(scales.length, 16, "Q3_K fixture must contain one signed scale per 16 values");
  const output = Buffer.alloc(110);
  output.writeUInt16LE(scaleBits, 0);
  for (let group = 0; group < 16; group += 1) {
    const scale = scales[group]!;
    assert.ok(scale >= -32 && scale <= 31, "Q3_K scales must be signed six-bit values");
    const encoded = scale + 32;
    output[98 + (group % 8)]! |= (encoded & 0x0f) << (group < 8 ? 0 : 4);
    output[106 + (group % 4)]! |= ((encoded >>> 4) & 0x03) << (2 * Math.floor(group / 4));
  }
  for (let index = 0; index < 256; index += 1) {
    const value = values[index]!;
    assert.ok(value >= -4 && value <= 3, "Q3_K fixture codes must be signed 3-bit values");
    const encoded = value + 4;
    if (encoded > 3) output[2 + (index % 32)]! |= 1 << Math.floor(index / 32);
    const plane = Math.floor((index % 128) / 32);
    output[34 + Math.floor(index / 128) * 32 + (index % 32)]! |= (encoded & 0x03) << (plane * 2);
  }
  return output;
}
function q4_0(values: number[], scaleBits = 0x3800): Buffer {
  assert.equal(values.length % 32, 0, "Q4_0 fixture must contain complete 32-value blocks");
  const output = Buffer.alloc((values.length / 32) * 18);
  for (let block = 0; block < values.length / 32; block += 1) {
    const offset = block * 18;
    output.writeUInt16LE(scaleBits, offset);
    for (let index = 0; index < 16; index += 1) {
      const low = values[block * 32 + index]!;
      const high = values[block * 32 + 16 + index]!;
      assert.ok(low >= -8 && low <= 7 && high >= -8 && high <= 7, "Q4_0 fixture codes must be signed 4-bit values");
      output[offset + 2 + index] = (low + 8) | ((high + 8) << 4);
    }
  }
  return output;
}
function q4_1(values: number[], scaleBits = 0x3800, minimumBits = 0xbc00): Buffer {
  assert.equal(values.length % 32, 0, "Q4_1 fixture must contain complete 32-value blocks");
  const output = Buffer.alloc((values.length / 32) * 20);
  for (let block = 0; block < values.length / 32; block += 1) {
    const offset = block * 20;
    output.writeUInt16LE(scaleBits, offset);
    output.writeUInt16LE(minimumBits, offset + 2);
    for (let index = 0; index < 16; index += 1) {
      const low = values[block * 32 + index]!;
      const high = values[block * 32 + 16 + index]!;
      assert.ok(low >= 0 && low <= 15 && high >= 0 && high <= 15, "Q4_1 fixture codes must be unsigned 4-bit values");
      output[offset + 4 + index] = low | (high << 4);
    }
  }
  return output;
}
function q4_k(values: number[], scales: number[], minimums: number[], scaleBits = 0x3800, minimumBits = 0x3800): Buffer {
  assert.equal(values.length, 256, "Q4_K fixture must contain exactly one 256-value block");
  assert.equal(scales.length, 8, "Q4_K fixture must contain eight scale fields");
  assert.equal(minimums.length, 8, "Q4_K fixture must contain eight minimum fields");
  const output = Buffer.alloc(144);
  output.writeUInt16LE(scaleBits, 0);
  output.writeUInt16LE(minimumBits, 2);
  for (let group = 0; group < 4; group += 1) {
    const scale = scales[group]!;
    const minimum = minimums[group]!;
    assert.ok(scale >= 0 && scale <= 63 && minimum >= 0 && minimum <= 63, "Q4_K packed fields must be unsigned six-bit values");
    output[4 + group] = scale;
    output[8 + group] = minimum;
  }
  for (let group = 4; group < 8; group += 1) {
    const scale = scales[group]!;
    const minimum = minimums[group]!;
    assert.ok(scale >= 0 && scale <= 63 && minimum >= 0 && minimum <= 63, "Q4_K packed fields must be unsigned six-bit values");
    output[4 + group - 4]! |= (scale >>> 4) << 6;
    output[8 + group - 4]! |= (minimum >>> 4) << 6;
    output[8 + group] = (scale & 0x0f) | ((minimum & 0x0f) << 4);
  }
  for (let group = 0; group < 8; group += 1) {
    const codeOffset = 16 + Math.floor(group / 2) * 32;
    const highNibble = group % 2 === 1;
    for (let index = 0; index < 32; index += 1) {
      const code = values[group * 32 + index]!;
      assert.ok(code >= 0 && code <= 15, "Q4_K fixture codes must be unsigned 4-bit values");
      output[codeOffset + index]! |= code << (highNibble ? 4 : 0);
    }
  }
  return output;
}
function q5_k(values: number[], scales: number[], minimums: number[], scaleBits = 0x3800, minimumBits = 0x3800): Buffer {
  assert.equal(values.length, 256, "Q5_K fixture must contain exactly one 256-value block");
  assert.equal(scales.length, 8, "Q5_K fixture must contain eight scale fields");
  assert.equal(minimums.length, 8, "Q5_K fixture must contain eight minimum fields");
  const output = Buffer.alloc(176);
  output.writeUInt16LE(scaleBits, 0);
  output.writeUInt16LE(minimumBits, 2);
  for (let group = 0; group < 4; group += 1) {
    const scale = scales[group]!;
    const minimum = minimums[group]!;
    assert.ok(scale >= 0 && scale <= 63 && minimum >= 0 && minimum <= 63, "Q5_K packed fields must be unsigned six-bit values");
    output[4 + group] = scale;
    output[8 + group] = minimum;
  }
  for (let group = 4; group < 8; group += 1) {
    const scale = scales[group]!;
    const minimum = minimums[group]!;
    assert.ok(scale >= 0 && scale <= 63 && minimum >= 0 && minimum <= 63, "Q5_K packed fields must be unsigned six-bit values");
    output[4 + group - 4]! |= (scale >>> 4) << 6;
    output[8 + group - 4]! |= (minimum >>> 4) << 6;
    output[8 + group] = (scale & 0x0f) | ((minimum & 0x0f) << 4);
  }
  for (let group = 0; group < 8; group += 1) {
    const codeOffset = 48 + Math.floor(group / 2) * 32;
    const highNibble = group % 2 === 1;
    for (let index = 0; index < 32; index += 1) {
      const code = values[group * 32 + index]!;
      assert.ok(code >= 0 && code <= 31, "Q5_K fixture codes must be unsigned 5-bit values");
      output[codeOffset + index]! |= (code & 0x0f) << (highNibble ? 4 : 0);
      const logicalIndex = group * 32 + index;
      output[16 + Math.floor(logicalIndex / 8)]! |= ((code >>> 4) & 1) << (logicalIndex % 8);
    }
  }
  return output;
}
function q5_0(values: number[], scaleBits = 0x3800): Buffer {
  assert.equal(values.length % 32, 0, "Q5_0 fixture must contain complete 32-value blocks");
  const output = Buffer.alloc((values.length / 32) * 22);
  for (let block = 0; block < values.length / 32; block += 1) {
    const offset = block * 22;
    output.writeUInt16LE(scaleBits, offset);
    for (let index = 0; index < 16; index += 1) {
      const low = values[block * 32 + index]!;
      const high = values[block * 32 + 16 + index]!;
      assert.ok(low >= -16 && low <= 15 && high >= -16 && high <= 15, "Q5_0 fixture codes must be signed 5-bit values");
      const lowEncoded = low + 16;
      const highEncoded = high + 16;
      output[offset + 6 + index] = (lowEncoded & 0x0f) | ((highEncoded & 0x0f) << 4);
      output[offset + 2 + Math.floor(index / 8)]! |= ((lowEncoded >>> 4) & 1) << (index % 8);
      const highIndex = 16 + index;
      output[offset + 2 + Math.floor(highIndex / 8)]! |= ((highEncoded >>> 4) & 1) << (highIndex % 8);
    }
  }
  return output;
}
function q5_1(values: number[], scaleBits = 0x3800, minimumBits = 0xbc00): Buffer {
  assert.equal(values.length % 32, 0, "Q5_1 fixture must contain complete 32-value blocks");
  const output = Buffer.alloc((values.length / 32) * 24);
  for (let block = 0; block < values.length / 32; block += 1) {
    const offset = block * 24;
    output.writeUInt16LE(scaleBits, offset);
    output.writeUInt16LE(minimumBits, offset + 2);
    for (let index = 0; index < 16; index += 1) {
      const low = values[block * 32 + index]!;
      const high = values[block * 32 + 16 + index]!;
      assert.ok(low >= 0 && low <= 31 && high >= 0 && high <= 31, "Q5_1 fixture codes must be unsigned 5-bit values");
      output[offset + 8 + index] = (low & 0x0f) | ((high & 0x0f) << 4);
      output[offset + 4 + Math.floor(index / 8)]! |= ((low >>> 4) & 1) << (index % 8);
      const highIndex = 16 + index;
      output[offset + 4 + Math.floor(highIndex / 8)]! |= ((high >>> 4) & 1) << (highIndex % 8);
    }
  }
  return output;
}
function q6_k(values: number[], scales: number[], scaleBits = 0x3800): Buffer {
  assert.equal(values.length, 256, "Q6_K fixture must contain exactly one 256-value block");
  assert.equal(scales.length, 16, "Q6_K fixture must contain one signed scale per 16 values");
  const output = Buffer.alloc(210);
  for (let plane = 0; plane < 8; plane += 1) {
    const lowByteBase = Math.floor(plane / 4) * 64 + (plane % 2) * 32;
    const lowShift = plane % 4 >= 2 ? 4 : 0;
    const highByteBase = 128 + Math.floor(plane / 4) * 32;
    const highShift = (plane % 4) * 2;
    for (let index = 0; index < 32; index += 1) {
      const value = values[plane * 32 + index]!;
      assert.ok(value >= -32 && value <= 31, "Q6_K fixture codes must be signed 6-bit values");
      const encoded = value + 32;
      output[lowByteBase + index]! |= ((encoded & 0x0f) << lowShift);
      output[highByteBase + index]! |= (((encoded >>> 4) & 0x03) << highShift);
    }
  }
  for (let index = 0; index < 16; index += 1) output.writeInt8(scales[index]!, 192 + index);
  output.writeUInt16LE(scaleBits, 208);
  return output;
}
function bf16(bits: number[]): Buffer {
  const output = Buffer.alloc(bits.length * 2);
  for (let index = 0; index < bits.length; index += 1) output.writeUInt16LE(bits[index]!, index * 2);
  return output;
}
function fixture(metadataEntries: Buffer[], tensors: Buffer[], payload: Buffer): Buffer {
  const prefix = Buffer.concat([Buffer.from("GGUF"), u32(3), u64(tensors.length), u64(metadataEntries.length), ...metadataEntries, ...tensors]);
  const padding = Buffer.alloc((32 - (prefix.length % 32)) % 32);
  return Buffer.concat([prefix, padding, payload]);
}

async function withFixture(bytes: Buffer, run: (file: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(path.join(os.tmpdir(), "llm-inner-gguf-"));
  const file = path.join(directory, "fixture.gguf");
  try { await writeFile(file, bytes); await run(file); } finally { await rm(directory, { recursive: true, force: true }); }
}

test("native GGUF reader catalogs v3 typed metadata, aligned F32 payloads and declared dimensions", async () => {
  const bytes = fixture(
    [metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.block_count", 1), metadataStringArray("test.labels", ["alpha", "beta"])],
    [tensor("token_embd.weight", [2, 3], 0, 0)],
    Buffer.alloc(24),
  );
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      assert.equal(catalog.format, "gguf");
      assert.deepEqual(catalog.rawMetadata["test.labels"], ["alpha", "beta"]);
      assert.deepEqual(catalog.config, { model_type: "llama", block_count: 1 });
      const weight = catalog.tensors.get("token_embd.weight");
      assert.deepEqual(weight?.storageShape, [2, 3]);
      assert.deepEqual(weight?.logicalShape, [2, 3]);
      assert.equal(weight?.storageDtype, "F32");
      assert.equal(weight?.byteLength, 24);
      assert.equal(weight!.byteOffset! % 32, 0);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects still-unimplemented packed GGML types instead of guessing a dequantizer", async () => {
  const bytes = fixture(
    [metadataString("general.architecture", "llama")],
    [tensor("blk.0.attn_q.weight", [32], 16, 0)],
    Buffer.alloc(32),
  );
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /GGML_TYPE_IQ2_XXS.*sem decodificador/); } finally { await reader.close(); }
  });
});

test("native GGUF reader widens verified dense F16 storage to F32 without reclassifying it as quantized", async () => {
  const payload = Buffer.alloc(6);
  payload.writeUInt16LE(0x3c00, 0); // 1
  payload.writeUInt16LE(0xc000, 2); // -2
  payload.writeUInt16LE(0x7c00, 4); // +Infinity
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("dense", [3], 1, 0)], payload);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const loaded = await reader.readDenseAsF32(catalog.tensors.get("dense")!);
      assert.deepEqual([...loaded.values], [1, -2, Infinity]);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader widens verified dense BF16 storage with IEEE-754 edge cases", async () => {
  const payload = bf16([0x0000, 0x3f80, 0xc000, 0x0001, 0x7f80, 0x7fc1]);
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("dense", [6], 25, 0)], payload);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("dense")!;
      assert.equal(source.storageDtype, "BF16");
      assert.equal(source.quantization, undefined);
      assert.equal(source.byteLength, 12);
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values.slice(0, 5)], [0, 1, -2, 2 ** -133, Infinity]);
      assert.ok(Number.isNaN(loaded.values[5]!));
      assert.equal(loaded.sourceQuantization, undefined);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects BF16 tensors whose declared dense interval exceeds the payload", async () => {
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("truncated", [3], 25, 0)], Buffer.alloc(4));
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /truncated: intervalo GGUF ultrapassa o payload declarado/); } finally { await reader.close(); }
  });
});

test("native GGUF reader decodes the documented Q8_0 half-scale plus signed-byte block layout", async () => {
  const quantized = q8_0(Array.from({ length: 32 }, (_, index) => index - 16));
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("quantized", [32], 8, 0)], quantized);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("quantized")!;
      assert.equal(source.storageDtype, "GGML_Q8_0");
      assert.deepEqual(source.quantization, { family: "gguf", mode: "q8_0", bits: 8, groupSize: 32, tensorType: "GGML_TYPE_Q8_0" });
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values.slice(0, 4)], [-8, -7.5, -7, -6.5]);
      assert.deepEqual([...loaded.values.slice(-4)], [6, 6.5, 7, 7.5]);
      assert.deepEqual(loaded.sourceQuantization, source.quantization);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader decodes Q8_1 binary32 scale and auxiliary sum without using Q8_0 offsets", async () => {
  const quantized = q8_1(Array.from({ length: 32 }, (_, index) => index - 16));
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("quantized", [32], 9, 0)], quantized);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("quantized")!;
      assert.equal(source.storageDtype, "GGML_Q8_1");
      assert.deepEqual(source.quantization, { family: "gguf", mode: "q8_1", bits: 8, groupSize: 32, tensorType: "GGML_TYPE_Q8_1" });
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values.slice(0, 4)], [-8, -7.5, -7, -6.5]);
      assert.deepEqual([...loaded.values.slice(-4)], [6, 6.5, 7, 7.5]);
      assert.deepEqual(loaded.sourceQuantization, source.quantization);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader decodes Q8_K binary32 scale plus 256 signed-byte codes", async () => {
  const codes = Array.from({ length: 256 }, (_, index) => (index % 255) - 127);
  const quantized = q8_k(codes, 0.25);
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("quantized", [256], 15, 0)], quantized);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("quantized")!;
      assert.equal(source.storageDtype, "GGML_Q8_K");
      assert.deepEqual(source.quantization, { family: "gguf", mode: "q8_k", bits: 8, groupSize: 256, tensorType: "GGML_TYPE_Q8_K" });
      assert.equal(source.byteLength, 260);
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values], codes.map((code) => 0.25 * code));
      assert.deepEqual(loaded.sourceQuantization, source.quantization);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader decodes Q2_K packed scale/minimum nibbles and four two-bit code planes", async () => {
  const codes = Array.from({ length: 256 }, (_, index) => (index * 7) % 4);
  const scales = Array.from({ length: 16 }, (_, index) => index % 16);
  const minimums = Array.from({ length: 16 }, (_, index) => 15 - index);
  const quantized = q2_k(codes, scales, minimums);
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("quantized", [256], 10, 0)], quantized);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("quantized")!;
      assert.equal(source.storageDtype, "GGML_Q2_K");
      assert.deepEqual(source.quantization, { family: "gguf", mode: "q2_k", bits: 2, groupSize: 256, tensorType: "GGML_TYPE_Q2_K" });
      assert.equal(source.byteLength, 84);
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values], codes.map((code, index) => 0.5 * scales[Math.floor(index / 16)]! * code - 0.5 * minimums[Math.floor(index / 16)]!));
      assert.deepEqual(loaded.sourceQuantization, source.quantization);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader decodes Q3_K packed signed scales, high-bit mask, and four 2-bit planes", async () => {
  const codes = Array.from({ length: 256 }, (_, index) => ((index * 5) % 8) - 4);
  const scales = Array.from({ length: 16 }, (_, index) => index * 4 - 30);
  const quantized = q3_k(codes, scales);
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("quantized", [256], 11, 0)], quantized);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("quantized")!;
      assert.equal(source.storageDtype, "GGML_Q3_K");
      assert.deepEqual(source.quantization, { family: "gguf", mode: "q3_k", bits: 3, groupSize: 256, tensorType: "GGML_TYPE_Q3_K" });
      assert.equal(source.byteLength, 110);
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values], codes.map((code, index) => 0.5 * scales[Math.floor(index / 16)]! * code));
      assert.deepEqual(loaded.sourceQuantization, source.quantization);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader decodes Q4_0 low then high nibbles with the centered -8 formula", async () => {
  const quantized = q4_0([...Array.from({ length: 16 }, (_, index) => index - 8), ...Array.from({ length: 16 }, (_, index) => 7 - index)]);
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("quantized", [32], 2, 0)], quantized);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("quantized")!;
      assert.equal(source.storageDtype, "GGML_Q4_0");
      assert.deepEqual(source.quantization, { family: "gguf", mode: "q4_0", bits: 4, groupSize: 32, tensorType: "GGML_TYPE_Q4_0" });
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values.slice(0, 4)], [-4, -3.5, -3, -2.5]);
      assert.deepEqual([...loaded.values.slice(16, 20)], [3.5, 3, 2.5, 2]);
      assert.deepEqual(loaded.sourceQuantization, source.quantization);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader decodes Q4_1 scale and minimum with low then high unsigned nibbles", async () => {
  const quantized = q4_1([...Array.from({ length: 16 }, (_, index) => index), ...Array.from({ length: 16 }, (_, index) => 15 - index)]);
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("quantized", [32], 3, 0)], quantized);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("quantized")!;
      assert.equal(source.storageDtype, "GGML_Q4_1");
      assert.deepEqual(source.quantization, { family: "gguf", mode: "q4_1", bits: 4, groupSize: 32, tensorType: "GGML_TYPE_Q4_1" });
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values.slice(0, 4)], [-1, -0.5, 0, 0.5]);
      assert.deepEqual([...loaded.values.slice(16, 20)], [6.5, 6, 5.5, 5]);
      assert.deepEqual(loaded.sourceQuantization, source.quantization);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader decodes Q5_0 high-bit plane plus low then high nibbles with the centered -16 formula", async () => {
  const quantized = q5_0([...Array.from({ length: 16 }, (_, index) => index - 16), ...Array.from({ length: 16 }, (_, index) => 15 - index)]);
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("quantized", [32], 6, 0)], quantized);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("quantized")!;
      assert.equal(source.storageDtype, "GGML_Q5_0");
      assert.deepEqual(source.quantization, { family: "gguf", mode: "q5_0", bits: 5, groupSize: 32, tensorType: "GGML_TYPE_Q5_0" });
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values.slice(0, 4)], [-8, -7.5, -7, -6.5]);
      assert.deepEqual([...loaded.values.slice(16, 20)], [7.5, 7, 6.5, 6]);
      assert.deepEqual(loaded.sourceQuantization, source.quantization);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader decodes Q5_1 scale, minimum, high-bit plane and low then high affine codes", async () => {
  const quantized = q5_1([...Array.from({ length: 16 }, (_, index) => index), ...Array.from({ length: 16 }, (_, index) => 31 - index)]);
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("quantized", [32], 7, 0)], quantized);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("quantized")!;
      assert.equal(source.storageDtype, "GGML_Q5_1");
      assert.deepEqual(source.quantization, { family: "gguf", mode: "q5_1", bits: 5, groupSize: 32, tensorType: "GGML_TYPE_Q5_1" });
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values.slice(0, 4)], [-1, -0.5, 0, 0.5]);
      assert.deepEqual([...loaded.values.slice(16, 20)], [14.5, 14, 13.5, 13]);
      assert.deepEqual(loaded.sourceQuantization, source.quantization);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader decodes Q6_K low and high planes with signed per-16-value scales", async () => {
  const codes = Array.from({ length: 256 }, (_, index) => ((index * 13) % 64) - 32);
  const scales = Array.from({ length: 16 }, (_, index) => index - 8);
  const quantized = q6_k(codes, scales);
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("quantized", [256], 14, 0)], quantized);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("quantized")!;
      assert.equal(source.storageDtype, "GGML_Q6_K");
      assert.deepEqual(source.quantization, { family: "gguf", mode: "q6_k", bits: 6, groupSize: 256, tensorType: "GGML_TYPE_Q6_K" });
      assert.equal(source.byteLength, 210);
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values], codes.map((code, index) => 0.5 * scales[Math.floor(index / 16)]! * code));
      assert.deepEqual(loaded.sourceQuantization, source.quantization);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader decodes Q4_K packed scale/minimum hierarchy and code planes", async () => {
  const codes = Array.from({ length: 256 }, (_, index) => (index * 11) % 16);
  const scales = [1, 2, 3, 4, 17, 33, 49, 63];
  const minimums = [4, 5, 6, 7, 18, 34, 50, 62];
  const quantized = q4_k(codes, scales, minimums);
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("quantized", [256], 12, 0)], quantized);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("quantized")!;
      assert.equal(source.storageDtype, "GGML_Q4_K");
      assert.deepEqual(source.quantization, { family: "gguf", mode: "q4_k", bits: 4, groupSize: 256, tensorType: "GGML_TYPE_Q4_K" });
      assert.equal(source.byteLength, 144);
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values], codes.map((code, index) => 0.5 * scales[Math.floor(index / 32)]! * code - 0.5 * minimums[Math.floor(index / 32)]!));
      assert.deepEqual(loaded.sourceQuantization, source.quantization);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader decodes Q5_K packed scale/minimum hierarchy, high-bit plane, and code planes", async () => {
  const codes = Array.from({ length: 256 }, (_, index) => (index * 19) % 32);
  const scales = [1, 2, 3, 4, 17, 33, 49, 63];
  const minimums = [4, 5, 6, 7, 18, 34, 50, 62];
  const quantized = q5_k(codes, scales, minimums);
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("quantized", [256], 13, 0)], quantized);
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const source = catalog.tensors.get("quantized")!;
      assert.equal(source.storageDtype, "GGML_Q5_K");
      assert.deepEqual(source.quantization, { family: "gguf", mode: "q5_k", bits: 5, groupSize: 256, tensorType: "GGML_TYPE_Q5_K" });
      assert.equal(source.byteLength, 176);
      const loaded = await reader.readDenseAsF32(source);
      assert.deepEqual([...loaded.values], codes.map((code, index) => 0.5 * scales[Math.floor(index / 32)]! * code - 0.5 * minimums[Math.floor(index / 32)]!));
      assert.deepEqual(loaded.sourceQuantization, source.quantization);
    } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q8_0 shapes whose fastest GGML dimension cannot contain complete blocks", async () => {
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("bad", [31], 8, 0)], Buffer.alloc(34));
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q8_0 exige a primeira dimensão GGML positiva e múltipla de 32/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q8_1 shapes whose fastest GGML dimension cannot contain complete blocks", async () => {
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("bad", [31], 9, 0)], Buffer.alloc(40));
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q8_1 exige a primeira dimensão GGML positiva e múltipla de 32/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q8_K shapes and intervals that cannot contain complete 256-value blocks", async () => {
  const malformedShape = fixture([metadataString("general.architecture", "llama")], [tensor("bad-shape", [255], 15, 0)], Buffer.alloc(260));
  await withFixture(malformedShape, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q8_K exige a primeira dimensão GGML positiva e múltipla de 256/); } finally { await reader.close(); }
  });
  const truncated = fixture([metadataString("general.architecture", "llama")], [tensor("bad-interval", [256], 15, 0)], Buffer.alloc(259));
  await withFixture(truncated, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /bad-interval: intervalo GGUF ultrapassa o payload declarado/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q3_K shapes and intervals that cannot contain complete 256-value blocks", async () => {
  const malformedShape = fixture([metadataString("general.architecture", "llama")], [tensor("bad-shape", [255], 11, 0)], Buffer.alloc(110));
  await withFixture(malformedShape, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q3_K exige a primeira dimensão GGML positiva e múltipla de 256/); } finally { await reader.close(); }
  });
  const truncated = fixture([metadataString("general.architecture", "llama")], [tensor("bad-interval", [256], 11, 0)], Buffer.alloc(109));
  await withFixture(truncated, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /bad-interval: intervalo GGUF ultrapassa o payload declarado/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q2_K shapes and intervals that cannot contain complete 256-value blocks", async () => {
  const malformedShape = fixture([metadataString("general.architecture", "llama")], [tensor("bad-shape", [255], 10, 0)], Buffer.alloc(84));
  await withFixture(malformedShape, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q2_K exige a primeira dimensão GGML positiva e múltipla de 256/); } finally { await reader.close(); }
  });
  const truncated = fixture([metadataString("general.architecture", "llama")], [tensor("bad-interval", [256], 10, 0)], Buffer.alloc(83));
  await withFixture(truncated, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /bad-interval: intervalo GGUF ultrapassa o payload declarado/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q4_0 shapes whose fastest GGML dimension cannot contain complete blocks", async () => {
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("bad", [31], 2, 0)], Buffer.alloc(18));
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q4_0 exige a primeira dimensão GGML positiva e múltipla de 32/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q4_1 shapes whose fastest GGML dimension cannot contain complete blocks", async () => {
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("bad", [31], 3, 0)], Buffer.alloc(20));
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q4_1 exige a primeira dimensão GGML positiva e múltipla de 32/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q5_0 shapes whose fastest GGML dimension cannot contain complete blocks", async () => {
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("bad", [31], 6, 0)], Buffer.alloc(22));
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q5_0 exige a primeira dimensão GGML positiva e múltipla de 32/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q5_1 shapes whose fastest GGML dimension cannot contain complete blocks", async () => {
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("bad", [31], 7, 0)], Buffer.alloc(24));
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q5_1 exige a primeira dimensão GGML positiva e múltipla de 32/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q6_K shapes and intervals that cannot contain complete 256-value blocks", async () => {
  const malformedShape = fixture([metadataString("general.architecture", "llama")], [tensor("bad-shape", [255], 14, 0)], Buffer.alloc(210));
  await withFixture(malformedShape, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q6_K exige a primeira dimensão GGML positiva e múltipla de 256/); } finally { await reader.close(); }
  });
  const truncated = fixture([metadataString("general.architecture", "llama")], [tensor("bad-interval", [256], 14, 0)], Buffer.alloc(209));
  await withFixture(truncated, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /bad-interval: intervalo GGUF ultrapassa o payload declarado/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q4_K shapes and intervals that cannot contain complete 256-value blocks", async () => {
  const malformedShape = fixture([metadataString("general.architecture", "llama")], [tensor("bad-shape", [255], 12, 0)], Buffer.alloc(144));
  await withFixture(malformedShape, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q4_K exige a primeira dimensão GGML positiva e múltipla de 256/); } finally { await reader.close(); }
  });
  const truncated = fixture([metadataString("general.architecture", "llama")], [tensor("bad-interval", [256], 12, 0)], Buffer.alloc(143));
  await withFixture(truncated, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /bad-interval: intervalo GGUF ultrapassa o payload declarado/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q5_K shapes and intervals that cannot contain complete 256-value blocks", async () => {
  const malformedShape = fixture([metadataString("general.architecture", "llama")], [tensor("bad-shape", [255], 13, 0)], Buffer.alloc(176));
  await withFixture(malformedShape, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q5_K exige a primeira dimensão GGML positiva e múltipla de 256/); } finally { await reader.close(); }
  });
  const truncated = fixture([metadataString("general.architecture", "llama")], [tensor("bad-interval", [256], 13, 0)], Buffer.alloc(175));
  await withFixture(truncated, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /bad-interval: intervalo GGUF ultrapassa o payload declarado/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects unaligned tensor offsets and unsupported container versions", async () => {
  const unaligned = fixture(
    [metadataString("general.architecture", "llama")],
    [tensor("token_embd.weight", [2], 0, 4)],
    Buffer.alloc(16),
  );
  await withFixture(unaligned, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /offset GGUF 4 não é alinhado/); } finally { await reader.close(); }
  });

  const wrongVersion = Buffer.concat([Buffer.from("GGUF"), u32(4), u64(0), u64(0)]);
  await withFixture(wrongVersion, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Versão GGUF 4 não suportada/); } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter reverses only documented GGML matrix dimensions and materializes dense F32 ranges", async () => {
  const shapes: Array<[string, number[]]> = [
    ["token_embd.weight", [4, 3]], ["blk.0.attn_norm.weight", [4]],
    ["blk.0.attn_q.weight", [4, 4]], ["blk.0.attn_k.weight", [4, 2]], ["blk.0.attn_v.weight", [4, 2]],
    ["blk.0.attn_output.weight", [4, 4]], ["blk.0.ffn_norm.weight", [4]],
    ["blk.0.ffn_gate.weight", [4, 6]], ["blk.0.ffn_up.weight", [4, 6]], ["blk.0.ffn_down.weight", [6, 4]],
    ["output_norm.weight", [4]], ["output.weight", [4, 3]],
  ];
  const payloadParts: Buffer[] = [];
  let offset = 0;
  const directory: Buffer[] = [];
  for (const [name, ggmlShape] of shapes) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const values = Buffer.alloc(ggmlShape.reduce((a, b) => a * b, 1) * 4);
    for (let index = 0; index < values.length / 4; index += 1) values.writeFloatLE(index + 0.25, index * 4);
    directory.push(tensor(name, ggmlShape, 0, offset));
    payloadParts.push(values); offset += values.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32),
    metadataU32("llama.embedding_length", 4), metadataU32("llama.block_count", 1),
    metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 2), metadataU32("llama.feed_forward_length", 6),
    metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const gate = ir.layers[0]!.operations.find((operation) => operation.id === "layer_0_gate_proj");
      assert.equal(gate?.op, "linear");
      if (gate?.op === "linear") assert.deepEqual(gate.weight.shape, [6, 4]);
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      assert.deepEqual(constants.get("token_embd.weight")?.shape, [3, 4]);
      assert.deepEqual([...constants.get("token_embd.weight")!.values.slice(0, 4)], [0.25, 1.25, 2.25, 3.25]);
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes dense BF16 constants", async () => {
  const shapes: Array<[string, number[]]> = [
    ["token_embd.weight", [4, 3]], ["blk.0.attn_norm.weight", [4]],
    ["blk.0.attn_q.weight", [4, 4]], ["blk.0.attn_k.weight", [4, 2]], ["blk.0.attn_v.weight", [4, 2]],
    ["blk.0.attn_output.weight", [4, 4]], ["blk.0.ffn_norm.weight", [4]],
    ["blk.0.ffn_gate.weight", [4, 6]], ["blk.0.ffn_up.weight", [4, 6]], ["blk.0.ffn_down.weight", [6, 4]],
    ["output_norm.weight", [4]], ["output.weight", [4, 3]],
  ];
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of shapes) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const payload = bf16(Array.from({ length: ggmlShape.reduce((left, right) => left * right, 1) }, () => 0x3fc0)); // 1.5
    directory.push(tensor(name, ggmlShape, 25, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32),
    metadataU32("llama.embedding_length", 4), metadataU32("llama.block_count", 1),
    metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 2), metadataU32("llama.feed_forward_length", 6),
    metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const embedding = constants.get("token_embd.weight")!;
      assert.deepEqual(embedding.shape, [3, 4]);
      assert.deepEqual([...embedding.values.slice(0, 4)], [1.5, 1.5, 1.5, 1.5]);
      assert.equal(embedding.sourceQuantization, undefined);
    } finally { await reader.close(); }
  });
});

test("Llama GGUF adapter rejects a documented matrix with non-matrix GGML dimensions", async () => {
  const bytes = fixture(
    [metadataString("general.architecture", "llama"), metadataU32("llama.embedding_length", 2), metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 1), metadataU32("llama.attention.key_length", 2), metadataU32("llama.feed_forward_length", 4)],
    [tensor("token_embd.weight", [2, 2, 1], 0, 0)], Buffer.alloc(16),
  );
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      await assert.rejects(() => buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false }), /tensor matricial Llama GGUF deve declarar exatamente duas dimensões/);
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes verified Q8_0 matrices with provenance", async () => {
  const matrixShapes: Array<[string, number[]]> = [
    ["token_embd.weight", [32, 32]], ["blk.0.attn_q.weight", [32, 32]], ["blk.0.attn_k.weight", [32, 16]],
    ["blk.0.attn_v.weight", [32, 16]], ["blk.0.attn_output.weight", [32, 32]], ["blk.0.ffn_gate.weight", [32, 32]],
    ["blk.0.ffn_up.weight", [32, 32]], ["blk.0.ffn_down.weight", [32, 32]], ["output.weight", [32, 32]],
  ];
  const vectorShapes: Array<[string, number[]]> = [["blk.0.attn_norm.weight", [32]], ["blk.0.ffn_norm.weight", [32]], ["output_norm.weight", [32]]];
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of [...matrixShapes, ...vectorShapes]) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const payload = matrixShapes.some(([matrixName]) => matrixName === name)
      ? q8_0(Array.from({ length: ggmlShape[0]! * ggmlShape[1]! }, (_, index) => (index % 32) - 16))
      : Buffer.alloc(ggmlShape[0]! * 4, 0);
    directory.push(tensor(name, ggmlShape, matrixShapes.some(([matrixName]) => matrixName === name) ? 8 : 0, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 32),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 16), metadataU32("llama.feed_forward_length", 32), metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const embedding = ir.prelude[0]!;
      assert.equal(embedding.op, "embedding");
      if (embedding.op === "embedding") assert.deepEqual(embedding.weight.shape, [32, 32]);
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const weight = constants.get("token_embd.weight")!;
      assert.deepEqual([...weight.values.slice(0, 4)], [-8, -7.5, -7, -6.5]);
      assert.deepEqual(weight.sourceQuantization, { family: "gguf", mode: "q8_0", bits: 8, groupSize: 32, tensorType: "GGML_TYPE_Q8_0" });
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes verified Q8_1 matrices with provenance", async () => {
  const matrixShapes: Array<[string, number[]]> = [
    ["token_embd.weight", [32, 32]], ["blk.0.attn_q.weight", [32, 32]], ["blk.0.attn_k.weight", [32, 16]],
    ["blk.0.attn_v.weight", [32, 16]], ["blk.0.attn_output.weight", [32, 32]], ["blk.0.ffn_gate.weight", [32, 32]],
    ["blk.0.ffn_up.weight", [32, 32]], ["blk.0.ffn_down.weight", [32, 32]], ["output.weight", [32, 32]],
  ];
  const vectorShapes: Array<[string, number[]]> = [["blk.0.attn_norm.weight", [32]], ["blk.0.ffn_norm.weight", [32]], ["output_norm.weight", [32]]];
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of [...matrixShapes, ...vectorShapes]) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const payload = matrixShapes.some(([matrixName]) => matrixName === name)
      ? q8_1(Array.from({ length: ggmlShape[0]! * ggmlShape[1]! }, (_, index) => (index % 32) - 16))
      : Buffer.alloc(ggmlShape[0]! * 4, 0);
    directory.push(tensor(name, ggmlShape, matrixShapes.some(([matrixName]) => matrixName === name) ? 9 : 0, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 32),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 16), metadataU32("llama.feed_forward_length", 32), metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const weight = constants.get("token_embd.weight")!;
      assert.deepEqual(weight.shape, [32, 32]);
      assert.deepEqual([...weight.values.slice(0, 4)], [-8, -7.5, -7, -6.5]);
      assert.deepEqual(weight.sourceQuantization, { family: "gguf", mode: "q8_1", bits: 8, groupSize: 32, tensorType: "GGML_TYPE_Q8_1" });
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes verified Q4_0 matrices with provenance", async () => {
  const matrixShapes: Array<[string, number[]]> = [
    ["token_embd.weight", [32, 32]], ["blk.0.attn_q.weight", [32, 32]], ["blk.0.attn_k.weight", [32, 16]],
    ["blk.0.attn_v.weight", [32, 16]], ["blk.0.attn_output.weight", [32, 32]], ["blk.0.ffn_gate.weight", [32, 32]],
    ["blk.0.ffn_up.weight", [32, 32]], ["blk.0.ffn_down.weight", [32, 32]], ["output.weight", [32, 32]],
  ];
  const vectorShapes: Array<[string, number[]]> = [["blk.0.attn_norm.weight", [32]], ["blk.0.ffn_norm.weight", [32]], ["output_norm.weight", [32]]];
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of [...matrixShapes, ...vectorShapes]) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const payload = matrixShapes.some(([matrixName]) => matrixName === name)
      ? q4_0(Array.from({ length: ggmlShape[0]! * ggmlShape[1]! }, (_, index) => (index % 16) - 8))
      : Buffer.alloc(ggmlShape[0]! * 4, 0);
    directory.push(tensor(name, ggmlShape, matrixShapes.some(([matrixName]) => matrixName === name) ? 2 : 0, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 32),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 16), metadataU32("llama.feed_forward_length", 32), metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const weight = constants.get("token_embd.weight")!;
      assert.deepEqual(weight.shape, [32, 32]);
      assert.deepEqual([...weight.values.slice(0, 4)], [-4, -3.5, -3, -2.5]);
      assert.deepEqual(weight.sourceQuantization, { family: "gguf", mode: "q4_0", bits: 4, groupSize: 32, tensorType: "GGML_TYPE_Q4_0" });
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes verified Q4_1 matrices with provenance", async () => {
  const matrixShapes: Array<[string, number[]]> = [
    ["token_embd.weight", [32, 32]], ["blk.0.attn_q.weight", [32, 32]], ["blk.0.attn_k.weight", [32, 16]],
    ["blk.0.attn_v.weight", [32, 16]], ["blk.0.attn_output.weight", [32, 32]], ["blk.0.ffn_gate.weight", [32, 32]],
    ["blk.0.ffn_up.weight", [32, 32]], ["blk.0.ffn_down.weight", [32, 32]], ["output.weight", [32, 32]],
  ];
  const vectorShapes: Array<[string, number[]]> = [["blk.0.attn_norm.weight", [32]], ["blk.0.ffn_norm.weight", [32]], ["output_norm.weight", [32]]];
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of [...matrixShapes, ...vectorShapes]) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const payload = matrixShapes.some(([matrixName]) => matrixName === name)
      ? q4_1(Array.from({ length: ggmlShape[0]! * ggmlShape[1]! }, (_, index) => index % 16))
      : Buffer.alloc(ggmlShape[0]! * 4, 0);
    directory.push(tensor(name, ggmlShape, matrixShapes.some(([matrixName]) => matrixName === name) ? 3 : 0, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 32),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 16), metadataU32("llama.feed_forward_length", 32), metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const weight = constants.get("token_embd.weight")!;
      assert.deepEqual(weight.shape, [32, 32]);
      assert.deepEqual([...weight.values.slice(0, 4)], [-1, -0.5, 0, 0.5]);
      assert.deepEqual(weight.sourceQuantization, { family: "gguf", mode: "q4_1", bits: 4, groupSize: 32, tensorType: "GGML_TYPE_Q4_1" });
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes verified Q5_0 matrices with provenance", async () => {
  const matrixShapes: Array<[string, number[]]> = [
    ["token_embd.weight", [32, 32]], ["blk.0.attn_q.weight", [32, 32]], ["blk.0.attn_k.weight", [32, 16]],
    ["blk.0.attn_v.weight", [32, 16]], ["blk.0.attn_output.weight", [32, 32]], ["blk.0.ffn_gate.weight", [32, 32]],
    ["blk.0.ffn_up.weight", [32, 32]], ["blk.0.ffn_down.weight", [32, 32]], ["output.weight", [32, 32]],
  ];
  const vectorShapes: Array<[string, number[]]> = [["blk.0.attn_norm.weight", [32]], ["blk.0.ffn_norm.weight", [32]], ["output_norm.weight", [32]]];
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of [...matrixShapes, ...vectorShapes]) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const payload = matrixShapes.some(([matrixName]) => matrixName === name)
      ? q5_0(Array.from({ length: ggmlShape[0]! * ggmlShape[1]! }, (_, index) => (index % 32) - 16))
      : Buffer.alloc(ggmlShape[0]! * 4, 0);
    directory.push(tensor(name, ggmlShape, matrixShapes.some(([matrixName]) => matrixName === name) ? 6 : 0, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 32),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 16), metadataU32("llama.feed_forward_length", 32), metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const weight = constants.get("token_embd.weight")!;
      assert.deepEqual(weight.shape, [32, 32]);
      assert.deepEqual([...weight.values.slice(0, 4)], [-8, -7.5, -7, -6.5]);
      assert.deepEqual(weight.sourceQuantization, { family: "gguf", mode: "q5_0", bits: 5, groupSize: 32, tensorType: "GGML_TYPE_Q5_0" });
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes verified Q5_1 matrices with affine provenance", async () => {
  const matrixShapes: Array<[string, number[]]> = [
    ["token_embd.weight", [32, 32]], ["blk.0.attn_q.weight", [32, 32]], ["blk.0.attn_k.weight", [32, 16]],
    ["blk.0.attn_v.weight", [32, 16]], ["blk.0.attn_output.weight", [32, 32]], ["blk.0.ffn_gate.weight", [32, 32]],
    ["blk.0.ffn_up.weight", [32, 32]], ["blk.0.ffn_down.weight", [32, 32]], ["output.weight", [32, 32]],
  ];
  const vectorShapes: Array<[string, number[]]> = [["blk.0.attn_norm.weight", [32]], ["blk.0.ffn_norm.weight", [32]], ["output_norm.weight", [32]]];
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of [...matrixShapes, ...vectorShapes]) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const payload = matrixShapes.some(([matrixName]) => matrixName === name)
      ? q5_1(Array.from({ length: ggmlShape[0]! * ggmlShape[1]! }, (_, index) => index % 32))
      : Buffer.alloc(ggmlShape[0]! * 4, 0);
    directory.push(tensor(name, ggmlShape, matrixShapes.some(([matrixName]) => matrixName === name) ? 7 : 0, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 32),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 16), metadataU32("llama.feed_forward_length", 32), metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const weight = constants.get("token_embd.weight")!;
      assert.deepEqual(weight.shape, [32, 32]);
      assert.deepEqual([...weight.values.slice(0, 4)], [-1, -0.5, 0, 0.5]);
      assert.deepEqual(weight.sourceQuantization, { family: "gguf", mode: "q5_1", bits: 5, groupSize: 32, tensorType: "GGML_TYPE_Q5_1" });
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes verified Q6_K matrices with provenance", async () => {
  const matrixShapes: Array<[string, number[]]> = [
    ["token_embd.weight", [256, 256]], ["blk.0.attn_q.weight", [256, 256]], ["blk.0.attn_k.weight", [256, 128]],
    ["blk.0.attn_v.weight", [256, 128]], ["blk.0.attn_output.weight", [256, 256]], ["blk.0.ffn_gate.weight", [256, 256]],
    ["blk.0.ffn_up.weight", [256, 256]], ["blk.0.ffn_down.weight", [256, 256]], ["output.weight", [256, 256]],
  ];
  const vectorShapes: Array<[string, number[]]> = [["blk.0.attn_norm.weight", [256]], ["blk.0.ffn_norm.weight", [256]], ["output_norm.weight", [256]]];
  const codes = Array.from({ length: 256 }, (_, index) => ((index * 13) % 64) - 32);
  const scales = Array.from({ length: 16 }, (_, index) => index - 8);
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of [...matrixShapes, ...vectorShapes]) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const isMatrix = matrixShapes.some(([matrixName]) => matrixName === name);
    const payload = isMatrix
      ? Buffer.concat(Array.from({ length: (ggmlShape[0]! * ggmlShape[1]!) / 256 }, () => q6_k(codes, scales)))
      : Buffer.alloc(ggmlShape[0]! * 4, 0);
    directory.push(tensor(name, ggmlShape, isMatrix ? 14 : 0, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 256),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 128), metadataU32("llama.feed_forward_length", 256), metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const weight = constants.get("token_embd.weight")!;
      assert.deepEqual(weight.shape, [256, 256]);
      assert.deepEqual([...weight.values.slice(0, 32)], codes.slice(0, 32).map((code, index) => 0.5 * scales[Math.floor(index / 16)]! * code));
      assert.deepEqual(weight.sourceQuantization, { family: "gguf", mode: "q6_k", bits: 6, groupSize: 256, tensorType: "GGML_TYPE_Q6_K" });
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes verified Q4_K matrices with provenance", async () => {
  const matrixShapes: Array<[string, number[]]> = [
    ["token_embd.weight", [256, 256]], ["blk.0.attn_q.weight", [256, 256]], ["blk.0.attn_k.weight", [256, 128]],
    ["blk.0.attn_v.weight", [256, 128]], ["blk.0.attn_output.weight", [256, 256]], ["blk.0.ffn_gate.weight", [256, 256]],
    ["blk.0.ffn_up.weight", [256, 256]], ["blk.0.ffn_down.weight", [256, 256]], ["output.weight", [256, 256]],
  ];
  const vectorShapes: Array<[string, number[]]> = [["blk.0.attn_norm.weight", [256]], ["blk.0.ffn_norm.weight", [256]], ["output_norm.weight", [256]]];
  const codes = Array.from({ length: 256 }, (_, index) => (index * 11) % 16);
  const scales = [1, 2, 3, 4, 17, 33, 49, 63];
  const minimums = [4, 5, 6, 7, 18, 34, 50, 62];
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of [...matrixShapes, ...vectorShapes]) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const isMatrix = matrixShapes.some(([matrixName]) => matrixName === name);
    const payload = isMatrix
      ? Buffer.concat(Array.from({ length: (ggmlShape[0]! * ggmlShape[1]!) / 256 }, () => q4_k(codes, scales, minimums)))
      : Buffer.alloc(ggmlShape[0]! * 4, 0);
    directory.push(tensor(name, ggmlShape, isMatrix ? 12 : 0, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 256),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 128), metadataU32("llama.feed_forward_length", 256), metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const weight = constants.get("token_embd.weight")!;
      assert.deepEqual(weight.shape, [256, 256]);
      assert.deepEqual([...weight.values.slice(0, 32)], codes.slice(0, 32).map((code) => 0.5 * scales[0]! * code - 0.5 * minimums[0]!));
      assert.deepEqual(weight.sourceQuantization, { family: "gguf", mode: "q4_k", bits: 4, groupSize: 256, tensorType: "GGML_TYPE_Q4_K" });
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes verified Q5_K matrices with provenance", async () => {
  const matrixShapes: Array<[string, number[]]> = [
    ["token_embd.weight", [256, 256]], ["blk.0.attn_q.weight", [256, 256]], ["blk.0.attn_k.weight", [256, 128]],
    ["blk.0.attn_v.weight", [256, 128]], ["blk.0.attn_output.weight", [256, 256]], ["blk.0.ffn_gate.weight", [256, 256]],
    ["blk.0.ffn_up.weight", [256, 256]], ["blk.0.ffn_down.weight", [256, 256]], ["output.weight", [256, 256]],
  ];
  const vectorShapes: Array<[string, number[]]> = [["blk.0.attn_norm.weight", [256]], ["blk.0.ffn_norm.weight", [256]], ["output_norm.weight", [256]]];
  const codes = Array.from({ length: 256 }, (_, index) => (index * 19) % 32);
  const scales = [1, 2, 3, 4, 17, 33, 49, 63];
  const minimums = [4, 5, 6, 7, 18, 34, 50, 62];
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of [...matrixShapes, ...vectorShapes]) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const isMatrix = matrixShapes.some(([matrixName]) => matrixName === name);
    const payload = isMatrix
      ? Buffer.concat(Array.from({ length: (ggmlShape[0]! * ggmlShape[1]!) / 256 }, () => q5_k(codes, scales, minimums)))
      : Buffer.alloc(ggmlShape[0]! * 4, 0);
    directory.push(tensor(name, ggmlShape, isMatrix ? 13 : 0, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 256),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 128), metadataU32("llama.feed_forward_length", 256), metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const weight = constants.get("token_embd.weight")!;
      assert.deepEqual(weight.shape, [256, 256]);
      assert.deepEqual([...weight.values.slice(0, 32)], codes.slice(0, 32).map((code) => 0.5 * scales[0]! * code - 0.5 * minimums[0]!));
      assert.deepEqual(weight.sourceQuantization, { family: "gguf", mode: "q5_k", bits: 5, groupSize: 256, tensorType: "GGML_TYPE_Q5_K" });
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes verified Q8_K matrices with provenance", async () => {
  const matrixShapes: Array<[string, number[]]> = [
    ["token_embd.weight", [256, 256]], ["blk.0.attn_q.weight", [256, 256]], ["blk.0.attn_k.weight", [256, 128]],
    ["blk.0.attn_v.weight", [256, 128]], ["blk.0.attn_output.weight", [256, 256]], ["blk.0.ffn_gate.weight", [256, 256]],
    ["blk.0.ffn_up.weight", [256, 256]], ["blk.0.ffn_down.weight", [256, 256]], ["output.weight", [256, 256]],
  ];
  const vectorShapes: Array<[string, number[]]> = [["blk.0.attn_norm.weight", [256]], ["blk.0.ffn_norm.weight", [256]], ["output_norm.weight", [256]]];
  const codes = Array.from({ length: 256 }, (_, index) => (index % 255) - 127);
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of [...matrixShapes, ...vectorShapes]) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const isMatrix = matrixShapes.some(([matrixName]) => matrixName === name);
    const payload = isMatrix
      ? Buffer.concat(Array.from({ length: (ggmlShape[0]! * ggmlShape[1]!) / 256 }, () => q8_k(codes, 0.25)))
      : Buffer.alloc(ggmlShape[0]! * 4, 0);
    directory.push(tensor(name, ggmlShape, isMatrix ? 15 : 0, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 256),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 128), metadataU32("llama.feed_forward_length", 256), metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const weight = constants.get("token_embd.weight")!;
      assert.deepEqual(weight.shape, [256, 256]);
      assert.deepEqual([...weight.values.slice(0, 32)], codes.slice(0, 32).map((code) => 0.25 * code));
      assert.deepEqual(weight.sourceQuantization, { family: "gguf", mode: "q8_k", bits: 8, groupSize: 256, tensorType: "GGML_TYPE_Q8_K" });
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes verified Q3_K matrices with provenance", async () => {
  const matrixShapes: Array<[string, number[]]> = [
    ["token_embd.weight", [256, 256]], ["blk.0.attn_q.weight", [256, 256]], ["blk.0.attn_k.weight", [256, 128]],
    ["blk.0.attn_v.weight", [256, 128]], ["blk.0.attn_output.weight", [256, 256]], ["blk.0.ffn_gate.weight", [256, 256]],
    ["blk.0.ffn_up.weight", [256, 256]], ["blk.0.ffn_down.weight", [256, 256]], ["output.weight", [256, 256]],
  ];
  const vectorShapes: Array<[string, number[]]> = [["blk.0.attn_norm.weight", [256]], ["blk.0.ffn_norm.weight", [256]], ["output_norm.weight", [256]]];
  const codes = Array.from({ length: 256 }, (_, index) => ((index * 5) % 8) - 4);
  const scales = Array.from({ length: 16 }, (_, index) => index * 4 - 30);
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of [...matrixShapes, ...vectorShapes]) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const isMatrix = matrixShapes.some(([matrixName]) => matrixName === name);
    const payload = isMatrix
      ? Buffer.concat(Array.from({ length: (ggmlShape[0]! * ggmlShape[1]!) / 256 }, () => q3_k(codes, scales)))
      : Buffer.alloc(ggmlShape[0]! * 4, 0);
    directory.push(tensor(name, ggmlShape, isMatrix ? 11 : 0, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 256),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 128), metadataU32("llama.feed_forward_length", 256), metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const weight = constants.get("token_embd.weight")!;
      assert.deepEqual(weight.shape, [256, 256]);
      assert.deepEqual([...weight.values.slice(0, 32)], codes.slice(0, 32).map((code, index) => 0.5 * scales[Math.floor(index / 16)]! * code));
      assert.deepEqual(weight.sourceQuantization, { family: "gguf", mode: "q3_k", bits: 3, groupSize: 256, tensorType: "GGML_TYPE_Q3_K" });
    } finally { await reader.close(); }
  });
});

test("explicit Llama GGUF adapter lowers and materializes verified Q2_K matrices with provenance", async () => {
  const matrixShapes: Array<[string, number[]]> = [
    ["token_embd.weight", [256, 256]], ["blk.0.attn_q.weight", [256, 256]], ["blk.0.attn_k.weight", [256, 128]],
    ["blk.0.attn_v.weight", [256, 128]], ["blk.0.attn_output.weight", [256, 256]], ["blk.0.ffn_gate.weight", [256, 256]],
    ["blk.0.ffn_up.weight", [256, 256]], ["blk.0.ffn_down.weight", [256, 256]], ["output.weight", [256, 256]],
  ];
  const vectorShapes: Array<[string, number[]]> = [["blk.0.attn_norm.weight", [256]], ["blk.0.ffn_norm.weight", [256]], ["output_norm.weight", [256]]];
  const codes = Array.from({ length: 256 }, (_, index) => (index * 7) % 4);
  const scales = Array.from({ length: 16 }, (_, index) => index % 16);
  const minimums = Array.from({ length: 16 }, (_, index) => 15 - index);
  const payloadParts: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, ggmlShape] of [...matrixShapes, ...vectorShapes]) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloadParts.push(Buffer.alloc(padding)); offset += padding; }
    const isMatrix = matrixShapes.some(([matrixName]) => matrixName === name);
    const payload = isMatrix
      ? Buffer.concat(Array.from({ length: (ggmlShape[0]! * ggmlShape[1]!) / 256 }, () => q2_k(codes, scales, minimums)))
      : Buffer.alloc(ggmlShape[0]! * 4, 0);
    directory.push(tensor(name, ggmlShape, isMatrix ? 10 : 0, offset));
    payloadParts.push(payload); offset += payload.length;
  }
  const epsilon = Buffer.alloc(4); epsilon.writeFloatLE(1e-5);
  const metadataEntries = [
    metadataString("general.architecture", "llama"), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 256),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 2), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 128), metadataU32("llama.feed_forward_length", 256), metadata("llama.attention.layer_norm_rms_epsilon", 6, epsilon),
  ];
  await withFixture(fixture(metadataEntries, directory, Buffer.concat(payloadParts)), async (file) => {
    const reader = new GgufCatalogReader(file);
    try {
      const catalog = await reader.inspect();
      const ir = await buildModelIR(catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
      const constants = await materializeReferenceF32Constants(ir, catalog, reader);
      const weight = constants.get("token_embd.weight")!;
      assert.deepEqual(weight.shape, [256, 256]);
      assert.deepEqual([...weight.values.slice(0, 32)], codes.slice(0, 32).map((code, index) => 0.5 * scales[Math.floor(index / 16)]! * code - 0.5 * minimums[Math.floor(index / 16)]!));
      assert.deepEqual(weight.sourceQuantization, { family: "gguf", mode: "q2_k", bits: 2, groupSize: 256, tensorType: "GGML_TYPE_Q2_K" });
    } finally { await reader.close(); }
  });
});
