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

test("native GGUF reader rejects unsupported packed GGML types instead of guessing a dequantizer", async () => {
  const bytes = fixture(
    [metadataString("general.architecture", "llama")],
    [tensor("blk.0.attn_q.weight", [32], 3, 0)],
    Buffer.alloc(32),
  );
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /GGML_TYPE_Q4_1.*sem decodificador/); } finally { await reader.close(); }
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

test("native GGUF reader rejects Q8_0 shapes whose fastest GGML dimension cannot contain complete blocks", async () => {
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("bad", [31], 8, 0)], Buffer.alloc(34));
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q8_0 exige a primeira dimensão GGML positiva e múltipla de 32/); } finally { await reader.close(); }
  });
});

test("native GGUF reader rejects Q4_0 shapes whose fastest GGML dimension cannot contain complete blocks", async () => {
  const bytes = fixture([metadataString("general.architecture", "llama")], [tensor("bad", [31], 2, 0)], Buffer.alloc(18));
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /Q4_0 exige a primeira dimensão GGML positiva e múltipla de 32/); } finally { await reader.close(); }
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
