import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { GgufCatalogReader } from "../src/gguf.js";

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
    [tensor("blk.0.attn_q.weight", [32], 2, 0)],
    Buffer.alloc(32),
  );
  await withFixture(bytes, async (file) => {
    const reader = new GgufCatalogReader(file);
    try { await assert.rejects(() => reader.inspect(), /GGML_TYPE_Q4_0.*sem decodificador/); } finally { await reader.close(); }
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
