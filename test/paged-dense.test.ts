import assert from "node:assert/strict";
import test from "node:test";
import { readLiteralDenseF32Tensor } from "../src/paged-dense.js";
import type { TensorInfo } from "../src/types.js";

test("literal dense materialization decodes an arbitrary-rank BF16 tensor through a bounded range read", async () => {
  const tensor: TensorInfo = { name: "audio.weight", storageDtype: "BF16", storageShape: [1, 1, 3], logicalShape: [1, 1, 3] };
  const bytes = Buffer.alloc(6);
  [0x3f80, 0xc000, 0x3f00].forEach((bits, index) => bytes.writeUInt16LE(bits, index * 2));
  const reads: Array<[number, number]> = [];
  const result = await readLiteralDenseF32Tensor(tensor, {
    readTensorBytesRange: async (_tensor, offset, byteLength) => { reads.push([offset, byteLength]); return bytes.subarray(offset, offset + byteLength); },
  }, 6);
  assert.deepEqual(result.shape, [1, 1, 3]);
  assert.deepEqual([...result.values], [1, -2, 0.5]);
  assert.deepEqual(reads, [[0, 6]]);
});

test("literal dense materialization rejects oversized and truncated payloads", async () => {
  const tensor: TensorInfo = { name: "audio.weight", storageDtype: "F32", storageShape: [2, 2], logicalShape: [2, 2] };
  const reader = { readTensorBytesRange: async () => Buffer.alloc(15) };
  await assert.rejects(readLiteralDenseF32Tensor(tensor, reader, 15), /excede maxTensorBytes=15/);
  await assert.rejects(readLiteralDenseF32Tensor(tensor, reader, 16), /retornou 15 bytes; esperados 16/);
});
