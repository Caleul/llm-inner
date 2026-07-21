import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Gemma4BinaryConstantPool } from "../src/gemma4-binary-constant-pool.js";
import type { TensorInfo } from "../src/types.js";

test("constant pool binário lê somente o range Safetensors indexado e valida identidade", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-binary-pool-"));
  const payload = Buffer.from([0x40, 0x40, 0x80, 0x40]);
  let header = JSON.stringify({ weight: { dtype: "BF16", shape: [1, 2], data_offsets: [0, payload.length] } });
  while (Buffer.byteLength(header) % 8 !== 0) header += " ";
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(Buffer.byteLength(header)));
  await writeFile(join(directory, "config.json"), "{}\n"); await writeFile(join(directory, "model.safetensors"), Buffer.concat([prefix, Buffer.from(header), payload]));
  const pool = await Gemma4BinaryConstantPool.open(directory);
  try {
    const tensor: TensorInfo = { name: "weight", storageDtype: "BF16", storageShape: [1, 2], logicalShape: [1, 2] };
    assert.deepEqual(await pool.readTensorBytesRange(tensor, 2, 2), payload.subarray(2));
    await assert.rejects(() => pool.readTensorBytesRange({ ...tensor, logicalShape: [2, 1] }, 0, 2), /diverge do programa literal/);
  } finally { await pool.close(); await rm(directory, { recursive: true, force: true }); }
});
