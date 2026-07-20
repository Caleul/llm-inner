import assert from "node:assert/strict";
import test from "node:test";
import {
  assertGemma4LiteralPayloadChunkBytes,
  buildGemma4LiteralPayloadIntegrity,
  GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES,
  validateGemma4LiteralPayloadIntegrityMetadata,
} from "../src/gemma4-literal-payload-integrity.js";

test("Gemma 4 payload integrity partitions decoded bytes into ordered gap-free commitments", () => {
  const bytes = Buffer.alloc(GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES + 7);
  for (let index = 0; index < bytes.length; index += 1) bytes[index] = index % 251;
  const integrity = buildGemma4LiteralPayloadIntegrity("model.tensor", bytes);

  validateGemma4LiteralPayloadIntegrityMetadata(integrity, "model.tensor", bytes.length);
  assert.deepEqual(integrity.chunking.chunks.map((chunk) => [chunk.ordinal, chunk.byteOffset, chunk.byteLength]), [
    [0, 0, GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES],
    [1, GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES, 7],
  ]);
  for (const chunk of integrity.chunking.chunks) {
    assertGemma4LiteralPayloadChunkBytes(integrity, chunk, bytes.subarray(chunk.byteOffset, chunk.byteOffset + chunk.byteLength));
  }

  const shifted = structuredClone(integrity);
  const shiftedSecondChunk = shifted.chunking.chunks[1]!;
  shiftedSecondChunk.byteOffset += 1;
  assert.throws(
    () => validateGemma4LiteralPayloadIntegrityMetadata(shifted, "model.tensor", bytes.length),
    /inválido ou fora de ordem/,
  );
  const corrupted = Buffer.from(bytes.subarray(0, GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES));
  corrupted[0] = corrupted[0]! ^ 1;
  assert.throws(
    () => assertGemma4LiteralPayloadChunkBytes(integrity, integrity.chunking.chunks[0]!, corrupted),
    /diverge do digest incorporado/,
  );
});
