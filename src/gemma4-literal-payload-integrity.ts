import { createHash } from "node:crypto";

/** Divisible by three so every non-final Base64 chunk has no padding. */
export const GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES = 12 * 1024 * 1024;

export interface Gemma4LiteralPayloadIntegrityChunk {
  ordinal: number;
  byteOffset: number;
  byteLength: number;
  sha256: string;
}

export interface Gemma4CompositeLiteralPayloadIntegrityEntry {
  name: string;
  payloadBytes: number;
  sha256: string;
  chunking: {
    schemaVersion: 1;
    chunkBytes: typeof GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES;
    algorithm: "SHA-256";
    coverage: "ordered-gap-free-decoded-payload-bytes";
    chunks: Gemma4LiteralPayloadIntegrityChunk[];
  };
}

export interface Gemma4LiteralAuthenticatedPayloadRange {
  kind: "gemma4-literal-authenticated-payload-range";
  schemaVersion: 1;
  tensor: string;
  requestedRange: {
    byteOffset: number;
    byteLength: number;
  };
  payloadCommitment: {
    payloadBytes: number;
    sha256: string;
    pointer: string;
  };
  authenticatedChunks: Array<Gemma4LiteralPayloadIntegrityChunk & { pointer: string }>;
  artifactCommitment: {
    algorithm: "SHA-256";
    rootSha256: string;
    payloadIntegritySectionSha256: string;
    payloadIntegritySectionPointer: string;
  };
  verification: "requested decoded bytes were returned only after every covering canonical Base64 chunk matched its integrity-manifest-bound SHA-256";
}

/** Describes the exact integrity chain already enforced by an authenticated range read. */
export function buildGemma4LiteralAuthenticatedPayloadRange(
  entry: Gemma4CompositeLiteralPayloadIntegrityEntry,
  byteOffset: number,
  byteLength: number,
  payloadIntegrityIndex: number,
  artifactCommitment: {
    rootSha256: string;
    payloadIntegritySectionSha256: string;
    payloadIntegritySectionIndex: number;
  },
): Gemma4LiteralAuthenticatedPayloadRange {
  validateGemma4LiteralPayloadIntegrityMetadata(entry, entry.name, entry.payloadBytes);
  if (!Number.isSafeInteger(byteOffset) || !Number.isSafeInteger(byteLength) || byteOffset < 0 || byteLength <= 0 ||
    byteOffset + byteLength > entry.payloadBytes || !Number.isSafeInteger(payloadIntegrityIndex) || payloadIntegrityIndex < 0 ||
    !Number.isSafeInteger(artifactCommitment.payloadIntegritySectionIndex) || artifactCommitment.payloadIntegritySectionIndex < 0 ||
    !isSha256(artifactCommitment.rootSha256) || !isSha256(artifactCommitment.payloadIntegritySectionSha256)) {
    throw new Error(`${entry.name}: range autenticado ou compromisso estrutural inválido.`);
  }
  const first = Math.floor(byteOffset / entry.chunking.chunkBytes);
  const last = Math.floor((byteOffset + byteLength - 1) / entry.chunking.chunkBytes);
  const payloadPointer = `/payloadIntegrity/${payloadIntegrityIndex}`;
  return {
    kind: "gemma4-literal-authenticated-payload-range",
    schemaVersion: 1,
    tensor: entry.name,
    requestedRange: { byteOffset, byteLength },
    payloadCommitment: {
      payloadBytes: entry.payloadBytes,
      sha256: entry.sha256,
      pointer: payloadPointer,
    },
    authenticatedChunks: entry.chunking.chunks.slice(first, last + 1).map((chunk) => ({
      ...structuredClone(chunk),
      pointer: `${payloadPointer}/chunking/chunks/${chunk.ordinal}`,
    })),
    artifactCommitment: {
      algorithm: "SHA-256",
      rootSha256: artifactCommitment.rootSha256,
      payloadIntegritySectionSha256: artifactCommitment.payloadIntegritySectionSha256,
      payloadIntegritySectionPointer: `/integrityManifest/sections/${artifactCommitment.payloadIntegritySectionIndex}`,
    },
    verification: "requested decoded bytes were returned only after every covering canonical Base64 chunk matched its integrity-manifest-bound SHA-256",
  };
}

/** Builds whole-payload and independently readable chunk commitments. */
export function buildGemma4LiteralPayloadIntegrity(
  name: string,
  bytes: Buffer,
): Gemma4CompositeLiteralPayloadIntegrityEntry {
  const chunks: Gemma4LiteralPayloadIntegrityChunk[] = [];
  for (let byteOffset = 0, ordinal = 0; byteOffset < bytes.length; byteOffset += GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES, ordinal += 1) {
    const byteLength = Math.min(GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES, bytes.length - byteOffset);
    chunks.push({
      ordinal,
      byteOffset,
      byteLength,
      sha256: createHash("sha256").update(bytes.subarray(byteOffset, byteOffset + byteLength)).digest("hex"),
    });
  }
  return payloadIntegrityEntry(name, bytes.length, createHash("sha256").update(bytes).digest("hex"), chunks);
}

/** Finalizes the streaming writer's already-computed whole and per-chunk digests. */
export function payloadIntegrityEntry(
  name: string,
  payloadBytes: number,
  sha256: string,
  chunks: Gemma4LiteralPayloadIntegrityChunk[],
): Gemma4CompositeLiteralPayloadIntegrityEntry {
  const entry: Gemma4CompositeLiteralPayloadIntegrityEntry = {
    name,
    payloadBytes,
    sha256,
    chunking: {
      schemaVersion: 1,
      chunkBytes: GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES,
      algorithm: "SHA-256",
      coverage: "ordered-gap-free-decoded-payload-bytes",
      chunks,
    },
  };
  validateGemma4LiteralPayloadIntegrityMetadata(entry, name, payloadBytes);
  return entry;
}

/** Validates complete, canonical coverage without trusting the payload itself. */
export function validateGemma4LiteralPayloadIntegrityMetadata(
  entry: Gemma4CompositeLiteralPayloadIntegrityEntry,
  expectedName: string,
  expectedPayloadBytes: number,
): void {
  if (entry.name !== expectedName || entry.payloadBytes !== expectedPayloadBytes || !isSha256(entry.sha256) ||
    entry.chunking?.schemaVersion !== 1 || entry.chunking.chunkBytes !== GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES ||
    entry.chunking.algorithm !== "SHA-256" || entry.chunking.coverage !== "ordered-gap-free-decoded-payload-bytes" ||
    !Array.isArray(entry.chunking.chunks)) {
    throw new Error(`${expectedName}: compromisso de integridade de payload inválido.`);
  }
  const expectedChunks = Math.ceil(expectedPayloadBytes / GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES);
  if (expectedChunks <= 0 || entry.chunking.chunks.length !== expectedChunks) {
    throw new Error(`${expectedName}: cobertura de chunks do payload incompleta.`);
  }
  for (let ordinal = 0; ordinal < expectedChunks; ordinal += 1) {
    const chunk = entry.chunking.chunks[ordinal];
    const byteOffset = ordinal * GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES;
    const byteLength = Math.min(GEMMA4_LITERAL_PAYLOAD_CHUNK_BYTES, expectedPayloadBytes - byteOffset);
    if (!chunk || chunk.ordinal !== ordinal || chunk.byteOffset !== byteOffset || chunk.byteLength !== byteLength || !isSha256(chunk.sha256)) {
      throw new Error(`${expectedName}: chunk de integridade ${ordinal} inválido ou fora de ordem.`);
    }
  }
}

export function assertGemma4LiteralPayloadChunkBytes(
  entry: Gemma4CompositeLiteralPayloadIntegrityEntry,
  chunk: Gemma4LiteralPayloadIntegrityChunk,
  bytes: Buffer,
): void {
  if (bytes.length !== chunk.byteLength || createHash("sha256").update(bytes).digest("hex") !== chunk.sha256) {
    throw new Error(`${entry.name}: chunk literal ${chunk.ordinal} diverge do digest incorporado.`);
  }
}

function isSha256(value: unknown): value is string {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}
