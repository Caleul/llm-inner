import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { open, readFile, readdir, stat, type FileHandle } from "node:fs/promises";
import * as path from "node:path";
import { isDeepStrictEqual } from "node:util";
import type { ModelCatalog } from "./types.js";

const SHA256 = /^[a-f0-9]{64}$/;
const IMMUTABLE_REVISION = /^[a-f0-9]{40}$/;
const MODEL_ID = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/;
const MAX_EMBEDDED_METADATA_BYTES = 48 * 1024 * 1024;
const MAX_EMBEDDED_SAFETENSORS_STRUCTURE_BYTES = 128 * 1024 * 1024;
const SOURCE_RECONSTRUCTION_CHUNK_BYTES = 12 * 1024 * 1024;
const SOURCE_FILE_ROLES = new Set<Gemma4LiteralSourceFileRole>([
  "weights",
  "weights-index",
  "model-config",
  "generation-config",
  "processor-config",
  "tokenizer-config",
  "tokenizer",
  "package-metadata",
]);

export type Gemma4LiteralSourceFileRole =
  | "weights"
  | "weights-index"
  | "model-config"
  | "generation-config"
  | "processor-config"
  | "tokenizer-config"
  | "tokenizer"
  | "package-metadata";

export interface Gemma4LiteralSourceFileCommitment {
  path: string;
  role: Gemma4LiteralSourceFileRole;
  bytes: number;
  sha256: string;
  content:
    | {
        storage: "embedded-metadata-base64";
        encoding: "base64";
        payloadBase64: string;
        decode: "base64-to-original-bytes";
      }
    | {
        storage: "embedded-tensor-constants";
        mapping: "safetensors-file-byte-segments-v1";
        reconstruction: "concatenate segments by ascending byteOffset; embedded bytes are decoded from base64 and tensor bytes are read from the named literal constant";
        segments: Gemma4LiteralSafetensorsFileSegment[];
      };
}

export type Gemma4LiteralSafetensorsFileSegment =
  | {
      kind: "embedded-bytes";
      byteOffset: number;
      byteLength: number;
      encoding: "base64";
      payloadBase64: string;
      sha256: string;
    }
  | {
      kind: "tensor-constant";
      byteOffset: number;
      byteLength: number;
      tensorName: string;
    };

export interface Gemma4LiteralSourceConstantMetadata {
  name: string;
  storageDtype: string;
  storageShape: number[];
  payloadBytes: number;
}

/** Immutable package identity carried by the calculation artifact itself. */
export interface Gemma4LiteralSourceIdentity {
  modelId: string;
  revision: string;
  sourceFormat: "safetensors";
  semanticAdapter: "gemma4-composite-v1";
  files: Gemma4LiteralSourceFileCommitment[];
}

export interface Gemma4LiteralSourceIdentitySummary
  extends Omit<Gemma4LiteralSourceIdentity, "files"> {
  files: Array<
    Omit<Gemma4LiteralSourceFileCommitment, "content"> & {
      content:
        | {
            storage: "embedded-metadata-base64";
            encoding: "base64";
            decode: "base64-to-original-bytes";
          }
        | {
            storage: "embedded-tensor-constants";
            mapping: "safetensors-file-byte-segments-v1";
            reconstruction: "concatenate segments by ascending byteOffset; embedded bytes are decoded from base64 and tensor bytes are read from the named literal constant";
            segments: Array<
              | { kind: "embedded-bytes"; byteOffset: number; byteLength: number; sha256: string }
              | { kind: "tensor-constant"; byteOffset: number; byteLength: number; tensorName: string }
            >;
          };
    }
  >;
  totalFiles: number;
  totalBytes: number;
  embeddedMetadataBytes: number;
  embeddedSafetensorsStructureBytes: number;
}

export interface Gemma4LiteralEmbeddedMetadataRange {
  path: string;
  role: Gemma4LiteralSourceFileRole;
  sourceBytes: number;
  sourceSha256: string;
  offset: number;
  byteLength: number;
  sha256: string;
  encoding: "base64";
  dataBase64: string;
}

export interface Gemma4LiteralSourceFileReconstruction {
  files: number;
  bytes: number;
  weightFiles: number;
  reconstructedWeightBytes: number;
  fileCommitments: Array<{ path: string; role: Gemma4LiteralSourceFileRole; bytes: number; sha256: string }>;
}

/**
 * Navigable proof that a literal tensor range is the same byte range carried
 * by one immutable Safetensors source file. The source identity and its
 * tensor-to-file segment mapping are themselves committed by the artifact
 * integrity manifest.
 */
export interface Gemma4LiteralSourceTensorRangeProvenance {
  kind: "gemma4-literal-source-tensor-range-provenance";
  schemaVersion: 1;
  modelId: string;
  revision: string;
  tensor: string;
  tensorRange: {
    byteOffset: number;
    byteLength: number;
  };
  sourceFile: {
    path: string;
    role: "weights";
    bytes: number;
    sha256: string;
    pointer: string;
  };
  safetensorsRange: {
    byteOffset: number;
    byteLength: number;
    segmentPointer: string;
  };
  artifactCommitment: {
    algorithm: "SHA-256";
    rootSha256: string;
    sourceIdentitySectionSha256: string;
    sourceIdentitySectionPointer: string;
  };
  mapping: "tensor-relative bytes map to the unique source Safetensors tensor segment at segment.byteOffset + tensorRange.byteOffset";
}

/** Resolves a literal tensor range back to its exact immutable source bytes. */
export function buildGemma4LiteralSourceTensorRangeProvenance(
  identity: Gemma4LiteralSourceIdentity,
  tensor: string,
  byteOffset: number,
  byteLength: number,
  artifactCommitment: {
    rootSha256: string;
    sourceIdentitySectionSha256: string;
    sourceIdentitySectionIndex: number;
  },
): Gemma4LiteralSourceTensorRangeProvenance {
  validateGemma4LiteralSourceIdentity(identity);
  if (!tensor || !Number.isSafeInteger(byteOffset) || !Number.isSafeInteger(byteLength) || byteOffset < 0 || byteLength <= 0 ||
    !Number.isSafeInteger(artifactCommitment.sourceIdentitySectionIndex) || artifactCommitment.sourceIdentitySectionIndex < 0 ||
    !SHA256.test(artifactCommitment.rootSha256) || !SHA256.test(artifactCommitment.sourceIdentitySectionSha256)) {
    throw new Error(`${tensor || "tensor"}: range de source ou compromisso estrutural inválido.`);
  }
  const matches: Array<{
    file: Gemma4LiteralSourceFileCommitment;
    fileIndex: number;
    segment: Extract<Gemma4LiteralSafetensorsFileSegment, { kind: "tensor-constant" }>;
    segmentIndex: number;
  }> = [];
  identity.files.forEach((file, fileIndex) => {
    if (file.content.storage !== "embedded-tensor-constants") return;
    file.content.segments.forEach((segment, segmentIndex) => {
      if (segment.kind === "tensor-constant" && segment.tensorName === tensor) {
        matches.push({ file, fileIndex, segment, segmentIndex });
      }
    });
  });
  const match = matches[0];
  if (matches.length !== 1 || !match || byteOffset + byteLength > match.segment.byteLength) {
    throw new Error(`${tensor}: range literal não possui um único mapeamento Safetensors integral.`);
  }
  const sourceFilePointer = `/sourceIdentity/files/${match.fileIndex}`;
  return {
    kind: "gemma4-literal-source-tensor-range-provenance",
    schemaVersion: 1,
    modelId: identity.modelId,
    revision: identity.revision,
    tensor,
    tensorRange: { byteOffset, byteLength },
    sourceFile: {
      path: match.file.path,
      role: "weights",
      bytes: match.file.bytes,
      sha256: match.file.sha256,
      pointer: sourceFilePointer,
    },
    safetensorsRange: {
      byteOffset: match.segment.byteOffset + byteOffset,
      byteLength,
      segmentPointer: `${sourceFilePointer}/content/segments/${match.segmentIndex}`,
    },
    artifactCommitment: {
      algorithm: "SHA-256",
      rootSha256: artifactCommitment.rootSha256,
      sourceIdentitySectionSha256: artifactCommitment.sourceIdentitySectionSha256,
      sourceIdentitySectionPointer: `/integrityManifest/sections/${artifactCommitment.sourceIdentitySectionIndex}`,
    },
    mapping: "tensor-relative bytes map to the unique source Safetensors tensor segment at segment.byteOffset + tensorRange.byteOffset",
  };
}

/** Keeps immutable source identity navigable without expanding large metadata payloads. */
export function summarizeGemma4LiteralSourceIdentity(
  identity: Gemma4LiteralSourceIdentity,
): Gemma4LiteralSourceIdentitySummary {
  validateGemma4LiteralSourceIdentity(identity);
  return {
    modelId: identity.modelId,
    revision: identity.revision,
    sourceFormat: identity.sourceFormat,
    semanticAdapter: identity.semanticAdapter,
    files: identity.files.map((file) => ({
      path: file.path,
      role: file.role,
      bytes: file.bytes,
      sha256: file.sha256,
      content: file.content.storage === "embedded-metadata-base64"
        ? {
            storage: file.content.storage,
            encoding: file.content.encoding,
            decode: file.content.decode,
          }
        : {
            storage: file.content.storage,
            mapping: file.content.mapping,
            reconstruction: file.content.reconstruction,
            segments: file.content.segments.map((segment) => segment.kind === "embedded-bytes"
              ? {
                  kind: segment.kind,
                  byteOffset: segment.byteOffset,
                  byteLength: segment.byteLength,
                  sha256: segment.sha256,
                }
              : {
                  kind: segment.kind,
                  byteOffset: segment.byteOffset,
                  byteLength: segment.byteLength,
                  tensorName: segment.tensorName,
                }),
          },
    })),
    totalFiles: identity.files.length,
    totalBytes: identity.files.reduce((total, file) => total + file.bytes, 0),
    embeddedMetadataBytes: identity.files.reduce((total, file) =>
      total + (file.content.storage === "embedded-metadata-base64" ? file.bytes : 0), 0),
    embeddedSafetensorsStructureBytes: identity.files.reduce((total, file) =>
      total + (file.content.storage === "embedded-tensor-constants"
        ? file.content.segments.reduce((fileTotal, segment) => fileTotal + (segment.kind === "embedded-bytes" ? segment.byteLength : 0), 0)
        : 0), 0),
  };
}

/** Reads one bounded range from metadata or a reconstructed Safetensors shard. */
export async function readGemma4LiteralSourceFileRange(
  identity: Gemma4LiteralSourceIdentity,
  constants: ReadonlyMap<string, Gemma4LiteralSourceConstantMetadata>,
  readTensorBytesRange: (tensor: Gemma4LiteralSourceConstantMetadata, offset: number, byteLength: number) => Promise<Buffer>,
  sourcePath: string,
  offset = 0,
  requestedByteLength?: number,
): Promise<Gemma4LiteralEmbeddedMetadataRange> {
  validateGemma4LiteralSourceIdentity(identity);
  validateGemma4LiteralSourceWeightMappings(identity, constants.values());
  const file = uniqueSourceFile(identity, sourcePath);
  const byteLength = sourceWindowByteLength(file, offset, requestedByteLength);
  if (file.content.storage === "embedded-metadata-base64") {
    return readGemma4LiteralEmbeddedMetadataRange(identity, sourcePath, offset, byteLength);
  }
  const end = offset + byteLength;
  const chunks: Buffer[] = [];
  for (const segment of file.content.segments) {
    const segmentEnd = segment.byteOffset + segment.byteLength;
    const overlapStart = Math.max(offset, segment.byteOffset);
    const overlapEnd = Math.min(end, segmentEnd);
    if (overlapStart >= overlapEnd) continue;
    const relativeOffset = overlapStart - segment.byteOffset;
    const overlapBytes = overlapEnd - overlapStart;
    if (segment.kind === "embedded-bytes") {
      chunks.push(Buffer.from(segment.payloadBase64, "base64").subarray(relativeOffset, relativeOffset + overlapBytes));
    } else {
      const constant = constants.get(segment.tensorName)!;
      chunks.push(await readTensorBytesRange(constant, relativeOffset, overlapBytes));
    }
  }
  const range = Buffer.concat(chunks);
  if (range.length !== byteLength) throw new Error(`${sourcePath}: reconstrução Safetensors produziu uma janela incompleta.`);
  return sourceFileRangeResult(file, offset, range);
}

/** Re-hashes every original package file solely from artifact-owned bytes. */
export async function verifyGemma4LiteralSourceFileReconstruction(
  identity: Gemma4LiteralSourceIdentity,
  constants: ReadonlyMap<string, Gemma4LiteralSourceConstantMetadata>,
  readTensorBytesRange: (tensor: Gemma4LiteralSourceConstantMetadata, offset: number, byteLength: number) => Promise<Buffer>,
): Promise<Gemma4LiteralSourceFileReconstruction> {
  validateGemma4LiteralSourceIdentity(identity);
  validateGemma4LiteralSourceWeightMappings(identity, constants.values());
  const fileCommitments: Gemma4LiteralSourceFileReconstruction["fileCommitments"] = [];
  let reconstructedWeightBytes = 0;
  for (const file of identity.files) {
    const digest = createHash("sha256");
    if (file.content.storage === "embedded-metadata-base64") {
      digest.update(Buffer.from(file.content.payloadBase64, "base64"));
    } else {
      for (const segment of file.content.segments) {
        if (segment.kind === "embedded-bytes") {
          digest.update(Buffer.from(segment.payloadBase64, "base64"));
          continue;
        }
        const constant = constants.get(segment.tensorName)!;
        for (let offset = 0; offset < segment.byteLength; offset += SOURCE_RECONSTRUCTION_CHUNK_BYTES) {
          const byteLength = Math.min(SOURCE_RECONSTRUCTION_CHUNK_BYTES, segment.byteLength - offset);
          const bytes = await readTensorBytesRange(constant, offset, byteLength);
          if (bytes.length !== byteLength) throw new Error(`${segment.tensorName}: reconstrução retornou range incompleto.`);
          digest.update(bytes);
        }
      }
      reconstructedWeightBytes += file.bytes;
    }
    const sha256 = digest.digest("hex");
    if (sha256 !== file.sha256) throw new Error(`${file.path}: reconstrução do arquivo fonte diverge do SHA-256 imutável.`);
    fileCommitments.push({ path: file.path, role: file.role, bytes: file.bytes, sha256 });
  }
  return {
    files: identity.files.length,
    bytes: identity.files.reduce((total, file) => total + file.bytes, 0),
    weightFiles: identity.files.filter((file) => file.role === "weights").length,
    reconstructedWeightBytes,
    fileCommitments,
  };
}

/** Decodes one explicit, bounded metadata range entirely from the artifact-owned identity. */
export function readGemma4LiteralEmbeddedMetadataRange(
  identity: Gemma4LiteralSourceIdentity,
  sourcePath: string,
  offset = 0,
  requestedByteLength?: number,
): Gemma4LiteralEmbeddedMetadataRange {
  validateGemma4LiteralSourceIdentity(identity);
  const matches = identity.files.filter((file) => file.path === sourcePath);
  const file = matches[0];
  if (matches.length !== 1 || !file) {
    throw new Error(`${sourcePath}: metadata de source Gemma 4 não encontrada.`);
  }
  if (file.content.storage !== "embedded-metadata-base64") {
    throw new Error(`${sourcePath}: bytes de weights devem ser navegados pelas constantes incorporadas.`);
  }
  if (!Number.isSafeInteger(offset) || offset < 0 || offset >= file.bytes) {
    throw new Error(`${sourcePath}: offset de metadata Gemma 4 fora do arquivo incorporado.`);
  }
  const byteLength = requestedByteLength ?? Math.min(4096, file.bytes - offset);
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || offset + byteLength > file.bytes) {
    throw new Error(`${sourcePath}: janela de metadata Gemma 4 fora do arquivo incorporado.`);
  }
  const decoded = Buffer.from(file.content.payloadBase64, "base64");
  const range = decoded.subarray(offset, offset + byteLength);
  return {
    path: file.path,
    role: file.role,
    sourceBytes: file.bytes,
    sourceSha256: file.sha256,
    offset,
    byteLength: range.length,
    sha256: createHash("sha256").update(range).digest("hex"),
    encoding: "base64",
    dataBase64: range.toString("base64"),
  };
}

/**
 * Hashes the exact source package and embeds every top-level JSON metadata
 * file. Safetensors payloads are already embedded as named constants; their
 * complete container files remain bound by SHA-256, byte length and shard
 * name without duplicating multi-GiB bytes a second time.
 */
export async function buildGemma4LiteralSourceIdentity(
  catalog: ModelCatalog,
  modelId: string,
  revision: string,
): Promise<Gemma4LiteralSourceIdentity> {
  if (catalog.format !== "safetensors") throw new Error("Identidade Gemma 4 literal requer pacote Safetensors denso.");
  const source = catalog.source;
  const entries = await readdir(source, { withFileTypes: true });
  const jsonFiles = entries.filter((entry) => entry.isFile() && entry.name.endsWith(".json")).map((entry) => entry.name);
  const shards = [...new Set([...catalog.tensors.values()].map((tensor) => tensor.shard))];
  if (shards.some((shard) => !shard || path.basename(shard) !== shard)) {
    throw new Error("Identidade Gemma 4 literal encontrou shard ausente ou fora da raiz do pacote.");
  }
  const weightFiles = shards as string[];
  const selected = [...new Set([...weightFiles, ...jsonFiles])].sort((left, right) => left.localeCompare(right, "en"));
  if (!selected.includes("config.json") || weightFiles.length === 0) {
    throw new Error("Identidade Gemma 4 literal requer config.json e ao menos um shard Safetensors.");
  }

  let embeddedMetadataBytes = 0;
  let embeddedSafetensorsStructureBytes = 0;
  const files: Gemma4LiteralSourceFileCommitment[] = [];
  for (const name of selected) {
    const absolute = path.join(source, name);
    const info = await stat(absolute);
    if (!info.isFile() || !Number.isSafeInteger(info.size) || info.size < 0) {
      throw new Error(`${name}: arquivo de identidade Gemma 4 inválido.`);
    }
    const weights = weightFiles.includes(name);
    if (weights) {
      const mapped = await buildSafetensorsFileMapping(
        absolute,
        info.size,
        [...catalog.tensors.values()].filter((tensor) => tensor.shard === name),
      );
      embeddedSafetensorsStructureBytes += mapped.embeddedBytes;
      if (embeddedSafetensorsStructureBytes > MAX_EMBEDDED_SAFETENSORS_STRUCTURE_BYTES) {
        throw new Error(`Estrutura Safetensors Gemma 4 excede ${MAX_EMBEDDED_SAFETENSORS_STRUCTURE_BYTES} bytes incorporáveis.`);
      }
      files.push({
        path: name,
        role: "weights",
        bytes: info.size,
        sha256: await sha256File(absolute),
        content: mapped.content,
      });
      continue;
    }
    embeddedMetadataBytes += info.size;
    if (embeddedMetadataBytes > MAX_EMBEDDED_METADATA_BYTES) {
      throw new Error(`Metadados Gemma 4 excedem ${MAX_EMBEDDED_METADATA_BYTES} bytes incorporáveis.`);
    }
    const bytes = await readFile(absolute);
    try { JSON.parse(bytes.toString("utf8")); } catch (error) {
      throw new Error(`${name}: metadata JSON inválido: ${(error as Error).message}`);
    }
    files.push({
      path: name,
      role: sourceRole(name),
      bytes: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      content: {
        storage: "embedded-metadata-base64",
        encoding: "base64",
        payloadBase64: bytes.toString("base64"),
        decode: "base64-to-original-bytes",
      },
    });
  }
  const identity: Gemma4LiteralSourceIdentity = {
    modelId,
    revision,
    sourceFormat: "safetensors",
    semanticAdapter: "gemma4-composite-v1",
    files,
  };
  validateGemma4LiteralSourceIdentity(identity);
  return identity;
}

/** Recomputes every commitment from the live package before source removal. */
export async function verifyGemma4LiteralSourceIdentityAgainstCatalog(
  identity: Gemma4LiteralSourceIdentity,
  catalog: ModelCatalog,
): Promise<{ files: number; bytes: number }> {
  validateGemma4LiteralSourceIdentity(identity);
  const actual = await buildGemma4LiteralSourceIdentity(catalog, identity.modelId, identity.revision);
  if (!isDeepStrictEqual(identity, actual)) {
    throw new Error("Identidade imutável Gemma 4 diverge dos arquivos do pacote fonte.");
  }
  return {
    files: actual.files.length,
    bytes: actual.files.reduce((total, file) => total + file.bytes, 0),
  };
}

export function validateGemma4LiteralSourceIdentity(identity: Gemma4LiteralSourceIdentity): void {
  if (!identity || !MODEL_ID.test(identity.modelId) || !IMMUTABLE_REVISION.test(identity.revision) ||
    identity.sourceFormat !== "safetensors" || identity.semanticAdapter !== "gemma4-composite-v1" || !Array.isArray(identity.files)) {
    throw new Error("Programa literal Gemma 4 possui identidade de source inválida.");
  }
  const names = new Set<string>();
  const tensorNames = new Set<string>();
  let configs = 0, weights = 0;
  for (const [index, file] of identity.files.entries()) {
    if (!file || !file.path || path.basename(file.path) !== file.path || names.has(file.path) ||
      !Number.isSafeInteger(file.bytes) || file.bytes < 0 || !SHA256.test(file.sha256) || !SOURCE_FILE_ROLES.has(file.role)) {
      throw new Error(`${file?.path ?? "arquivo"}: compromisso de source Gemma 4 inválido.`);
    }
    if (index > 0 && identity.files[index - 1]!.path.localeCompare(file.path, "en") >= 0) {
      throw new Error(`${file.path}: compromissos de source Gemma 4 não estão em ordem determinística.`);
    }
    names.add(file.path);
    if (file.role === "weights") {
      weights += 1;
      if (!file.path.endsWith(".safetensors") || file.content.storage !== "embedded-tensor-constants" ||
        file.content.mapping !== "safetensors-file-byte-segments-v1" ||
        file.content.reconstruction !== "concatenate segments by ascending byteOffset; embedded bytes are decoded from base64 and tensor bytes are read from the named literal constant") {
        throw new Error(`${file.path}: compromisso de weights Gemma 4 inválido.`);
      }
      validateSafetensorsSegments(file, tensorNames);
      continue;
    }
    if (file.path === "config.json" && file.role === "model-config") configs += 1;
    if (!file.path.endsWith(".json") || file.content.storage !== "embedded-metadata-base64" ||
      file.content.encoding !== "base64" || file.content.decode !== "base64-to-original-bytes") {
      throw new Error(`${file.path}: metadata incorporada Gemma 4 inválida.`);
    }
    const decoded = Buffer.from(file.content.payloadBase64, "base64");
    if (decoded.toString("base64") !== file.content.payloadBase64 || decoded.length !== file.bytes ||
      createHash("sha256").update(decoded).digest("hex") !== file.sha256) {
      throw new Error(`${file.path}: bytes incorporados não correspondem ao compromisso de source.`);
    }
    try { JSON.parse(decoded.toString("utf8")); } catch {
      throw new Error(`${file.path}: bytes incorporados não são JSON válido.`);
    }
  }
  if (configs !== 1 || weights === 0) throw new Error("Identidade Gemma 4 literal requer um config.json e ao menos um Safetensors.");
}

/** Binds every source header tensor entry to exactly one embedded literal constant. */
export function validateGemma4LiteralSourceWeightMappings(
  identity: Gemma4LiteralSourceIdentity,
  constants: Iterable<Gemma4LiteralSourceConstantMetadata>,
): void {
  validateGemma4LiteralSourceIdentity(identity);
  const byName = new Map<string, Gemma4LiteralSourceConstantMetadata>();
  for (const constant of constants) {
    if (!constant?.name || byName.has(constant.name) || !Number.isSafeInteger(constant.payloadBytes) || constant.payloadBytes <= 0) {
      throw new Error(`${constant?.name ?? "constante"}: metadata de constante para identidade Safetensors inválida.`);
    }
    byName.set(constant.name, constant);
  }
  const mapped = new Set<string>();
  for (const file of identity.files) {
    if (file.content.storage !== "embedded-tensor-constants") continue;
    const header = safetensorsHeader(file);
    for (const segment of file.content.segments) {
      if (segment.kind !== "tensor-constant") continue;
      const constant = byName.get(segment.tensorName);
      const entry = header[segment.tensorName];
      if (!constant || !entry || constant.payloadBytes !== segment.byteLength || entry.dtype !== constant.storageDtype ||
        !isDeepStrictEqual(entry.shape, constant.storageShape) || mapped.has(segment.tensorName)) {
        throw new Error(`${segment.tensorName}: range de source Safetensors diverge da constante literal incorporada.`);
      }
      mapped.add(segment.tensorName);
    }
  }
  if (mapped.size !== byName.size || [...byName.keys()].some((name) => !mapped.has(name))) {
    throw new Error("Identidade Safetensors não mapeia exatamente todas as constantes literais incorporadas.");
  }
}

type ParsedSafetensorsHeader = Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;

function validateSafetensorsSegments(file: Gemma4LiteralSourceFileCommitment, globalTensorNames: Set<string>): void {
  if (file.content.storage !== "embedded-tensor-constants" || !Array.isArray(file.content.segments) || file.content.segments.length < 2) {
    throw new Error(`${file.path}: segmentos Safetensors ausentes.`);
  }
  let cursor = 0;
  for (const segment of file.content.segments) {
    if (!segment || !Number.isSafeInteger(segment.byteOffset) || !Number.isSafeInteger(segment.byteLength) ||
      segment.byteOffset !== cursor || segment.byteLength <= 0) {
      throw new Error(`${file.path}: cobertura de segmentos Safetensors inválida.`);
    }
    cursor += segment.byteLength;
    if (segment.kind === "embedded-bytes") {
      const decoded = Buffer.from(segment.payloadBase64, "base64");
      if (segment.encoding !== "base64" || decoded.toString("base64") !== segment.payloadBase64 ||
        decoded.length !== segment.byteLength || createHash("sha256").update(decoded).digest("hex") !== segment.sha256) {
        throw new Error(`${file.path}: bytes estruturais Safetensors incorporados inválidos.`);
      }
    } else if (segment.kind === "tensor-constant") {
      if (!segment.tensorName || globalTensorNames.has(segment.tensorName)) {
        throw new Error(`${file.path}: tensor Safetensors ausente ou duplicado no mapeamento.`);
      }
      globalTensorNames.add(segment.tensorName);
    } else {
      throw new Error(`${file.path}: tipo de segmento Safetensors desconhecido.`);
    }
  }
  if (cursor !== file.bytes) throw new Error(`${file.path}: segmentos Safetensors não cobrem o arquivo inteiro.`);
  const header = safetensorsHeader(file);
  const segments = new Map(file.content.segments.filter((segment) => segment.kind === "tensor-constant").map((segment) => [segment.tensorName, segment]));
  if (Object.keys(header).length !== segments.size) throw new Error(`${file.path}: header e segmentos Safetensors possuem cardinalidade divergente.`);
  const structural = file.content.segments[0]!;
  if (structural.kind !== "embedded-bytes" || structural.byteOffset !== 0) throw new Error(`${file.path}: prefixo/header Safetensors não está incorporado.`);
  const headerLength = Number(Buffer.from(structural.payloadBase64, "base64").readBigUInt64LE(0));
  for (const [name, entry] of Object.entries(header)) {
    const segment = segments.get(name);
    if (!segment || segment.byteOffset !== 8 + headerLength + entry.data_offsets[0] ||
      segment.byteLength !== entry.data_offsets[1] - entry.data_offsets[0]) {
      throw new Error(`${name}: segmento não corresponde ao data_offsets do header Safetensors.`);
    }
  }
}

function safetensorsHeader(file: Gemma4LiteralSourceFileCommitment): ParsedSafetensorsHeader {
  if (file.content.storage !== "embedded-tensor-constants") throw new Error(`${file.path}: arquivo não é Safetensors reconstruível.`);
  const first = file.content.segments[0];
  if (!first || first.kind !== "embedded-bytes") throw new Error(`${file.path}: prefixo Safetensors ausente.`);
  const bytes = Buffer.from(first.payloadBase64, "base64");
  if (bytes.length < 8) throw new Error(`${file.path}: prefixo Safetensors truncado.`);
  const headerLengthBig = bytes.readBigUInt64LE(0);
  if (headerLengthBig > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${file.path}: header Safetensors excede inteiro seguro.`);
  const headerLength = Number(headerLengthBig);
  if (headerLength <= 0 || 8 + headerLength > bytes.length) throw new Error(`${file.path}: header Safetensors não está integralmente incorporado.`);
  let raw: unknown;
  try { raw = JSON.parse(bytes.subarray(8, 8 + headerLength).toString("utf8")); } catch {
    throw new Error(`${file.path}: header Safetensors incorporado não é JSON válido.`);
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error(`${file.path}: header Safetensors incorporado inválido.`);
  const result: ParsedSafetensorsHeader = {};
  for (const [name, value] of Object.entries(raw as Record<string, unknown>)) {
    if (name === "__metadata__") continue;
    const entry = value as Partial<ParsedSafetensorsHeader[string]>;
    if (!entry || typeof entry !== "object" || typeof entry.dtype !== "string" || !Array.isArray(entry.shape) ||
      entry.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 0) || !Array.isArray(entry.data_offsets) ||
      entry.data_offsets.length !== 2 || entry.data_offsets.some((offset) => !Number.isSafeInteger(offset) || offset < 0) || entry.data_offsets[1]! < entry.data_offsets[0]!) {
      throw new Error(`${name}: entrada do header Safetensors incorporado inválida.`);
    }
    result[name] = { dtype: entry.dtype, shape: [...entry.shape], data_offsets: [...entry.data_offsets] as [number, number] };
  }
  return result;
}

async function buildSafetensorsFileMapping(
  absolute: string,
  fileBytes: number,
  tensors: Array<{ name: string; byteOffset?: number; byteLength?: number }>,
): Promise<{ content: Extract<Gemma4LiteralSourceFileCommitment["content"], { storage: "embedded-tensor-constants" }>; embeddedBytes: number }> {
  if (tensors.length === 0) throw new Error(`${path.basename(absolute)}: shard não possui tensores catalogados.`);
  const ordered = [...tensors].sort((left, right) => (left.byteOffset ?? -1) - (right.byteOffset ?? -1) || left.name.localeCompare(right.name, "en"));
  const file = await open(absolute, "r");
  try {
    const segments: Gemma4LiteralSafetensorsFileSegment[] = [];
    let cursor = 0, embeddedBytes = 0;
    for (const tensor of ordered) {
      if (!Number.isSafeInteger(tensor.byteOffset) || !Number.isSafeInteger(tensor.byteLength) || tensor.byteOffset! < cursor || tensor.byteLength! <= 0 || tensor.byteOffset! + tensor.byteLength! > fileBytes) {
        throw new Error(`${tensor.name}: range absoluto Safetensors inválido para identidade literal.`);
      }
      if (tensor.byteOffset! > cursor) {
        const embedded = await embeddedFileSegment(file, cursor, tensor.byteOffset! - cursor);
        segments.push(embedded);
        embeddedBytes += embedded.byteLength;
      }
      segments.push({ kind: "tensor-constant", byteOffset: tensor.byteOffset!, byteLength: tensor.byteLength!, tensorName: tensor.name });
      cursor = tensor.byteOffset! + tensor.byteLength!;
    }
    if (cursor < fileBytes) {
      const embedded = await embeddedFileSegment(file, cursor, fileBytes - cursor);
      segments.push(embedded);
      embeddedBytes += embedded.byteLength;
    }
    return {
      content: {
        storage: "embedded-tensor-constants",
        mapping: "safetensors-file-byte-segments-v1",
        reconstruction: "concatenate segments by ascending byteOffset; embedded bytes are decoded from base64 and tensor bytes are read from the named literal constant",
        segments,
      },
      embeddedBytes,
    };
  } finally {
    await file.close();
  }
}

async function embeddedFileSegment(file: FileHandle, byteOffset: number, byteLength: number): Promise<Extract<Gemma4LiteralSafetensorsFileSegment, { kind: "embedded-bytes" }>> {
  const bytes = Buffer.allocUnsafe(byteLength);
  let read = 0;
  while (read < byteLength) {
    const result = await file.read(bytes, read, byteLength - read, byteOffset + read);
    if (result.bytesRead === 0) throw new Error("Arquivo Safetensors terminou durante a incorporação estrutural.");
    read += result.bytesRead;
  }
  return {
    kind: "embedded-bytes",
    byteOffset,
    byteLength,
    encoding: "base64",
    payloadBase64: bytes.toString("base64"),
    sha256: createHash("sha256").update(bytes).digest("hex"),
  };
}

function uniqueSourceFile(identity: Gemma4LiteralSourceIdentity, sourcePath: string): Gemma4LiteralSourceFileCommitment {
  const matches = identity.files.filter((file) => file.path === sourcePath);
  if (matches.length !== 1) throw new Error(`${sourcePath}: arquivo de source Gemma 4 não encontrado.`);
  return matches[0]!;
}

function sourceWindowByteLength(file: Gemma4LiteralSourceFileCommitment, offset: number, requestedByteLength?: number): number {
  if (!Number.isSafeInteger(offset) || offset < 0 || offset >= file.bytes) throw new Error(`${file.path}: offset de source Gemma 4 fora do arquivo incorporado.`);
  const byteLength = requestedByteLength ?? Math.min(4096, file.bytes - offset);
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || offset + byteLength > file.bytes) throw new Error(`${file.path}: janela de source Gemma 4 fora do arquivo incorporado.`);
  return byteLength;
}

function sourceFileRangeResult(file: Gemma4LiteralSourceFileCommitment, offset: number, range: Buffer): Gemma4LiteralEmbeddedMetadataRange {
  return {
    path: file.path,
    role: file.role,
    sourceBytes: file.bytes,
    sourceSha256: file.sha256,
    offset,
    byteLength: range.length,
    sha256: createHash("sha256").update(range).digest("hex"),
    encoding: "base64",
    dataBase64: range.toString("base64"),
  };
}

function sourceRole(name: string): Gemma4LiteralSourceFileRole {
  if (name.endsWith(".safetensors.index.json")) return "weights-index";
  if (name === "config.json") return "model-config";
  if (name === "generation_config.json") return "generation-config";
  if (name === "processor_config.json") return "processor-config";
  if (name === "tokenizer_config.json") return "tokenizer-config";
  if (name === "tokenizer.json") return "tokenizer";
  return "package-metadata";
}

async function sha256File(file: string): Promise<string> {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(file)) digest.update(chunk as Buffer);
  return digest.digest("hex");
}
