import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, readdir, stat } from "node:fs/promises";
import * as path from "node:path";
import { isDeepStrictEqual } from "node:util";
import type { ModelCatalog } from "./types.js";

const SHA256 = /^[a-f0-9]{64}$/;
const IMMUTABLE_REVISION = /^[a-f0-9]{40}$/;
const MODEL_ID = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/;
const MAX_EMBEDDED_METADATA_BYTES = 48 * 1024 * 1024;
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
        mapping: "safetensors-header-ranges-to-named-constants";
      };
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
            mapping: "safetensors-header-ranges-to-named-constants";
          };
    }
  >;
  totalFiles: number;
  totalBytes: number;
  embeddedMetadataBytes: number;
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
        : { storage: file.content.storage, mapping: file.content.mapping },
    })),
    totalFiles: identity.files.length,
    totalBytes: identity.files.reduce((total, file) => total + file.bytes, 0),
    embeddedMetadataBytes: identity.files.reduce((total, file) =>
      total + (file.content.storage === "embedded-metadata-base64" ? file.bytes : 0), 0),
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
  const files: Gemma4LiteralSourceFileCommitment[] = [];
  for (const name of selected) {
    const absolute = path.join(source, name);
    const info = await stat(absolute);
    if (!info.isFile() || !Number.isSafeInteger(info.size) || info.size < 0) {
      throw new Error(`${name}: arquivo de identidade Gemma 4 inválido.`);
    }
    const weights = weightFiles.includes(name);
    if (weights) {
      files.push({
        path: name,
        role: "weights",
        bytes: info.size,
        sha256: await sha256File(absolute),
        content: { storage: "embedded-tensor-constants", mapping: "safetensors-header-ranges-to-named-constants" },
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
        file.content.mapping !== "safetensors-header-ranges-to-named-constants") {
        throw new Error(`${file.path}: compromisso de weights Gemma 4 inválido.`);
      }
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
