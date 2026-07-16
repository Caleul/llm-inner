import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { openCatalog } from "./catalog.js";

const PAYLOAD_MARKER = /"payloadBase64"\s*:\s*"/;
const PAYLOAD_MARKER_CARRY = 64;

export interface LiteralArtifactAudit {
  artifact: string;
  artifactBytes: number;
  artifactSha256: string;
  constants: number;
  embeddedPayloadBytes: number;
  payloadBase64Characters: number;
  forbiddenSourcePathPresent: boolean;
}

/**
 * Streams a literal JSON artifact without JSON.parse, which is essential for
 * real checkpoints whose base64 JSON is larger than the available V8 heap.
 * It recognizes only the deliberately constrained payload field emitted by
 * the literal writers: base64 contains no quote, so the closing quote is an
 * unambiguous streaming delimiter.
 */
export async function auditLiteralArtifact(
  artifact: string,
  expected?: { constants: number; embeddedPayloadBytes: number; forbiddenSourcePath?: string },
): Promise<LiteralArtifactAudit> {
  const info = await stat(artifact);
  const hash = createHash("sha256");
  let constants = 0;
  let embeddedPayloadBytes = 0;
  let payloadBase64Characters = 0;
  let forbiddenSourcePathPresent = false;
  let outside = "";
  let inPayload = false;
  let currentCharacters = 0;
  let currentTail = "";
  const forbidden = expected?.forbiddenSourcePath;

  const finishPayload = (): void => {
    if (currentCharacters === 0 || currentCharacters % 4 !== 0) throw new Error(`Payload literal ${constants} possui base64 inválido por comprimento.`);
    const padding = currentTail.endsWith("==") ? 2 : currentTail.endsWith("=") ? 1 : 0;
    embeddedPayloadBytes += currentCharacters / 4 * 3 - padding;
    payloadBase64Characters += currentCharacters;
    currentCharacters = 0;
    currentTail = "";
  };

  for await (const chunk of createReadStream(artifact, { highWaterMark: 8 * 1024 * 1024 })) {
    const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    hash.update(bytes);
    let text = outside + bytes.toString("latin1");
    if (forbidden && text.includes(forbidden)) forbiddenSourcePathPresent = true;
    outside = "";
    while (text.length > 0) {
      if (inPayload) {
        const end = text.indexOf('"');
        const portion = end === -1 ? text : text.slice(0, end);
        if (!/^[A-Za-z0-9+/=]*$/.test(portion)) throw new Error(`Payload literal ${constants} contém caractere base64 inválido.`);
        currentCharacters += portion.length;
        currentTail = (currentTail + portion).slice(-2);
        if (end === -1) break;
        finishPayload();
        inPayload = false;
        text = text.slice(end + 1);
        continue;
      }
      const marker = PAYLOAD_MARKER.exec(text);
      if (!marker || marker.index === undefined) {
        outside = text.slice(-Math.max(PAYLOAD_MARKER_CARRY, (forbidden?.length ?? 0) - 1));
        break;
      }
      constants += 1;
      text = text.slice(marker.index + marker[0].length);
      inPayload = true;
    }
  }
  if (inPayload) throw new Error("Artifact literal terminou dentro de um payload base64.");
  const result: LiteralArtifactAudit = {
    artifact,
    artifactBytes: info.size,
    artifactSha256: hash.digest("hex"),
    constants,
    embeddedPayloadBytes,
    payloadBase64Characters,
    forbiddenSourcePathPresent,
  };
  if (expected && (result.constants !== expected.constants || result.embeddedPayloadBytes !== expected.embeddedPayloadBytes || result.forbiddenSourcePathPresent)) {
    throw new Error(`Auditoria literal divergente: constants=${result.constants}/${expected.constants}, payloadBytes=${result.embeddedPayloadBytes}/${expected.embeddedPayloadBytes}, sourcePath=${result.forbiddenSourcePathPresent}.`);
  }
  return result;
}

/** Binds a streamed artifact to the exact catalog payload count/byte total. */
export async function auditLiteralArtifactAgainstCatalog(artifact: string, source: string): Promise<LiteralArtifactAudit> {
  const opened = await openCatalog(source, false);
  try {
    const payloadBytes = [...opened.catalog.tensors.values()].reduce((total, tensor) => total + (tensor.byteLength ?? 0), 0);
    if (!Number.isSafeInteger(payloadBytes) || payloadBytes <= 0 || [...opened.catalog.tensors.values()].some((tensor) => tensor.byteLength === undefined)) {
      throw new Error("Catálogo não declara ranges completos para auditoria literal.");
    }
    return auditLiteralArtifact(artifact, {
      constants: opened.catalog.tensors.size,
      embeddedPayloadBytes: payloadBytes,
      forbiddenSourcePath: opened.catalog.source,
    });
  } finally {
    await opened.close();
  }
}
