import { createHash } from "node:crypto";
import { openCatalog } from "./catalog.js";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import type { TensorInfo } from "./types.js";

const DEFAULT_CHUNK_BYTES = 12 * 1024 * 1024;

export interface Gemma4CompositeLiteralPayloadVerification {
  artifact: string;
  source: string;
  constants: number;
  comparedPayloadBytes: number;
  sourceStorageSha256: string;
  literalStorageSha256: string;
}

/**
 * Establishes the export boundary before a checkpoint is removed: every
 * original Safetensors storage byte must equal the matching embedded literal
 * payload byte. The comparison is bounded and name-addressed, so equal totals
 * or a different constant ordering can never masquerade as a lossless export.
 */
export async function verifyGemma4CompositeLiteralPayloadsAgainstCatalog(options: {
  artifact: string;
  source: string;
  maxReadBytes?: number;
}): Promise<Gemma4CompositeLiteralPayloadVerification> {
  const maxReadBytes = options.maxReadBytes ?? DEFAULT_CHUNK_BYTES;
  if (!Number.isSafeInteger(maxReadBytes) || maxReadBytes <= 0) throw new Error("Verificação de payload Gemma 4 requer maxReadBytes inteiro positivo.");

  const source = await openCatalog(options.source, false);
  try {
    const artifact = await openGemma4CompositeLiteralArtifact(options.artifact);
    try {
      if (source.catalog.format !== "safetensors") throw new Error(`Verificação de payload Gemma 4 requer Safetensors denso; recebeu ${source.catalog.format}.`);
      const reader = source.reader as typeof source.reader & { readTensorBytesRange?: (tensor: TensorInfo, offset: number, byteLength: number) => Promise<Buffer> };
      if (typeof reader.readTensorBytesRange !== "function") throw new Error("Leitor Safetensors não oferece readTensorBytesRange para verificação de payload.");
      if (artifact.constants.size !== source.catalog.tensors.size) throw new Error(`Verificação de payload Gemma 4 encontrou ${artifact.constants.size} constantes literais para ${source.catalog.tensors.size} tensores fonte.`);

      const sourceHash = createHash("sha256");
      const literalHash = createHash("sha256");
      let comparedPayloadBytes = 0;
      for (const name of [...source.catalog.tensors.keys()].sort()) {
        const sourceTensor = source.catalog.tensors.get(name)!;
        const literal = artifact.constants.get(name);
        if (!literal || sourceTensor.quantization || sourceTensor.byteLength === undefined || sourceTensor.storageDtype !== literal.storageDtype ||
          !sameShape(sourceTensor.storageShape, literal.storageShape) || !sameShape(sourceTensor.logicalShape, literal.logicalShape) || sourceTensor.byteLength !== literal.payloadBytes) {
          throw new Error(`${name}: constante literal não corresponde ao tensor Safetensors declarado.`);
        }
        for (let offset = 0; offset < sourceTensor.byteLength; offset += maxReadBytes) {
          const byteLength = Math.min(maxReadBytes, sourceTensor.byteLength - offset);
          const [sourceBytes, literalBytes] = await Promise.all([
            reader.readTensorBytesRange(sourceTensor, offset, byteLength),
            artifact.readTensorBytesRange(sourceTensor, offset, byteLength),
          ]);
          if (!sourceBytes.equals(literalBytes)) throw new Error(`${name}: payload literal diverge do Safetensors no offset ${offset} (${byteLength} bytes comparados).`);
          sourceHash.update(sourceBytes);
          literalHash.update(literalBytes);
          comparedPayloadBytes += byteLength;
        }
      }
      return {
        artifact: artifact.artifact,
        source: options.source,
        constants: artifact.constants.size,
        comparedPayloadBytes,
        sourceStorageSha256: sourceHash.digest("hex"),
        literalStorageSha256: literalHash.digest("hex"),
      };
    } finally {
      await artifact.close();
    }
  } finally {
    await source.close();
  }
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
