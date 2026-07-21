import { createReadStream } from "node:fs";

export interface Gemma4GlobalSsaOutput {
  assignment: string;
  value: string;
  parameters: string[];
  coordinate: string[];
  finalQuantization: "BF16-round-to-nearest-ties-to-even";
}

/**
 * Reads one output binding from the large, one-line SSA export without parsing
 * its hundreds of MiB of statements and functions into memory.
 */
export async function readGemma4GlobalSsaOutput(path: string, ordinal: number): Promise<Gemma4GlobalSsaOutput> {
  if (!Number.isSafeInteger(ordinal) || ordinal < 0) throw new Error("Ordinal de saída SSA deve ser inteiro não negativo.");
  const marker = Buffer.from('"outputs":['), stream = createReadStream(path);
  let carry: Buffer<ArrayBufferLike> = Buffer.alloc(0); let found = false, objectDepth = 0, inString = false, escaped = false, currentOrdinal = -1;
  const bytes: number[] = [];
  for await (const chunk_ of stream) {
    const chunk = chunk_ as Buffer; let data = chunk;
    if (!found) {
      const searchable = carry.length ? Buffer.concat([carry, chunk]) : chunk, markerIndex = searchable.indexOf(marker);
      if (markerIndex < 0) { carry = searchable.subarray(Math.max(0, searchable.length - marker.length + 1)); continue; }
      found = true; data = searchable.subarray(markerIndex + marker.length);
    }
    for (const byte of data) {
      if (!inString && objectDepth === 0) {
        if (byte === 0x5d) throw new Error(`SSA global não possui output no ordinal ${ordinal}.`);
        if (byte !== 0x7b) continue;
        currentOrdinal += 1; objectDepth = 1;
        if (currentOrdinal === ordinal) bytes.push(byte);
        continue;
      }
      if (currentOrdinal === ordinal) bytes.push(byte);
      if (inString) {
        if (escaped) escaped = false;
        else if (byte === 0x5c) escaped = true;
        else if (byte === 0x22) inString = false;
        continue;
      }
      if (byte === 0x22) inString = true;
      else if (byte === 0x7b) objectDepth += 1;
      else if (byte === 0x7d) {
        objectDepth -= 1;
        if (objectDepth === 0 && currentOrdinal === ordinal) return validateOutput(JSON.parse(Buffer.from(bytes).toString("utf8")) as unknown, ordinal);
      }
    }
  }
  if (!found) throw new Error("SSA global não contém o array outputs.");
  throw new Error(`SSA global terminou antes do output no ordinal ${ordinal}.`);
}

function validateOutput(value: unknown, ordinal: number): Gemma4GlobalSsaOutput {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`Output SSA ${ordinal} não é objeto.`);
  const output = value as Partial<Gemma4GlobalSsaOutput>;
  if (typeof output.assignment !== "string" || !/^calc_[a-z0-9_]+_\d+$/.test(output.assignment) || typeof output.value !== "string" || !/^sha256:[0-9a-f]{64}$/.test(output.value) ||
    !Array.isArray(output.parameters) || output.parameters.some((entry) => typeof entry !== "string") || !Array.isArray(output.coordinate) || output.coordinate.some((entry) => typeof entry !== "string" || !/^sha256:[0-9a-f]{64}$/.test(entry)) ||
    output.finalQuantization !== "BF16-round-to-nearest-ties-to-even") throw new Error(`Output SSA ${ordinal} possui contrato inválido.`);
  return output as Gemma4GlobalSsaOutput;
}
