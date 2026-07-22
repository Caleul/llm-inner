import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { once } from "node:events";
import { readFile, stat } from "node:fs/promises";

export interface Gemma4FinalFormulaMapResult { functions: number; orderedRootsSha256: string }
export interface Gemma4FinalFormulaMapValidation extends Gemma4FinalFormulaMapResult { fileSha256: string }

/** Writes the requested calc_final_n -> F_n(x) object without loading the large SSA in memory. */
export async function writeGemma4FinalFormulaMap(globalSsa: string, output: string, firstOutput: number, functions: number): Promise<Gemma4FinalFormulaMapResult> {
  if (!Number.isSafeInteger(firstOutput) || firstOutput < 0 || !Number.isSafeInteger(functions) || functions < 1) throw new Error("Faixa de fórmulas finais inválida.");
  const input = createReadStream(globalSsa), target = createWriteStream(output, { flags: "wx" });
  const marker = Buffer.from('"outputs":['); let carry: Buffer<ArrayBufferLike> = Buffer.alloc(0), found = false, outputsEnded = false, depth = 0, inString = false, escaped = false, ordinal = -1, written = 0;
  let objectBytes: number[] = []; const rootsDigest = createHash("sha256");
  try {
    await write(target, "{");
    for await (const chunk_ of input) {
      const chunk = chunk_ as Buffer; let data = chunk;
      if (!found) {
        const searchable = carry.length ? Buffer.concat([carry, chunk]) : chunk, index = searchable.indexOf(marker);
        if (index < 0) { carry = searchable.subarray(Math.max(0, searchable.length - marker.length + 1)); continue; }
        found = true; data = searchable.subarray(index + marker.length);
      }
      for (const byte of data) {
        if (!inString && depth === 0) {
          if (byte === 0x5d) { outputsEnded = true; break; }
          if (byte !== 0x7b) continue;
          ordinal += 1; depth = 1; objectBytes = [byte]; continue;
        }
        objectBytes.push(byte);
        if (inString) {
          if (escaped) escaped = false;
          else if (byte === 0x5c) escaped = true;
          else if (byte === 0x22) inString = false;
          continue;
        }
        if (byte === 0x22) inString = true;
        else if (byte === 0x7b) depth += 1;
        else if (byte === 0x7d) {
          depth -= 1;
          if (depth === 0 && ordinal >= firstOutput && ordinal < firstOutput + functions) {
            const dimension = ordinal - firstOutput, binding = parseBinding(Buffer.from(objectBytes).toString("utf8"), dimension);
            const expression = `BF16_RNE(EVAL_EXACT_DAG(${JSON.stringify(binding.value)}, x))`;
            await write(target, `${written === 0 ? "" : ","}${JSON.stringify(`calc_final_${dimension}`)}:${JSON.stringify(expression)}`);
            rootsDigest.update(`${dimension}:${binding.value}\n`, "utf8");
            written += 1;
          }
        }
      }
      if (written === functions || outputsEnded) break;
    }
    if (!found) throw new Error("SSA global não contém outputs.");
    if (written !== functions) throw new Error(`SSA global forneceu ${written}/${functions} fórmulas finais.`);
    await write(target, "}\n"); target.end(); await once(target, "close");
    return { functions: written, orderedRootsSha256: rootsDigest.digest("hex") };
  } catch (error) {
    input.destroy(); target.end(); await Promise.allSettled([once(target, "close")]); throw error;
  }
}

/** Authenticates the public calc_final_n map and every ordered DAG root before execution. */
export async function validateGemma4FinalFormulaMapFile(path: string, expected: { fileSha256: string; functions: number; orderedRootsSha256: string }): Promise<Gemma4FinalFormulaMapValidation> {
  if (!/^[0-9a-f]{64}$/.test(expected.fileSha256) || !/^[0-9a-f]{64}$/.test(expected.orderedRootsSha256) || !Number.isSafeInteger(expected.functions) || expected.functions < 1) throw new Error("Contrato esperado do mapa final é inválido.");
  const info = await stat(path);
  if (!info.isFile() || info.size < 3 || info.size > 256 * 1024 * 1024) throw new Error("Mapa final deve ser arquivo JSON entre 3 bytes e 256 MiB.");
  const bytes = await readFile(path), fileSha256 = createHash("sha256").update(bytes).digest("hex");
  if (fileSha256 !== expected.fileSha256) throw new Error("Mapa final diverge do SHA-256 autenticado.");
  const parsed = JSON.parse(bytes.toString("utf8")) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Mapa final deve ser objeto JSON.");
  const entries = Object.entries(parsed as Record<string, unknown>);
  if (entries.length !== expected.functions) throw new Error(`Mapa final possui ${entries.length}/${expected.functions} funções.`);
  const rootsDigest = createHash("sha256");
  for (let dimension = 0; dimension < entries.length; dimension += 1) {
    const [key, expression] = entries[dimension]!, expectedKey = `calc_final_${dimension}`;
    if (key !== expectedKey || typeof expression !== "string") throw new Error(`Mapa final diverge em ${expectedKey}.`);
    const match = /^BF16_RNE\(EVAL_EXACT_DAG\("(sha256:[0-9a-f]{64})", x\)\)$/.exec(expression);
    if (!match) throw new Error(`${expectedKey} não possui a forma executável autenticada.`);
    rootsDigest.update(`${dimension}:${match[1]}\n`, "utf8");
  }
  const orderedRootsSha256 = rootsDigest.digest("hex");
  if (orderedRootsSha256 !== expected.orderedRootsSha256) throw new Error("Raízes ordenadas do mapa final divergem do compromisso autenticado.");
  return { functions: entries.length, fileSha256, orderedRootsSha256 };
}

function parseBinding(json: string, dimension: number): { value: string } {
  const value = JSON.parse(json) as { assignment?: unknown; value?: unknown; parameters?: unknown; finalQuantization?: unknown };
  if (value.assignment !== `calc_terminal_logit_${dimension}` || typeof value.value !== "string" || value.value.length === 0 || !Array.isArray(value.parameters) || value.finalQuantization !== "BF16-round-to-nearest-ties-to-even") throw new Error(`Binding terminal_logit[${dimension}] inválido.`);
  return { value: value.value };
}

async function write(stream: ReturnType<typeof createWriteStream>, value: string): Promise<void> {
  if (!stream.write(value, "utf8")) await once(stream, "drain");
}
