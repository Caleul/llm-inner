import { createReadStream, createWriteStream } from "node:fs";
import { once } from "node:events";
import { finished } from "node:stream/promises";
import { compileF16SiluBranches } from "./fixed-f16-conditional-linear.js";

const hiddenCall = /\bhidden_(\d+)\(t,(\d+)\)/g;
const carryLength = 256;
let cachedSiluBody: string | undefined;

/** Substitute one generated MLP activation, preserving its F16 boundaries. */
export function inlineFixedF16HiddenCall(layerId: string, dimension: string): string {
  const body = cachedSiluBody ??= (() => {
    const silu = compileF16SiluBranches().source;
    return silu.slice("function(bits){".length, -1);
  })();
  return `(()=>{const g=gate_${layerId}(t,${dimension}),u=up_${layerId}(t,${dimension});` +
    `const activated=(()=>{const bits=g;${body}})();` +
    `return f16Bits(f16(activated)*f16(u));})()`;
}

/** Materialize the requested output expression without holding its expansion in a JS string. */
export async function writeFixedF16HiddenExpandedFormula(formula: string, path: string): Promise<{
  replacements: number; bytes: number;
}> {
  return rewriteFixedF16HiddenChunks((async function* () { yield formula; })(), path);
}

/** File-to-file pass: the expanded expression never becomes a JS string. */
export async function rewriteFixedF16HiddenFile(inputPath: string, outputPath: string, chunkSize = 64 * 1024): Promise<{
  replacements: number; bytes: number;
}> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const input = createReadStream(inputPath, { encoding: "utf8", highWaterMark: chunkSize });
  return rewriteFixedF16HiddenChunks(input, outputPath);
}

async function rewriteFixedF16HiddenChunks(chunks: AsyncIterable<string>, path: string): Promise<{
  replacements: number; bytes: number;
}> {
  const output = createWriteStream(path, { encoding: "utf8" });
  const write = async (chunk: string): Promise<void> => {
    if (!output.write(chunk)) await once(output, "drain");
  };
  let pending = "", replacements = 0, bytes = 0;
  const emit = async (source: string, final: boolean): Promise<void> => {
    const safeEnd = final ? source.length : Math.max(0, source.length - carryLength);
    let cursor = 0, writableEnd = safeEnd;
    hiddenCall.lastIndex = 0;
    for (const match of source.matchAll(hiddenCall)) {
      if (match.index >= safeEnd) break;
      if (!final && match.index + match[0].length > safeEnd) {
        writableEnd = match.index;
        break;
      }
      const prefix = source.slice(cursor, match.index);
      const replacement = inlineFixedF16HiddenCall(match[1]!, match[2]!);
      await write(prefix);
      await write(replacement);
      bytes += Buffer.byteLength(prefix) + Buffer.byteLength(replacement);
      replacements++;
      cursor = match.index + match[0].length;
    }
    const suffix = source.slice(cursor, writableEnd);
    await write(suffix);
    bytes += Buffer.byteLength(suffix);
    pending = source.slice(writableEnd);
  };
  try {
    for await (const chunk of chunks) {
      await emit(pending + chunk, false);
    }
    await emit(pending, true);
    output.end();
    await finished(output);
    return { replacements, bytes };
  } catch (error) {
    output.destroy();
    throw error;
  }
}
