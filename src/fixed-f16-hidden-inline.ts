import { createWriteStream } from "node:fs";
import { once } from "node:events";
import { finished } from "node:stream/promises";
import { compileF16SiluBranches } from "./fixed-f16-conditional-linear.js";

const hiddenCall = /\bhidden_(\d+)\(t,(\d+)\)/g;
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
  const output = createWriteStream(path, { encoding: "utf8" });
  const write = async (chunk: string): Promise<void> => {
    if (!output.write(chunk)) await once(output, "drain");
  };
  let cursor = 0, replacements = 0, bytes = 0;
  try {
    for (const match of formula.matchAll(hiddenCall)) {
      const prefix = formula.slice(cursor, match.index);
      const replacement = inlineFixedF16HiddenCall(match[1]!, match[2]!);
      await write(prefix);
      await write(replacement);
      bytes += Buffer.byteLength(prefix) + Buffer.byteLength(replacement);
      replacements++;
      cursor = match.index + match[0].length;
    }
    const suffix = formula.slice(cursor);
    await write(suffix);
    bytes += Buffer.byteLength(suffix);
    output.end();
    await finished(output);
    return { replacements, bytes };
  } catch (error) {
    output.destroy();
    throw error;
  }
}
