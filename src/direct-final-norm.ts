import { mkdtemp, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { rewriteTextualCalls } from "./direct-numeric-stream.js";
import { discoverDirectOutput, writeDirectOutputRow, type DirectOutputDiscovery } from "./direct-output-row.js";
import { SafetensorsCatalogReader } from "./safetensors.js";
import { f16BitsToDyadic } from "./fixed-f16-projection.js";

/** Substitute the final normalization in the selected literal logit row. */
export async function writeDirectOutputThroughFinalNorm(
  directory: string, python: string, dimension: number, outputPath: string,
): Promise<DirectOutputDiscovery> {
  const discovered = await discoverDirectOutput(directory, python);
  if (!discovered.finalNormWeight || !Number.isFinite(discovered.finalNormEpsilon) ||
    discovered.finalNormEpsilon < 0) throw new Error("Normalização final não descoberta");
  const reader = new SafetensorsCatalogReader(directory);
  const temporary = await mkdtemp(join(dirname(outputPath), ".direct-final-norm-"));
  try {
    const tensor = (await reader.inspect()).tensors.get(discovered.finalNormWeight);
    const width = discovered.shape[1];
    if (!tensor || tensor.storageDtype !== "F16" || tensor.logicalShape.length !== 1 ||
      tensor.logicalShape[0] !== width) throw new Error("Peso da normalização final inválido");
    const weights = await reader.readTensorBytes(tensor);
    let sum = "0";
    for (let coordinate = 0; coordinate < width; coordinate++) {
      const input = `f16(layerOutput[t][${coordinate}])`;
      sum = `Math.fround(${sum}+${input}*${input})`;
    }
    const epsilon = Math.fround(discovered.finalNormEpsilon);
    const factor = `Math.fround(1/Math.sqrt(Math.fround(Math.fround(${sum}/${width})+${epsilon})))`;
    const rowPath = join(temporary, "row");
    await writeDirectOutputRow(directory, python, dimension, rowPath);
    const pattern = /\bhidden\[t\]\[(\d+)\]/g;
    const result = await rewriteTextualCalls(rowPath, outputPath, pattern, (match) => {
      const coordinate = Number(match[1]);
      if (!Number.isSafeInteger(coordinate) || coordinate < 0 || coordinate >= width) {
        throw new RangeError("Dimensão da normalização fora do domínio");
      }
      const bits = weights.readUInt16LE(coordinate * 2);
      const dyadic = f16BitsToDyadic(bits);
      const weight = Number(dyadic.coefficient) * 2 ** dyadic.exponent;
      const normalized = `f16Bits(Math.fround(f16(layerOutput[t][${coordinate}])*rmsFactor))`;
      return `f16Bits(f16(${normalized})*${Object.is(weight, -0) ? "-0" : weight})`;
    }, 64 * 1024, `(()=>{const rmsFactor=${factor};return `, "})()");
    if (result.replacements !== width) {
      throw new Error(`Esperadas ${width} substituições da normalização; houve ${result.replacements}`);
    }
    return discovered;
  } finally {
    await reader.close();
    await rm(temporary, { recursive: true, force: true });
  }
}
