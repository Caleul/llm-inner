import { createWriteStream } from "node:fs";
import { execFile } from "node:child_process";
import { once } from "node:events";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { finished } from "node:stream/promises";
import { promisify } from "node:util";
import { SafetensorsCatalogReader } from "./safetensors.js";
import { f16BitsToDyadic } from "./fixed-f16-projection.js";

const runFile = promisify(execFile);

export interface DirectOutputDiscovery {
  architecture: string;
  forwardSourceSha256: string;
  weight: string;
  safetensors: string;
  shape: [number, number];
  dtype: string;
  finalNormWeight: string;
  finalNormEpsilon: number;
  finalNormSourceSha256: string;
  decoderLayers: string[];
  transformers: string;
  torch: string;
}

/** Inspect the installed forward source and tensor metadata, without model IR. */
export async function discoverDirectOutput(
  directory: string, python: string,
): Promise<DirectOutputDiscovery> {
  const helper = new URL("../../helpers/discover_direct_output.py", import.meta.url).pathname;
  const { stdout } = await runFile(python, [helper, directory], { maxBuffer: 1024 * 1024 });
  const discovered = JSON.parse(stdout) as DirectOutputDiscovery;
  if (!Array.isArray(discovered.shape) || discovered.shape.length !== 2 || discovered.dtype !== "F16" ||
    !Number.isSafeInteger(discovered.shape[0]) || !Number.isSafeInteger(discovered.shape[1]) ||
    discovered.shape[0] <= 0 || discovered.shape[1] <= 0 || !discovered.weight) {
    throw new Error("Projeção de logits descoberta não suportada");
  }
  return discovered;
}

/** Emit the chosen logit row with literal weights and ordered four-lane sums. */
export async function writeDirectOutputRow(
  directory: string, python: string, dimension: number, outputPath: string,
): Promise<DirectOutputDiscovery> {
  const discovered = await discoverDirectOutput(directory, python);
  if (!Number.isSafeInteger(dimension) || dimension < 0 || dimension >= discovered.shape[0]) {
    throw new RangeError("Dimensão final fora do vocabulário");
  }
  const reader = new SafetensorsCatalogReader(directory);
  await mkdir(dirname(outputPath), { recursive: true });
  const output = createWriteStream(outputPath, { encoding: "utf8" });
  const write = async (text: string): Promise<void> => {
    if (!output.write(text)) await once(output, "drain");
  };
  try {
    const catalog = await reader.inspect();
    const tensor = catalog.tensors.get(discovered.weight);
    if (!tensor || tensor.storageDtype !== "F16" || tensor.logicalShape.length !== 2 ||
      tensor.logicalShape[0] !== discovered.shape[0] || tensor.logicalShape[1] !== discovered.shape[1]) {
      throw new Error("Peso de saída incompatível com a fonte do forward");
    }
    const bytes = await reader.readTensorBytes(tensor);
    const width = discovered.shape[1];
    await write("f16Bits(Math.fround(Math.fround(");
    for (let lane = 0; lane < 4; lane++) {
      if (lane === 2) await write(")+Math.fround(");
      else if (lane > 0) await write("+");
      const indices = Array.from({ length: width }, (_, input) => input).filter((input) => input % 4 === lane);
      for (const _index of indices) await write("Math.fround(");
      await write("0");
      for (const input of indices) {
        const bits = bytes.readUInt16LE((dimension * width + input) * 2);
        const dyadic = f16BitsToDyadic(bits);
        const weight = Number(dyadic.coefficient) * 2 ** dyadic.exponent;
        await write(`+f16(hidden[t][${input}])*${Object.is(weight, -0) ? "-0" : weight})`);
      }
    }
    await write(")))");
    output.end();
    await finished(output);
    return discovered;
  } catch (error) {
    output.destroy();
    throw error;
  } finally {
    await reader.close();
  }
}
