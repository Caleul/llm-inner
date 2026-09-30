import { execFile } from "node:child_process";
import { createWriteStream } from "node:fs";
import { once } from "node:events";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { finished } from "node:stream/promises";
import { promisify } from "node:util";
import { SafetensorsCatalogReader } from "./safetensors.js";
import { readDirectF16Literal } from "./direct-weight-literal.js";
import type { TensorInfo } from "./types.js";
import { compileF16SiluBranches } from "./fixed-f16-conditional-linear.js";

export interface Projection { weight: string; shape: [number, number] }
export interface DiscoveredMlp {
  layer: string;
  mlp: { gate: Projection; up: Projection; down: Projection };
  normalizations: { weight: string; epsilon: number }[];
}

const runFile = promisify(execFile);

export async function discoverDirectMlp(directory: string, python: string, layerIndex: number): Promise<DiscoveredMlp> {
  if (!Number.isSafeInteger(layerIndex) || layerIndex < 0) throw new RangeError("Índice de camada inválido");
  const helper = new URL("../../helpers/discover_direct_mlp.py", import.meta.url).pathname;
  const { stdout } = await runFile(python, [helper, directory, String(layerIndex)], { maxBuffer: 1024 * 1024 });
  return JSON.parse(stdout) as DiscoveredMlp;
}

/** Emit one MLP output coordinate, with every reached weight decoded to a literal. */
export async function writeDirectMlpOutput(
  directory: string, python: string, layerIndex: number, dimension: number, outputPath: string,
): Promise<DiscoveredMlp> {
  const discovered = await discoverDirectMlp(directory, python, layerIndex);
  const reader = new SafetensorsCatalogReader(directory);
  await mkdir(dirname(outputPath), { recursive: true });
  const output = createWriteStream(outputPath, { encoding: "utf8" });
  const write = async (source: string): Promise<void> => {
    if (!output.write(source)) await once(output, "drain");
  };
  try {
    const catalog = await reader.inspect();
    const read = async (projection: Projection): Promise<TensorInfo> => {
      const tensor = catalog.tensors.get(projection.weight);
      if (!tensor || tensor.storageDtype !== "F16" || tensor.logicalShape.length !== 2 ||
        tensor.logicalShape[0] !== projection.shape[0] || tensor.logicalShape[1] !== projection.shape[1]) {
        throw new Error(`Projeção ${projection.weight} incompatível com a fonte`);
      }
      return tensor;
    };
    const gate = await read(discovered.mlp.gate);
    const up = await read(discovered.mlp.up);
    const down = await read(discovered.mlp.down);
    const hiddenSize = discovered.mlp.down.shape[0];
    const intermediateSize = discovered.mlp.down.shape[1];
    if (!Number.isSafeInteger(dimension) || dimension < 0 || dimension >= hiddenSize) {
      throw new RangeError("Dimensão MLP inválida");
    }
    const literal = (tensor: TensorInfo, width: number, row: number, column: number): Promise<string> =>
      readDirectF16Literal(reader, tensor, row * width + column);
    const emitLinear = async (
      bytes: TensorInfo, width: number, row: number, emitInput: (column: number) => Promise<void>,
    ): Promise<void> => {
      await write("f16Bits(Math.fround(Math.fround(");
      for (let lane = 0; lane < 4; lane++) {
        if (lane === 2) await write(")+Math.fround(");
        else if (lane > 0) await write("+");
        const indices = Array.from({ length: width }, (_, column) => column).filter((column) => column % 4 === lane);
        for (const _index of indices) await write("Math.fround(");
        await write("0");
        for (const column of indices) {
          await write("+");
          await emitInput(column);
          await write(`*${await literal(bytes, width, row, column)})`);
        }
      }
      await write(")))");
    };
    const silu = compileF16SiluBranches().source.slice("function(bits){".length, -1);
    const emitHidden = async (column: number): Promise<void> => {
      await write("f16Bits(f16((()=>{const bits=");
      await emitLinear(gate, hiddenSize, column, async (input) => write(`f16(mlpInput[t][${input}])`));
      await write(`;${silu}})())*f16(`);
      await emitLinear(up, hiddenSize, column, async (input) => write(`f16(mlpInput[t][${input}])`));
      await write("))");
    };
    await emitLinear(down, intermediateSize, dimension, async (column) => {
      await write("f16(");
      await emitHidden(column);
      await write(")");
    });
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
