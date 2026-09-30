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

interface Projection { weight: string; shape: [number, number] }
export interface DiscoveredAttention {
  layer: string;
  attention: { projections: { q: Projection; k: Projection; v: Projection; o: Projection };
    heads: number; kvHeads: number; headDim: number; scaling: number; ropeTheta: number; maxPosition: number };
}
const runFile = promisify(execFile);

export async function discoverDirectAttention(
  directory: string, python: string, layerIndex: number,
): Promise<DiscoveredAttention> {
  const helper = new URL("../../helpers/discover_direct_attention.py", import.meta.url).pathname;
  const { stdout } = await runFile(python, [helper, directory, String(layerIndex)], { maxBuffer: 1024 * 1024 });
  return JSON.parse(stdout) as DiscoveredAttention;
}

/** Emit one attention output coordinate for runtime n and t, without activations or learned-weight access. */
export async function writeDirectAttentionOutput(
  directory: string, python: string, layerIndex: number, dimension: number, outputPath: string,
): Promise<DiscoveredAttention> {
  const discovered = await discoverDirectAttention(directory, python, layerIndex);
  const spec = discovered.attention;
  const width = spec.projections.o.shape[1];
  if (!Number.isInteger(dimension) || dimension < 0 || dimension >= spec.projections.o.shape[0] ||
    !Number.isInteger(spec.heads) || !Number.isInteger(spec.kvHeads) ||
    !Number.isInteger(spec.headDim) || spec.headDim <= 0 || spec.headDim % 2 !== 0 ||
    spec.heads % spec.kvHeads !== 0 || width !== spec.heads * spec.headDim) {
    throw new Error("Geometria da atenção não suportada");
  }
  const reader = new SafetensorsCatalogReader(directory);
  await mkdir(dirname(outputPath), { recursive: true });
  const output = createWriteStream(outputPath, { encoding: "utf8" });
  const write = async (source: string): Promise<void> => {
    if (!output.write(source)) await once(output, "drain");
  };
  try {
    const catalog = await reader.inspect();
    const weights: Record<"q" | "k" | "v" | "o", TensorInfo> = {} as never;
    for (const role of ["q", "k", "v", "o"] as const) {
      const projection = spec.projections[role];
      const tensor = catalog.tensors.get(projection.weight);
      if (!tensor || tensor.storageDtype !== "F16" || tensor.logicalShape.length !== 2 ||
        tensor.logicalShape[0] !== projection.shape[0] || tensor.logicalShape[1] !== projection.shape[1]) {
        throw new Error(`Peso da atenção inválido: ${projection.weight}`);
      }
      weights[role] = tensor;
    }
    const literal = (role: "q" | "k" | "v" | "o", row: number, column: number): Promise<string> =>
      readDirectF16Literal(reader, weights[role], row * spec.projections[role].shape[1] + column);
    const emitReduction = async (size: number, term: (coordinate: number) => Promise<void>): Promise<void> => {
      await write("f16Bits(Math.fround(Math.fround(");
      for (let lane = 0; lane < 4; lane++) {
        if (lane === 2) await write(")+Math.fround(");
        else if (lane > 0) await write("+");
        const coordinates = Array.from({ length: size }, (_, index) => index).filter((index) => index % 4 === lane);
        for (const _coordinate of coordinates) await write("Math.fround(");
        await write("0");
        for (const coordinate of coordinates) {
          await write("+");
          await term(coordinate);
          await write(")");
        }
      }
      await write(")))");
    };
    const emitProjection = async (role: "q" | "k" | "v" | "o", row: number, position: string,
      input?: (coordinate: number) => Promise<void>): Promise<void> => {
      const size = spec.projections[role].shape[1];
      await emitReduction(size, async (coordinate) => {
        if (input) await input(coordinate);
        else await write(`f16(attnInput[${position}][${coordinate}])`);
        await write(`*${await literal(role, row, coordinate)}`);
      });
    };
    const emitRotated = async (role: "q" | "k", head: number, coordinate: number, position: string): Promise<void> => {
      const half = spec.headDim / 2;
      const projectionHead = role === "q" ? head : Math.floor(head / (spec.heads / spec.kvHeads));
      const row = projectionHead * spec.headDim + coordinate;
      const partner = projectionHead * spec.headDim + ((coordinate + half) % spec.headDim);
      const cos = `ropeBits(${position},${coordinate},${spec.headDim},${spec.ropeTheta},0)`;
      const sin = `ropeBits(${position},${coordinate},${spec.headDim},${spec.ropeTheta},1)`;
      await write("f16Bits(f16(f16Bits(f16(");
      await emitProjection(role, row, position);
      await write(`)*f16(${cos})))+f16(f16Bits(`);
      if (coordinate < half) await write("-");
      await write("f16(");
      await emitProjection(role, partner, position);
      await write(`)*f16(${sin}))))`);
    };
    const emitScore = async (head: number): Promise<void> => {
      await write("f16Bits(f16(");
      await emitReduction(spec.headDim, async (coordinate) => {
        await write("f16(");
        await emitRotated("q", head, coordinate, "t");
        await write(")*f16(");
        await emitRotated("k", head, coordinate, "j");
        await write(")");
      });
      await write(`)*${spec.scaling})`);
    };
    const emitMaskedScore = async (head: number): Promise<void> => {
      await write("(()=>{const raw=");
      await emitScore(head);
      await write(";return j>t?f16Bits(f16(raw)-65504):raw})()");
    };
    const emitContext = async (head: number, coordinate: number): Promise<void> => {
      await write("(()=>{let maximum=-Infinity;for(let j=0;j<n;j++){const candidate=f16(");
      await emitMaskedScore(head);
      await write(");if(candidate>maximum)maximum=candidate;}let denominator=Math.fround(0);");
      await write("for(let j=0;j<n;j++){denominator=Math.fround(denominator+Math.fround(Math.exp(Math.fround(f16(");
      await emitMaskedScore(head);
      await write(")-maximum))));}let acc=Math.fround(0);for(let j=0;j<n;j++){");
      await write("const probability=f16Bits(Math.fround(Math.fround(Math.exp(Math.fround(f16(");
      await emitMaskedScore(head);
      await write(")-maximum)))/denominator));acc=Math.fround(acc+Math.fround(f16(probability)*f16(");
      const kvHead = Math.floor(head / (spec.heads / spec.kvHeads));
      await emitProjection("v", kvHead * spec.headDim + coordinate, "j");
      await write(")));}return f16Bits(acc);})()");
    };
    await emitProjection("o", dimension, "t", async (coordinate) => {
      await write("f16(");
      await emitContext(Math.floor(coordinate / spec.headDim), coordinate % spec.headDim);
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
