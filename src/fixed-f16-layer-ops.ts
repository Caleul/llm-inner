import { type SafetensorsCatalogReader } from "./safetensors.js";
import { addDyadic, compileFixedF16Projection, f16BitsToDyadic, f32BitsToDyadic, multiplyDyadic, roundDyadicToF16IfElse, type FixedF16Projection } from "./fixed-f16-projection.js";

const bits = new DataView(new ArrayBuffer(4));
export function f16Number(value: number): number {
  const dyadic = f16BitsToDyadic(value);
  return Number(dyadic.coefficient) * 2 ** dyadic.exponent;
}
export function f32Number(value: number): number {
  bits.setUint32(0, value, true);
  return bits.getFloat32(0, true);
}
export function f32Bits(value: number): number {
  bits.setFloat32(0, value, true);
  return bits.getUint32(0, true);
}
export function f16Bits(value: number): number {
  return roundDyadicToF16IfElse(f32BitsToDyadic(f32Bits(value)));
}

export async function readFixedF16Vector(reader: SafetensorsCatalogReader, name: string, size: number): Promise<number[]> {
  const tensor = (await reader.inspect()).tensors.get(name);
  if (!tensor || tensor.storageDtype !== "F16" || tensor.logicalShape.length !== 1 || tensor.logicalShape[0] !== size) {
    throw new Error(`${name}: vetor F16 incompatível.`);
  }
  const bytes = await reader.readTensorBytes(tensor);
  return Array.from({ length: size }, (_, index) => bytes.readUInt16LE(index * 2));
}

/** LlamaRMSNorm CPU candidate, preserving its F32 and F16 cast boundaries. */
export function rmsNormF16(input: readonly number[], weight: readonly number[], epsilon = 1e-6): number[] {
  if (input.length !== weight.length || input.length === 0) throw new Error("RMSNorm: shape inválido.");
  let variance = Math.fround(0);
  for (const value of input) {
    const x = f16Number(value);
    variance = Math.fround(variance + Math.fround(x * x));
  }
  variance = Math.fround(variance / input.length);
  const inverse = Math.fround(1 / Math.sqrt(Math.fround(variance + Math.fround(epsilon))));
  return input.map((value, index) => {
    const normalized = f16Bits(Math.fround(f16Number(value) * inverse));
    return f16Bits(f16Number(normalized) * f16Number(weight[index]!));
  });
}

export function addF16Bits(left: number, right: number): number {
  return roundDyadicToF16IfElse(addDyadic(f16BitsToDyadic(left), f16BitsToDyadic(right)));
}
export function multiplyF16Bits(left: number, right: number): number {
  return roundDyadicToF16IfElse(multiplyDyadic(f16BitsToDyadic(left), f16BitsToDyadic(right)));
}
export interface FixedMlpProgram { gate: FixedF16Projection; up: FixedF16Projection; down: FixedF16Projection }
export async function compileFixedMlp(reader: SafetensorsCatalogReader, layer: number): Promise<FixedMlpProgram> {
  const base = `model.layers.${layer}.mlp.`;
  return { gate: await compileFixedF16Projection(reader, `${base}gate_proj.weight`),
    up: await compileFixedF16Projection(reader, `${base}up_proj.weight`),
    down: await compileFixedF16Projection(reader, `${base}down_proj.weight`) };
}
export function evaluateFixedMlp(program: FixedMlpProgram, input: readonly number[]): number[] {
  const gate = evaluateFixedFourLaneProjection(program.gate, input);
  const up = evaluateFixedFourLaneProjection(program.up, input);
  const hidden = gate.map((value, index) => {
    const x = f16Number(value);
    const silu = f16Bits(x / (1 + Math.exp(-x)));
    return multiplyF16Bits(silu, up[index]!);
  });
  return evaluateFixedFourLaneProjection(program.down, hidden);
}

/** CPU lm_head uses four interleaved F32 accumulators and a pairwise fold. */
export function evaluateFixedFourLaneProjection(program: FixedF16Projection, input: readonly number[]): number[] {
  if (input.length !== program.inputSize || program.inputSize % 4 !== 0) throw new Error("Projeção de quatro lanes requer tamanho múltiplo de quatro.");
  const inputs = input.map(f16Number);
  return program.rows.map((row) => {
    const lanes = [0, 0, 0, 0];
    for (const term of row.terms) {
      const lane = term.input & 3;
      lanes[lane] = Math.fround(lanes[lane]! + Math.fround(inputs[term.input]! * f16Number(term.weightBits)));
    }
    return f16Bits(Math.fround(Math.fround(lanes[0]! + lanes[1]!) + Math.fround(lanes[2]! + lanes[3]!)));
  });
}

/** Dense packed form of the same four-lane head, without one object per weight. */
export function evaluateFixedFourLaneDenseHead(weights: Buffer, rows: number, input: readonly number[]): number[] {
  if (input.length !== 16 || weights.length !== rows * 32) throw new Error("lm_head denso F16 incompatível.");
  const inputs = input.map(f16Number);
  const result = Array<number>(rows);
  for (let row = 0; row < rows; row++) {
    const lanes = [0, 0, 0, 0];
    for (let index = 0; index < 16; index++) {
      const lane = index & 3;
      lanes[lane] = Math.fround(lanes[lane]! + Math.fround(inputs[index]! * f16Number(weights.readUInt16LE(row * 32 + index * 2))));
    }
    result[row] = f16Bits(Math.fround(Math.fround(lanes[0]! + lanes[1]!) + Math.fround(lanes[2]! + lanes[3]!)));
  }
  return result;
}
