import { SafetensorsCatalogReader } from "./safetensors.js";

/** A finite binary number, kept exact until the declared rounding boundary. */
export interface Dyadic { coefficient: bigint; exponent: number }
export interface FixedF16ProjectionRow {
  output: number;
  terms: Array<{ input: number; weightBits: number }>;
}
export interface FixedF16Projection {
  kind: "fixed-f16-projection";
  tensor: string;
  inputSize: number;
  outputSize: number;
  rows: FixedF16ProjectionRow[];
  arithmetic: "f32-ascending-products-and-sum";
  rounding: "binary16-nearest-ties-to-even-conditional";
}

export async function compileFixedF16Projection(
  reader: SafetensorsCatalogReader, tensorName: string,
): Promise<FixedF16Projection> {
  const catalog = await reader.inspect();
  const tensor = catalog.tensors.get(tensorName);
  if (!tensor || tensor.storageDtype !== "F16" || tensor.quantization || tensor.logicalShape.length !== 2) {
    throw new Error(`${tensorName}: projeção requer matriz Safetensors F16 densa.`);
  }
  const [outputSize, inputSize] = tensor.logicalShape;
  if (!outputSize || !inputSize) throw new Error(`${tensorName}: shape inválido.`);
  const bytes = await reader.readTensorBytes(tensor);
  const rows = Array.from({ length: outputSize }, (_, output) => ({
    output,
    terms: Array.from({ length: inputSize }, (_, input) => ({ input, weightBits: bytes.readUInt16LE(2 * (output * inputSize + input)) }))
      .filter(({ weightBits }) => (weightBits & 0x7fff) !== 0),
  }));
  return { kind: "fixed-f16-projection", tensor: tensorName, inputSize, outputSize, rows,
    arithmetic: "f32-ascending-products-and-sum", rounding: "binary16-nearest-ties-to-even-conditional" };
}

export function f16BitsToDyadic(bits: number): Dyadic {
  if (!Number.isInteger(bits) || bits < 0 || bits > 0xffff) throw new Error("Bits F16 inválidos.");
  const fraction = bits & 0x3ff, exponent = (bits >>> 10) & 31;
  if (exponent === 31) throw new Error("F16 não finito fora do contrato inicial.");
  const sign = (bits & 0x8000) ? -1n : 1n;
  return { coefficient: sign * BigInt(exponent === 0 ? fraction : 1024 + fraction), exponent: exponent === 0 ? -24 : exponent - 25 };
}

export function addDyadic(a: Dyadic, b: Dyadic): Dyadic {
  const exponent = Math.min(a.exponent, b.exponent);
  return { coefficient: (a.coefficient << BigInt(a.exponent - exponent)) + (b.coefficient << BigInt(b.exponent - exponent)), exponent };
}
export function multiplyDyadic(a: Dyadic, b: Dyadic): Dyadic {
  return { coefficient: a.coefficient * b.coefficient, exponent: a.exponent + b.exponent };
}
function compareDyadic(a: Dyadic, b: Dyadic): number {
  const difference = addDyadic(a, { coefficient: -b.coefficient, exponent: b.exponent });
  return difference.coefficient < 0n ? -1 : difference.coefficient > 0n ? 1 : 0;
}

export function f32BitsToDyadic(bits: number): Dyadic {
  const fraction = bits & 0x7fffff, exponent = (bits >>> 23) & 0xff;
  if (exponent === 0xff) throw new Error("F32 não finito fora do contrato inicial.");
  const sign = (bits & 0x80000000) ? -1n : 1n;
  return { coefficient: sign * BigInt(exponent === 0 ? fraction : 0x800000 + fraction), exponent: exponent === 0 ? -149 : exponent - 150 };
}

export function roundDyadicToF32IfElse(value: Dyadic): number {
  if (value.coefficient === 0n) return 0;
  const negative = value.coefficient < 0n;
  const magnitude = negative ? { ...value, coefficient: -value.coefficient } : value;
  if (compareDyadic(magnitude, { coefficient: 0x2000000n - 1n, exponent: 103 }) >= 0) return negative ? 0xff800000 : 0x7f800000;
  let low = 0, high = 0x7f7fffff;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (compareDyadic(f32BitsToDyadic(middle), magnitude) <= 0) low = middle;
    else high = middle - 1;
  }
  if (low === 0x7f7fffff) return (low | (negative ? 0x80000000 : 0)) >>> 0;
  const midpoint = multiplyDyadic(addDyadic(f32BitsToDyadic(low), f32BitsToDyadic(low + 1)), { coefficient: 1n, exponent: -1 });
  const comparison = compareDyadic(magnitude, midpoint);
  const chosen = comparison < 0 || (comparison === 0 && (low & 1) === 0) ? low : low + 1;
  return (chosen | (negative ? 0x80000000 : 0)) >>> 0;
}

/** Binary search is a compact if/else tree over adjacent F16 values and their midpoints. */
export function roundDyadicToF16IfElse(value: Dyadic): number {
  if (value.coefficient === 0n) return 0;
  const negative = value.coefficient < 0n;
  const magnitude = negative ? { ...value, coefficient: -value.coefficient } : value;
  const overflowMidpoint: Dyadic = { coefficient: 65520n, exponent: 0 };
  if (compareDyadic(magnitude, overflowMidpoint) >= 0) return negative ? 0xfc00 : 0x7c00;
  let low = 0, high = 0x7bff;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (compareDyadic(f16BitsToDyadic(middle), magnitude) <= 0) low = middle;
    else high = middle - 1;
  }
  if (low === 0x7bff) return low | (negative ? 0x8000 : 0);
  const lower = f16BitsToDyadic(low), upper = f16BitsToDyadic(low + 1);
  const midpoint = multiplyDyadic(addDyadic(lower, upper), { coefficient: 1n, exponent: -1 });
  const comparison = compareDyadic(magnitude, midpoint);
  const chosen = comparison < 0 || (comparison === 0 && (low & 1) === 0) ? low : low + 1;
  return chosen | (negative ? 0x8000 : 0);
}

export function evaluateFixedF16Projection(program: FixedF16Projection, inputBits: readonly number[]): number[] {
  if (inputBits.length !== program.inputSize) throw new Error("Tamanho da entrada da projeção inválido.");
  const input = inputBits.map(f16BitsToDyadic);
  return program.rows.map((row) => {
    let accumulator: Dyadic = { coefficient: 0n, exponent: 0 };
    for (const term of row.terms) {
      const product = multiplyDyadic(input[term.input]!, f16BitsToDyadic(term.weightBits));
      const productBits = roundDyadicToF32IfElse(product);
      if ((productBits & 0x7f800000) === 0x7f800000) throw new Error("Produto F32 não finito fora do contrato inicial.");
      const sumBits = roundDyadicToF32IfElse(addDyadic(accumulator, f32BitsToDyadic(productBits)));
      if ((sumBits & 0x7f800000) === 0x7f800000) throw new Error("Acumulador F32 não finito fora do contrato inicial.");
      accumulator = f32BitsToDyadic(sumBits);
    }
    return roundDyadicToF16IfElse(accumulator);
  });
}
