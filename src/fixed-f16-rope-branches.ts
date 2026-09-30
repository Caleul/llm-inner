import { f32BitsToDyadic, roundDyadicToF16IfElse } from "./fixed-f16-projection.js";

/** Match the declared JS/F32/F16 RoPE coefficient policy at compile time. */
export function fixedF16RopeLiteral(
  position: number, dimension: number, headDim: number, theta: number, sine: number,
): number {
  const frequency = 1 / theta ** (2 * (dimension % (headDim / 2)) / headDim);
  const angle = Math.fround(position * Math.fround(frequency));
  const value = sine ? Math.sin(angle) : Math.cos(angle);
  const buffer = new DataView(new ArrayBuffer(4));
  buffer.setFloat32(0, value, true);
  return roundDyadicToF16IfElse(f32BitsToDyadic(buffer.getUint32(0, true)));
}

/** Emit conditionals whose leaves are literal RoPE coefficients for every valid position. */
export function compileFixedF16RopeBranches(
  maxSequenceLength: number, headDim: number, theta: number, sine: number,
): string {
  if (!Number.isSafeInteger(maxSequenceLength) || maxSequenceLength <= 0 ||
    !Number.isSafeInteger(headDim) || headDim <= 0 || headDim % 2 !== 0 ||
    !Number.isFinite(theta) || theta <= 0 || (sine !== 0 && sine !== 1)) {
    throw new Error("Parâmetros RoPE inválidos.");
  }
  const emit = (values: number[], low: number, high: number): string => {
    if (low === high) return `return ${values[low]};`;
    const middle = Math.floor((low + high) / 2);
    return `if(pos<=${middle}){${emit(values, low, middle)}}else{${emit(values, middle + 1, high)}}`;
  };
  let body = `if(pos<0||pos>=${maxSequenceLength})throw new RangeError("Posição RoPE fora do contexto");`;
  for (let dimension = 0; dimension < headDim; dimension++) {
    const values = Array.from({ length: maxSequenceLength }, (_, position) =>
      fixedF16RopeLiteral(position, dimension, headDim, theta, sine));
    body += `if(dim===${dimension}){${emit(values, 0, maxSequenceLength - 1)}}`;
  }
  return body + `throw new RangeError("Dimensão RoPE inválida");`;
}

/** Constant dimension removes the entire dimension dispatch during backward substitution. */
export function compileFixedF16RopePositionBranches(
  maxSequenceLength: number, headDim: number, theta: number, sine: number, dimension: number,
): string {
  if (!Number.isSafeInteger(maxSequenceLength) || maxSequenceLength <= 0 ||
    !Number.isSafeInteger(headDim) || headDim <= 0 || headDim % 2 !== 0 ||
    !Number.isInteger(dimension) || dimension < 0 || dimension >= headDim ||
    !Number.isFinite(theta) || theta <= 0 || (sine !== 0 && sine !== 1)) {
    throw new Error("Parâmetros RoPE inválidos.");
  }
  const values = Array.from({ length: maxSequenceLength }, (_, position) =>
    fixedF16RopeLiteral(position, dimension, headDim, theta, sine));
  const emit = (low: number, high: number): string => {
    if (low === high) return `return ${values[low]};`;
    const middle = Math.floor((low + high) / 2);
    return `if(pos<=${middle}){${emit(low, middle)}}else{${emit(middle + 1, high)}}`;
  };
  return `if(pos<0||pos>=${maxSequenceLength})throw new RangeError("Posição RoPE inválida");${emit(0, maxSequenceLength - 1)}`;
}
