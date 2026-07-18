/**
 * Scalar transcriptions of the SLEEF ADVSIMD kernels selected by the pinned
 * PyTorch wheel. Every helper below mirrors one binary32 ADVSIMD or FMA
 * boundary; the pair representation mirrors SLEEF's `vfloat2` double-float.
 *
 * Authoritative sources:
 * - PyTorch 7269437d655783a26cba32aa88195b741ff496aa
 * - bundled SLEEF 5a1d179df9cf652951b59010a2d2075372d67f68
 */

interface FloatPair { x: number; y: number }

const f32 = Math.fround;
const add = (left: number, right: number): number => f32(left + right);
const subtract = (left: number, right: number): number => f32(left - right);
const multiply = (left: number, right: number): number => f32(left * right);
/** Exact F32 operands remain exact in binary64 until this single F32 boundary. */
const fma = (left: number, right: number, addend: number): number => f32(left * right + addend);
const pair = (x: number, y = 0): FloatPair => ({ x: f32(x), y: f32(y) });

/**
 * First 416 binary32 entries of SLEEF's `Sleef_rempitabsp`.  Entry 408 is
 * selected by the largest finite F32 exponent and entries 412..415 preserve
 * the source's Inf/NaN lane behavior.  The literal artifact carries the same
 * little-endian payload and its SHA-256, so large-angle replay does not depend
 * on an external SLEEF binary or source checkout.
 */
export const SLEEF_REMPITABSP_F32_LE_BASE64 = "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";
export const SLEEF_REMPITABSP_F32_LE_SHA256 = "9a623b9ff705f726ddb129e4b1c3c0311ac19b86a30667cdd395e6ea98d8c5c5";
const REMPI_TABLE = decodeF32LeTable(SLEEF_REMPITABSP_F32_LE_BASE64, 416);

export function sleefTanhF32(value: number): number {
  const input = f32(value);
  if (Number.isNaN(input)) return Number.NaN;
  const magnitude = Math.abs(input);
  if (magnitude > f32(8.664339742)) return input < 0 || Object.is(input, -0) ? -1 : 1;
  const exponential = expPair(pair(magnitude));
  const reciprocal = reciprocalPair(exponential);
  const quotient = dividePair(subtractPair(exponential, reciprocal), addPair(exponential, reciprocal));
  let output = add(quotient.x, quotient.y);
  if (Number.isNaN(output)) output = 1;
  return input < 0 || Object.is(input, -0) ? -output : output;
}

/** Exact ADVSIMD u10 sine over the complete finite binary32 argument range. */
export function sleefSinF32(value: number): number {
  const input = f32(value);
  if (!Number.isFinite(input)) return Number.NaN;
  if (Math.abs(input) >= 125) return sleefSinLargeF32(input);
  const quadrantFloat = f32(roundTiesToEven(multiply(input, f32(0.31830988618379067154))));
  const quadrant = Math.trunc(quadrantFloat);
  const first = fma(quadrantFloat, f32(-3.1414794921875), input);
  let reduced = addFloatFloat2(first, multiply(quadrantFloat, f32(-0.00011315941810607910156)));
  reduced = addPairFloat(reduced, multiply(quadrantFloat, f32(-1.9841872589410058936e-9)));
  let output = sinReducedF32(reduced);
  if ((quadrant & 1) === 1) output = -output;
  return Object.is(input, -0) ? -0 : output;
}

/** Exact ADVSIMD u10 cosine over the complete finite binary32 argument range. */
export function sleefCosF32(value: number): number {
  const input = f32(value);
  if (!Number.isFinite(input)) return Number.NaN;
  if (Math.abs(input) >= 125) return sleefCosLargeF32(input);
  const rounded = f32(roundTiesToEven(fma(input, f32(0.31830988618379067154), -0.5)));
  const quadrantFloat = fma(rounded, 2, 1);
  const quadrant = Math.trunc(quadrantFloat);
  let reduced = addFloatFloat2(input, multiply(quadrantFloat, f32(-3.1414794921875 * 0.5)));
  reduced = addPairFloat2(reduced, multiply(quadrantFloat, f32(-0.00011315941810607910156 * 0.5)));
  reduced = addPairFloat2(reduced, multiply(quadrantFloat, f32(-1.9841872589410058936e-9 * 0.5)));
  let output = sinReducedF32(reduced);
  if ((quadrant & 2) === 0) output = -output;
  return output;
}

/** Exact ADVSIMD u10 exponential selected by PyTorch's F32 softmax kernel. */
export function sleefExpF32(value: number): number {
  const input = f32(value);
  if (Number.isNaN(input)) return Number.NaN;
  if (input < -104) return 0;
  if (input > 100) return Number.POSITIVE_INFINITY;
  const exponent = roundTiesToEven(multiply(input, f32(1.4426950408889634)));
  let reduced = fma(f32(exponent), f32(-0.693145751953125), input);
  reduced = fma(f32(exponent), f32(-1.428606765330187e-6), reduced);
  let polynomial = f32(0.000198527617612853646278381);
  polynomial = fma(polynomial, reduced, f32(0.00139304355252534151077271));
  polynomial = fma(polynomial, reduced, f32(0.00833336077630519866943359));
  polynomial = fma(polynomial, reduced, f32(0.0416664853692054748535156));
  polynomial = fma(polynomial, reduced, f32(0.166666671633720397949219));
  polynomial = fma(polynomial, reduced, f32(0.5));
  const result = add(1, fma(multiply(reduced, reduced), polynomial, reduced));
  return scalePowerOfTwo(result, exponent);
}

function sleefSinLargeF32(input: number): number {
  const reduction = rempiF32(input);
  let quadrant = reduction.quadrant & 3;
  quadrant = ((quadrant + quadrant) + (reduction.reduced.x > 0 ? 2 : 1)) >> 2;
  if ((reduction.quadrant & 1) === 1) {
    const correction = pair(
      xorSign(f32(-1.5707963705062866), reduction.reduced.x),
      xorSign(f32(4.371138828673793e-8), reduction.reduced.x),
    );
    reduction.reduced = addPair2(reduction.reduced, correction);
  }
  const output = sinReducedF32(normalizePair(reduction.reduced));
  return (quadrant & 1) === 1 ? xorSign(output, -1) : output;
}

function sleefCosLargeF32(input: number): number {
  const reduction = rempiF32(input);
  let quadrant = reduction.quadrant & 3;
  quadrant = ((quadrant + quadrant) + (reduction.reduced.x > 0 ? 8 : 7)) >> 1;
  if ((reduction.quadrant & 1) === 0) {
    const sign = reduction.reduced.x > 0 ? 0 : -1;
    reduction.reduced = addPair2(reduction.reduced, pair(
      xorSign(f32(-1.5707963705062866), sign),
      xorSign(f32(4.371138828673793e-8), sign),
    ));
  }
  const output = sinReducedF32(normalizePair(reduction.reduced));
  return (quadrant & 2) === 0 ? xorSign(output, -1) : output;
}

function rempiF32(input: number): { reduced: FloatPair; quadrant: number } {
  const bits = f32Bits(input);
  let exponent = ((bits >>> 23) & 0xff) - 0x7f - 25;
  const scaleExponent = exponent > 65 ? -64 : 0;
  const scaledInput = f32(input * (2 ** scaleExponent));
  exponent = Math.max(0, exponent) * 4;
  if (exponent + 3 >= REMPI_TABLE.length) throw new Error(`SLEEF rempif table index ${exponent} is outside the declared F32 table.`);

  let reduced = multiplyFloatFloatPair(scaledInput, REMPI_TABLE[exponent]!);
  let sub = rempiSubF32(reduced.x);
  let quadrant = sub.quadrant;
  reduced = normalizePair(pair(sub.remainder, reduced.y));

  reduced = addPair2(reduced, multiplyFloatFloatPair(scaledInput, REMPI_TABLE[exponent + 1]!));
  sub = rempiSubF32(reduced.x);
  quadrant += sub.quadrant;
  reduced = normalizePair(pair(sub.remainder, reduced.y));

  reduced = addPair2(reduced, multiplyPairFloat(pair(REMPI_TABLE[exponent + 2]!, REMPI_TABLE[exponent + 3]!), scaledInput));
  reduced = normalizePair(reduced);
  reduced = multiplyPair(reduced, pair(f32(6.2831854820251465), f32(-1.7484555314695172e-7)));
  return { reduced, quadrant };
}

function rempiSubF32(input: number): { remainder: number; quadrant: number } {
  const roundedFour = f32(roundTiesToEven(multiply(input, 4)));
  const roundedOne = f32(roundTiesToEven(input));
  return {
    remainder: subtract(input, multiply(roundedFour, f32(0.25))),
    quadrant: Math.trunc(subtract(roundedFour, multiply(roundedOne, 4))),
  };
}

function multiplyFloatFloatPair(left: number, right: number): FloatPair {
  const product = multiply(left, right);
  return pair(product, fma(left, right, -product));
}

function normalizePair(value: FloatPair): FloatPair {
  const sum = add(value.x, value.y);
  return pair(sum, add(subtract(value.x, sum), value.y));
}

function xorSign(value: number, signSource: number): number {
  return signSource < 0 || Object.is(signSource, -0) ? -value : value;
}

function f32Bits(value: number): number {
  const bytes = new ArrayBuffer(4);
  const view = new DataView(bytes);
  view.setFloat32(0, value, true);
  return view.getUint32(0, true);
}

function decodeF32LeTable(encoded: string, entries: number): Float32Array {
  const bytes = Buffer.from(encoded, "base64");
  if (bytes.length !== entries * 4) throw new Error(`SLEEF rempif table has ${bytes.length} bytes; expected ${entries * 4}.`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Float32Array.from({ length: entries }, (_, index) => view.getFloat32(index * 4, true));
}

function sinReducedF32(reduced: FloatPair): number {
  const squared = squarePair(reduced);
  let polynomial = f32(2.6083159809786593541503e-6);
  polynomial = fma(polynomial, squared.x, f32(-0.0001981069071916863322258));
  polynomial = fma(polynomial, squared.x, f32(0.00833307858556509017944336));
  const coefficient = addFloatFloat(f32(-0.166666597127914428710938), multiply(polynomial, squared.x));
  const correction = multiplyPair(coefficient, squared);
  const expansion = addFloatPair(1, correction);
  return multiplyPairsToFloat(reduced, expansion);
}

function expPair(input: FloatPair): FloatPair {
  const inverseLn2 = f32(1.4426950408889634);
  const exponent = roundTiesToEven(multiply(add(input.x, input.y), inverseLn2));
  let reduced = addPairFloat2(input, multiply(f32(exponent), f32(-0.693145751953125)));
  reduced = addPairFloat2(reduced, multiply(f32(exponent), f32(-1.428606765330187e-6)));

  let polynomial = f32(0.1980960224e-3);
  polynomial = fma(polynomial, reduced.x, f32(0.1394256484e-2));
  polynomial = fma(polynomial, reduced.x, f32(0.8333456703e-2));
  polynomial = fma(polynomial, reduced.x, f32(0.4166637361e-1));

  let correction = addPairFloat2(multiplyPairFloat(reduced, polynomial), f32(0.16666665941423424));
  correction = addPairFloat2(multiplyPair(reduced, correction), f32(0.5));
  correction = addPair2(reduced, multiplyPair(squarePair(reduced), correction));
  correction = addFloatPair2(1, correction);
  correction.x = scalePowerOfTwo(correction.x, exponent);
  correction.y = scalePowerOfTwo(correction.y, exponent);
  return input.x < -104 ? pair(0) : correction;
}

function addPairFloat2(left: FloatPair, right: number): FloatPair {
  const sum = add(left.x, right);
  const virtual = subtract(sum, left.x);
  const error = add(subtract(left.x, subtract(sum, virtual)), subtract(right, virtual));
  return pair(sum, add(error, left.y));
}

function addPairFloat(left: FloatPair, right: number): FloatPair {
  const sum = add(left.x, right);
  return pair(sum, add(add(subtract(left.x, sum), right), left.y));
}

function addFloatFloat(left: number, right: number): FloatPair {
  const sum = add(left, right);
  return pair(sum, add(subtract(left, sum), right));
}

function addFloatFloat2(left: number, right: number): FloatPair {
  const sum = add(left, right);
  const virtual = subtract(sum, left);
  return pair(sum, add(subtract(left, subtract(sum, virtual)), subtract(right, virtual)));
}

function addFloatPair(left: number, right: FloatPair): FloatPair {
  const sum = add(left, right.x);
  return pair(sum, add(add(subtract(left, sum), right.x), right.y));
}

function addFloatPair2(left: number, right: FloatPair): FloatPair {
  const sum = add(left, right.x);
  const virtual = subtract(sum, left);
  return pair(sum, add(add(subtract(left, subtract(sum, virtual)), subtract(right.x, virtual)), right.y));
}

function addPair(left: FloatPair, right: FloatPair): FloatPair {
  const sum = add(left.x, right.x);
  return pair(sum, add(add(subtract(left.x, sum), right.x), add(left.y, right.y)));
}

function addPair2(left: FloatPair, right: FloatPair): FloatPair {
  const sum = add(left.x, right.x);
  const virtual = subtract(sum, left.x);
  const error = add(subtract(left.x, subtract(sum, virtual)), subtract(right.x, virtual));
  return pair(sum, add(error, add(left.y, right.y)));
}

function subtractPair(left: FloatPair, right: FloatPair): FloatPair {
  const difference = subtract(left.x, right.x);
  let error = subtract(left.x, difference);
  error = subtract(error, right.x);
  error = add(error, left.y);
  return pair(difference, subtract(error, right.y));
}

function multiplyPairFloat(left: FloatPair, right: number): FloatPair {
  const product = multiply(left.x, right);
  return pair(product, fma(left.y, right, fma(left.x, right, -product)));
}

function multiplyPair(left: FloatPair, right: FloatPair): FloatPair {
  const product = multiply(left.x, right.x);
  return pair(product, fma(left.x, right.y, fma(left.y, right.x, fma(left.x, right.x, -product))));
}

function multiplyPairsToFloat(left: FloatPair, right: FloatPair): number {
  return fma(left.x, right.x, fma(left.y, right.x, multiply(left.x, right.y)));
}

function squarePair(value: FloatPair): FloatPair {
  const square = multiply(value.x, value.x);
  return pair(square, fma(add(value.x, value.x), value.y, fma(value.x, value.x, -square)));
}

function reciprocalPair(value: FloatPair): FloatPair {
  const reciprocal = f32(1 / value.x);
  return pair(reciprocal, multiply(reciprocal, fma(-value.y, reciprocal, fma(-value.x, reciprocal, 1))));
}

function dividePair(numerator: FloatPair, denominator: FloatPair): FloatPair {
  const reciprocal = f32(1 / denominator.x);
  const quotient = multiply(numerator.x, reciprocal);
  const numeratorError = fma(reciprocal, numerator.x, -quotient);
  const denominatorError = fma(-denominator.y, reciprocal, fma(-denominator.x, reciprocal, 1));
  return pair(quotient, fma(quotient, denominatorError, fma(numerator.y, reciprocal, numeratorError)));
}

function scalePowerOfTwo(value: number, exponent: number): number {
  const half = exponent >> 1;
  return multiply(multiply(value, f32(2 ** half)), f32(2 ** (exponent - half)));
}

function roundTiesToEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}
