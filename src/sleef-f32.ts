/**
 * Scalar transcription of SLEEF `Sleef_tanhf4_u10advsimd` as shipped by the
 * pinned PyTorch wheel. Every helper below mirrors one binary32 ADVSIMD or FMA
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

/** Exact ADVSIMD u10 sine fast path used by Gemma RoPE angles below 125. */
export function sleefSinF32(value: number): number {
  const input = f32(value);
  assertFastTrigRange(input);
  const quadrantFloat = f32(roundTiesToEven(multiply(input, f32(0.31830988618379067154))));
  const quadrant = Math.trunc(quadrantFloat);
  const first = fma(quadrantFloat, f32(-3.1414794921875), input);
  let reduced = addFloatFloat2(first, multiply(quadrantFloat, f32(-0.00011315941810607910156)));
  reduced = addPairFloat(reduced, multiply(quadrantFloat, f32(-1.9841872589410058936e-9)));
  let output = sinReducedF32(reduced);
  if ((quadrant & 1) === 1) output = -output;
  return Object.is(input, -0) ? -0 : output;
}

/** Exact ADVSIMD u10 cosine fast path used by Gemma RoPE angles below 125. */
export function sleefCosF32(value: number): number {
  const input = f32(value);
  assertFastTrigRange(input);
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

function assertFastTrigRange(value: number): void {
  if (!Number.isFinite(value) || Math.abs(value) >= 125) {
    throw new Error(`SLEEF ADVSIMD trig argument ${value} requires the unimplemented rempif range reducer.`);
  }
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
