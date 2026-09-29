import { addDyadic, f16BitsToDyadic, f32BitsToDyadic, roundDyadicToF16IfElse, roundDyadicToF32IfElse, type Dyadic } from "./fixed-f16-projection.js";

interface Rational { numerator: bigint; denominator: bigint }
function gcd(a: bigint, b: bigint): bigint { while (b !== 0n) [a, b] = [b, a % b]; return a < 0n ? -a : a; }
function rational(numerator: bigint, denominator: bigint): Rational {
  if (denominator <= 0n) throw new Error("Denominador inválido.");
  const divisor = gcd(numerator, denominator);
  return { numerator: numerator / divisor, denominator: denominator / divisor };
}
function toRational(value: Dyadic): Rational {
  return value.exponent >= 0 ? rational(value.coefficient << BigInt(value.exponent), 1n) : rational(value.coefficient, 1n << BigInt(-value.exponent));
}
function add(a: Rational, b: Rational): Rational { return rational(a.numerator * b.denominator + b.numerator * a.denominator, a.denominator * b.denominator); }
function multiply(a: Rational, b: Rational): Rational { return rational(a.numerator * b.numerator, a.denominator * b.denominator); }
function compare(a: Rational, b: Rational): number {
  const difference = a.numerator * b.denominator - b.numerator * a.denominator;
  return difference < 0n ? -1 : difference > 0n ? 1 : 0;
}

/** Enclosed rational interval around exp(x), for -128 < x <= 0.
 * Taylor is evaluated within [-1,0], then outward-rounded fixed-point
 * intervals are squared. The fixed denominator prevents explosive growth. */
function expNegativeInterval(value: Dyadic): [Rational, Rational] {
  const original = toRational(value);
  const x = rational(original.numerator, original.denominator * 128n);
  if (compare(x, rational(-1n, 1n)) < 0 || x.numerator > 0n) throw new Error("Argumento exp fora do intervalo certificado [-1,0].");
  let sum = rational(1n, 1n), term = sum;
  let lower = sum, upper = sum;
  for (let index = 1n; index <= 40n; index++) {
    term = multiply(term, rational(x.numerator, x.denominator * index));
    sum = add(sum, term);
    if (index === 39n) lower = sum;
    if (index === 40n) upper = sum;
  }
  const scale = 1n << 256n;
  let low = lower.numerator * scale / lower.denominator;
  let high = (upper.numerator * scale + upper.denominator - 1n) / upper.denominator;
  for (let i = 0; i < 7; i++) {
    low = low * low / scale;
    high = (high * high + scale - 1n) / scale;
  }
  return [{ numerator: low, denominator: scale }, { numerator: high, denominator: scale }];
}

function roundRationalToF32IfElse(value: Rational): number {
  if (value.numerator < 0n || value.numerator > value.denominator) throw new Error("Racional de softmax fora de [0,1].");
  let low = 0, high = 0x3f800000;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (compare(toRational(f32BitsToDyadic(middle)), value) <= 0) low = middle;
    else high = middle - 1;
  }
  if (low === 0x3f800000) return low;
  const lower = toRational(f32BitsToDyadic(low)), upper = toRational(f32BitsToDyadic(low + 1));
  const midpoint = multiply(add(lower, upper), rational(1n, 2n));
  const side = compare(value, midpoint);
  return side < 0 || side === 0 && (low & 1) === 0 ? low : low + 1;
}

function roundedExpF32(value: Dyadic): number {
  const comparison = addDyadic(value, { coefficient: 128n, exponent: 0 });
  if (comparison.coefficient <= 0n) return 0; // exp(-128) < half the least F32 subnormal
  const [lower, upper] = expNegativeInterval(value);
  const lowBits = roundRationalToF32IfElse(lower), highBits = roundRationalToF32IfElse(upper);
  if (lowBits !== highBits) throw new Error("Exp requer mais termos para certificar o arredondamento F32.");
  return lowBits;
}

/** Two-logit softmax with explicit F32 boundaries and certified exp rounding. */
export function softmaxTwoF16IfElse(leftBits: number, rightBits: number): [number, number] {
  const left = f16BitsToDyadic(leftBits), right = f16BitsToDyadic(rightBits);
  const max = toRational(left).numerator * toRational(right).denominator >= toRational(right).numerator * toRational(left).denominator ? left : right;
  const differences = [left, right].map((value) => f32BitsToDyadic(roundDyadicToF32IfElse(addDyadic(value, { coefficient: -max.coefficient, exponent: max.exponent }))));
  const exponentials = differences.map((value) => f32BitsToDyadic(roundedExpF32(value)));
  const sum = f32BitsToDyadic(roundDyadicToF32IfElse(addDyadic(exponentials[0]!, exponentials[1]!)));
  return exponentials.map((value) => {
    const quotient = rational(toRational(value).numerator * toRational(sum).denominator,
      toRational(value).denominator * toRational(sum).numerator);
    return roundDyadicToF16IfElse(f32BitsToDyadic(roundRationalToF32IfElse(quotient)));
  }) as [number, number];
}
