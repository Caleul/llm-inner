/** Compile-time path facts only: no stored calculation tree or runtime representation. */
export interface Rational { numerator: bigint; denominator: bigint }
export interface Bound { value: Rational; inclusive: boolean }
export interface Interval { lower?: Bound; upper?: Bound }
export type Comparison = "<" | "<=" | ">" | ">=";
const compare = (a: Rational, b: Rational): number => {
  const delta = a.numerator * b.denominator - b.numerator * a.denominator;
  return delta < 0n ? -1 : delta > 0n ? 1 : 0;
};
export function rational(numerator: bigint, denominator = 1n): Rational {
  if (denominator === 0n) throw new RangeError("Zero denominator");
  if(denominator<0n){numerator=-numerator;denominator=-denominator;}
  let a=numerator<0n?-numerator:numerator,b=denominator;
  while(b!==0n){const remainder=a%b;a=b;b=remainder;}
  return {numerator:numerator/a,denominator:denominator/a};
}
const subtract = (a: Rational, b: Rational): Rational => rational(
  a.numerator * b.denominator - b.numerator * a.denominator, a.denominator * b.denominator);
const divide = (a: Rational, b: Rational): Rational => rational(a.numerator * b.denominator, a.denominator * b.numerator);
export function intersectInterval(a: Interval, b: Interval): Interval | undefined {
  const result: Interval = {};
  for (const side of ["lower", "upper"] as const) {
    const x = a[side], y = b[side];
    if (!x && !y) continue;
    if (!x || !y) { result[side] = (x ?? y)!; continue; }
    const c = compare(x.value, y.value);
    result[side] = c === 0 ? { value: x.value, inclusive: x.inclusive && y.inclusive } :
      (side === "lower" ? c > 0 : c < 0) ? x : y;
  }
  if (result.lower && result.upper) {
    const c = compare(result.lower.value, result.upper.value);
    if (c > 0 || (c === 0 && !(result.lower.inclusive && result.upper.inclusive))) return undefined;
  }
  return result;
}
function boundInterval(op: Comparison, value: Rational): Interval {
  return op.startsWith("<") ? { upper: { value, inclusive: op === "<=" } } :
    { lower: { value, inclusive: op === ">=" } };
}
const inverse: Record<Comparison, Comparison> = { "<": ">=", "<=": ">", ">": "<=", ">=": "<" };
const reverse: Record<Comparison, Comparison> = { "<": ">", "<=": ">=", ">": "<", ">=": "<=" };

/** Normalize a*x+b against c only when the caller has proved the transform exact.
 * Rounded transforms must first supply their actual numeric range; treating them
 * as real affine arithmetic would invalidate IEEE midpoint comparisons.
 */
export function normalizeExactAffineComparison(scale: Rational, offset: Rational,
  op: Comparison, rhs: Rational): { op: Comparison; value: Rational } | boolean {
  if (scale.numerator === 0n) {
    const c = compare(offset, rhs);
    return op === "<" ? c < 0 : op === "<=" ? c <= 0 : op === ">" ? c > 0 : c >= 0;
  }
  return { op: scale.numerator < 0n ? reverse[op] : op, value: divide(subtract(rhs, offset), scale) };
}
export class DirectBranchDomain {
  constructor(private readonly intervals: ReadonlyMap<string, Interval> = new Map()) {}
  entries(): IterableIterator<[string, Interval]> { return this.intervals.entries(); }
  split(variable: string, op: Comparison, value: Rational): {
    truth: DirectBranchDomain | undefined; falsity: DirectBranchDomain | undefined;
  } {
    const existing = this.intervals.get(variable) ?? {};
    const refine = (comparison: Comparison): DirectBranchDomain | undefined => {
      const narrowed = intersectInterval(existing, boundInterval(comparison, value));
      if (!narrowed) return undefined;
      const next = new Map(this.intervals); next.set(variable, narrowed);
      return new DirectBranchDomain(next);
    };
    return { truth: refine(op), falsity: refine(inverse[op]) };
  }
}
