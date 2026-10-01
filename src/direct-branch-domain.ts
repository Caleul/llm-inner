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
  constructor(private readonly intervals: ReadonlyMap<string, Interval> = new Map(),
    private readonly disjoint: ReadonlyMap<string, readonly Interval[]> = new Map()) {}
  /** Hulls support conservative range arithmetic; regions retain excluded
   * gaps as compile-time conditions, never as a runtime calculation graph. */
  entries(): IterableIterator<[string, Interval]> { return this.intervals.entries(); }
  regions(variable:string):readonly Interval[]{return this.disjoint.get(variable)??[this.intervals.get(variable)??{}];}
  get hasDisjointIntervals():boolean{return this.disjoint.size!==0;}
  select(keep:(key:string)=>boolean):DirectBranchDomain{
    return new DirectBranchDomain(new Map([...this.intervals].filter(([key])=>keep(key))),
      new Map([...this.disjoint].filter(([key])=>keep(key))));
  }
  intersectRegions(variable:string,regions:readonly Interval[]):DirectBranchDomain|undefined{
    const intersections:Interval[]=[];
    for(const a of this.regions(variable))for(const b of regions){
      const intersection=intersectInterval(a,b);if(intersection)intersections.push(intersection);
    }
    const nextRegions=normalizeRegions(intersections);if(!nextRegions.length)return undefined;
    if(sameRegions(this.regions(variable),nextRegions))return this;
    const first=nextRegions[0]!,last=nextRegions.at(-1)!;
    const hull:Interval={...(first.lower?{lower:first.lower}:{}),...(last.upper?{upper:last.upper}:{})};
    const next=new Map(this.intervals);next.set(variable,hull);
    const disjoint=new Map(this.disjoint);
    if(nextRegions.length>1)disjoint.set(variable,nextRegions);else disjoint.delete(variable);
    return new DirectBranchDomain(next,disjoint);
  }
  split(variable: string, op: Comparison, value: Rational): {
    truth: DirectBranchDomain | undefined; falsity: DirectBranchDomain | undefined;
  } {
    const refine = (comparison: Comparison): DirectBranchDomain | undefined =>{
      if(this.disjoint.has(variable))return this.intersectRegions(variable,[boundInterval(comparison,value)]);
      const existing=this.intervals.get(variable)??{};
      const narrowed=intersectInterval(existing,boundInterval(comparison,value));if(!narrowed)return undefined;
      if(sameRegions([existing],[narrowed]))return this;
      const next=new Map(this.intervals);next.set(variable,narrowed);
      return new DirectBranchDomain(next,this.disjoint);
    };
    return { truth: refine(op), falsity: refine(inverse[op]) };
  }
}

function sameRegions(a:readonly Interval[],b:readonly Interval[]):boolean{
  return a.length===b.length&&a.every((x,i)=>(["lower","upper"] as const).every(side=>{
    const y=b[i]![side],bound=x[side];
    return !bound||!y?bound===y:bound.inclusive===y.inclusive&&compare(bound.value,y.value)===0;
  }));
}
/** Merge overlaps, but preserve a missing midpoint when BOTH sides exclude
 * equality. Endpoints are exact rationals; no approximate adjacency test. */
function normalizeRegions(regions:readonly Interval[]):Interval[]{
  const sorted=regions.filter(x=>intersectInterval(x,{})!==undefined).sort((a,b)=>{
    if(!a.lower)return b.lower?-1:0;if(!b.lower)return 1;
    return compare(a.lower.value,b.lower.value)||(a.lower.inclusive===b.lower.inclusive?0:a.lower.inclusive?-1:1);
  });
  const result:Interval[]=[];
  for(const region of sorted){
    const previous=result.at(-1);
    if(!previous){result.push(region);continue;}
    const separation=previous.upper&&region.lower?compare(previous.upper.value,region.lower.value):-1;
    if(previous.upper&&region.lower&&(separation<0||separation===0&&!previous.upper.inclusive&&!region.lower.inclusive)){
      result.push(region);continue;
    }
    let upper:Bound|undefined;
    if(previous.upper&&region.upper){
      const c=compare(previous.upper.value,region.upper.value);
      upper=c>0?previous.upper:c<0?region.upper:{value:previous.upper.value,inclusive:previous.upper.inclusive||region.upper.inclusive};
    }
    result[result.length-1]={...(previous.lower?{lower:previous.lower}:{}),...(upper?{upper}:{})};
  }
  return result;
}
