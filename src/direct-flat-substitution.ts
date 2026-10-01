import { createHash } from "node:crypto";
import { DirectBranchDomain, intersectInterval, normalizeExactAffineComparison, rational, type Interval, type Rational } from "./direct-branch-domain.js";
import { exactNumberRational, finiteIeeeValue, normalizeFiniteArithmeticComparison, normalizeFiniteAffineRunComparison, normalizePositiveReciprocalComparison, normalizePositiveSqrtComparison, normalizeRoundedAffineComparison } from "./direct-round-preimage.js";
import { DirectRustStream, rustF64, type RustExpression } from "./direct-rust-stream.js";
import { emitRustExp, emitRustSilu, foldCertifiedF32Sqrt, type NumericAffineRun, type NumericRunConsumer } from "./direct-rust-numeric.js";
import { f32BitsToDyadic, roundDyadicToF16IfElse } from "./fixed-f16-projection.js";
import { decodeIeeeF16ToF32 } from "./utils.js";
import { DirectSourceBounds, type SourceBounds } from "./direct-source-bounds.js";

/** Only conditions on the currently visited path are retained. Producers call
 * their consumer immediately; no expression nodes, replay files or memoized
 * values are constructed. A leaf callback writes its substituted arithmetic.
 */
export interface FlatInput {
  emit: RustExpression;
  minimum: number;
  maximum: number;
  precision?: "f16" | "f32";
  literal?:number;
  /** Proven power-of-two lattice, not cached values. */
  quantum?:number;
  /** This source is a fundamental F16 input in every path, not merely a
   * locally rounded expression whose dtype depends on its exponent guard. */
  fundamentalF16?:true;
  positiveZero?:true;
  restrict?: (path:FlatConditions,interval:Interval)=>Promise<FlatConditions|undefined>;
}
export type FlatConsumer = (path: FlatConditions, input: FlatInput) => Promise<void>;
export type FlatProducer = (path: FlatConditions, consume: FlatConsumer) => Promise<void>;
export class FlatConditions {
  constructor(readonly domain = new DirectBranchDomain(),
    readonly guards: readonly { key: string; emit: RustExpression;fundamentalF16?:true }[] = [],
    readonly entryGuards:readonly RustExpression[] = [],readonly nonzero:ReadonlySet<string>=new Set(),
    readonly minimumMagnitude:ReadonlyMap<string,number>=new Map()) {}
  refine(key: string, emit: RustExpression, interval: Interval,fundamentalF16?:true): FlatConditions | undefined {
    let domain: DirectBranchDomain | undefined = this.domain;
    for (const side of ["lower", "upper"] as const) {
      const bound = interval[side];
      if (!bound || !domain) continue;
      domain = domain.split(key, side === "lower" ? (bound.inclusive ? ">=" : ">") :
        (bound.inclusive ? "<=" : "<"), bound.value).truth;
    }
    if (!domain) return undefined;
    const bounds=new Map(domain.entries()).get(key);
    if(this.nonzero.has(key)&&bounds?.lower?.value.numerator===0n&&bounds.upper?.value.numerator===0n)return undefined;
    return new FlatConditions(domain, this.guards.some(g => g.key === key) ?
      (fundamentalF16?this.guards.map(g=>g.key===key?{...g,fundamentalF16}:g):this.guards) :
      [...this.guards, {key, emit,...(fundamentalF16?{fundamentalF16}:{})}],this.entryGuards,this.nonzero,this.minimumMagnitude);
  }
  requireNonzero(key:string):FlatConditions|undefined{
    const interval=new Map(this.domain.entries()).get(key);
    if(interval?.lower?.value.numerator===0n&&interval.upper?.value.numerator===0n)return undefined;
    return this.nonzero.has(key)?this:new FlatConditions(this.domain,this.guards,this.entryGuards,new Set([...this.nonzero,key]),this.minimumMagnitude);
  }
  requireMagnitude(key:string,value:number):FlatConditions|undefined{
    const path=this.requireNonzero(key);if(!path)return undefined;
    const magnitude=Math.max(path.minimumMagnitude.get(key)??0,value);
    return new FlatConditions(path.domain,path.guards,path.entryGuards,path.nonzero,new Map([...path.minimumMagnitude,[key,magnitude]]));
  }
}

/** Substitution is continuation passing: the consumer runs inside each
 * surviving producer path, BEFORE any Rust branch is emitted. Consequently
 * outer conditions never contain an inline conditional producer.
 */
export class DirectFlatSubstitution {
  private pendingLeaf:{path:FlatConditions;label:string;input:FlatInput;key:string}|undefined;
  constructor(readonly stream: DirectRustStream) {}
  async literal(path:FlatConditions,value:number,consume:FlatConsumer):Promise<void>{
    if(!Number.isFinite(value))throw new Error("Nonfinite literal needs explicit numerical semantics");
    const exact=exactNumberRational(value);let numerator=exact.numerator<0n?-exact.numerator:exact.numerator;
    let exponent=-(exact.denominator.toString(2).length-1);
    if(numerator)while((numerator&1n)===0n){numerator>>=1n;exponent++;}
    const significantBits=numerator.toString(2).length;
    const precision:FlatInput["precision"]=value===0||Math.abs(value)<=65504&&exponent>=-24&&significantBits<=11?"f16":
      Math.abs(value)<=3.4028234663852886e38&&exponent>=-149&&significantBits<=24?"f32":undefined;
    await consume(path,{literal:value,minimum:value,maximum:value,...(precision?{precision}:{}),
      ...(!Object.is(value,-0)?{positiveZero:true as const}:{}),...(numerator?{quantum:2**exponent}:{}),
      emit:()=>this.stream.write(rustF64(Object.is(value,-0)?"-0":value))});
  }
  async input(path: FlatConditions, emit: RustExpression, minimum: number, maximum: number,
    consume: FlatConsumer): Promise<void> {
    if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum)
      throw new Error("Flat substitution requires a proved finite scalar domain");
    const interval=new Map(path.domain.entries()).get(await this.key(emit));
    if(interval?.lower)minimum=Math.max(minimum,Number(interval.lower.value.numerator)/Number(interval.lower.value.denominator));
    if(interval?.upper)maximum=Math.min(maximum,Number(interval.upper.value.numerator)/Number(interval.upper.value.denominator));
    if(minimum>maximum)return;
    await consume(path, {emit, minimum, maximum});
  }
  async f16Input(path:FlatConditions,emit:RustExpression,consume:FlatConsumer):Promise<void>{
    const interval=new Map(path.domain.entries()).get(await this.key(emit));
    const satisfies=(index:number,side:"lower"|"upper")=>{
      const bound=interval?.[side];if(!bound)return true;
      const c=compareRational(exactNumberRational(finiteIeeeValue("f16",index)),bound.value);
      return side==="lower"?(c>0||(c===0&&bound.inclusive)):(c<0||(c===0&&bound.inclusive));
    };
    let lower=0,upper=63486;
    while(lower<upper){const mid=Math.floor((lower+upper)/2);if(satisfies(mid,"lower"))upper=mid;else lower=mid+1;}
    const first=lower;lower=0;upper=63486;
    while(lower<upper){const mid=Math.ceil((lower+upper)/2);if(satisfies(mid,"upper"))lower=mid;else upper=mid-1;}
    const last=lower;
    if(first>last||!satisfies(first,"lower")||!satisfies(last,"upper")){this.stream.eliminatedBranches++;return;}
    const minimum=finiteIeeeValue("f16",first),maximum=finiteIeeeValue("f16",last);
    if(minimum===0&&maximum===0&&path.nonzero.has(await this.key(emit))){this.stream.eliminatedBranches++;return;}
    if(minimum===maximum&&minimum!==0){
      this.stream.eliminatedBranches++;await this.literal(path,minimum,(p,v)=>consume(p,{...v,precision:"f16",quantum:2**-24}));return;
    }
    await consume(path,{emit,minimum,maximum,precision:"f16",quantum:2**-24,fundamentalF16:true});
  }
  async binary(path: FlatConditions, left: FlatProducer, right: FlatProducer,
    operator: "+" | "-" | "*" | "/", consume: FlatConsumer): Promise<void> {
    await left(path, async (aPath, a) => right(aPath, async (bPath, b) => {
      const refreshedA=await this.refresh(bPath,a),refreshedB=await this.refresh(bPath,b);
      if(!refreshedA||!refreshedB){this.stream.eliminatedBranches++;return;}
      a=refreshedA;b=refreshedB;
      if (operator === "/" && b.minimum <= 0 && b.maximum >= 0)
        throw new Error("Division requires a domain excluding zero");
      const apply = (x: number, y: number) => operator === "+" ? x+y : operator === "-" ? x-y :
        operator === "*" ? x*y : x/y;
      const values = [apply(a.minimum,b.minimum),apply(a.minimum,b.maximum),
        apply(a.maximum,b.minimum),apply(a.maximum,b.maximum)];
      const minimum = Math.min(...values), maximum = Math.max(...values);
      if (!Number.isFinite(minimum) || !Number.isFinite(maximum))
        throw new Error("Binary64 finite arithmetic domain has not been proved");
      if(a.literal!==undefined&&b.literal!==undefined){await this.literal(bPath,apply(a.literal,b.literal),consume);return;}
      if((operator==="-"||operator==="+")&&a.literal===undefined&&b.literal===undefined&&await this.key(a.emit)===await this.key(b.emit)){
        this.stream.eliminatedBranches++;
        if(operator==="-"){await this.literal(bPath,0,consume);return;}
        // Doubling is exact binary64 in this finite domain, including -0.
        // Keep one producer and inverse-normalize future comparisons by two.
        const lower=Math.max(a.minimum,b.minimum),upper=Math.min(a.maximum,b.maximum);
        if(lower>upper)return;
        await consume(bPath,{minimum:2*lower,maximum:2*upper,...(a.positiveZero||b.positiveZero?{positiveZero:true as const}:{}),
          ...((a.precision||b.precision)&&Math.max(Math.abs(2*lower),Math.abs(2*upper))<=3.4028234663852886e38?
            {precision:"f32" as const}:{}),...(a.quantum?{quantum:2*a.quantum}:{}),
          restrict:async(p,interval)=>{
            const normalized=normalizeInterval(interval,(op,rhs)=>normalizeExactAffineComparison(rational(2n),rational(0n),op,rhs));
            return normalized===false?undefined:normalized===true?p:this.refine(p,a,normalized);
          },emit:async()=>{await this.stream.write("(");await a.emit();await this.stream.write("*2.0)");}});
        return;
      }
      if(operator==="+"&&a.literal===0&&(Object.is(a.literal,-0)||b.minimum>0||b.maximum<0||b.positiveZero)){
        this.stream.eliminatedBranches++;await consume(bPath,b);return;
      }
      if((operator==="+"||operator==="-")&&b.literal===0&&
        ((operator==="+"?Object.is(b.literal,-0):!Object.is(b.literal,-0))||a.minimum>0||a.maximum<0||a.positiveZero)){
        this.stream.eliminatedBranches++;await consume(bPath,a);return;
      }
      if(operator==="+"&&a.minimum===0&&a.maximum===0&&b.minimum===0&&b.maximum===0&&
        ((a.literal===0&&!Object.is(a.literal,-0))||(b.literal===0&&!Object.is(b.literal,-0)))){
        this.stream.eliminatedBranches++;await this.literal(bPath,0,consume);return;
      }
      if(operator==="*"&&b.literal===1){this.stream.eliminatedBranches++;await consume(bPath,a);return;}
      const emit=async () => {
        await this.stream.write("("); await a.emit(); await this.stream.write(operator);
        await b.emit(); await this.stream.write(")");
      };
      let restrict:FlatInput["restrict"];
      // A finite multiplication by a power of two >=1 is exact in binary64.
      // Other arithmetic retains its operation and its comparison verbatim.
      const scale=operator==="*"?b.literal:undefined;
      if(scale!==undefined&&Math.abs(scale)>=1&&Number.isInteger(Math.log2(Math.abs(scale)))){
        restrict=async(path,interval)=>{
          const normalized=normalizeInterval(interval,(op,rhs)=>normalizeExactAffineComparison(
            exactNumberRational(scale),rational(0n),op,rhs));
          return normalized===false?undefined:normalized===true?path:this.refine(path,a,normalized);
        };
      }
      if(b.literal!==undefined&&a.precision){
        restrict=async(path,interval)=>{
          const normalized=normalizeInterval(interval,(op,rhs)=>normalizeFiniteArithmeticComparison(a.precision!,operator,b.literal!,op,rhs));
          return normalized===false?undefined:normalized===true?path:this.refine(path,a,normalized);
        };
      }
      if(a.literal!==undefined&&b.precision&&(operator==="+"||operator==="*")){
        // Finite IEEE addition and multiplication commute bitwise, including
        // signed zero. Normalize the comparison without changing source order.
        restrict=async(path,interval)=>{
          const normalized=normalizeInterval(interval,(op,rhs)=>normalizeFiniteArithmeticComparison(
            b.precision!,operator,a.literal!,op,rhs));
          return normalized===false?undefined:normalized===true?path:this.refine(path,b,normalized);
        };
      }
      if(operator==="/"&&a.literal!==undefined&&a.literal>0&&b.precision&&b.minimum>0){
        restrict=async(path,interval)=>{
          const normalized=normalizeInterval(interval,(op,rhs)=>normalizePositiveReciprocalComparison(
            b.precision!,a.literal!,op,rhs));
          return normalized===false?undefined:normalized===true?path:this.refine(path,b,normalized);
        };
      }
      if(operator==="*"&&b.literal===undefined&&a.quantum&&
        ((b.minimum>0&&a.quantum*b.minimum>=2**-1022)||(b.maximum<0&&a.quantum*(-b.maximum)>=2**-1022))){
        const sign=b.minimum>0?1:-1;
        restrict=async(p,interval)=>{
          if(a.precision){
            for(const side of ["lower","upper"] as const){
              const bound=interval[side];if(!bound)continue;
              const positive=bound.value.numerator>=0n;
              const extreme=side==="lower"?positive:!positive;
              const factor=sign>0?(extreme?b.maximum:b.minimum):(extreme?b.minimum:b.maximum);
              // Choose the endpoint giving the necessary extremal product on
              // the side of zero selected by the comparison. The discrete
              // preimage evaluates the actual binary64 multiplication.
              const normalized=normalizeInterval({[side]:bound},(op,rhs)=>
                normalizeFiniteArithmeticComparison(a.precision!,"*",factor,op,rhs));
              if(normalized===false)return undefined;
              if(normalized!==true){const next=await this.refine(p,a,normalized);if(!next)return undefined;p=next;}
            }
          }
          const remaining:Interval={};
          for(const side of ["lower","upper"] as const){
            const bound=interval[side];if(!bound)continue;
            if((side==="lower"&&bound.value.numerator>=0n)||(side==="upper"&&bound.value.numerator<=0n)){
              const strict=bound.value.numerator!==0n||!bound.inclusive;
              const consequence=normalizeInterval({[side]:{value:rational(0n),inclusive:!strict}},
                (comparison,rhs)=>normalizeExactAffineComparison(rational(BigInt(sign)),rational(0n),comparison,rhs));
              if(consequence===false)return undefined;
              if(consequence!==true){const refined=await this.refine(p,a,consequence);if(!refined)return undefined;p=refined;}
              if(bound.value.numerator===0n)continue;
            }
            remaining[side]=bound;
          }
          if(!remaining.lower&&!remaining.upper)return p;
          const refreshed=await this.refresh(p,{emit,minimum,maximum});
          return refreshed?this.refine(p,refreshed,remaining):undefined;
        };
      }
      if(operator==="+"&&a.minimum>=0&&b.minimum>=0&&b.literal===undefined){
        restrict=async(path,interval)=>{
          const lower=interval.lower;
          if(lower){
            const before=path.guards;
            for(const [operand,otherMaximum] of [[a,b.maximum],[b,a.maximum]] as const){
              if(!operand.precision)continue;
              // Addition is monotone in both finite operands. If a+b >= r,
              // then a+max(b) >= r is necessary, including binary64 rounding.
              const normalized=normalizeInterval({lower},(op,rhs)=>
                normalizeFiniteArithmeticComparison(operand.precision!,"+",otherMaximum,op,rhs));
              if(normalized===false)return undefined;
              if(normalized!==true){const next=await this.refine(path,operand,normalized);if(!next)return undefined;path=next;}
            }
            path=new FlatConditions(path.domain,before,path.entryGuards,path.nonzero,path.minimumMagnitude);
          }
          const upper=interval.upper;
          if(upper&&upper.value.numerator===0n&&upper.inclusive){
            const zero:Interval={upper:{value:rational(0n),inclusive:true}};
            const first=await this.refine(path,a,zero);
            return first?this.refine(first,b,zero):undefined;
          }
          if(upper){
            const before=path.guards;
            const first=await this.refine(path,a,{upper});
            const second=first?await this.refine(first,b,{upper}):undefined;
            if(!second)return undefined;
            // Necessary consequences are compile-time facts. Their parent
            // comparison remains the emitted guard, so do not emit duplicates.
            path=new FlatConditions(second.domain,before,second.entryGuards,second.nonzero,second.minimumMagnitude);
          }
          return path.refine(await this.key(emit),emit,interval);
        };
      }
      const quantum=a.quantum&&b.quantum?(operator==="+"||operator==="-"?Math.min(a.quantum,b.quantum):
        operator==="*"?a.quantum*b.quantum:operator==="/"&&b.literal!==undefined&&
          Number.isInteger(Math.log2(Math.abs(b.literal)))?a.quantum/Math.abs(b.literal):undefined):undefined;
      const power=b.literal!==undefined?(operator==="*"?Math.abs(b.literal):operator==="/"?1/Math.abs(b.literal):undefined):undefined;
      let precision:FlatInput["precision"];
      if(operator==="*"&&a.precision==="f16"&&b.precision==="f16")precision="f32";
      if((operator==="+"||operator==="-")&&a.precision&&b.precision&&quantum){
        const magnitude=Math.max(Math.abs(minimum),Math.abs(maximum));
        if(quantum>=2**-24&&magnitude<=65504&&magnitude/quantum<=2**11)precision="f16";
        else if(quantum>=2**-149&&magnitude<=3.4028234663852886e38&&magnitude/quantum<=2**24)precision="f32";
      }
      if(power&&a.precision&&Number.isInteger(Math.log2(power))){
        const lattice=(a.quantum??(a.precision==="f16"?2**-24:2**-149))*power;
        const magnitude=Math.max(Math.abs(minimum),Math.abs(maximum));
        if(a.precision==="f16"&&lattice>=2**-24&&magnitude<=65504)precision="f16";
        else if(lattice>=2**-149&&magnitude<=3.4028234663852886e38)precision="f32";
      }
      const positiveZero=operator==="+"&&(a.positiveZero||b.positiveZero)||a.positiveZero&&b.minimum>0&&
        (operator==="*"?(a.minimum===0&&a.maximum===0||a.quantum&&a.quantum*b.minimum>=2**-1022):
          operator==="/"?(a.minimum===0&&a.maximum===0||a.quantum&&a.quantum/b.maximum>=2**-1022):false);
      await consume(bPath,{minimum,maximum,emit,...(positiveZero?{positiveZero:true as const}:{}),
        ...(quantum?{quantum}:{}),...(precision?{precision}:{}),...(restrict?{restrict}:{})});
    }));
  }
  async square(path:FlatConditions,producer:FlatProducer,consume:FlatConsumer):Promise<void>{
    await producer(path,async(path,input)=>{
      if(input.literal!==undefined){await this.literal(path,input.literal*input.literal,consume);return;}
      if(input.minimum===0&&input.maximum===0){this.stream.eliminatedBranches++;await this.literal(path,0,consume);return;}
      const minimum=input.minimum<=0&&input.maximum>=0?0:Math.min(input.minimum**2,input.maximum**2);
      const maximum=Math.max(input.minimum**2,input.maximum**2);
      if(!Number.isFinite(maximum))throw new Error("Finite square domain not proved");
      const emit=async()=>{
        await this.stream.write("(");await input.emit();await this.stream.write("*");await input.emit();await this.stream.write(")");
      };
      const restrict:NonNullable<FlatInput["restrict"]>=async(path,interval)=>{
        if(interval.lower&&(interval.lower.value.numerator>0n||
          (interval.lower.value.numerator===0n&&!interval.lower.inclusive))){
          const nonzero=path.requireNonzero(await this.key(input.emit));if(!nonzero)return undefined;path=nonzero;
          if(input.precision){
            const kind=input.precision,last=kind==="f16"?31743:0x7f7fffff,bound=interval.lower;
            let lo=1,hi=last;
            while(lo<hi){const mid=Math.floor((lo+hi)/2),x=finiteIeeeValue(kind,last+mid);
              const c=compareRational(exactNumberRational(x*x),bound.value);
              if(c>0||(c===0&&bound.inclusive))hi=mid;else lo=mid+1;}
            const magnitude=path.requireMagnitude(await this.key(input.emit),finiteIeeeValue(kind,last+lo));
            if(!magnitude)return undefined;path=magnitude;
          }
        }
        if(interval.upper?.value.numerator===0n&&interval.upper.inclusive){
          if(!input.precision&&!(input.quantum&&input.quantum>=2**-537))
            return path.refine(await this.key(emit),emit,interval);
          return this.refine(path,input,{lower:{value:rational(0n),inclusive:true},upper:{value:rational(0n),inclusive:true}});
        }
        if(interval.upper&&input.precision==="f16"){
          const upper=interval.upper;let lo=0,hi=31743;
          while(lo<hi){const mid=Math.ceil((lo+hi)/2),x=decodeIeeeF16ToF32(mid);
            const c=compareRational(exactNumberRational(x*x),upper.value);
            if(c<0||(c===0&&upper.inclusive))lo=mid;else hi=mid-1;}
          const limit=decodeIeeeF16ToF32(lo);
          const narrowed=await this.refine(path,input,{lower:{value:exactNumberRational(-limit),inclusive:true},
            upper:{value:exactNumberRational(limit),inclusive:true}});
          if(!narrowed)return undefined;
          if(!interval.lower)return narrowed;
          return narrowed.refine(await this.key(emit),emit,interval);
        }
        return path.refine(await this.key(emit),emit,interval);
      };
      await consume(path,{minimum,maximum,restrict,positiveZero:true,
        ...(input.precision==="f16"?{precision:"f32" as const,quantum:2**-48}:{}),emit});
    });
  }
  private async key(emit: RustExpression): Promise<string> {
    const hash = createHash("sha256");
    // Hash direct streamed arithmetic, never retain a source string. Equal
    // substituted producers share condition bounds without a scalar cache.
    await this.stream.inspectExpression(emit, chunk => { hash.update(chunk); });
    return hash.digest("hex");
  }
  private async refine(path:FlatConditions,input:FlatInput,interval:Interval):Promise<FlatConditions|undefined>{
    interval={...interval};
    const inputKey=path.nonzero.size?await this.key(input.emit):undefined;
    if(inputKey&&path.nonzero.has(inputKey)){
      if(input.minimum===0&&input.maximum===0)return undefined;
      for(const side of ["lower","upper"] as const){
        const bound=interval[side];if(bound?.value.numerator===0n&&bound.inclusive)interval[side]={...bound,inclusive:false};
      }
      const magnitude=path.minimumMagnitude.get(inputKey);
      if(magnitude){
        if(input.minimum>=0||interval.lower&&interval.lower.value.numerator>=0n){
          const bound={value:exactNumberRational(magnitude),inclusive:true};
          if(!interval.lower||compareRational(interval.lower.value,bound.value)<0)interval.lower=bound;
        }
        if(input.maximum<=0||interval.upper&&interval.upper.value.numerator<=0n){
          const bound={value:exactNumberRational(-magnitude),inclusive:true};
          if(!interval.upper||compareRational(interval.upper.value,bound.value)>0)interval.upper=bound;
        }
      }
    }
    // IEEE value ranges and the proved lattice settle comparisons before
    // dispatching to a producer preimage. This exposes zero sums of squares.
    for(const side of ["lower","upper"] as const){
      const bound=interval[side];if(!bound)continue;
      const minimum=compareRational(exactNumberRational(input.minimum),bound.value);
      const maximum=compareRational(exactNumberRational(input.maximum),bound.value);
      if(side==="lower"){
        if(maximum<0||(maximum===0&&!bound.inclusive))return undefined;
        if(minimum>0||(minimum===0&&bound.inclusive))delete interval.lower;
      }else{
        if(minimum>0||(minimum===0&&!bound.inclusive))return undefined;
        if(maximum<0||(maximum===0&&bound.inclusive))delete interval.upper;
      }
    }
    if(input.minimum>=0&&input.quantum&&interval.upper){
      const upper=interval.upper;
      const c=compareRational(upper.value,exactNumberRational(input.quantum));
      if(c<0||(c===0&&!upper.inclusive))interval.upper={value:rational(0n),inclusive:true};
    }
    // Lattice narrowing may make a formerly nonempty real interval empty,
    // e.g. 0 < sum < 2^-126 for F16 squares on the 2^-48 lattice.
    const consistent=intersectInterval(interval,{});if(!consistent)return undefined;interval=consistent;
    if(!interval.lower&&!interval.upper)return path;
    if(input.restrict)return input.restrict(path,interval);
    const narrowed=path.refine(await this.key(input.emit),input.emit,interval,input.fundamentalF16);
    if(!narrowed||!input.fundamentalF16)return narrowed;
    // Necessary consequences are safe for contradiction proofs, even when
    // they were derived from this guard. They are NOT sufficient to delete
    // a guard as invariant: that requires the independent leaf checks below.
    for(const guard of narrowed.guards){
      if(guard.fundamentalF16)continue;
      const bounds=await this.sourceBounds(guard.emit,narrowed.domain);
      if(bounds.opaque)continue;
      const constraint=new Map(narrowed.domain.entries()).get(guard.key)!;
      for(const side of ["lower","upper"] as const){
        const bound=constraint[side];if(!bound)continue;
        const min=compareRational(exactNumberRational(bounds.minimum),bound.value),
          max=compareRational(exactNumberRational(bounds.maximum),bound.value);
        if(side==="lower"?(max<0||(max===0&&!bound.inclusive)):(min>0||(min===0&&!bound.inclusive))){
          this.stream.eliminatedBranches++;return undefined;
        }
      }
    }
    return narrowed;
  }
  async comparison(path:FlatConditions,producer:FlatProducer,op:"<"|"<="|">"|">=",rhs:Rational,
    truth:FlatConsumer,falsity:FlatConsumer):Promise<void>{
    const inverse={"<":">=","<=":">",">":"<=",">=":"<"} as const;
    await producer(path,async(inputPath,input)=>{
      for(const [comparison,consume] of [[op,truth],[inverse[op],falsity]] as const){
        const interval:Interval=comparison.startsWith("<")?
          {upper:{value:rhs,inclusive:comparison==="<="}}:{lower:{value:rhs,inclusive:comparison===">="}};
        const narrowed=await this.refine(inputPath,input,interval);
        if(narrowed)await consume(narrowed,input);else this.stream.eliminatedBranches++;
      }
    });
  }
  async round(path: FlatConditions, producer: FlatProducer, kind: "f16" | "f32",
    alreadyF32: boolean, consume: FlatConsumer): Promise<void> {
    await producer(path, async (inputPath, input) => {
      const refreshed=await this.refresh(inputPath,input);
      if(!refreshed){this.stream.eliminatedBranches++;return;}input=refreshed;
      if(input.precision===kind || (kind==="f32"&&(input.precision==="f16"||alreadyF32))){
        this.stream.eliminatedBranches++;await consume(inputPath,input);return;
      }
      if(input.literal!==undefined || (input.minimum===input.maximum&&(input.minimum!==0||input.positiveZero))){
        const value=input.literal??(input.minimum===0&&input.positiveZero?0:input.minimum);
        let rounded=Math.fround(value);
        if(kind==="f16"){
          const data=new DataView(new ArrayBuffer(4));data.setFloat32(0,rounded,true);
          const bits=data.getUint32(0,true);
          rounded=(bits&0x7fffffff)===0?(bits>>>31?-0:0):
            decodeIeeeF16ToF32(roundDyadicToF16IfElse(f32BitsToDyadic(bits)));
        }
        if(!Number.isFinite(rounded))throw new Error("Nonfinite rounding needs explicit domain semantics");
        this.stream.eliminatedBranches++;
        await this.literal(inputPath,rounded,(p,v)=>consume(p,{...v,precision:kind}));return;
      }
      const composed = kind === "f16" && !alreadyF32 && input.precision!=="f32";
      const bits = kind === "f16" ? 10 : 23, first = kind === "f16" ? -14 : -126;
      const last = kind === "f16" ? 15 : 127;
      const overflow = (kind === "f16" ? 65520 : 2**128-2**103) -
        (composed ? 2**-9 : 0);
      if (input.minimum <= -overflow || input.maximum >= overflow)
        throw new Error("Flat finite rounding requires an overflow-free producer domain");
      if(input.minimum<=0&&input.maximum>=0){
        const zero=exactNumberRational(0);
        const zeroPath=await this.refine(inputPath,input,{lower:{value:zero,inclusive:true},upper:{value:zero,inclusive:true}});
        if(zeroPath)await consume(zeroPath,{emit:input.emit,minimum:0,maximum:0,precision:kind,quantum:kind==="f16"?2**-24:2**-149});
      }
      const emitBand = async (negative: boolean, exponent: number, low: number, high: number) => {
        const minimum = negative ? -high : low, maximum = negative ? -low : high;
        if (input.maximum < minimum || input.minimum > maximum) return;
        const interval: Interval = {
          lower: {value: exactNumberRational(minimum), inclusive: !negative && minimum!==0},
          upper: {value: exactNumberRational(maximum), inclusive: negative && maximum!==0},
        };
        const narrowed = await this.refine(inputPath,input,interval);
        if (!narrowed) { this.stream.eliminatedBranches++; return; }
        const unit = 2**(Math.max(first,exponent)-bits);
        const unit32 = 2**(exponent-23), ratio = unit/unit32;
        const roundValue = (x:number) => {
          let y = negative ? -x : x;
          y = composed ? (((y/unit32+2**52)-2**52)/ratio) : y/unit;
          y = ((y+2**52)-2**52)*unit;
          return negative ? -y : y;
        };
        const from = Math.max(input.minimum,minimum), to = Math.min(input.maximum,maximum);
        const restrict:NonNullable<FlatInput["restrict"]>=async(path,interval)=>{
          const normalized=normalizeInterval(interval,(op,rhs)=>normalizeRoundedAffineComparison(
            composed?"f32-f16":kind,rational(1n),rational(0n),op,rhs));
          return normalized===false?undefined:normalized===true?path:this.refine(path,input,normalized);
        };
        const roundedMinimum=roundValue(from),roundedMaximum=roundValue(to);
        if(Object.is(roundedMinimum,roundedMaximum)){
          this.stream.eliminatedBranches++;await this.literal(narrowed,roundedMinimum,(p,v)=>consume(p,{...v,precision:kind}));return;
        }
        await consume(narrowed,{minimum:roundedMinimum,maximum:roundedMaximum,precision:kind,
          quantum:Math.max(input.quantum??unit,unit),restrict,emit:async()=>{
          await this.stream.write(negative ? "-(((" : "(((");
          if(composed){
            await this.stream.write("((");if(negative)await this.stream.write("-");
            await this.stream.write("(");await input.emit();await this.stream.write(")");
            await this.stream.write(`/${rustF64(unit32)}+4503599627370496.0)-4503599627370496.0)/${rustF64(ratio)}`);
          }else{
            if(negative)await this.stream.write("-");await this.stream.write("(");await input.emit();
            await this.stream.write(`)/${rustF64(unit)}`);
          }
          await this.stream.write(`+4503599627370496.0)-4503599627370496.0)*${rustF64(unit)})`);
        }});
      };
      // Subnormals and zeros are included in the signed first bands. Returning
      // the operand for an exact zero preserves its sign; rounded negative
      // underflows retain unary minus in their arithmetic leaf.
      for (let exponent = composed ? -25 : first-1; exponent <= last; exponent++) {
        const low = !composed && exponent === first-1 ? 0 : 2**exponent;
        const high = exponent === first-1 ? 2**first : 2**(exponent+1);
        await emitBand(false,exponent,low,high);await emitBand(true,exponent,low,high);
      }
      if(composed){
        // Values below the smallest composed exponent round to signed zero.
        // Keeping the double-round sequence also handles its endpoint tie.
        await emitBand(false,-25,0,2**-25);await emitBand(true,-25,0,2**-25);
      }
    });
  }
  /** Numeric rule visitors produce one checked affine run at a time. The
   * consumer receives its intersected path before emission; numeric choices
   * cannot introduce an inline if in an arithmetic producer.
   */
  private async numeric(path:FlatConditions,producer:FlatProducer,precision:"f16"|"f32",
    visit:(input:FlatInput,consume:NumericRunConsumer)=>Promise<void>,consume:FlatConsumer):Promise<void>{
    await producer(path,async(inputPath,input)=>{
      const refreshed=await this.refresh(inputPath,input);
      if(!refreshed){this.stream.eliminatedBranches++;return;}input=refreshed;
      if(input.precision!==precision && !(precision==="f32"&&input.precision==="f16"))
        throw new Error("Numeric run substitution requires a proved discrete input dtype");
      await visit(input,async(run:NumericAffineRun)=>{
        const minimum=Math.max(input.minimum,run.minimum),maximum=Math.min(input.maximum,run.maximum);
        if(minimum>maximum){this.stream.eliminatedBranches++;return;}
        const narrowed=await this.refine(inputPath,input,{
          lower:{value:exactNumberRational(minimum),inclusive:true},
          upper:{value:exactNumberRational(maximum),inclusive:true}});
        if(!narrowed){this.stream.eliminatedBranches++;return;}
        if(run.rounded){
          // This alternative was checked against every reached F16 policy
          // input by the visitor. The raw affine value has <=13 bits here,
          // so its two scalar operations are exact F32 before half rounding.
          const raw:FlatInput={minimum:run.slope*minimum+run.offset,maximum:run.slope*maximum+run.offset,
            precision:"f32",quantum:2**-26,
            restrict:async(p,interval)=>{
              const normalized=normalizeInterval(interval,(op,rhs)=>normalizeFiniteAffineRunComparison(
                input.precision!,run.slope,run.offset,op,rhs));
              return normalized===false?undefined:normalized===true?p:this.refine(p,input,normalized);
            },emit:async()=>{await this.stream.write(`(${rustF64(run.slope)}*(`);await input.emit();
              await this.stream.write(`)+${rustF64(run.offset)})`);}};
          await this.round(narrowed,(p,k)=>k(p,raw),run.rounded,true,consume);return;
        }
        const apply=(x:number)=>run.constant===undefined?run.slope*x+run.offset:run.constant;
        const endpoints=[apply(minimum),apply(maximum)];
        const restrict:NonNullable<FlatInput["restrict"]>=async(path,interval)=>{
          const normalized=normalizeInterval(interval,(op,rhs)=>normalizeFiniteAffineRunComparison(
            input.precision!,run.constant===undefined?run.slope:0,run.constant??run.offset,op,rhs));
          return normalized===false?undefined:normalized===true?path:this.refine(path,input,normalized);
        };
        await consume(narrowed,{precision,restrict,...(run.constant!==undefined?{literal:run.constant}:{}),minimum:Math.min(...endpoints),maximum:Math.max(...endpoints),emit:async()=>{
          if(run.constant!==undefined){await this.stream.write(rustF64(Object.is(run.constant,-0)?"-0":run.constant));return;}
          if(run.slope===1&&run.offset===0){await input.emit();return;}
          await this.stream.write(`(${rustF64(run.slope)}*(`);await input.emit();
          await this.stream.write(`)+${rustF64(run.offset)})`);
        }});
      });
    });
  }
  async silu(path:FlatConditions,producer:FlatProducer,consume:FlatConsumer):Promise<void>{
    await this.numeric(path,producer,"f16",(input,visit)=>emitRustSilu(this.stream,input.emit,
      Math.max(Math.abs(input.minimum),Math.abs(input.maximum)),visit,
      {minimum:input.minimum,maximum:input.maximum}),consume);
  }
  async sqrt(path:FlatConditions,producer:FlatProducer,consume:FlatConsumer):Promise<void>{
    await producer(path,async(path,input)=>{
      const refreshed=await this.refresh(path,input);if(!refreshed){this.stream.eliminatedBranches++;return;}input=refreshed;
      if((input.precision!=="f32"&&input.precision!=="f16")||!(input.minimum>0))
        throw new Error("Positive discrete F32 square-root domain required");
      if(input.minimum===input.maximum){
        await this.literal(path,foldCertifiedF32Sqrt(input.minimum),(p,v)=>consume(p,{...v,precision:"f32"}));return;
      }
      // Normalize x=4^e*m, 1<=m<4. On cells of width h=2^-12,
      // a chord through rounded F32 endpoints differs from sqrt(m) by
      // at most 2^-24+h^2/32 < one F32 ulp (2^-23).
      // The slope has <=11 significant bits: slope*m+offset is exact F64
      // for every F32 m. Rounding that chord gives a candidate at most one
      // F32 value away. Exact midpoint squares correct it. A positive F32
      // input cannot equal these odd 49/50-bit squared midpoints: no tie
      // parity branch is needed. All consumer paths stay flattened.
      for(let exponent=Math.floor(Math.log2(input.minimum)/2);exponent<=Math.floor(Math.log2(input.maximum)/2);exponent++){
        const scale=2**(2*exponent),rootScale=2**exponent,unit=2**(exponent-23);
        const first=Math.max(0,Math.min(12287,Math.floor((Math.max(1,input.minimum/scale)-1)*4096)));
        const last=Math.max(0,Math.min(12287,Math.floor((Math.min(4,input.maximum/scale)-1)*4096)));
        for(let cell=first;cell<=last;cell++){
          const lo=1+cell/4096,hi=1+(cell+1)/4096;
          const narrowed=await this.refine(path,input,{lower:{value:exactNumberRational(lo*scale),inclusive:true},
            upper:{value:exactNumberRational(hi*scale),inclusive:false}});
          if(!narrowed){this.stream.eliminatedBranches++;continue;}
          const y0=foldCertifiedF32Sqrt(lo),y1=foldCertifiedF32Sqrt(hi),slope=(y1-y0)*4096,offset=y0-slope*lo;
          const candidate:FlatInput={precision:"f32",quantum:unit,minimum:y0*rootScale,maximum:y1*rootScale,emit:async()=>{
            await this.stream.write(`((((${rustF64(slope)}*(`);await input.emit();
            await this.stream.write(`/${rustF64(scale)})+${rustF64(offset)})/${rustF64(2**-23)}+4503599627370496.0)-4503599627370496.0)*${rustF64(unit)})`);
          }};
          const original:FlatProducer=(p,k)=>k(p,input),c:FlatProducer=(p,k)=>k(p,candidate);
          const literal=(x:number):FlatProducer=>(p,k)=>this.literal(p,x,k);
          const midpoint=(sign:number):FlatProducer=>(p,k)=>this.binary(p,c,literal(sign*unit/2),"+",k);
          const delta=(sign:number):FlatProducer=>(p,k)=>this.binary(p,original,
            (p,k)=>this.square(p,midpoint(sign),k),"-",k);
          const result=(p:FlatConditions,correction:number)=>{
            const restrict:NonNullable<FlatInput["restrict"]>=async(p,interval)=>{
              const normalized=normalizeInterval(interval,(op,rhs)=>normalizePositiveSqrtComparison(op,rhs));
              return normalized===false?undefined:normalized===true?p:this.refine(p,input,normalized);
            };
            const deliver=(p:FlatConditions,v:FlatInput)=>consume(p,{...v,precision:"f32",quantum:unit,restrict,
              minimum:Math.max(y0*rootScale,v.minimum),maximum:Math.min(y1*rootScale,v.maximum)});
            return correction===0?deliver(p,candidate):this.binary(p,c,literal(correction*unit),"+",deliver);
          };
          await this.comparison(narrowed,delta(-1),"<",rational(0n),p=>result(p,-1),
            p=>this.comparison(p,delta(1),">",rational(0n),p=>result(p,1),p=>result(p,0)));
        }
      }
    });
  }
  async exponential(path:FlatConditions,producer:FlatProducer,consume:FlatConsumer,quantum?:number):Promise<void>{
    await this.numeric(path,producer,"f32",(input,visit)=>emitRustExp(this.stream,input.emit,true,
      undefined,{minimum:input.minimum,maximum:input.maximum},quantum,visit),consume);
  }
  async leaf(path: FlatConditions, label: string, input: FlatInput): Promise<void> {
    const result=await this.sourceBounds(input.emit,path.domain);
    if(!result.opaque&&Object.is(result.minimum,result.maximum)){
      const value=result.minimum;
      input={...input,literal:value,minimum:value,maximum:value,
        emit:()=>this.stream.write(rustF64(Object.is(value,-0)?"-0":value))};
    }
    let guards=[...path.guards];
    // Sequential implication checks avoid circular elimination of two
    // equivalent guards. Derived facts with no emitted guard are excluded.
    for(const guard of path.guards){
      const others=new DirectBranchDomain(new Map([...path.domain.entries()]
        .filter(([key])=>key!==guard.key&&guards.some(g=>g.key===key))));
      const range=await this.sourceBounds(guard.emit,others);
      if(range.opaque)continue;
      const interval=new Map(path.domain.entries()).get(guard.key)!;
      let always=true;
      for(const side of ["lower","upper"] as const){
        const bound=interval[side];if(!bound)continue;
        const min=compareRational(exactNumberRational(range.minimum),bound.value);
        const max=compareRational(exactNumberRational(range.maximum),bound.value);
        const impossible=side==="lower"?(max<0||(max===0&&!bound.inclusive)):(min>0||(min===0&&!bound.inclusive));
        if(impossible){this.stream.eliminatedBranches++;return;}
        always&&=side==="lower"?(min>0||(min===0&&bound.inclusive)):(max<0||(max===0&&bound.inclusive));
      }
      if(always){guards=guards.filter(g=>g.key!==guard.key);this.stream.eliminatedBranches++;}
    }
    path=new FlatConditions(path.domain,guards,path.entryGuards,path.nonzero,path.minimumMagnitude);
    const key=!result.opaque&&Object.is(result.minimum,result.maximum)?await this.key(input.emit):result.sourceKey;
    const previous=this.pendingLeaf;
    if(previous&&previous.key===key&&previous.label===label){
      const merged=unionLeafConditions(previous.path,path);
      if(merged){this.pendingLeaf={path:merged,label,input:previous.input,key};this.stream.eliminatedBranches++;return;}
    }
    await this.finishRound();this.pendingLeaf={path,label,input,key};
  }
  private async sourceBounds(emit:RustExpression,domain:DirectBranchDomain):Promise<SourceBounds&{sourceKey:string}>{
    const intervals=new Map(domain.entries());
    const sourceHash=createHash("sha256");
    const parser=new DirectSourceBounds(name=>{
      const hash=createHash("sha256").update(name).digest("hex"),interval=intervals.get(hash);
      if(/^input_tokens\[\d+\]\[\d+\]$/.test(name)){
        const satisfies=(index:number,side:"lower"|"upper")=>{
          const bound=interval?.[side];if(!bound)return true;
          const c=compareRational(exactNumberRational(finiteIeeeValue("f16",index)),bound.value);
          return side==="lower"?(c>0||(c===0&&bound.inclusive)):(c<0||(c===0&&bound.inclusive));
        };
        let lo=0,hi=63486;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(satisfies(mid,"lower"))hi=mid;else lo=mid+1;}
        const first=lo;lo=0;hi=63486;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(satisfies(mid,"upper"))lo=mid;else hi=mid-1;}
        if(first>lo||!satisfies(first,"lower")||!satisfies(lo,"upper"))return {minimum:-Infinity,maximum:Infinity,key:hash,opaque:true};
        const minimum=finiteIeeeValue("f16",first),maximum=finiteIeeeValue("f16",lo);
        return {minimum:minimum===0?-0:minimum,maximum,key:hash};
      }
      return {minimum:-Infinity,maximum:Infinity,key:hash,opaque:true};
    });
    await this.stream.inspectExpression(emit,chunk=>{sourceHash.update(chunk);parser.accept(chunk);});
    return {...parser.finish(),sourceKey:sourceHash.digest("hex")};
  }
  /** Later producers can narrow the fundamental inputs of an earlier operand.
   * Re-read its scalar source under the current path before expanding another
   * numerical choice; no expression or activation is stored by the reducer.
   */
  private async refresh(path:FlatConditions,input:FlatInput):Promise<FlatInput|undefined>{
    if(input.literal!==undefined||path.domain.entries().next().done)return input;
    const range=await this.sourceBounds(input.emit,path.domain);
    const intervals=new Map(path.domain.entries());
    let interval=intervals.get(range.sourceKey);
    // Re-normalize inherited comparisons under the current fundamental input
    // bindings. Retain every original guard: these are necessary consequences,
    // and cannot justify deleting the guard that supplied them.
    if(!range.opaque){
      for(const guard of path.guards){
        if(guard.fundamentalF16||guard.key===range.sourceKey)continue;
        const inherited=await this.sourceBounds(guard.emit,path.domain);
        if(inherited.opaque||inherited.key!==range.key)continue;
        const constraint=intervals.get(guard.key);
        if(!constraint)continue;
        const intersection=intersectInterval(interval??{},constraint);
        if(!intersection)return undefined;
        interval=intersection;
      }
    }
    if(interval){
      const bounded=intersectInterval(interval,{lower:{value:exactNumberRational(input.minimum),inclusive:true},
        upper:{value:exactNumberRational(input.maximum),inclusive:true}});
      if(!bounded)return undefined;
      if(input.precision){
        const kind=input.precision,last=kind==="f16"?63486:2*0x7f7fffff;
        const satisfies=(index:number,side:"lower"|"upper")=>{
          const bound=bounded[side];if(!bound)return true;
          const c=compareRational(exactNumberRational(finiteIeeeValue(kind,index)),bound.value);
          return side==="lower"?(c>0||(c===0&&bound.inclusive)):(c<0||(c===0&&bound.inclusive));
        };
        let lo=0,hi=last;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(satisfies(mid,"lower"))hi=mid;else lo=mid+1;}
        const first=lo;lo=0;hi=last;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(satisfies(mid,"upper"))lo=mid;else hi=mid-1;}
        if(first>lo||!satisfies(first,"lower")||!satisfies(lo,"upper"))return undefined;
        input={...input,minimum:finiteIeeeValue(kind,first),maximum:finiteIeeeValue(kind,lo)};
      }else{
        input={...input,minimum:Math.max(input.minimum,Number(bounded.lower!.value.numerator)/Number(bounded.lower!.value.denominator)),
          maximum:Math.min(input.maximum,Number(bounded.upper!.value.numerator)/Number(bounded.upper!.value.denominator))};
      }
    }
    if(input.minimum===input.maximum&&input.minimum!==0){
      const value=input.minimum;
      return {...input,literal:value,emit:()=>this.stream.write(rustF64(value))};
    }
    if(range.opaque)return input;
    const minimum=Math.max(input.minimum,range.minimum),maximum=Math.min(input.maximum,range.maximum);
    if(minimum>maximum)return undefined;
    if(Object.is(range.minimum,range.maximum)){
      const value=range.minimum;
      return {...input,minimum:value,maximum:value,literal:value,
        emit:()=>this.stream.write(rustF64(Object.is(value,-0)?"-0":value))};
    }
    return {...input,minimum,maximum};
  }
  /** Commit the current affine/constant run before leaving a result block.
   * Only one not-yet-emitted leaf is retained for adjacent condition merging.
   */
  async finishRound():Promise<void>{
    const pending=this.pendingLeaf;if(!pending)return;this.pendingLeaf=undefined;
    const {path,label,input}=pending;
    if(!/^[a-z][a-z0-9_]*$/.test(label))throw new Error("Invalid flat result label");
    let first=true;
    for(const entry of path.entryGuards){await this.stream.write(first?"if ":" && ");first=false;await entry();}
    for(const guard of path.guards){
      const interval = new Map(path.domain.entries()).get(guard.key)!;
      for(const side of ["lower","upper"] as const){
        const bound=interval[side];if(!bound)continue;
        await this.stream.write(first?"if ":" && ");first=false;
        await this.stream.write("(");await guard.emit();await this.stream.write(")");
        await this.stream.write(side==="lower"?(bound.inclusive?">=":">"):(bound.inclusive?"<=":"<"));
        await this.stream.write(rustBoundary(bound.value));
      }
    }
    if(!first)await this.stream.write(" {");
    await this.stream.write(`break '${label} `);await input.emit();await this.stream.write(";");
    if(!first)await this.stream.write("}");
  }
}
function unionLeafConditions(a:FlatConditions,b:FlatConditions):FlatConditions|undefined{
  if(a.entryGuards!==b.entryGuards||a.guards.length!==b.guards.length)return undefined;
  const ai=new Map(a.domain.entries()),bi=new Map(b.domain.entries()),merged=new Map<string,Interval>();
  let differences=0;
  for(const guard of a.guards){
    const other=b.guards.find(g=>g.key===guard.key);if(!other)return undefined;
    const x=ai.get(guard.key)!,y=bi.get(guard.key)!;
    if(sameInterval(x,y)){merged.set(guard.key,x);continue;}
    if(++differences>1)return undefined;
    const earlier=lessLower(x.lower,y.lower)?x:y,later=earlier===x?y:x;
    if(earlier.upper&&later.lower){
      const c=compareRational(earlier.upper.value,later.lower.value);
      if(c<0||(c===0&&!earlier.upper.inclusive&&!later.lower.inclusive)){
        if(!guard.fundamentalF16||!other.fundamentalF16||f16GapContainsValue(earlier.upper,later.lower))return undefined;
      }
    }
    const union:Interval={};
    if(earlier.lower)union.lower=earlier.lower;
    if(x.upper&&y.upper){
      const c=compareRational(x.upper.value,y.upper.value);
      union.upper=c>0?x.upper:c<0?y.upper:{value:x.upper.value,inclusive:x.upper.inclusive||y.upper.inclusive};
    }
    merged.set(guard.key,union);
  }
  return new FlatConditions(new DirectBranchDomain(merged),a.guards,a.entryGuards);
}
function f16GapContainsValue(upper:NonNullable<Interval["upper"]>,lower:NonNullable<Interval["lower"]>):boolean{
  let lo=0,hi=63487;
  while(lo<hi){const mid=Math.floor((lo+hi)/2);
    if(mid===63487){hi=mid;continue;}
    const c=compareRational(exactNumberRational(finiteIeeeValue("f16",mid)),upper.value);
    if(c>0||(c===0&&!upper.inclusive))hi=mid;else lo=mid+1;
  }
  if(lo===63487)return false;
  const c=compareRational(exactNumberRational(finiteIeeeValue("f16",lo)),lower.value);
  return c<0||(c===0&&!lower.inclusive);
}
function sameInterval(a:Interval,b:Interval):boolean{
  return (["lower","upper"] as const).every(side=>{
    const x=a[side],y=b[side];return !x&&!y||!!x&&!!y&&x.inclusive===y.inclusive&&compareRational(x.value,y.value)===0;
  });
}
function lessLower(a:Interval["lower"],b:Interval["lower"]):boolean{
  if(!a)return true;if(!b)return false;
  const c=compareRational(a.value,b.value);return c<0||(c===0&&(a.inclusive||!b.inclusive));
}
function normalizeInterval(interval:Interval,normalize:(op:"<"|"<="|">"|">=",rhs:Rational)=>
  {op:"<"|"<="|">"|">=",value:Rational}|boolean):Interval|boolean{
  const result:Interval={};
  for(const side of ["lower","upper"] as const){
    const bound=interval[side];if(!bound)continue;
    const op=side==="lower"?(bound.inclusive?">=":">"):(bound.inclusive?"<=":"<");
    const transformed=normalize(op,bound.value);
    if(transformed===false)return false;if(transformed===true)continue;
    result[transformed.op.startsWith("<")?"upper":"lower"]={value:transformed.value,
      inclusive:transformed.op.length===2};
  }
  return result;
}
function compareRational(a:Rational,b:Rational):number{
  const difference=a.numerator*b.denominator-b.numerator*a.denominator;
  return difference<0n?-1:difference>0n?1:0;
}
function rustBoundary(value:Rational):string {
  const exponent = -(value.denominator.toString(2).length-1);
  const number = Number(value.numerator)*2**exponent;
  const exact = exactNumberRational(number);
  if(exact.numerator!==value.numerator||exact.denominator!==value.denominator)
    throw new Error("Inexact flat condition boundary");
  return rustF64(number);
}
