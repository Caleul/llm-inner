import { createWriteStream } from "node:fs";
import { once } from "node:events";
import { finished } from "node:stream/promises";
import { DirectBranchDomain, type Comparison, type Rational, normalizeExactAffineComparison } from "./direct-branch-domain.js";
import { exactNumberRational, normalizeReciprocalRootAffineComparison, normalizeRoundedAffineComparison, type DirectRoundedFormat } from "./direct-round-preimage.js";
import { DirectReductionGate } from "./direct-reduction-gate.js";

const completedWrite = Promise.resolve();

export type RustExpression = () => Promise<void>;
export interface PositiveRoundRange { minimum: number; maximum: number }
/** Writes direct source with bounded memory; expression callbacks emit immediately. */
export class DirectRustStream {
  private readonly output;
  bytes = 0;
  eliminatedBranches = 0;
  inspectedExpressions = 0;
  private reductionGate?:DirectReductionGate;
  private inspection: ((source:string)=>void) | undefined;
  private cancellation:Error|undefined;
  private fragments:string[]=[];
  private bufferedBytes=0;
  outputWrites=0;
  constructor(path: string, private readonly bufferLimit=65536) {
    if(!Number.isSafeInteger(bufferLimit)||bufferLimit<0)throw new RangeError("Invalid stream buffer limit");
    this.output = createWriteStream(path, { encoding: "utf8" });
  }
  write(source: string): Promise<void> {
    try {
    if(this.cancellation)throw this.cancellation;
    if(this.inspection){this.inspection(source);return completedWrite;}
    this.reductionGate?.accept(source);
    // Numeric fragments are complete lexemes at emission boundaries. Mark
    // floating literals explicitly to avoid millions of unresolved operator
    // obligations in Rust's single-function type checker. Authored strings
    // contain no floating literals; checkpoint data never supplies source text.
    source=source.replace(/(?<![A-Za-z0-9_])(?:\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![A-Za-z0-9_.])/g,"$&_f64");
    const size=Buffer.byteLength(source);this.bytes += size;
    // Buffer source bytes only: never retain calculations, weights or values.
    // The bound applies between sequential writes, including oversized chunks.
    if(this.bufferLimit===0){this.outputWrites++;return this.output.write(source)?completedWrite:once(this.output,"drain").then(()=>{});}
    if(size>=this.bufferLimit)return this.flushThenWrite(source);
    if(this.bufferedBytes+size>this.bufferLimit)return this.flush().then(()=>{
      this.fragments.push(source);this.bufferedBytes=size;
    });
    this.fragments.push(source);this.bufferedBytes+=size;
    return this.bufferedBytes>=this.bufferLimit?this.flush():completedWrite;
    }catch(error){return Promise.reject(error);}
  }
  private flush():Promise<void>{
    if(!this.bufferedBytes)return completedWrite;
    const source=this.fragments.join("");this.fragments=[];this.bufferedBytes=0;this.outputWrites++;
    return this.output.write(source)?completedWrite:once(this.output,"drain").then(()=>{});
  }
  private async flushThenWrite(source:string):Promise<void>{
    await this.flush();this.outputWrites++;
    if(!this.output.write(source))await once(this.output,"drain");
  }
  beginReducedExpression():void{this.reductionGate=new DirectReductionGate();}
  /** Inspect already substituted arithmetic with bounded memory. Inspection
   * never writes, caches a source fragment or bypasses the flat-source gate.
   */
  async inspectExpression(expression:RustExpression,consume:(source:string)=>void):Promise<void>{
    if(this.inspection)throw new Error("Nested expression inspection");
    this.inspectedExpressions++;
    const gate=new DirectReductionGate();
    this.inspection=source=>{gate.accept(source);consume(source);};
    try{await expression();gate.finish();}finally{this.inspection=undefined;}
  }
  async close(): Promise<void> { this.reductionGate?.finish();await this.flush();this.output.end(); await finished(this.output); }
  destroy(): void { this.fragments=[];this.bufferedBytes=0;this.output.on("error", () => {}); this.output.destroy(); }
  cancel(reason:string):void{this.cancellation=new Error(reason);}
  async exactAffineBranch(domain: DirectBranchDomain, variable: string, scale: Rational, offset: Rational,
    op: Comparison, rhs: Rational, truth: (domain: DirectBranchDomain) => Promise<void>,
    falsity: (domain: DirectBranchDomain) => Promise<void>): Promise<void> {
    if (!/^[a-z][a-z0-9_]*$/.test(variable)) throw new Error("Invalid variable");
    const normalized = normalizeExactAffineComparison(scale, offset, op, rhs);
    if (typeof normalized === "boolean") {
      this.eliminatedBranches++; await (normalized ? truth : falsity)(domain); return;
    }
    const paths = domain.split(variable, normalized.op, normalized.value);
    if (!paths.truth) { this.eliminatedBranches++; await falsity(paths.falsity!); return; }
    if (!paths.falsity) { this.eliminatedBranches++; await truth(paths.truth); return; }
    await this.write(`if ${variable} ${normalized.op} ${rustExactBoundary(normalized.value)} {`);
    await truth(paths.truth); await this.write("} else {"); await falsity(paths.falsity); await this.write("}");
  }
  /** Descend with semantic path facts; emit no parent branch or stored tree. */
  async flattenedAffinePaths(domain: DirectBranchDomain, variable: string, scale: Rational, offset: Rational,
    op: Comparison, rhs: Rational, truth: (domain: DirectBranchDomain) => Promise<void>,
    falsity: (domain: DirectBranchDomain) => Promise<void>): Promise<void> {
    const normalized=normalizeExactAffineComparison(scale,offset,op,rhs);
    if(typeof normalized==="boolean") {this.eliminatedBranches++;await (normalized?truth:falsity)(domain);return;}
    const paths=domain.split(variable,normalized.op,normalized.value);
    if(paths.truth)await truth(paths.truth);else this.eliminatedBranches++;
    if(paths.falsity)await falsity(paths.falsity);else this.eliminatedBranches++;
  }
  /** Propagate a rounded comparison back to its exact affine producer.
   * Descend under the preimage facts; only flattened leaves emit conditions.
   */
  async flattenedRoundedAffinePaths(domain:DirectBranchDomain,variable:string,format:DirectRoundedFormat,
    scale:Rational,offset:Rational,op:Comparison,rhs:Rational,
    truth:(domain:DirectBranchDomain)=>Promise<void>,falsity:(domain:DirectBranchDomain)=>Promise<void>):Promise<void>{
    const normalized=normalizeRoundedAffineComparison(format,scale,offset,op,rhs);
    if(typeof normalized==="boolean"){this.eliminatedBranches++;await (normalized?truth:falsity)(domain);return;}
    const paths=domain.split(variable,normalized.op,normalized.value);
    if(paths.truth)await truth(paths.truth);else this.eliminatedBranches++;
    if(paths.falsity)await falsity(paths.falsity);else this.eliminatedBranches++;
  }
  async flattenedReciprocalRootPaths(domain:DirectBranchDomain,variable:string,scale:Rational,offset:Rational,
    op:Comparison,rhs:Rational,truth:(domain:DirectBranchDomain)=>Promise<void>,
    falsity:(domain:DirectBranchDomain)=>Promise<void>):Promise<void>{
    const normalized=normalizeReciprocalRootAffineComparison(scale,offset,op,rhs);
    if(typeof normalized==="boolean"){this.eliminatedBranches++;await (normalized?truth:falsity)(domain);return;}
    const paths=domain.split(variable,normalized.op,normalized.value);
    if(paths.truth)await truth(paths.truth);else this.eliminatedBranches++;
    if(paths.falsity)await falsity(paths.falsity);else this.eliminatedBranches++;
  }
  /** Each surviving leaf gets one conjunction of reduced bounds, no nested if. */
  async affineLeaf(domain:DirectBranchDomain,label:string,body:RustExpression,assumed?:DirectBranchDomain):Promise<void>{
    if(!/^[a-z][a-z0-9_]*$/.test(label))throw new Error("Invalid Rust label");
    const conditions:string[]=[];
    for(const [variable,interval] of domain.entries()){
      if(!/^[a-z][a-z0-9_]*$/.test(variable))throw new Error("Invalid variable");
      for(const side of ["lower","upper"] as const){
        const bound=interval[side];if(!bound)continue;
        const comparison=side==="lower"?(bound.inclusive?">=":">"):(bound.inclusive?"<=":"<");
        if(assumed&&!assumed.split(variable,comparison,bound.value).falsity){this.eliminatedBranches++;continue;}
        conditions.push(`${variable}${comparison}${rustExactBoundary(bound.value)}`);
      }
    }
    if(conditions.length)await this.write(`if ${conditions.join(" && ")} {`);
    await this.write(`break '${label} `);await body();await this.write(";");
    if(conditions.length)await this.write("}");
  }
  /** Exact IEEE nearest-even lowering using only exponent predicates and
   * linear binary64 arithmetic. Adding 2^52 rounds a nonnegative normalized
   * significand (<2^24) to an integer; subtraction is exact by Sterbenz.
   * This pair MUST NOT be cancelled by real-number algebra. No binding,
   * scratch, mantissa loop or numeric helper survives the emitted operation.
   */
  async round(kind: "f16" | "f32", input: RustExpression, alreadyF32 = false, positiveRange?: PositiveRoundRange): Promise<void> {
    if(positiveRange && (!Number.isFinite(positiveRange.minimum)||!Number.isFinite(positiveRange.maximum)||
      positiveRange.minimum<=0||positiveRange.maximum<positiveRange.minimum))throw new Error("Invalid positive rounding range");
    const operand=input,composedHalf=kind==="f16"&&!alreadyF32;
    const half=kind==="f16",firstExponent=half?-14:-126,lastExponent=half?15:127,bits=half?10:23;
    const largest=half?65504:3.4028234663852886e38;
    const overflow=largest+2**(lastExponent-bits-1)-(composedHalf?2**(lastExponent-24):0);
    const zeroThreshold=2**-25+(composedHalf?2**-49:0);
    const value=async()=>{await this.write("(");await operand();await this.write(")");};
    const linear=async(unit:number,negative:boolean,exponent:number)=>{
      await this.write(negative?"-(((":"(((");
      if(composedHalf){
        const unit32=2**(exponent-23),ratio=unit/unit32;
        await this.write("((");if(negative)await this.write("-");await value();
        await this.write(`/${rustF64(unit32)}+4503599627370496.0)-4503599627370496.0)/${rustF64(ratio)}`);
      }else{if(negative)await this.write("-");await value();await this.write(`/${rustF64(unit)}`);}
      await this.write(`+4503599627370496.0)-4503599627370496.0)*${rustF64(unit)})`);
    };
    await this.write("'rounding:{");
    if(!positiveRange){
      await this.write("if ");await value();await this.write("==0.0 {break 'rounding ");await value();await this.write(";}");
    }
    if(!positiveRange || positiveRange.maximum>=overflow){
      await this.write("if ");await value();await this.write(`>=${rustF64(overflow)} {break 'rounding f64::INFINITY;}`);
    }
    if(!positiveRange){
      await this.write("if ");await value();await this.write(`<=${rustF64(-overflow)} {break 'rounding f64::NEG_INFINITY;}`);
    }
    if(half){
      if(!positiveRange || positiveRange.minimum<=zeroThreshold){
        await this.write("if ");
        if(!positiveRange){await value();await this.write(">0.0 && ");}
        await value();await this.write(`<=${rustF64(zeroThreshold)} {break 'rounding 0.0;}`);
      }
      if(!positiveRange){
        await this.write("if ");await value();await this.write("<0.0 && ");await value();
        await this.write(`>=${rustF64(-zeroThreshold)} {break 'rounding -0.0;}`);
      }
    }
    for(let exponent=composedHalf?-25:firstExponent-1;exponent<=lastExponent;exponent++){
      const subnormal=!composedHalf&&exponent<firstExponent,lo=subnormal?0:2**exponent,hi=subnormal?2**firstExponent:2**(exponent+1);
      const unit=2**(Math.max(firstExponent,exponent)-bits);
      if(positiveRange && (positiveRange.maximum<lo||positiveRange.minimum>=hi)){this.eliminatedBranches++;continue;}
      const unconditional=positiveRange && positiveRange.maximum<hi;
      if(!unconditional){
        await this.write("if ");
        if(!positiveRange){await value();await this.write(`${subnormal?">":">="}${rustF64(lo)} && `);}
        await value();await this.write(`<${rustF64(hi)} {`);
      }
      await this.write("break 'rounding ");await linear(unit,false,exponent);await this.write(";");
      if(!unconditional)await this.write("}");else{await this.write("}");return;}
      if(!positiveRange){
        await this.write("if ");await value();await this.write(`${subnormal?"<":"<="}${rustF64(-lo)} && `);
        await value();await this.write(`>${rustF64(-hi)} {break 'rounding `);await linear(unit,true,exponent);await this.write(";}");
      }
    }
    await this.write('panic!("undefined numeric domain")}');
  }

}

export function rustF64(value: number | string): string {
  const text = String(value);
  return /[.eE]/.test(text) ? `${text}_f64` : `${text}.0_f64`;
}

/** Emit a dyadic boundary only when decimal parsing reconstructs its exact
 * binary64 value. This includes small IEEE preimages whose denominators are
 * larger than 2^53; converting a rational approximately is never allowed.
 */
function rustExactBoundary(value:Rational):string{
  const denominator=value.denominator;
  if((denominator&(denominator-1n))!==0n)throw new Error("Non-dyadic branch requires explicit IEEE boundary lowering");
  let numerator=value.numerator,exponent=-(denominator.toString(2).length-1);
  if(numerator===0n)return "0.0_f64";
  while((numerator&1n)===0n){numerator>>=1n;exponent++;}
  const result=Number(numerator)*2**exponent;
  if(!Number.isFinite(result))throw new Error("Branch boundary exceeds binary64 domain");
  const exact=exactNumberRational(result);
  if(exact.numerator!==value.numerator||exact.denominator!==value.denominator)
    throw new Error("Branch boundary requires explicit IEEE lowering");
  return rustF64(result);
}
