import { createWriteStream } from "node:fs";
import { once } from "node:events";
import { finished } from "node:stream/promises";
import { DirectBranchDomain, type Comparison, type Rational, normalizeExactAffineComparison } from "./direct-branch-domain.js";

export type RustExpression = () => Promise<void>;
/** Writes direct source with bounded memory; expression callbacks emit immediately. */
export class DirectRustStream {
  private readonly output;
  bytes = 0;
  eliminatedBranches = 0;
  private reusableNumericScratch=false;
  constructor(path: string) { this.output = createWriteStream(path, { encoding: "utf8" }); }
  async write(source: string): Promise<void> {
    // Numeric fragments are complete lexemes at emission boundaries. Mark
    // floating literals explicitly to avoid millions of unresolved operator
    // obligations in Rust's single-function type checker. Authored strings
    // contain no floating literals; checkpoint data never supplies source text.
    source=source.replace(/(?<![A-Za-z0-9_])(?:\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![A-Za-z0-9_.])/g,"$&_f64");
    this.bytes += Buffer.byteLength(source);
    if (!this.output.write(source)) await once(this.output, "drain");
  }
  async close(): Promise<void> { this.output.end(); await finished(this.output); }
  destroy(): void { this.output.on("error", () => {}); this.output.destroy(); }
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
    const { numerator, denominator } = normalized.value;
    // Comparing a rounded decimal approximation would change the branch domain.
    // Emit exact integer ratios only within binary64's exact integer range.
    if (numerator > 9007199254740992n || numerator < -9007199254740992n || denominator > 9007199254740992n ||
      (denominator & (denominator - 1n)) !== 0n) throw new Error("Non-dyadic branch requires explicit IEEE boundary lowering");
    await this.write(`if ${variable} ${normalized.op} (${numerator}.0_f64/${denominator}.0_f64) {`);
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
  /** Each surviving leaf gets one conjunction of reduced bounds, no nested if. */
  async affineLeaf(domain:DirectBranchDomain,label:string,body:RustExpression):Promise<void>{
    if(!/^[a-z][a-z0-9_]*$/.test(label))throw new Error("Invalid Rust label");
    const conditions:string[]=[];
    for(const [variable,interval] of domain.entries()){
      if(!/^[a-z][a-z0-9_]*$/.test(variable))throw new Error("Invalid variable");
      for(const side of ["lower","upper"] as const){
        const bound=interval[side];if(!bound)continue;
        const {numerator,denominator}=bound.value;
        if(numerator>9007199254740992n||numerator< -9007199254740992n||denominator>9007199254740992n||
          (denominator&(denominator-1n))!==0n)throw new Error("Branch boundary requires explicit IEEE lowering");
        const comparison=side==="lower"?(bound.inclusive?">=":">"):(bound.inclusive?"<=":"<");
        conditions.push(`${variable}${comparison}(${numerator}.0_f64/${denominator}.0_f64)`);
      }
    }
    if(conditions.length)await this.write(`if ${conditions.join(" && ")} {`);
    await this.write(`break '${label} `);await body();await this.write(";");
    if(conditions.length)await this.write("}");
  }
  /** Reusable operator scratch is overwritten at each operation, never memoized
   * by coordinate or used to retain a computed activation between consumers.
   */
  async declareRoundingScratch():Promise<void>{
    await this.write("let mut round_input:f64;let mut round_negative:bool;let mut round_magnitude:f64;let mut round_base:f64;let mut round_unit:f64;let mut round_exponent:i32;let mut round_normal:bool;let mut round_lower:f64;let mut round_parity:bool;let mut round_midpoint:f64;");
    this.reusableNumericScratch=true;
  }
  private async writeRound(source:string):Promise<void>{
    if(this.reusableNumericScratch){
      source=source.replace(/\b(input|negative|magnitude|base|unit|exponent|normal|lower|parity|midpoint)\b/g,"round_$1");
      source=source.replace(/\blet(?: mut)? (round_\w+)(?::\s*(?:f64|bool|i32))?\s*=/g,"$1=");
    }
    await this.write(source);
  }
  /** All rounding arithmetic is emitted at the use site; no numeric function survives. */
  async round(kind: "f16" | "f32", input: RustExpression, alreadyF32 = false): Promise<void> {
    await this.writeRound("{let input: f64 = ");
    if (kind === "f16" && !alreadyF32) await this.round("f32", input);
    else await input();
    await this.writeRound(";assert!(input == input);let negative:bool = input < 0.0 || (input == 0.0 && 1.0/input < 0.0);let magnitude:f64 = if negative {-input} else {input};");
    const half = kind === "f16", min = half ? "0.00006103515625" : "1.1754943508222875e-38",
      unit = half ? "5.960464477539063e-8" : "1.401298464324817e-45", max = half ? "65504.0" : "3.4028234663852886e38";
    await this.writeRound(`let mut base: f64 = ${min};let mut unit: f64 = ${unit};let mut exponent:i32 = 1;let normal:bool = magnitude >= base;`);
    for (const step of half ? [16, 8, 4, 2, 1] : [128, 64, 32, 16, 8, 4, 2, 1]) {
      await this.writeRound(`if normal && exponent+${step}<=${half ? 30 : 254} && magnitude>=base*${rustF64(2 ** step)} {base*=${rustF64(2 ** step)};unit*=${rustF64(2 ** step)};exponent+=${step};}`);
    }
    await this.writeRound("if !normal {base=0.0;}let mut lower:f64=base;let mut parity:bool=false;");
    // Mantissa decisions have one conditional level and no runtime loop.
    for (let bit = half ? 9 : 22; bit >= 0; bit--) {
      const step = 2 ** bit;
      await this.writeRound(`if magnitude >= lower+unit*${step}.0 {lower+=unit*${step}.0;${bit === 0 ? "parity=true;" : ""}}`);
    }
    await this.writeRound(`let midpoint:f64=lower+unit/2.0;if magnitude>midpoint || (magnitude==midpoint && parity) {lower+=unit;}if magnitude>=${max}+unit/2.0 {lower=f64::INFINITY;}if negative {-lower} else {lower}}`);
  }
}

export function rustF64(value: number | string): string {
  const text = String(value);
  return /[.eE]/.test(text) ? `${text}_f64` : `${text}.0_f64`;
}
