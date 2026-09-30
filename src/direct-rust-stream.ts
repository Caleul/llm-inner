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
  /** All rounding arithmetic is emitted at the use site; no numeric function survives. */
  async round(kind: "f16" | "f32", input: RustExpression, alreadyF32 = false): Promise<void> {
    await this.write("{let input: f64 = ");
    if (kind === "f16" && !alreadyF32) await this.round("f32", input);
    else await input();
    await this.write(";assert!(input == input);let negative:bool = input < 0.0 || (input == 0.0 && 1.0/input < 0.0);let magnitude:f64 = if negative {-input} else {input};");
    const half = kind === "f16", min = half ? "0.00006103515625" : "1.1754943508222875e-38",
      unit = half ? "5.960464477539063e-8" : "1.401298464324817e-45", max = half ? "65504.0" : "3.4028234663852886e38";
    await this.write(`let mut base: f64 = ${min};let mut unit: f64 = ${unit};let mut exponent:i32 = 1;let normal:bool = magnitude >= base;`);
    for (const step of half ? [16, 8, 4, 2, 1] : [128, 64, 32, 16, 8, 4, 2, 1]) {
      await this.write(`if normal && exponent+${step}<=${half ? 30 : 254} && magnitude>=base*${rustF64(2 ** step)} {base*=${rustF64(2 ** step)};unit*=${rustF64(2 ** step)};exponent+=${step};}`);
    }
    await this.write("if !normal {base=0.0;}let mut lower:f64=base;let mut parity:bool=false;");
    // Mantissa decisions have one conditional level and no runtime loop.
    for (let bit = half ? 9 : 22; bit >= 0; bit--) {
      const step = 2 ** bit;
      await this.write(`if magnitude >= lower+unit*${step}.0 {lower+=unit*${step}.0;${bit === 0 ? "parity=true;" : ""}}`);
    }
    await this.write(`let midpoint:f64=lower+unit/2.0;if magnitude>midpoint || (magnitude==midpoint && parity) {lower+=unit;}if magnitude>=${max}+unit/2.0 {lower=f64::INFINITY;}if negative {-lower} else {lower}}`);
  }
}

export function rustF64(value: number | string): string {
  const text = String(value);
  return /[.eE]/.test(text) ? `${text}_f64` : `${text}.0_f64`;
}
