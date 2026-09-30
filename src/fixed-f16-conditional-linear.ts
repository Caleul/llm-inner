import { readFileSync } from "node:fs";
import { f16BitsToDyadic, multiplyDyadic, roundDyadicToF16IfElse } from "./fixed-f16-projection.js";

export interface F16ConditionalLinearRule {
  source: string;
  intervals: number;
  characters: number;
}

/** Lower any fully specified unary F16 numeric rule to affine if/else leaves. */
export function compileFiniteF16UnaryBranches(evaluate: (bits: number) => number): F16ConditionalLinearRule {
  const intervals: Array<{ start: number; end: number; first: number; slope: number | undefined }> = [];
  for (let bits = 0; bits <= 0xffff; bits++) {
    if ((bits & 0x7c00) === 0x7c00) continue;
    const output = evaluate(bits);
    if (!Number.isInteger(output) || output < 0 || output > 0xffff) {
      throw new Error(`Regra F16 inválida para ${bits}.`);
    }
    const last = intervals[intervals.length - 1];
    if (last && last.end === bits - 1) {
      const previous = last.first + (last.slope ?? 0) * (last.end - last.start);
      const slope = output - previous;
      if (last.slope === undefined || last.slope === slope) {
        last.slope = slope;
        last.end = bits;
        continue;
      }
    }
    intervals.push({ start: bits, end: bits, first: output, slope: undefined });
  }
  const emit = (from: number, to: number): string => {
    if (from === to) {
      const interval = intervals[from]!;
      const slope = interval.slope ?? 0;
      const intercept = interval.first - slope * interval.start;
      const affine = slope === 0 ? `${intercept}` : slope === 1 ? `bits+${intercept}` :
        slope === -1 ? `-bits+${intercept}` : `${slope}*bits+${intercept}`;
      return `return ${affine};`;
    }
    const middle = Math.floor((from + to) / 2);
    return `if(bits<=${intervals[middle]!.end}){${emit(from, middle)}}else{${emit(middle + 1, to)}}`;
  };
  const source = `function(bits){if(bits<0||bits>65535||` +
    `(bits>=31744&&bits<32768)||bits>=64512)throw new RangeError("F16 não finito");` +
    `${emit(0, intervals.length - 1)}}`;
  return { source, intervals: intervals.length, characters: source.length };
}

/** Checkpoint-literal multiplication, rounded with the declared F16 rule. */
export function compileF16LiteralMultiplyBranches(weightBits: number): F16ConditionalLinearRule {
  if (!Number.isInteger(weightBits) || weightBits < 0 || weightBits > 0xffff ||
    (weightBits & 0x7c00) === 0x7c00) throw new Error("Peso F16 finito necessário.");
  const weight = f16BitsToDyadic(weightBits);
  return compileFiniteF16UnaryBranches((bits) =>
    roundDyadicToF16IfElse(multiplyDyadic(f16BitsToDyadic(bits), weight)));
}

/** Lower the captured PyTorch 2.12.1 CPU F16 SiLU profile, without a runtime exp. */
export function compileF16SiluBranches(): F16ConditionalLinearRule {
  const values = readFileSync(new URL("../../numeric-profiles/pytorch-2.12.1-cpu-f16-silu.bin", import.meta.url));
  if (values.length !== 65536 * 2) throw new Error("Perfil SiLU F16 incompleto.");
  return compileFiniteF16UnaryBranches((bits) => values.readUInt16LE(bits * 2));
}
