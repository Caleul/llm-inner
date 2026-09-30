import { createReadStream, createWriteStream } from "node:fs";
import { mkdtemp, rename, rm } from "node:fs/promises";
import { once } from "node:events";
import { finished } from "node:stream/promises";
import { dirname, join } from "node:path";
import { directF32ExpPrefix } from "./fixed-f16-direct-exp.js";
import { directSqrtPrefix } from "./fixed-f16-direct-sqrt.js";
import { directRoundPrefix } from "./fixed-f16-direct-rounding.js";
import { compileF16BitDecodeBranches } from "./fixed-f16-bit-decode-branches.js";
import { compileFixedF16RopeBranches, compileFixedF16RopePositionBranches } from "./fixed-f16-rope-branches.js";
import { f16BitsToDyadic, f32BitsToDyadic, roundDyadicToF16IfElse } from "./fixed-f16-projection.js";
import { sleefExpF32 } from "./sleef-f32.js";

/** Numeric lowering independent of any model IR, cache, or stage executor. */
export async function rewriteDirectNumericFile(
  inputPath: string, outputPath: string, operation: "exp" | "sqrt" | "round" | "decode" | "fold",
  chunkSize = 64 * 1024,
): Promise<{ replacements: number; bytes: number }> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const pattern = operation === "fold" ?
    /\b(f16Bits|f16|Math\.fround|Math\.exp|Math\.sqrt)\((-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)/g :
    operation === "exp" ? /\bMath\.exp\(/g : operation === "sqrt" ? /\bMath\.sqrt\(/g :
    operation === "decode" ? /\bf16\(/g : /\b(f16Bits|Math\.fround)\(/g;
  const substitute = (match: RegExpMatchArray): string => operation === "fold" ? foldNumericLiteral(match[1]!, match[2]!) :
    operation === "exp" ? directF32ExpPrefix() :
    operation === "sqrt" ? directSqrtPrefix() : operation === "decode" ?
      `((bits)=>{${compileF16BitDecodeBranches()}})(` :
      directRoundPrefix(match[1] as "f16Bits" | "Math.fround");
  return rewriteTextualCalls(inputPath, outputPath, pattern, substitute, chunkSize);
}

function foldNumericLiteral(kind: string, literal: string): string {
  const value = Number(literal);
  if (!Number.isFinite(value)) throw new RangeError(`Constante não finita: ${literal}`);
  if (kind === "Math.fround") return formatLiteral(Math.fround(value));
  if (kind === "Math.exp") return formatLiteral(sleefExpF32(value));
  if (kind === "Math.sqrt") return formatLiteral(Math.sqrt(value));
  if (kind === "f16") {
    if (!Number.isInteger(value) || value < 0 || value > 0xffff) throw new RangeError("Bits F16 inválidos");
    if (value === 0x8000) return "-0";
    const decoded = f16BitsToDyadic(value);
    return formatLiteral(Number(decoded.coefficient) * 2 ** decoded.exponent);
  }
  if (kind === "f16Bits") {
    const data = new DataView(new ArrayBuffer(4));
    data.setFloat32(0, value, true);
    const bits = data.getUint32(0, true);
    if ((bits & 0x7fffffff) === 0 && (bits & 0x80000000)) return "32768";
    return String(roundDyadicToF16IfElse(f32BitsToDyadic(bits)));
  }
  throw new Error(`Operação numérica desconhecida: ${kind}`);
}

function formatLiteral(value: number): string {
  if (!Number.isFinite(value)) throw new RangeError("Resultado não finito fora da política numérica");
  return Object.is(value, -0) ? "-0" : String(value);
}

export interface DirectNumericLoweringResult {
  passes: { name: string; replacements: number; bytes: number }[];
  unresolved: Record<string, number>;
}

/**
 * Lower numeric calls in a source expression without loading the expression as
 * a JavaScript string. Temporary files exist only during compilation.
 */
export async function lowerDirectNumericExpressionFile(
  inputPath: string, outputPath: string, maxSequenceLength: number,
): Promise<DirectNumericLoweringResult> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const temporary = await mkdtemp(join(dirname(outputPath), ".direct-numeric-"));
  const passes: DirectNumericLoweringResult["passes"] = [];
  let input = inputPath;
  try {
    for (const operation of ["fold", "exp", "sqrt", "rope", "round", "decode"] as const) {
      const output = join(temporary, operation);
      const result = operation === "rope" ?
        await rewriteDirectRopeFile(input, output, maxSequenceLength) :
        await rewriteDirectNumericFile(input, output, operation);
      passes.push({ name: operation, ...result });
      if (input !== inputPath) await rm(input);
      input = output;
    }
    const unresolved = await auditDirectNumericFile(input);
    if (Object.keys(unresolved).length !== 0) {
      throw new Error(`Operações numéricas não substituídas: ${JSON.stringify(unresolved)}`);
    }
    await rename(input, outputPath);
    return { passes, unresolved };
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

export async function auditDirectNumericFile(path: string): Promise<Record<string, number>> {
  const pattern = /\b(?:f16|f16Bits|ropeBits|Math\.exp|Math\.sqrt|Math\.fround)\(/g;
  const counts: Record<string, number> = {};
  let pending = "";
  const count = (source: string, final: boolean): void => {
    const limit = final ? source.length : Math.max(0, source.length - 256);
    for (const match of source.matchAll(pattern)) {
      if (match.index >= limit) break;
      const name = match[0]!.slice(0, -1);
      counts[name] = (counts[name] ?? 0) + 1;
    }
    pending = source.slice(limit);
  };
  for await (const chunk of createReadStream(path, { encoding: "utf8" })) count(pending + chunk, false);
  count(pending, true);
  return counts;
}

/** Inline position/dimension coefficients from the declared context and RoPE parameters. */
export async function rewriteDirectRopeFile(
  inputPath: string, outputPath: string, maxSequenceLength: number, chunkSize = 64 * 1024,
): Promise<{ replacements: number; bytes: number }> {
  const bodies = new Map<string, string>();
  const pattern = /\bropeBits\(([a-z][a-z0-9]*),([a-z][a-z0-9]*|\d+),(\d+),(\d+(?:\.\d+)?),([01])\)/g;
  return rewriteTextualCalls(inputPath, outputPath, pattern, (match) => {
    const position = match[1]!, dimension = match[2]!, rawHeadDim = match[3]!,
      rawTheta = match[4]!, rawSine = match[5]!;
    const key = `${rawHeadDim}/${rawTheta}/${rawSine}/${dimension}`;
    let body = bodies.get(key);
    if (body === undefined) {
      body = /^\d+$/.test(dimension) ?
        compileFixedF16RopePositionBranches(maxSequenceLength, Number(rawHeadDim), Number(rawTheta),
          Number(rawSine), Number(dimension)) :
        compileFixedF16RopeBranches(maxSequenceLength, Number(rawHeadDim), Number(rawTheta), Number(rawSine));
      bodies.set(key, body);
    }
    return /^\d+$/.test(dimension) ? `((pos)=>{${body}})(${position})` :
      `((pos,dim)=>{${body}})(${position},${dimension})`;
  }, chunkSize);
}

/** Generic bounded-window substitution for generated direct expressions. */
export async function rewriteTextualCalls(
  inputPath: string, outputPath: string, pattern: RegExp,
  substitute: (match: RegExpMatchArray) => string, chunkSize: number,
  prefix = "", suffix = "",
): Promise<{ replacements: number; bytes: number }> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const output = createWriteStream(outputPath, { encoding: "utf8" });
  let pending = "", replacements = 0, bytes = 0;
  const write = async (source: string): Promise<void> => {
    if (!source) return;
    bytes += Buffer.byteLength(source);
    if (!output.write(source)) await once(output, "drain");
  };
  const consume = async (source: string, final: boolean): Promise<void> => {
    const limit = final ? source.length : Math.max(0, source.length - 256);
    let cursor = 0, writableEnd = limit;
    for (const match of source.matchAll(pattern)) {
      if (match.index >= limit) break;
      if (!final && match.index + match[0].length > limit) {
        writableEnd = match.index;
        break;
      }
      await write(source.slice(cursor, match.index));
      await write(substitute(match));
      cursor = match.index + match[0].length;
      replacements++;
    }
    await write(source.slice(cursor, writableEnd));
    pending = source.slice(writableEnd);
  };
  try {
    await write(prefix);
    for await (const chunk of createReadStream(inputPath, { encoding: "utf8", highWaterMark: chunkSize })) {
      await consume(pending + chunk, false);
    }
    await consume(pending, true);
    await write(suffix);
    output.end();
    await finished(output);
    return { replacements, bytes };
  } catch (error) {
    output.destroy();
    throw error;
  }
}
