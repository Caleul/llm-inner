import { createReadStream, createWriteStream } from "node:fs";
import { once } from "node:events";
import { finished } from "node:stream/promises";
import { compileF16SiluBranches } from "./fixed-f16-conditional-linear.js";
import { fixedF16ScalarCases } from "./fixed-f16-cache-inline.js";
import { splitFixedF16Declarations } from "./fixed-f16-source-prune.js";
import type { FixedF16CachedScalarSource } from "./fixed-f16-parametric-formulas.js";
import { f32BitsToDyadic, roundDyadicToF16IfElse } from "./fixed-f16-projection.js";
import { compileF16BitDecodeBranches } from "./fixed-f16-bit-decode-branches.js";
import { compileFixedF16RopeBranches } from "./fixed-f16-rope-branches.js";

const hiddenCall = /\bhidden_(\d+)\(t,(\d+)\)/g;
const contextCall = /\bcontext_(\d+)\(t,(\d+)\)/g;
const scoreCall = /\bscore_(\d+)\(p,j,h\)/g;
const rotatedCall = /\b(qRot|kRot)_(\d+)\((p|j),(h|Math\.floor\(h\/\d+\)),d\)/g;
const projectionPrefix = /\b(q|k|v)_(\d+)\(/g;
const constantNumericCall = /\b(f16Bits|Math\.fround)\((-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)/g;
const f16Prefix = /\bf16\(/g;
const bitArithmeticPrefix = /\b(add16|mul16|neg16)\(/g;
const ropeCall = /\bropeBits\((p|j),d,(\d+),(\d+(?:\.\d+)?),([01])\)/g;
const carryLength = 256;
let cachedSiluBody: string | undefined;

/** Substitute one generated MLP activation, preserving its F16 boundaries. */
export function inlineFixedF16HiddenCall(layerId: string, dimension: string): string {
  const body = cachedSiluBody ??= (() => {
    const silu = compileF16SiluBranches().source;
    return silu.slice("function(bits){".length, -1);
  })();
  return `(()=>{const g=gate_${layerId}(t,${dimension}),u=up_${layerId}(t,${dimension});` +
    `const activated=(()=>{const bits=g;${body}})();` +
    `return f16Bits(f16(activated)*f16(u));})()`;
}

/** Resolve gate/up selector cases before emitting the MLP branch for one coordinate. */
export function fixedF16MlpHiddenReplacer(program: FixedF16CachedScalarSource):
  (layerId: string, dimension: string) => string {
  const definitions = new Map<string, string[]>();
  for (const statement of splitFixedF16Declarations(program.declarations)) {
    const match = /^const\s+((?:gate|up)_\d+)\s*=/.exec(statement);
    if (match) definitions.set(match[1]!, fixedF16ScalarCases(statement));
  }
  return (layerId, dimension) => {
    const selected = (name: string): string => {
      const formula = definitions.get(`${name}_${layerId}`)?.[Number(dimension)];
      if (formula === undefined) throw new Error(`${name}_${layerId}(t,${dimension}): fórmula ausente`);
      return formula;
    };
    const gate = selected("gate"), up = selected("up");
    const body = cachedSiluBody ??= (() => {
      const silu = compileF16SiluBranches().source;
      return silu.slice("function(bits){".length, -1);
    })();
    return `(()=>{const g=(${gate}),u=(${up});` +
      `const activated=(()=>{const bits=g;${body}})();` +
      `return f16Bits(f16(activated)*f16(u));})()`;
  };
}

/** Embed the causal attention body at one literal output coordinate. */
export function fixedF16ContextReplacer(program: FixedF16CachedScalarSource):
  (attentionId: string, dimension: string) => string {
  const bodies = new Map<string, string>();
  for (const statement of splitFixedF16Declarations(program.declarations)) {
    const match = /^const\s+(context_\d+)\s*=/.exec(statement);
    if (!match) continue;
    const start = statement.indexOf("const h=Math.floor(coordinate/");
    const end = statement.indexOf("const value=f16Bits(acc);", start);
    if (start < 0 || end < 0) throw new Error(`${match[1]}: corpo da atenção não reconhecido`);
    bodies.set(match[1]!, statement.slice(start, end));
  }
  return (attentionId, dimension) => {
    const body = bodies.get(`context_${attentionId}`);
    if (!body) throw new Error(`context_${attentionId}: corpo ausente`);
    return `(()=>{const p=t,coordinate=${dimension};${body}return f16Bits(acc);})()`;
  };
}

/** Embed the ordered Q·K reduction and score scaling into the attention expression. */
export function fixedF16ScoreReplacer(program: FixedF16CachedScalarSource):
  (attentionId: string, unused: string) => string {
  const bodies = new Map<string, string>();
  for (const statement of splitFixedF16Declarations(program.declarations)) {
    const match = /^const\s+(score_\d+)\s*=/.exec(statement);
    if (!match) continue;
    const start = statement.indexOf("let total=Math.fround(0);");
    const end = statement.indexOf("const value=mul16(", start);
    const tail = statement.indexOf(";", end);
    if (start < 0 || end < 0 || tail < 0) throw new Error(`${match[1]}: redução QK não reconhecida`);
    const expression = statement.slice(end + "const value=".length, tail);
    bodies.set(match[1]!, `(()=>{${statement.slice(start, end)}return ${expression};})()`);
  }
  return (attentionId) => {
    const body = bodies.get(`score_${attentionId}`);
    if (!body) throw new Error(`score_${attentionId}: corpo ausente`);
    return body;
  };
}

/** Substitute RoPE coordinate arithmetic while retaining its exact F16 operations. */
export function fixedF16RotatedReplacer(program: FixedF16CachedScalarSource):
  (kind: string, attentionId: string, position: string, head: string) => string {
  const bodies = new Map<string, string>();
  for (const statement of splitFixedF16Declarations(program.declarations)) {
    const match = /^const\s+((?:qRot|kRot)_\d+)\s*=/.exec(statement);
    if (!match) continue;
    const start = statement.indexOf("const partner=");
    const end = statement.lastIndexOf("};");
    if (start < 0 || end < start) throw new Error(`${match[1]}: RoPE não reconhecido`);
    bodies.set(match[1]!, statement.slice(start, end));
  }
  return (kind, attentionId, position, head) => {
    const body = bodies.get(`${kind}_${attentionId}`);
    if (!body) throw new Error(`${kind}_${attentionId}: corpo ausente`);
    return `((p,h,d)=>{${body}})(${position},${head},d)`;
  };
}

/** Inline a learned projection selector; all case leaves contain literal weights. */
export function fixedF16ProjectionPrefixReplacer(program: FixedF16CachedScalarSource):
  (kind: string, attentionId: string) => string {
  const cases = new Map<string, string[]>();
  for (const statement of splitFixedF16Declarations(program.declarations)) {
    const match = /^const\s+((?:q|k|v)_\d+)\s*=/.exec(statement);
    if (match) cases.set(match[1]!, fixedF16ScalarCases(statement));
  }
  return (kind, attentionId) => {
    const selected = cases.get(`${kind}_${attentionId}`);
    if (!selected) throw new Error(`${kind}_${attentionId}: projeção ausente`);
    return `((p,d)=>{const t=p;switch(d){${selected.map((formula, dimension) =>
      `case ${dimension}:return ${formula};`).join("")}default:throw new RangeError("Dimensão inválida");}})(`;
  };
}

/** Fold numeric literals without moving any input-dependent rounding boundary. */
export function foldFixedF16ConstantCall(kind: string, rawValue: string): string {
  const value = Number(rawValue);
  if (!Number.isFinite(value)) throw new Error(`Constante numérica inválida: ${rawValue}`);
  if (kind === "Math.fround") {
    const result = Math.fround(value);
    return Object.is(result, -0) ? "-0" : String(result);
  }
  if (kind === "f16Bits") {
    const buffer = new DataView(new ArrayBuffer(4));
    buffer.setFloat32(0, value, true);
    return String(roundDyadicToF16IfElse(f32BitsToDyadic(buffer.getUint32(0, true))));
  }
  throw new Error(`Operação constante não suportada: ${kind}`);
}

/** Materialize the requested output expression without holding its expansion in a JS string. */
export async function writeFixedF16HiddenExpandedFormula(formula: string, path: string): Promise<{
  replacements: number; bytes: number;
}> {
  return rewriteFixedF16HiddenChunks((async function* () { yield formula; })(), path);
}

/** File-to-file pass: the expanded expression never becomes a JS string. */
export async function rewriteFixedF16HiddenFile(inputPath: string, outputPath: string, chunkSize = 64 * 1024): Promise<{
  replacements: number; bytes: number;
}> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const input = createReadStream(inputPath, { encoding: "utf8", highWaterMark: chunkSize });
  return rewriteFixedF16HiddenChunks(input, outputPath);
}

/** Fuse gate/up literal projections while replacing hidden calls in a large file. */
export async function rewriteFixedF16MlpFile(
  inputPath: string, outputPath: string, program: FixedF16CachedScalarSource, chunkSize = 64 * 1024,
): Promise<{ replacements: number; bytes: number }> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const input = createReadStream(inputPath, { encoding: "utf8", highWaterMark: chunkSize });
  return rewriteFixedF16Chunks(input, outputPath, hiddenCall, fixedF16MlpHiddenReplacer(program));
}

export async function rewriteFixedF16ContextFile(
  inputPath: string, outputPath: string, program: FixedF16CachedScalarSource, chunkSize = 64 * 1024,
): Promise<{ replacements: number; bytes: number }> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const input = createReadStream(inputPath, { encoding: "utf8", highWaterMark: chunkSize });
  return rewriteFixedF16Chunks(input, outputPath, contextCall, fixedF16ContextReplacer(program));
}

export async function rewriteFixedF16ScoreFile(
  inputPath: string, outputPath: string, program: FixedF16CachedScalarSource, chunkSize = 64 * 1024,
): Promise<{ replacements: number; bytes: number }> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const input = createReadStream(inputPath, { encoding: "utf8", highWaterMark: chunkSize });
  return rewriteFixedF16Chunks(input, outputPath, scoreCall, fixedF16ScoreReplacer(program));
}

export async function rewriteFixedF16RotatedFile(
  inputPath: string, outputPath: string, program: FixedF16CachedScalarSource, chunkSize = 64 * 1024,
): Promise<{ replacements: number; bytes: number }> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const input = createReadStream(inputPath, { encoding: "utf8", highWaterMark: chunkSize });
  const replace = fixedF16RotatedReplacer(program);
  return rewriteFixedF16Chunks(input, outputPath, rotatedCall, (kind, id, position, head) =>
    replace(kind, id, position, head));
}

export async function rewriteFixedF16ProjectionFile(
  inputPath: string, outputPath: string, program: FixedF16CachedScalarSource, chunkSize = 64 * 1024,
): Promise<{ replacements: number; bytes: number }> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const input = createReadStream(inputPath, { encoding: "utf8", highWaterMark: chunkSize });
  return rewriteFixedF16Chunks(input, outputPath, projectionPrefix, fixedF16ProjectionPrefixReplacer(program));
}

export async function rewriteFixedF16NumericConstantsFile(
  inputPath: string, outputPath: string, chunkSize = 64 * 1024,
): Promise<{ replacements: number; bytes: number }> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const input = createReadStream(inputPath, { encoding: "utf8", highWaterMark: chunkSize });
  return rewriteFixedF16Chunks(input, outputPath, constantNumericCall, foldFixedF16ConstantCall);
}

/** Replace each finite F16 decode by an in-place conditional affine expression. */
export async function rewriteFixedF16DecodeFile(
  inputPath: string, outputPath: string, chunkSize = 64 * 1024,
): Promise<{ replacements: number; bytes: number }> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const input = createReadStream(inputPath, { encoding: "utf8", highWaterMark: chunkSize });
  const prefix = `((bits)=>{${compileF16BitDecodeBranches()}})(`;
  return rewriteFixedF16Chunks(input, outputPath, f16Prefix, () => prefix);
}

/** Embed F16 arithmetic bodies before lowering their decode and rounding sites. */
export function fixedF16BitArithmeticPrefix(kind: string): string {
  if (kind === "add16") return "((left,right)=>f16Bits(f16(left)+f16(right)))(";
  if (kind === "mul16") return "((left,right)=>f16Bits(f16(left)*f16(right)))(";
  if (kind === "neg16") return "((bits)=>bits<32768?bits+32768:bits-32768)(";
  throw new Error(`Aritmética F16 desconhecida: ${kind}`);
}

export async function rewriteFixedF16BitArithmeticFile(
  inputPath: string, outputPath: string, chunkSize = 64 * 1024,
): Promise<{ replacements: number; bytes: number }> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const input = createReadStream(inputPath, { encoding: "utf8", highWaterMark: chunkSize });
  return rewriteFixedF16Chunks(input, outputPath, bitArithmeticPrefix, fixedF16BitArithmeticPrefix);
}

/** Replace RoPE coefficient calls by position/dimension branches in the compiled context. */
export function fixedF16RopeReplacer(maxSequenceLength: number):
  (position: string, headDim: string, theta: string, sine: string) => string {
  const bodies = new Map<string, string>();
  return (position, headDim, theta, sine) => {
    const key = `${headDim}/${theta}/${sine}`;
    let body = bodies.get(key);
    if (!body) {
      body = compileFixedF16RopeBranches(maxSequenceLength, Number(headDim), Number(theta), Number(sine));
      bodies.set(key, body);
    }
    return `((pos,dim)=>{${body}})(${position},d)`;
  };
}

export async function rewriteFixedF16RopeFile(
  inputPath: string, outputPath: string, maxSequenceLength: number, chunkSize = 64 * 1024,
): Promise<{ replacements: number; bytes: number }> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  const input = createReadStream(inputPath, { encoding: "utf8", highWaterMark: chunkSize });
  return rewriteFixedF16Chunks(input, outputPath, ropeCall, fixedF16RopeReplacer(maxSequenceLength));
}

async function rewriteFixedF16HiddenChunks(
  chunks: AsyncIterable<string>, path: string,
  replacement: (layerId: string, dimension: string) => string = inlineFixedF16HiddenCall,
): Promise<{
  replacements: number; bytes: number;
}> {
  return rewriteFixedF16Chunks(chunks, path, hiddenCall, replacement);
}

async function rewriteFixedF16Chunks(
  chunks: AsyncIterable<string>, path: string, call: RegExp,
  replacement: (...captures: string[]) => string,
): Promise<{ replacements: number; bytes: number }> {
  const output = createWriteStream(path, { encoding: "utf8" });
  let buffer = "";
  const write = async (chunk: string): Promise<void> => {
    if (buffer.length + chunk.length < 1_000_000) { buffer += chunk; return; }
    if (!output.write(buffer + chunk)) await once(output, "drain");
    buffer = "";
  };
  let pending = "", replacements = 0, bytes = 0;
  const emit = async (source: string, final: boolean): Promise<void> => {
    const safeEnd = final ? source.length : Math.max(0, source.length - carryLength);
    let cursor = 0, writableEnd = safeEnd;
    call.lastIndex = 0;
    for (const match of source.matchAll(call)) {
      if (match.index >= safeEnd) break;
      if (!final && match.index + match[0].length > safeEnd) {
        writableEnd = match.index;
        break;
      }
      const prefix = source.slice(cursor, match.index);
      const expanded = replacement(...match.slice(1).map((capture) => capture ?? ""));
      await write(prefix);
      await write(expanded);
      bytes += Buffer.byteLength(prefix) + Buffer.byteLength(expanded);
      replacements++;
      cursor = match.index + match[0].length;
    }
    const suffix = source.slice(cursor, writableEnd);
    await write(suffix);
    bytes += Buffer.byteLength(suffix);
    pending = source.slice(writableEnd);
  };
  try {
    for await (const chunk of chunks) {
      await emit(pending + chunk, false);
    }
    await emit(pending, true);
    if (buffer && !output.write(buffer)) await once(output, "drain");
    output.end();
    await finished(output);
    return { replacements, bytes };
  } catch (error) {
    output.destroy();
    throw error;
  }
}
