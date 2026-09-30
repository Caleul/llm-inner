import { createReadStream } from "node:fs";
import { mkdtemp, mkdir, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { compileFixedF16ScalarDimensionFromDirectory } from "./fixed-f16-ir-scalar-compiler.js";
import { rewriteFixedF16ContextFile, rewriteFixedF16MlpFile, rewriteFixedF16ProjectionFile,
  rewriteFixedF16RotatedFile, rewriteFixedF16ScoreFile,
  rewriteFixedF16NumericConstantsFile, rewriteFixedF16DecodeFile,
  rewriteFixedF16BitArithmeticFile, rewriteFixedF16RopeFile } from "./fixed-f16-hidden-inline.js";

export interface FixedF16StreamDiagnostic {
  outputDimension: number;
  maxSequenceLength: number;
  inputSize: number;
  stages: { name: string; replacements: number; bytes: number }[];
  remainingCalls: Record<string, number>;
  status: "numeric-lowering-and-full-parity-pending";
}

const remainingPattern = /\b(?:hidden|gate|up|context|score|qRot|kRot|q|k|v)_\d+\(|\b(?:f16Bits|f16|add16|mul16|neg16|ropeBits)\(|\bMath\.(?:exp|sqrt|fround)\(/g;

/** Count unresolved operators without reading the generated file into memory. */
export async function auditFixedF16StreamFile(path: string): Promise<Record<string, number>> {
  const counts: Record<string, number> = {};
  let pending = "";
  const inspect = (source: string, final: boolean): void => {
    const cutoff = final ? source.length : Math.max(0, source.length - 64);
    for (const match of source.matchAll(remainingPattern)) {
      if (match.index >= cutoff) break;
      const name = match[0]!.slice(0, -1).replace(/_\d+$/, "_*");
      counts[name] = (counts[name] ?? 0) + 1;
    }
    pending = source.slice(cutoff);
  };
  for await (const chunk of createReadStream(path, { encoding: "utf8" })) inspect(pending + chunk, false);
  inspect(pending, true);
  return counts;
}

/** Emit the backward-expanded dimension without materializing the growing formula in V8. */
export async function writeFixedF16StreamDiagnostic(
  checkpointDirectory: string, outputDimension: number, outputPath: string,
  options: { decodeF16?: boolean; lowerRope?: boolean } = {},
): Promise<FixedF16StreamDiagnostic> {
  const source = await compileFixedF16ScalarDimensionFromDirectory(checkpointDirectory, outputDimension);
  if (source.formulas.length !== 1 || source.maxSequenceLength === undefined) {
    throw new Error("Dimensão ou limite de contexto ausente.");
  }
  await mkdir(dirname(outputPath), { recursive: true });
  const temporary = await mkdtemp(join(dirname(outputPath), ".scalar-stream-"));
  const stages: FixedF16StreamDiagnostic["stages"] = [];
  try {
    let input = join(temporary, "input");
    await writeFile(input, source.formulas[0]!, "utf8");
    const passes = [
      ["mlp", rewriteFixedF16MlpFile], ["context", rewriteFixedF16ContextFile],
      ["score", rewriteFixedF16ScoreFile], ["rope", rewriteFixedF16RotatedFile],
      ["qkv", rewriteFixedF16ProjectionFile],
    ] as const;
    for (const [name, rewrite] of passes) {
      const next = join(temporary, name);
      const result = await rewrite(input, next, source);
      stages.push({ name, ...result });
      await rm(input);
      input = next;
    }
    const numeric = join(temporary, "numeric-constants");
    const folded = await rewriteFixedF16NumericConstantsFile(input, numeric);
    stages.push({ name: "numeric-constants", ...folded });
    await rm(input);
    input = numeric;
    if (options.lowerRope) {
      const rope = join(temporary, "rope-coefficients");
      const lowered = await rewriteFixedF16RopeFile(input, rope, source.maxSequenceLength);
      stages.push({ name: "rope-coefficients", ...lowered });
      await rm(input);
      input = rope;
    }
    const arithmetic = join(temporary, "bit-arithmetic");
    const embedded = await rewriteFixedF16BitArithmeticFile(input, arithmetic);
    stages.push({ name: "bit-arithmetic", ...embedded });
    await rm(input);
    input = arithmetic;
    if (options.decodeF16) {
      const decoded = join(temporary, "decode-f16");
      const lowered = await rewriteFixedF16DecodeFile(input, decoded);
      stages.push({ name: "decode-f16", ...lowered });
      await rm(input);
      input = decoded;
    }
    const remainingCalls = await auditFixedF16StreamFile(input);
    await rename(input, outputPath);
    return { outputDimension, maxSequenceLength: source.maxSequenceLength,
      inputSize: source.inputSize, stages, remainingCalls,
      status: "numeric-lowering-and-full-parity-pending" };
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}
