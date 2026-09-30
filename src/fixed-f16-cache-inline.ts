import type { FixedF16CachedScalarSource } from "./fixed-f16-parametric-formulas.js";
import { pruneFixedF16ScalarDeclarations, splitFixedF16Declarations } from "./fixed-f16-source-prune.js";

function expressionEnd(source: string, start: number): number {
  let braces = 0, parentheses = 0, brackets = 0, quote = "", escaped = false;
  for (let index = start; index < source.length; index++) {
    const character = source[index]!;
    if (quote) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === quote) quote = "";
      continue;
    }
    if (character === '"' || character === "'" || character === "`") { quote = character; continue; }
    if (character === "{") braces++;
    else if (character === "}") braces--;
    else if (character === "(") parentheses++;
    else if (character === ")") parentheses--;
    else if (character === "[") brackets++;
    else if (character === "]") brackets--;
    else if (character === ";" && braces === 0 && parentheses === 0 && brackets === 0) return index;
    if (braces < 0 || parentheses < 0 || brackets < 0) throw new Error("Fórmula de cache desbalanceada.");
  }
  throw new Error("Fórmula de cache incompleta.");
}

export function fixedF16ScalarCases(declaration: string): string[] {
  const marker = /switch\s*\(d\)\s*\{/.exec(declaration);
  if (!marker || marker.index === undefined) throw new Error("Seletor escalar ausente.");
  const formulas: string[] = [];
  let cursor = marker.index + marker[0].length;
  while (true) {
    const remainder = declaration.slice(cursor);
    if (/^\s*default\s*:/.test(remainder)) break;
    const match = /^\s*case\s+(\d+)\s*:\s*return\s*/.exec(remainder);
    if (!match) throw new Error("Caso escalar não suportado.");
    const dimension = Number(match[1]);
    if (dimension !== formulas.length) throw new Error("Dimensões escalares fora de ordem.");
    const start = cursor + match[0].length;
    const end = expressionEnd(declaration, start);
    formulas.push(declaration.slice(start, end).trim());
    cursor = end + 1;
  }
  return formulas;
}

/** Substitute the next constant-dimension scalar cache in requested outputs. */
export function inlineNextFixedF16ScalarCache(
  program: FixedF16CachedScalarSource,
): { source: FixedF16CachedScalarSource; inlined: string | null } {
  const ids = program.formulas.flatMap((formula) =>
    [...formula.matchAll(/\bscalar_cache_(\d+)\(t,\d+\)/g)].map((match) => Number(match[1])));
  if (ids.length === 0) return { source: program, inlined: null };
  const id = Math.max(...ids);
  const name = `scalar_cache_${id}`;
  const statement = splitFixedF16Declarations(program.declarations).find((item) =>
    new RegExp(`^const\\s+${name}\\s*=`).test(item));
  if (!statement) throw new Error(`Definição de ${name} ausente.`);
  const cases = fixedF16ScalarCases(statement);
  const call = new RegExp(`\\b${name}\\(t,(\\d+)\\)`, "g");
  const formulas = program.formulas.map((formula) => formula.replace(call, (_match, rawDimension: string) => {
    const replacement = cases[Number(rawDimension)];
    if (replacement === undefined) throw new Error(`Dimensão ${rawDimension} ausente em ${name}.`);
    return `(${replacement})`;
  }));
  return { source: pruneFixedF16ScalarDeclarations({ ...program, formulas }), inlined: name };
}
