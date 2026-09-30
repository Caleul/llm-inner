import type { FixedF16CachedScalarSource } from "./fixed-f16-parametric-formulas.js";

/** Split only the top-level const declarations emitted by this compiler. */
export function splitFixedF16Declarations(source: string): string[] {
  const result: string[] = [];
  let start = 0, braces = 0, parentheses = 0, brackets = 0;
  let quote = "", escaped = false;
  for (let index = 0; index < source.length; index++) {
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
    else if (character === ";" && braces === 0 && parentheses === 0 && brackets === 0) {
      const statement = source.slice(start, index + 1).trim();
      if (statement) result.push(statement);
      start = index + 1;
    }
    if (braces < 0 || parentheses < 0 || brackets < 0) throw new Error("Declarações escalares desbalanceadas.");
  }
  if (source.slice(start).trim() || quote || braces || parentheses || brackets) {
    throw new Error("Declarações escalares incompletas.");
  }
  return result;
}

/** Remove generated scalar declarations that no requested output can reach. */
export function pruneFixedF16ScalarDeclarations(
  program: FixedF16CachedScalarSource,
): FixedF16CachedScalarSource {
  const statements = splitFixedF16Declarations(program.declarations);
  const owner = new Map<string, number>();
  statements.forEach((statement, index) => {
    const match = /^const\s+([A-Za-z_$][\w$]*)\s*=/.exec(statement);
    if (!match) throw new Error("Declaração escalar não suportada.");
    owner.set(match[1]!, index);
  });
  const references = (source: string): number[] => {
    const found = new Set<number>();
    for (const match of source.matchAll(/[A-Za-z_$][\w$]*/g)) {
      const index = owner.get(match[0]);
      if (index !== undefined) found.add(index);
    }
    return [...found];
  };
  const live = new Set<number>();
  const queue = program.formulas.flatMap(references);
  while (queue.length > 0) {
    const index = queue.pop()!;
    if (live.has(index)) continue;
    live.add(index);
    queue.push(...references(statements[index]!));
  }
  return { ...program, declarations: statements.filter((_statement, index) => live.has(index)).join("\n") + "\n" };
}
