import { resolve } from "node:path";
import { compileModel } from "./compiler.js";

interface CliArgs {
  source: string;
  output: string;
  equations?: string;
  maxFeatures: number;
  maxTerms: number;
  includeWeights: boolean;
}

function parseArgs(argv: string[]): CliArgs {
  const values = new Map<string, string>();
  const flags = new Set<string>();
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]!;
    if (!token.startsWith("--")) throw new Error(`Argumento inválido: ${token}`);
    if (token === "--include-weights") {
      flags.add(token);
      continue;
    }
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Valor ausente para ${token}`);
    values.set(token, value);
    index += 1;
  }
  const source = values.get("--source");
  const output = values.get("--output");
  if (!source || !output) {
    throw new Error(
      "Uso: --source <dir|model.gguf> --output <model.ir.json> [--equations out.txt] " +
        "[--max-features 10] [--max-terms 10] [--include-weights]",
    );
  }
  const maxFeatures = Number(values.get("--max-features") ?? 10);
  const maxTerms = Number(values.get("--max-terms") ?? 10);
  if (!Number.isInteger(maxFeatures) || maxFeatures <= 0) throw new Error("--max-features inválido.");
  if (!Number.isInteger(maxTerms) || maxTerms <= 0) throw new Error("--max-terms inválido.");
  return {
    source: resolve(source),
    output: resolve(output),
    ...(values.get("--equations") ? { equations: resolve(values.get("--equations")!) } : {}),
    maxFeatures,
    maxTerms,
    includeWeights: flags.has("--include-weights"),
  };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  await compileModel({
    source: args.source,
    output: args.output,
    ...(args.equations ? { equationsOutput: args.equations } : {}),
    preview: {
      outputRows: args.maxFeatures,
      inputTerms: args.maxTerms,
      includeWeights: args.includeWeights,
    },
  });
  console.log(`IR escrito em ${args.output}`);
  if (args.equations) console.log(`Equações escritas em ${args.equations}`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
