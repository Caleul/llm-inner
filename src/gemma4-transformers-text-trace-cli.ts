import { resolve } from "node:path";
import { captureGemma4TransformersTextGenerationTrace } from "./gemma4-transformers-text-trace.js";

function value(argv: string[], name: string, required = true): string | undefined {
  const index = argv.indexOf(name);
  if (index < 0) { if (required) throw new Error(`Valor ausente para ${name}.`); return undefined; }
  const result = argv[index + 1];
  if (!result || result.startsWith("--")) throw new Error(`Valor ausente para ${name}.`);
  return result;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const inputTokens = value(argv, "--input-tokens")!.split(",").map(Number);
  const positionIds = value(argv, "--position-ids", false)?.split(",").map(Number);
  const maxNewTokens = Number(value(argv, "--max-new-tokens")!);
  await captureGemma4TransformersTextGenerationTrace({
    source: resolve(value(argv, "--source")!), output: resolve(value(argv, "--output")!), inputTokens,
    ...(positionIds ? { positionIds } : {}), maxNewTokens, python: resolve(value(argv, "--python")!),
    model: value(argv, "--model")!, revisionOrChecksum: value(argv, "--revision")!,
  });
  console.log(`Trace nativo Gemma 4 escrito em ${resolve(value(argv, "--output")!)}.`);
}

main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
