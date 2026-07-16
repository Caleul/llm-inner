import { resolve } from "node:path";
import { captureTransformersLlamaTrace } from "./transformers-trace-capture.js";

function value(argv: string[], name: string, required = true): string | undefined {
  const index = argv.indexOf(name);
  if (index < 0) { if (required) throw new Error(`Valor ausente para ${name}.`); return undefined; }
  const result = argv[index + 1];
  if (!result || result.startsWith("--")) throw new Error(`Valor ausente para ${name}.`);
  return result;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const source = value(argv, "--source")!;
  const output = value(argv, "--output")!;
  const inputTokens = value(argv, "--input-tokens")!.split(",").map(Number);
  const positions = value(argv, "--position-ids", false)?.split(",").map(Number);
  const maxNewTokens = value(argv, "--max-new-tokens", false);
  const kind = await captureTransformersLlamaTrace({
    source: resolve(source), output: resolve(output), inputTokens,
    ...(positions ? { positionIds: positions } : {}),
    ...(maxNewTokens !== undefined ? { maxNewTokens: Number(maxNewTokens) } : {}),
    python: resolve(value(argv, "--python")!),
    model: value(argv, "--model")!, revisionOrChecksum: value(argv, "--revision")!,
  });
  console.log(`Trace Transformers ${kind} escrito em ${resolve(output)}.`);
}

main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
