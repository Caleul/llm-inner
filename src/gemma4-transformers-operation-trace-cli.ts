import { resolve } from "node:path";
import { captureGemma4TransformersOperationTrace } from "./gemma4-transformers-operation-trace.js";

function value(argv: string[], name: string, required = true): string | undefined {
  const index = argv.indexOf(name);
  if (index < 0) { if (required) throw new Error(`Valor ausente para ${name}.`); return undefined; }
  const result = argv[index + 1];
  if (!result || result.startsWith("--")) throw new Error(`Valor ausente para ${name}.`);
  return result;
}

function executionDevice(argv: string[]): "cpu" | "mps" {
  const device = value(argv, "--device");
  if (device === "cpu" || device === "mps") return device;
  throw new Error("--device requer cpu ou mps.");
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const inputTokens = value(argv, "--input-tokens")!.split(",").map(Number);
  const positionIds = value(argv, "--position-ids", false)?.split(",").map(Number);
  await captureGemma4TransformersOperationTrace({
    source: resolve(value(argv, "--source")!), output: resolve(value(argv, "--output")!), inputTokens,
    ...(positionIds ? { positionIds } : {}), python: resolve(value(argv, "--python")!),
    model: value(argv, "--model")!, revisionOrChecksum: value(argv, "--revision")!, executionDevice: executionDevice(argv),
  });
  console.log(`Checkpoints nativos Gemma 4 escritos em ${resolve(value(argv, "--output")!)}.`);
}

main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
