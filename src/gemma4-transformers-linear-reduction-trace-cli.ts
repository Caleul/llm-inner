import { resolve } from "node:path";
import { captureGemma4TransformersLinearReductionTrace } from "./gemma4-transformers-operation-trace.js";

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
  const operationId = value(argv, "--operation-id")!;
  const output = resolve(value(argv, "--output")!);
  const activationScale = integer(value(argv, "--activation-scale", false) ?? "1", "--activation-scale");
  const activationBf16ScaleBits = optionalInteger(value(argv, "--activation-bf16-scale-bits", false), "--activation-bf16-scale-bits");
  await captureGemma4TransformersLinearReductionTrace({
    source: resolve(value(argv, "--source")!), output, operationId, inputTokens,
    activationScale,
    ...(activationBf16ScaleBits === undefined ? {} : { activationBf16ScaleBits }),
    ...(positionIds ? { positionIds } : {}), python: resolve(value(argv, "--python")!),
    model: value(argv, "--model")!, revisionOrChecksum: value(argv, "--revision")!, executionDevice: executionDevice(argv),
  });
  console.log(`Checkpoint linear nativo Gemma 4 escrito em ${output}.`);
}

function integer(value: string, flag: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`${flag} requer inteiro seguro.`);
  return parsed;
}

function optionalInteger(value: string | undefined, flag: string): number | undefined {
  return value === undefined ? undefined : integer(value, flag);
}

main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
