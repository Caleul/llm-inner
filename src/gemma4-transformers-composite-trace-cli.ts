import { resolve } from "node:path";
import { captureGemma4TransformersCompositeTrace } from "./gemma4-transformers-composite-trace.js";

function value(argv: string[], name: string, required = true): string | undefined {
  const index = argv.indexOf(name);
  if (index < 0) { if (required) throw new Error(`Valor ausente para ${name}.`); return undefined; }
  const result = argv[index + 1];
  if (!result || result.startsWith("--")) throw new Error(`Valor ausente para ${name}.`);
  return result;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const imageTokenId = Number(value(argv, "--image-token-id", false) ?? "258880");
  const textTokenId = Number(value(argv, "--text-token-id", false) ?? "2");
  const maxNewTokens = Number(value(argv, "--max-new-tokens", false) ?? "1");
  const start = Number(value(argv, "--pixel-start", false) ?? "0");
  const end = Number(value(argv, "--pixel-end", false) ?? "1");
  if (![imageTokenId, textTokenId, maxNewTokens].every(Number.isSafeInteger) || imageTokenId < 0 || textTokenId < 0 || maxNewTokens < 0 || !Number.isFinite(start) || !Number.isFinite(end)) {
    throw new Error("IDs, maxNewTokens ou faixa de pixels inválidos.");
  }
  const patches = 9, width = 768, count = patches * width;
  const pixels = Array.from({ length: count }, (_, index) => Math.fround(start + (end - start) * index / Math.max(count - 1, 1)));
  const positions = [Array.from({ length: patches }, (_, index) => [index % 3, Math.floor(index / 3)])];
  await captureGemma4TransformersCompositeTrace({
    source: resolve(value(argv, "--source")!),
    output: resolve(value(argv, "--output")!),
    inputTokens: [imageTokenId, textTokenId],
    positionIds: [0, 1],
    mmTokenTypeIds: [1, 0],
    pixelValues: { shape: [1, patches, width], values: pixels },
    imagePositionIds: positions,
    maxNewTokens,
    python: resolve(value(argv, "--python")!),
    model: value(argv, "--model")!,
    revisionOrChecksum: value(argv, "--revision")!,
  });
  console.log(`Trace composite Gemma 4 escrito em ${resolve(value(argv, "--output")!)}.`);
}

main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
