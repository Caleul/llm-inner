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
  const modality = value(argv, "--modality", false) ?? "image";
  if (modality !== "image" && modality !== "video" && modality !== "audio") throw new Error("--modality requer image, video ou audio.");
  const modalityTokenId = Number(value(argv, "--modality-token-id", false) ??
    (modality === "image" ? value(argv, "--image-token-id", false) ?? "258880" : modality === "video" ? "258884" : "258881"));
  const textTokenId = Number(value(argv, "--text-token-id", false) ?? "2");
  const maxNewTokens = Number(value(argv, "--max-new-tokens", false) ?? "1");
  const start = Number(value(argv, modality === "audio" ? "--feature-start" : "--pixel-start", false) ?? (modality === "audio" ? "-0.25" : "0"));
  const end = Number(value(argv, modality === "audio" ? "--feature-end" : "--pixel-end", false) ?? (modality === "audio" ? "0.25" : "1"));
  if (![modalityTokenId, textTokenId, maxNewTokens].every(Number.isSafeInteger) || modalityTokenId < 0 || textTokenId < 0 || maxNewTokens < 0 || !Number.isFinite(start) || !Number.isFinite(end)) {
    throw new Error("IDs, maxNewTokens ou faixa de pixels inválidos.");
  }
  const patches = 9, width = 768, frames = Number(value(argv, "--frames", false) ?? "1");
  if (!Number.isSafeInteger(frames) || frames <= 0) throw new Error("--frames requer inteiro positivo.");
  const count = modality === "audio" ? frames * 128 : frames * patches * width;
  const values = Array.from({ length: count }, (_, index) => Math.fround(start + (end - start) * index / Math.max(count - 1, 1)));
  const positions = [Array.from({ length: patches }, (_, index) => [index % 3, Math.floor(index / 3)])];
  await captureGemma4TransformersCompositeTrace({
    source: resolve(value(argv, "--source")!),
    output: resolve(value(argv, "--output")!),
    modality,
    inputTokens: [modalityTokenId, textTokenId],
    positionIds: [0, 1],
    mmTokenTypeIds: [modality === "image" ? 1 : modality === "video" ? 2 : 0, 0],
    ...(modality === "image" ? { pixelValues: { shape: [1, patches, width], values }, imagePositionIds: positions } : {}),
    ...(modality === "video" ? {
      pixelValuesVideos: { shape: [1, frames, patches, width], values },
      videoPositionIds: [Array.from({ length: frames }, () => positions[0]!.map((position) => [...position]))],
    } : {}),
    ...(modality === "audio" ? {
      inputFeatures: { shape: [1, frames, 128], values },
      inputFeaturesMask: [Array.from({ length: frames }, () => true)],
    } : {}),
    maxNewTokens,
    python: resolve(value(argv, "--python")!),
    model: value(argv, "--model")!,
    revisionOrChecksum: value(argv, "--revision")!,
  });
  console.log(`Trace composite Gemma 4 escrito em ${resolve(value(argv, "--output")!)}.`);
}

main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
