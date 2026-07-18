import { resolve } from "node:path";
import { captureGemma4TransformersAudioTrace } from "./gemma4-transformers-audio-trace.js";

function value(argv: string[], name: string, required = true): string | undefined {
  const index = argv.indexOf(name);
  if (index < 0) { if (required) throw new Error(`Valor ausente para ${name}.`); return undefined; }
  const result = argv[index + 1];
  if (!result || result.startsWith("--")) throw new Error(`Valor ausente para ${name}.`);
  return result;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const frames = Number(value(argv, "--frames", false) ?? "1");
  const start = Number(value(argv, "--feature-start", false) ?? "-0.25");
  const end = Number(value(argv, "--feature-end", false) ?? "0.25");
  if (!Number.isSafeInteger(frames) || frames <= 0 || !Number.isFinite(start) || !Number.isFinite(end)) throw new Error("Faixa de input_features inválida.");
  const count = frames * 128;
  const values = Float32Array.from({ length: count }, (_, index) => Math.fround(start + (end - start) * index / Math.max(count - 1, 1)));
  const output = resolve(value(argv, "--output")!);
  await captureGemma4TransformersAudioTrace({
    source: resolve(value(argv, "--source")!),
    output,
    inputFeatures: { shape: [1, frames, 128], values },
    inputFeaturesMask: [Array.from({ length: frames }, () => true)],
    python: resolve(value(argv, "--python")!),
    model: value(argv, "--model")!,
    revisionOrChecksum: value(argv, "--revision")!,
    executionDevice: "cpu",
  });
  console.log(`Checkpoints nativos de áudio Gemma 4 escritos em ${output}.`);
}

main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
