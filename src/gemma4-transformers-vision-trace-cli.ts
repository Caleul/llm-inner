import { resolve } from "node:path";
import { captureGemma4TransformersVisionTrace } from "./gemma4-transformers-vision-trace.js";
import type { Gemma4VisionInvocation } from "./gemma4-literal-vision.js";

function value(argv: string[], name: string, required = true): string | undefined { const index = argv.indexOf(name); if (index < 0) { if (required) throw new Error(`Valor ausente para ${name}.`); return undefined; } const result = argv[index + 1]; if (!result || result.startsWith("--")) throw new Error(`Valor ausente para ${name}.`); return result; }

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const invocation = (value(argv, "--invocation", false) ?? "image") as Gemma4VisionInvocation;
  const frames = Number(value(argv, "--frames", false) ?? (invocation === "video" ? "2" : "1"));
  const start = Number(value(argv, "--pixel-start", false) ?? "0");
  const end = Number(value(argv, "--pixel-end", false) ?? "1");
  if ((invocation !== "image" && invocation !== "video") || !Number.isSafeInteger(frames) || frames <= 0 || (invocation === "image" && frames !== 1) || !Number.isFinite(start) || !Number.isFinite(end)) throw new Error("Invocação/faixa vision inválida.");
  const patches = 9, width = 768, count = frames * patches * width;
  const values = Float32Array.from({ length: count }, (_, index) => Math.fround(start + (end - start) * index / Math.max(count - 1, 1)));
  const framePositions = (): number[][] => Array.from({ length: patches }, (_, index) => [index % 3, Math.floor(index / 3)]);
  const pixelPositionIds = invocation === "image" ? [framePositions()] : [Array.from({ length: frames }, framePositions)];
  await captureGemma4TransformersVisionTrace({
    source: resolve(value(argv, "--source")!), output: resolve(value(argv, "--output")!), invocation,
    pixelValues: { shape: invocation === "image" ? [1, patches, width] : [1, frames, patches, width], values },
    pixelPositionIds, python: resolve(value(argv, "--python")!), model: value(argv, "--model")!, revisionOrChecksum: value(argv, "--revision")!, executionDevice: "cpu",
  });
  console.log(`Checkpoints nativos Gemma 4 ${invocation} escritos.`);
}
main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
