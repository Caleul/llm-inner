import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { compareGemma4LiteralCompositeModalitySuite } from "./gemma4-literal-composite-modality-suite.js";
import { Gemma4TorchRuntimeReductionProvider } from "./gemma4-torch-runtime-reduction-provider.js";

function value(argv: string[], name: string): string {
  const index = argv.indexOf(name);
  const result = index < 0 ? undefined : argv[index + 1];
  if (!result || result.startsWith("--")) throw new Error(`Valor ausente para ${name}.`);
  return result;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  if (!argv.includes("--allow-unverified-fidelity")) {
    throw new Error("A suite requer --allow-unverified-fidelity enquanto as 100 BMM permanecem runtime-defined.");
  }
  const maxReadMiB = Number(argv.includes("--max-read-mib") ? value(argv, "--max-read-mib") : "16");
  const maxTowerTensorMiB = Number(argv.includes("--max-tower-tensor-mib") ? value(argv, "--max-tower-tensor-mib") : "64");
  const topK = Number(argv.includes("--top-k") ? value(argv, "--top-k") : "10");
  if (![maxReadMiB, maxTowerTensorMiB, topK].every((entry) => Number.isSafeInteger(entry) && entry > 0)) {
    throw new Error("Opções numéricas da suite composite são inválidas.");
  }
  const report = resolve(value(argv, "--report"));
  const artifact = resolve(value(argv, "--artifact"));
  const runtimeReductionProvider = argv.includes("--runtime-reduction-python")
    ? await Gemma4TorchRuntimeReductionProvider.fromArtifact(value(argv, "--runtime-reduction-python"), artifact)
    : undefined;
  const result = await compareGemma4LiteralCompositeModalitySuite({
    artifact,
    traces: {
      image: resolve(value(argv, "--image-trace")),
      video: resolve(value(argv, "--video-trace")),
      audio: resolve(value(argv, "--audio-trace")),
    },
    maxReadBytes: maxReadMiB * 1024 * 1024,
    maxTowerTensorBytes: maxTowerTensorMiB * 1024 * 1024,
    maxAbsoluteError: 0,
    maxRelativeError: 0,
    topK,
    assertSourceUnavailable: resolve(value(argv, "--assert-source-unavailable")),
    allowUnverifiedFidelity: true,
    ...(runtimeReductionProvider ? { runtimeReductionProvider } : {}),
  });
  await mkdir(dirname(report), { recursive: true });
  await writeFile(report, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  console.log(`Suite composite Gemma 4 escrita em ${report} (${result.summary.zeroToleranceGenerationPasses}/3 gerações, ${result.summary.runtimeDefinedReductions} BMM fail-closed).`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
