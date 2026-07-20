import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { compareGemma4LiteralCompositeTrace } from "./gemma4-literal-composite-differential.js";
import { Gemma4TorchRuntimeReductionProvider } from "./gemma4-torch-runtime-reduction-provider.js";

function value(argv: string[], name: string, required = true): string | undefined {
  const index = argv.indexOf(name);
  if (index < 0) { if (required) throw new Error(`Valor ausente para ${name}.`); return undefined; }
  const result = argv[index + 1];
  if (!result || result.startsWith("--")) throw new Error(`Valor ausente para ${name}.`);
  return result;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const maxReadMiB = Number(value(argv, "--max-read-mib", false) ?? "16");
  const maxTowerTensorMiB = Number(value(argv, "--max-tower-tensor-mib", false) ?? "64");
  const absolute = Number(value(argv, "--absolute-tolerance", false) ?? "0");
  const relative = Number(value(argv, "--relative-tolerance", false) ?? "0");
  const topK = Number(value(argv, "--top-k", false) ?? "10");
  if (![maxReadMiB, maxTowerTensorMiB, topK].every((entry) => Number.isSafeInteger(entry) && entry > 0) ||
    ![absolute, relative].every((entry) => Number.isFinite(entry) && entry >= 0)) throw new Error("Opções numéricas composite inválidas.");
  const report = resolve(value(argv, "--report")!);
  const artifact = resolve(value(argv, "--artifact")!);
  const runtimeReductionPython = value(argv, "--runtime-reduction-python", false);
  const runtimeReductionProvider = runtimeReductionPython
    ? await Gemma4TorchRuntimeReductionProvider.fromArtifact(runtimeReductionPython, artifact)
    : undefined;
  const comparison = await compareGemma4LiteralCompositeTrace({
    artifact,
    trace: resolve(value(argv, "--trace")!),
    maxReadBytes: maxReadMiB * 1024 * 1024,
    maxTowerTensorBytes: maxTowerTensorMiB * 1024 * 1024,
    maxAbsoluteError: absolute,
    maxRelativeError: relative,
    topK,
    ...(value(argv, "--assert-source-unavailable", false) ? { assertSourceUnavailable: resolve(value(argv, "--assert-source-unavailable")!) } : {}),
    ...(argv.includes("--allow-unverified-fidelity") ? { allowUnverifiedFidelity: true } : {}),
    ...(runtimeReductionProvider ? { runtimeReductionProvider } : {}),
  });
  await mkdir(dirname(report), { recursive: true });
  await writeFile(report, `${JSON.stringify({
    kind: "gemma4-embedded-literal-composite-generation-differential",
    sourceCheckpointAccessed: false,
    candidateFidelityAcknowledged: argv.includes("--allow-unverified-fidelity"),
    runtimeReductionProvider: runtimeReductionProvider?.contractId ?? null,
    ...comparison,
  }, null, 2)}\n`, "utf8");
  console.log(`Relatório composite Gemma 4 escrito em ${report} (prefill=${comparison.prefill.fidelityClass}, generation=${comparison.generation.fidelityClass}).`);
}

main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
