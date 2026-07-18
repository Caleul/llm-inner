import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { compareGemma4LiteralAudioTrace } from "./gemma4-literal-audio.js";
import { readGemma4AudioDifferentialTrace } from "./gemma4-transformers-audio-trace.js";

function value(argv: string[], name: string, required = true): string | undefined {
  const index = argv.indexOf(name);
  if (index < 0) { if (required) throw new Error(`Valor ausente para ${name}.`); return undefined; }
  const result = argv[index + 1];
  if (!result || result.startsWith("--")) throw new Error(`Valor ausente para ${name}.`);
  return result;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const maxTensorMiB = Number(value(argv, "--max-tensor-mib", false) ?? "16");
  const absolute = Number(value(argv, "--absolute-tolerance", false) ?? "0");
  const relative = Number(value(argv, "--relative-tolerance", false) ?? "0");
  const topK = Number(value(argv, "--top-k", false) ?? "10");
  if (!Number.isSafeInteger(maxTensorMiB) || maxTensorMiB <= 0 || !Number.isFinite(absolute) || absolute < 0 ||
    !Number.isFinite(relative) || relative < 0 || !Number.isSafeInteger(topK) || topK <= 0) throw new Error("Opções diferenciais de áudio inválidas.");
  const report = resolve(value(argv, "--report")!);
  const trace = await readGemma4AudioDifferentialTrace(resolve(value(argv, "--trace")!));
  const result = await compareGemma4LiteralAudioTrace({
    artifact: resolve(value(argv, "--artifact")!),
    trace,
    maxTensorBytes: maxTensorMiB * 1024 * 1024,
    maxAbsoluteError: absolute,
    maxRelativeError: relative,
    topK,
    ...(value(argv, "--assert-source-unavailable", false) ? { assertSourceUnavailable: resolve(value(argv, "--assert-source-unavailable")!) } : {}),
  });
  await mkdir(dirname(report), { recursive: true });
  await writeFile(report, `${JSON.stringify({
    kind: "gemma4-embedded-literal-audio-differential",
    sourceCheckpointAccessed: false,
    model: trace.source.model,
    revisionOrChecksum: trace.source.revisionOrChecksum,
    referenceRuntime: trace.reference.runtime,
    inputFeatures: { shape: trace.reference.inputFeatures.shape, values: Array.from(trace.reference.inputFeatures.values) },
    inputFeaturesMask: trace.reference.inputFeaturesMask,
    ...result,
  }, null, 2)}\n`, "utf8");
  console.log(`Relatório diferencial de áudio Gemma 4 escrito em ${report} (${result.fidelityClass}).`);
}

main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
