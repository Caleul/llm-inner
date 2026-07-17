import { mkdir, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { compareGemma4PagedTextLiteralGenerationTrace } from "./gemma4-paged-text-differential.js";

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
  const topK = value(argv, "--top-k", false);
  const maxAbsoluteError = value(argv, "--max-absolute-error", false);
  const maxRelativeError = value(argv, "--max-relative-error", false);
  if (!Number.isSafeInteger(maxReadMiB) || maxReadMiB <= 0 || (topK !== undefined && (!Number.isSafeInteger(Number(topK)) || Number(topK) <= 0)) ||
    [maxAbsoluteError, maxRelativeError].some((entry) => entry !== undefined && (!Number.isFinite(Number(entry)) || Number(entry) < 0))) throw new Error("Opções numéricas inválidas.");
  const report = resolve(value(argv, "--report")!);
  const result = await compareGemma4PagedTextLiteralGenerationTrace({
    artifact: resolve(value(argv, "--artifact")!), trace: resolve(value(argv, "--trace")!),
    maxReadBytes: maxReadMiB * 1024 * 1024,
    ...(topK === undefined ? {} : { topK: Number(topK) }),
    ...(maxAbsoluteError === undefined ? {} : { maxAbsoluteError: Number(maxAbsoluteError) }),
    ...(maxRelativeError === undefined ? {} : { maxRelativeError: Number(maxRelativeError) }),
    ...(value(argv, "--assert-source-unavailable", false) ? { assertSourceUnavailable: resolve(value(argv, "--assert-source-unavailable")!) } : {}),
    ...(argv.includes("--allow-unverified-fidelity") ? { allowUnverifiedFidelity: true } : {}),
  });
  await mkdir(dirname(report), { recursive: true });
  await writeFile(report, `${JSON.stringify({ kind: "gemma4-paged-text-native-generation-differential", sourceCheckpointAccessed: false, candidateFidelityAcknowledged: argv.includes("--allow-unverified-fidelity"), ...result }, null, 2)}\n`, "utf8");
  console.log(`Relatório diferencial Gemma 4 escrito em ${report} (${result.fidelityClass}).`);
}

main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
