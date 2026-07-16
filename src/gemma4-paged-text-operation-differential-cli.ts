import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { compareGemma4PagedTextLiteralOperationCheckpoints } from "./gemma4-paged-text-operation-differential.js";

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
  if (!Number.isSafeInteger(maxReadMiB) || maxReadMiB <= 0 || (topK !== undefined && (!Number.isSafeInteger(Number(topK)) || Number(topK) <= 0))) throw new Error("Opções numéricas inválidas.");
  const report = resolve(value(argv, "--report")!);
  const result = await compareGemma4PagedTextLiteralOperationCheckpoints({
    artifact: resolve(value(argv, "--artifact")!), trace: resolve(value(argv, "--trace")!), maxReadBytes: maxReadMiB * 1024 * 1024,
    ...(topK === undefined ? {} : { topK: Number(topK) }),
    ...(value(argv, "--assert-source-unavailable", false) ? { assertSourceUnavailable: resolve(value(argv, "--assert-source-unavailable")!) } : {}),
  });
  await mkdir(dirname(report), { recursive: true });
  await writeFile(report, `${JSON.stringify({ kind: "gemma4-paged-text-native-operation-checkpoints", sourceCheckpointAccessed: false, ...result }, null, 2)}\n`, "utf8");
  console.log(`Relatório de checkpoints Gemma 4 escrito em ${report} (${result.fidelityClass}).`);
}

main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
