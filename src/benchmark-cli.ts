import { mkdir, writeFile } from "node:fs/promises";
import * as path from "node:path";
import { runNativeFixtureBenchmarks } from "./benchmark.js";

function option(argv: string[], name: string, required = true): string | undefined {
  const index = argv.indexOf(name);
  if (index < 0) {
    if (required) throw new Error(`Valor ausente para ${name}.`);
    return undefined;
  }
  const value = argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`Valor ausente para ${name}.`);
  return value;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const output = path.resolve(option(argv, "--output")!);
  const samples = Number(option(argv, "--samples", false) ?? 5);
  const warmupSamples = Number(option(argv, "--warmup-samples", false) ?? 1);
  const report = await runNativeFixtureBenchmarks({ samples, warmupSamples });
  await mkdir(path.dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log(`Relatório de benchmark escrito em ${output}.`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
