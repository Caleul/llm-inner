import { resolve } from "node:path";
import { runExecutionTraceComparison } from "./trace-runner.js";

function value(argv: string[], name: string, required = true): string | undefined {
  const index = argv.indexOf(name);
  if (index < 0) {
    if (required) throw new Error(`Valor ausente para ${name}.`);
    return undefined;
  }
  const result = argv[index + 1];
  if (!result || result.startsWith("--")) throw new Error(`Valor ausente para ${name}.`);
  return result;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const source = value(argv, "--source")!;
  const trace = value(argv, "--trace")!;
  const report = value(argv, "--report")!;
  const topK = value(argv, "--top-k", false);
  const maxAbsoluteError = value(argv, "--max-absolute-error", false);
  const maxRelativeError = value(argv, "--max-relative-error", false);
  const parsed = [topK, maxAbsoluteError, maxRelativeError].map((entry) => entry === undefined ? undefined : Number(entry));
  if ((parsed[0] !== undefined && (!Number.isInteger(parsed[0]) || parsed[0]! <= 0)) || parsed.slice(1).some((entry) => entry !== undefined && (!Number.isFinite(entry) || entry < 0))) {
    throw new Error("Tolerâncias ou --top-k inválidos.");
  }
  const result = await runExecutionTraceComparison({
    source: resolve(source), trace: resolve(trace), report: resolve(report),
    ...(parsed[0] !== undefined ? { topK: parsed[0] } : {}),
    ...(parsed[1] !== undefined ? { maxAbsoluteError: parsed[1] } : {}),
    ...(parsed[2] !== undefined ? { maxRelativeError: parsed[2] } : {}),
  });
  console.log(`Relatório diferencial escrito em ${resolve(report)} (${result.fidelityClass}).`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
