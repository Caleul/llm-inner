import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { compareGemma4LiteralVisionTrace } from "./gemma4-literal-vision.js";
import { Gemma4TorchRuntimeReductionProvider } from "./gemma4-torch-runtime-reduction-provider.js";
import { gemma4RuntimeReductionReplayEvidence } from "./gemma4-runtime-reduction-provider.js";
import { readGemma4VisionDifferentialTrace } from "./gemma4-transformers-vision-trace.js";

function value(argv: string[], name: string, required = true): string | undefined { const index = argv.indexOf(name); if (index < 0) { if (required) throw new Error(`Valor ausente para ${name}.`); return undefined; } const result = argv[index + 1]; if (!result || result.startsWith("--")) throw new Error(`Valor ausente para ${name}.`); return result; }

async function main(): Promise<void> {
  const argv = process.argv.slice(2), maxTensorMiB = Number(value(argv, "--max-tensor-mib", false) ?? "64"), absolute = Number(value(argv, "--absolute-tolerance", false) ?? "0"), relative = Number(value(argv, "--relative-tolerance", false) ?? "0"), topK = Number(value(argv, "--top-k", false) ?? "10");
  if (!Number.isSafeInteger(maxTensorMiB) || maxTensorMiB <= 0 || !Number.isFinite(absolute) || absolute < 0 || !Number.isFinite(relative) || relative < 0 || !Number.isSafeInteger(topK) || topK <= 0) throw new Error("Opções diferenciais vision inválidas.");
  const report = resolve(value(argv, "--report")!), trace = await readGemma4VisionDifferentialTrace(resolve(value(argv, "--trace")!));
  const runtimeReductionPython = value(argv, "--runtime-reduction-python", false);
  const runtimeReductionProvider = runtimeReductionPython ? new Gemma4TorchRuntimeReductionProvider(runtimeReductionPython) : undefined;
  const result = await compareGemma4LiteralVisionTrace({ artifact: resolve(value(argv, "--artifact")!), trace, maxTensorBytes: maxTensorMiB * 1024 * 1024, maxAbsoluteError: absolute, maxRelativeError: relative, topK, ...(value(argv, "--assert-source-unavailable", false) ? { assertSourceUnavailable: resolve(value(argv, "--assert-source-unavailable")!) } : {}), ...(runtimeReductionProvider ? { runtimeReductionProvider } : {}) });
  await mkdir(dirname(report), { recursive: true });
  await writeFile(report, `${JSON.stringify({
    kind: "gemma4-embedded-literal-vision-differential", sourceCheckpointAccessed: false,
    invocation: trace.invocation, model: trace.source.model, revisionOrChecksum: trace.source.revisionOrChecksum,
    referenceRuntime: trace.reference.runtime,
    runtimeReductionProvider: runtimeReductionProvider?.contractId ?? null,
    runtimeReductionReplay: runtimeReductionProvider ? gemma4RuntimeReductionReplayEvidence(runtimeReductionProvider, 0) : null,
    traceEvidence: { captureId: trace.reference.captureId, sourceFiles: trace.source.files, runtimeReductionCoverage: trace.reference.runtimeReductionCoverage },
    pixelValues: { shape: trace.reference.pixelValues.shape, values: Array.from(trace.reference.pixelValues.values) },
    pixelPositionIds: trace.reference.pixelPositionIds, ...result,
  }, null, 2)}\n`, "utf8");
  console.log(`Relatório diferencial Gemma 4 ${trace.invocation} escrito em ${report} (${result.fidelityClass}).`);
}
main().catch((error: unknown) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
