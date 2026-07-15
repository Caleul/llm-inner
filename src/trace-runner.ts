import { mkdir, writeFile } from "node:fs/promises";
import * as path from "node:path";
import { buildModelIR } from "./architecture.js";
import { openCatalog } from "./catalog.js";
import { compareExecutionTrace } from "./differential.js";
import { executeReferenceF32 } from "./executor.js";
import { materializeReferenceF32Constants } from "./materialize.js";
import { fingerprintIR, readExecutionTraceBundle, verifyTraceSource } from "./trace.js";
import type { DifferentialComparisonReport, ModelIR, Operation } from "./types.js";

export async function runExecutionTraceComparison(options: {
  source: string;
  trace: string;
  report: string;
  topK?: number;
  maxAbsoluteError?: number;
  maxRelativeError?: number;
}): Promise<DifferentialComparisonReport> {
  const decoded = await readExecutionTraceBundle(options.trace);
  const opened = await openCatalog(options.source, true);
  try {
    await verifyTraceSource(opened.catalog, decoded.bundle.source.files);
    const ir = await buildModelIR(opened.catalog, { outputRows: 1, inputTerms: 1, includeWeights: false }, opened.bridge);
    if (fingerprintIR(ir) !== decoded.bundle.irFingerprint) throw new Error("Fingerprint do IR diverge; trace foi capturado para outra semântica/adaptador.");
    applyF32Policy(ir);
    const tensors = await materializeReferenceF32Constants(ir, opened.catalog, opened.reader, opened.bridge);
    const candidate = executeReferenceF32(ir, { inputIds: decoded.reference.inputTokens, tensors });
    const report = compareExecutionTrace(ir, candidate, decoded.reference, {
      candidateRuntime: decoded.bundle.candidatePolicy.runtime,
      ...(options.topK !== undefined ? { topK: options.topK } : {}),
      tolerance: {
        maxAbsoluteError: options.maxAbsoluteError ?? 0,
        maxRelativeError: options.maxRelativeError ?? 0,
      },
    });
    await mkdir(path.dirname(options.report), { recursive: true });
    await writeFile(options.report, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    return report;
  } finally {
    await opened.close();
  }
}

function applyF32Policy(ir: ModelIR): void {
  for (const operation of allOperations(ir)) {
    operation.dtypePolicy = { computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
}

function allOperations(ir: ModelIR): Operation[] {
  return [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue];
}
