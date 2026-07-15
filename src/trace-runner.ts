import { mkdir, writeFile } from "node:fs/promises";
import * as path from "node:path";
import { buildModelIR } from "./architecture.js";
import { openCatalog } from "./catalog.js";
import { compareExecutionTrace, compareGenerationTrace } from "./differential.js";
import { executeReferenceF32, executeReferenceF64, generateReferenceF32, generateReferenceF64 } from "./executor.js";
import { materializeReferenceF32Constants, materializeReferenceF64Constants } from "./materialize.js";
import { fingerprintIR, readExecutionTraceBundle, readGenerationTraceBundle, verifyTraceSource } from "./trace.js";
import type { DifferentialComparisonReport, DifferentialGenerationComparisonReport, ModelIR, Operation } from "./types.js";

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
    const candidate = decoded.bundle.candidatePolicy.dtype === "F32"
      ? await executeF32Trace(ir, opened, decoded.reference.inputTokens)
      : await executeF64Trace(ir, opened, decoded.reference.inputTokens);
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

/**
 * Replays an integrity-bound authoritative greedy-generation capture. The
 * same verified checkpoint and lowered IR are used for prompt prefill and
 * every incremental decode, so token agreement cannot hide a stale cache or
 * changed absolute-position policy.
 */
export async function runGenerationTraceComparison(options: {
  source: string;
  trace: string;
  report: string;
  topK?: number;
  maxAbsoluteError?: number;
  maxRelativeError?: number;
}): Promise<DifferentialGenerationComparisonReport> {
  const decoded = await readGenerationTraceBundle(options.trace);
  const opened = await openCatalog(options.source, true);
  try {
    await verifyTraceSource(opened.catalog, decoded.bundle.source.files);
    const ir = await buildModelIR(opened.catalog, { outputRows: 1, inputTerms: 1, includeWeights: false }, opened.bridge);
    if (fingerprintIR(ir) !== decoded.bundle.irFingerprint) throw new Error("Fingerprint do IR diverge; trace foi capturado para outra semântica/adaptador.");
    const candidate = decoded.bundle.candidatePolicy.dtype === "F32"
      ? await generateF32Trace(ir, opened, decoded.reference)
      : await generateF64Trace(ir, opened, decoded.reference);
    const report = compareGenerationTrace(candidate, decoded.reference, {
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

async function executeF32Trace(ir: ModelIR, opened: Awaited<ReturnType<typeof openCatalog>>, inputIds: number[][]) {
  applyF32Policy(ir);
  const tensors = await materializeReferenceF32Constants(ir, opened.catalog, opened.reader, opened.bridge);
  return executeReferenceF32(ir, { inputIds, tensors });
}

async function executeF64Trace(ir: ModelIR, opened: Awaited<ReturnType<typeof openCatalog>>, inputIds: number[][]) {
  applyF64Policy(ir);
  if (!("readDenseF64" in opened.reader) || typeof opened.reader.readDenseF64 !== "function") {
    throw new Error("Trace F64 requer um contêiner com materializador F64 denso verificado.");
  }
  const tensors = await materializeReferenceF64Constants(ir, opened.catalog, opened.reader);
  return executeReferenceF64(ir, { inputIds, tensors });
}

async function generateF32Trace(ir: ModelIR, opened: Awaited<ReturnType<typeof openCatalog>>, reference: Awaited<ReturnType<typeof readGenerationTraceBundle>>["reference"]) {
  applyF32Policy(ir);
  const tensors = await materializeReferenceF32Constants(ir, opened.catalog, opened.reader, opened.bridge);
  return generateReferenceF32(ir, {
    inputIds: [reference.inputTokens], positionIds: [reference.promptPositionIds], tensors, maxNewTokens: reference.maxNewTokens,
    ...(reference.eosTokenId !== undefined ? { eosTokenId: reference.eosTokenId } : {}),
  });
}

async function generateF64Trace(ir: ModelIR, opened: Awaited<ReturnType<typeof openCatalog>>, reference: Awaited<ReturnType<typeof readGenerationTraceBundle>>["reference"]) {
  applyF64Policy(ir);
  if (!("readDenseF64" in opened.reader) || typeof opened.reader.readDenseF64 !== "function") {
    throw new Error("Trace F64 requer um contêiner com materializador F64 denso verificado.");
  }
  const tensors = await materializeReferenceF64Constants(ir, opened.catalog, opened.reader);
  return generateReferenceF64(ir, {
    inputIds: [reference.inputTokens], positionIds: [reference.promptPositionIds], tensors, maxNewTokens: reference.maxNewTokens,
    ...(reference.eosTokenId !== undefined ? { eosTokenId: reference.eosTokenId } : {}),
  });
}

function applyF32Policy(ir: ModelIR): void {
  for (const operation of allOperations(ir)) {
    operation.dtypePolicy = { computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
}

function applyF64Policy(ir: ModelIR): void {
  for (const operation of allOperations(ir)) {
    operation.dtypePolicy = { computeDtype: "F64", accumulationDtype: "F64", outputDtype: "F64" };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F64";
  }
}

function allOperations(ir: ModelIR): Operation[] {
  return [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue];
}
