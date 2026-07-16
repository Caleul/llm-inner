import { mkdir, writeFile } from "node:fs/promises";
import * as path from "node:path";
import { buildModelIR } from "./architecture.js";
import { openCatalog } from "./catalog.js";
import { compareExecutionTrace, compareGenerationTrace } from "./differential.js";
import { executeReferenceF32, executeReferenceF64, generateReferenceF32, generateReferenceF64 } from "./executor.js";
import { materializeReferenceF32Constants, materializeReferenceF64Constants } from "./materialize.js";
import { applyReferenceF32Policy, applyReferenceF64Policy } from "./reference-policy.js";
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
      ? await executeF32Trace(ir, opened, decoded.reference)
      : await executeF64Trace(ir, opened, decoded.reference);
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
    assertGenerationTraceCacheCoverage(ir, decoded.reference);
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

async function executeF32Trace(
  ir: ModelIR,
  opened: Awaited<ReturnType<typeof openCatalog>>,
  reference: Awaited<ReturnType<typeof readExecutionTraceBundle>>["reference"],
) {
  applyReferenceF32Policy(ir);
  const tensors = await materializeReferenceF32Constants(ir, opened.catalog, opened.reader, opened.bridge);
  return executeReferenceF32(ir, { inputIds: reference.inputTokens, ...(reference.positionIds ? { positionIds: reference.positionIds } : {}), tensors });
}

async function executeF64Trace(
  ir: ModelIR,
  opened: Awaited<ReturnType<typeof openCatalog>>,
  reference: Awaited<ReturnType<typeof readExecutionTraceBundle>>["reference"],
) {
  applyReferenceF64Policy(ir);
  if (!("readDenseF64" in opened.reader) || typeof opened.reader.readDenseF64 !== "function") {
    throw new Error("Trace F64 requer um contêiner com materializador F64 denso verificado.");
  }
  const tensors = await materializeReferenceF64Constants(ir, opened.catalog, opened.reader);
  return executeReferenceF64(ir, { inputIds: reference.inputTokens, ...(reference.positionIds ? { positionIds: reference.positionIds } : {}), tensors });
}

async function generateF32Trace(ir: ModelIR, opened: Awaited<ReturnType<typeof openCatalog>>, reference: Awaited<ReturnType<typeof readGenerationTraceBundle>>["reference"]) {
  applyReferenceF32Policy(ir);
  const tensors = await materializeReferenceF32Constants(ir, opened.catalog, opened.reader, opened.bridge);
  return generateReferenceF32(ir, {
    inputIds: [reference.inputTokens], positionIds: [reference.promptPositionIds], tensors, maxNewTokens: reference.maxNewTokens,
    ...(reference.eosTokenId !== undefined ? { eosTokenId: reference.eosTokenId } : {}),
  });
}

async function generateF64Trace(ir: ModelIR, opened: Awaited<ReturnType<typeof openCatalog>>, reference: Awaited<ReturnType<typeof readGenerationTraceBundle>>["reference"]) {
  applyReferenceF64Policy(ir);
  if (!("readDenseF64" in opened.reader) || typeof opened.reader.readDenseF64 !== "function") {
    throw new Error("Trace F64 requer um contêiner com materializador F64 denso verificado.");
  }
  const tensors = await materializeReferenceF64Constants(ir, opened.catalog, opened.reader);
  return generateReferenceF64(ir, {
    inputIds: [reference.inputTokens], positionIds: [reference.promptPositionIds], tensors, maxNewTokens: reference.maxNewTokens,
    ...(reference.eosTokenId !== undefined ? { eosTokenId: reference.eosTokenId } : {}),
  });
}


/**
 * A generation trace must prove continuation state for every independent
 * attention layer declared by the lowered IR.  Comparing only the union of
 * cache layers supplied by candidate and reference would allow both sides to
 * omit the same layer and still appear complete.
 *
 * A shared-KV consumer deliberately has no cache entry of its own. Its
 * producer is already part of `required`, so the trace proves the canonical
 * state exactly once and rejects a duplicate consumer-owned snapshot.
 */
function assertGenerationTraceCacheCoverage(
  ir: ModelIR,
  reference: Awaited<ReturnType<typeof readGenerationTraceBundle>>["reference"],
): void {
  const required = new Set(
    allOperations(ir)
      .filter((operation): operation is Extract<Operation, { op: "scaled_dot_product_attention" }> =>
        operation.op === "scaled_dot_product_attention" && operation.layer !== undefined && !operation.kvSharing)
      .map((operation) => operation.layer!),
  );
  const validate = (caches: readonly { layer: number }[], location: string) => {
    const actual = new Set(caches.map((cache) => cache.layer));
    const missing = [...required].filter((layer) => !actual.has(layer));
    const unexpected = [...actual].filter((layer) => !required.has(layer));
    if (missing.length > 0 || unexpected.length > 0) {
      throw new Error(`${location} não cobre exatamente as camadas KV independentes do IR; ausentes [${missing.join(", ")}], extras [${unexpected.join(", ")}].`);
    }
  };
  reference.stepPastKeyValues.forEach((caches, index) => validate(caches, `Trace de geração cache KV pós-decode ${index}`));
  validate(reference.pastKeyValues, "Trace de geração cache KV final");
}

function allOperations(ir: ModelIR): Operation[] {
  return [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue];
}
