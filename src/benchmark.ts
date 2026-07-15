import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { buildModelIR } from "./architecture.js";
import { writeNativeBenchmarkFixtures, type BenchmarkFixture } from "./benchmark-fixtures.js";
import { openCatalog } from "./catalog.js";
import { executeReferenceF32 } from "./executor.js";
import { materializeReferenceF32Constants } from "./materialize.js";
import { applyReferenceF32Policy } from "./reference-policy.js";
import type { ModelIR } from "./types.js";

const PREVIEW = { outputRows: 1, inputTerms: 1, includeWeights: false } as const;
const PREFILL_TOKENS = [1, 2, 3, 4];
const DECODE_TOKEN = 5;

export interface NativeBenchmarkOptions {
  samples?: number;
  warmupSamples?: number;
}

export interface NativeBenchmarkReport {
  schemaVersion: 1;
  benchmark: "native-container-and-reference-executor";
  generatedAt: string;
  machine: { node: string; platform: string; arch: string };
  sampling: { warmupSamples: number; measuredSamples: number; clock: "process.hrtime.bigint nanoseconds" };
  workloads: NativeBenchmarkWorkload[];
  caveats: string[];
}

export interface NativeBenchmarkWorkload {
  id: BenchmarkFixture["id"];
  model: "deterministic one-layer Llama benchmark fixture";
  containerFormat: BenchmarkFixture["format"];
  quantization: string;
  fixtureFiles: BenchmarkFixture["files"];
  architecture: { hiddenSize: number; layers: number; attentionHeads: number; keyValueHeads: number; intermediateSize: number; vocabSize: number };
  input: { prefillTokens: number[]; decodeToken: number; decodePosition: number };
  memory: MemoryObservation;
  stages: Record<"catalogInspection" | "irLowering" | "rangeMaterialization" | "prefillExecution" | "incrementalDecode", TimingSummary>;
}

export interface TimingSummary {
  samplesNanoseconds: number[];
  minimumNanoseconds: number;
  medianNanoseconds: number;
  p95Nanoseconds: number;
}

export interface MemoryObservation {
  /** Exact retained typed-array payload for constants used by the F32 executor. */
  materializedConstantBytes: number;
  /** Process snapshot immediately after one complete materialization; not a portable limit. */
  processAfterMaterializationBytes: { rss: number; heapUsed: number; arrayBuffers: number };
}

/**
 * Measures only the native paths that the reference executor can actually
 * replay today. It intentionally makes no speed assertion: wall-clock values
 * are machine-scoped evidence, while workload hashes make runs comparable.
 */
export async function runNativeFixtureBenchmarks(options: NativeBenchmarkOptions = {}): Promise<NativeBenchmarkReport> {
  const samples = options.samples ?? 5;
  const warmupSamples = options.warmupSamples ?? 1;
  validateSamples(samples, warmupSamples);
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-native-benchmark-"));
  try {
    const fixtures = await writeNativeBenchmarkFixtures(root);
    return {
      schemaVersion: 1,
      benchmark: "native-container-and-reference-executor",
      generatedAt: new Date().toISOString(),
      machine: { node: process.version, platform: process.platform, arch: process.arch },
      sampling: { warmupSamples, measuredSamples: samples, clock: "process.hrtime.bigint nanoseconds" },
      workloads: await Promise.all(fixtures.map((fixture) => benchmarkFixture(fixture, samples, warmupSamples))),
      caveats: [
        "Results are machine-scoped wall-clock observations, not cross-machine performance claims.",
        "Fixtures are deterministic synthetic Llama packages used to exercise native container paths; they are not released-model or authoritative-runtime fidelity evidence.",
        "Catalog inspection, lowering, materialization, prefill, and single-token cache-backed decode are timed separately; fixture creation and checksum calculation are excluded.",
      ],
    };
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

async function benchmarkFixture(fixture: BenchmarkFixture, samples: number, warmups: number): Promise<NativeBenchmarkWorkload> {
  const catalogInspection = await measure(samples, warmups, async () => {
    const opened = await openCatalog(fixture.source, false);
    await opened.close();
  });
  const irLowering = await withOpened(fixture.source, (opened) => measure(samples, warmups, async () => {
    await buildModelIR(opened.catalog, PREVIEW, opened.bridge);
  }));
  const rangeMaterialization = await withPrepared(fixture.source, (ir, opened) => measure(samples, warmups, async () => {
    await materializeReferenceF32Constants(ir, opened.catalog, opened.reader, opened.bridge);
  }));
  const prefillExecution = await withMaterialized(fixture.source, (ir, tensors) => measure(samples, warmups, async () => {
    executeReferenceF32(ir, { inputIds: [PREFILL_TOKENS], tensors });
  }));
  const incrementalDecode = await withMaterialized(fixture.source, (ir, tensors) => measurePrepared(samples, warmups,
    async () => executeReferenceF32(ir, { inputIds: [PREFILL_TOKENS], tensors }),
    async (prefill) => { executeReferenceF32(ir, { inputIds: [[DECODE_TOKEN]], positionIds: [[PREFILL_TOKENS.length]], pastKeyValues: prefill.pastKeyValues, tensors }); },
  ));
  const memory = await withMaterialized(fixture.source, async (_ir, tensors) => ({
    materializedConstantBytes: [...tensors.values()].reduce((total, tensor) => total + tensor.values.byteLength, 0),
    processAfterMaterializationBytes: processMemoryBytes(),
  }));

  return {
    id: fixture.id,
    model: "deterministic one-layer Llama benchmark fixture",
    containerFormat: fixture.format,
    quantization: fixture.quantization,
    fixtureFiles: fixture.files,
    architecture: { hiddenSize: 256, layers: 1, attentionHeads: 4, keyValueHeads: 4, intermediateSize: 256, vocabSize: 256 },
    input: { prefillTokens: [...PREFILL_TOKENS], decodeToken: DECODE_TOKEN, decodePosition: PREFILL_TOKENS.length },
    memory,
    stages: { catalogInspection, irLowering, rangeMaterialization, prefillExecution, incrementalDecode },
  };
}

async function withOpened<T>(source: string, run: (opened: Awaited<ReturnType<typeof openCatalog>>) => Promise<T>): Promise<T> {
  const opened = await openCatalog(source, false);
  try {
    return await run(opened);
  } finally {
    await opened.close();
  }
}

async function withPrepared<T>(source: string, run: (ir: ModelIR, opened: Awaited<ReturnType<typeof openCatalog>>) => Promise<T>): Promise<T> {
  return withOpened(source, async (opened) => {
    const ir = await buildModelIR(opened.catalog, PREVIEW, opened.bridge);
    applyReferenceF32Policy(ir);
    return run(ir, opened);
  });
}

async function withMaterialized<T>(source: string, run: (ir: ModelIR, tensors: Awaited<ReturnType<typeof materializeReferenceF32Constants>>) => Promise<T>): Promise<T> {
  return withPrepared(source, async (ir, opened) => {
    const tensors = await materializeReferenceF32Constants(ir, opened.catalog, opened.reader, opened.bridge);
    return run(ir, tensors);
  });
}

async function measure(samples: number, warmups: number, run: () => Promise<void>): Promise<TimingSummary> {
  for (let index = 0; index < warmups; index += 1) await run();
  const values: number[] = [];
  for (let index = 0; index < samples; index += 1) {
    const started = process.hrtime.bigint();
    await run();
    values.push(Number(process.hrtime.bigint() - started));
  }
  const sorted = [...values].sort((left, right) => left - right);
  return {
    samplesNanoseconds: values,
    minimumNanoseconds: sorted[0]!,
    medianNanoseconds: percentile(sorted, 0.5),
    p95Nanoseconds: percentile(sorted, 0.95),
  };
}

async function measurePrepared<T>(
  samples: number,
  warmups: number,
  prepare: () => Promise<T>,
  run: (prepared: T) => Promise<void>,
): Promise<TimingSummary> {
  for (let index = 0; index < warmups; index += 1) await run(await prepare());
  const values: number[] = [];
  for (let index = 0; index < samples; index += 1) {
    const prepared = await prepare();
    const started = process.hrtime.bigint();
    await run(prepared);
    values.push(Number(process.hrtime.bigint() - started));
  }
  const sorted = [...values].sort((left, right) => left - right);
  return {
    samplesNanoseconds: values,
    minimumNanoseconds: sorted[0]!,
    medianNanoseconds: percentile(sorted, 0.5),
    p95Nanoseconds: percentile(sorted, 0.95),
  };
}

function percentile(sorted: readonly number[], percentileValue: number): number {
  return sorted[Math.ceil(percentileValue * sorted.length) - 1]!;
}

function validateSamples(samples: number, warmups: number): void {
  if (!Number.isInteger(samples) || samples <= 0) throw new Error("samples deve ser inteiro positivo.");
  if (!Number.isInteger(warmups) || warmups < 0) throw new Error("warmupSamples deve ser inteiro não negativo.");
}

function processMemoryBytes(): MemoryObservation["processAfterMaterializationBytes"] {
  const memory = process.memoryUsage();
  return { rss: memory.rss, heapUsed: memory.heapUsed, arrayBuffers: memory.arrayBuffers };
}
