import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import * as path from "node:path";
import type {
  DenseF32Tensor,
  DifferentialKeyValueCacheSample,
  DifferentialReferenceTrace,
  ModelCatalog,
  ModelIR,
} from "./types.js";

export interface TraceSourceFile {
  path: string;
  sha256: string;
}

interface SerializedF32Tensor {
  dtype: "F32";
  shape: number[];
  /** Little-endian IEEE-754 binary32 payload, encoded without JSON rounding. */
  valuesBase64: string;
}

interface SerializedOperation {
  operationId: string;
  output: string;
  tensor: SerializedF32Tensor;
}

interface SerializedCache {
  layer: number;
  key: SerializedF32Tensor;
  value: SerializedF32Tensor;
}

/**
 * Portable, integrity-bound capture consumed by the F32 reference executor.
 * The producing runtime remains responsible for authoritative hooks; this
 * format deliberately refuses decimal arrays, implicit dtype conversion, and
 * unbound checkpoint files.
 */
export interface ExecutionTraceBundle {
  schemaVersion: 1;
  kind: "execution";
  source: { files: TraceSourceFile[] };
  irFingerprint: string;
  candidatePolicy: {
    dtype: "F32";
    runtime: string;
  };
  reference: Omit<DifferentialReferenceTrace, "operations" | "pastKeyValues"> & {
    operations: SerializedOperation[];
    pastKeyValues: SerializedCache[];
  };
}

export interface DecodedExecutionTrace {
  bundle: ExecutionTraceBundle;
  reference: DifferentialReferenceTrace;
}

export async function readExecutionTraceBundle(file: string): Promise<DecodedExecutionTrace> {
  const raw: unknown = JSON.parse(await readFile(file, "utf8"));
  const bundle = validateBundle(raw);
  return {
    bundle,
    reference: {
      ...bundle.reference,
      inputTokens: bundle.reference.inputTokens.map((row) => [...row]),
      operations: bundle.reference.operations.map((sample) => ({
        operationId: sample.operationId,
        output: sample.output,
        tensor: decodeTensor(sample.tensor, `operação ${sample.operationId}`),
      })),
      pastKeyValues: bundle.reference.pastKeyValues.map((cache) => ({
        layer: cache.layer,
        key: decodeTensor(cache.key, `cache KV ${cache.layer}.key`),
        value: decodeTensor(cache.value, `cache KV ${cache.layer}.value`),
      })),
    },
  };
}

/** Stable hash of the complete executable IR, including tensor contracts. */
export function fingerprintIR(ir: ModelIR): string {
  return createHash("sha256").update(JSON.stringify(ir)).digest("hex");
}

/**
 * Require exact checksums for every file that contributes config or tensor
 * bytes. A trace from a different revision cannot silently validate a local
 * checkpoint with the same model label.
 */
export async function verifyTraceSource(catalog: ModelCatalog, files: readonly TraceSourceFile[]): Promise<void> {
  const expected = sourceFilesForCatalog(catalog);
  const declared = new Map<string, string>();
  for (const entry of files) {
    assertRelativePath(entry.path, "arquivo de source trace");
    if (!/^[a-f0-9]{64}$/.test(entry.sha256)) throw new Error(`Checksum SHA-256 inválido para ${entry.path}.`);
    if (declared.has(entry.path)) throw new Error(`Arquivo de source trace duplicado: ${entry.path}.`);
    declared.set(entry.path, entry.sha256);
  }
  if (declared.size !== expected.size || [...expected].some((file) => !declared.has(file))) {
    throw new Error(`Trace não cobre exatamente os arquivos do checkpoint local; esperado ${[...expected].sort().join(", ")}.`);
  }
  const base = catalog.format === "gguf" ? path.dirname(catalog.source) : catalog.source;
  for (const file of expected) {
    const bytes = await readFile(path.join(base, file));
    const actual = createHash("sha256").update(bytes).digest("hex");
    if (actual !== declared.get(file)) throw new Error(`Checksum divergente para ${file}; trace e checkpoint não são a mesma revisão.`);
  }
}

function sourceFilesForCatalog(catalog: ModelCatalog): Set<string> {
  if (catalog.format === "gguf") return new Set([path.basename(catalog.source)]);
  const files = new Set<string>(["config.json"]);
  for (const tensor of catalog.tensors.values()) {
    if (!tensor.shard) throw new Error(`${tensor.name}: catálogo Safetensors não declara shard para verificação de trace.`);
    assertRelativePath(tensor.shard, `shard de ${tensor.name}`);
    files.add(tensor.shard);
  }
  return files;
}

function validateBundle(raw: unknown): ExecutionTraceBundle {
  const value = object(raw, "Trace");
  if (value.schemaVersion !== 1 || value.kind !== "execution") throw new Error("Trace requer schemaVersion=1 e kind=execution.");
  const source = object(value.source, "Trace source");
  if (!Array.isArray(source.files) || source.files.length === 0) throw new Error("Trace requer source.files não vazio.");
  const candidatePolicy = object(value.candidatePolicy, "Trace candidatePolicy");
  if (candidatePolicy.dtype !== "F32" || typeof candidatePolicy.runtime !== "string" || candidatePolicy.runtime.trim() === "") {
    throw new Error("Trace requer candidatePolicy F32 e runtime não vazio.");
  }
  if (typeof value.irFingerprint !== "string" || !/^[a-f0-9]{64}$/.test(value.irFingerprint)) throw new Error("Trace requer irFingerprint SHA-256.");
  const reference = object(value.reference, "Trace reference");
  for (const field of ["runtime", "model", "revisionOrChecksum", "containerFormat", "quantization", "dtypePolicy"] as const) {
    if (typeof reference[field] !== "string" || reference[field].trim() === "") throw new Error(`Trace reference requer ${field} não vazio.`);
  }
  if (!Array.isArray(reference.inputTokens) || reference.inputTokens.length === 0 || !reference.inputTokens.every(tokenRow)) throw new Error("Trace reference requer inputTokens inteiros não negativos.");
  if (!Array.isArray(reference.operations) || !Array.isArray(reference.pastKeyValues)) throw new Error("Trace reference requer operations e pastKeyValues arrays.");
  const files = source.files.map((entry) => {
    const file = object(entry, "Trace source file");
    if (typeof file.path !== "string" || typeof file.sha256 !== "string") throw new Error("Trace source file inválido.");
    return { path: file.path, sha256: file.sha256 };
  });
  const operations = reference.operations.map((entry) => {
    const sample = object(entry, "Trace operation");
    if (typeof sample.operationId !== "string" || sample.operationId === "" || typeof sample.output !== "string" || sample.output === "") throw new Error("Trace operation requer operationId e output.");
    return { operationId: sample.operationId, output: sample.output, tensor: validateSerializedTensor(sample.tensor, `operação ${sample.operationId}`) };
  });
  const pastKeyValues = reference.pastKeyValues.map((entry) => {
    const cache = object(entry, "Trace cache KV");
    if (!Number.isInteger(cache.layer) || (cache.layer as number) < 0) throw new Error("Trace cache KV requer layer inteiro não negativo.");
    return { layer: cache.layer as number, key: validateSerializedTensor(cache.key, "cache key"), value: validateSerializedTensor(cache.value, "cache value") };
  });
  return {
    schemaVersion: 1, kind: "execution", source: { files }, irFingerprint: value.irFingerprint,
    candidatePolicy: { dtype: "F32", runtime: candidatePolicy.runtime },
    reference: {
      runtime: reference.runtime as string, model: reference.model as string, revisionOrChecksum: reference.revisionOrChecksum as string,
      containerFormat: reference.containerFormat as string, quantization: reference.quantization as string,
      inputTokens: reference.inputTokens as number[][], dtypePolicy: reference.dtypePolicy as string,
      operations, pastKeyValues,
    },
  };
}

function validateSerializedTensor(raw: unknown, label: string): SerializedF32Tensor {
  const tensor = object(raw, `Tensor ${label}`);
  if (tensor.dtype !== "F32" || !Array.isArray(tensor.shape) || !tensor.shape.every((dimension) => Number.isInteger(dimension) && (dimension as number) > 0) || typeof tensor.valuesBase64 !== "string") {
    throw new Error(`Tensor ${label} deve declarar dtype F32, shape positivo e valuesBase64.`);
  }
  decodeTensor({ dtype: "F32", shape: tensor.shape as number[], valuesBase64: tensor.valuesBase64 }, label);
  return { dtype: "F32", shape: tensor.shape as number[], valuesBase64: tensor.valuesBase64 };
}

function decodeTensor(tensor: SerializedF32Tensor, label: string): DenseF32Tensor {
  const elements = tensor.shape.reduce((total, dimension) => total * dimension, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0) throw new Error(`Tensor ${label} possui shape inseguro.`);
  const bytes = Buffer.from(tensor.valuesBase64, "base64");
  if (bytes.length !== elements * Float32Array.BYTES_PER_ELEMENT) throw new Error(`Tensor ${label} possui ${bytes.length} bytes F32, esperado ${elements * 4}.`);
  const copy = new Uint8Array(bytes);
  return { shape: [...tensor.shape], values: new Float32Array(copy.buffer) };
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`${label} deve ser objeto JSON.`);
  return value as Record<string, unknown>;
}

function tokenRow(row: unknown): boolean {
  return Array.isArray(row) && row.length > 0 && row.every((token) => Number.isInteger(token) && (token as number) >= 0);
}

function assertRelativePath(file: string, label: string): void {
  if (path.isAbsolute(file) || file === "" || file.split(path.sep).includes("..") || file.includes("/../") || file.includes("\\\\..\\\\")) throw new Error(`${label} deve ser caminho relativo seguro: ${file}.`);
}
