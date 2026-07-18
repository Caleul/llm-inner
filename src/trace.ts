import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile } from "node:fs/promises";
import * as path from "node:path";
import type {
  DenseTensor,
  DenseF32Tensor,
  DifferentialGenerationReferenceTrace,
  DifferentialKeyValueCacheSample,
  DifferentialNativeKernelEnvironment,
  DifferentialOperationLayout,
  DifferentialReductionProbeInput,
  DifferentialReferenceTrace,
  ModelCatalog,
  ModelIR,
} from "./types.js";
import { selectGreedyToken } from "./generation.js";

export interface TraceSourceFile {
  path: string;
  sha256: string;
}

/** Stream source-file identities so a real multi-gigabyte checkpoint never
 * needs to fit in Node's Buffer limit merely to bind an authoritative trace. */
export async function sha256File(file: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

interface SerializedTensor {
  dtype: "F32" | "F64";
  shape: number[];
  /** Little-endian IEEE-754 payload in the declared dtype, without JSON rounding. */
  valuesBase64: string;
}

interface SerializedOperation {
  operationId: string;
  output: string;
  tensor: SerializedTensor;
}

interface SerializedCache {
  layer: number;
  key: SerializedTensor;
  value: SerializedTensor;
}

interface SerializedGenerationStep {
  tokenId: number;
  positionId: number;
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
  /** Per-capture nonce; optional for historical generic traces. */
  captureId?: string;
  source: { files: TraceSourceFile[] };
  irFingerprint: string;
  candidatePolicy: {
    dtype: "F32" | "F64";
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

/** Persisted F32 greedy-generation evidence, bound to checkpoint and IR. */
export interface GenerationTraceBundle {
  schemaVersion: 1;
  kind: "generation";
  source: { files: TraceSourceFile[] };
  irFingerprint: string;
  candidatePolicy: { dtype: "F32" | "F64"; runtime: string };
  reference: Omit<DifferentialGenerationReferenceTrace, "logits" | "pastKeyValues" | "stepPastKeyValues" | "steps" | "selectionLogits"> & {
    logits: SerializedTensor;
    pastKeyValues: SerializedCache[];
    steps: SerializedGenerationStep[];
    selectionLogits: SerializedTensor[];
    stepPastKeyValues: SerializedCache[][];
  };
}

export interface DecodedGenerationTrace {
  bundle: GenerationTraceBundle;
  reference: DifferentialGenerationReferenceTrace;
}

export async function readExecutionTraceBundle(file: string): Promise<DecodedExecutionTrace> {
  const raw: unknown = JSON.parse(await readFile(file, "utf8"));
  const bundle = validateBundle(raw);
  return {
    bundle,
    reference: {
      ...bundle.reference,
      inputTokens: bundle.reference.inputTokens.map((row) => [...row]),
      ...(bundle.reference.positionIds ? { positionIds: bundle.reference.positionIds.map((row) => [...row]) } : {}),
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

export async function readGenerationTraceBundle(file: string): Promise<DecodedGenerationTrace> {
  const raw: unknown = JSON.parse(await readFile(file, "utf8"));
  const bundle = validateGenerationBundle(raw);
  return {
    bundle,
    reference: {
      ...bundle.reference,
      inputTokens: [...bundle.reference.inputTokens],
      promptPositionIds: [...bundle.reference.promptPositionIds],
      generatedTokenIds: [...bundle.reference.generatedTokenIds],
      steps: bundle.reference.steps.map((step) => ({ ...step })),
      selectionLogits: bundle.reference.selectionLogits.map((tensor, index) => decodeTensor(tensor, `logits de seleção ${index}`)),
      stepPastKeyValues: bundle.reference.stepPastKeyValues.map((step, index) => step.map((cache) => ({
        layer: cache.layer,
        key: decodeTensor(cache.key, `cache KV pós-decode ${index}/${cache.layer}.key`),
        value: decodeTensor(cache.value, `cache KV pós-decode ${index}/${cache.layer}.value`),
      }))),
      logits: decodeTensor(bundle.reference.logits, "logits terminais"),
      pastKeyValues: bundle.reference.pastKeyValues.map((cache) => ({
        layer: cache.layer,
        key: decodeTensor(cache.key, `cache KV ${cache.layer}.key`),
        value: decodeTensor(cache.value, `cache KV ${cache.layer}.value`),
      })),
    },
  };
}

/**
 * Stable hash of mathematical IR semantics and tensor contracts.  Source
 * location is bound separately by `source.files`, while preview controls are
 * diagnostics only and must never make two equivalent full programs appear
 * semantically different.
 */
export function fingerprintIR(ir: ModelIR): string {
  const semantic = structuredClone(ir);
  semantic.source.path = "<source-bound-by-checksums>";
  semantic.preview = { outputRows: 0, inputTerms: 0, includeWeights: false };
  return createHash("sha256").update(JSON.stringify(semantic)).digest("hex");
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
    const actual = await sha256File(path.join(base, file));
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
  if ((candidatePolicy.dtype !== "F32" && candidatePolicy.dtype !== "F64") || typeof candidatePolicy.runtime !== "string" || candidatePolicy.runtime.trim() === "") {
    throw new Error("Trace requer candidatePolicy F32/F64 e runtime não vazio.");
  }
  if (typeof value.irFingerprint !== "string" || !/^[a-f0-9]{64}$/.test(value.irFingerprint)) throw new Error("Trace requer irFingerprint SHA-256.");
  const reference = object(value.reference, "Trace reference");
  for (const field of ["runtime", "model", "revisionOrChecksum", "containerFormat", "quantization", "dtypePolicy"] as const) {
    if (typeof reference[field] !== "string" || reference[field].trim() === "") throw new Error(`Trace reference requer ${field} não vazio.`);
  }
  if (reference.executionDevice !== undefined && (typeof reference.executionDevice !== "string" || reference.executionDevice.trim() === "")) {
    throw new Error("Trace reference executionDevice deve ser string não vazia quando declarado.");
  }
  if (reference.executionDeviceDetail !== undefined && (typeof reference.executionDeviceDetail !== "string" || reference.executionDeviceDetail.trim() === "")) {
    throw new Error("Trace reference executionDeviceDetail deve ser string não vazia quando declarado.");
  }
  if (!Array.isArray(reference.inputTokens) || reference.inputTokens.length === 0 || !reference.inputTokens.every(tokenRow)) throw new Error("Trace reference requer inputTokens inteiros não negativos.");
  const inputTokens = reference.inputTokens as number[][];
  const positionIds = reference.positionIds;
  if (positionIds !== undefined && (!Array.isArray(positionIds) || positionIds.length !== inputTokens.length ||
    !positionIds.every((row, index) => tokenRow(row) && row.length === inputTokens[index]!.length))) {
    throw new Error("Trace reference positionIds requer o mesmo shape de inputTokens com posições inteiras não negativas.");
  }
  if (!Array.isArray(reference.operations) || !Array.isArray(reference.pastKeyValues)) throw new Error("Trace reference requer operations e pastKeyValues arrays.");
  const files = source.files.map((entry) => {
    const file = object(entry, "Trace source file");
    if (typeof file.path !== "string" || typeof file.sha256 !== "string") throw new Error("Trace source file inválido.");
    return { path: file.path, sha256: file.sha256 };
  });
  const dtype = candidatePolicy.dtype as "F32" | "F64";
  const operations = reference.operations.map((entry) => {
    const sample = object(entry, "Trace operation");
    if (typeof sample.operationId !== "string" || sample.operationId === "" || typeof sample.output !== "string" || sample.output === "") throw new Error("Trace operation requer operationId e output.");
    return { operationId: sample.operationId, output: sample.output, tensor: validateSerializedTensor(sample.tensor, `operação ${sample.operationId}`, dtype) };
  });
  const operationIds = new Set(operations.map((operation) => operation.operationId));
  const operationDtypes = reference.operationDtypes === undefined ? undefined : parseOperationDtypes(reference.operationDtypes, operationIds);
  const operationLayouts = reference.operationLayouts === undefined ? undefined : parseOperationLayouts(reference.operationLayouts, operations);
  const nativeKernelEnvironment = reference.nativeKernelEnvironment === undefined
    ? undefined
    : parseNativeKernelEnvironment(reference.nativeKernelEnvironment);
  const reductionProbeInput = reference.reductionProbeInput === undefined
    ? undefined
    : parseReductionProbeInput(reference.reductionProbeInput);
  const pastKeyValues = reference.pastKeyValues.map((entry) => {
    const cache = object(entry, "Trace cache KV");
    if (!Number.isInteger(cache.layer) || (cache.layer as number) < 0) throw new Error("Trace cache KV requer layer inteiro não negativo.");
    return { layer: cache.layer as number, key: validateSerializedTensor(cache.key, "cache key", dtype), value: validateSerializedTensor(cache.value, "cache value", dtype) };
  });
  const captureId = value.captureId;
  if (captureId !== undefined && (typeof captureId !== "string" || captureId.trim() === "")) throw new Error("Trace captureId deve ser string não vazia quando declarado.");
  return {
    schemaVersion: 1, kind: "execution", ...(captureId === undefined ? {} : { captureId }), source: { files }, irFingerprint: value.irFingerprint,
    candidatePolicy: { dtype: candidatePolicy.dtype as "F32" | "F64", runtime: candidatePolicy.runtime },
    reference: {
      runtime: reference.runtime as string, model: reference.model as string, revisionOrChecksum: reference.revisionOrChecksum as string,
      ...(reference.executionMode === undefined ? {} : { executionMode: nonemptyTraceContext(reference.executionMode, "executionMode") }),
      ...(reference.attentionImplementation === undefined ? {} : { attentionImplementation: nonemptyTraceContext(reference.attentionImplementation, "attentionImplementation") }),
      ...(reference.executionDevice === undefined ? {} : { executionDevice: reference.executionDevice as string }),
      ...(reference.executionDeviceDetail === undefined ? {} : { executionDeviceDetail: reference.executionDeviceDetail as string }),
      containerFormat: reference.containerFormat as string, quantization: reference.quantization as string,
      inputTokens, ...(positionIds !== undefined ? { positionIds: positionIds.map((row) => [...row] as number[]) } : {}), dtypePolicy: reference.dtypePolicy as string,
      operations, ...(operationDtypes === undefined ? {} : { operationDtypes }), ...(operationLayouts === undefined ? {} : { operationLayouts }),
      ...(nativeKernelEnvironment === undefined ? {} : { nativeKernelEnvironment }),
      ...(reductionProbeInput === undefined ? {} : { reductionProbeInput }), pastKeyValues,
    },
  };
}

function parseReductionProbeInput(raw: unknown): DifferentialReductionProbeInput {
  const value = object(raw, "Trace reductionProbeInput");
  if (value.kind === "model-forward") return { kind: "model-forward" };
  if (value.kind === "bf16-power-of-two-scale") {
    const factor = value.factor;
    if (typeof factor !== "number" || !Number.isSafeInteger(factor) || factor < 2 || factor > 256 || (factor & (factor - 1)) !== 0) {
      throw new Error("Trace reductionProbeInput bf16-power-of-two-scale requer factor potência de dois entre 2 e 256.");
    }
    return { kind: "bf16-power-of-two-scale", factor };
  }
  if (value.kind === "bf16-scalar-scale") {
    const factorBf16Bits = value.factorBf16Bits;
    if (typeof factorBf16Bits !== "number") {
      throw new Error("Trace reductionProbeInput bf16-scalar-scale requer factorBf16Bits finito, não zero e uint16.");
    }
    const exponent = (factorBf16Bits >>> 7) & 0xff;
    if (!Number.isSafeInteger(factorBf16Bits) || factorBf16Bits < 0 || factorBf16Bits > 0xffff || exponent === 0xff || (factorBf16Bits & 0x7fff) === 0) {
      throw new Error("Trace reductionProbeInput bf16-scalar-scale requer factorBf16Bits finito, não zero e uint16.");
    }
    return { kind: "bf16-scalar-scale", factorBf16Bits };
  }
  throw new Error("Trace reductionProbeInput requer kind model-forward, bf16-power-of-two-scale ou bf16-scalar-scale.");
}

function parseNativeKernelEnvironment(raw: unknown): DifferentialNativeKernelEnvironment {
  const value = object(raw, "Trace nativeKernelEnvironment");
  if (typeof value.torchBuildConfigSha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.torchBuildConfigSha256)) {
    throw new Error("Trace nativeKernelEnvironment requer torchBuildConfigSha256 SHA-256.");
  }
  for (const field of ["intraopThreads", "interopThreads"] as const) {
    if (!Number.isSafeInteger(value[field]) || (value[field] as number) <= 0) {
      throw new Error(`Trace nativeKernelEnvironment requer ${field} inteiro positivo seguro.`);
    }
  }
  for (const field of ["deterministicAlgorithms", "mkldnnAvailable", "mkldnnEnabled"] as const) {
    if (typeof value[field] !== "boolean") throw new Error(`Trace nativeKernelEnvironment requer ${field} booleano.`);
  }
  return {
    torchBuildConfigSha256: value.torchBuildConfigSha256,
    intraopThreads: value.intraopThreads as number,
    interopThreads: value.interopThreads as number,
    deterministicAlgorithms: value.deterministicAlgorithms as boolean,
    mkldnnAvailable: value.mkldnnAvailable as boolean,
    mkldnnEnabled: value.mkldnnEnabled as boolean,
  };
}

function parseOperationDtypes(raw: unknown, operationIds: ReadonlySet<string>): Array<{ operationId: string; inputDtype: string; outputDtype: string; parameterDtype?: string }> {
  if (!Array.isArray(raw)) throw new Error("Trace reference operationDtypes deve ser array quando declarado.");
  const seen = new Set<string>();
  return raw.map((entry) => {
    const value = object(entry, "Trace operation dtype");
    for (const field of ["operationId", "inputDtype", "outputDtype"] as const) {
      if (typeof value[field] !== "string" || value[field].trim() === "") throw new Error(`Trace operation dtype requer ${field} não vazio.`);
    }
    if (value.parameterDtype !== undefined && (typeof value.parameterDtype !== "string" || value.parameterDtype.trim() === "")) {
      throw new Error("Trace operation dtype parameterDtype deve ser string não vazia quando declarado.");
    }
    const operationId = value.operationId as string;
    if (!operationIds.has(operationId)) throw new Error(`Trace operation dtype referencia operação ausente ${operationId}.`);
    if (seen.has(operationId)) throw new Error(`Trace operation dtype duplicou ${operationId}.`);
    seen.add(operationId);
    return {
      operationId,
      inputDtype: value.inputDtype as string,
      outputDtype: value.outputDtype as string,
      ...(value.parameterDtype === undefined ? {} : { parameterDtype: value.parameterDtype as string }),
    };
  });
}

function parseOperationLayouts(raw: unknown, operations: readonly { operationId: string; tensor: { shape: number[] } }[]): DifferentialOperationLayout[] {
  if (!Array.isArray(raw)) throw new Error("Trace operationLayouts deve ser array quando declarado.");
  const samples = new Map(operations.map((operation) => [operation.operationId, operation]));
  const seen = new Set<string>();
  return raw.map((entry) => {
    const value = object(entry, "Trace operation layout");
    if (typeof value.operationId !== "string" || value.operationId.trim() === "") throw new Error("Trace operation layout requer operationId não vazio.");
    const sample = samples.get(value.operationId);
    if (!sample) throw new Error(`Trace operation layout referencia operação ausente ${value.operationId}.`);
    if (seen.has(value.operationId)) throw new Error(`Trace operation layout duplicou ${value.operationId}.`);
    seen.add(value.operationId);
    const input = parseTensorLayout(value.input, `Trace operation layout ${value.operationId} input`);
    const output = parseTensorLayout(value.output, `Trace operation layout ${value.operationId} output`);
    if (!sameShape(output.shape, sample.tensor.shape)) throw new Error(`Trace operation layout ${value.operationId} output diverge do tensor serializado.`);
    const parameter = value.parameter === undefined ? undefined : parseTensorLayout(value.parameter, `Trace operation layout ${value.operationId} parameter`);
    return { operationId: value.operationId, input, output, ...(parameter === undefined ? {} : { parameter }) };
  });
}

function parseTensorLayout(raw: unknown, label: string): DifferentialOperationLayout["input"] {
  const value = object(raw, label);
  if (!Array.isArray(value.shape) || value.shape.length === 0 || !value.shape.every((dimension) => Number.isSafeInteger(dimension) && (dimension as number) > 0)) {
    throw new Error(`${label} requer shape de dimensões positivas seguras.`);
  }
  if (!Array.isArray(value.strides) || value.strides.length !== value.shape.length || !value.strides.every((stride) => Number.isSafeInteger(stride) && (stride as number) >= 0)) {
    throw new Error(`${label} requer strides não negativos e compatíveis com shape.`);
  }
  if (!Number.isSafeInteger(value.storageOffset) || (value.storageOffset as number) < 0 || typeof value.isContiguous !== "boolean") {
    throw new Error(`${label} requer storageOffset seguro e isContiguous booleano.`);
  }
  return {
    shape: value.shape as number[], strides: value.strides as number[], storageOffset: value.storageOffset as number,
    isContiguous: value.isContiguous as boolean,
  };
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((dimension, index) => dimension === right[index]);
}

function validateGenerationBundle(raw: unknown): GenerationTraceBundle {
  const value = object(raw, "Trace de geração");
  validateEnvelope(value, "generation");
  const source = object(value.source, "Trace source");
  const candidatePolicy = object(value.candidatePolicy, "Trace candidatePolicy");
  const reference = object(value.reference, "Trace generation reference");
  for (const field of ["runtime", "model", "revisionOrChecksum", "containerFormat", "quantization", "dtypePolicy"] as const) {
    if (typeof reference[field] !== "string" || reference[field].trim() === "") throw new Error(`Trace de geração requer ${field} não vazio.`);
  }
  if (reference.executionDevice !== undefined && (typeof reference.executionDevice !== "string" || reference.executionDevice.trim() === "")) {
    throw new Error("Trace de geração executionDevice deve ser string não vazia quando declarado.");
  }
  if (reference.executionDeviceDetail !== undefined && (typeof reference.executionDeviceDetail !== "string" || reference.executionDeviceDetail.trim() === "")) {
    throw new Error("Trace de geração executionDeviceDetail deve ser string não vazia quando declarado.");
  }
  if (!tokenVector(reference.inputTokens)) throw new Error("Trace de geração requer inputTokens inteiros não negativos.");
  if (!tokenVector(reference.promptPositionIds) || reference.promptPositionIds.length !== reference.inputTokens.length) {
    throw new Error("Trace de geração requer promptPositionIds inteiros não negativos com o comprimento do prompt.");
  }
  if (!tokenVector(reference.generatedTokenIds)) throw new Error("Trace de geração contém generatedTokenIds inválido.");
  const inputTokens = reference.inputTokens;
  const promptPositionIds = reference.promptPositionIds;
  const generatedTokenIds = reference.generatedTokenIds;
  if (!Number.isInteger(reference.maxNewTokens) || (reference.maxNewTokens as number) < 0 || generatedTokenIds.length > (reference.maxNewTokens as number)) {
    throw new Error("Trace de geração contém maxNewTokens ou generatedTokenIds inválido.");
  }
  if (reference.eosTokenId !== undefined && (!Number.isInteger(reference.eosTokenId) || (reference.eosTokenId as number) < 0)) throw new Error("Trace de geração contém eosTokenId inválido.");
  if (!Array.isArray(reference.steps) || reference.steps.length !== generatedTokenIds.length) throw new Error("Trace de geração requer um step para cada token emitido.");
  if (!Array.isArray(reference.selectionLogits) || reference.selectionLogits.length !== generatedTokenIds.length) throw new Error("Trace de geração requer logits de seleção para cada token emitido.");
  if (!Array.isArray(reference.stepPastKeyValues) || reference.stepPastKeyValues.length !== generatedTokenIds.length) throw new Error("Trace de geração requer cache KV pós-decode para cada token emitido.");
  const steps = reference.steps.map((entry, index) => {
    const step = object(entry, `Trace de geração step ${index}`);
    if (!Number.isInteger(step.tokenId) || (step.tokenId as number) < 0 || !Number.isInteger(step.positionId) || (step.positionId as number) < 0 || step.tokenId !== generatedTokenIds[index]) {
      throw new Error(`Trace de geração contém step inválido no índice ${index}.`);
    }
    return { tokenId: step.tokenId as number, positionId: step.positionId as number };
  });
  const selectionLogits = reference.selectionLogits.map((entry, index) =>
    validateSerializedTensor(entry, `logits de seleção ${index}`, candidatePolicy.dtype as "F32" | "F64"));
  for (const [index, logits] of selectionLogits.entries()) {
    if (selectGreedyToken(decodeTensor(logits, `logits de seleção ${index}`)) !== generatedTokenIds[index]) {
      throw new Error(`Trace de geração seleciona token ${generatedTokenIds[index]} no índice ${index}, mas os logits determinísticos exigem outro argmax.`);
    }
  }
  const stepPastKeyValues = reference.stepPastKeyValues.map((entry, index) => {
    if (!Array.isArray(entry)) throw new Error(`Trace de geração cache KV pós-decode ${index} deve ser array.`);
    const caches = entry.map((cache) => serializedCache(cache, `Trace de geração cache KV pós-decode ${index}`, candidatePolicy.dtype as "F32" | "F64"));
    uniqueCacheLayers(caches);
    return caches;
  });
  const eosIndex = reference.eosTokenId === undefined ? -1 : generatedTokenIds.indexOf(reference.eosTokenId as number);
  if (eosIndex >= 0 && eosIndex !== generatedTokenIds.length - 1) throw new Error("Trace de geração não pode emitir tokens após EOS.");
  if (!Array.isArray(reference.pastKeyValues)) throw new Error("Trace de geração requer pastKeyValues array.");
  const pastKeyValues = reference.pastKeyValues.map((entry) => serializedCache(entry, "Trace de geração cache KV", candidatePolicy.dtype as "F32" | "F64"));
  uniqueCacheLayers(pastKeyValues);
  const files = parseSourceFiles(source.files);
  return {
    schemaVersion: 1, kind: "generation", source: { files }, irFingerprint: value.irFingerprint as string,
    candidatePolicy: { dtype: candidatePolicy.dtype as "F32" | "F64", runtime: candidatePolicy.runtime as string },
    reference: {
      runtime: reference.runtime as string, model: reference.model as string, revisionOrChecksum: reference.revisionOrChecksum as string,
      ...(reference.executionMode === undefined ? {} : { executionMode: nonemptyTraceContext(reference.executionMode, "executionMode") }),
      ...(reference.attentionImplementation === undefined ? {} : { attentionImplementation: nonemptyTraceContext(reference.attentionImplementation, "attentionImplementation") }),
      ...(reference.executionDevice === undefined ? {} : { executionDevice: reference.executionDevice as string }),
      ...(reference.executionDeviceDetail === undefined ? {} : { executionDeviceDetail: reference.executionDeviceDetail as string }),
      containerFormat: reference.containerFormat as string, quantization: reference.quantization as string,
      inputTokens: [...inputTokens], promptPositionIds: [...promptPositionIds],
      dtypePolicy: reference.dtypePolicy as string, maxNewTokens: reference.maxNewTokens as number,
      ...(reference.eosTokenId !== undefined ? { eosTokenId: reference.eosTokenId as number } : {}),
      generatedTokenIds: [...generatedTokenIds], steps, selectionLogits, stepPastKeyValues,
      logits: validateSerializedTensor(reference.logits, "logits terminais", candidatePolicy.dtype as "F32" | "F64"), pastKeyValues,
    },
  };
}

function nonemptyTraceContext(value: unknown, field: string): string {
  if (typeof value !== "string" || value.trim() === "") throw new Error(`Trace ${field} deve ser string não vazia quando declarado.`);
  return value;
}

function validateEnvelope(value: Record<string, unknown>, kind: "execution" | "generation"): void {
  if (value.schemaVersion !== 1 || value.kind !== kind) throw new Error(`Trace requer schemaVersion=1 e kind=${kind}.`);
  const source = object(value.source, "Trace source");
  if (!Array.isArray(source.files) || source.files.length === 0) throw new Error("Trace requer source.files não vazio.");
  const candidatePolicy = object(value.candidatePolicy, "Trace candidatePolicy");
  if ((candidatePolicy.dtype !== "F32" && candidatePolicy.dtype !== "F64") || typeof candidatePolicy.runtime !== "string" || candidatePolicy.runtime.trim() === "") throw new Error("Trace requer candidatePolicy F32/F64 e runtime não vazio.");
  if (typeof value.irFingerprint !== "string" || !/^[a-f0-9]{64}$/.test(value.irFingerprint)) throw new Error("Trace requer irFingerprint SHA-256.");
}

function parseSourceFiles(raw: unknown): TraceSourceFile[] {
  if (!Array.isArray(raw)) throw new Error("Trace requer source.files não vazio.");
  return raw.map((entry) => {
    const file = object(entry, "Trace source file");
    if (typeof file.path !== "string" || typeof file.sha256 !== "string") throw new Error("Trace source file inválido.");
    return { path: file.path, sha256: file.sha256 };
  });
}

function serializedCache(raw: unknown, label: string, dtype: "F32" | "F64"): SerializedCache {
  const cache = object(raw, label);
  if (!Number.isInteger(cache.layer) || (cache.layer as number) < 0) throw new Error(`${label} requer layer inteiro não negativo.`);
  return { layer: cache.layer as number, key: validateSerializedTensor(cache.key, "cache key", dtype), value: validateSerializedTensor(cache.value, "cache value", dtype) };
}

function uniqueCacheLayers(caches: readonly SerializedCache[]): void {
  const layers = new Set<number>();
  for (const cache of caches) {
    if (layers.has(cache.layer)) throw new Error(`Trace de geração contém cache KV duplicado para camada ${cache.layer}.`);
    layers.add(cache.layer);
  }
}

function validateSerializedTensor(raw: unknown, label: string, expectedDtype?: "F32" | "F64"): SerializedTensor {
  const tensor = object(raw, `Tensor ${label}`);
  if ((tensor.dtype !== "F32" && tensor.dtype !== "F64") || (expectedDtype !== undefined && tensor.dtype !== expectedDtype) || !Array.isArray(tensor.shape) || !tensor.shape.every((dimension) => Number.isInteger(dimension) && (dimension as number) > 0) || typeof tensor.valuesBase64 !== "string") {
    throw new Error(`Tensor ${label} deve declarar dtype ${expectedDtype ?? "F32/F64"}, shape positivo e valuesBase64.`);
  }
  decodeTensor({ dtype: tensor.dtype, shape: tensor.shape as number[], valuesBase64: tensor.valuesBase64 }, label);
  return { dtype: tensor.dtype, shape: tensor.shape as number[], valuesBase64: tensor.valuesBase64 };
}

function decodeTensor(tensor: SerializedTensor, label: string): DenseF32Tensor | DenseTensor {
  const elements = tensor.shape.reduce((total, dimension) => total * dimension, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0) throw new Error(`Tensor ${label} possui shape inseguro.`);
  const bytes = Buffer.from(tensor.valuesBase64, "base64");
  const bytesPerElement = tensor.dtype === "F32" ? Float32Array.BYTES_PER_ELEMENT : Float64Array.BYTES_PER_ELEMENT;
  if (bytes.length !== elements * bytesPerElement) throw new Error(`Tensor ${label} possui ${bytes.length} bytes ${tensor.dtype}, esperado ${elements * bytesPerElement}.`);
  const copy = new Uint8Array(bytes);
  if (tensor.dtype === "F32") return { shape: [...tensor.shape], values: new Float32Array(copy.buffer) };
  return { shape: [...tensor.shape], values: new Float64Array(copy.buffer) };
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`${label} deve ser objeto JSON.`);
  return value as Record<string, unknown>;
}

function tokenRow(row: unknown): boolean {
  return Array.isArray(row) && row.length > 0 && row.every((token) => Number.isInteger(token) && (token as number) >= 0);
}

function tokenVector(value: unknown): value is number[] {
  return Array.isArray(value) && value.length > 0 && value.every((token) => Number.isInteger(token) && (token as number) >= 0);
}

function assertRelativePath(file: string, label: string): void {
  if (path.isAbsolute(file) || file === "" || file.split(path.sep).includes("..") || file.includes("/../") || file.includes("\\\\..\\\\")) throw new Error(`${label} deve ser caminho relativo seguro: ${file}.`);
}
