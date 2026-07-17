import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { openCatalog } from "./catalog.js";
import { buildGemma4CompositeProgram } from "./gemma4-composite.js";
import { fingerprintIR, sha256File, type ExecutionTraceBundle, type TraceSourceFile } from "./trace.js";
import type { Operation } from "./types.js";

export interface Gemma4TransformersOperationTraceOptions {
  source: string;
  output: string;
  inputTokens: number[];
  positionIds?: number[];
  python: string;
  model: string;
  revisionOrChecksum: string;
  /** Deliberate native kernel contract; never infer this from host capability. */
  executionDevice: "cpu" | "mps";
}

export interface Gemma4TransformersLinearReductionTraceOptions extends Omit<Gemma4TransformersOperationTraceOptions, "output"> {
  output: string;
  /** A declared, bias-free Gemma4Text linear projection such as layer_1_o_proj. */
  operationId: string;
  /**
   * Diagnostic-only BF16 activation scale for a native Linear probe. It is
   * recorded in the trace and cannot stand in for model-forward fidelity.
   */
  activationScale?: number;
  /** Exact BF16 scalar bits for a non-power-of-two diagnostic activation. */
  activationBf16ScaleBits?: number;
}

interface NativeOperationCapture {
  runtime: string;
  executionDevice: string;
  executionDeviceDetail: string;
  nativeKernelEnvironment?: ExecutionTraceBundle["reference"]["nativeKernelEnvironment"];
  operations: ExecutionTraceBundle["reference"]["operations"];
  operationDtypes?: ExecutionTraceBundle["reference"]["operationDtypes"];
  operationLayouts?: ExecutionTraceBundle["reference"]["operationLayouts"];
  pastKeyValues: ExecutionTraceBundle["reference"]["pastKeyValues"];
}

/**
 * Captures only a declared MLP projection and the named assignment that
 * produces its activation.  Unlike the full 1,229-assignment trace, this
 * narrow evidence bundle is small enough to repeat across a reduction-profile
 * campaign.  It remains bound to the complete adapter fingerprint, immutable
 * source checksums, and a native forward that is bitwise unchanged by hooks.
 */
export async function captureGemma4TransformersLinearReductionTrace(options: Gemma4TransformersLinearReductionTraceOptions): Promise<void> {
  validateIds(options.inputTokens, "inputTokens");
  const positions = options.positionIds ?? options.inputTokens.map((_, index) => index);
  validateIds(positions, "positionIds");
  if (positions.length !== options.inputTokens.length) throw new Error("Gemma 4 checkpoint capture requer uma posição para cada token.");
  const activationScale = options.activationScale ?? 1;
  assertActivationScale(activationScale);
  if (options.activationBf16ScaleBits !== undefined && activationScale !== 1) {
    throw new Error("Gemma 4 activationScale e activationBf16ScaleBits são mutuamente exclusivos.");
  }
  if (options.activationBf16ScaleBits !== undefined) assertActivationBf16ScaleBits(options.activationBf16ScaleBits);
  const opened = await openCatalog(options.source, false);
  try {
    if (opened.catalog.format !== "safetensors" || opened.catalog.config.model_type !== "gemma4") {
      throw new Error("Gemma 4 checkpoint capture requer pacote Safetensors Gemma4 registrado.");
    }
    for (const tensor of opened.catalog.tensors.values()) {
      if (tensor.storageDtype !== "BF16" || tensor.quantization) throw new Error(`${tensor.name}: checkpoint capture requer armazenamento BF16 denso sem quantização.`);
    }
    const program = buildGemma4CompositeProgram(opened.catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
    const target = selectGemma4TextLinearCaptureTarget(program.textProgram, options.operationId);
    const native = await invoke<NativeOperationCapture>(options.python, {
      source: options.source,
      inputTokens: options.inputTokens,
      positionIds: positions,
      executionDevice: options.executionDevice,
      mode: "linear-reduction-checkpoint",
      layerIndex: target.layer,
      projection: target.projection,
      operationId: target.operationId,
      producerOperationId: target.producerOperationId,
      producerOutput: target.producerOutput,
      output: target.output,
      activationScale,
      ...(options.activationBf16ScaleBits === undefined ? {} : { activationBf16ScaleBits: options.activationBf16ScaleBits }),
    });
    assertNativeExecutionDevice(native.executionDevice, options.executionDevice);
    assertNativeKernelEnvironment(native.nativeKernelEnvironment);
    assertGemma4NativeOperationCoverage([
      { id: target.producerOperationId, output: target.producerOutput },
      { id: target.operationId, output: target.output },
    ], native.operations);
    assertNativeLinearDtypeCoverage(native.operationDtypes, target.operationId);
    assertNativeLinearLayoutCoverage(native.operationLayouts, target.operationId);
    const bundle: ExecutionTraceBundle = {
      schemaVersion: 1,
      kind: "execution",
      captureId: randomUUID(),
      source: { files: await checksums(opened.catalog.source, opened.catalog.tensors.values()) },
      irFingerprint: fingerprintIR(program.textProgram),
      candidatePolicy: { dtype: "F32", runtime: "llm-inner paged Gemma4Text literal F32" },
      reference: {
        runtime: native.runtime,
        executionDevice: native.executionDevice,
        executionDeviceDetail: native.executionDeviceDetail,
        nativeKernelEnvironment: native.nativeKernelEnvironment,
        model: options.model,
        revisionOrChecksum: options.revisionOrChecksum,
        containerFormat: "safetensors",
        quantization: "none; dense BF16 storage",
        inputTokens: [options.inputTokens],
        positionIds: [positions],
        dtypePolicy: "native eager BF16 bounded Gemma4Text linear projection checkpoints captured as F32",
        reductionProbeInput: options.activationBf16ScaleBits !== undefined
          ? { kind: "bf16-scalar-scale" as const, factorBf16Bits: options.activationBf16ScaleBits }
          : activationScale === 1
            ? { kind: "model-forward" as const }
            : { kind: "bf16-power-of-two-scale" as const, factor: activationScale },
        operations: native.operations,
        operationDtypes: native.operationDtypes,
        operationLayouts: native.operationLayouts,
        pastKeyValues: native.pastKeyValues,
      },
    };
    await mkdir(path.dirname(options.output), { recursive: true });
    await writeFile(options.output, `${JSON.stringify(bundle, null, 2)}\n`, "utf8");
  } finally {
    await opened.close();
  }
}

function assertActivationScale(value: number): void {
  if (!Number.isSafeInteger(value) || value < 1 || value > 256 || (value & (value - 1)) !== 0) {
    throw new Error("Gemma 4 activationScale requer potência de dois inteira entre 1 e 256.");
  }
}

function assertActivationBf16ScaleBits(value: number): void {
  const exponent = Number.isSafeInteger(value) ? (value >>> 7) & 0xff : 0xff;
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff || exponent === 0xff || (value & 0x7fff) === 0) {
    throw new Error("Gemma 4 activationBf16ScaleBits requer BF16 finito, não zero e uint16.");
  }
}

function assertNativeLinearDtypeCoverage(
  operationDtypes: ExecutionTraceBundle["reference"]["operationDtypes"],
  operationId: string,
): asserts operationDtypes is NonNullable<ExecutionTraceBundle["reference"]["operationDtypes"]> {
  const record = operationDtypes?.find((entry) => entry.operationId === operationId);
  if (!record || !record.inputDtype || !record.outputDtype || !record.parameterDtype) {
    throw new Error(`Gemma 4 native trace não declarou dtypes de entrada, saída e parâmetro para ${operationId}.`);
  }
}

function assertNativeLinearLayoutCoverage(
  operationLayouts: ExecutionTraceBundle["reference"]["operationLayouts"],
  operationId: string,
): asserts operationLayouts is NonNullable<ExecutionTraceBundle["reference"]["operationLayouts"]> {
  const record = operationLayouts?.find((entry) => entry.operationId === operationId);
  if (!record || !record.parameter) throw new Error(`Gemma 4 native trace não declarou layouts de entrada, saída e parâmetro para ${operationId}.`);
}

function assertNativeKernelEnvironment(
  environment: NativeOperationCapture["nativeKernelEnvironment"],
): asserts environment is NonNullable<NativeOperationCapture["nativeKernelEnvironment"]> {
  if (!environment || !/^[a-f0-9]{64}$/.test(environment.torchBuildConfigSha256) ||
    !Number.isSafeInteger(environment.intraopThreads) || environment.intraopThreads <= 0 ||
    !Number.isSafeInteger(environment.interopThreads) || environment.interopThreads <= 0 ||
    typeof environment.deterministicAlgorithms !== "boolean" || typeof environment.mkldnnAvailable !== "boolean" ||
    typeof environment.mkldnnEnabled !== "boolean") {
    throw new Error("Gemma 4 helper não declarou nativeKernelEnvironment válido.");
  }
}

export interface Gemma4TextLinearCaptureTarget {
  operationId: string;
  producerOperationId: string;
  producerOutput: string;
  output: string;
  layer: number;
  projection: "gate_proj" | "up_proj" | "down_proj" | "o_proj";
}

/**
 * Restricts bounded native captures to declared Gemma4Text projections with a
 * module-owned native hook target. Attention `o_proj` is included because its
 * exact input is the explicit attention assignment, not an inferred MLP-like
 * surrogate. Q/K/V remain deliberately outside this probe contract.
 */
export function selectGemma4TextLinearCaptureTarget(program: ReturnType<typeof buildGemma4CompositeProgram>["textProgram"], operationId: string): Gemma4TextLinearCaptureTarget {
  const operation = [...program.prelude, ...program.layers.flatMap((layer) => layer.operations), ...program.epilogue]
    .find((entry) => entry.id === operationId);
  if (!operation || operation.op !== "linear" || operation.layer === undefined || operation.bias || !operation.transposeWeight) {
    throw new Error(`${operationId}: captura limitada requer uma projeção linear Gemma4Text declarada, sem bias e transposta.`);
  }
  const matched = /^layer_(\d+)_(gate|up|down|o)_proj$/.exec(operation.id);
  if (!matched || Number(matched[1]) !== operation.layer) {
    throw new Error(`${operationId}: captura limitada aceita somente layer_<n>_(gate|up|down|o)_proj.`);
  }
  const producer = [...program.prelude, ...program.layers.flatMap((layer) => layer.operations), ...program.epilogue]
    .find((entry) => entry.output === operation.input);
  if (!producer) throw new Error(`${operationId}: não encontrou atribuição produtora declarada para '${operation.input}'.`);
  return {
    operationId: operation.id,
    producerOperationId: producer.id,
    producerOutput: producer.output,
    output: operation.output,
    layer: operation.layer,
    projection: `${matched[2]}_proj` as Gemma4TextLinearCaptureTarget["projection"],
  };
}

/**
 * Compatibility export for existing callers that intentionally need the
 * narrower MLP-only policy. New reduction investigations should use the
 * Gemma4Text selector above so each declared native module can opt in
 * explicitly rather than being classified by matrix shape.
 */
export function selectGemma4MlpLinearCaptureTarget(program: ReturnType<typeof buildGemma4CompositeProgram>["textProgram"], operationId: string): Gemma4TextLinearCaptureTarget {
  if (!/^layer_\d+_(gate|up|down)_proj$/.test(operationId)) {
    throw new Error(`${operationId}: captura limitada aceita somente layer_<n>_(gate|up|down)_proj.`);
  }
  const target = selectGemma4TextLinearCaptureTarget(program, operationId);
  return target;
}

/**
 * The native helper may only emit a complete assignment trace.  Its IDs and
 * outputs are tied to the registered adapter rather than accepted as an
 * opportunistic collection of module hooks.
 */
export function assertGemma4NativeOperationCoverage(
  expected: readonly Pick<Operation, "id" | "output">[],
  actual: readonly Pick<ExecutionTraceBundle["reference"]["operations"][number], "operationId" | "output">[],
): void {
  const seen = new Set<string>();
  const byId = new Map<string, string>();
  for (const operation of actual) {
    if (seen.has(operation.operationId)) throw new Error(`Gemma 4 native trace duplicou a operação ${operation.operationId}.`);
    seen.add(operation.operationId);
    byId.set(operation.operationId, operation.output);
  }
  for (const operation of expected) {
    const output = byId.get(operation.id);
    if (output === undefined) throw new Error(`Gemma 4 native trace não capturou a atribuição declarada ${operation.id}.`);
    if (output !== operation.output) throw new Error(`Gemma 4 native trace declarou output '${output}' para ${operation.id}; esperado '${operation.output}'.`);
  }
  const expectedIds = new Set(expected.map((operation) => operation.id));
  const unexpected = actual.find((operation) => !expectedIds.has(operation.operationId));
  if (unexpected) throw new Error(`Gemma 4 native trace contém operação não declarada ${unexpected.operationId}.`);
}

/**
 * Captures every declared Gemma4Text assignment from the pinned native eager
 * BF16 path. The Python helper rejects instrumentation that changes final
 * logits or KV cache before this complete trace is written.
 */
export async function captureGemma4TransformersOperationTrace(options: Gemma4TransformersOperationTraceOptions): Promise<void> {
  validateIds(options.inputTokens, "inputTokens");
  const positions = options.positionIds ?? options.inputTokens.map((_, index) => index);
  validateIds(positions, "positionIds");
  if (positions.length !== options.inputTokens.length) throw new Error("Gemma 4 checkpoint capture requer uma posição para cada token.");
  const opened = await openCatalog(options.source, false);
  try {
    if (opened.catalog.format !== "safetensors" || opened.catalog.config.model_type !== "gemma4") {
      throw new Error("Gemma 4 checkpoint capture requer pacote Safetensors Gemma4 registrado.");
    }
    for (const tensor of opened.catalog.tensors.values()) {
      if (tensor.storageDtype !== "BF16" || tensor.quantization) throw new Error(`${tensor.name}: checkpoint capture requer armazenamento BF16 denso sem quantização.`);
    }
    const program = buildGemma4CompositeProgram(opened.catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
    const native = await invoke<NativeOperationCapture>(options.python, {
      source: options.source,
      inputTokens: options.inputTokens,
      positionIds: positions,
      executionDevice: options.executionDevice,
      mode: "operation-checkpoints",
    });
    assertNativeExecutionDevice(native.executionDevice, options.executionDevice);
    assertNativeKernelEnvironment(native.nativeKernelEnvironment);
    const expected = [...program.textProgram.prelude, ...program.textProgram.layers.flatMap((layer) => layer.operations), ...program.textProgram.epilogue];
    assertGemma4NativeOperationCoverage(expected, native.operations);
    const bundle: ExecutionTraceBundle = {
      schemaVersion: 1,
      kind: "execution",
      captureId: randomUUID(),
      source: { files: await checksums(opened.catalog.source, opened.catalog.tensors.values()) },
      irFingerprint: fingerprintIR(program.textProgram),
      candidatePolicy: { dtype: "F32", runtime: "llm-inner paged Gemma4Text literal F32" },
      reference: {
        runtime: native.runtime,
        executionDevice: native.executionDevice,
        executionDeviceDetail: native.executionDeviceDetail,
        nativeKernelEnvironment: native.nativeKernelEnvironment,
        model: options.model,
        revisionOrChecksum: options.revisionOrChecksum,
        containerFormat: "safetensors",
        quantization: "none; dense BF16 storage",
        inputTokens: [options.inputTokens],
        positionIds: [positions],
        dtypePolicy: "native eager BF16 module checkpoints captured as F32",
        operations: native.operations,
        pastKeyValues: native.pastKeyValues,
      },
    };
    await mkdir(path.dirname(options.output), { recursive: true });
    await writeFile(options.output, `${JSON.stringify(bundle, null, 2)}\n`, "utf8");
  } finally {
    await opened.close();
  }
}

function assertNativeExecutionDevice(actual: string, expected: "cpu" | "mps"): void {
  if (actual !== expected) throw new Error(`Gemma 4 helper declarou executionDevice '${actual}', esperado '${expected}'.`);
}

function validateIds(values: readonly number[], label: string): void {
  if (values.length === 0 || values.some((value) => !Number.isSafeInteger(value) || value < 0)) throw new Error(`Gemma 4 checkpoint capture requer ${label} inteiros não negativos e não vazios.`);
}

async function checksums(source: string, tensors: Iterable<{ shard?: string }>): Promise<TraceSourceFile[]> {
  const files = new Set<string>(["config.json"]);
  for (const tensor of tensors) {
    if (!tensor.shard) throw new Error("Tensor Gemma 4 sem shard não pode gerar checkpoint trace verificável.");
    files.add(tensor.shard);
  }
  return Promise.all([...files].sort().map(async (file) => ({ path: file, sha256: await sha256File(path.join(source, file)) })));
}

async function invoke<T>(python: string, request: object): Promise<T> {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-checkpoint-capture-"));
  try {
    const requestPath = path.join(directory, "request.json");
    await writeFile(requestPath, JSON.stringify(request), "utf8");
    const helper = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../helpers/transformers_gemma4_text_trace_capture.py");
    const output = await new Promise<string>((resolve, reject) => {
      const child = spawn(python, [helper, requestPath], { stdio: ["ignore", "pipe", "pipe"] });
      let stdout = "", stderr = "";
      child.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString("utf8"); });
      child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString("utf8"); });
      child.on("error", reject);
      child.on("close", (code) => code === 0 ? resolve(stdout) : reject(new Error(`Gemma 4 checkpoint helper encerrou com código ${code}: ${stderr.trim()}`)));
    });
    return JSON.parse(output) as T;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
