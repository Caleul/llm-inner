import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { buildModelIR } from "./architecture.js";
import { openCatalog } from "./catalog.js";
import { fingerprintIR, type ExecutionTraceBundle, type GenerationTraceBundle, type TraceSourceFile } from "./trace.js";

export interface MlxCaptureOptions {
  source: string;
  output: string;
  inputTokens: number[];
  positionIds?: number[];
  maxNewTokens?: number;
  python: string;
  model: string;
  revisionOrChecksum: string;
}

/**
 * Capture a deliberately narrow independent reference trace through MLX
 * kernels. This consumes a prevalidated IR for one of the explicitly listed
 * adapters; it never guesses architecture or dequantization semantics, and
 * remains separate from the scalar candidate executor used by trace replay.
 */
export async function captureMlxTrace(options: MlxCaptureOptions): Promise<"execution" | "generation"> {
  if (options.inputTokens.length === 0 || options.inputTokens.some((token) => !Number.isInteger(token) || token < 0)) throw new Error("MLX capture requer inputTokens inteiros não negativos.");
  const positions = options.positionIds ?? options.inputTokens.map((_, index) => index);
  if (positions.length !== options.inputTokens.length || positions.some((position) => !Number.isInteger(position) || position < 0)) throw new Error("MLX capture requer positionIds inteiros não negativos para todo input token.");
  const opened = await openCatalog(options.source, false);
  try {
    if (opened.catalog.format !== "safetensors" && opened.catalog.format !== "mlx-safetensors") {
      throw new Error(`MLX capture requer Safetensors denso ou MLX affine; recebeu ${opened.catalog.format}.`);
    }
    const captureQuantization = validateMlxCaptureStorage(opened.catalog.tensors.values());
    const ir = await buildModelIR(opened.catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
    if (!MLX_CAPTURE_MODEL_TYPES.has(ir.architecture.modelType)) {
      throw new Error(`MLX capture não possui contrato independente para ${ir.architecture.modelType}; suportados: ${[...MLX_CAPTURE_MODEL_TYPES].join(", ")}.`);
    }
    if (captureQuantization && ir.architecture.modelType !== "llama") {
      throw new Error(`MLX capture quantizado possui contrato independente somente para llama; recebeu ${ir.architecture.modelType}.`);
    }
    const source = { files: await checksums(opened.catalog.source, opened.catalog.tensors.values()) };
    const common = {
      schemaVersion: 1 as const,
      source,
      irFingerprint: fingerprintIR(ir),
      candidatePolicy: { dtype: "F32" as const, runtime: "llm-inner scalar IEEE-754 F32" },
    };
    if (options.maxNewTokens === undefined) {
      const reference = await invoke(options.python, { kind: "execution", source: options.source, ir, inputTokens: options.inputTokens, positionIds: positions });
      const bundle: ExecutionTraceBundle = {
        ...common, kind: "execution",
        reference: {
          runtime: mlxRuntime(ir.architecture.modelType, captureQuantization), model: options.model, revisionOrChecksum: options.revisionOrChecksum,
          containerFormat: captureQuantization ? "mlx-safetensors" : "safetensors", quantization: mlxQuantizationLabel(captureQuantization), inputTokens: [options.inputTokens], positionIds: [positions], dtypePolicy: "MLX float32 kernel capture",
          operations: reference.operations, pastKeyValues: reference.pastKeyValues,
        },
      };
      await mkdir(path.dirname(options.output), { recursive: true });
      await writeFile(options.output, `${JSON.stringify(bundle, null, 2)}\n`, "utf8");
      return "execution";
    }
    if (!Number.isInteger(options.maxNewTokens) || options.maxNewTokens < 0) throw new Error("maxNewTokens deve ser inteiro não negativo.");
    const reference = await invoke(options.python, { kind: "generation", source: options.source, ir, inputTokens: options.inputTokens, positionIds: positions, maxNewTokens: options.maxNewTokens });
    const bundle: GenerationTraceBundle = {
      ...common, kind: "generation",
      reference: {
        runtime: mlxRuntime(ir.architecture.modelType, captureQuantization), model: options.model, revisionOrChecksum: options.revisionOrChecksum,
        containerFormat: captureQuantization ? "mlx-safetensors" : "safetensors", quantization: mlxQuantizationLabel(captureQuantization), inputTokens: [...options.inputTokens], promptPositionIds: positions,
        dtypePolicy: "MLX float32 kernel capture", maxNewTokens: options.maxNewTokens, generatedTokenIds: reference.generatedTokenIds,
        steps: reference.steps, selectionLogits: reference.selectionLogits, stepPastKeyValues: reference.stepPastKeyValues,
        logits: reference.logits, pastKeyValues: reference.pastKeyValues,
      },
    };
    await mkdir(path.dirname(options.output), { recursive: true });
    await writeFile(options.output, `${JSON.stringify(bundle, null, 2)}\n`, "utf8");
    return "generation";
  } finally {
    await opened.close();
  }
}

// Each entry has an end-to-end MLX regression that exercises semantics not
// shared by the Llama baseline. New adapters must be added here deliberately;
// a generic "supported IR" fallback would hide unreviewed model behavior.
const MLX_CAPTURE_MODEL_TYPES = new Set(["llama", "mistral", "gemma", "gemma2", "qwen2", "qwen3"]);

interface MlxAffineCaptureSpec { bits: number; groupSize: number; parameterDtype: "F32" | "F16" | "BF16"; }

/**
 * A checkpoint may intentionally override MLX's root affine settings for
 * individual modules. Capture provenance must retain every admitted contract:
 * reducing this to one checkpoint-wide bit width or parameter dtype would
 * make the reference trace describe a different storage program.
 */
interface MlxAffineCaptureStorage { contracts: readonly MlxAffineCaptureSpec[]; }

/**
 * This is intentionally stricter than cataloging: MLX is the independent
 * backend only for affine U32 tensors whose exact dequantize invocation is
 * represented in the IR. Other MLX modes remain fail-closed here even if a
 * bridge supports them for candidate materialization.
 */
function validateMlxCaptureStorage(tensors: Iterable<{ name: string; storageDtype: string; quantization?: { family: string; mode: string; bits?: number; groupSize?: number; scaleTensor?: string; biasTensor?: string } }>): MlxAffineCaptureStorage | undefined {
  const catalogued = new Map<string, { name: string; storageDtype: string; quantization?: { family: string; mode: string; bits?: number; groupSize?: number; scaleTensor?: string; biasTensor?: string } }>();
  for (const tensor of tensors) catalogued.set(tensor.name, tensor);
  const contracts = new Map<string, MlxAffineCaptureSpec>();
  const affineParameterNames = new Set<string>();
  for (const tensor of catalogued.values()) {
    if (!tensor.quantization) {
      continue;
    }
    const q = tensor.quantization;
    if (q.family !== "mlx" || q.mode !== "affine" || tensor.storageDtype !== "U32" || !Number.isInteger(q.bits) || ![2, 3, 4, 5, 6, 8].includes(q.bits!) || ![32, 64].includes(q.groupSize ?? -1)) {
      throw new Error(`${tensor.name}: MLX capture não possui contrato independente para ${q.family}/${q.mode} ${tensor.storageDtype}; requer affine U32 com bits {2,3,4,5,6,8} e group_size MLX 32 ou 64 validado.`);
    }
    if (!q.scaleTensor) throw new Error(`${tensor.name}: MLX capture affine requer tensor de scales declarado.`);
    const scales = catalogued.get(q.scaleTensor);
    const biases = q.biasTensor ? catalogued.get(q.biasTensor) : undefined;
    if (!scales || !["F32", "F16", "BF16"].includes(scales.storageDtype) || (q.biasTensor && (!biases || biases.storageDtype !== scales.storageDtype))) {
      throw new Error(`${tensor.name}: MLX capture affine requer scales e biases opcionais no mesmo dtype F32, F16 ou BF16.`);
    }
    affineParameterNames.add(q.scaleTensor);
    if (q.biasTensor) affineParameterNames.add(q.biasTensor);
    const current = { bits: q.bits!, groupSize: q.groupSize!, parameterDtype: scales.storageDtype as "F32" | "F16" | "BF16" };
    contracts.set(affineContractKey(current), current);
  }
  for (const tensor of catalogued.values()) {
    if (!tensor.quantization && tensor.storageDtype !== "F32" && !affineParameterNames.has(tensor.name)) {
      throw new Error(`${tensor.name}: MLX capture requer tensor denso F32, peso MLX affine U32, ou parâmetro affine F16/BF16 declarado.`);
    }
  }
  return contracts.size === 0
    ? undefined
    : { contracts: [...contracts.values()].sort(compareAffineContracts) };
}

function mlxRuntime(modelType: string, affine: MlxAffineCaptureStorage | undefined): string {
  return affine
    ? `MLX 0.32 ${mlxAffineCaptureDescription(affine, "-", "affine-U32")} ${modelType} independent IR-kernel capture`
    : `MLX 0.32 dense-F32 ${modelType} independent IR-kernel capture`;
}

function mlxQuantizationLabel(affine: MlxAffineCaptureStorage | undefined): string {
  return affine ? `MLX affine U32 ${mlxAffineCaptureDescription(affine, "_", "")}` : "none";
}

function mlxAffineCaptureDescription(storage: MlxAffineCaptureStorage, separator: "-" | "_", prefix: "affine-U32" | ""): string {
  const groupLabel = separator === "-" ? "group-" : "group_size=";
  const contracts = storage.contracts.map((contract) => `${contract.bits}-bit ${groupLabel}${contract.groupSize} ${contract.parameterDtype}-parameters`);
  return contracts.length === 1
    ? `${prefix}${prefix ? " " : ""}${contracts[0]}`
    : `${prefix}${prefix ? " " : ""}mixed-contracts [${contracts.join("; ")}]`;
}

function affineContractKey(contract: MlxAffineCaptureSpec): string {
  return `${contract.bits}/${contract.groupSize}/${contract.parameterDtype}`;
}

function compareAffineContracts(left: MlxAffineCaptureSpec, right: MlxAffineCaptureSpec): number {
  return left.bits - right.bits || left.groupSize - right.groupSize || left.parameterDtype.localeCompare(right.parameterDtype);
}

async function checksums(source: string, tensors: Iterable<{ shard?: string }>): Promise<TraceSourceFile[]> {
  const files = new Set<string>(["config.json"]);
  for (const tensor of tensors) {
    if (!tensor.shard) throw new Error("Safetensors catalogado sem shard não pode gerar trace MLX verificável.");
    files.add(tensor.shard);
  }
  return Promise.all([...files].sort().map(async (file) => ({ path: file, sha256: createHash("sha256").update(await readFile(path.join(source, file))).digest("hex") })));
}

async function invoke(python: string, request: object): Promise<any> {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-mlx-capture-"));
  try {
    const requestPath = path.join(directory, "request.json");
    await writeFile(requestPath, JSON.stringify(request), "utf8");
    const helper = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../helpers/mlx_trace_capture.py");
    const output = await new Promise<string>((resolve, reject) => {
      const process = spawn(python, [helper, requestPath], { stdio: ["ignore", "pipe", "pipe"] });
      let stdout = ""; let stderr = "";
      process.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString("utf8"); });
      process.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString("utf8"); });
      process.on("error", reject);
      process.on("close", (code) => code === 0 ? resolve(stdout) : reject(new Error(`MLX capture helper encerrou com código ${code}: ${stderr.trim()}`)));
    });
    return JSON.parse(output) as any;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
