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
 * kernels. This consumes a prevalidated dense F32 IR for one of the explicitly
 * listed adapters; it never guesses architecture or dequantization semantics,
 * and remains separate from the scalar candidate executor used by trace replay.
 */
export async function captureMlxTrace(options: MlxCaptureOptions): Promise<"execution" | "generation"> {
  if (options.inputTokens.length === 0 || options.inputTokens.some((token) => !Number.isInteger(token) || token < 0)) throw new Error("MLX capture requer inputTokens inteiros não negativos.");
  const positions = options.positionIds ?? options.inputTokens.map((_, index) => index);
  if (positions.length !== options.inputTokens.length || positions.some((position) => !Number.isInteger(position) || position < 0)) throw new Error("MLX capture requer positionIds inteiros não negativos para todo input token.");
  const opened = await openCatalog(options.source, false);
  try {
    if (opened.catalog.format !== "safetensors") throw new Error(`MLX capture requer Safetensors denso; recebeu ${opened.catalog.format}.`);
    for (const tensor of opened.catalog.tensors.values()) {
      if (tensor.storageDtype !== "F32" || tensor.quantization) throw new Error(`${tensor.name}: MLX capture requer todos os tensors F32 densos e não quantizados.`);
    }
    const ir = await buildModelIR(opened.catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
    if (!MLX_CAPTURE_MODEL_TYPES.has(ir.architecture.modelType)) {
      throw new Error(`MLX capture não possui contrato independente para ${ir.architecture.modelType}; suportados: ${[...MLX_CAPTURE_MODEL_TYPES].join(", ")}.`);
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
          runtime: `MLX 0.32 dense-F32 ${ir.architecture.modelType} independent IR-kernel capture`, model: options.model, revisionOrChecksum: options.revisionOrChecksum,
          containerFormat: "safetensors", quantization: "none", inputTokens: [options.inputTokens], dtypePolicy: "MLX float32 kernel capture",
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
        runtime: `MLX 0.32 dense-F32 ${ir.architecture.modelType} independent IR-kernel capture`, model: options.model, revisionOrChecksum: options.revisionOrChecksum,
        containerFormat: "safetensors", quantization: "none", inputTokens: [...options.inputTokens], promptPositionIds: positions,
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
