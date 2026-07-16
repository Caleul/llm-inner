import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { buildModelIR } from "./architecture.js";
import { openCatalog } from "./catalog.js";
import { fingerprintIR, type ExecutionTraceBundle, type GenerationTraceBundle, type TraceSourceFile } from "./trace.js";

export interface TransformersCaptureOptions {
  source: string;
  output: string;
  inputTokens: number[];
  positionIds?: number[];
  maxNewTokens?: number;
  python: string;
  model: string;
  revisionOrChecksum: string;
}

interface TransformersExecutionCapture {
  runtime: string;
  operations: ExecutionTraceBundle["reference"]["operations"];
  pastKeyValues: ExecutionTraceBundle["reference"]["pastKeyValues"];
}

interface TransformersGenerationCapture {
  runtime: string;
  generatedTokenIds: number[];
  steps: GenerationTraceBundle["reference"]["steps"];
  selectionLogits: GenerationTraceBundle["reference"]["selectionLogits"];
  stepPastKeyValues: GenerationTraceBundle["reference"]["stepPastKeyValues"];
  logits: GenerationTraceBundle["reference"]["logits"];
  pastKeyValues: GenerationTraceBundle["reference"]["pastKeyValues"];
}

/**
 * Captures a native Hugging Face Transformers Llama execution without sending
 * the candidate IR to Python.  The helper is intentionally a narrowly pinned
 * architecture adapter: it validates a dense-F32 LlamaForCausalLM package,
 * forces its documented eager attention path, and rejects every different
 * model family or storage contract instead of reverse-engineering behavior
 * from tensor names.
 */
export async function captureTransformersLlamaTrace(options: TransformersCaptureOptions): Promise<"execution" | "generation"> {
  validateTokens(options.inputTokens, "inputTokens");
  const positions = options.positionIds ?? options.inputTokens.map((_, index) => index);
  validateTokens(positions, "positionIds");
  if (positions.length !== options.inputTokens.length) throw new Error("Transformers capture requer positionIds para todo input token.");

  const opened = await openCatalog(options.source, false);
  try {
    if (opened.catalog.format !== "safetensors") {
      throw new Error(`Transformers Llama capture requer Safetensors denso; recebeu ${opened.catalog.format}.`);
    }
    for (const tensor of opened.catalog.tensors.values()) {
      if (tensor.storageDtype !== "F32" || tensor.quantization !== undefined) {
        throw new Error(`${tensor.name}: Transformers Llama capture requer todo tensor em storage F32 denso sem quantização.`);
      }
    }
    const ir = await buildModelIR(opened.catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
    if (ir.architecture.modelType !== "llama") {
      throw new Error(`Transformers capture possui adaptador nativo somente para llama; recebeu ${ir.architecture.modelType}.`);
    }
    const source = { files: await checksums(opened.catalog.source, opened.catalog.tensors.values()) };
    const common = {
      schemaVersion: 1 as const,
      source,
      irFingerprint: fingerprintIR(ir),
      candidatePolicy: { dtype: "F32" as const, runtime: "llm-inner scalar IEEE-754 F32" },
    };
    if (options.maxNewTokens === undefined) {
      const reference = await invoke<TransformersExecutionCapture>(options.python, {
        kind: "execution", source: options.source, inputTokens: options.inputTokens, positionIds: positions,
      });
      const bundle: ExecutionTraceBundle = {
        ...common,
        kind: "execution",
        reference: {
          runtime: reference.runtime,
          model: options.model,
          revisionOrChecksum: options.revisionOrChecksum,
          containerFormat: "safetensors",
          quantization: "none",
          inputTokens: [options.inputTokens],
          positionIds: [positions],
          dtypePolicy: "PyTorch float32 eager LlamaForCausalLM native capture",
          operations: reference.operations,
          pastKeyValues: reference.pastKeyValues,
        },
      };
      await writeBundle(options.output, bundle);
      return "execution";
    }
    if (!Number.isInteger(options.maxNewTokens) || options.maxNewTokens < 0) throw new Error("maxNewTokens deve ser inteiro não negativo.");
    const reference = await invoke<TransformersGenerationCapture>(options.python, {
      kind: "generation", source: options.source, inputTokens: options.inputTokens, positionIds: positions, maxNewTokens: options.maxNewTokens,
    });
    const bundle: GenerationTraceBundle = {
      ...common,
      kind: "generation",
      reference: {
        runtime: reference.runtime,
        model: options.model,
        revisionOrChecksum: options.revisionOrChecksum,
        containerFormat: "safetensors",
        quantization: "none",
        inputTokens: [...options.inputTokens],
        promptPositionIds: positions,
        dtypePolicy: "PyTorch float32 eager LlamaForCausalLM native capture",
        maxNewTokens: options.maxNewTokens,
        generatedTokenIds: reference.generatedTokenIds,
        steps: reference.steps,
        selectionLogits: reference.selectionLogits,
        stepPastKeyValues: reference.stepPastKeyValues,
        logits: reference.logits,
        pastKeyValues: reference.pastKeyValues,
      },
    };
    await writeBundle(options.output, bundle);
    return "generation";
  } finally {
    await opened.close();
  }
}

function validateTokens(values: readonly number[], name: string): void {
  if (values.length === 0 || values.some((value) => !Number.isInteger(value) || value < 0)) {
    throw new Error(`Transformers capture requer ${name} inteiros não negativos e não vazios.`);
  }
}

async function writeBundle(output: string, bundle: ExecutionTraceBundle | GenerationTraceBundle): Promise<void> {
  await mkdir(path.dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(bundle, null, 2)}\n`, "utf8");
}

async function checksums(source: string, tensors: Iterable<{ shard?: string }>): Promise<TraceSourceFile[]> {
  const files = new Set<string>(["config.json"]);
  for (const tensor of tensors) {
    if (!tensor.shard) throw new Error("Safetensors catalogado sem shard não pode gerar trace Transformers verificável.");
    files.add(tensor.shard);
  }
  return Promise.all([...files].sort().map(async (file) => ({
    path: file,
    sha256: createHash("sha256").update(await readFile(path.join(source, file))).digest("hex"),
  })));
}

async function invoke<T>(python: string, request: object): Promise<T> {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-transformers-capture-"));
  try {
    const requestPath = path.join(directory, "request.json");
    await writeFile(requestPath, JSON.stringify(request), "utf8");
    const helper = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../helpers/transformers_llama_trace_capture.py");
    const output = await new Promise<string>((resolve, reject) => {
      const process = spawn(python, [helper, requestPath], { stdio: ["ignore", "pipe", "pipe"] });
      let stdout = "";
      let stderr = "";
      process.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString("utf8"); });
      process.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString("utf8"); });
      process.on("error", reject);
      process.on("close", (code) => code === 0
        ? resolve(stdout)
        : reject(new Error(`Transformers Llama capture helper encerrou com código ${code}: ${stderr.trim()}`)));
    });
    return JSON.parse(output) as T;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
