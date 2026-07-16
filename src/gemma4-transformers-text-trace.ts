import { spawn } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { openCatalog } from "./catalog.js";
import { buildGemma4CompositeProgram } from "./gemma4-composite.js";
import { fingerprintIR, readGenerationTraceBundle, sha256File, type GenerationTraceBundle, type TraceSourceFile } from "./trace.js";

export interface Gemma4TransformersTextTraceOptions {
  source: string;
  output: string;
  inputTokens: number[];
  positionIds?: number[];
  maxNewTokens: number;
  python: string;
  model: string;
  revisionOrChecksum: string;
  eosTokenId?: number;
}

interface NativeGenerationCapture {
  runtime: string;
  generatedTokenIds: number[];
  steps: GenerationTraceBundle["reference"]["steps"];
  selectionLogits: GenerationTraceBundle["reference"]["selectionLogits"];
  stepPastKeyValues: GenerationTraceBundle["reference"]["stepPastKeyValues"];
  logits: GenerationTraceBundle["reference"]["logits"];
  pastKeyValues: GenerationTraceBundle["reference"]["pastKeyValues"];
}

/**
 * Captures the registered dense BF16 Gemma4 text path from Transformers 5.5.0.
 * It intentionally captures only token-only forward/decode; the composite
 * image/video/audio path has separate semantics and remains outside this trace.
 */
export async function captureGemma4TransformersTextGenerationTrace(options: Gemma4TransformersTextTraceOptions): Promise<void> {
  validateIds(options.inputTokens, "inputTokens");
  if (!Number.isInteger(options.maxNewTokens) || options.maxNewTokens < 0) throw new Error("Gemma 4 Transformers capture requer maxNewTokens inteiro não negativo.");
  const positions = options.positionIds ?? options.inputTokens.map((_, index) => index);
  validateIds(positions, "positionIds");
  if (positions.length !== options.inputTokens.length) throw new Error("Gemma 4 Transformers capture requer posição para cada token.");
  if (options.eosTokenId !== undefined) throw new Error("Gemma 4 Transformers capture ainda não aceita eosTokenId: a captura deve registrar todos os passos solicitados.");

  const opened = await openCatalog(options.source, false);
  try {
    if (opened.catalog.format !== "safetensors" || opened.catalog.config.model_type !== "gemma4") {
      throw new Error("Gemma 4 Transformers capture requer pacote Safetensors Gemma4 registrado.");
    }
    for (const tensor of opened.catalog.tensors.values()) {
      if (tensor.storageDtype !== "BF16" || tensor.quantization) throw new Error(`${tensor.name}: captura Gemma 4 requer armazenamento BF16 denso sem quantização.`);
    }
    const program = buildGemma4CompositeProgram(opened.catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
    const native = await invoke<NativeGenerationCapture>(options.python, {
      source: options.source, inputTokens: options.inputTokens, positionIds: positions, maxNewTokens: options.maxNewTokens,
    });
    const bundle: GenerationTraceBundle = {
      schemaVersion: 1,
      kind: "generation",
      source: { files: await checksums(opened.catalog.source, opened.catalog.tensors.values()) },
      irFingerprint: fingerprintIR(program.textProgram),
      candidatePolicy: { dtype: "F32", runtime: "llm-inner paged Gemma4Text literal F32" },
      reference: {
        runtime: native.runtime,
        model: options.model,
        revisionOrChecksum: options.revisionOrChecksum,
        containerFormat: "safetensors",
        quantization: "none; dense BF16 storage",
        inputTokens: [...options.inputTokens],
        promptPositionIds: positions,
        dtypePolicy: "native eager BF16 operations, tensors captured as F32 for comparison",
        maxNewTokens: options.maxNewTokens,
        generatedTokenIds: native.generatedTokenIds,
        steps: native.steps,
        selectionLogits: native.selectionLogits,
        stepPastKeyValues: native.stepPastKeyValues,
        logits: native.logits,
        pastKeyValues: native.pastKeyValues,
      },
    };
    await mkdir(path.dirname(options.output), { recursive: true });
    await writeFile(options.output, `${JSON.stringify(bundle, null, 2)}\n`, "utf8");
    await readGenerationTraceBundle(options.output);
  } finally {
    await opened.close();
  }
}

function validateIds(values: readonly number[], label: string): void {
  if (values.length === 0 || values.some((value) => !Number.isSafeInteger(value) || value < 0)) throw new Error(`Gemma 4 Transformers capture requer ${label} inteiros não negativos e não vazios.`);
}

async function checksums(source: string, tensors: Iterable<{ shard?: string }>): Promise<TraceSourceFile[]> {
  const files = new Set<string>(["config.json"]);
  for (const tensor of tensors) {
    if (!tensor.shard) throw new Error("Tensor Gemma 4 sem shard não pode gerar trace verificável.");
    files.add(tensor.shard);
  }
  return Promise.all([...files].sort().map(async (file) => ({ path: file, sha256: await sha256File(path.join(source, file)) })));
}

async function invoke<T>(python: string, request: object): Promise<T> {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-transformers-capture-"));
  try {
    const requestPath = path.join(directory, "request.json");
    await writeFile(requestPath, JSON.stringify(request), "utf8");
    const helper = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../helpers/transformers_gemma4_text_trace_capture.py");
    const stdout = await new Promise<string>((resolve, reject) => {
      const child = spawn(python, [helper, requestPath], { stdio: ["ignore", "pipe", "pipe"] });
      let output = "", error = "";
      child.stdout.on("data", (chunk: Buffer) => { output += chunk.toString("utf8"); });
      child.stderr.on("data", (chunk: Buffer) => { error += chunk.toString("utf8"); });
      child.on("error", reject);
      child.on("close", (code) => code === 0 ? resolve(output) : reject(new Error(`Gemma 4 Transformers helper encerrou com código ${code}: ${error.trim()}`)));
    });
    return JSON.parse(stdout) as T;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
