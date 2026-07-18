import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { openCatalog } from "./catalog.js";
import { buildGemma4CompositeProgram } from "./gemma4-composite.js";
import { fingerprintIR, readGenerationTraceBundle, sha256File, type GenerationTraceBundle } from "./trace.js";

interface SerializedF32Tensor {
  dtype: "F32";
  shape: number[];
  valuesBase64: string;
}

interface SerializedOperation {
  operationId: string;
  output: string;
  tensor: SerializedF32Tensor;
}

interface NativeCompositeCapture {
  runtime: string;
  executionDevice: string;
  executionDeviceDetail: string;
  generatedTokenIds: number[];
  steps: GenerationTraceBundle["reference"]["steps"];
  selectionLogits: GenerationTraceBundle["reference"]["selectionLogits"];
  stepPastKeyValues: GenerationTraceBundle["reference"]["stepPastKeyValues"];
  logits: GenerationTraceBundle["reference"]["logits"];
  pastKeyValues: GenerationTraceBundle["reference"]["pastKeyValues"];
  prefillOperations: SerializedOperation[];
}

export interface Gemma4TransformersCompositeTraceOptions {
  source: string;
  output: string;
  inputTokens: number[];
  positionIds: number[];
  mmTokenTypeIds: number[];
  pixelValues: { shape: number[]; values: number[] };
  imagePositionIds: number[][][];
  maxNewTokens: number;
  python: string;
  model: string;
  revisionOrChecksum: string;
}

/** Captures real image-prefill plus cached greedy decode from pinned Transformers. */
export async function captureGemma4TransformersCompositeTrace(options: Gemma4TransformersCompositeTraceOptions): Promise<void> {
  validate(options);
  const opened = await openCatalog(options.source, false);
  try {
    if (opened.catalog.format !== "safetensors" || opened.catalog.config.model_type !== "gemma4") {
      throw new Error("Captura composite requer pacote Safetensors Gemma 4 registrado.");
    }
    for (const tensor of opened.catalog.tensors.values()) {
      if (tensor.storageDtype !== "BF16" || tensor.quantization) throw new Error(`${tensor.name}: captura composite requer BF16 denso sem quantização.`);
    }
    const program = buildGemma4CompositeProgram(opened.catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
    const native = await invoke<NativeCompositeCapture>(options.python, {
      source: options.source,
      inputTokens: options.inputTokens,
      positionIds: options.positionIds,
      mmTokenTypeIds: options.mmTokenTypeIds,
      pixelValues: options.pixelValues,
      imagePositionIds: options.imagePositionIds,
      maxNewTokens: options.maxNewTokens,
      executionDevice: "cpu",
    });
    if (native.executionDevice !== "cpu" || native.prefillOperations.length !== 5) {
      throw new Error("Helper composite não declarou CPU e cinco fronteiras de prefill.");
    }
    const bundle: GenerationTraceBundle & {
      compositeInputs: Omit<Gemma4TransformersCompositeTraceOptions, "source" | "output" | "python" | "model" | "revisionOrChecksum">;
      prefillOperations: SerializedOperation[];
    } = {
      schemaVersion: 1,
      kind: "generation",
      source: { files: [
        { path: "config.json", sha256: await sha256File(path.join(options.source, "config.json")) },
        { path: "model.safetensors", sha256: await sha256File(path.join(options.source, "model.safetensors")) },
      ] },
      irFingerprint: fingerprintIR(program.textProgram),
      candidatePolicy: { dtype: "F32", runtime: "llm-inner embedded-literal Gemma4 composite BF16-policy executor" },
      reference: {
        runtime: native.runtime,
        executionDevice: native.executionDevice,
        executionDeviceDetail: native.executionDeviceDetail,
        model: options.model,
        revisionOrChecksum: options.revisionOrChecksum,
        containerFormat: "safetensors",
        quantization: "none; dense BF16 storage",
        inputTokens: [...options.inputTokens],
        promptPositionIds: [...options.positionIds],
        dtypePolicy: "native eager BF16 composite operations, tensors captured as F32 for comparison",
        maxNewTokens: options.maxNewTokens,
        generatedTokenIds: native.generatedTokenIds,
        steps: native.steps,
        selectionLogits: native.selectionLogits,
        stepPastKeyValues: native.stepPastKeyValues,
        logits: native.logits,
        pastKeyValues: native.pastKeyValues,
      },
      compositeInputs: {
        inputTokens: [...options.inputTokens],
        positionIds: [...options.positionIds],
        mmTokenTypeIds: [...options.mmTokenTypeIds],
        pixelValues: { shape: [...options.pixelValues.shape], values: [...options.pixelValues.values] },
        imagePositionIds: options.imagePositionIds.map((batch) => batch.map((position) => [...position])),
        maxNewTokens: options.maxNewTokens,
      },
      prefillOperations: native.prefillOperations,
    };
    await mkdir(path.dirname(options.output), { recursive: true });
    await writeFile(options.output, `${JSON.stringify(bundle)}\n`, "utf8");
    await readGenerationTraceBundle(options.output);
    const raw = JSON.parse(await readFile(options.output, "utf8")) as { prefillOperations?: unknown[] };
    if (raw.prefillOperations?.length !== 5) throw new Error("Trace composite perdeu fronteiras de prefill na serialização.");
  } finally {
    await opened.close();
  }
}

function validate(options: Gemma4TransformersCompositeTraceOptions): void {
  const vectors = [options.inputTokens, options.positionIds, options.mmTokenTypeIds];
  if (vectors.some((values) => values.length === 0 || values.some((value) => !Number.isSafeInteger(value) || value < 0)) ||
    options.inputTokens.length !== options.positionIds.length || options.inputTokens.length !== options.mmTokenTypeIds.length) {
    throw new Error("Captura composite requer tokens, posições e tipos multimodais inteiros e alinhados.");
  }
  const validPixelShape = options.pixelValues.shape.length === 3 && options.pixelValues.shape.at(-1) === 768 &&
    options.pixelValues.shape.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0);
  const elements = validPixelShape ? options.pixelValues.shape.reduce((total, dimension) => total * dimension, 1) : 0;
  if (!validPixelShape || elements !== options.pixelValues.values.length ||
    options.pixelValues.values.some((value) => !Number.isFinite(value)) || options.imagePositionIds.length !== options.pixelValues.shape[0] ||
    options.imagePositionIds.some((batch) => batch.length !== options.pixelValues.shape[1] || batch.some((position) => position.length !== 2)) ||
    !Number.isSafeInteger(options.maxNewTokens) || options.maxNewTokens < 0) {
    throw new Error("Captura composite requer pixel_values F32 [images,patches,768], posições [images,patches,2] e maxNewTokens válido.");
  }
}

async function invoke<T>(python: string, request: object): Promise<T> {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-composite-capture-"));
  try {
    const requestPath = path.join(directory, "request.json");
    await writeFile(requestPath, JSON.stringify(request), "utf8");
    const helper = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../helpers/transformers_gemma4_composite_trace_capture.py");
    const stdout = await new Promise<string>((resolve, reject) => {
      const child = spawn(python, [helper, requestPath], { stdio: ["ignore", "pipe", "pipe"] });
      let output = "", error = "";
      child.stdout.on("data", (chunk: Buffer) => { output += chunk.toString("utf8"); });
      child.stderr.on("data", (chunk: Buffer) => { error += chunk.toString("utf8"); });
      child.on("error", reject);
      child.on("close", (code) => code === 0 ? resolve(output) : reject(new Error(`Helper composite encerrou com código ${code}: ${error.trim()}`)));
    });
    return JSON.parse(stdout) as T;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
