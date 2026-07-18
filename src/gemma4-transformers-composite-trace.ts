import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { openCatalog } from "./catalog.js";
import { buildGemma4CompositeProgram } from "./gemma4-composite.js";
import { assertGemma4AuthoritativeRuntime } from "./gemma4-authoritative-runtime.js";
import { gemma4CompositeTraceProfile, type Gemma4CompositeTraceModality } from "./gemma4-composite-trace-profile.js";
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
  modality: Gemma4CompositeTraceModality;
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
  modality: Gemma4CompositeTraceModality;
  pixelValues?: { shape: number[]; values: number[] };
  imagePositionIds?: number[][][];
  pixelValuesVideos?: { shape: number[]; values: number[] };
  videoPositionIds?: number[][][][];
  inputFeatures?: { shape: number[]; values: number[] };
  inputFeaturesMask?: boolean[][];
  maxNewTokens: number;
  python: string;
  model: string;
  revisionOrChecksum: string;
}

/** Captures real image, video, or audio prefill plus cached greedy decode from pinned Transformers. */
export async function captureGemma4TransformersCompositeTrace(options: Gemma4TransformersCompositeTraceOptions): Promise<void> {
  validateGemma4CompositeTraceOptions(options);
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
      modality: options.modality,
      inputTokens: options.inputTokens,
      positionIds: options.positionIds,
      mmTokenTypeIds: options.mmTokenTypeIds,
      ...(options.pixelValues ? { pixelValues: options.pixelValues } : {}),
      ...(options.imagePositionIds ? { imagePositionIds: options.imagePositionIds } : {}),
      ...(options.pixelValuesVideos ? { pixelValuesVideos: options.pixelValuesVideos } : {}),
      ...(options.videoPositionIds ? { videoPositionIds: options.videoPositionIds } : {}),
      ...(options.inputFeatures ? { inputFeatures: options.inputFeatures } : {}),
      ...(options.inputFeaturesMask ? { inputFeaturesMask: options.inputFeaturesMask } : {}),
      maxNewTokens: options.maxNewTokens,
      executionDevice: "cpu",
    });
    assertGemma4AuthoritativeRuntime("composite", native.runtime);
    if (native.executionDevice !== "cpu" || native.modality !== options.modality || native.prefillOperations.length !== 5) {
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
        modality: options.modality,
        inputTokens: [...options.inputTokens],
        positionIds: [...options.positionIds],
        mmTokenTypeIds: [...options.mmTokenTypeIds],
        ...(options.pixelValues ? { pixelValues: cloneTensor(options.pixelValues) } : {}),
        ...(options.imagePositionIds ? { imagePositionIds: options.imagePositionIds.map((batch) => batch.map((position) => [...position])) } : {}),
        ...(options.pixelValuesVideos ? { pixelValuesVideos: cloneTensor(options.pixelValuesVideos) } : {}),
        ...(options.videoPositionIds ? { videoPositionIds: options.videoPositionIds.map((video) => video.map((frame) => frame.map((position) => [...position]))) } : {}),
        ...(options.inputFeatures ? { inputFeatures: cloneTensor(options.inputFeatures) } : {}),
        ...(options.inputFeaturesMask ? { inputFeaturesMask: options.inputFeaturesMask.map((row) => [...row]) } : {}),
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

export function validateGemma4CompositeTraceOptions(options: Gemma4TransformersCompositeTraceOptions): void {
  const vectors = [options.inputTokens, options.positionIds, options.mmTokenTypeIds];
  if (vectors.some((values) => values.length === 0 || values.some((value) => !Number.isSafeInteger(value) || value < 0)) ||
    options.inputTokens.length !== options.positionIds.length || options.inputTokens.length !== options.mmTokenTypeIds.length) {
    throw new Error("Captura composite requer tokens, posições e tipos multimodais inteiros e alinhados.");
  }
  if (!Number.isSafeInteger(options.maxNewTokens) || options.maxNewTokens < 0) throw new Error("Captura composite requer maxNewTokens válido.");
  const profile = gemma4CompositeTraceProfile(options.modality);
  if (options.mmTokenTypeIds.filter((value) => value === profile.tokenTypeId).length !== (profile.tokenTypeId === 0 ? options.mmTokenTypeIds.length : 1)) {
    throw new Error(`Captura composite ${options.modality} possui mm_token_type_ids incompatível.`);
  }
  const provided = [options.pixelValues !== undefined || options.imagePositionIds !== undefined,
    options.pixelValuesVideos !== undefined || options.videoPositionIds !== undefined,
    options.inputFeatures !== undefined || options.inputFeaturesMask !== undefined];
  const expected = options.modality === "image" ? 0 : options.modality === "video" ? 1 : 2;
  if (provided.filter(Boolean).length !== 1 || !provided[expected]) throw new Error(`Captura composite ${options.modality} requer somente os inputs da modalidade declarada.`);
  if (options.modality === "image") validateVisionInput(options.pixelValues, options.imagePositionIds, false);
  if (options.modality === "video") validateVisionInput(options.pixelValuesVideos, options.videoPositionIds, true);
  if (options.modality === "audio") validateAudioInput(options.inputFeatures, options.inputFeaturesMask);
}

function validateVisionInput(
  tensor: { shape: number[]; values: number[] } | undefined,
  positions: number[][][] | number[][][][] | undefined,
  video: boolean,
): void {
  const rank = video ? 4 : 3;
  if (!tensor || !positions || tensor.shape.length !== rank || tensor.shape.at(-1) !== 768 ||
    tensor.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0) ||
    tensor.shape.reduce((total, dimension) => total * dimension, 1) !== tensor.values.length || tensor.values.some((value) => !Number.isFinite(value))) {
    throw new Error(`Captura composite requer pixels ${video ? "video [videos,frames,patches,768]" : "image [images,patches,768]"} válidos.`);
  }
  const flattened = video ? (positions as number[][][][]).flat(1) : positions as number[][][];
  const batches = video ? tensor.shape[0]! * tensor.shape[1]! : tensor.shape[0]!;
  const patches = tensor.shape.at(-2)!;
  if (flattened.length !== batches || flattened.some((batch) => batch.length !== patches || batch.some((position) => position.length !== 2 || position.some((value) => !Number.isSafeInteger(value))))) {
    throw new Error(`Captura composite requer posições ${video ? "video" : "image"} alinhadas aos pixels.`);
  }
}

function validateAudioInput(tensor: { shape: number[]; values: number[] } | undefined, mask: boolean[][] | undefined): void {
  if (!tensor || !mask || tensor.shape.length !== 3 || tensor.shape[2] !== 128 ||
    tensor.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0) ||
    tensor.shape.reduce((total, dimension) => total * dimension, 1) !== tensor.values.length || tensor.values.some((value) => !Number.isFinite(value)) ||
    mask.length !== tensor.shape[0] || mask.some((row) => row.length !== tensor.shape[1] || row.some((value) => typeof value !== "boolean"))) {
    throw new Error("Captura composite requer input_features [batch,frames,128] e mask alinhada.");
  }
}

function cloneTensor(tensor: { shape: number[]; values: number[] }): { shape: number[]; values: number[] } {
  return { shape: [...tensor.shape], values: [...tensor.values] };
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
