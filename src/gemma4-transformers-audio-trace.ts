import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { openCatalog } from "./catalog.js";
import type { Gemma4AudioDifferentialTrace } from "./gemma4-literal-audio.js";
import type { DenseF32Tensor, DifferentialOperationSample } from "./types.js";

interface SerializedTensor { shape: number[]; values: number[] }
interface NativeAudioCapture {
  runtime: string;
  executionDevice: "cpu";
  dtypePolicy: string;
  inputFeatures: SerializedTensor;
  inputFeaturesMask: boolean[][];
  operations: Array<{ operationId: string; output: string; tensor: SerializedTensor }>;
}

export interface Gemma4TransformersAudioTraceOptions {
  source: string;
  output: string;
  inputFeatures: DenseF32Tensor;
  inputFeaturesMask: boolean[][];
  python: string;
  model: string;
  revisionOrChecksum: string;
  executionDevice: "cpu";
}

/** Capture real audio module boundaries from the pinned package/runtime. */
export async function captureGemma4TransformersAudioTrace(options: Gemma4TransformersAudioTraceOptions): Promise<void> {
  const opened = await openCatalog(options.source, false);
  try {
    if (opened.catalog.format !== "safetensors" || opened.catalog.config.model_type !== "gemma4") {
      throw new Error("Captura de áudio Gemma 4 requer pacote Safetensors Gemma4 registrado.");
    }
    for (const tensor of opened.catalog.tensors.values()) {
      if (tensor.storageDtype !== "BF16" || tensor.quantization) throw new Error(`${tensor.name}: captura de áudio requer armazenamento BF16 denso sem quantização.`);
    }
    assertInput(options.inputFeatures, options.inputFeaturesMask);
    const native = await invoke<NativeAudioCapture>(options.python, {
      source: options.source,
      executionDevice: options.executionDevice,
      inputFeatures: { shape: options.inputFeatures.shape, values: Array.from(options.inputFeatures.values) },
      inputFeaturesMask: options.inputFeaturesMask,
    });
    const trace: Gemma4AudioDifferentialTrace = {
      schemaVersion: 1,
      kind: "gemma4-audio-checkpoints",
      source: {
        model: options.model,
        revisionOrChecksum: options.revisionOrChecksum,
        containerFormat: "safetensors",
        quantization: "none; dense BF16 storage",
      },
      reference: {
        runtime: native.runtime,
        executionDevice: native.executionDevice,
        dtypePolicy: native.dtypePolicy,
        inputFeatures: dense(native.inputFeatures),
        inputFeaturesMask: native.inputFeaturesMask,
        operations: native.operations.map((operation): DifferentialOperationSample => ({
          operationId: operation.operationId,
          output: operation.output,
          tensor: dense(operation.tensor),
        })),
      },
    };
    await writeFile(options.output, `${JSON.stringify(serializeTrace(trace), null, 2)}\n`, "utf8");
  } finally {
    await opened.close();
  }
}

export async function readGemma4AudioDifferentialTrace(file: string): Promise<Gemma4AudioDifferentialTrace> {
  const parsed = JSON.parse(await readFile(file, "utf8")) as Record<string, unknown>;
  const reference = parsed.reference as NativeAudioCapture | undefined;
  if (!reference || !reference.inputFeatures || !Array.isArray(reference.operations)) throw new Error("Trace de áudio Gemma 4 não possui referência válida.");
  return {
    ...(parsed as unknown as Gemma4AudioDifferentialTrace),
    reference: {
      ...reference,
      inputFeatures: dense(reference.inputFeatures),
      operations: reference.operations.map((operation) => ({ ...operation, tensor: dense(operation.tensor) })),
    },
  };
}

function serializeTrace(trace: Gemma4AudioDifferentialTrace): object {
  return {
    ...trace,
    reference: {
      ...trace.reference,
      inputFeatures: serializeTensor(trace.reference.inputFeatures),
      operations: trace.reference.operations.map((operation) => ({ ...operation, tensor: serializeTensor(operation.tensor as DenseF32Tensor) })),
    },
  };
}

function serializeTensor(tensor: DenseF32Tensor): SerializedTensor {
  return { shape: [...tensor.shape], values: Array.from(tensor.values) };
}

function dense(tensor: SerializedTensor): DenseF32Tensor {
  if (!Array.isArray(tensor.shape) || !Array.isArray(tensor.values) || tensor.shape.some((value) => !Number.isSafeInteger(value) || value < 0) ||
    tensor.values.length !== tensor.shape.reduce((total, dimension) => total * dimension, 1)) throw new Error("Trace de áudio Gemma 4 contém tensor inválido.");
  return { shape: [...tensor.shape], values: Float32Array.from(tensor.values) };
}

function assertInput(input: DenseF32Tensor, mask: boolean[][]): void {
  if (input.shape.length !== 3 || input.shape[2] !== 128 || input.values.length !== input.shape.reduce((total, dimension) => total * dimension, 1) ||
    mask.length !== input.shape[0] || mask.some((row) => row.length !== input.shape[1])) {
    throw new Error("Captura de áudio Gemma 4 requer input_features [B,T,128] e mask [B,T] compatíveis.");
  }
}

async function invoke<T>(python: string, request: object): Promise<T> {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-audio-capture-"));
  try {
    const requestPath = path.join(directory, "request.json");
    await writeFile(requestPath, JSON.stringify(request), "utf8");
    const helper = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../helpers/transformers_gemma4_audio_trace_capture.py");
    const stdout = await new Promise<string>((resolve, reject) => {
      const child = spawn(python, [helper, requestPath], { stdio: ["ignore", "pipe", "pipe"] });
      let output = "", error = "";
      child.stdout.on("data", (chunk: Buffer) => { output += chunk.toString("utf8"); });
      child.stderr.on("data", (chunk: Buffer) => { error += chunk.toString("utf8"); });
      child.on("error", reject);
      child.on("close", (code) => code === 0 ? resolve(output) : reject(new Error(`Helper de áudio Gemma 4 encerrou com código ${code}: ${error.trim()}`)));
    });
    return JSON.parse(stdout) as T;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
