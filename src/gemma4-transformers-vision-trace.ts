import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { openCatalog } from "./catalog.js";
import type { Gemma4VisionDifferentialTrace, Gemma4VisionInvocation } from "./gemma4-literal-vision.js";
import type { DenseF32Tensor, DifferentialOperationSample } from "./types.js";

interface SerializedTensor { shape: number[]; values: number[] }
interface NativeVisionCapture {
  runtime: string;
  executionDevice: "cpu";
  dtypePolicy: string;
  pixelValues: SerializedTensor;
  pixelPositionIds: number[][][] | number[][][][];
  operations: Array<{ operationId: string; output: string; tensor: SerializedTensor }>;
}

export interface Gemma4TransformersVisionTraceOptions {
  source: string;
  output: string;
  invocation: Gemma4VisionInvocation;
  pixelValues: DenseF32Tensor;
  pixelPositionIds: number[][][] | number[][][][];
  python: string;
  model: string;
  revisionOrChecksum: string;
  executionDevice: "cpu";
}

/** Capture real image/video boundaries from the pinned package/runtime. */
export async function captureGemma4TransformersVisionTrace(options: Gemma4TransformersVisionTraceOptions): Promise<void> {
  const opened = await openCatalog(options.source, false);
  try {
    if (opened.catalog.format !== "safetensors" || opened.catalog.config.model_type !== "gemma4") throw new Error("Captura vision Gemma 4 requer pacote Safetensors Gemma4 registrado.");
    for (const tensor of opened.catalog.tensors.values()) if (tensor.storageDtype !== "BF16" || tensor.quantization) throw new Error(`${tensor.name}: captura vision requer armazenamento BF16 denso sem quantização.`);
    assertInput(options.invocation, options.pixelValues, options.pixelPositionIds);
    const native = await invoke<NativeVisionCapture>(options.python, {
      source: options.source,
      invocation: options.invocation,
      executionDevice: options.executionDevice,
      pixelValues: serializeTensor(options.pixelValues),
      pixelPositionIds: options.pixelPositionIds,
    });
    const trace: Gemma4VisionDifferentialTrace = {
      schemaVersion: 1,
      kind: "gemma4-vision-checkpoints",
      invocation: options.invocation,
      source: { model: options.model, revisionOrChecksum: options.revisionOrChecksum, containerFormat: "safetensors", quantization: "none; dense BF16 storage" },
      reference: {
        runtime: native.runtime,
        executionDevice: native.executionDevice,
        dtypePolicy: native.dtypePolicy,
        pixelValues: dense(native.pixelValues),
        pixelPositionIds: native.pixelPositionIds,
        operations: native.operations.map((operation): DifferentialOperationSample => ({ operationId: operation.operationId, output: operation.output, tensor: dense(operation.tensor) })),
      },
    };
    await writeFile(options.output, `${JSON.stringify(serializeTrace(trace), null, 2)}\n`, "utf8");
  } finally {
    await opened.close();
  }
}

export async function readGemma4VisionDifferentialTrace(file: string): Promise<Gemma4VisionDifferentialTrace> {
  const parsed = JSON.parse(await readFile(file, "utf8")) as Record<string, unknown>;
  const reference = parsed.reference as NativeVisionCapture | undefined;
  if (!reference?.pixelValues || !Array.isArray(reference.pixelPositionIds) || !Array.isArray(reference.operations)) throw new Error("Trace vision Gemma 4 não possui referência válida.");
  return {
    ...(parsed as unknown as Gemma4VisionDifferentialTrace),
    reference: { ...reference, pixelValues: dense(reference.pixelValues), operations: reference.operations.map((operation) => ({ ...operation, tensor: dense(operation.tensor) })) },
  };
}

function serializeTrace(trace: Gemma4VisionDifferentialTrace): object {
  return { ...trace, reference: { ...trace.reference, pixelValues: serializeTensor(trace.reference.pixelValues), operations: trace.reference.operations.map((operation) => ({ ...operation, tensor: serializeTensor(operation.tensor as DenseF32Tensor) })) } };
}
function serializeTensor(tensor: DenseF32Tensor): SerializedTensor { return { shape: [...tensor.shape], values: Array.from(tensor.values) }; }
function dense(tensor: SerializedTensor): DenseF32Tensor {
  if (!Array.isArray(tensor.shape) || !Array.isArray(tensor.values) || tensor.shape.some((value) => !Number.isSafeInteger(value) || value < 0) || tensor.values.length !== tensor.shape.reduce((total, dimension) => total * dimension, 1)) throw new Error("Trace vision Gemma 4 contém tensor inválido.");
  return { shape: [...tensor.shape], values: Float32Array.from(tensor.values) };
}
function assertInput(invocation: Gemma4VisionInvocation, input: DenseF32Tensor, positions: number[][][] | number[][][][]): void {
  const rank = invocation === "image" ? 3 : 4;
  if (input.shape.length !== rank || input.shape.at(-1) !== 768 || input.values.length !== input.shape.reduce((total, dimension) => total * dimension, 1) || positions.length !== input.shape[0]) throw new Error(`Captura Gemma 4 ${invocation} requer pixel_values rank ${rank} com width 768 e posições compatíveis.`);
}
async function invoke<T>(python: string, request: object): Promise<T> {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-vision-capture-"));
  try {
    const requestPath = path.join(directory, "request.json");
    await writeFile(requestPath, JSON.stringify(request), "utf8");
    const helper = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../helpers/transformers_gemma4_vision_trace_capture.py");
    const stdout = await new Promise<string>((resolve, reject) => {
      const child = spawn(python, [helper, requestPath], { stdio: ["ignore", "pipe", "pipe"] });
      let output = "", error = "";
      child.stdout.on("data", (chunk: Buffer) => { output += chunk.toString("utf8"); });
      child.stderr.on("data", (chunk: Buffer) => { error += chunk.toString("utf8"); });
      child.on("error", reject);
      child.on("close", (code) => code === 0 ? resolve(output) : reject(new Error(`Helper vision Gemma 4 encerrou com código ${code}: ${error.trim()}`)));
    });
    return JSON.parse(stdout) as T;
  } finally { await rm(directory, { recursive: true, force: true }); }
}
