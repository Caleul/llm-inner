import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface, type Interface } from "node:readline";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { DenseF32Tensor, JsonObject, LinearPreview, ModelCatalog, QuantizationSpec, TensorInfo } from "./types.js";

interface RpcResponse {
  id: number;
  result?: unknown;
  error?: { message: string; stack?: string };
}

export class TensorBridge {
  readonly #source: string;
  readonly #process: ChildProcessWithoutNullStreams;
  readonly #readline: Interface;
  readonly #pending = new Map<
    number,
    { resolve: (value: unknown) => void; reject: (error: Error) => void }
  >();
  #nextId = 1;
  #stderr = "";

  constructor(source: string, python = "python3") {
    this.#source = source;
    const here = path.dirname(fileURLToPath(import.meta.url));
    const helper = path.resolve(here, "../../helpers/tensor_bridge.py");
    this.#process = spawn(python, [helper, source], { stdio: ["pipe", "pipe", "pipe"] });
    this.#readline = createInterface({ input: this.#process.stdout });
    this.#readline.on("line", (line) => this.#onLine(line));
    this.#process.stderr.on("data", (chunk: Buffer) => {
      this.#stderr += chunk.toString("utf8");
      if (this.#stderr.length > 16_384) this.#stderr = this.#stderr.slice(-16_384);
    });
    this.#process.on("exit", (code) => {
      const message = `Tensor bridge encerrou com código ${code}. ${this.#stderr}`.trim();
      for (const pending of this.#pending.values()) pending.reject(new Error(message));
      this.#pending.clear();
    });
  }

  async close(): Promise<void> {
    if (this.#process.exitCode === null) {
      try {
        await this.#call("close", {});
      } catch {
        // O processo pode já estar encerrando.
      }
      this.#process.stdin.end();
    }
    this.#readline.close();
  }

  async inspectGguf(): Promise<{
    config: JsonObject;
    rawMetadata: JsonObject;
    tensors: TensorInfo[];
  }> {
    return (await this.#call("inspect", {})) as {
      config: JsonObject;
      rawMetadata: JsonObject;
      tensors: TensorInfo[];
    };
  }

  async readLinearPreview(
    catalog: ModelCatalog,
    weightName: string,
    outputRows: number,
    inputTerms: number,
  ): Promise<LinearPreview> {
    const info = catalog.tensors.get(weightName);
    if (!info || info.logicalShape.length !== 2) {
      throw new Error(`Tensor linear inválido: ${weightName}`);
    }
    const [outFeatures, inFeatures] = info.logicalShape;
    if (outFeatures === undefined || inFeatures === undefined) throw new Error(`Shape inválido: ${weightName}`);
    const rows = Math.min(outFeatures, outputRows);
    const cols = Math.min(inFeatures, inputTerms);
    const result = (await this.#call("read_rows", {
      tensor: weightName,
      row_start: 0,
      row_end: rows,
      col_start: 0,
      col_end: cols,
    })) as { values: number[][] };

    return {
      rows: result.values.map((values, outputIndex) => ({
        outputIndex,
        terms: values.map((weight, inputIndex) => ({ inputIndex, weight })),
        omittedInputTerms: Math.max(0, inFeatures - cols),
      })),
      omittedOutputRows: Math.max(0, outFeatures - rows),
    };
  }

  /**
   * Materialize one complete MLX quantized tensor through mlx.core.dequantize.
   * The request carries the catalogued storage and logical layouts so the
   * Python backend cannot silently dequantize a similarly named tensor with a
   * different packing contract.
   */
  async readMlxDequantizedF32(catalog: ModelCatalog, tensorName: string): Promise<DenseF32Tensor> {
    if (catalog.format !== "mlx-safetensors") {
      throw new Error(`Leitura MLX completa requer catálogo mlx-safetensors, recebeu ${catalog.format}.`);
    }
    const info = catalog.tensors.get(tensorName);
    if (!info) throw new Error(`Tensor MLX não catalogado: ${tensorName}`);
    const quantization = requireMlxQuantization(info);
    const result = await this.#call("read_tensor_f32", {
      tensor: tensorName,
      storage_dtype: info.storageDtype,
      storage_shape: [...info.storageShape],
      logical_shape: [...info.logicalShape],
      quantization: {
        family: quantization.family,
        mode: quantization.mode,
        bits: quantization.bits,
        group_size: quantization.groupSize,
        scale_tensor: quantization.scaleTensor,
        bias_tensor: quantization.biasTensor,
        global_scale_tensor: quantization.globalScaleTensor,
      },
    }) as { shape: unknown; f32leBase64: unknown };
    return decodeMlxF32Payload(result, info.logicalShape, quantization);
  }

  async #call(method: string, params: JsonObject): Promise<unknown> {
    const id = this.#nextId++;
    const payload = JSON.stringify({ id, method, params, source: this.#source });
    const promise = new Promise<unknown>((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
    });
    this.#process.stdin.write(`${payload}\n`);
    return promise;
  }

  #onLine(line: string): void {
    let response: RpcResponse;
    try {
      response = JSON.parse(line) as RpcResponse;
    } catch {
      return;
    }
    const pending = this.#pending.get(response.id);
    if (!pending) return;
    this.#pending.delete(response.id);
    if (response.error) {
      pending.reject(new Error(response.error.message));
    } else {
      pending.resolve(response.result);
    }
  }
}

function requireMlxQuantization(info: TensorInfo): QuantizationSpec {
  const quantization = info.quantization;
  if (
    info.storageDtype !== "U32" ||
    quantization?.family !== "mlx" ||
    !quantization.mode ||
    !Number.isInteger(quantization.bits) || quantization.bits! <= 0 || quantization.bits! > 32 ||
    !Number.isInteger(quantization.groupSize) || quantization.groupSize! <= 0 ||
    !quantization.scaleTensor
  ) {
    throw new Error(`${info.name}: tensor não possui contrato MLX U32 completo e verificável.`);
  }
  if (info.logicalShape.length !== 2 || info.storageShape.length !== 2) {
    throw new Error(`${info.name}: executor MLX atual requer shapes 2D de storage e lógicos.`);
  }
  return quantization;
}

/** Exported for byte-level regression tests; backend responses never use JSON number arrays. */
export function decodeMlxF32Payload(
  payload: { shape: unknown; f32leBase64: unknown },
  expectedShape: readonly number[],
  sourceQuantization: QuantizationSpec,
): DenseF32Tensor {
  if (!Array.isArray(payload.shape) || !payload.shape.every((dimension) => Number.isSafeInteger(dimension) && dimension >= 0)) {
    throw new Error("Backend MLX retornou shape F32 inválido.");
  }
  if (payload.shape.length !== expectedShape.length || payload.shape.some((dimension, index) => dimension !== expectedShape[index])) {
    throw new Error(`Backend MLX retornou shape [${payload.shape.join(", ")}], esperado [${expectedShape.join(", ")}].`);
  }
  if (typeof payload.f32leBase64 !== "string" || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(payload.f32leBase64)) {
    throw new Error("Backend MLX retornou payload F32 base64 inválido.");
  }
  const bytes = Buffer.from(payload.f32leBase64, "base64");
  const elements = expectedShape.reduce((product, dimension) => product * dimension, 1);
  if (bytes.byteLength !== elements * Float32Array.BYTES_PER_ELEMENT) {
    throw new Error(`Backend MLX retornou ${bytes.byteLength} bytes F32, esperado ${elements * Float32Array.BYTES_PER_ELEMENT}.`);
  }
  const copied = Uint8Array.from(bytes);
  return {
    shape: [...expectedShape],
    values: new Float32Array(copied.buffer),
    sourceQuantization: { ...sourceQuantization },
  };
}
