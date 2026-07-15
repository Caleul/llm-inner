import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface, type Interface } from "node:readline";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { JsonObject, LinearPreview, ModelCatalog, TensorInfo } from "./types.js";

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
