import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Gemma4RuntimeReductionProvider, Gemma4RuntimeReductionRequest } from "./gemma4-runtime-reduction-provider.js";
import type { DenseF32Tensor } from "./types.js";

interface SerializedTensor { shape: number[]; values: number[] }
interface HelperResponse {
  schemaVersion: number;
  contractId: string;
  sourceCheckpointAccessed: boolean;
  output: SerializedTensor;
}

/**
 * Source-independent adapter for the exact pinned PyTorch CPU matmul path.
 * Each request contains only already-produced literal intermediates and the
 * structured tower contract; the helper has no checkpoint argument or model
 * loading capability.
 */
export class Gemma4TorchRuntimeReductionProvider implements Gemma4RuntimeReductionProvider {
  readonly contractId = "torch-2.12.1-cpu-inference-matmul-v1" as const;
  readonly #python: string;
  readonly #helper: string;

  constructor(python: string, helper = resolve(dirname(fileURLToPath(import.meta.url)), "../../helpers/torch_gemma4_runtime_reductions.py")) {
    if (!python) throw new Error("Provedor PyTorch Gemma 4 requer executável Python explícito.");
    this.#python = resolve(python);
    this.#helper = resolve(helper);
  }

  execute(request: Gemma4RuntimeReductionRequest): DenseF32Tensor {
    const directory = mkdtempSync(join(tmpdir(), "llm-inner-gemma4-runtime-reduction-"));
    try {
      const requestPath = join(directory, "request.json");
      writeFileSync(requestPath, JSON.stringify(serializeRequest(request)), "utf8");
      const child = spawnSync(this.#python, [this.#helper, requestPath], {
        encoding: "utf8",
        maxBuffer: 128 * 1024 * 1024,
      });
      if (child.error) throw child.error;
      if (child.status !== 0) {
        throw new Error(`${request.operationId}: helper PyTorch encerrou com código ${String(child.status)}: ${child.stderr.trim()}`);
      }
      const response = parseResponse(child.stdout, request.operationId);
      return { shape: [...response.output.shape], values: Float32Array.from(response.output.values) };
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  }
}

function serializeRequest(request: Gemma4RuntimeReductionRequest): object {
  const tower = request.scope === "vision"
    ? {
      attentionHeads: request.program.tower.attentionHeads,
      headDim: request.program.tower.headDim,
      runtimeDtype: request.program.runtimeDtype,
    }
    : {
      attentionHeads: request.program.tower.attentionHeads,
      headDim: request.program.tower.headDim,
      attentionChunkSize: request.program.tower.attentionChunkSize,
      attentionContextLeft: request.program.tower.attentionContextLeft,
      attentionContextRight: request.program.tower.attentionContextRight,
      runtimeDtype: request.program.runtimeDtype,
    };
  return {
    schemaVersion: 1,
    contractId: "torch-2.12.1-cpu-inference-matmul-v1",
    scope: request.scope,
    operationId: request.operationId,
    operation: request.operation,
    tower,
    operands: request.operands.map(serializeTensor),
  };
}

function serializeTensor(tensor: DenseF32Tensor): SerializedTensor {
  return { shape: [...tensor.shape], values: Array.from(tensor.values) };
}

function parseResponse(stdout: string, operationId: string): HelperResponse {
  let parsed: HelperResponse;
  try { parsed = JSON.parse(stdout) as HelperResponse; } catch (error) {
    throw new Error(`${operationId}: helper PyTorch retornou JSON inválido: ${(error as Error).message}`);
  }
  if (parsed.schemaVersion !== 1 || parsed.contractId !== "torch-2.12.1-cpu-inference-matmul-v1" ||
    parsed.sourceCheckpointAccessed !== false || !parsed.output || !Array.isArray(parsed.output.shape) || !Array.isArray(parsed.output.values)) {
    throw new Error(`${operationId}: helper PyTorch retornou envelope incompatível.`);
  }
  return parsed;
}
