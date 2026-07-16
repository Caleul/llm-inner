import { spawn } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { openCatalog } from "./catalog.js";
import { buildGemma4CompositeProgram } from "./gemma4-composite.js";
import { fingerprintIR, sha256File, type ExecutionTraceBundle, type TraceSourceFile } from "./trace.js";
import type { Operation } from "./types.js";

export interface Gemma4TransformersOperationTraceOptions {
  source: string;
  output: string;
  inputTokens: number[];
  positionIds?: number[];
  python: string;
  model: string;
  revisionOrChecksum: string;
}

interface NativeOperationCapture {
  runtime: string;
  operations: ExecutionTraceBundle["reference"]["operations"];
  pastKeyValues: ExecutionTraceBundle["reference"]["pastKeyValues"];
}

/**
 * The native helper may only emit a complete assignment trace.  Its IDs and
 * outputs are tied to the registered adapter rather than accepted as an
 * opportunistic collection of module hooks.
 */
export function assertGemma4NativeOperationCoverage(
  expected: readonly Pick<Operation, "id" | "output">[],
  actual: readonly Pick<ExecutionTraceBundle["reference"]["operations"][number], "operationId" | "output">[],
): void {
  const seen = new Set<string>();
  const byId = new Map<string, string>();
  for (const operation of actual) {
    if (seen.has(operation.operationId)) throw new Error(`Gemma 4 native trace duplicou a operação ${operation.operationId}.`);
    seen.add(operation.operationId);
    byId.set(operation.operationId, operation.output);
  }
  for (const operation of expected) {
    const output = byId.get(operation.id);
    if (output === undefined) throw new Error(`Gemma 4 native trace não capturou a atribuição declarada ${operation.id}.`);
    if (output !== operation.output) throw new Error(`Gemma 4 native trace declarou output '${output}' para ${operation.id}; esperado '${operation.output}'.`);
  }
  const expectedIds = new Set(expected.map((operation) => operation.id));
  const unexpected = actual.find((operation) => !expectedIds.has(operation.operationId));
  if (unexpected) throw new Error(`Gemma 4 native trace contém operação não declarada ${unexpected.operationId}.`);
}

/**
 * Captures every declared Gemma4Text assignment from the pinned native eager
 * BF16 path. The Python helper rejects instrumentation that changes final
 * logits or KV cache before this complete trace is written.
 */
export async function captureGemma4TransformersOperationTrace(options: Gemma4TransformersOperationTraceOptions): Promise<void> {
  validateIds(options.inputTokens, "inputTokens");
  const positions = options.positionIds ?? options.inputTokens.map((_, index) => index);
  validateIds(positions, "positionIds");
  if (positions.length !== options.inputTokens.length) throw new Error("Gemma 4 checkpoint capture requer uma posição para cada token.");
  const opened = await openCatalog(options.source, false);
  try {
    if (opened.catalog.format !== "safetensors" || opened.catalog.config.model_type !== "gemma4") {
      throw new Error("Gemma 4 checkpoint capture requer pacote Safetensors Gemma4 registrado.");
    }
    for (const tensor of opened.catalog.tensors.values()) {
      if (tensor.storageDtype !== "BF16" || tensor.quantization) throw new Error(`${tensor.name}: checkpoint capture requer armazenamento BF16 denso sem quantização.`);
    }
    const program = buildGemma4CompositeProgram(opened.catalog, { outputRows: 1, inputTerms: 1, includeWeights: false });
    const native = await invoke<NativeOperationCapture>(options.python, {
      source: options.source,
      inputTokens: options.inputTokens,
      positionIds: positions,
      mode: "operation-checkpoints",
    });
    const expected = [...program.textProgram.prelude, ...program.textProgram.layers.flatMap((layer) => layer.operations), ...program.textProgram.epilogue];
    assertGemma4NativeOperationCoverage(expected, native.operations);
    const bundle: ExecutionTraceBundle = {
      schemaVersion: 1,
      kind: "execution",
      source: { files: await checksums(opened.catalog.source, opened.catalog.tensors.values()) },
      irFingerprint: fingerprintIR(program.textProgram),
      candidatePolicy: { dtype: "F32", runtime: "llm-inner paged Gemma4Text literal F32" },
      reference: {
        runtime: native.runtime,
        model: options.model,
        revisionOrChecksum: options.revisionOrChecksum,
        containerFormat: "safetensors",
        quantization: "none; dense BF16 storage",
        inputTokens: [options.inputTokens],
        positionIds: [positions],
        dtypePolicy: "native eager BF16 module checkpoints captured as F32",
        operations: native.operations,
        pastKeyValues: native.pastKeyValues,
      },
    };
    await mkdir(path.dirname(options.output), { recursive: true });
    await writeFile(options.output, `${JSON.stringify(bundle, null, 2)}\n`, "utf8");
  } finally {
    await opened.close();
  }
}

function validateIds(values: readonly number[], label: string): void {
  if (values.length === 0 || values.some((value) => !Number.isSafeInteger(value) || value < 0)) throw new Error(`Gemma 4 checkpoint capture requer ${label} inteiros não negativos e não vazios.`);
}

async function checksums(source: string, tensors: Iterable<{ shard?: string }>): Promise<TraceSourceFile[]> {
  const files = new Set<string>(["config.json"]);
  for (const tensor of tensors) {
    if (!tensor.shard) throw new Error("Tensor Gemma 4 sem shard não pode gerar checkpoint trace verificável.");
    files.add(tensor.shard);
  }
  return Promise.all([...files].sort().map(async (file) => ({ path: file, sha256: await sha256File(path.join(source, file)) })));
}

async function invoke<T>(python: string, request: object): Promise<T> {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-checkpoint-capture-"));
  try {
    const requestPath = path.join(directory, "request.json");
    await writeFile(requestPath, JSON.stringify(request), "utf8");
    const helper = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../helpers/transformers_gemma4_text_trace_capture.py");
    const output = await new Promise<string>((resolve, reject) => {
      const child = spawn(python, [helper, requestPath], { stdio: ["ignore", "pipe", "pipe"] });
      let stdout = "", stderr = "";
      child.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString("utf8"); });
      child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString("utf8"); });
      child.on("error", reject);
      child.on("close", (code) => code === 0 ? resolve(stdout) : reject(new Error(`Gemma 4 checkpoint helper encerrou com código ${code}: ${stderr.trim()}`)));
    });
    return JSON.parse(output) as T;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
