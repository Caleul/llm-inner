import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import type { Gemma4LiteralGenerationAssignmentExecution, Gemma4LiteralGenerationValue } from "./gemma4-literal-generation.js";
import { executeGemma4PagedTextLiteralF32, generateGemma4PagedTextLiteralF32 } from "./gemma4-paged-text.js";
import type { DenseF32Tensor, ReferenceF32ExecutionResult, ReferenceF32KeyValueCache } from "./types.js";

interface Arguments {
  artifact: string;
  inputIds: number[];
  maxNewTokens: number;
  eosTokenId?: number;
  maxReadBytes: number;
  allowUnverifiedFidelity: boolean;
  output?: string;
}

const args = parseArguments(process.argv.slice(2));
const artifact = await openGemma4CompositeLiteralArtifact(args.artifact);
const started = performance.now();
const rssBefore = process.memoryUsage().rss;
try {
  const request = { inputIds: [args.inputIds] };
  const result = args.maxNewTokens === 0
    ? await executeGemma4PagedTextLiteralF32(artifact, request, { maxReadBytes: args.maxReadBytes, allowUnverifiedFidelity: args.allowUnverifiedFidelity })
    : await generateGemma4PagedTextLiteralF32(artifact, { ...request, maxNewTokens: args.maxNewTokens, ...(args.eosTokenId === undefined ? {} : { eosTokenId: args.eosTokenId }) }, { maxReadBytes: args.maxReadBytes, allowUnverifiedFidelity: args.allowUnverifiedFidelity });
  const logits = result.logits;
  const hash = createHash("sha256").update(Buffer.from(logits.values.buffer, logits.values.byteOffset, logits.values.byteLength)).digest("hex");
  const report = {
    kind: "gemma4-paged-text-literal-replay",
    artifact: artifact.artifact,
    artifactBytes: artifact.artifactBytes,
    sourceCheckpointAccessed: false,
    executionFidelity: artifact.program.textProgram.fidelity.exactByConstruction
      ? "exact-by-construction"
      : "unverified-fidelity-explicitly-acknowledged",
    executionScope: "text-only; image/video/audio inputs are intentionally unsupported by this command",
    inputIds: args.inputIds,
    maxReadBytes: args.maxReadBytes,
    maxNewTokens: args.maxNewTokens,
    ...("generatedTokenIds" in result ? { generatedTokenIds: result.generatedTokenIds } : {}),
    ...("assignmentExecutions" in result ? { generationProgramExecution: result.assignmentExecutions.map(summarizeAssignmentExecution) } : {}),
    logitsShape: logits.shape,
    logitsSha256: hash,
    kvProducerLayers: [...result.pastKeyValues.keys()],
    elapsedMilliseconds: Math.round((performance.now() - started) * 1000) / 1000,
    rssBefore,
    rssAfter: process.memoryUsage().rss,
    // Node normalizes ru_maxrss to KiB. Unlike rssAfter, this preserves the
    // process high-water mark for a real paged replay report.
    maxRssKiB: process.resourceUsage().maxRSS,
  };
  const json = `${JSON.stringify(report, null, 2)}\n`;
  if (args.output) await writeFile(args.output, json);
  else process.stdout.write(json);
} finally {
  await artifact.close();
}

function parseArguments(argv: string[]): Arguments {
  let artifact: string | undefined, inputIds: number[] | undefined, output: string | undefined, eosTokenId: number | undefined;
  let maxNewTokens = 0, maxReadMiB = 16, allowUnverifiedFidelity = false;
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index], next = argv[index + 1];
    if (value === "--artifact") { artifact = next; index += 1; }
    else if (value === "--input-ids") { inputIds = parseIds(next); index += 1; }
    else if (value === "--max-new-tokens") { maxNewTokens = parseInteger(next, value); index += 1; }
    else if (value === "--eos-token-id") { eosTokenId = parseInteger(next, value); index += 1; }
    else if (value === "--max-read-mib") { maxReadMiB = parseInteger(next, value); index += 1; }
    else if (value === "--allow-unverified-fidelity") { allowUnverifiedFidelity = true; }
    else if (value === "--output") { output = next; index += 1; }
    else throw new Error(`Argumento desconhecido: ${value}.`);
  }
  if (!artifact || !inputIds || inputIds.length === 0 || maxNewTokens < 0 || maxReadMiB <= 0) {
    throw new Error("Uso: --artifact <literal.json> --input-ids <id,id,...> [--max-new-tokens N] [--eos-token-id N] [--max-read-mib N] [--allow-unverified-fidelity] [--output report.json].");
  }
  const maxReadBytes = maxReadMiB * 1024 * 1024;
  if (!Number.isSafeInteger(maxReadBytes)) throw new Error("--max-read-mib excede limite seguro.");
  return { artifact, inputIds, maxNewTokens, ...(eosTokenId === undefined ? {} : { eosTokenId }), maxReadBytes, allowUnverifiedFidelity, ...(output ? { output } : {}) };
}

function parseIds(value: string | undefined): number[] {
  if (!value) throw new Error("--input-ids requer ao menos um ID.");
  const ids = value.split(",").map((entry) => parseInteger(entry, "--input-ids"));
  if (ids.some((id) => id < 0)) throw new Error("--input-ids requer IDs não negativos.");
  return ids;
}

function parseInteger(value: string | undefined, flag: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`${flag} requer inteiro seguro.`);
  return parsed;
}

function summarizeAssignmentExecution(execution: Gemma4LiteralGenerationAssignmentExecution): Record<string, unknown> {
  return {
    assignmentId: execution.assignmentId,
    operation: execution.operation,
    ...(execution.step === undefined ? {} : { step: execution.step }),
    output: execution.output,
    value: summarizeGenerationValue(execution.value),
  };
}

function summarizeGenerationValue(value: Gemma4LiteralGenerationValue): unknown {
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (Array.isArray(value)) {
    const entries = value as unknown[];
    if (entries.length === 0 || typeof entries[0] === "number") return [...entries as number[]];
    const snapshots = entries as Array<ReadonlyMap<number, ReferenceF32KeyValueCache>>;
    return { cacheSnapshots: snapshots.length, terminal: summarizeCache(snapshots.at(-1)!) };
  }
  if (value instanceof Map) return summarizeCache(value);
  if (isTensor(value)) return summarizeTensor(value);
  if ("logits" in value && "pastKeyValues" in value) {
    const state = value as ReferenceF32ExecutionResult;
    return { logits: summarizeTensor(state.logits), pastKeyValues: summarizeCache(state.pastKeyValues) };
  }
  if ("inputIds" in value && "positionIds" in value && "pastKeyValues" in value) {
    return { inputIds: value.inputIds, positionIds: value.positionIds, pastKeyValues: summarizeCache(value.pastKeyValues) };
  }
  throw new Error("Valor de atribuição do programa de geração Gemma 4 não reconhecido.");
}

function summarizeTensor(tensor: DenseF32Tensor): Record<string, unknown> {
  return {
    dtype: "F32",
    shape: tensor.shape,
    sha256: createHash("sha256").update(Buffer.from(tensor.values.buffer, tensor.values.byteOffset, tensor.values.byteLength)).digest("hex"),
  };
}

function summarizeCache(cache: ReadonlyMap<number, ReferenceF32KeyValueCache>): Record<string, unknown> {
  return {
    producerLayers: [...cache.keys()],
    shapes: [...cache].map(([layer, entry]) => ({ layer, key: entry.key.shape, value: entry.value.shape })),
  };
}

function isTensor(value: Gemma4LiteralGenerationValue): value is DenseF32Tensor {
  return typeof value === "object" && value !== null && "shape" in value && "values" in value;
}
