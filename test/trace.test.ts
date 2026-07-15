import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { buildModelIR } from "../src/architecture.js";
import { openCatalog } from "../src/catalog.js";
import { executeReferenceF32, generateReferenceF32 } from "../src/executor.js";
import { materializeReferenceF32Constants } from "../src/materialize.js";
import { fingerprintIR, readExecutionTraceBundle } from "../src/trace.js";
import { runExecutionTraceComparison, runGenerationTraceComparison } from "../src/trace-runner.js";
import type { ModelIR } from "../src/types.js";
import type { ReferenceF32ExecutionResult } from "../src/types.js";

const preview = { outputRows: 1, inputTerms: 1, includeWeights: false } as const;

test("integrity-bound F32 trace runs catalog-to-materializer-to-executor differential comparison", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-trace-"));
  try {
    await writeTinyF32Model(directory);
    const source = path.join(directory, "model");
    const opened = await openCatalog(source, false);
    let ir: ModelIR;
    let candidate: ReferenceF32ExecutionResult;
    try {
      ir = await buildModelIR(opened.catalog, preview);
      setF32Policy(ir);
      const tensors = await materializeReferenceF32Constants(ir, opened.catalog, opened.reader);
      candidate = executeReferenceF32(ir, { inputIds: [[1]], tensors });
    } finally {
      await opened.close();
    }
    // The persisted trace must bind to lowering before its explicitly declared
    // F32 scalar execution policy is applied by the comparison runner.
    const fingerprintSource = await openCatalog(source, false);
    let fingerprint: string;
    try { fingerprint = fingerprintIR(await buildModelIR(fingerprintSource.catalog, preview)); } finally { await fingerprintSource.close(); }
    const trace = path.join(directory, "trace.json");
    await writeFile(trace, JSON.stringify({
      schemaVersion: 1,
      kind: "execution",
      source: { files: await checksums(source, ["config.json", "model.safetensors"]) },
      irFingerprint: fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "fixture authoritative F32", model: "tiny-llama", revisionOrChecksum: "fixture-sha256",
        containerFormat: "safetensors", quantization: "none", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture",
        operations: operations(ir!).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(candidate!.values.get(operation.output)!) })),
        pastKeyValues: [...candidate!.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const reportPath = path.join(directory, "report.json");
    const report = await runExecutionTraceComparison({ source, trace, report: reportPath, topK: 3 });
    assert.equal(report.fidelityClass, "lossless-within-dtype");
    assert.equal(report.firstDivergentOperation, null);
    assert.equal(JSON.parse(await readFile(reportPath, "utf8")).logits.maxAbsoluteError, 0);

    await writeFile(path.join(source, "config.json"), "{}\n");
    await assert.rejects(() => runExecutionTraceComparison({ source, trace, report: reportPath }), /configuração obrigatória|Checksum divergente/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("trace bundle rejects decimal-array tensor payloads and unsafe checkpoint paths", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-trace-invalid-"));
  try {
    const trace = path.join(directory, "invalid.json");
    await writeFile(trace, JSON.stringify({
      schemaVersion: 1, kind: "execution", irFingerprint: "0".repeat(64),
      source: { files: [{ path: "../model.safetensors", sha256: "0".repeat(64) }] },
      candidatePolicy: { dtype: "F32", runtime: "test" },
      reference: { runtime: "test", model: "test", revisionOrChecksum: "test", containerFormat: "safetensors", quantization: "none", inputTokens: [[0]], dtypePolicy: "F32", operations: [{ operationId: "x", output: "x", tensor: { dtype: "F32", shape: [1], values: [1] } }], pastKeyValues: [] },
    }));
    await assert.rejects(() => readExecutionTraceBundle(trace), /valuesBase64/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("integrity-bound F32 generation trace verifies positions, terminal logits, and canonical KV cache", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-generation-trace-"));
  try {
    await writeTinyF32Model(directory);
    const source = path.join(directory, "model");
    const opened = await openCatalog(source, false);
    let ir: ModelIR;
    let generated: ReturnType<typeof generateReferenceF32>;
    try {
      ir = await buildModelIR(opened.catalog, preview);
      setF32Policy(ir);
      const tensors = await materializeReferenceF32Constants(ir, opened.catalog, opened.reader);
      generated = generateReferenceF32(ir, { inputIds: [[1]], tensors, maxNewTokens: 2 });
    } finally {
      await opened.close();
    }
    const fingerprintSource = await openCatalog(source, false);
    let fingerprint: string;
    try { fingerprint = fingerprintIR(await buildModelIR(fingerprintSource.catalog, preview)); } finally { await fingerprintSource.close(); }
    const trace = path.join(directory, "generation-trace.json");
    await writeFile(trace, JSON.stringify({
      schemaVersion: 1,
      kind: "generation",
      source: { files: await checksums(source, ["config.json", "model.safetensors"]) },
      irFingerprint: fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "fixture authoritative F32", model: "tiny-llama", revisionOrChecksum: "fixture-sha256",
        containerFormat: "safetensors", quantization: "none", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture", maxNewTokens: 2, generatedTokenIds: generated!.generatedTokenIds,
        steps: generated!.steps, logits: serialized(generated!.logits),
        pastKeyValues: [...generated!.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const reportPath = path.join(directory, "generation-report.json");
    const report = await runGenerationTraceComparison({ source, trace, report: reportPath, topK: 3 });
    assert.equal(report.fidelityClass, "lossless-within-dtype");
    assert.equal(report.firstDivergence, null);
    assert.equal(JSON.parse(await readFile(reportPath, "utf8")).terminalLogits.maxAbsoluteError, 0);

    const malformed = JSON.parse(await readFile(trace, "utf8"));
    malformed.reference.pastKeyValues.push(malformed.reference.pastKeyValues[0]);
    await writeFile(trace, JSON.stringify(malformed));
    await assert.rejects(() => runGenerationTraceComparison({ source, trace, report: reportPath }), /cache KV duplicado/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("dense GGUF Llama replays Safetensors forward and greedy-generation evidence through the same IR semantics", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-cross-container-trace-"));
  try {
    await writeTinyF32Model(directory);
    const safetensorsSource = path.join(directory, "model");
    const ggufSource = path.join(directory, "model.gguf");
    await writeTinyF32GgufModel(ggufSource);

    const safetensors = await executeFixture(safetensorsSource, [[1]]);
    const gguf = await executeFixture(ggufSource, [[1]]);
    assert.deepEqual(serialized(gguf.candidate.values.get("logits")!), serialized(safetensors.candidate.values.get("logits")!));
    assert.deepEqual([...gguf.candidate.pastKeyValues], [...safetensors.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "gguf-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["model.gguf"]) },
      irFingerprint: gguf.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "paired Safetensors F32 fixture", model: "tiny-llama", revisionOrChecksum: "paired-fixture-v1",
        containerFormat: "safetensors", quantization: "none", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture",
        operations: operations(gguf.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(safetensors.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...safetensors.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: ggufSource, trace: executionTrace, report: path.join(directory, "gguf-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const safetensorsGeneration = await generateFixture(safetensorsSource);
    const generationTrace = path.join(directory, "gguf-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["model.gguf"]) },
      irFingerprint: gguf.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "paired Safetensors F32 fixture", model: "tiny-llama", revisionOrChecksum: "paired-fixture-v1",
        containerFormat: "safetensors", quantization: "none", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture", maxNewTokens: 2, generatedTokenIds: safetensorsGeneration.generatedTokenIds,
        steps: safetensorsGeneration.steps, logits: serialized(safetensorsGeneration.logits),
        pastKeyValues: [...safetensorsGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: ggufSource, trace: generationTrace, report: path.join(directory, "gguf-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Q8_0 GGUF Llama replays independently materialized dense F32 forward and greedy-generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-q8_0-trace-"));
  try {
    const denseSource = path.join(directory, "dense-reference.gguf");
    const quantizedSource = path.join(directory, "q8_0.gguf");
    await writeQ8_0TraceFixture(denseSource, quantizedSource);

    // The F32 package is constructed directly from the declared Q8_0 formula
    // (F32[i] = F16(d) * int8(q[i])), rather than by using the candidate
    // reader. It is therefore independent evidence for the full trace path.
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "q8_0-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["q8_0.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q8_0 F32 fixture", model: "q8_0-llama-256", revisionOrChecksum: "q8_0-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q8_0", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared Q8_0 formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "q8_0-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "q8_0-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["q8_0.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q8_0 F32 fixture", model: "q8_0-llama-256", revisionOrChecksum: "q8_0-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q8_0", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared Q8_0 formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "q8_0-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    // A trace is bound to the exact packed source, not merely its model name
    // or a claim that it uses Q8_0. Changing one packed code must fail before
    // the candidate result can be accepted.
    const corrupted = await readFile(quantizedSource);
    const last = corrupted.length - 1;
    corrupted[last] = corrupted[last]! ^ 1;
    await writeFile(quantizedSource, corrupted);
    await assert.rejects(() => runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "corrupt-report.json") }), /Checksum divergente/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

function operations(ir: ModelIR) { return [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]; }

function setF32Policy(ir: ModelIR): void {
  for (const operation of operations(ir)) {
    operation.dtypePolicy = { computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
}

function serialized(tensor: { shape: number[]; values: Float32Array }) {
  return { dtype: "F32", shape: tensor.shape, valuesBase64: Buffer.from(tensor.values.buffer, tensor.values.byteOffset, tensor.values.byteLength).toString("base64") };
}

async function checksums(directory: string, files: string[]) {
  return Promise.all(files.map(async (file) => ({ path: file, sha256: createHash("sha256").update(await readFile(path.join(directory, file))).digest("hex") })));
}

async function executeFixture(source: string, inputIds: number[][]): Promise<{ ir: ModelIR; fingerprint: string; candidate: ReferenceF32ExecutionResult }> {
  const opened = await openCatalog(source, false);
  try {
    const ir = await buildModelIR(opened.catalog, preview);
    const fingerprint = fingerprintIR(ir);
    setF32Policy(ir);
    const tensors = await materializeReferenceF32Constants(ir, opened.catalog, opened.reader);
    return { ir, fingerprint, candidate: executeReferenceF32(ir, { inputIds, tensors }) };
  } finally { await opened.close(); }
}

async function generateFixture(source: string) {
  const opened = await openCatalog(source, false);
  try {
    const ir = await buildModelIR(opened.catalog, preview);
    setF32Policy(ir);
    const tensors = await materializeReferenceF32Constants(ir, opened.catalog, opened.reader);
    return generateReferenceF32(ir, { inputIds: [[1]], tensors, maxNewTokens: 2 });
  } finally { await opened.close(); }
}

async function writeTinyF32Model(root: string): Promise<void> {
  const directory = path.join(root, "model");
  await mkdir(directory);
  const weights: Array<[string, number[], number[]]> = [
    ["model.embed_tokens.weight", [3, 2], [0, 0, 3, 4, 0, 0]],
    ["model.layers.0.input_layernorm.weight", [2], [1, 1]],
    ...["q_proj", "k_proj", "v_proj", "o_proj"].map((projection): [string, number[], number[]] => [`model.layers.0.self_attn.${projection}.weight`, [2, 2], [1, 0, 0, 1]]),
    ["model.layers.0.post_attention_layernorm.weight", [2], [1, 1]],
    ["model.layers.0.mlp.gate_proj.weight", [2, 2], [0, 0, 0, 0]], ["model.layers.0.mlp.up_proj.weight", [2, 2], [0, 0, 0, 0]], ["model.layers.0.mlp.down_proj.weight", [2, 2], [0, 0, 0, 0]],
    ["model.norm.weight", [2], [1, 1]], ["lm_head.weight", [3, 2], [1, 0, 0, 1, 1, 1]],
  ];
  const header: Record<string, unknown> = {};
  let offset = 0;
  const payloads = weights.map(([name, shape, values]) => {
    const payload = Buffer.alloc(values.length * 4);
    values.forEach((value, index) => payload.writeFloatLE(value, index * 4));
    header[name] = { dtype: "F32", shape, data_offsets: [offset, offset + payload.length] };
    offset += payload.length;
    return payload;
  });
  const encoded = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(encoded.length));
  await writeFile(path.join(directory, "config.json"), JSON.stringify({ model_type: "llama", hidden_size: 2, intermediate_size: 2, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1, head_dim: 2, vocab_size: 3, rms_norm_eps: 1e-6, hidden_act: "silu" }));
  await writeFile(path.join(directory, "model.safetensors"), Buffer.concat([prefix, encoded, ...payloads]));
}

/** Writes the same logical weights as writeTinyF32Model in native GGUF order. */
async function writeTinyF32GgufModel(file: string): Promise<void> {
  const weights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [2, 3], [0, 0, 3, 4, 0, 0]],
    ["blk.0.attn_norm.weight", [2], [1, 1]],
    ...["attn_q", "attn_k", "attn_v", "attn_output"].map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [2, 2], [1, 0, 0, 1]]),
    ["blk.0.ffn_norm.weight", [2], [1, 1]],
    ...["ffn_gate", "ffn_up", "ffn_down"].map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [2, 2], [0, 0, 0, 0]]),
    ["output_norm.weight", [2], [1, 1]], ["output.weight", [2, 3], [1, 0, 0, 1, 1, 1]],
  ];
  const text = (value: string) => { const bytes = Buffer.from(value); const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(bytes.length)); return Buffer.concat([length, bytes]); };
  const u32 = (value: number) => { const bytes = Buffer.alloc(4); bytes.writeUInt32LE(value); return bytes; };
  const u64 = (value: number) => { const bytes = Buffer.alloc(8); bytes.writeBigUInt64LE(BigInt(value)); return bytes; };
  const metadata = (key: string, type: number, value: Buffer) => Buffer.concat([text(key), u32(type), value]);
  const metadataU32 = (key: string, value: number) => metadata(key, 4, u32(value));
  const metadataF32 = (key: string, value: number) => { const bytes = Buffer.alloc(4); bytes.writeFloatLE(value); return metadata(key, 6, bytes); };
  const metadataEntries = [
    metadata("general.architecture", 8, text("llama")), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 2),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 1), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 2), metadataU32("llama.feed_forward_length", 2), metadataF32("llama.attention.layer_norm_rms_epsilon", 1e-6),
  ];
  const payloads: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, shape, values] of weights) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloads.push(Buffer.alloc(padding)); offset += padding; }
    const payload = Buffer.alloc(values.length * 4);
    values.forEach((value, index) => payload.writeFloatLE(value, index * 4));
    directory.push(Buffer.concat([text(name), u32(shape.length), ...shape.map(u64), u32(0), u64(offset)]));
    payloads.push(payload); offset += payload.length;
  }
  const prefix = Buffer.concat([Buffer.from("GGUF"), u32(3), u64(weights.length), u64(metadataEntries.length), ...metadataEntries, ...directory]);
  await writeFile(file, Buffer.concat([prefix, Buffer.alloc((32 - (prefix.length % 32)) % 32), ...payloads]));
}

/**
 * Writes a one-layer 256-wide Llama whose matrix payloads are Q8_0 blocks
 * with d=0.5.  The paired dense file is formed from exactly d*q, so it can
 * exercise every materialized projection while respecting Q8_0's 32-value
 * block boundary on GGML's first matrix dimension.
 */
async function writeQ8_0TraceFixture(denseFile: string, quantizedFile: string): Promise<void> {
  const width = 256;
  const identity = Array.from({ length: width * width }, (_, index) => (Math.floor(index / width) === index % width ? 1 : 0));
  const embedding = Array.from({ length: width * width }, (_, index) => (Math.floor(index / width) === 1 && index % width < 2 ? (index % width === 0 ? 1 : 0.5) : 0));
  const zeros = new Array<number>(width * width).fill(0);
  const ones = new Array<number>(width).fill(1);
  const denseWeights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [width, width], embedding], ["blk.0.attn_norm.weight", [width], ones],
    ...["attn_q", "attn_k", "attn_v", "attn_output"].map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], identity]),
    ["blk.0.ffn_norm.weight", [width], ones],
    ...["ffn_gate", "ffn_up", "ffn_down"].map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], zeros]),
    ["output_norm.weight", [width], ones], ["output.weight", [width, width], identity],
  ];
  await writeGgufLlamaFixture(denseFile, denseWeights.map(([name, shape, values]) => [name, shape, 0, values]));
  await writeGgufLlamaFixture(quantizedFile, denseWeights.map(([name, shape, values]) => [name, shape, values.length === width ? 0 : 8, values]));
}

async function writeGgufLlamaFixture(file: string, weights: Array<[string, number[], number, number[]]>): Promise<void> {
  const text = (value: string) => { const bytes = Buffer.from(value); const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(bytes.length)); return Buffer.concat([length, bytes]); };
  const u32 = (value: number) => { const bytes = Buffer.alloc(4); bytes.writeUInt32LE(value); return bytes; };
  const u64 = (value: number) => { const bytes = Buffer.alloc(8); bytes.writeBigUInt64LE(BigInt(value)); return bytes; };
  const metadata = (key: string, type: number, value: Buffer) => Buffer.concat([text(key), u32(type), value]);
  const metadataU32 = (key: string, value: number) => metadata(key, 4, u32(value));
  const metadataF32 = (key: string, value: number) => { const bytes = Buffer.alloc(4); bytes.writeFloatLE(value); return metadata(key, 6, bytes); };
  const metadataEntries = [
    metadata("general.architecture", 8, text("llama")), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", 256),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 1), metadataU32("llama.attention.head_count_kv", 1),
    metadataU32("llama.attention.key_length", 256), metadataU32("llama.feed_forward_length", 256), metadataF32("llama.attention.layer_norm_rms_epsilon", 1e-6),
  ];
  const payloads: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, shape, ggmlType, values] of weights) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloads.push(Buffer.alloc(padding)); offset += padding; }
    const payload = ggmlType === 8 ? q8_0Payload(values) : f32Payload(values);
    directory.push(Buffer.concat([text(name), u32(shape.length), ...shape.map(u64), u32(ggmlType), u64(offset)]));
    payloads.push(payload); offset += payload.length;
  }
  const prefix = Buffer.concat([Buffer.from("GGUF"), u32(3), u64(weights.length), u64(metadataEntries.length), ...metadataEntries, ...directory]);
  await writeFile(file, Buffer.concat([prefix, Buffer.alloc((32 - (prefix.length % 32)) % 32), ...payloads]));
}

function f32Payload(values: number[]): Buffer {
  const payload = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => payload.writeFloatLE(value, index * 4));
  return payload;
}

function q8_0Payload(values: number[]): Buffer {
  assert.equal(values.length % 32, 0, "Q8_0 fixture requires complete 32-value blocks");
  const payload = Buffer.alloc((values.length / 32) * 34);
  for (let block = 0; block < values.length / 32; block += 1) {
    // 0x3800 is IEEE-754 binary16 0.5. All fixture values are multiples of
    // 0.5, producing exact signed Q8_0 codes without a candidate dequantizer.
    payload.writeUInt16LE(0x3800, block * 34);
    for (let index = 0; index < 32; index += 1) payload.writeInt8(values[block * 32 + index]! * 2, block * 34 + 2 + index);
  }
  return payload;
}
