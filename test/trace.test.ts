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

test("Q8_1 GGUF Llama replays independently constructed dense F32 forward and greedy-generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-q8_1-trace-"));
  try {
    const denseSource = path.join(directory, "dense-reference.gguf");
    const quantizedSource = path.join(directory, "q8_1.gguf");
    await writeQ8_1TraceFixture(denseSource, quantizedSource);

    // The paired dense package calculates d*q directly with binary32 d=0.25.
    // The Q8_1 writer independently emits s=d*sum(qs) for every 32-code
    // block; s is an auxiliary dot-product field and deliberately does not
    // become an element offset or a Q8_0 binary16 scale in the dense formula.
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "q8_1-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["q8_1.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q8_1 F32 fixture", model: "q8_1-llama-256", revisionOrChecksum: "q8_1-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q8_1", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared Q8_1 formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "q8_1-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "q8_1-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["q8_1.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q8_1 F32 fixture", model: "q8_1-llama-256", revisionOrChecksum: "q8_1-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q8_1", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared Q8_1 formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "q8_1-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    // This flips the binary32 auxiliary s field in the final Q8_1 block.
    // Integrity verification must reject the modified packed source before it
    // can be reported as faithful, even though s is not an element offset.
    const corrupted = await readFile(quantizedSource);
    const lastAuxiliaryField = corrupted.length - 36;
    corrupted[lastAuxiliaryField] = corrupted[lastAuxiliaryField]! ^ 1;
    await writeFile(quantizedSource, corrupted);
    await assert.rejects(() => runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "corrupt-report.json") }), /Checksum divergente/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Q4_K GGUF Llama replays independently constructed affine dense F32 forward and greedy-generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-q4_k-trace-"));
  try {
    const denseSource = path.join(directory, "dense-reference.gguf");
    const quantizedSource = path.join(directory, "q4_k.gguf");
    await writeQ4_KTraceFixture(denseSource, quantizedSource);

    // The F32 package applies Q4_K's declared d*scale*q - dmin*minimum
    // formula directly. It deliberately does not use the candidate reader,
    // while covering both six-bit packing branches (groups 0..3 and 4..7).
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "q4_k-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["q4_k.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q4_K F32 fixture", model: "q4_k-llama-256", revisionOrChecksum: "q4_k-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q4_K", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared Q4_K formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "q4_k-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "q4_k-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["q4_k.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q4_K F32 fixture", model: "q4_k-llama-256", revisionOrChecksum: "q4_k-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q4_K", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared Q4_K formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "q4_k-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    const corrupted = await readFile(quantizedSource);
    corrupted[corrupted.length - 1] = corrupted[corrupted.length - 1]! ^ 1;
    await writeFile(quantizedSource, corrupted);
    await assert.rejects(() => runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "corrupt-report.json") }), /Checksum divergente/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Q5_K GGUF Llama replays independently constructed affine dense F32 forward and greedy-generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-q5_k-trace-"));
  try {
    const denseSource = path.join(directory, "dense-reference.gguf");
    const quantizedSource = path.join(directory, "q5_k.gguf");
    await writeQ5_KTraceFixture(denseSource, quantizedSource);

    // The paired F32 package applies Q5_K's declared affine formula directly,
    // including the independently packed fifth-bit plane. It never delegates
    // reconstruction to the candidate GGUF reader.
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "q5_k-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["q5_k.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q5_K F32 fixture", model: "q5_k-llama-256", revisionOrChecksum: "q5_k-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q5_K", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared Q5_K formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "q5_k-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "q5_k-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["q5_k.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q5_K F32 fixture", model: "q5_k-llama-256", revisionOrChecksum: "q5_k-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q5_K", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared Q5_K formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "q5_k-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    const corrupted = await readFile(quantizedSource);
    corrupted[corrupted.length - 1] = corrupted[corrupted.length - 1]! ^ 1;
    await writeFile(quantizedSource, corrupted);
    await assert.rejects(() => runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "corrupt-report.json") }), /Checksum divergente/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Q6_K GGUF Llama replays independently constructed signed dense F32 forward and greedy-generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-q6_k-trace-"));
  try {
    const denseSource = path.join(directory, "dense-reference.gguf");
    const quantizedSource = path.join(directory, "q6_k.gguf");
    await writeQ6_KTraceFixture(denseSource, quantizedSource);

    // The paired F32 package applies Q6_K's declared d*scale*(code-32)
    // formula directly. It never reads the candidate payload and exercises
    // every ql/qh two-bit state and both signs of each per-16-value scale.
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "q6_k-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["q6_k.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q6_K F32 fixture", model: "q6_k-llama-256", revisionOrChecksum: "q6_k-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q6_K", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared Q6_K formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "q6_k-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "q6_k-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["q6_k.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q6_K F32 fixture", model: "q6_k-llama-256", revisionOrChecksum: "q6_k-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q6_K", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared Q6_K formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "q6_k-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    const corrupted = await readFile(quantizedSource);
    corrupted[corrupted.length - 1] = corrupted[corrupted.length - 1]! ^ 1;
    await writeFile(quantizedSource, corrupted);
    await assert.rejects(() => runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "corrupt-report.json") }), /Checksum divergente/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Q3_K GGUF Llama replays independently constructed signed dense F32 forward and greedy-generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-q3_k-trace-"));
  try {
    const denseSource = path.join(directory, "dense-reference.gguf");
    const quantizedSource = path.join(directory, "q3_k.gguf");
    await writeQ3_KTraceFixture(denseSource, quantizedSource);

    // The paired F32 package applies Q3_K's d*signedScale*signedCode formula
    // directly. It never reads the candidate payload and exercises all four
    // two-bit planes, both hmask states, and signed six-bit scale fields.
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "q3_k-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["q3_k.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q3_K F32 fixture", model: "q3_k-llama-256", revisionOrChecksum: "q3_k-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q3_K", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared Q3_K formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "q3_k-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "q3_k-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["q3_k.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q3_K F32 fixture", model: "q3_k-llama-256", revisionOrChecksum: "q3_k-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q3_K", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared Q3_K formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "q3_k-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    const corrupted = await readFile(quantizedSource);
    corrupted[corrupted.length - 1] = corrupted[corrupted.length - 1]! ^ 1;
    await writeFile(quantizedSource, corrupted);
    await assert.rejects(() => runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "corrupt-report.json") }), /Checksum divergente/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Q2_K GGUF Llama replays independently constructed affine dense F32 forward and greedy-generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-q2_k-trace-"));
  try {
    const denseSource = path.join(directory, "dense-reference.gguf");
    const quantizedSource = path.join(directory, "q2_k.gguf");
    await writeQ2_KTraceFixture(denseSource, quantizedSource);

    // The F32 package applies Q2_K's d*scale*code - dmin*minimum formula
    // directly. It never reads the candidate payload and exercises all 16
    // packed scale/minimum nibbles and all four two-bit code planes.
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "q2_k-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["q2_k.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q2_K F32 fixture", model: "q2_k-llama-256", revisionOrChecksum: "q2_k-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q2_K", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared Q2_K formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "q2_k-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "q2_k-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["q2_k.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q2_K F32 fixture", model: "q2_k-llama-256", revisionOrChecksum: "q2_k-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q2_K", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared Q2_K formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "q2_k-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    const corrupted = await readFile(quantizedSource);
    corrupted[corrupted.length - 1] = corrupted[corrupted.length - 1]! ^ 1;
    await writeFile(quantizedSource, corrupted);
    await assert.rejects(() => runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "corrupt-report.json") }), /Checksum divergente/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Q8_K GGUF Llama replays independently constructed signed dense F32 forward and greedy-generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-q8_k-trace-"));
  try {
    const denseSource = path.join(directory, "dense-reference.gguf");
    const quantizedSource = path.join(directory, "q8_k.gguf");
    await writeQ8_KTraceFixture(denseSource, quantizedSource);

    // The paired F32 package calculates d*q directly with d=0.25. It does
    // not decode the candidate bytes, and each 256-value block exercises the
    // complete signed int8 range, including negative, zero, and positive q.
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "q8_k-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["q8_k.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q8_K F32 fixture", model: "q8_k-llama-256", revisionOrChecksum: "q8_k-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q8_K", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared Q8_K formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "q8_k-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "q8_k-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["q8_k.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q8_K F32 fixture", model: "q8_k-llama-256", revisionOrChecksum: "q8_k-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q8_K", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared Q8_K formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "q8_k-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    const corrupted = await readFile(quantizedSource);
    corrupted[corrupted.length - 1] = corrupted[corrupted.length - 1]! ^ 1;
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

/** Builds a Q8_1 fixture with all signed int8 codes and verified auxiliary sums live. */
async function writeQ8_1TraceFixture(denseFile: string, quantizedFile: string): Promise<void> {
  const width = 256;
  const ones = new Array<number>(width).fill(1);
  const matrixNames = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
  const denseWeights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [width, width], q8_1DenseValues(width * width)], ["blk.0.attn_norm.weight", [width], ones],
    ...matrixNames.map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], q8_1DenseValues(width * width)]),
    ["blk.0.ffn_norm.weight", [width], ones], ["output_norm.weight", [width], ones], ["output.weight", [width, width], q8_1DenseValues(width * width)],
  ];
  await writeGgufLlamaFixture(denseFile, denseWeights.map(([name, shape, values]) => [name, shape, 0, values]));
  await writeGgufLlamaFixture(quantizedFile, denseWeights.map(([name, shape, values]) => [name, shape, values.length === width ? 0 : 9, values]));
}

/** Builds a Q4_K fixture with all eight group fields and the nibble planes live. */
async function writeQ4_KTraceFixture(denseFile: string, quantizedFile: string): Promise<void> {
  const width = 256;
  const ones = new Array<number>(width).fill(1);
  const matrixNames = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
  const denseWeights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [width, width], q4KDenseValues(width * width)], ["blk.0.attn_norm.weight", [width], ones],
    ...matrixNames.map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], q4KDenseValues(width * width)]),
    ["blk.0.ffn_norm.weight", [width], ones], ["output_norm.weight", [width], ones], ["output.weight", [width, width], q4KDenseValues(width * width)],
  ];
  await writeGgufLlamaFixture(denseFile, denseWeights.map(([name, shape, values]) => [name, shape, 0, values]));
  await writeGgufLlamaFixture(quantizedFile, denseWeights.map(([name, shape, values]) => [name, shape, values.length === width ? 0 : 12, values]));
}

/** Builds a Q5_K fixture with all affine groups, nibble planes, and qh bits live. */
async function writeQ5_KTraceFixture(denseFile: string, quantizedFile: string): Promise<void> {
  const width = 256;
  const ones = new Array<number>(width).fill(1);
  const matrixNames = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
  const denseWeights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [width, width], q5KDenseValues(width * width)], ["blk.0.attn_norm.weight", [width], ones],
    ...matrixNames.map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], q5KDenseValues(width * width)]),
    ["blk.0.ffn_norm.weight", [width], ones], ["output_norm.weight", [width], ones], ["output.weight", [width, width], q5KDenseValues(width * width)],
  ];
  await writeGgufLlamaFixture(denseFile, denseWeights.map(([name, shape, values]) => [name, shape, 0, values]));
  await writeGgufLlamaFixture(quantizedFile, denseWeights.map(([name, shape, values]) => [name, shape, values.length === width ? 0 : 13, values]));
}

/** Builds a Q6_K fixture with all signed scales and ql/qh code planes live. */
async function writeQ6_KTraceFixture(denseFile: string, quantizedFile: string): Promise<void> {
  const width = 256;
  const ones = new Array<number>(width).fill(1);
  const matrixNames = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
  const denseWeights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [width, width], q6KDenseValues(width * width)], ["blk.0.attn_norm.weight", [width], ones],
    ...matrixNames.map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], q6KDenseValues(width * width)]),
    ["blk.0.ffn_norm.weight", [width], ones], ["output_norm.weight", [width], ones], ["output.weight", [width, width], q6KDenseValues(width * width)],
  ];
  await writeGgufLlamaFixture(denseFile, denseWeights.map(([name, shape, values]) => [name, shape, 0, values]));
  await writeGgufLlamaFixture(quantizedFile, denseWeights.map(([name, shape, values]) => [name, shape, values.length === width ? 0 : 14, values]));
}

/** Builds a Q3_K fixture with live hmask, every low-code plane, and signed scales. */
async function writeQ3_KTraceFixture(denseFile: string, quantizedFile: string): Promise<void> {
  const width = 256;
  const ones = new Array<number>(width).fill(1);
  const matrixNames = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
  const denseWeights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [width, width], q3KDenseValues(width * width)], ["blk.0.attn_norm.weight", [width], ones],
    ...matrixNames.map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], q3KDenseValues(width * width)]),
    ["blk.0.ffn_norm.weight", [width], ones], ["output_norm.weight", [width], ones], ["output.weight", [width, width], q3KDenseValues(width * width)],
  ];
  await writeGgufLlamaFixture(denseFile, denseWeights.map(([name, shape, values]) => [name, shape, 0, values]));
  await writeGgufLlamaFixture(quantizedFile, denseWeights.map(([name, shape, values]) => [name, shape, values.length === width ? 0 : 11, values]));
}

/** Builds a Q2_K fixture with every scale/minimum nibble and code plane live. */
async function writeQ2_KTraceFixture(denseFile: string, quantizedFile: string): Promise<void> {
  const width = 256;
  const ones = new Array<number>(width).fill(1);
  const matrixNames = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
  const denseWeights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [width, width], q2KDenseValues(width * width)], ["blk.0.attn_norm.weight", [width], ones],
    ...matrixNames.map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], q2KDenseValues(width * width)]),
    ["blk.0.ffn_norm.weight", [width], ones], ["output_norm.weight", [width], ones], ["output.weight", [width, width], q2KDenseValues(width * width)],
  ];
  await writeGgufLlamaFixture(denseFile, denseWeights.map(([name, shape, values]) => [name, shape, 0, values]));
  await writeGgufLlamaFixture(quantizedFile, denseWeights.map(([name, shape, values]) => [name, shape, values.length === width ? 0 : 10, values]));
}

/** Builds a Q8_K fixture with the complete signed int8 code domain live. */
async function writeQ8_KTraceFixture(denseFile: string, quantizedFile: string): Promise<void> {
  const width = 256;
  const ones = new Array<number>(width).fill(1);
  const matrixNames = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
  const denseWeights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [width, width], q8KDenseValues(width * width)], ["blk.0.attn_norm.weight", [width], ones],
    ...matrixNames.map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], q8KDenseValues(width * width)]),
    ["blk.0.ffn_norm.weight", [width], ones], ["output_norm.weight", [width], ones], ["output.weight", [width, width], q8KDenseValues(width * width)],
  ];
  await writeGgufLlamaFixture(denseFile, denseWeights.map(([name, shape, values]) => [name, shape, 0, values]));
  await writeGgufLlamaFixture(quantizedFile, denseWeights.map(([name, shape, values]) => [name, shape, values.length === width ? 0 : 15, values]));
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
    const payload = ggmlType === 8 ? q8_0Payload(values) : ggmlType === 9 ? q8_1Payload(values.length) : ggmlType === 10 ? q2KPayload(values.length) : ggmlType === 11 ? q3KPayload(values.length) : ggmlType === 12 ? q4KPayload(values.length) : ggmlType === 13 ? q5KPayload(values.length) : ggmlType === 14 ? q6KPayload(values.length) : ggmlType === 15 ? q8KPayload(values.length) : f32Payload(values);
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

function q8KDenseValues(length: number): number[] {
  assert.equal(length % 256, 0, "Q8_K fixture requires complete 256-value blocks");
  return Array.from({ length }, (_, index) => 0.25 * ((index % 256) - 128));
}

function q8_1DenseValues(length: number): number[] {
  assert.equal(length % 32, 0, "Q8_1 fixture requires complete 32-value blocks");
  return Array.from({ length }, (_, index) => 0.25 * ((index % 256) - 128));
}

function q8_1Payload(length: number): Buffer {
  assert.equal(length % 32, 0, "Q8_1 fixture requires complete 32-value blocks");
  const payload = Buffer.alloc((length / 32) * 40);
  for (let block = 0; block < length / 32; block += 1) {
    const offset = block * 40;
    const codes = Array.from({ length: 32 }, (_, index) => ((block * 32 + index) % 256) - 128);
    const scale = 0.25;
    const auxiliarySum = Math.fround(scale * codes.reduce((sum, code) => sum + code, 0));
    payload.writeFloatLE(scale, offset);
    payload.writeFloatLE(auxiliarySum, offset + 4);
    assert.equal(payload.readFloatLE(offset + 4), auxiliarySum, "Q8_1 fixture must persist s=d*sum(qs) as binary32");
    for (let index = 0; index < 32; index += 1) payload.writeInt8(codes[index]!, offset + 8 + index);
  }
  return payload;
}

function q8KPayload(length: number): Buffer {
  assert.equal(length % 256, 0, "Q8_K fixture requires complete 256-value blocks");
  const payload = Buffer.alloc((length / 256) * 260);
  for (let block = 0; block < length / 256; block += 1) {
    const offset = block * 260;
    payload.writeFloatLE(0.25, offset);
    for (let index = 0; index < 256; index += 1) payload.writeInt8(index - 128, offset + 4 + index);
  }
  return payload;
}

function q4KDenseValues(length: number): number[] {
  assert.equal(length % 256, 0, "Q4_K fixture requires complete 256-value blocks");
  const scales = [1, 2, 3, 4, 17, 34, 51, 63];
  const minimums = [5, 6, 7, 8, 18, 35, 52, 63];
  return Array.from({ length }, (_, index) => {
    const within = index % 256;
    const group = Math.floor(within / 32);
    const code = (within * 7 + group * 3) & 0x0f;
    return 0.5 * scales[group]! * code - 0.25 * minimums[group]!;
  });
}

function q4KPayload(length: number): Buffer {
  assert.equal(length % 256, 0, "Q4_K fixture requires complete 256-value blocks");
  const scales = [1, 2, 3, 4, 17, 34, 51, 63];
  const minimums = [5, 6, 7, 8, 18, 35, 52, 63];
  const payload = Buffer.alloc((length / 256) * 144);
  for (let block = 0; block < length / 256; block += 1) {
    const offset = block * 144;
    // d=0.5 and dmin=0.25 as IEEE-754 binary16. The independent dense side
    // applies those declared values, rather than decoding this payload.
    payload.writeUInt16LE(0x3800, offset);
    payload.writeUInt16LE(0x3400, offset + 2);
    for (let group = 0; group < 4; group += 1) {
      payload[offset + 4 + group] = (scales[group]! & 0x3f) | ((scales[group + 4]! >>> 4) << 6);
      payload[offset + 8 + group] = (minimums[group]! & 0x3f) | ((minimums[group + 4]! >>> 4) << 6);
      payload[offset + 12 + group] = (scales[group + 4]! & 0x0f) | ((minimums[group + 4]! & 0x0f) << 4);
    }
    for (let group = 0; group < 8; group += 1) {
      const codeOffset = offset + 16 + Math.floor(group / 2) * 32;
      const highNibble = group % 2 === 1;
      for (let index = 0; index < 32; index += 1) {
        const code = ((group * 32 + index) * 7 + group * 3) & 0x0f;
        payload[codeOffset + index]! |= code << (highNibble ? 4 : 0);
      }
    }
  }
  return payload;
}

function q5KDenseValues(length: number): number[] {
  assert.equal(length % 256, 0, "Q5_K fixture requires complete 256-value blocks");
  const scales = [1, 2, 3, 4, 17, 34, 51, 63];
  const minimums = [5, 6, 7, 8, 18, 35, 52, 63];
  return Array.from({ length }, (_, index) => {
    const within = index % 256;
    const group = Math.floor(within / 32);
    const code = (within * 13 + group * 5) & 0x1f;
    return 0.5 * scales[group]! * code - 0.25 * minimums[group]!;
  });
}

function q5KPayload(length: number): Buffer {
  assert.equal(length % 256, 0, "Q5_K fixture requires complete 256-value blocks");
  const scales = [1, 2, 3, 4, 17, 34, 51, 63];
  const minimums = [5, 6, 7, 8, 18, 35, 52, 63];
  const payload = Buffer.alloc((length / 256) * 176);
  for (let block = 0; block < length / 256; block += 1) {
    const offset = block * 176;
    payload.writeUInt16LE(0x3800, offset); // d = 0.5
    payload.writeUInt16LE(0x3400, offset + 2); // dmin = 0.25
    for (let group = 0; group < 4; group += 1) {
      payload[offset + 4 + group] = (scales[group]! & 0x3f) | ((scales[group + 4]! >>> 4) << 6);
      payload[offset + 8 + group] = (minimums[group]! & 0x3f) | ((minimums[group + 4]! >>> 4) << 6);
      payload[offset + 12 + group] = (scales[group + 4]! & 0x0f) | ((minimums[group + 4]! & 0x0f) << 4);
    }
    for (let group = 0; group < 8; group += 1) {
      const codeOffset = offset + 48 + Math.floor(group / 2) * 32;
      const highNibble = group % 2 === 1;
      for (let index = 0; index < 32; index += 1) {
        const blockIndex = group * 32 + index;
        const code = (blockIndex * 13 + group * 5) & 0x1f;
        payload[codeOffset + index]! |= (code & 0x0f) << (highNibble ? 4 : 0);
        payload[offset + 16 + Math.floor(blockIndex / 8)]! |= ((code >>> 4) & 1) << (blockIndex % 8);
      }
    }
  }
  return payload;
}

function q6KDenseValues(length: number): number[] {
  assert.equal(length % 256, 0, "Q6_K fixture requires complete 256-value blocks");
  const scales = [-8, -5, -3, -1, 1, 2, 4, 7, -7, -4, -2, -1, 1, 3, 5, 8];
  return Array.from({ length }, (_, index) => {
    const within = index % 256;
    const code = (within * 29 + Math.floor(within / 16) * 7) & 0x3f;
    return 0.5 * scales[Math.floor(within / 16)]! * (code - 32);
  });
}

function q6KPayload(length: number): Buffer {
  assert.equal(length % 256, 0, "Q6_K fixture requires complete 256-value blocks");
  const scales = [-8, -5, -3, -1, 1, 2, 4, 7, -7, -4, -2, -1, 1, 3, 5, 8];
  const payload = Buffer.alloc((length / 256) * 210);
  for (let block = 0; block < length / 256; block += 1) {
    const offset = block * 210;
    for (let plane = 0; plane < 8; plane += 1) {
      const lowByteBase = offset + Math.floor(plane / 4) * 64 + (plane % 2) * 32;
      const lowShift = plane % 4 >= 2 ? 4 : 0;
      const highByteBase = offset + 128 + Math.floor(plane / 4) * 32;
      const highShift = (plane % 4) * 2;
      for (let index = 0; index < 32; index += 1) {
        const blockIndex = plane * 32 + index;
        const code = (blockIndex * 29 + Math.floor(blockIndex / 16) * 7) & 0x3f;
        payload[lowByteBase + index]! |= (code & 0x0f) << lowShift;
        payload[highByteBase + index]! |= ((code >>> 4) & 0x03) << highShift;
      }
    }
    for (let group = 0; group < 16; group += 1) payload.writeInt8(scales[group]!, offset + 192 + group);
    payload.writeUInt16LE(0x3800, offset + 208); // d = 0.5
  }
  return payload;
}

function q3KDenseValues(length: number): number[] {
  assert.equal(length % 256, 0, "Q3_K fixture requires complete 256-value blocks");
  const scales = [-31, -17, -9, -3, 1, 5, 12, 31, -30, -14, -6, -1, 2, 8, 19, 30];
  return Array.from({ length }, (_, index) => {
    const within = index % 256;
    const code = ((within * 5 + Math.floor(within / 32) * 3) & 0x07) - 4;
    return 0.5 * scales[Math.floor(within / 16)]! * code;
  });
}

function q3KPayload(length: number): Buffer {
  assert.equal(length % 256, 0, "Q3_K fixture requires complete 256-value blocks");
  const scales = [-31, -17, -9, -3, 1, 5, 12, 31, -30, -14, -6, -1, 2, 8, 19, 30];
  const payload = Buffer.alloc((length / 256) * 110);
  for (let block = 0; block < length / 256; block += 1) {
    const offset = block * 110;
    payload.writeUInt16LE(0x3800, offset); // d = 0.5
    for (let group = 0; group < 16; group += 1) {
      const encoded = scales[group]! + 32;
      payload[offset + 98 + (group % 8)]! |= (encoded & 0x0f) << (group < 8 ? 0 : 4);
      payload[offset + 106 + (group % 4)]! |= ((encoded >>> 4) & 0x03) << (2 * Math.floor(group / 4));
    }
    for (let index = 0; index < 256; index += 1) {
      const code = ((index * 5 + Math.floor(index / 32) * 3) & 0x07) - 4;
      const encoded = code + 4;
      if (encoded > 3) payload[offset + 2 + (index % 32)]! |= 1 << Math.floor(index / 32);
      const plane = Math.floor((index % 128) / 32);
      payload[offset + 34 + Math.floor(index / 128) * 32 + (index % 32)]! |= (encoded & 0x03) << (plane * 2);
    }
  }
  return payload;
}

function q2KDenseValues(length: number): number[] {
  assert.equal(length % 256, 0, "Q2_K fixture requires complete 256-value blocks");
  const scales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1];
  const minimums = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 15];
  return Array.from({ length }, (_, index) => {
    const within = index % 256;
    const group = Math.floor(within / 16);
    const code = (within * 5 + group) & 0x03;
    return 0.5 * scales[group]! * code - 0.25 * minimums[group]!;
  });
}

function q2KPayload(length: number): Buffer {
  assert.equal(length % 256, 0, "Q2_K fixture requires complete 256-value blocks");
  const scales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1];
  const minimums = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 15];
  const payload = Buffer.alloc((length / 256) * 84);
  for (let block = 0; block < length / 256; block += 1) {
    const offset = block * 84;
    for (let group = 0; group < 16; group += 1) {
      payload[offset + group] = scales[group]! | (minimums[group]! << 4);
      const codeOffset = offset + 16 + Math.floor(group / 4) * 16;
      const shift = (group % 4) * 2;
      for (let index = 0; index < 16; index += 1) {
        const logicalIndex = group * 16 + index;
        const code = (logicalIndex * 5 + group) & 0x03;
        payload[codeOffset + index]! |= code << shift;
      }
    }
    payload.writeUInt16LE(0x3800, offset + 80); // d = 0.5
    payload.writeUInt16LE(0x3400, offset + 82); // dmin = 0.25
  }
  return payload;
}
