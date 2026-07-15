import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { buildModelIR } from "../src/architecture.js";
import { openCatalog } from "../src/catalog.js";
import { executeReferenceF32, executeReferenceF64, generateReferenceF32, generateReferenceF64 } from "../src/executor.js";
import { materializeReferenceF32Constants, materializeReferenceF64Constants } from "../src/materialize.js";
import { fingerprintIR, readExecutionTraceBundle } from "../src/trace.js";
import { runExecutionTraceComparison, runGenerationTraceComparison } from "../src/trace-runner.js";
import { captureMlxTrace } from "../src/mlx-trace-capture.js";
import type { ModelIR } from "../src/types.js";
import type { ReferenceF32ExecutionResult } from "../src/types.js";

const preview = { outputRows: 1, inputTerms: 1, includeWeights: false } as const;

test("MLX kernel capture independently records dense Llama execution and greedy generation traces", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-mlx-kernel-capture-"));
  try {
    await writeTinyF32Model(directory);
    const source = path.join(directory, "model");
    const python = path.resolve("venv/bin/python");
    const executionTrace = path.join(directory, "mlx-execution.json");
    assert.equal(await captureMlxTrace({ source, output: executionTrace, inputTokens: [1], python, model: "tiny-llama-mlx", revisionOrChecksum: "mlx-kernel-fixture-v1" }), "execution");
    const execution = await runExecutionTraceComparison({ source, trace: executionTrace, report: path.join(directory, "mlx-execution-report.json"), topK: 3, maxAbsoluteError: 1e-5, maxRelativeError: 1e-4 });
    assert.equal(execution.fidelityClass, "lossless-within-dtype");
    assert.equal(execution.reference.runtime, "MLX 0.32 dense-F32 llama independent IR-kernel capture");
    assert.equal(execution.operations.length, 22);

    const generationTrace = path.join(directory, "mlx-generation.json");
    assert.equal(await captureMlxTrace({ source, output: generationTrace, inputTokens: [1], maxNewTokens: 2, python, model: "tiny-llama-mlx", revisionOrChecksum: "mlx-kernel-fixture-v1" }), "generation");
    const generation = await runGenerationTraceComparison({ source, trace: generationTrace, report: path.join(directory, "mlx-generation-report.json"), topK: 3, maxAbsoluteError: 1e-5, maxRelativeError: 1e-4 });
    assert.equal(generation.fidelityClass, "numerically-equivalent");
    assert.equal(generation.generatedTokenIds.length, 2);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("MLX kernel capture independently records Mistral sliding-window, Gemma unit-offset, and Qwen 2/3 attention traces", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-mlx-multi-adapter-capture-"));
  try {
    await writeTinyMistralF32Model(directory);
    await writeTinyGemmaF32Model(directory);
    await writeTinyQwen2F32Model(directory);
    await writeTinyQwen3F32Model(directory);
    const python = path.resolve("venv/bin/python");
    for (const fixture of [
      { name: "mistral", model: "tiny-mistral-mlx", revision: "mistral-mlx-fixture-v1", inputTokens: [1, 1, 1], expectedOperations: 22, executionFidelity: "numerically-equivalent" },
      { name: "gemma", model: "tiny-gemma-mlx", revision: "gemma-mlx-fixture-v1", expectedOperations: 22, executionFidelity: "lossless-within-dtype" },
      // Qwen 2 uses bias-bearing attention projections but deliberately has
      // no Q/K head norms. This keeps its independently captured contract
      // distinct from both the Llama baseline and Qwen 3.
      { name: "qwen2", model: "tiny-qwen2-mlx", revision: "qwen2-mlx-fixture-v1", expectedOperations: 22, executionFidelity: "numerically-equivalent" },
      // The Qwen 3 projections include nonzero bias and Q/K RMSNorm. MLX
      // reduction boundaries therefore differ from the scalar F32 candidate,
      // while the complete trace remains inside the declared tolerance.
      { name: "qwen3", model: "tiny-qwen3-mlx", revision: "qwen3-mlx-fixture-v1", expectedOperations: 24, executionFidelity: "numerically-equivalent" },
    ]) {
      const source = path.join(directory, fixture.name);
      const executionTrace = path.join(directory, `${fixture.name}-mlx-execution.json`);
      const inputTokens = fixture.inputTokens ?? [1];
      assert.equal(await captureMlxTrace({ source, output: executionTrace, inputTokens, python, model: fixture.model, revisionOrChecksum: fixture.revision }), "execution");
      const execution = await runExecutionTraceComparison({ source, trace: executionTrace, report: path.join(directory, `${fixture.name}-mlx-execution-report.json`), topK: 3, maxAbsoluteError: 1e-5, maxRelativeError: 1e-4 });
      assert.equal(execution.fidelityClass, fixture.executionFidelity, fixture.name);
      assert.equal(execution.reference.runtime, `MLX 0.32 dense-F32 ${fixture.name} independent IR-kernel capture`);
      assert.equal(execution.operations.length, fixture.expectedOperations, fixture.name);

      const generationTrace = path.join(directory, `${fixture.name}-mlx-generation.json`);
      assert.equal(await captureMlxTrace({ source, output: generationTrace, inputTokens, maxNewTokens: 2, python, model: fixture.model, revisionOrChecksum: fixture.revision }), "generation");
      const generation = await runGenerationTraceComparison({ source, trace: generationTrace, report: path.join(directory, `${fixture.name}-mlx-generation-report.json`), topK: 3, maxAbsoluteError: 1e-5, maxRelativeError: 1e-4 });
      assert.equal(generation.fidelityClass, "numerically-equivalent", fixture.name);
      assert.equal(generation.generatedTokenIds.length, 2, fixture.name);
    }

    const unsupportedConfig = path.join(directory, "qwen2", "config.json");
    const qwen2Config = JSON.parse(await readFile(unsupportedConfig, "utf8"));
    await writeFile(unsupportedConfig, JSON.stringify({ ...qwen2Config, rope_scaling: { rope_type: "linear", factor: 2 } }));
    await assert.rejects(
      () => captureMlxTrace({ source: path.join(directory, "qwen2"), output: path.join(directory, "qwen2-mlx.json"), inputTokens: [1], python, model: "tiny-qwen2", revisionOrChecksum: "qwen2-mlx-fixture-v1" }),
      /RoPE 'linear' não possui adaptador matemático registrado/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

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

test("F64 Safetensors replays binary64 operations, KV cache, logits, and greedy generation without F32 narrowing", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-f64-trace-"));
  try {
    const source = await writeTinyF64Model(directory);
    const opened = await openCatalog(source, false);
    let ir: ModelIR;
    let execution: ReturnType<typeof executeReferenceF64>;
    let generation: ReturnType<typeof generateReferenceF64>;
    try {
      ir = await buildModelIR(opened.catalog, preview);
      setF64Policy(ir);
      if (!("readDenseF64" in opened.reader) || typeof opened.reader.readDenseF64 !== "function") throw new Error("F64 Safetensors reader unavailable.");
      const tensors = await materializeReferenceF64Constants(ir, opened.catalog, opened.reader);
      execution = executeReferenceF64(ir, { inputIds: [[1]], tensors });
      generation = generateReferenceF64(ir, { inputIds: [[1]], tensors, maxNewTokens: 2 });
    } finally { await opened.close(); }
    const fingerprintOpened = await openCatalog(source, false);
    let fingerprint: string;
    try { fingerprint = fingerprintIR(await buildModelIR(fingerprintOpened.catalog, preview)); } finally { await fingerprintOpened.close(); }
    const trace = path.join(directory, "f64-execution.json");
    await writeFile(trace, JSON.stringify({
      schemaVersion: 1, kind: "execution", source: { files: await checksums(source, ["config.json", "model.safetensors"]) }, irFingerprint: fingerprint,
      candidatePolicy: { dtype: "F64", runtime: "llm-inner scalar IEEE-754 F64" },
      reference: {
        runtime: "independent exact F64 fixture", model: "tiny-llama-f64", revisionOrChecksum: "f64-fixture-v1", containerFormat: "safetensors", quantization: "none", inputTokens: [[1]], dtypePolicy: "F64 scalar fixture",
        operations: operations(ir!).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serializedF64(execution!.values.get(operation.output)!) })),
        pastKeyValues: [...execution!.pastKeyValues].map(([layer, cache]) => ({ layer, key: serializedF64(cache.key), value: serializedF64(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source, trace, report: path.join(directory, "f64-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const mixedPrecision = JSON.parse(await readFile(trace, "utf8"));
    mixedPrecision.reference.operations[0].tensor.dtype = "F32";
    await writeFile(path.join(directory, "f64-mixed.json"), JSON.stringify(mixedPrecision));
    await assert.rejects(() => runExecutionTraceComparison({ source, trace: path.join(directory, "f64-mixed.json"), report: path.join(directory, "f64-mixed-report.json") }), /dtype F64/);

    const generationTrace = path.join(directory, "f64-generation.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation", source: { files: await checksums(source, ["config.json", "model.safetensors"]) }, irFingerprint: fingerprint,
      candidatePolicy: { dtype: "F64", runtime: "llm-inner scalar IEEE-754 F64" },
      reference: {
        runtime: "independent exact F64 fixture", model: "tiny-llama-f64", revisionOrChecksum: "f64-fixture-v1", containerFormat: "safetensors", quantization: "none", inputTokens: [1], promptPositionIds: [0], dtypePolicy: "F64 scalar fixture", maxNewTokens: 2,
        generatedTokenIds: generation!.generatedTokenIds, steps: generation!.steps, selectionLogits: generation!.selectionLogits.map(serializedF64), stepPastKeyValues: serializedStepCaches(generation!.stepPastKeyValues, serializedF64), logits: serializedF64(generation!.logits),
        pastKeyValues: [...generation!.pastKeyValues].map(([layer, cache]) => ({ layer, key: serializedF64(cache.key), value: serializedF64(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source, trace: generationTrace, report: path.join(directory, "f64-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    const corrupt = await readFile(path.join(source, "model.safetensors"));
    corrupt[corrupt.length - 1] = corrupt[corrupt.length - 1]! ^ 1;
    await writeFile(path.join(source, "model.safetensors"), corrupt);
    await assert.rejects(() => runExecutionTraceComparison({ source, trace, report: path.join(directory, "f64-corrupt.json") }), /Checksum divergente/);
  } finally { await rm(directory, { recursive: true, force: true }); }
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
        steps: generated!.steps, selectionLogits: generated!.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(generated!.stepPastKeyValues, serialized), logits: serialized(generated!.logits),
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

    delete malformed.reference.selectionLogits;
    malformed.reference.pastKeyValues.pop();
    await writeFile(trace, JSON.stringify(malformed));
    await assert.rejects(() => runGenerationTraceComparison({ source, trace, report: reportPath }), /logits de seleção para cada token/);

    malformed.reference.selectionLogits = generated!.selectionLogits.map(serialized);
    delete malformed.reference.stepPastKeyValues;
    await writeFile(trace, JSON.stringify(malformed));
    await assert.rejects(() => runGenerationTraceComparison({ source, trace, report: reportPath }), /cache KV pós-decode para cada token/);

    malformed.reference.stepPastKeyValues = serializedStepCaches(generated!.stepPastKeyValues, serialized);
    malformed.reference.generatedTokenIds[0] = 0;
    malformed.reference.steps[0].tokenId = 0;
    await writeFile(trace, JSON.stringify(malformed));
    await assert.rejects(() => runGenerationTraceComparison({ source, trace, report: reportPath }), /logits determinísticos exigem outro argmax/);

    const missingCache = JSON.parse(await readFile(trace, "utf8"));
    missingCache.reference.generatedTokenIds = generated!.generatedTokenIds;
    missingCache.reference.steps = generated!.steps;
    missingCache.reference.stepPastKeyValues = serializedStepCaches(generated!.stepPastKeyValues, serialized).map(() => []);
    missingCache.reference.pastKeyValues = [];
    await writeFile(trace, JSON.stringify(missingCache));
    await assert.rejects(
      () => runGenerationTraceComparison({ source, trace, report: reportPath }),
      /não cobre exatamente as camadas KV independentes do IR; ausentes \[0\]/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("generation traces bind shared-KV continuation state to its producer exactly once", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-shared-kv-generation-trace-"));
  try {
    const source = await writeTinySharedKvF32Model(directory);
    const executed = await executeFixture(source, [[1]]);
    const generated = await generateFixture(source);
    const attentions = executed.ir.layers.flatMap((layer) => layer.operations).filter((operation) => operation.op === "scaled_dot_product_attention");
    assert.equal(attentions.length, 2);
    assert.equal(attentions[1]?.op, "scaled_dot_product_attention");
    if (attentions[1]?.op === "scaled_dot_product_attention") assert.deepEqual(attentions[1].kvSharing, { enabled: true, producerLayer: 0, group: "full_attention" });
    assert.deepEqual([...generated.pastKeyValues.keys()], [0]);

    const trace = path.join(directory, "shared-kv-generation-trace.json");
    await writeFile(trace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(source, ["config.json", "model.safetensors"]) },
      irFingerprint: executed.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent shared-KV F32 fixture", model: "tiny-llama-shared-kv", revisionOrChecksum: "shared-kv-fixture-v1",
        containerFormat: "safetensors", quantization: "none", inputTokens: [1], promptPositionIds: [0], dtypePolicy: "F32 scalar shared-KV fixture", maxNewTokens: 2,
        generatedTokenIds: generated.generatedTokenIds, steps: generated.steps, selectionLogits: generated.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(generated.stepPastKeyValues, serialized), logits: serialized(generated.logits),
        pastKeyValues: [...generated.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const report = await runGenerationTraceComparison({ source, trace, report: path.join(directory, "shared-kv-generation-report.json"), topK: 3 });
    assert.equal(report.fidelityClass, "lossless-within-dtype");
    assert.equal(report.firstDivergence, null);

    const duplicateConsumer = JSON.parse(await readFile(trace, "utf8"));
    const producer = duplicateConsumer.reference.pastKeyValues[0];
    duplicateConsumer.reference.pastKeyValues.push({ ...producer, layer: 1 });
    duplicateConsumer.reference.stepPastKeyValues = duplicateConsumer.reference.stepPastKeyValues.map((snapshot: unknown[]) => [...snapshot, { ...producer, layer: 1 }]);
    await writeFile(trace, JSON.stringify(duplicateConsumer));
    await assert.rejects(
      () => runGenerationTraceComparison({ source, trace, report: path.join(directory, "shared-kv-duplicate-report.json") }),
      /não cobre exatamente as camadas KV independentes do IR; ausentes \[\], extras \[1\]/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("F16 and BF16 dense Safetensors and GGUF packages replay independent F32 operation, cache, logits, and generation traces", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-dense-16-trace-"));
  try {
    await writeTinyF32Model(directory);
    const denseSource = path.join(directory, "model");
    const denseExecution = await executeFixture(denseSource, [[1]]);
    const denseGeneration = await generateFixture(denseSource);

    for (const storageDtype of ["F16", "BF16"] as const) {
      const safetensorsSource = path.join(directory, `safetensors-${storageDtype.toLowerCase()}`);
      await writeTiny16SafetensorsModel(safetensorsSource, storageDtype);
      const ggufSource = path.join(directory, `llama-${storageDtype.toLowerCase()}.gguf`);
      await writeTiny16GgufModel(ggufSource, storageDtype);

      for (const [label, source, files, format] of [
        ["safetensors", safetensorsSource, ["config.json", "model.safetensors"], "safetensors"],
        ["gguf", ggufSource, [path.basename(ggufSource)], "gguf v3"],
      ] as const) {
        const target = await executeFixture(source, [[1]]);
        const checksumRoot = label === "safetensors" ? source : path.dirname(source);
        const executionTrace = path.join(directory, `${label}-${storageDtype.toLowerCase()}-execution-trace.json`);
        await writeFile(executionTrace, JSON.stringify({
          schemaVersion: 1, kind: "execution",
          source: { files: await checksums(checksumRoot, files) }, irFingerprint: target.fingerprint,
          candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
          reference: {
            runtime: "independent exact F32 dense-storage fixture", model: `tiny-llama-${label}-${storageDtype.toLowerCase()}`,
            revisionOrChecksum: `dense-${storageDtype.toLowerCase()}-${label}-fixture-v1`, containerFormat: format,
            quantization: "none", inputTokens: [[1]], dtypePolicy: `F32 scalar fixture from exact ${storageDtype} storage values`,
            operations: operations(denseExecution.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(denseExecution.candidate.values.get(operation.output)!) })),
            pastKeyValues: [...denseExecution.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
          },
        }, null, 2));
        const executionReport = await runExecutionTraceComparison({ source, trace: executionTrace, report: path.join(directory, `${label}-${storageDtype.toLowerCase()}-execution-report.json`), topK: 3 });
        assert.equal(executionReport.fidelityClass, "lossless-within-dtype", `${label} ${storageDtype} execution trace`);
        assert.equal(executionReport.firstDivergentOperation, null);

        const generationTrace = path.join(directory, `${label}-${storageDtype.toLowerCase()}-generation-trace.json`);
        await writeFile(generationTrace, JSON.stringify({
          schemaVersion: 1, kind: "generation",
          source: { files: await checksums(checksumRoot, files) }, irFingerprint: target.fingerprint,
          candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
          reference: {
            runtime: "independent exact F32 dense-storage fixture", model: `tiny-llama-${label}-${storageDtype.toLowerCase()}`,
            revisionOrChecksum: `dense-${storageDtype.toLowerCase()}-${label}-fixture-v1`, containerFormat: format,
            quantization: "none", inputTokens: [1], promptPositionIds: [0],
            dtypePolicy: `F32 scalar fixture from exact ${storageDtype} storage values`, maxNewTokens: 2,
            generatedTokenIds: denseGeneration.generatedTokenIds, steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
            pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
          },
        }, null, 2));
        const generationReport = await runGenerationTraceComparison({ source, trace: generationTrace, report: path.join(directory, `${label}-${storageDtype.toLowerCase()}-generation-report.json`), topK: 3 });
        assert.equal(generationReport.fidelityClass, "lossless-within-dtype", `${label} ${storageDtype} generation trace`);
        assert.equal(generationReport.firstDivergence, null);

        const mutableFile = label === "safetensors" ? path.join(source, "model.safetensors") : source;
        const mutated = await readFile(mutableFile);
        mutated[mutated.length - 1] = mutated[mutated.length - 1]! ^ 1;
        await writeFile(mutableFile, mutated);
        await assert.rejects(() => runExecutionTraceComparison({ source, trace: executionTrace, report: path.join(directory, "corrupt-report.json") }), /Checksum divergente/);
      }
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Gemma F32 Safetensors replays complete execution and greedy generation with embedding scale and unit-offset norms", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma-trace-"));
  try {
    await writeTinyGemmaF32Model(directory);
    const source = path.join(directory, "gemma");
    const executed = await executeFixture(source, [[1]]);

    // Gemma's embedding scale and (1 + weight) RMSNorm transform are not
    // interchangeable with the Llama defaults. Assert the lowered contract
    // before using its operation captures as evidence for replay.
    assert.equal(executed.ir.architecture.modelType, "gemma");
    assert.equal(executed.ir.prelude[0]?.op, "embedding");
    assert.equal(executed.ir.prelude[0]?.scale, Math.sqrt(2));
    assert.ok(operations(executed.ir).filter((operation) => operation.op === "rms_norm").every((operation) => operation.weightTransform === "one_plus_weight"));
    const activation = operations(executed.ir).find((operation) => operation.id === "layer_0_activation");
    assert.equal(activation?.op, "activation");
    if (activation?.op !== "activation") throw new Error("Gemma fixture did not lower its declared activation.");
    assert.equal(activation.function, "gelu");
    assert.equal(activation.approximation, "tanh");

    const executionTrace = path.join(directory, "gemma-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(source, ["config.json", "model.safetensors"]) },
      irFingerprint: executed.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent Gemma F32 formula fixture", model: "tiny-gemma-unit-offset", revisionOrChecksum: "gemma-formula-fixture-v1",
        containerFormat: "safetensors", quantization: "none", inputTokens: [[1]], dtypePolicy: "F32 scalar Gemma fixture",
        operations: operations(executed.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(executed.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...executed.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source, trace: executionTrace, report: path.join(directory, "gemma-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const generated = await generateFixture(source);
    const generationTrace = path.join(directory, "gemma-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(source, ["config.json", "model.safetensors"]) },
      irFingerprint: executed.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent Gemma F32 formula fixture", model: "tiny-gemma-unit-offset", revisionOrChecksum: "gemma-formula-fixture-v1",
        containerFormat: "safetensors", quantization: "none", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar Gemma fixture", maxNewTokens: 2, generatedTokenIds: generated.generatedTokenIds, steps: generated.steps, selectionLogits: generated.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(generated.stepPastKeyValues, serialized),
        logits: serialized(generated.logits), pastKeyValues: [...generated.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source, trace: generationTrace, report: path.join(directory, "gemma-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    const corrupted = await readFile(path.join(source, "model.safetensors"));
    corrupted[corrupted.length - 1] = corrupted[corrupted.length - 1]! ^ 1;
    await writeFile(path.join(source, "model.safetensors"), corrupted);
    await assert.rejects(() => runExecutionTraceComparison({ source, trace: executionTrace, report: path.join(directory, "gemma-corrupt-report.json") }), /Checksum divergente/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Qwen 3 F32 Safetensors replays Q/K head norms, attention biases, and greedy generation", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-qwen3-trace-"));
  try {
    await writeTinyQwen3F32Model(directory);
    const source = path.join(directory, "qwen3");
    const executed = await executeFixture(source, [[1]]);

    // Qwen 3's Q/K norms act on [batch, heads, sequence, head_dim] after
    // reshape. They must not be lowered as a hidden-size norm before RoPE.
    assert.equal(executed.ir.architecture.modelType, "qwen3");
    const qNorm = executed.ir.layers[0]!.operations.find((operation) => operation.id === "layer_0_q_norm");
    const kNorm = executed.ir.layers[0]!.operations.find((operation) => operation.id === "layer_0_k_norm");
    assert.equal(qNorm?.op, "rms_norm");
    assert.equal(kNorm?.op, "rms_norm");
    if (qNorm?.op !== "rms_norm" || kNorm?.op !== "rms_norm") throw new Error("Qwen 3 fixture did not lower Q/K RMSNorm.");
    assert.equal(qNorm.input, "layer_0_q_heads");
    assert.equal(kNorm.input, "layer_0_k_heads");
    assert.equal(qNorm.weightTransform, "direct");
    assert.equal(kNorm.weightTransform, "direct");
    const attentionLinears = executed.ir.layers[0]!.operations.filter((operation) => operation.op === "linear" && ["layer_0_q_proj", "layer_0_k_proj", "layer_0_v_proj", "layer_0_o_proj"].includes(operation.id));
    assert.equal(attentionLinears.length, 4);
    assert.ok(attentionLinears.every((operation) => operation.op === "linear" && operation.bias !== undefined));

    const executionTrace = path.join(directory, "qwen3-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(source, ["config.json", "model.safetensors"]) },
      irFingerprint: executed.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "deterministic Qwen 3 F32 formula fixture", model: "tiny-qwen3-qk-norm", revisionOrChecksum: "qwen3-qk-norm-fixture-v1",
        containerFormat: "safetensors", quantization: "none", inputTokens: [[1]], dtypePolicy: "F32 scalar Qwen 3 fixture",
        operations: operations(executed.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(executed.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...executed.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source, trace: executionTrace, report: path.join(directory, "qwen3-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const generated = await generateFixture(source);
    const generationTrace = path.join(directory, "qwen3-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(source, ["config.json", "model.safetensors"]) },
      irFingerprint: executed.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "deterministic Qwen 3 F32 formula fixture", model: "tiny-qwen3-qk-norm", revisionOrChecksum: "qwen3-qk-norm-fixture-v1",
        containerFormat: "safetensors", quantization: "none", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar Qwen 3 fixture", maxNewTokens: 2, generatedTokenIds: generated.generatedTokenIds, steps: generated.steps, selectionLogits: generated.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(generated.stepPastKeyValues, serialized),
        logits: serialized(generated.logits), pastKeyValues: [...generated.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source, trace: generationTrace, report: path.join(directory, "qwen3-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);
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
        steps: safetensorsGeneration.steps, selectionLogits: safetensorsGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(safetensorsGeneration.stepPastKeyValues, serialized), logits: serialized(safetensorsGeneration.logits),
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

test("MLX affine U32 Llama replays independently constructed dense F32 execution and generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-mlx-affine-trace-"));
  try {
    const { denseSource, quantizedSource } = await writeMlxAffineTraceFixture(directory);

    // The dense package is produced directly from the declared affine formula
    // (scale * packedCode + bias). It never invokes the MLX catalog reader or
    // dequantizer, so it witnesses the complete native U32 materialization
    // boundary rather than merely replaying candidate output.
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "mlx-affine-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(quantizedSource, ["config.json", "model.safetensors"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared MLX affine F32 fixture", model: "mlx-affine-llama-32", revisionOrChecksum: "mlx-affine-formula-fixture-v1",
        containerFormat: "mlx-safetensors", quantization: "MLX affine U32 4-bit", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared MLX affine formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "mlx-affine-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "mlx-affine-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(quantizedSource, ["config.json", "model.safetensors"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared MLX affine F32 fixture", model: "mlx-affine-llama-32", revisionOrChecksum: "mlx-affine-formula-fixture-v1",
        containerFormat: "mlx-safetensors", quantization: "MLX affine U32 4-bit", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared MLX affine formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "mlx-affine-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    const corrupted = await readFile(path.join(quantizedSource, "model.safetensors"));
    corrupted[corrupted.length - 1] = corrupted[corrupted.length - 1]! ^ 1;
    await writeFile(path.join(quantizedSource, "model.safetensors"), corrupted);
    await assert.rejects(() => runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "corrupt-report.json") }), /Checksum divergente/);
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
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
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
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
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

test("Q5_1 GGUF Llama replays independently constructed affine dense F32 forward and greedy-generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-q5_1-trace-"));
  try {
    const denseSource = path.join(directory, "dense-reference.gguf");
    const quantizedSource = path.join(directory, "q5_1.gguf");
    await writeQ5_1TraceFixture(denseSource, quantizedSource);

    // The dense package applies d*q+m directly with d=0.5 and m=-1 to each
    // declared unsigned five-bit code. The packed writer independently puts
    // q[0..15] in low nibbles, q[16..31] in high nibbles, and bit 4 in qh.
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "q5_1-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["q5_1.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q5_1 F32 fixture", model: "q5_1-llama-256", revisionOrChecksum: "q5_1-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q5_1", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared Q5_1 formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "q5_1-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "q5_1-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["q5_1.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q5_1 F32 fixture", model: "q5_1-llama-256", revisionOrChecksum: "q5_1-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q5_1", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared Q5_1 formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "q5_1-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    // Flip qh's last high-bit byte. The trace must be bound to this exact
    // packed affine source rather than only its semantic fixture description.
    const corrupted = await readFile(quantizedSource);
    const lastHighBitByte = corrupted.length - 20;
    corrupted[lastHighBitByte] = corrupted[lastHighBitByte]! ^ 1;
    await writeFile(quantizedSource, corrupted);
    await assert.rejects(() => runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "corrupt-report.json") }), /Checksum divergente/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Q5_0 GGUF Llama replays independently constructed centered dense F32 forward and greedy-generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-q5_0-trace-"));
  try {
    const denseSource = path.join(directory, "dense-reference.gguf");
    const quantizedSource = path.join(directory, "q5_0.gguf");
    await writeQ5_0TraceFixture(denseSource, quantizedSource);

    // The dense package independently applies d*(q-16) with d=0.5 to every
    // declared five-bit code. The packed writer puts q[0..15] in low
    // nibbles, q[16..31] in high nibbles, and bit 4 in the qh bit plane.
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "q5_0-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["q5_0.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q5_0 F32 fixture", model: "q5_0-llama-256", revisionOrChecksum: "q5_0-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q5_0", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared Q5_0 formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "q5_0-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "q5_0-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["q5_0.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q5_0 F32 fixture", model: "q5_0-llama-256", revisionOrChecksum: "q5_0-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q5_0", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared Q5_0 formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "q5_0-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    // A high-bit-plane mutation changes centered codes but must be rejected
    // by source integrity verification before any trace can be accepted.
    const corrupted = await readFile(quantizedSource);
    const lastHighBitByte = corrupted.length - 20;
    corrupted[lastHighBitByte] = corrupted[lastHighBitByte]! ^ 1;
    await writeFile(quantizedSource, corrupted);
    await assert.rejects(() => runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "corrupt-report.json") }), /Checksum divergente/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Q4_0 GGUF Llama replays independently constructed centered dense F32 forward and greedy-generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-q4_0-trace-"));
  try {
    const denseSource = path.join(directory, "dense-reference.gguf");
    const quantizedSource = path.join(directory, "q4_0.gguf");
    await writeQ4_0TraceFixture(denseSource, quantizedSource);

    // The dense package independently applies d*(q-8) with d=0.5 to every
    // declared four-bit code. The packed writer puts q[0..15] in low nibbles
    // and q[16..31] in high nibbles without using the candidate reader.
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "q4_0-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["q4_0.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q4_0 F32 fixture", model: "q4_0-llama-256", revisionOrChecksum: "q4_0-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q4_0", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared Q4_0 formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "q4_0-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "q4_0-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["q4_0.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q4_0 F32 fixture", model: "q4_0-llama-256", revisionOrChecksum: "q4_0-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q4_0", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared Q4_0 formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "q4_0-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    // The final payload byte packs both a low and high centered code. Source
    // integrity must reject any mutation before a trace is accepted.
    const corrupted = await readFile(quantizedSource);
    corrupted[corrupted.length - 1] = corrupted[corrupted.length - 1]! ^ 1;
    await writeFile(quantizedSource, corrupted);
    await assert.rejects(() => runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "corrupt-report.json") }), /Checksum divergente/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Q4_1 GGUF Llama replays independently constructed affine dense F32 forward and greedy-generation evidence", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-q4_1-trace-"));
  try {
    const denseSource = path.join(directory, "dense-reference.gguf");
    const quantizedSource = path.join(directory, "q4_1.gguf");
    await writeQ4_1TraceFixture(denseSource, quantizedSource);

    // The dense package applies d*q+m directly with d=0.5 and m=-1 to all
    // unsigned four-bit codes. The packed writer places q[0..15] in low
    // nibbles and q[16..31] in high nibbles without using the candidate reader.
    const dense = await executeFixture(denseSource, [[1]]);
    const quantized = await executeFixture(quantizedSource, [[1]]);
    assert.deepEqual(serialized(quantized.candidate.values.get("logits")!), serialized(dense.candidate.values.get("logits")!));
    assert.deepEqual([...quantized.candidate.pastKeyValues], [...dense.candidate.pastKeyValues]);

    const executionTrace = path.join(directory, "q4_1-execution-trace.json");
    await writeFile(executionTrace, JSON.stringify({
      schemaVersion: 1, kind: "execution",
      source: { files: await checksums(directory, ["q4_1.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q4_1 F32 fixture", model: "q4_1-llama-256", revisionOrChecksum: "q4_1-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q4_1", inputTokens: [[1]], dtypePolicy: "F32 scalar fixture from declared Q4_1 formula",
        operations: operations(quantized.ir).map((operation) => ({ operationId: operation.id, output: operation.output, tensor: serialized(dense.candidate.values.get(operation.output)!) })),
        pastKeyValues: [...dense.candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const executionReport = await runExecutionTraceComparison({ source: quantizedSource, trace: executionTrace, report: path.join(directory, "q4_1-execution-report.json"), topK: 3 });
    assert.equal(executionReport.fidelityClass, "lossless-within-dtype");
    assert.equal(executionReport.firstDivergentOperation, null);

    const denseGeneration = await generateFixture(denseSource);
    const generationTrace = path.join(directory, "q4_1-generation-trace.json");
    await writeFile(generationTrace, JSON.stringify({
      schemaVersion: 1, kind: "generation",
      source: { files: await checksums(directory, ["q4_1.gguf"]) },
      irFingerprint: quantized.fingerprint,
      candidatePolicy: { dtype: "F32", runtime: "llm-inner scalar IEEE-754 F32" },
      reference: {
        runtime: "independent declared GGML Q4_1 F32 fixture", model: "q4_1-llama-256", revisionOrChecksum: "q4_1-formula-fixture-v1",
        containerFormat: "gguf v3", quantization: "GGML_TYPE_Q4_1", inputTokens: [1], promptPositionIds: [0],
        dtypePolicy: "F32 scalar fixture from declared Q4_1 formula", maxNewTokens: 2, generatedTokenIds: denseGeneration.generatedTokenIds,
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
        pastKeyValues: [...denseGeneration.pastKeyValues].map(([layer, cache]) => ({ layer, key: serialized(cache.key), value: serialized(cache.value) })),
      },
    }, null, 2));
    const generationReport = await runGenerationTraceComparison({ source: quantizedSource, trace: generationTrace, report: path.join(directory, "q4_1-generation-report.json"), topK: 3 });
    assert.equal(generationReport.fidelityClass, "lossless-within-dtype");
    assert.equal(generationReport.firstDivergence, null);

    // The final Q4_1 block stores its F16 minimum at byte -18. Integrity is
    // over the packed source, so a minimum mutation fails before comparison.
    const corrupted = await readFile(quantizedSource);
    corrupted[corrupted.length - 18] = corrupted[corrupted.length - 18]! ^ 1;
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
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
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
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
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
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
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
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
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
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
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
        steps: denseGeneration.steps, selectionLogits: denseGeneration.selectionLogits.map(serialized), stepPastKeyValues: serializedStepCaches(denseGeneration.stepPastKeyValues, serialized), logits: serialized(denseGeneration.logits),
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

function setF64Policy(ir: ModelIR): void {
  for (const operation of operations(ir)) {
    operation.dtypePolicy = { computeDtype: "F64", accumulationDtype: "F64", outputDtype: "F64" };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F64";
  }
}

function serialized(tensor: { shape: number[]; values: Float32Array }) {
  return { dtype: "F32", shape: tensor.shape, valuesBase64: Buffer.from(tensor.values.buffer, tensor.values.byteOffset, tensor.values.byteLength).toString("base64") };
}

function serializedF64(tensor: { shape: number[]; values: Float64Array }) {
  return { dtype: "F64", shape: tensor.shape, valuesBase64: Buffer.from(tensor.values.buffer, tensor.values.byteOffset, tensor.values.byteLength).toString("base64") };
}

function serializedStepCaches<T extends Float32Array | Float64Array>(
  snapshots: ReadonlyArray<ReadonlyMap<number, { key: { shape: number[]; values: T }; value: { shape: number[]; values: T } }>>,
  serialize: (tensor: { shape: number[]; values: T }) => unknown,
) {
  return snapshots.map((snapshot) => [...snapshot].map(([layer, cache]) => ({
    layer,
    key: serialize(cache.key),
    value: serialize(cache.value),
  })));
}

async function checksums(directory: string, files: readonly string[]) {
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
  const weights = tinyLlamaWeights();
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
  await writeFile(path.join(directory, "config.json"), JSON.stringify(tinyLlamaConfig()));
  await writeFile(path.join(directory, "model.safetensors"), Buffer.concat([prefix, encoded, ...payloads]));
}

/** A two-layer Llama contract where layer 1 reuses layer 0 post-RoPE KV. */
async function writeTinySharedKvF32Model(root: string): Promise<string> {
  const directory = path.join(root, "shared-kv-model");
  await mkdir(directory);
  const weights = tinyLlamaWeights();
  for (const [name, shape, values] of tinyLlamaWeights()) {
    if (!name.startsWith("model.layers.0.") || name.includes("self_attn.k_proj") || name.includes("self_attn.v_proj")) continue;
    weights.push([name.replace("model.layers.0", "model.layers.1"), shape, values]);
  }
  await writeFile(path.join(directory, "config.json"), JSON.stringify({
    ...tinyLlamaConfig(), num_hidden_layers: 2, num_kv_shared_layers: 1,
  }));
  await writeSafetensorsFixture(path.join(directory, "model.safetensors"), weights.map(([name, shape, values]) => [name, "F32", shape, values]));
  return directory;
}

async function writeTinyF64Model(root: string): Promise<string> {
  const directory = path.join(root, "model");
  await mkdir(directory);
  const header: Record<string, unknown> = {};
  let offset = 0;
  const payloads = tinyLlamaWeights().map(([name, shape, values]) => {
    const payload = Buffer.alloc(values.length * 8);
    values.forEach((value, index) => payload.writeDoubleLE(value + (value !== 0 ? 1e-13 : 0), index * 8));
    header[name] = { dtype: "F64", shape, data_offsets: [offset, offset + payload.length] };
    offset += payload.length;
    return payload;
  });
  const encoded = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(encoded.length));
  await writeFile(path.join(directory, "config.json"), JSON.stringify(tinyLlamaConfig()));
  await writeFile(path.join(directory, "model.safetensors"), Buffer.concat([prefix, encoded, ...payloads]));
  return directory;
}

async function writeTiny16SafetensorsModel(directory: string, storageDtype: "F16" | "BF16"): Promise<void> {
  await mkdir(directory);
  await writeFile(path.join(directory, "config.json"), JSON.stringify(tinyLlamaConfig()));
  const header: Record<string, unknown> = {};
  let offset = 0;
  const payloads = tinyLlamaWeights().map(([name, shape, values]) => {
    const payload = dense16Payload(values, storageDtype);
    header[name] = { dtype: storageDtype, shape, data_offsets: [offset, offset + payload.length] };
    offset += payload.length;
    return payload;
  });
  const encoded = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(encoded.length));
  await writeFile(path.join(directory, "model.safetensors"), Buffer.concat([prefix, encoded, ...payloads]));
}

function tinyLlamaConfig() {
  return { model_type: "llama", hidden_size: 2, intermediate_size: 2, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1, head_dim: 2, vocab_size: 3, rms_norm_eps: 1e-6, hidden_act: "silu" };
}

function tinyLlamaWeights(): Array<[string, number[], number[]]> {
  return [
    ["model.embed_tokens.weight", [3, 2], [0, 0, 3, 4, 0, 0]],
    ["model.layers.0.input_layernorm.weight", [2], [1, 1]],
    ...["q_proj", "k_proj", "v_proj", "o_proj"].map((projection): [string, number[], number[]] => [`model.layers.0.self_attn.${projection}.weight`, [2, 2], [1, 0, 0, 1]]),
    ["model.layers.0.post_attention_layernorm.weight", [2], [1, 1]],
    ["model.layers.0.mlp.gate_proj.weight", [2, 2], [0, 0, 0, 0]], ["model.layers.0.mlp.up_proj.weight", [2, 2], [0, 0, 0, 0]], ["model.layers.0.mlp.down_proj.weight", [2, 2], [0, 0, 0, 0]],
    ["model.norm.weight", [2], [1, 1]], ["lm_head.weight", [3, 2], [1, 0, 0, 1, 1, 1]],
  ];
}

/** Mistral's declared local window must remain active through prompt and decode capture. */
async function writeTinyMistralF32Model(root: string): Promise<void> {
  const directory = path.join(root, "mistral");
  await mkdir(directory);
  await writeFile(path.join(directory, "config.json"), JSON.stringify({ ...tinyLlamaConfig(), model_type: "mistral", sliding_window: 2 }));
  await writeSafetensorsFixture(path.join(directory, "model.safetensors"), tinyLlamaWeights().map(([name, shape, values]) => [name, "F32", shape, values]));
}

/** A non-Llama decoder fixture whose nonzero path exercises Gemma-specific lowering semantics. */
async function writeTinyGemmaF32Model(root: string): Promise<void> {
  const directory = path.join(root, "gemma");
  await mkdir(directory);
  const identity = [1, 0, 0, 1];
  const zeroNorm = [0, 0]; // Gemma consumes this as (1 + weight), not directly.
  const weights: Array<[string, "F32", number[], number[]]> = [
    ["model.embed_tokens.weight", "F32", [3, 2], [0, 0, 3, 4, 1, -1]],
    ["model.layers.0.input_layernorm.weight", "F32", [2], zeroNorm],
    ...["q_proj", "k_proj", "v_proj", "o_proj"].map((projection): [string, "F32", number[], number[]] => [`model.layers.0.self_attn.${projection}.weight`, "F32", [2, 2], identity]),
    ["model.layers.0.post_attention_layernorm.weight", "F32", [2], zeroNorm],
    ["model.layers.0.mlp.gate_proj.weight", "F32", [2, 2], identity], ["model.layers.0.mlp.up_proj.weight", "F32", [2, 2], identity], ["model.layers.0.mlp.down_proj.weight", "F32", [2, 2], identity],
    ["model.norm.weight", "F32", [2], zeroNorm],
  ];
  await writeFile(path.join(directory, "config.json"), JSON.stringify({
    model_type: "gemma", hidden_size: 2, intermediate_size: 2, num_hidden_layers: 1, num_attention_heads: 1,
    num_key_value_heads: 1, head_dim: 2, vocab_size: 3, rms_norm_eps: 1e-6, hidden_act: "gelu_pytorch_tanh", tie_word_embeddings: true,
  }));
  await writeSafetensorsFixture(path.join(directory, "model.safetensors"), weights);
}

/** Qwen 2 keeps Q/K unnormalized but makes attention bias an explicit contract. */
async function writeTinyQwen2F32Model(root: string): Promise<void> {
  const directory = path.join(root, "qwen2");
  await mkdir(directory);
  const identity = [1, 0, 0, 1];
  const weights: Array<[string, "F32", number[], number[]]> = [
    ["model.embed_tokens.weight", "F32", [3, 2], [0, 0, 2, -1, -1, 1]],
    ["model.layers.0.input_layernorm.weight", "F32", [2], [1, 1]],
    ...["q_proj", "k_proj", "v_proj", "o_proj"].flatMap((projection, index): Array<[string, "F32", number[], number[]]> => [
      [`model.layers.0.self_attn.${projection}.weight`, "F32", [2, 2], identity],
      [`model.layers.0.self_attn.${projection}.bias`, "F32", [2], [0.125 * (index + 1), -0.0625 * (index + 1)]],
    ]),
    ["model.layers.0.post_attention_layernorm.weight", "F32", [2], [1, 1]],
    ["model.layers.0.mlp.gate_proj.weight", "F32", [2, 2], identity],
    ["model.layers.0.mlp.up_proj.weight", "F32", [2, 2], identity],
    ["model.layers.0.mlp.down_proj.weight", "F32", [2, 2], identity],
    ["model.norm.weight", "F32", [2], [1, 1]],
    ["lm_head.weight", "F32", [3, 2], [1, 0, 0, 1, 1, -1]],
  ];
  await writeFile(path.join(directory, "config.json"), JSON.stringify({
    model_type: "qwen2", hidden_size: 2, intermediate_size: 2, num_hidden_layers: 1, num_attention_heads: 1,
    num_key_value_heads: 1, head_dim: 2, vocab_size: 3, rms_norm_eps: 1e-6, hidden_act: "silu", attention_bias: true,
  }));
  await writeSafetensorsFixture(path.join(directory, "model.safetensors"), weights);
}

/** A Qwen 3 fixture with declared Q/K head norms and bias-bearing attention projections. */
async function writeTinyQwen3F32Model(root: string): Promise<void> {
  const directory = path.join(root, "qwen3");
  await mkdir(directory);
  const identity = [1, 0, 0, 1];
  const weights: Array<[string, "F32", number[], number[]]> = [
    ["model.embed_tokens.weight", "F32", [3, 2], [0, 0, 2, -1, -1, 1]],
    ["model.layers.0.input_layernorm.weight", "F32", [2], [1, 1]],
    ...["q_proj", "k_proj", "v_proj", "o_proj"].flatMap((projection, index): Array<[string, "F32", number[], number[]]> => [
      [`model.layers.0.self_attn.${projection}.weight`, "F32", [2, 2], identity],
      [`model.layers.0.self_attn.${projection}.bias`, "F32", [2], [0.125 * (index + 1), -0.0625 * (index + 1)]],
    ]),
    ["model.layers.0.self_attn.q_norm.weight", "F32", [2], [1.5, 0.5]],
    ["model.layers.0.self_attn.k_norm.weight", "F32", [2], [0.75, 1.25]],
    ["model.layers.0.post_attention_layernorm.weight", "F32", [2], [1, 1]],
    ["model.layers.0.mlp.gate_proj.weight", "F32", [2, 2], identity],
    ["model.layers.0.mlp.up_proj.weight", "F32", [2, 2], identity],
    ["model.layers.0.mlp.down_proj.weight", "F32", [2, 2], identity],
    ["model.norm.weight", "F32", [2], [1, 1]],
    ["lm_head.weight", "F32", [3, 2], [1, 0, 0, 1, 1, -1]],
  ];
  await writeFile(path.join(directory, "config.json"), JSON.stringify({
    model_type: "qwen3", hidden_size: 2, intermediate_size: 2, num_hidden_layers: 1, num_attention_heads: 1,
    num_key_value_heads: 1, head_dim: 2, vocab_size: 3, rms_norm_eps: 1e-6, hidden_act: "silu", attention_bias: true,
  }));
  await writeSafetensorsFixture(path.join(directory, "model.safetensors"), weights);
}

/**
 * Builds paired 32-wide Llama packages for the native MLX affine reader. Each
 * U32 row holds eight 4-bit codes per word; the F32 side computes every value
 * independently as scale[group] * code + bias[group].
 */
async function writeMlxAffineTraceFixture(root: string): Promise<{ denseSource: string; quantizedSource: string }> {
  const denseSource = path.join(root, "dense");
  const quantizedSource = path.join(root, "mlx-affine");
  await mkdir(denseSource);
  await mkdir(quantizedSource);
  const width = 32;
  const groups = 4;
  const groupSize = width / groups;
  const matrixNames = [
    "model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight", "lm_head.weight",
  ];
  const denseTensors: Array<[string, "F32", number[], number[]]> = [];
  const mlxTensors: Array<[string, "F32" | "U32", number[], number[]]> = [];
  for (const [matrixIndex, name] of matrixNames.entries()) {
    const scales = Array.from({ length: width * groups }, (_, index) => Math.fround(0.03125 * (1 + ((index + matrixIndex) % 4))));
    const biases = Array.from({ length: width * groups }, (_, index) => Math.fround(-0.25 + 0.0625 * ((index + matrixIndex) % 5)));
    const codes = Array.from({ length: width * width }, (_, index) => (index * 7 + matrixIndex * 3 + Math.floor(index / width)) & 0x0f);
    const denseValues = codes.map((code, index) => Math.fround(Math.fround(scales[Math.floor(index / width) * groups + Math.floor((index % width) / groupSize)]!) * code + biases[Math.floor(index / width) * groups + Math.floor((index % width) / groupSize)]!));
    const packed = packMlxAffineCodes(codes, width, width, 4);
    denseTensors.push([name, "F32", [width, width], denseValues]);
    mlxTensors.push([name, "U32", [width, width / 8], packed], [name.slice(0, -".weight".length) + ".scales", "F32", [width, groups], scales], [name.slice(0, -".weight".length) + ".biases", "F32", [width, groups], biases]);
  }
  for (const name of ["model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight", "model.norm.weight"]) {
    denseTensors.push([name, "F32", [width], new Array<number>(width).fill(1)]);
    mlxTensors.push([name, "F32", [width], new Array<number>(width).fill(1)]);
  }
  const config = {
    model_type: "llama", hidden_size: width, intermediate_size: width, num_hidden_layers: 1, num_attention_heads: 1,
    num_key_value_heads: 1, head_dim: width, vocab_size: width, rms_norm_eps: 1e-6, hidden_act: "silu",
  };
  await writeFile(path.join(denseSource, "config.json"), JSON.stringify(config));
  await writeFile(path.join(quantizedSource, "config.json"), JSON.stringify({ ...config, quantization: { bits: 4, group_size: groupSize, mode: "affine" } }));
  await writeSafetensorsFixture(path.join(denseSource, "model.safetensors"), denseTensors);
  await writeSafetensorsFixture(path.join(quantizedSource, "model.safetensors"), mlxTensors);
  return { denseSource, quantizedSource };
}

function packMlxAffineCodes(codes: readonly number[], rows: number, columns: number, bits: number): number[] {
  assert.equal(codes.length, rows * columns);
  assert.equal((columns * bits) % 32, 0, "MLX fixture requires whole U32 rows");
  const wordsPerRow = columns * bits / 32;
  const packed = new Array<number>(rows * wordsPerRow).fill(0);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const bitOffset = column * bits;
      const word = row * wordsPerRow + Math.floor(bitOffset / 32);
      packed[word] = (packed[word]! | (codes[row * columns + column]! << (bitOffset % 32))) >>> 0;
    }
  }
  return packed;
}

async function writeSafetensorsFixture(file: string, tensors: Array<[string, "F32" | "U32", number[], number[]]>): Promise<void> {
  const header: Record<string, unknown> = {};
  let offset = 0;
  const payloads = tensors.map(([name, dtype, shape, values]) => {
    const payload = Buffer.alloc(values.length * 4);
    values.forEach((value, index) => dtype === "F32" ? payload.writeFloatLE(value, index * 4) : payload.writeUInt32LE(value, index * 4));
    header[name] = { dtype, shape, data_offsets: [offset, offset + payload.length] };
    offset += payload.length;
    return payload;
  });
  const encoded = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(encoded.length));
  await writeFile(file, Buffer.concat([prefix, encoded, ...payloads]));
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

/** Writes the tiny Llama fixture in a verified dense GGML storage dtype. */
async function writeTiny16GgufModel(file: string, storageDtype: "F16" | "BF16"): Promise<void> {
  const weights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [2, 3], [0, 0, 3, 4, 0, 0]], ["blk.0.attn_norm.weight", [2], [1, 1]],
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
  const ggmlType = storageDtype === "F16" ? 1 : 25;
  const payloads: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, shape, values] of weights) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloads.push(Buffer.alloc(padding)); offset += padding; }
    const payload = dense16Payload(values, storageDtype);
    directory.push(Buffer.concat([text(name), u32(shape.length), ...shape.map(u64), u32(ggmlType), u64(offset)]));
    payloads.push(payload); offset += payload.length;
  }
  const prefix = Buffer.concat([Buffer.from("GGUF"), u32(3), u64(weights.length), u64(metadataEntries.length), ...metadataEntries, ...directory]);
  await writeFile(file, Buffer.concat([prefix, Buffer.alloc((32 - (prefix.length % 32)) % 32), ...payloads]));
}

function dense16Payload(values: readonly number[], storageDtype: "F16" | "BF16"): Buffer {
  const payload = Buffer.alloc(values.length * 2);
  for (const [index, value] of values.entries()) payload.writeUInt16LE(storageDtype === "F16" ? encodeTinyF16(value) : encodeBF16(value), index * 2);
  return payload;
}

function encodeTinyF16(value: number): number {
  const bits = new Map<number, number>([[0, 0x0000], [1, 0x3c00], [3, 0x4200], [4, 0x4400]]).get(value);
  if (bits === undefined) throw new Error(`Fixture F16 não possui codificação exata para ${value}.`);
  return bits;
}

function encodeBF16(value: number): number {
  const bits = new ArrayBuffer(4);
  const view = new DataView(bits);
  view.setFloat32(0, value, true);
  return view.getUint16(2, true);
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

/** Builds a Q5_1 fixture with every affine code, qh bit, and nibble half live. */
async function writeQ5_1TraceFixture(denseFile: string, quantizedFile: string): Promise<void> {
  const width = 256;
  const ones = new Array<number>(width).fill(1);
  const matrixNames = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
  const denseWeights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [width, width], q5_1DenseValues(width * width)], ["blk.0.attn_norm.weight", [width], ones],
    ...matrixNames.map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], q5_1DenseValues(width * width)]),
    ["blk.0.ffn_norm.weight", [width], ones], ["output_norm.weight", [width], ones], ["output.weight", [width, width], q5_1DenseValues(width * width)],
  ];
  await writeGgufLlamaFixture(denseFile, denseWeights.map(([name, shape, values]) => [name, shape, 0, values]));
  await writeGgufLlamaFixture(quantizedFile, denseWeights.map(([name, shape, values]) => [name, shape, values.length === width ? 0 : 7, values]));
}

/** Builds a Q5_0 fixture with every centered code, qh bit, and nibble half live. */
async function writeQ5_0TraceFixture(denseFile: string, quantizedFile: string): Promise<void> {
  const width = 256;
  const ones = new Array<number>(width).fill(1);
  const matrixNames = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
  const denseWeights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [width, width], q5_0DenseValues(width * width)], ["blk.0.attn_norm.weight", [width], ones],
    ...matrixNames.map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], q5_0DenseValues(width * width)]),
    ["blk.0.ffn_norm.weight", [width], ones], ["output_norm.weight", [width], ones], ["output.weight", [width, width], q5_0DenseValues(width * width)],
  ];
  await writeGgufLlamaFixture(denseFile, denseWeights.map(([name, shape, values]) => [name, shape, 0, values]));
  await writeGgufLlamaFixture(quantizedFile, denseWeights.map(([name, shape, values]) => [name, shape, values.length === width ? 0 : 6, values]));
}

/** Builds a Q4_0 fixture with every centered code, nibble half, and F16 scale live. */
async function writeQ4_0TraceFixture(denseFile: string, quantizedFile: string): Promise<void> {
  const width = 256;
  const ones = new Array<number>(width).fill(1);
  const matrixNames = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
  const denseWeights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [width, width], q4_0DenseValues(width * width)], ["blk.0.attn_norm.weight", [width], ones],
    ...matrixNames.map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], q4_0DenseValues(width * width)]),
    ["blk.0.ffn_norm.weight", [width], ones], ["output_norm.weight", [width], ones], ["output.weight", [width, width], q4_0DenseValues(width * width)],
  ];
  await writeGgufLlamaFixture(denseFile, denseWeights.map(([name, shape, values]) => [name, shape, 0, values]));
  await writeGgufLlamaFixture(quantizedFile, denseWeights.map(([name, shape, values]) => [name, shape, values.length === width ? 0 : 2, values]));
}

/** Builds a Q4_1 fixture with every affine code, nibble half, F16 scale, and F16 minimum live. */
async function writeQ4_1TraceFixture(denseFile: string, quantizedFile: string): Promise<void> {
  const width = 256;
  const ones = new Array<number>(width).fill(1);
  const matrixNames = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
  const denseWeights: Array<[string, number[], number[]]> = [
    ["token_embd.weight", [width, width], q4_1DenseValues(width * width)], ["blk.0.attn_norm.weight", [width], ones],
    ...matrixNames.map((projection): [string, number[], number[]] => [`blk.0.${projection}.weight`, [width, width], q4_1DenseValues(width * width)]),
    ["blk.0.ffn_norm.weight", [width], ones], ["output_norm.weight", [width], ones], ["output.weight", [width, width], q4_1DenseValues(width * width)],
  ];
  await writeGgufLlamaFixture(denseFile, denseWeights.map(([name, shape, values]) => [name, shape, 0, values]));
  await writeGgufLlamaFixture(quantizedFile, denseWeights.map(([name, shape, values]) => [name, shape, values.length === width ? 0 : 3, values]));
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
    const payload = ggmlType === 2 ? q4_0Payload(values.length) : ggmlType === 3 ? q4_1Payload(values.length) : ggmlType === 6 ? q5_0Payload(values.length) : ggmlType === 7 ? q5_1Payload(values.length) : ggmlType === 8 ? q8_0Payload(values) : ggmlType === 9 ? q8_1Payload(values.length) : ggmlType === 10 ? q2KPayload(values.length) : ggmlType === 11 ? q3KPayload(values.length) : ggmlType === 12 ? q4KPayload(values.length) : ggmlType === 13 ? q5KPayload(values.length) : ggmlType === 14 ? q6KPayload(values.length) : ggmlType === 15 ? q8KPayload(values.length) : f32Payload(values);
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

function q5_1DenseValues(length: number): number[] {
  assert.equal(length % 32, 0, "Q5_1 fixture requires complete 32-value blocks");
  return Array.from({ length }, (_, index) => {
    const withinBlock = index % 32;
    const code = withinBlock < 16 ? withinBlock : 31 - (withinBlock - 16);
    return 0.5 * code - 1;
  });
}

function q5_0DenseValues(length: number): number[] {
  assert.equal(length % 32, 0, "Q5_0 fixture requires complete 32-value blocks");
  return Array.from({ length }, (_, index) => {
    const withinBlock = index % 32;
    const code = withinBlock < 16 ? withinBlock : 31 - (withinBlock - 16);
    return 0.5 * (code - 16);
  });
}

function q4_0DenseValues(length: number): number[] {
  assert.equal(length % 32, 0, "Q4_0 fixture requires complete 32-value blocks");
  return Array.from({ length }, (_, index) => {
    const withinBlock = index % 32;
    const code = withinBlock < 16 ? withinBlock : 15 - (withinBlock - 16);
    return 0.5 * (code - 8);
  });
}

function q4_1DenseValues(length: number): number[] {
  assert.equal(length % 32, 0, "Q4_1 fixture requires complete 32-value blocks");
  return Array.from({ length }, (_, index) => {
    const withinBlock = index % 32;
    const code = withinBlock < 16 ? withinBlock : 15 - (withinBlock - 16);
    return 0.5 * code - 1;
  });
}

function q4_0Payload(length: number): Buffer {
  assert.equal(length % 32, 0, "Q4_0 fixture requires complete 32-value blocks");
  const payload = Buffer.alloc((length / 32) * 18);
  for (let block = 0; block < length / 32; block += 1) {
    const offset = block * 18;
    payload.writeUInt16LE(0x3800, offset); // d = 0.5
    for (let index = 0; index < 16; index += 1) payload[offset + 2 + index] = index | ((15 - index) << 4);
  }
  return payload;
}

function q4_1Payload(length: number): Buffer {
  assert.equal(length % 32, 0, "Q4_1 fixture requires complete 32-value blocks");
  const payload = Buffer.alloc((length / 32) * 20);
  for (let block = 0; block < length / 32; block += 1) {
    const offset = block * 20;
    payload.writeUInt16LE(0x3800, offset); // d = 0.5
    payload.writeUInt16LE(0xbc00, offset + 2); // m = -1
    for (let index = 0; index < 16; index += 1) {
      const lowCode = index;
      const highCode = 15 - index;
      payload[offset + 4 + index] = lowCode | (highCode << 4);
    }
  }
  return payload;
}

function q5_0Payload(length: number): Buffer {
  assert.equal(length % 32, 0, "Q5_0 fixture requires complete 32-value blocks");
  const payload = Buffer.alloc((length / 32) * 22);
  for (let block = 0; block < length / 32; block += 1) {
    const offset = block * 22;
    payload.writeUInt16LE(0x3800, offset); // d = 0.5
    for (let index = 0; index < 16; index += 1) {
      const lowCode = index;
      const highCode = 31 - index;
      payload[offset + 6 + index] = (lowCode & 0x0f) | ((highCode & 0x0f) << 4);
      payload[offset + 2 + Math.floor(index / 8)]! |= ((lowCode >>> 4) & 1) << (index % 8);
      const highIndex = 16 + index;
      payload[offset + 2 + Math.floor(highIndex / 8)]! |= ((highCode >>> 4) & 1) << (highIndex % 8);
    }
  }
  return payload;
}

function q5_1Payload(length: number): Buffer {
  assert.equal(length % 32, 0, "Q5_1 fixture requires complete 32-value blocks");
  const payload = Buffer.alloc((length / 32) * 24);
  for (let block = 0; block < length / 32; block += 1) {
    const offset = block * 24;
    payload.writeUInt16LE(0x3800, offset); // d = 0.5
    payload.writeUInt16LE(0xbc00, offset + 2); // m = -1
    for (let index = 0; index < 16; index += 1) {
      const lowCode = index;
      const highCode = 31 - index;
      payload[offset + 8 + index] = (lowCode & 0x0f) | ((highCode & 0x0f) << 4);
      payload[offset + 4 + Math.floor(index / 8)]! |= ((lowCode >>> 4) & 1) << (index % 8);
      const highIndex = 16 + index;
      payload[offset + 4 + Math.floor(highIndex / 8)]! |= ((highCode >>> 4) & 1) << (highIndex % 8);
    }
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
