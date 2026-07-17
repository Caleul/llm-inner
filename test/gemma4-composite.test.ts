import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import * as path from "node:path";
import test from "node:test";
import { tmpdir } from "node:os";
import { executeReferenceF32WithPreparedPrelude } from "../src/executor.js";
import { buildGemma4CompositeProgram, executeGemma4CompositeF32, generateGemma4CompositeF32 } from "../src/gemma4-composite.js";
import {
  buildGemma4CompositeLiteralCalculationProgram,
  executeGemma4CompositeLiteralF32,
  generateGemma4CompositeLiteralF32,
  validateGemma4CompositeLiteralCalculationProgram,
  writeGemma4CompositeLiteralCalculationProgram,
} from "../src/gemma4-composite-literal.js";
import { openGemma4CompositeLiteralArtifact } from "../src/gemma4-composite-literal-reader.js";
import {
  verifyGemma4CompositeLiteralEmbeddedPayloadIntegrity,
  verifyGemma4CompositeLiteralPayloadsAgainstCatalog,
} from "../src/gemma4-composite-literal-payload-verification.js";
import { probeGemma4LiteralLinearReductionProfiles } from "../src/gemma4-linear-reduction-probe.js";
import { executeGemma4PagedTextLiteralF32, generateGemma4PagedTextLiteralF32 } from "../src/gemma4-paged-text.js";
import { createPagedDenseF32Matrix, pagedEmbeddingF32, pagedLinearF32 } from "../src/paged-dense.js";
import { fingerprintIR } from "../src/trace.js";
import type { DenseF32Tensor, ModelCatalog, TensorInfo } from "../src/types.js";

const preview = { outputRows: 1, inputTerms: 1, includeWeights: false } as const;

test("Gemma 4 reduction probe CLI rejects unknown and duplicate targeted profile IDs before opening artifacts", () => {
  const fixedArguments = [
    new URL("../src/gemma4-linear-reduction-probe-cli.js", import.meta.url).pathname,
    "--artifact", "/tmp/nonexistent-artifact.json", "--trace", "/tmp/trace-a.json", "--trace", "/tmp/trace-b.json",
    "--operation-id", "layer_0_up_proj", "--output", "/tmp/unused-report.json",
  ];
  const unknown = spawnSync(process.execPath, [...fixedArguments, "--profile-id", "unsupported-profile-id"], { encoding: "utf8" });
  assert.notEqual(unknown.status, 0);
  assert.match(unknown.stderr, /--profile-id não reconhece: unsupported-profile-id/);
  const duplicate = spawnSync(process.execPath, [
    ...fixedArguments,
    "--profile-id", "interleaved-f32-lanes-32-balanced-pairwise",
    "--profile-id", "interleaved-f32-lanes-32-balanced-pairwise",
  ], { encoding: "utf8" });
  assert.notEqual(duplicate.status, 0);
  assert.match(duplicate.stderr, /--profile-id requer IDs não vazios e sem repetição/);
});

test("Gemma 4 composite prelude replaces PAD-backed image/video/audio slots before context PLE and enters text core", () => {
  const catalog = fixture();
  const program = buildGemma4CompositeProgram(catalog, preview);
  assert.equal(program.kind, "gemma4-composite-prelude");
  assert.deepEqual(program.assignments.map((assignment) => assignment.id).slice(0, 8), [
    "composite_placeholder_masks", "composite_block_sequence_ids", "composite_full_attention_mask", "composite_sliding_attention_mask",
    "composite_pad_substitution", "composite_text_embedding", "composite_ple_identity", "composite_image_features",
  ]);
  const ids = [[1, 99, 97, 98, 2]];
  const tensors = materialize(catalog);
  const noFeatures = executeGemma4CompositeF32(program, { inputIds: ids, tensors });
  const result = executeGemma4CompositeF32(program, {
    inputIds: ids,
    tensors,
    pixelValues: patterned([1, 4, 12]),
    imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]],
    pixelValuesVideos: patterned([1, 1, 4, 12]),
    videoPositionIds: [[[[0, 0], [1, 0], [0, 1], [1, 1]]]],
    inputFeatures: patterned([1, 4, 16]),
    inputFeaturesMask: [[true, true, true, true]],
  });
  assert.deepEqual(result.llmInputIds, [[1, 0, 0, 0, 2]]);
  assert.deepEqual(result.text.logits.shape, [1, 5, 6]);
  assert.ok([...result.text.logits.values].every(Number.isFinite));
  assert.equal(result.values.has("image_features"), true);
  assert.equal(result.values.has("video_features"), true);
  assert.equal(result.values.has("audio_features"), true);
  assert.equal(result.values.has("layer_0_ple_input"), true);
  assert.notDeepEqual([...result.values.get("ple_context_packed")!.values], [...noFeatures.values.get("ple_context_packed")!.values], "PLE context must observe the post-scatter embeddings");
});

test("Gemma 4 composite lowers vision blocks into distinct full/sliding masks and rejects ambiguous caller masks", () => {
  const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), tensors = materialize(catalog);
  const result = executeGemma4CompositeF32(program, { inputIds: [[1, 99, 99, 2, 3]], mmTokenTypeIds: [[0, 1, 1, 0, 0]], tensors });
  assert.deepEqual([...result.values.get("vision_block_sequence_ids")!.values], [-1, 0, 0, -1, -1]);
  const full = result.values.get("full_attention_mask")!, sliding = result.values.get("sliding_attention_mask")!;
  assert.equal(full.shape.join(","), "1,1,5,5");
  assert.equal(full.values[1 * 5 + 2], -Infinity, "full attention stays causal across a vision block");
  assert.equal(sliding.values[1 * 5 + 2], 0, "sliding attention permits a future token in the same vision block");
  assert.equal(sliding.values[0 * 5 + 4], -Infinity, "sliding attention rejects unrelated future text");
  assert.ok([...result.text.logits.values].every(Number.isFinite));
  assert.throws(
    () => executeGemma4CompositeF32(program, { inputIds: [[1]], mmTokenTypeIds: [[0]], attentionMask: patterned([1, 1, 1, 1]), tensors }),
    /attentionMask 4-D fornecida pelo chamador/,
  );
  assert.throws(
    () => executeGemma4CompositeF32(program, { inputIds: [[1]], mmTokenTypeIds: [[0]], pastKeyValues: result.text.pastKeyValues, tensors }),
    /somente no prefill sem cache/,
  );
});

test("Gemma 4 composite reuses producer-owned KV after vision-aware prefill without reapplying block masks", () => {
  const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), tensors = materialize(catalog);
  const generated = generateGemma4CompositeF32(program, { inputIds: [[1, 99, 99, 2]], mmTokenTypeIds: [[0, 1, 1, 0]], tensors, maxNewTokens: 2 });
  assert.equal(generated.generatedTokenIds.length, 2);
  assert.equal(generated.stepPastKeyValues.length, 2);
  assert.equal(generated.prefill.text.pastKeyValues.size, 2);
  assert.equal(generated.text.pastKeyValues.size, 2);
  for (const cache of generated.text.pastKeyValues.values()) assert.equal(cache.key.shape[2], 6);
});

test("Gemma 4 composite literal embeds every tower weight and replays multimodal prefill plus cached decode after source bytes are removed", async () => {
  const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), sourceTensors = materialize(catalog);
  const expected = executeGemma4CompositeF32(program, {
    inputIds: [[1, 99, 2]], mmTokenTypeIds: [[0, 1, 0]], tensors: sourceTensors,
    pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]],
  });
  const expectedGeneration = generateGemma4CompositeF32(program, {
    inputIds: [[1, 99, 2]], mmTokenTypeIds: [[0, 1, 0]], tensors: sourceTensors,
    pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]], maxNewTokens: 2,
  });
  const literal = await buildGemma4CompositeLiteralCalculationProgram(program, catalog, {
    async readTensorBytes(info) {
      const tensor = sourceTensors.get(info.name);
      if (!tensor) throw new Error(`source tensor missing: ${info.name}`);
      const bytes = Buffer.alloc(tensor.values.length * 4);
      tensor.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
      return bytes;
    },
  });
  assert.equal(literal.constants.length, catalog.tensors.size);
  assert.equal(literal.storageDecoders.length, catalog.tensors.size);
  assert.equal(JSON.stringify(literal).includes(catalog.source), false);
  assert.equal(literal.program.textProgram.source.path, "embedded://gemma4-composite-literal");

  sourceTensors.clear();
  const replay = executeGemma4CompositeLiteralF32(literal, {
    inputIds: [[1, 99, 2]], mmTokenTypeIds: [[0, 1, 0]],
    pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]],
  });
  const generation = generateGemma4CompositeLiteralF32(literal, {
    inputIds: [[1, 99, 2]], mmTokenTypeIds: [[0, 1, 0]],
    pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]], maxNewTokens: 2,
  });
  assert.deepEqual([...replay.text.logits.values], [...expected.text.logits.values]);
  assert.deepEqual(generation.generatedTokenIds, expectedGeneration.generatedTokenIds);
  assert.deepEqual([...generation.text.logits.values], [...expectedGeneration.text.logits.values]);

  const missingMask = structuredClone(literal);
  missingMask.assignments.composite = missingMask.assignments.composite.filter((assignment) => assignment.id !== "composite_sliding_attention_mask");
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(missingMask), /omite uma transição de máscara ou cache obrigatória/);
  const external = structuredClone(literal);
  external.program.textProgram.source.path = "/checkpoint/model.safetensors";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(external), /reteve uma referência de source checkpoint/);
});

test("Gemma 4 composite streamed writer emits an atomic self-contained JSON file without accumulating source payloads", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-streamed-"));
  try {
    const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), sourceTensors = materialize(catalog);
    const expected = executeGemma4CompositeF32(program, { inputIds: [[1, 99, 2]], mmTokenTypeIds: [[0, 1, 0]], tensors: sourceTensors, pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]] });
    const output = path.join(root, "tiny.gemma4.literal.json");
    let maxRequestedBytes = 0;
    const written = await writeGemma4CompositeLiteralCalculationProgram(program, catalog, {
      async readTensorBytes(info) {
        const tensor = sourceTensors.get(info.name);
        if (!tensor) throw new Error(`source tensor missing: ${info.name}`);
        const bytes = Buffer.alloc(tensor.values.length * 4);
        tensor.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
        maxRequestedBytes = Math.max(maxRequestedBytes, bytes.length);
        return bytes;
      },
    }, output);
    const raw = await readFile(output);
    assert.equal(written.artifactSha256, createHash("sha256").update(raw).digest("hex"));
    assert.equal(written.constants, catalog.tensors.size);
    assert.equal(written.embeddedPayloadBytes, [...catalog.tensors.values()].reduce((total, tensor) => total + tensor.logicalShape.reduce((size, dimension) => size * dimension, 1) * 4, 0));
    assert.ok(maxRequestedBytes < written.embeddedPayloadBytes, "writer must request one payload at a time rather than a package buffer");
    const literal = JSON.parse(raw.toString("utf8"));
    assert.equal(JSON.stringify(literal).includes(catalog.source), false);
    assert.equal(literal.payloadIntegrity.length, catalog.tensors.size);
    const embedded = literal.payloadIntegrity.find((entry: { name: string }) => entry.name === "model.language_model.embed_tokens.weight")!;
    assert.equal(embedded.sha256, createHash("sha256").update(denseF32Bytes(sourceTensors.get(embedded.name)!)).digest("hex"));

    sourceTensors.clear();
    const replay = executeGemma4CompositeLiteralF32(literal, { inputIds: [[1, 99, 2]], mmTokenTypeIds: [[0, 1, 0]], pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]] });
    assert.deepEqual([...replay.text.logits.values], [...expected.text.logits.values]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("Gemma 4 streamed literal artifact indexes exact tensor ranges after its checkpoint path is removed", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-artifact-reader-"));
  try {
    const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), sourceTensors = materialize(catalog);
    const source = path.join(root, "removed-checkpoint.safetensors");
    const output = path.join(root, "tiny.gemma4.literal.json");
    catalog.source = source;
    await writeFile(source, "checkpoint bytes are intentionally unavailable after export");
    await writeGemma4CompositeLiteralCalculationProgram(program, catalog, {
      async readTensorBytes(info) {
        const tensor = sourceTensors.get(info.name);
        if (!tensor) throw new Error(`source tensor missing: ${info.name}`);
        const bytes = Buffer.alloc(tensor.values.length * 4);
        tensor.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
        return bytes;
      },
    }, output);
    const expected = sourceTensors.get("model.language_model.layers.0.self_attn.q_proj.weight")!;
    const expectedBytes = Buffer.alloc(expected.values.length * 4);
    expected.values.forEach((value, index) => expectedBytes.writeFloatLE(value, index * 4));
    sourceTensors.clear();
    await rm(source);

    const artifact = await openGemma4CompositeLiteralArtifact(output);
    try {
      assert.equal(artifact.constants.size, catalog.tensors.size);
      assert.equal(artifact.program.textProgram.source.path, "embedded://gemma4-composite-literal");
      const tensor = catalog.tensors.get("model.language_model.layers.0.self_attn.q_proj.weight")!;
      assert.deepEqual(await artifact.readTensorBytes(tensor), expectedBytes);
      assert.deepEqual(await artifact.readTensorBytesRange(tensor, 5, 23), expectedBytes.subarray(5, 28));
      assert.equal("payloadBase64" in artifact.constants.get(tensor.name)!, false);
      assert.equal(artifact.payloadIntegrity?.size, catalog.tensors.size);
    } finally {
      await artifact.close();
    }
    const corrupted = path.join(root, "corrupted.gemma4.literal.json");
    const raw = await readFile(output, "utf8");
    await writeFile(corrupted, raw.replace('"semantics":"exact IEEE-754 storage decode; no arithmetic narrowing"', '"semantics":"invalid"'));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(corrupted), /decoder denso do artefato literal/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("Gemma 4 literal payload verifier proves every embedded storage byte before source removal", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-payload-verification-"));
  try {
    const source = path.join(root, "source");
    const catalog = fixture();
    const sourceTensors = materialize(catalog);
    await writeFixtureSafetensors(source, catalog, sourceTensors);
    catalog.source = source;
    const program = buildGemma4CompositeProgram(catalog, preview);
    const artifact = path.join(root, "tiny.gemma4.literal.json");
    await writeGemma4CompositeLiteralCalculationProgram(program, catalog, {
      async readTensorBytes(info) { return denseF32Bytes(sourceTensors.get(info.name)!); },
    }, artifact);

    const verified = await verifyGemma4CompositeLiteralPayloadsAgainstCatalog({ artifact, source, maxReadBytes: 13 });
    assert.equal(verified.constants, catalog.tensors.size);
    assert.equal(verified.comparedPayloadBytes, [...sourceTensors.values()].reduce((total, tensor) => total + tensor.values.byteLength, 0));
    assert.equal(verified.sourceStorageSha256, verified.literalStorageSha256);

    const corrupt = path.join(root, "corrupt.gemma4.literal.json");
    const raw = await readFile(artifact, "utf8");
    await writeFile(corrupt, raw.replace(/"payloadBase64":"([A-Za-z0-9])/, (_match, first: string) => `"payloadBase64":"${first === "A" ? "B" : "A"}`), "utf8");
    await assert.rejects(
      () => verifyGemma4CompositeLiteralPayloadsAgainstCatalog({ artifact: corrupt, source, maxReadBytes: 13 }),
      /payload literal diverge do Safetensors/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("Gemma 4 literal artifact verifies its embedded payload commitments after source removal", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-embedded-payload-verification-"));
  try {
    const catalog = fixture(), sourceTensors = materialize(catalog);
    const artifact = path.join(root, "tiny.gemma4.literal.json");
    const unavailableSource = path.join(root, "checkpoint-removed");
    await writeGemma4CompositeLiteralCalculationProgram(buildGemma4CompositeProgram(catalog, preview), catalog, {
      async readTensorBytes(info) { return denseF32Bytes(sourceTensors.get(info.name)!); },
    }, artifact);
    sourceTensors.clear();

    const verified = await verifyGemma4CompositeLiteralEmbeddedPayloadIntegrity({ artifact, maxReadBytes: 13, assertSourceUnavailable: unavailableSource });
    assert.equal(verified.constants, catalog.tensors.size);
    assert.ok(verified.comparedPayloadBytes > 0);
    assert.equal(verified.sourceCheckpointAccessed, false);
    assert.equal(verified.assertedUnavailableSource, unavailableSource);

    await mkdir(unavailableSource);
    await assert.rejects(
      () => verifyGemma4CompositeLiteralEmbeddedPayloadIntegrity({ artifact, assertSourceUnavailable: unavailableSource }),
      /requer source indisponível/,
    );
    await rm(unavailableSource, { recursive: true });

    const corrupt = path.join(root, "corrupt.gemma4.literal.json");
    const raw = await readFile(artifact, "utf8");
    await writeFile(corrupt, raw.replace(/"payloadBase64":"([A-Za-z0-9])/, (_match, first: string) => `"payloadBase64":"${first === "A" ? "B" : "A"}`), "utf8");
    await assert.rejects(
      () => verifyGemma4CompositeLiteralEmbeddedPayloadIntegrity({ artifact: corrupt, maxReadBytes: 13 }),
      /payload literal diverge do digest incorporado/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("Gemma 4 literal headers expose and stream-validate operation-declared F64 reductions", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-f64-policy-"));
  try {
    const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), sourceTensors = materialize(catalog);
    for (const operation of [...program.textProgram.prelude, ...program.textProgram.layers.flatMap((layer) => layer.operations), ...program.textProgram.epilogue]) {
      if (operation.op === "linear" || operation.op === "rms_norm") {
        operation.dtypePolicy = { inputDtype: "BF16", computeDtype: "F32", accumulationDtype: "F64", outputDtype: "BF16", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } };
      }
    }
    const output = path.join(root, "f64.gemma4.literal.json");
    await writeGemma4CompositeLiteralCalculationProgram(program, catalog, {
      async readTensorBytes(info) {
        const tensor = sourceTensors.get(info.name)!;
        const bytes = Buffer.alloc(tensor.values.length * 4);
        tensor.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
        return bytes;
      },
    }, output);
    const artifact = await openGemma4CompositeLiteralArtifact(output);
    try {
      assert.deepEqual(artifact.numericPolicy, {
        inputDtype: "I32/F32/BOOL",
        computeDtype: "F32",
        accumulationDtype: "operation-declared",
        outputDtype: "operation-declared",
        scalarSemantics: "IEEE-754 binary32 products; each operation declares its ordered-scalar, contiguous blocked-term, blocked tiled-lane, or interleaved-lane F32/F64 reduction and F32 or BF16 result cast",
      });
    } finally {
      await artifact.close();
    }

    const corrupted = path.join(root, "f64-header-lie.gemma4.literal.json");
    const raw = await readFile(output, "utf8");
    const legacy = path.join(root, "legacy-f64-header.gemma4.literal.json");
    await writeFile(legacy, raw.replace(
      "IEEE-754 binary32 products; each operation declares its ordered-scalar, contiguous blocked-term, blocked tiled-lane, or interleaved-lane F32/F64 reduction and F32 or BF16 result cast",
      "IEEE-754 binary32 products; each operation declares its ordered-scalar or interleaved-lane F32/F64 reduction and F32 or BF16 result cast",
    ));
    const legacyArtifact = await openGemma4CompositeLiteralArtifact(legacy);
    await legacyArtifact.close();

    const blockedTiledProgram = buildGemma4CompositeProgram(catalog, preview);
    const blockedTiled = [...blockedTiledProgram.textProgram.prelude, ...blockedTiledProgram.textProgram.layers.flatMap((layer) => layer.operations), ...blockedTiledProgram.textProgram.epilogue]
      .find((operation) => operation.op === "linear");
    if (!blockedTiled || blockedTiled.op !== "linear") throw new Error("fixture requires a linear operation");
    blockedTiled.dtypePolicy = {
      inputDtype: "BF16", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "BF16",
      reduction: { kind: "blocked-tiled-f32-lanes", laneCount: 2, termsPerLane: 2, inputBlock: "tile-contiguous-terms", laneReductionOrder: "ascending", productBoundary: "separately-rounded-f32", blockOrder: "ascending" },
    };
    const blockedTiledOutput = path.join(root, "blocked-tiled.gemma4.literal.json");
    await writeGemma4CompositeLiteralCalculationProgram(blockedTiledProgram, catalog, {
      async readTensorBytes(info) {
        const tensor = sourceTensors.get(info.name)!;
        const bytes = Buffer.alloc(tensor.values.length * 4);
        tensor.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
        return bytes;
      },
    }, blockedTiledOutput);
    const staleBlockedTiledHeader = path.join(root, "blocked-tiled-stale-header.gemma4.literal.json");
    const blockedTiledRaw = await readFile(blockedTiledOutput, "utf8");
    await writeFile(staleBlockedTiledHeader, blockedTiledRaw.replace(
      "IEEE-754 binary32 products; each operation declares its ordered-scalar, contiguous blocked-term, blocked tiled-lane, or interleaved-lane F32/F64 reduction and F32 or BF16 result cast",
      "IEEE-754 binary32 products; each operation declares its ordered-scalar, contiguous blocked-term, or interleaved-lane F32/F64 reduction and F32 or BF16 result cast",
    ));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(staleBlockedTiledHeader), /política numérica incompatível com as atribuições declaradas/);

    await writeFile(corrupted, raw.replace(
      '"accumulationDtype":"operation-declared","outputDtype":"operation-declared","scalarSemantics":"IEEE-754 binary32 products; each operation declares its ordered-scalar, contiguous blocked-term, blocked tiled-lane, or interleaved-lane F32/F64 reduction and F32 or BF16 result cast"',
      '"accumulationDtype":"F32","outputDtype":"operation-declared","scalarSemantics":"IEEE-754 binary32 reductions; each operation declares its F32 or BF16 result cast"',
    ));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(corrupted), /política numérica incompatível com as atribuições declaradas/);
    const missingSchedule = path.join(root, "f64-missing-schedule.gemma4.literal.json");
    await writeFile(missingSchedule, raw.replace(
      '"reduction":{"kind":"ordered-scalar","indexOrder":"ascending"}',
      '"reduction":{"kind":"invalid","indexOrder":"ascending"}',
    ));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(missingSchedule), /agenda de (redução|lanes)/);

    const inMemory = await buildGemma4CompositeLiteralCalculationProgram(program, catalog, {
      async readTensorBytes(info) {
        const tensor = sourceTensors.get(info.name)!;
        const bytes = Buffer.alloc(tensor.values.length * 4);
        tensor.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
        return bytes;
      },
    });
    assert.throws(
      () => executeGemma4CompositeLiteralF32(inMemory, { inputIds: [[1]] }),
      /não pode apagar a política numérica declarada/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("Gemma 4 literal artifact supplies bounded BF16/F32-compatible embedding and linear kernels after source removal", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-paged-"));
  try {
    const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), sourceTensors = materialize(catalog);
    const output = path.join(root, "tiny.gemma4.literal.json");
    await writeGemma4CompositeLiteralCalculationProgram(program, catalog, {
      async readTensorBytes(info) {
        const source = sourceTensors.get(info.name)!;
        const bytes = Buffer.alloc(source.values.length * 4);
        source.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
        return bytes;
      },
    }, output);
    sourceTensors.clear();
    const artifact = await openGemma4CompositeLiteralArtifact(output);
    try {
      const embeddingInfo = artifact.constants.get("model.language_model.embed_tokens.weight")!;
      const projectionInfo = artifact.constants.get("model.language_model.layers.0.self_attn.q_proj.weight")!;
      const embedding = createPagedDenseF32Matrix({ name: embeddingInfo.name, storageDtype: embeddingInfo.storageDtype, storageShape: embeddingInfo.storageShape, logicalShape: embeddingInfo.logicalShape }, artifact, 16);
      const projection = createPagedDenseF32Matrix({ name: projectionInfo.name, storageDtype: projectionInfo.storageDtype, storageShape: projectionInfo.storageShape, logicalShape: projectionInfo.logicalShape }, artifact, 16);
      const embedded = await pagedEmbeddingF32([[1, 2, 1]], embedding, 2);
      assert.deepEqual(embedded.shape, [1, 3, 4]);
      assert.deepEqual(embedded.values, Float32Array.from([0.2, 0.04, 0.08, 0.12, 0.16, 0.2, 0.04, 0.08, 0.2, 0.04, 0.08, 0.12]));
      const linear = await pagedLinearF32({ shape: [1, 2, 4], values: Float32Array.from([0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1]) }, projection);
      assert.deepEqual(linear.shape, [1, 2, 4]);
      assert.deepEqual(linear.values, Float32Array.from([
        0.06000000238418579, 0.05000000074505806, 0.05000000447034836, 0.06000000238418579,
        0.03999999910593033, 0.06000000610947609, 0.07000000029802322, 0.07000000029802322,
      ]));
      await assert.rejects(() => projection.readRows(0, 2), /excede maxReadBytes/);
    } finally {
      await artifact.close();
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("Gemma 4 paged text interpreter replays prefill and cached greedy decode from literal ranges after source removal", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-paged-text-"));
  try {
    const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), sourceTensors = materialize(catalog);
    const expected = executeGemma4CompositeF32(program, { inputIds: [[1, 2, 3]], tensors: sourceTensors });
    const expectedGeneration = generateGemma4CompositeF32(program, { inputIds: [[1, 2, 3]], tensors: sourceTensors, maxNewTokens: 2 });
    const output = path.join(root, "tiny.gemma4.literal.json");
    await writeGemma4CompositeLiteralCalculationProgram(program, catalog, {
      async readTensorBytes(info) {
        const tensor = sourceTensors.get(info.name)!;
        const bytes = Buffer.alloc(tensor.values.length * 4);
        tensor.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
        return bytes;
      },
    }, output);
    sourceTensors.clear();
    const artifact = await openGemma4CompositeLiteralArtifact(output);
    try {
      await assert.rejects(
        () => executeGemma4PagedTextLiteralF32(artifact, { inputIds: [[1, 2, 3]] }, { maxReadBytes: 64 }),
        /fidelidade numérica não verificada/,
      );
      const replay = await executeGemma4PagedTextLiteralF32(artifact, { inputIds: [[1, 2, 3]] }, { maxReadBytes: 64, allowUnverifiedFidelity: true });
      const generation = await generateGemma4PagedTextLiteralF32(artifact, { inputIds: [[1, 2, 3]], maxNewTokens: 2 }, { maxReadBytes: 64, allowUnverifiedFidelity: true });
      assert.deepEqual(replay.logits.values, expected.text.logits.values);
      assert.deepEqual(generation.generatedTokenIds, expectedGeneration.generatedTokenIds);
      assert.deepEqual(generation.logits.values, expectedGeneration.text.logits.values);
      assert.deepEqual([...generation.pastKeyValues.keys()], [...expectedGeneration.text.pastKeyValues.keys()]);
      artifact.program.textProgram.fidelity.exactByConstruction = true;
      const exactReplay = await executeGemma4PagedTextLiteralF32(artifact, { inputIds: [[1, 2, 3]] }, { maxReadBytes: 64 });
      assert.deepEqual(exactReplay.logits.values, expected.text.logits.values);
    } finally {
      await artifact.close();
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("paged dense kernels widen BF16 only from declared literal ranges", async () => {
  const tensor: TensorInfo = { name: "embedded://bf16-matrix", storageDtype: "BF16", storageShape: [2, 2], logicalShape: [2, 2] };
  const storage = Buffer.from([0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40]); // [[1, 2], [3, 4]] BF16 LE
  let maxRead = 0;
  const matrix = createPagedDenseF32Matrix(tensor, {
    async readTensorBytesRange(_tensor, offset, byteLength) {
      maxRead = Math.max(maxRead, byteLength);
      return storage.subarray(offset, offset + byteLength);
    },
  }, 4);
  const embedded = await pagedEmbeddingF32([[1, 0]], matrix);
  assert.deepEqual(embedded.values, Float32Array.from([3, 4, 1, 2]));
  const bf16Output = await pagedEmbeddingF32([[0]], matrix, Math.fround(1.003), { roundOutputToBf16: true });
  assert.deepEqual(bf16Output.values, Float32Array.from([1, 2]), "Gemma BF16 embeddings cast the scaled output back to BF16");
  const output = await pagedLinearF32({ shape: [1, 2], values: Float32Array.from([1, 2]) }, matrix);
  assert.deepEqual(output.values, Float32Array.from([5, 11]));
  const narrowedLinear = await pagedLinearF32({ shape: [1, 2], values: Float32Array.from([1, Math.fround(1.003)]) }, matrix, { outputDtype: "BF16" });
  assert.deepEqual(narrowedLinear.values, Float32Array.from([3, 7]), "a declared BF16 linear result must narrow only after the F32 reduction");
  assert.equal(maxRead, 4, "the kernel must never request more than the declared row budget");
});

test("paged linear keeps the declared F64 accumulator distinct from F32 products and BF16 storage", async () => {
  const tensor: TensorInfo = { name: "embedded://cancellation", storageDtype: "F32", storageShape: [1, 3], logicalShape: [1, 3] };
  const storage = Buffer.alloc(12);
  storage.writeFloatLE(16_777_216, 0);
  storage.writeFloatLE(1, 4);
  storage.writeFloatLE(-16_777_216, 8);
  const matrix = createPagedDenseF32Matrix(tensor, {
    async readTensorBytesRange(_tensor, offset, byteLength) { return storage.subarray(offset, offset + byteLength); },
  }, 12);
  const input = { shape: [1, 3], values: Float32Array.from([1, 1, 1]) };
  const f32 = await pagedLinearF32(input, matrix, { outputDtype: "BF16", accumulationDtype: "F32" });
  const f64 = await pagedLinearF32(input, matrix, { outputDtype: "BF16", accumulationDtype: "F64" });
  assert.deepEqual(f32.values, Float32Array.from([0]));
  assert.deepEqual(f64.values, Float32Array.from([1]));
});

test("paged linear applies an explicit interleaved F32 lane schedule instead of selecting one from shape", async () => {
  const tensor: TensorInfo = { name: "embedded://lanes", storageDtype: "F32", storageShape: [1, 4], logicalShape: [1, 4] };
  const storage = Buffer.alloc(16);
  [16_777_216, 1, -16_777_216, 1].forEach((value, index) => storage.writeFloatLE(value, index * 4));
  const matrix = createPagedDenseF32Matrix(tensor, { async readTensorBytesRange(_tensor, offset, byteLength) { return storage.subarray(offset, offset + byteLength); } }, 16);
  const input = { shape: [1, 4], values: Float32Array.from([1, 1, 1, 1]) };
  const scalar = await pagedLinearF32(input, matrix, { accumulationDtype: "F32" });
  const lanes = await pagedLinearF32(input, matrix, {
    accumulationDtype: "F32",
    reduction: { kind: "interleaved-f32-lanes", laneCount: 2, inputLane: "index-modulo-lane-count", laneReductionOrder: "ascending" },
  });
  assert.deepEqual(scalar.values, Float32Array.from([1]));
  assert.deepEqual(lanes.values, Float32Array.from([2]));
});

test("paged linear preserves the declared horizontal lane fold instead of assuming ascending accumulation", async () => {
  const tensor: TensorInfo = { name: "embedded://lane-fold", storageDtype: "F32", storageShape: [1, 4], logicalShape: [1, 4] };
  const storage = Buffer.alloc(16);
  [1e20, 1, -1e20, 1].forEach((value, index) => storage.writeFloatLE(value, index * 4));
  const matrix = createPagedDenseF32Matrix(tensor, { async readTensorBytesRange(_tensor, offset, byteLength) { return storage.subarray(offset, offset + byteLength); } }, 16);
  const input = { shape: [1, 4], values: Float32Array.from([1, 1, 1, 1]) };
  const ascending = await pagedLinearF32(input, matrix, { accumulationDtype: "F32", reduction: { kind: "interleaved-f32-lanes", laneCount: 4, inputLane: "index-modulo-lane-count", laneReductionOrder: "ascending" } });
  const balanced = await pagedLinearF32(input, matrix, { accumulationDtype: "F32", reduction: { kind: "interleaved-f32-lanes", laneCount: 4, inputLane: "index-modulo-lane-count", laneReductionOrder: "balanced-pairwise" } });
  assert.deepEqual(ascending.values, Float32Array.from([1]));
  assert.deepEqual(balanced.values, Float32Array.from([0]));
});

test("paged linear replays the ARM BF16 dot register tree rather than a generic 32-lane fold", async () => {
  const tensor: TensorInfo = { name: "embedded://arm-neon-bf16-dot", storageDtype: "F32", storageShape: [1, 32], logicalShape: [1, 32] };
  const storage = Buffer.alloc(32 * 4);
  const values = new Float32Array(32);
  values[0] = 1e20; values[16] = -1e20; values[8] = 1; values[24] = 1;
  // Lane 4 belongs to the second surviving register vector. It proves the
  // final vector-vector merge occurs before the four-lane horizontal fold.
  values[4] = 7;
  values.forEach((value, index) => storage.writeFloatLE(value, index * 4));
  const matrix = createPagedDenseF32Matrix(tensor, { async readTensorBytesRange(_tensor, offset, byteLength) { return storage.subarray(offset, offset + byteLength); } }, storage.length);
  const input = { shape: [1, 32], values: Float32Array.from({ length: 32 }, () => 1) };
  const armTree = await pagedLinearF32(input, matrix, {
    accumulationDtype: "F32",
    reduction: { kind: "arm-neon-bf16-dot-fma", laneCount: 32, registerCount: 8, lanesPerRegister: 4, inputLane: "index-modulo-vector-lane-count", horizontalFold: "pairwise" },
  });
  const generic = await pagedLinearF32(input, matrix, {
    accumulationDtype: "F32",
    reduction: { kind: "interleaved-fma-lanes", laneCount: 32, inputLane: "index-modulo-lane-count", laneReductionOrder: "balanced-pairwise" },
  });
  assert.deepEqual(armTree.values, Float32Array.from([9]));
  assert.deepEqual(generic.values, Float32Array.from([0]));
});

test("paged linear distinguishes a tiled adjacent-product lane schedule from index-modulo lanes", async () => {
  const tensor: TensorInfo = { name: "embedded://tiled-lanes", storageDtype: "F32", storageShape: [1, 4], logicalShape: [1, 4] };
  const storage = Buffer.alloc(16);
  [1e20, 1, -1e20, 1].forEach((value, index) => storage.writeFloatLE(value, index * 4));
  const matrix = createPagedDenseF32Matrix(tensor, { async readTensorBytesRange(_tensor, offset, byteLength) { return storage.subarray(offset, offset + byteLength); } }, 16);
  const input = { shape: [1, 4], values: Float32Array.from([1, 1, 1, 1]) };
  const modulo = await pagedLinearF32(input, matrix, {
    accumulationDtype: "F32",
    reduction: { kind: "interleaved-f32-lanes", laneCount: 2, inputLane: "index-modulo-lane-count", laneReductionOrder: "ascending" },
  });
  const tiled = await pagedLinearF32(input, matrix, {
    accumulationDtype: "F32",
    reduction: { kind: "tiled-f32-lanes", laneCount: 2, termsPerLane: 2, inputLane: "tile-contiguous-terms", laneReductionOrder: "ascending" },
  });
  assert.deepEqual(modulo.values, Float32Array.from([2]));
  assert.deepEqual(tiled.values, Float32Array.from([0]));
});

test("paged linear keeps fused multiply-add lanes distinct from separately rounded products", async () => {
  const tensor: TensorInfo = { name: "embedded://fma", storageDtype: "F32", storageShape: [1, 4], logicalShape: [1, 4] };
  const storage = Buffer.alloc(16);
  [-0.05253555625677109, 0, -0.771535336971283, 0].forEach((value, index) => storage.writeFloatLE(value, index * 4));
  const matrix = createPagedDenseF32Matrix(tensor, { async readTensorBytesRange(_tensor, offset, byteLength) { return storage.subarray(offset, offset + byteLength); } }, 16);
  const input = { shape: [1, 4], values: Float32Array.from([-0.000940456404350698, 0, 13.010579109191895, 0]) };
  const separatelyRounded = await pagedLinearF32(input, matrix, { accumulationDtype: "F32", reduction: { kind: "interleaved-f32-lanes", laneCount: 2, inputLane: "index-modulo-lane-count", laneReductionOrder: "ascending" } });
  const fused = await pagedLinearF32(input, matrix, { accumulationDtype: "F32", reduction: { kind: "interleaved-fma-lanes", laneCount: 2, inputLane: "index-modulo-lane-count", laneReductionOrder: "ascending" } });
  assert.notEqual(separatelyRounded.values[0], fused.values[0]);
});

test("paged linear preserves ordered FMA and adjacent dot-product block boundaries", async () => {
  const fmaTensor: TensorInfo = { name: "embedded://ordered-fma", storageDtype: "F32", storageShape: [1, 4], logicalShape: [1, 4] };
  const fmaStorage = Buffer.alloc(16);
  [-0.05253555625677109, 0, -0.771535336971283, 0].forEach((value, index) => fmaStorage.writeFloatLE(value, index * 4));
  const fmaMatrix = createPagedDenseF32Matrix(fmaTensor, { async readTensorBytesRange(_tensor, offset, byteLength) { return fmaStorage.subarray(offset, offset + byteLength); } }, 16);
  const fmaInput = { shape: [1, 4], values: Float32Array.from([-0.000940456404350698, 0, 13.010579109191895, 0]) };
  const separate = await pagedLinearF32(fmaInput, fmaMatrix, { accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } });
  const orderedFma = await pagedLinearF32(fmaInput, fmaMatrix, { accumulationDtype: "F32", reduction: { kind: "ordered-fma", indexOrder: "ascending" } });
  assert.notEqual(separate.values[0], orderedFma.values[0]);

  const blockTensor: TensorInfo = { name: "embedded://blocked-terms", storageDtype: "F32", storageShape: [1, 4], logicalShape: [1, 4] };
  const blockStorage = Buffer.alloc(16);
  [1e20, 1, -1e20, 1].forEach((value, index) => blockStorage.writeFloatLE(value, index * 4));
  const blockMatrix = createPagedDenseF32Matrix(blockTensor, { async readTensorBytesRange(_tensor, offset, byteLength) { return blockStorage.subarray(offset, offset + byteLength); } }, 16);
  const blockInput = { shape: [1, 4], values: Float32Array.from([1, 1, 1, 1]) };
  const scalar = await pagedLinearF32(blockInput, blockMatrix, { accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } });
  const pairBlocks = await pagedLinearF32(blockInput, blockMatrix, {
    accumulationDtype: "F32",
    reduction: { kind: "blocked-f32-terms", termsPerBlock: 2, inputBlock: "contiguous-terms", termOrder: "ascending", productBoundary: "separately-rounded-f32", blockOrder: "ascending" },
  });
  assert.deepEqual(scalar.values, Float32Array.from([1]));
  assert.deepEqual(pairBlocks.values, Float32Array.from([0]));
});

test("paged linear keeps finite tiled blocks distinct from persistent tiled lanes", async () => {
  const tensor: TensorInfo = { name: "embedded://blocked-tiled", storageDtype: "F32", storageShape: [1, 8], logicalShape: [1, 8] };
  const storage = Buffer.alloc(32);
  [1e20, 0, 1, 0, -1e20, 0, 1, 0].forEach((value, index) => storage.writeFloatLE(value, index * 4));
  const matrix = createPagedDenseF32Matrix(tensor, { async readTensorBytesRange(_tensor, offset, byteLength) { return storage.subarray(offset, offset + byteLength); } }, 32);
  const input = { shape: [1, 8], values: Float32Array.from([1, 1, 1, 1, 1, 1, 1, 1]) };
  const persistent = await pagedLinearF32(input, matrix, {
    accumulationDtype: "F32",
    reduction: { kind: "tiled-f32-lanes", laneCount: 2, termsPerLane: 2, inputLane: "tile-contiguous-terms", laneReductionOrder: "ascending" },
  });
  const blocked = await pagedLinearF32(input, matrix, {
    accumulationDtype: "F32",
    reduction: { kind: "blocked-tiled-f32-lanes", laneCount: 2, termsPerLane: 2, inputBlock: "tile-contiguous-terms", laneReductionOrder: "ascending", productBoundary: "separately-rounded-f32", blockOrder: "ascending" },
  });
  assert.deepEqual(persistent.values, Float32Array.from([2]));
  assert.deepEqual(blocked.values, Float32Array.from([0]));
});

test("Gemma 4 linear reduction probe binds a candidate schedule to traced producer and output tensors", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-gemma4-reduction-probe-"));
  try {
    const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), tensors = materialize(catalog);
    const reader = {
      async readTensorBytes(info: TensorInfo) {
        const tensor = tensors.get(info.name)!;
        const bytes = Buffer.alloc(tensor.values.length * 4);
        tensor.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
        return bytes;
      },
    };
    const artifact = path.join(root, "fixture.literal.json");
    await writeGemma4CompositeLiteralCalculationProgram(program, catalog, reader, artifact);
    const native = executeGemma4CompositeF32(program, { inputIds: [[1]], tensors }).text;
    const target = program.textProgram.layers[0]!.operations.find((operation) => operation.id === "layer_0_gate_proj")!;
    if (target.op !== "linear") throw new Error("fixture gate must be linear");
    const producer = [...program.textProgram.prelude, ...program.textProgram.layers.flatMap((layer) => layer.operations), ...program.textProgram.epilogue].find((operation) => operation.output === target.input)!;
    const serialize = (tensor: DenseF32Tensor) => ({ dtype: "F32", shape: tensor.shape, valuesBase64: Buffer.from(tensor.values.buffer, tensor.values.byteOffset, tensor.values.byteLength).toString("base64") });
    const trace = path.join(root, "trace.json"), repeatedTrace = path.join(root, "trace-repeat.json");
    const tracePayload = {
      schemaVersion: 1, kind: "execution", captureId: "fixture-capture-a", source: { files: [{ path: "config.json", sha256: "a".repeat(64) }] },
      irFingerprint: fingerprintIR(program.textProgram), candidatePolicy: { dtype: "F32", runtime: "llm-inner paged Gemma4Text literal F32" },
      reference: {
        runtime: "fixture", executionDevice: "cpu", executionDeviceDetail: "cpu", model: "fixture", revisionOrChecksum: "fixture", containerFormat: "safetensors", quantization: "none", inputTokens: [[1]], dtypePolicy: "fixture F32",
        nativeKernelEnvironment: { torchBuildConfigSha256: "b".repeat(64), intraopThreads: 1, interopThreads: 1, deterministicAlgorithms: true, mkldnnAvailable: true, mkldnnEnabled: true },
        operations: [
          { operationId: producer.id, output: producer.output, tensor: serialize(native.values.get(producer.output)!) },
          { operationId: target.id, output: target.output, tensor: serialize(native.values.get(target.output)!) },
        ],
        operationDtypes: [{ operationId: target.id, inputDtype: "float32", outputDtype: "float32", parameterDtype: "float32" }],
        operationLayouts: [{
          operationId: target.id,
          input: { shape: [...native.values.get(producer.output)!.shape], strides: [target.inFeatures, target.inFeatures, 1], storageOffset: 0, isContiguous: true },
          output: { shape: [...native.values.get(target.output)!.shape], strides: [target.outFeatures, target.outFeatures, 1], storageOffset: 0, isContiguous: true },
          parameter: { shape: [target.outFeatures, target.inFeatures], strides: [target.inFeatures, 1], storageOffset: 0, isContiguous: true },
        }],
        pastKeyValues: [],
      },
    };
    await writeFile(trace, JSON.stringify(tracePayload), "utf8");
    const repeatedPayload = structuredClone(tracePayload);
    repeatedPayload.captureId = "fixture-capture-b";
    await writeFile(repeatedTrace, JSON.stringify(repeatedPayload), "utf8");
    const report = await probeGemma4LiteralLinearReductionProfiles({
      artifact, traces: [trace, repeatedTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
      profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }],
    });
    assert.equal(report.inputOperationId, producer.id);
    assert.equal(report.traceCount, 2);
    assert.deepEqual(report.reference.nativeKernelEnvironment, tracePayload.reference.nativeKernelEnvironment);
    assert.deepEqual(report.reference.nativeOperationLayout, tracePayload.reference.operationLayouts[0]);
    assert.deepEqual(report.exactProfileIds, ["ordered-f32"]);
    assert.equal(report.profiles[0]!.mismatchedElements, 0);
    const secondInput = structuredClone(tracePayload);
    secondInput.captureId = "fixture-capture-c";
    secondInput.reference.inputTokens = [[2]];
    const secondInputRepeat = structuredClone(secondInput);
    secondInputRepeat.captureId = "fixture-capture-d";
    const secondTrace = path.join(root, "trace-input-2.json"), secondTraceRepeat = path.join(root, "trace-input-2-repeat.json");
    await writeFile(secondTrace, JSON.stringify(secondInput), "utf8");
    await writeFile(secondTraceRepeat, JSON.stringify(secondInputRepeat), "utf8");
    const campaign = await probeGemma4LiteralLinearReductionProfiles({
      artifact, traces: [trace, repeatedTrace, secondTrace, secondTraceRepeat], operationId: target.id, maxReadBytes: 1024 * 1024, minDistinctInputs: 2,
      profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }],
    });
    assert.equal(campaign.inputGroups.length, 2);
    assert.deepEqual(campaign.inputGroups.map((group) => group.traces.length), [2, 2]);
    assert.deepEqual(campaign.exactProfileIds, ["ordered-f32"]);
    const fixtureOutputFeatures = native.values.get(target.output)!.shape.at(-1)!;
    assert.deepEqual(campaign.outputFeatureCoverage, {
      inputGroupCount: 2,
      outputFeatures: fixtureOutputFeatures,
      spans: [{ start: 0, endExclusive: fixtureOutputFeatures, profileIds: ["ordered-f32"] }],
      uncoveredOutputFeatures: [],
    });
    const incompatibleFeatureInput = structuredClone(secondInput);
    const incompatibleFeatureTensor = incompatibleFeatureInput.reference.operations.find((entry) => entry.operationId === target.id)!.tensor;
    const incompatibleFeatureValues = Float32Array.from(native.values.get(target.output)!.values);
    incompatibleFeatureValues[0] = Math.fround(incompatibleFeatureValues[0]! + 1);
    incompatibleFeatureTensor.valuesBase64 = serialize({ shape: [...native.values.get(target.output)!.shape], values: incompatibleFeatureValues }).valuesBase64;
    const incompatibleFeatureRepeat = structuredClone(incompatibleFeatureInput);
    incompatibleFeatureRepeat.captureId = "fixture-capture-e";
    const incompatibleFeatureTrace = path.join(root, "trace-input-2-output-mismatch.json"), incompatibleFeatureRepeatTrace = path.join(root, "trace-input-2-output-mismatch-repeat.json");
    await writeFile(incompatibleFeatureTrace, JSON.stringify(incompatibleFeatureInput), "utf8");
    await writeFile(incompatibleFeatureRepeatTrace, JSON.stringify(incompatibleFeatureRepeat), "utf8");
    const featureCoverage = await probeGemma4LiteralLinearReductionProfiles({
      artifact, traces: [trace, repeatedTrace, incompatibleFeatureTrace, incompatibleFeatureRepeatTrace], operationId: target.id, maxReadBytes: 1024 * 1024, minDistinctInputs: 2,
      profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }],
    });
    assert.deepEqual(featureCoverage.outputFeatureCoverage, {
      inputGroupCount: 2,
      outputFeatures: fixtureOutputFeatures,
      spans: [
        { start: 0, endExclusive: 1, profileIds: [] },
        { start: 1, endExclusive: fixtureOutputFeatures, profileIds: ["ordered-f32"] },
      ],
      uncoveredOutputFeatures: [0],
    });
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, repeatedTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "invalid-tiled", accumulationDtype: "F32", reduction: { kind: "tiled-f32-lanes", laneCount: 2, termsPerLane: 1, inputLane: "tile-contiguous-terms", laneReductionOrder: "ascending" } }], }),
      /mapeamento de lanes F32 inválido/,
    );
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, repeatedTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "invalid-blocked", accumulationDtype: "F32", reduction: { kind: "blocked-f32-terms", termsPerBlock: 1, inputBlock: "contiguous-terms", termOrder: "ascending", productBoundary: "fused-fma", blockOrder: "ascending" } }], }),
      /perfil de blocos F32 inválido/,
    );
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, repeatedTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "invalid-blocked-tiled", accumulationDtype: "F32", reduction: { kind: "blocked-tiled-f32-lanes", laneCount: 2, termsPerLane: 1, inputBlock: "tile-contiguous-terms", laneReductionOrder: "ascending", productBoundary: "fused-fma", blockOrder: "ascending" } }], }),
      /perfil de blocos tiled F32 inválido/,
    );
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, repeatedTrace], operationId: target.id, maxReadBytes: 1024 * 1024, minDistinctInputs: 2,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /2 entradas declaradas distintas/,
    );
    const mpsRepeat = structuredClone(repeatedPayload);
    mpsRepeat.captureId = "fixture-capture-mps";
    mpsRepeat.reference.executionDevice = "mps";
    const mpsRepeatTrace = path.join(root, "trace-mps-repeat.json");
    await writeFile(mpsRepeatTrace, JSON.stringify(mpsRepeat), "utf8");
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, mpsRepeatTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /contrato de referência diferente/,
    );
    const differentDeviceDetail = structuredClone(repeatedPayload);
    differentDeviceDetail.captureId = "fixture-capture-cpu-detail-drift";
    differentDeviceDetail.reference.executionDeviceDetail = "cpu:0";
    const differentDeviceDetailTrace = path.join(root, "trace-cpu-detail-drift.json");
    await writeFile(differentDeviceDetailTrace, JSON.stringify(differentDeviceDetail), "utf8");
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, differentDeviceDetailTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /contrato de referência diferente/,
    );
    const missingDevice = structuredClone(repeatedPayload);
    missingDevice.captureId = "fixture-capture-no-device";
    Object.defineProperty(missingDevice.reference, "executionDevice", { value: undefined, enumerable: true });
    const missingDeviceTrace = path.join(root, "trace-no-device.json");
    await writeFile(missingDeviceTrace, JSON.stringify(missingDevice), "utf8");
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, missingDeviceTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /executionDevice explícito cpu ou mps/,
    );
    const missingDtypes = structuredClone(repeatedPayload);
    missingDtypes.captureId = "fixture-capture-no-dtypes";
    Object.defineProperty(missingDtypes.reference, "operationDtypes", { value: undefined, enumerable: true });
    const missingDtypesTrace = path.join(root, "trace-no-dtypes.json");
    await writeFile(missingDtypesTrace, JSON.stringify(missingDtypes), "utf8");
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, missingDtypesTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /requer dtypes nativos/,
    );
    const missingLayouts = structuredClone(repeatedPayload);
    missingLayouts.captureId = "fixture-capture-no-layouts";
    Object.defineProperty(missingLayouts.reference, "operationLayouts", { value: undefined, enumerable: true });
    const missingLayoutsTrace = path.join(root, "trace-no-layouts.json");
    await writeFile(missingLayoutsTrace, JSON.stringify(missingLayouts), "utf8");
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, missingLayoutsTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /requer layouts nativos/,
    );
    const stridedParameter = structuredClone(repeatedPayload);
    stridedParameter.captureId = "fixture-capture-strided-parameter";
    stridedParameter.reference.operationLayouts![0]!.parameter!.strides = [1, target.outFeatures];
    const stridedParameterTrace = path.join(root, "trace-strided-parameter.json");
    await writeFile(stridedParameterTrace, JSON.stringify(stridedParameter), "utf8");
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, stridedParameterTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /layout row-major contíguo/,
    );
    const missingKernelEnvironment = structuredClone(repeatedPayload);
    missingKernelEnvironment.captureId = "fixture-capture-no-kernel-environment";
    Object.defineProperty(missingKernelEnvironment.reference, "nativeKernelEnvironment", { value: undefined, enumerable: true });
    const missingKernelEnvironmentTrace = path.join(root, "trace-no-kernel-environment.json");
    await writeFile(missingKernelEnvironmentTrace, JSON.stringify(missingKernelEnvironment), "utf8");
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, missingKernelEnvironmentTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /requer nativeKernelEnvironment explícito/,
    );
    const malformedKernelEnvironment = structuredClone(repeatedPayload);
    malformedKernelEnvironment.captureId = "fixture-capture-malformed-kernel-environment";
    malformedKernelEnvironment.reference.nativeKernelEnvironment.intraopThreads = 0;
    const malformedKernelEnvironmentTrace = path.join(root, "trace-malformed-kernel-environment.json");
    await writeFile(malformedKernelEnvironmentTrace, JSON.stringify(malformedKernelEnvironment), "utf8");
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, malformedKernelEnvironmentTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /requer intraopThreads inteiro positivo seguro/,
    );
    const differentKernelEnvironment = structuredClone(repeatedPayload);
    differentKernelEnvironment.captureId = "fixture-capture-kernel-environment-drift";
    differentKernelEnvironment.reference.nativeKernelEnvironment.intraopThreads = 2;
    const differentKernelEnvironmentTrace = path.join(root, "trace-kernel-environment-drift.json");
    await writeFile(differentKernelEnvironmentTrace, JSON.stringify(differentKernelEnvironment), "utf8");
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, differentKernelEnvironmentTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /contrato de referência diferente/,
    );
    const promotedDtypes = structuredClone(repeatedPayload);
    promotedDtypes.captureId = "fixture-capture-promoted-dtypes";
    promotedDtypes.reference.operationDtypes![0]!.outputDtype = "bfloat16";
    const promotedDtypesTrace = path.join(root, "trace-promoted-dtypes.json");
    await writeFile(promotedDtypesTrace, JSON.stringify(promotedDtypes), "utf8");
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, promotedDtypesTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /divergem do contrato literal/,
    );
    const divergent = structuredClone(repeatedPayload);
    const targetTrace = divergent.reference.operations.find((entry) => entry.operationId === target.id)!;
    const divergentValues = Float32Array.from(native.values.get(target.output)!.values);
    divergentValues[0] = Math.fround(divergentValues[0]! + 1);
    targetTrace.tensor.valuesBase64 = serialize({ shape: [...native.values.get(target.output)!.shape], values: divergentValues }).valuesBase64;
    await writeFile(repeatedTrace, JSON.stringify(divergent), "utf8");
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, repeatedTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /resultado nativo não repetível/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("Gemma 4 shared-KV consumers embed but explicitly label their checkpoint-local K/V tensors", async () => {
  const catalog = fixture({ layers: 3, layerTypes: ["sliding_attention", "full_attention", "sliding_attention"], sharedKeyValueLayers: 1 });
  const program = buildGemma4CompositeProgram(catalog, preview), sourceTensors = materialize(catalog);
  const literal = await buildGemma4CompositeLiteralCalculationProgram(program, catalog, {
    async readTensorBytes(info) {
      const tensor = sourceTensors.get(info.name);
      if (!tensor) throw new Error(`source tensor missing: ${info.name}`);
      const bytes = Buffer.alloc(tensor.values.length * 4);
      tensor.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
      return bytes;
    },
  });
  assert.equal(literal.constants.length, catalog.tensors.size);
  assert.deepEqual(literal.unreachableConstants, [
    { name: "model.language_model.layers.2.self_attn.k_norm.weight", reason: "shared-kv-consumer-local-kv-is-runtime-unreachable", producerLayer: 0 },
    { name: "model.language_model.layers.2.self_attn.k_proj.weight", reason: "shared-kv-consumer-local-kv-is-runtime-unreachable", producerLayer: 0 },
    { name: "model.language_model.layers.2.self_attn.v_proj.weight", reason: "shared-kv-consumer-local-kv-is-runtime-unreachable", producerLayer: 0 },
  ]);
  validateGemma4CompositeLiteralCalculationProgram(literal);
});

test("Gemma 4 composite fails closed for cardinality, partial modality inputs, and malformed vision blocks", () => {
  const catalog = fixture(), program = buildGemma4CompositeProgram(catalog, preview), tensors = materialize(catalog);
  assert.throws(
    () => executeGemma4CompositeF32(program, { inputIds: [[1, 99, 99, 2]], tensors, pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]] }),
    /placeholder count=2/,
  );
  assert.throws(
    () => executeGemma4CompositeF32(program, { inputIds: [[1]], tensors, inputFeatures: patterned([1, 4, 16]) }),
    /input_features e input_features_mask juntos/,
  );
  assert.throws(
    () => executeGemma4CompositeF32(program, { inputIds: [[1]], tensors, mmTokenTypeIds: [[0, 1]] }),
    /deve acompanhar input_ids/,
  );
  assert.throws(
    () => executeReferenceF32WithPreparedPrelude(program.textProgram, { inputIds: [[1]], tensors }, new Map([["hidden_states_0", patterned([1, 1, 4])]])),
    /saída obrigatória ple_token_identity/,
  );
});

function fixture(options: { layers?: number; layerTypes?: Array<"sliding_attention" | "full_attention">; sharedKeyValueLayers?: number } = {}): ModelCatalog {
  const tensors = new Map<string, TensorInfo>();
  const add = (name: string, shape: number[]): void => { tensors.set(name, { name, storageDtype: "F32", storageShape: shape, logicalShape: shape }); };
  const clipped = (prefix: string, shape: number[]): void => { add(`${prefix}.linear.weight`, shape); for (const suffix of ["input_min", "input_max", "output_min", "output_max"]) add(`${prefix}.${suffix}`, []); };
  const hidden = 4, layers = options.layers ?? 2, ple = 1, vocab = 6, text = "model.language_model";
  const layerTypes = options.layerTypes ?? ["full_attention", "sliding_attention"];
  const sharedKeyValueLayers = options.sharedKeyValueLayers ?? 0;
  if (layerTypes.length !== layers) throw new Error("fixture layerTypes must match layers");
  add(`${text}.embed_tokens.weight`, [vocab, hidden]); add(`${text}.embed_tokens_per_layer.weight`, [vocab, layers * ple]); add(`${text}.per_layer_model_projection.weight`, [layers * ple, hidden]); add(`${text}.per_layer_projection_norm.weight`, [ple]); add(`${text}.norm.weight`, [hidden]);
  for (let layer = 0; layer < layers; layer += 1) {
    const textLayer = `${text}.layers.${layer}`;
    for (const norm of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"]) add(`${textLayer}.${norm}.weight`, [hidden]);
    for (const projection of ["q_proj", "k_proj", "v_proj"]) add(`${textLayer}.self_attn.${projection}.weight`, [hidden, hidden]);
    add(`${textLayer}.self_attn.q_norm.weight`, [hidden]); add(`${textLayer}.self_attn.k_norm.weight`, [hidden]); add(`${textLayer}.self_attn.o_proj.weight`, [hidden, hidden]); add(`${textLayer}.per_layer_input_gate.weight`, [ple, hidden]); add(`${textLayer}.per_layer_projection.weight`, [hidden, ple]); add(`${textLayer}.layer_scalar`, [1]);
    for (const projection of ["gate_proj", "up_proj"]) add(`${textLayer}.mlp.${projection}.weight`, [8, hidden]); add(`${textLayer}.mlp.down_proj.weight`, [hidden, 8]);
  }
  const vision = "model.vision_tower";
  add(`${vision}.patch_embedder.input_proj.weight`, [hidden, 12]); add(`${vision}.patch_embedder.position_embedding_table`, [2, 4, hidden]);
  const visionLayer = `${vision}.encoder.layers.0`;
  for (const norm of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]) add(`${visionLayer}.${norm}.weight`, [hidden]);
  add(`${visionLayer}.self_attn.q_norm.weight`, [hidden]); add(`${visionLayer}.self_attn.k_norm.weight`, [hidden]);
  for (const projection of ["q_proj", "k_proj", "v_proj", "o_proj"]) clipped(`${visionLayer}.self_attn.${projection}`, [hidden, hidden]);
  clipped(`${visionLayer}.mlp.gate_proj`, [8, hidden]); clipped(`${visionLayer}.mlp.up_proj`, [8, hidden]); clipped(`${visionLayer}.mlp.down_proj`, [hidden, 8]); add("model.embed_vision.embedding_projection.weight", [hidden, hidden]);
  const audio = "model.audio_tower";
  add(`${audio}.subsample_conv_projection.layer0.conv.weight`, [1, 1, 3, 3]); add(`${audio}.subsample_conv_projection.layer0.norm.weight`, [1]); add(`${audio}.subsample_conv_projection.layer1.conv.weight`, [1, 1, 3, 3]); add(`${audio}.subsample_conv_projection.layer1.norm.weight`, [1]); add(`${audio}.subsample_conv_projection.input_proj_linear.weight`, [hidden, hidden]);
  const audioLayer = `${audio}.layers.0`;
  for (const norm of ["norm_pre_attn", "norm_post_attn", "norm_out"]) add(`${audioLayer}.${norm}.weight`, [hidden]); add(`${audioLayer}.self_attn.per_dim_scale`, [hidden]); add(`${audioLayer}.self_attn.relative_k_proj.weight`, [hidden, hidden]);
  for (const projection of ["q_proj", "k_proj", "v_proj", "post"]) clipped(`${audioLayer}.self_attn.${projection}`, [hidden, hidden]);
  for (const ffn of ["feed_forward1", "feed_forward2"]) { add(`${audioLayer}.${ffn}.pre_layer_norm.weight`, [hidden]); add(`${audioLayer}.${ffn}.post_layer_norm.weight`, [hidden]); clipped(`${audioLayer}.${ffn}.ffw_layer_1`, [hidden * 4, hidden]); clipped(`${audioLayer}.${ffn}.ffw_layer_2`, [hidden, hidden * 4]); }
  add(`${audioLayer}.lconv1d.pre_layer_norm.weight`, [hidden]); add(`${audioLayer}.lconv1d.conv_norm.weight`, [hidden]); add(`${audioLayer}.lconv1d.depthwise_conv1d.weight`, [hidden, 1, 5]); clipped(`${audioLayer}.lconv1d.linear_start`, [hidden * 2, hidden]); clipped(`${audioLayer}.lconv1d.linear_end`, [hidden, hidden]); add(`${audio}.output_proj.weight`, [hidden, hidden]); add(`${audio}.output_proj.bias`, [hidden]); add("model.embed_audio.embedding_projection.weight", [hidden, hidden]);
  return { source: "/tmp/tiny-gemma4-composite", format: "safetensors", rawMetadata: {}, tensors, config: {
    model_type: "gemma4", image_token_id: 99, video_token_id: 97, audio_token_id: 98,
    vision_config: { model_type: "gemma4_vision", hidden_size: hidden, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1, head_dim: hidden, intermediate_size: 8, patch_size: 2, position_embedding_size: 4, default_output_length: 1, pooling_kernel_size: 2, attention_bias: false, hidden_activation: "gelu_pytorch_tanh", use_clipped_linears: true, rms_norm_eps: 1e-6, rope_parameters: { rope_type: "default", rope_theta: 100 } },
    audio_config: { model_type: "gemma4_audio", hidden_size: hidden, num_hidden_layers: 1, num_attention_heads: 1, output_proj_dims: hidden, attention_chunk_size: 2, attention_context_left: 1, attention_context_right: 0, attention_logit_cap: 50, attention_invalid_logits_value: -1e9, gradient_clipping: 1e10, conv_kernel_size: 5, residual_weight: 0.5, subsampling_conv_channels: [1, 1], hidden_act: "silu", use_clipped_linears: true, rms_norm_eps: 1e-6 },
    text_config: { model_type: "gemma4_text", hidden_size: hidden, vocab_size: vocab, pad_token_id: 0, num_hidden_layers: layers, num_attention_heads: 1, num_key_value_heads: 1, global_head_dim: hidden, head_dim: hidden, intermediate_size: 8, num_kv_shared_layers: sharedKeyValueLayers, hidden_size_per_layer_input: ple, vocab_size_per_layer_input: vocab, attention_bias: false, attention_k_eq_v: false, enable_moe_block: false, use_double_wide_mlp: false, rms_norm_eps: 1e-6, sliding_window: 4, layer_types: layerTypes, rope_parameters: { sliding_attention: { rope_type: "default", rope_theta: 10_000 }, full_attention: { rope_type: "proportional", rope_theta: 1_000_000, partial_rotary_factor: 1 } } },
  } };
}

function patterned(shape: number[]): DenseF32Tensor { return { shape, values: Float32Array.from({ length: shape.reduce((total, dimension) => total * dimension, 1) }, (_, index) => Math.fround((index % 9 + 1) / 25)) }; }
function materialize(catalog: ModelCatalog): Map<string, DenseF32Tensor> { const result = new Map<string, DenseF32Tensor>(); for (const entry of catalog.tensors.values()) { const size = entry.logicalShape.reduce((total, dimension) => total * dimension, 1) || 1; const values = new Float32Array(size); if (entry.name.endsWith("input_min") || entry.name.endsWith("output_min")) values[0] = -100; else if (entry.name.endsWith("input_max") || entry.name.endsWith("output_max")) values[0] = 100; else if (entry.name.endsWith("norm.weight")) values.fill(1); else if (entry.name.endsWith("layer_scalar")) values.fill(1); else for (let index = 0; index < size; index += 1) values[index] = Math.fround((index % 5 + 1) / 50); result.set(entry.name, { shape: [...entry.logicalShape], values }); } return result; }
async function writeFixtureSafetensors(directory: string, catalog: ModelCatalog, tensors: ReadonlyMap<string, DenseF32Tensor>): Promise<void> {
  await mkdir(directory, { recursive: true });
  const header: Record<string, unknown> = {};
  const payloads: Buffer[] = [];
  let offset = 0;
  for (const tensor of catalog.tensors.values()) {
    const payload = denseF32Bytes(tensors.get(tensor.name)!);
    header[tensor.name] = { dtype: "F32", shape: tensor.logicalShape, data_offsets: [offset, offset + payload.length] };
    offset += payload.length;
    payloads.push(payload);
  }
  const encodedHeader = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(encodedHeader.length));
  await writeFile(path.join(directory, "config.json"), JSON.stringify(catalog.config));
  await writeFile(path.join(directory, "model.safetensors"), Buffer.concat([prefix, encodedHeader, ...payloads]));
}
function denseF32Bytes(tensor: DenseF32Tensor): Buffer { const bytes = Buffer.alloc(tensor.values.byteLength); tensor.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4)); return bytes; }
