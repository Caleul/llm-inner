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
import { executeGemma4LiteralCompositeF32, generateGemma4LiteralCompositeF32 } from "../src/gemma4-literal-composite.js";
import { buildGemma4LiteralGenerationNavigation, renderGemma4LiteralGenerationCalculationView } from "../src/gemma4-literal-generation-navigation.js";
import { selectGemma4LiteralGenerationToken } from "../src/gemma4-literal-generation-control.js";
import { executeGemma4LiteralForwardControlProgram } from "../src/gemma4-literal-forward-control.js";
import { executeGemma4LiteralInputContract } from "../src/gemma4-literal-input-contract.js";
import {
  executeGemma4LiteralForwardOutputContract,
  executeGemma4LiteralGenerationOutputContract,
} from "../src/gemma4-literal-output-contract.js";
import { buildGemma4LiteralCalculationSlice } from "../src/gemma4-literal-calculation-slice.js";
import { buildGemma4LiteralEndToEndCalculation } from "../src/gemma4-literal-end-to-end-calculation.js";
import {
  evaluateGemma4LiteralDimensionProgram,
  evaluateGemma4LiteralDimensionPrograms,
  evaluateGemma4LiteralShapeExpression,
} from "../src/gemma4-literal-dimension-programs.js";
import { evaluateGemma4LiteralLearnedOperandIndices } from "../src/gemma4-literal-learned-operands.js";
import {
  buildGemma4LiteralScalarCalculations,
  buildGemma4LiteralScalarStatementDataflow,
} from "../src/gemma4-literal-scalar-calculations.js";
import {
  bindGemma4LiteralScalarStatementPrograms,
  buildGemma4LiteralScalarStatementEnvironment,
  buildGemma4LiteralScalarStatementPrograms,
  parseGemma4LiteralScalarExpression,
  validateGemma4LiteralScalarStatementPrograms,
} from "../src/gemma4-literal-scalar-statement-programs.js";
import {
  evaluateGemma4LiteralReductionIndexDomains,
  gemma4LiteralReductionDomainLanguage,
} from "../src/gemma4-literal-reduction-domains.js";
import {
  executeGemma4LiteralAudioRelativeShiftSource,
  executeGemma4LiteralContiguousVisionGroupId,
  executeGemma4LiteralStableTrueCoordinateAtRank,
  executeGemma4LiteralStableTrueCount,
  executeGemma4LiteralStableTruePrefixRank,
  executeGemma4LiteralVisionPoolCellHasPatch,
  executeGemma4LiteralVisionPoolSlot,
  executeGemma4LiteralSoftmaxReductionProgram,
  gemma4LiteralIndexingPrograms,
  gemma4LiteralSoftmaxReductionPrograms,
  validateGemma4LiteralFormulaFunctionCoverage,
} from "../src/gemma4-literal-formula-language.js";
import { buildGemma4LiteralSourceIdentity, type Gemma4LiteralSourceIdentity } from "../src/gemma4-literal-source-identity.js";
import {
  listGemma4LiteralOperations,
  renderGemma4LiteralMultimodalScalarView,
} from "../src/gemma4-literal-multimodal-scalar-view.js";
import {
  listGemma4LiteralRuntimeReductionOperations,
  renderGemma4LiteralRuntimeReductionAudit,
} from "../src/gemma4-literal-runtime-reduction-audit.js";
import { buildGemma4LiteralFidelityGate } from "../src/gemma4-literal-fidelity-gate.js";
import { executeGemma4LiteralVisionF32 } from "../src/gemma4-literal-vision.js";
import { listGemma4LiteralTextOperations, renderGemma4LiteralScalarView, validateGemma4LiteralScalarView } from "../src/gemma4-literal-scalar-view.js";
import {
  buildGemma4LiteralLinearReductionAssignments,
  gemma4LiteralScalarProductFormula,
} from "../src/gemma4-literal-linear-reduction-view.js";
import {
  buildGemma4LiteralCascadeSquareReductionAssignments,
  buildGemma4LiteralWelfordAssignments,
} from "../src/gemma4-literal-normalization-reduction-view.js";
import {
  verifyGemma4CompositeLiteralEmbeddedPayloadIntegrity,
  verifyGemma4CompositeLiteralPayloadsAgainstCatalog,
} from "../src/gemma4-composite-literal-payload-verification.js";
import { probeGemma4LiteralLinearReductionProfiles } from "../src/gemma4-linear-reduction-probe.js";
import { executeGemma4PagedTextLiteralF32, generateGemma4PagedTextLiteralF32 } from "../src/gemma4-paged-text.js";
import { executeGemma4VisionF32 } from "../src/gemma4-vision.js";
import { GEMMA4_E4B_PYTORCH_BF16_ATTENTION_IMPLEMENTATION } from "../src/gemma4-text.js";
import {
  assertGemma4AuthoritativeRuntime,
  GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256,
  GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY,
  GEMMA4_RUNTIME_REDUCTION_EXECUTION_STATE,
  GEMMA4_RUNTIME_REDUCTION_PROCESS_ENVIRONMENT,
  GEMMA4_AUDIO_REFERENCE_RUNTIME,
  GEMMA4_COMPOSITE_REFERENCE_RUNTIME,
  GEMMA4_VISION_REFERENCE_RUNTIME,
} from "../src/gemma4-authoritative-runtime.js";
import { gemma4CompositeTraceProfile } from "../src/gemma4-composite-trace-profile.js";
import { validateGemma4CompositeTraceOptions } from "../src/gemma4-transformers-composite-trace.js";
import { createPagedDenseF32Matrix, pagedEmbeddingF32, pagedLinearF32 } from "../src/paged-dense.js";
import { fingerprintIR } from "../src/trace.js";
import { auditLiteralArtifact } from "../src/literal-artifact-audit.js";
import {
  buildGemma4LiteralTranscendentalPrograms,
  executeGemma4LiteralTranscendentalProgram,
  validateGemma4LiteralTranscendentalCoverage,
  validateGemma4LiteralTranscendentalPrograms,
} from "../src/gemma4-literal-transcendental-programs.js";
import {
  buildLiteralDenseDecoderLanguageContract,
  buildLiteralDenseStorageDecodeAssignment,
  decodeLiteralDenseElementF32,
  evaluateLiteralDenseElementAddress,
} from "../src/literal.js";
import {
  buildGemma4LiteralConsumerCoordinateNavigation,
  buildGemma4LiteralOutputCoordinateNavigation,
  evaluateGemma4LiteralCoordinateExpression,
  extractGemma4LiteralCoordinateAccesses,
  gemma4LiteralCoordinateExpressionLanguage,
  parseGemma4LiteralCoordinateExpression,
  validateGemma4LiteralCoordinateExpressionBindings,
} from "../src/gemma4-literal-coordinate-accesses.js";
import type { DenseF32Tensor, ModelCatalog, TensorInfo } from "../src/types.js";

const preview = { outputRows: 1, inputTerms: 1, includeWeights: false } as const;
const fixtureConfigBytes = Buffer.from("{}");

test("Gemma 4 scalar statement dataflow exposes local coordinates and bidirectional edges", () => {
  const dataflow = buildGemma4LiteralScalarStatementDataflow([
    "pair = head%2",
    "score[key] = F32(input[pair,key])",
    "maximum = ORDERED_F32_REDUCE_MAX(score[key],key=0..K-1)",
    "out[batch,head] = F32(score[pair]-maximum)",
  ], "out", "fixture:dataflow");
  assert.deepEqual(dataflow[1]?.writes[0], {
    kind: "indexed", name: "score", expression: "score[key]",
    coordinates: ["key"], coordinatePrograms: [{ kind: "symbol", name: "key" }], role: "local",
  });
  assert.deepEqual(dataflow[2]?.reads.map((read) => [read.expression, read.producerStatementOrdinal]), [["score[key]", 1]]);
  assert.deepEqual(dataflow[1]?.consumerStatementOrdinals, [2, 3]);
  assert.deepEqual(dataflow[3]?.reads.map((read) => [read.expression, read.producerStatementOrdinal]), [
    ["score[pair]", 1], ["pair", 0], ["maximum", 2],
  ]);
  assert.throws(() => buildGemma4LiteralScalarStatementDataflow([
    "tmp = F32(later)", "later = F32(0)", "out[0] = tmp",
  ], "out", "fixture:read-before-write"), /lê local later sem produtor anterior/);
  assert.throws(() => buildGemma4LiteralScalarStatementDataflow([
    "out[0] = F32(0)", "out[1] = F32(1)",
  ], "out", "fixture:duplicate-output"), /uma única escrita terminal/);
});

test("Gemma 4 scalar statement programs embed closed syntax trees and reject textual drift", () => {
  const assignments = [
    "score[key] = F32(input[batch,key]*decode(weight)[hidden,key])",
    "maximum = ORDERED_F32_REDUCE_MAX(score[key],key=0..K-1 where mask[batch,key])",
    "out[batch,hidden] = mask[batch,hidden] ? F32(score[hidden]-maximum) : F32(0)",
  ];
  const programs = buildGemma4LiteralScalarStatementPrograms(assignments, "out", "fixture:syntax");
  assert.equal(programs.length, assignments.length);
  assert.deepEqual(programs[0]?.targets[0], {
    name: "score", role: "local",
    coordinates: [{ kind: "identifier", name: "key" }],
  });
  assert.equal(programs[1]?.expression.kind, "call");
  assert.equal(programs[2]?.expression.kind, "conditional");
  assert.doesNotThrow(() => validateGemma4LiteralScalarStatementPrograms(programs, assignments, "out", "fixture:syntax"));
  const environment = buildGemma4LiteralScalarStatementEnvironment(programs, {
    output: "out",
    outputCoordinates: ["batch", "hidden"],
    orderedInputs: ["input", "mask"],
    learnedOperandRoles: ["weight"],
    reductions: [{ indices: ["key=0..K-1"], source: "assignment" }],
  }, "fixture:syntax");
  assert.deepEqual(environment.orderedInputs, [{ position: 0, name: "input" }, { position: 1, name: "mask" }]);
  assert.deepEqual(environment.reductions, [{
    index: "key", source: "assignment", domainOrdinal: 0, extentAlias: "K",
  }]);
  assert.ok(environment.intrinsics.includes("decode"));
  const freeIdentifier = buildGemma4LiteralScalarStatementPrograms([
    "out[batch] = F32(host_value)",
  ], "out", "fixture:free-identifier");
  assert.throws(() => buildGemma4LiteralScalarStatementEnvironment(freeIdentifier, {
    output: "out", outputCoordinates: ["batch"], orderedInputs: [], learnedOperandRoles: [], reductions: [],
  }, "fixture:free-identifier"), /identificador escalar livre host_value/);
  const unknownMember = buildGemma4LiteralScalarStatementPrograms([
    "out[batch] = input.host_stride",
  ], "out", "fixture:unknown-member");
  assert.throws(() => buildGemma4LiteralScalarStatementEnvironment(unknownMember, {
    output: "out", outputCoordinates: ["batch"], orderedInputs: ["input"], learnedOperandRoles: [], reductions: [],
  }, "fixture:unknown-member"), /acesso de membro host_stride não resolvido/);
  const boundAssignments = assignments.map((source) => source.replaceAll("input", "invocation/input"));
  const bound = bindGemma4LiteralScalarStatementPrograms(programs, new Map([["input", "invocation/input"]]), boundAssignments);
  assert.equal(bound[0]?.source, boundAssignments[0]);
  assert.match(JSON.stringify(bound[0]?.expression), /"name":"invocation\/input"/);
  const altered = structuredClone(programs);
  altered[2]!.source = altered[2]!.source.replace("F32(0)", "F32(1)");
  assert.throws(
    () => validateGemma4LiteralScalarStatementPrograms(altered, assignments, "out", "fixture:syntax"),
    /programa sintático escalar ausente ou divergente/,
  );
  assert.equal(parseGemma4LiteralScalarExpression(
    "pool_slot[patch]==pool_cell ? acc[patch-1] : F32(0), patch=0..patches-1 ascending",
  ).kind, "ordered-loop");
  assert.deepEqual(parseGemma4LiteralScalarExpression("decode(normalization-scale)[hidden-1]"), {
    kind: "index",
    target: {
      kind: "call",
      callee: { kind: "identifier", name: "decode" },
      arguments: [{ kind: "identifier", name: "normalization-scale" }],
    },
    coordinates: [{
      kind: "binary", operator: "-",
      left: { kind: "identifier", name: "hidden" },
      right: { kind: "literal", literalType: "number", source: "1" },
    }],
  });
  assert.throws(() => parseGemma4LiteralScalarExpression("F32(value) trailing"), /token inesperado trailing/);
});

test("Gemma 4 embeds one executable F32 runtime-math program class for every formula", () => {
  const programs = buildGemma4LiteralTranscendentalPrograms();
  assert.deepEqual(Object.keys(programs.programs), [
    "SLEEF_EXP_F32", "SLEEF_SIN_F32", "SLEEF_COS_F32", "SLEEF_TANH_F32",
    "SLEEF_LOG1P_F32", "ARM_SQRT_F32", "PYTORCH_POW_NEGATIVE_HALF_F32",
  ]);
  assert.equal(programs.rempiTable.entries, 416);
  assert.equal(Buffer.from(programs.rempiTable.payloadBase64, "base64").length, 416 * 4);
  assert.ok(programs.constants.every((constant) => /^0x[0-9a-f]{8}$/.test(constant.binary32Hex) && Math.fround(constant.value) === constant.value));
  assert.ok(programs.sharedSubprograms.some((program) => program.name === "REMPI_F32" && program.scalarAssignments.length >= 4));
  assert.ok(programs.sharedSubprograms.some((program) => program.name === "EXP_PAIR" && program.scalarAssignments.length >= 5));
  assert.equal(executeGemma4LiteralTranscendentalProgram(programs, "SLEEF_EXP_F32", 1), 2.7182817459106445);
  assert.equal(executeGemma4LiteralTranscendentalProgram(programs, "SLEEF_SIN_F32", 125), -0.6160404682159424);
  assert.equal(executeGemma4LiteralTranscendentalProgram(programs, "SLEEF_COS_F32", 125), 0.7877144813537598);
  assert.equal(executeGemma4LiteralTranscendentalProgram(programs, "SLEEF_TANH_F32", 1), 0.7615941762924194);
  assert.equal(executeGemma4LiteralTranscendentalProgram(programs, "SLEEF_LOG1P_F32", 1), 0.6931471824645996);
  assert.equal(executeGemma4LiteralTranscendentalProgram(programs, "SLEEF_LOG1P_F32", -0.5), -0.6931471824645996);
  assert.equal(executeGemma4LiteralTranscendentalProgram(programs, "SLEEF_LOG1P_F32", 20), 3.044522523880005);
  assert.equal(executeGemma4LiteralTranscendentalProgram(programs, "ARM_SQRT_F32", 4.022159099578857), 2.0055320262908936);
  assert.equal(executeGemma4LiteralTranscendentalProgram(programs, "PYTORCH_POW_NEGATIVE_HALF_F32", 4.022159099578857), 0.49862080812454224);
  const corrupted = structuredClone(programs);
  corrupted.programs.SLEEF_EXP_F32.scalarAssignments[0] = "result=host_exp(input)";
  assert.throws(() => validateGemma4LiteralTranscendentalPrograms(corrupted), /transcrição transcendental F32/);
  assert.throws(() => executeGemma4LiteralTranscendentalProgram(corrupted, "SLEEF_EXP_F32", 1), /transcrição transcendental F32/);
  assert.throws(() => validateGemma4LiteralTranscendentalCoverage(programs, ["y=exp(x)"]), /intrínseco matemático opaco/);
  assert.throws(() => validateGemma4LiteralTranscendentalCoverage(programs, ["y=SLEEF_ERF_F32(x)"]), /sem programa incorporado/);
});

test("Gemma 4 coordinate navigation parses nested indices without operation or layer dispatch", () => {
  assert.deepEqual(extractGemma4LiteralCoordinateAccesses("hidden/state", [
    "term[i]=F32(hidden/state[batch,key_index[i],floor((head*width+i)/2)]*weight[i])",
    "require key_index[i]<hidden/state.shape[1]",
  ]), [
    {
      kind: "tensor-element",
      expression: "hidden/state[batch,key_index[i],floor((head*width+i)/2)]",
      coordinates: ["batch", "key_index[i]", "floor((head*width+i)/2)"],
      coordinatePrograms: [
        { kind: "symbol", name: "batch" },
        { kind: "indexed-symbol", name: "key_index", indices: [{ kind: "symbol", name: "i" }] },
        {
          kind: "floor-divide",
          left: {
            kind: "add",
            left: { kind: "multiply", left: { kind: "symbol", name: "head" }, right: { kind: "symbol", name: "width" } },
            right: { kind: "symbol", name: "i" },
          },
          right: { kind: "constant", value: 2 },
        },
      ],
    },
    { kind: "tensor-shape", expression: "hidden/state.shape[1]", axis: "1", axisProgram: { kind: "constant", value: 1 } },
  ]);
  assert.deepEqual(extractGemma4LiteralCoordinateAccesses("cache", ["next=tuple(cache,delta)"]), [
    { kind: "whole-value", expression: "cache" },
  ]);
  assert.deepEqual(buildGemma4LiteralOutputCoordinateNavigation("result", [
    "require result.shape[0]==source.shape[0]",
    "result[batch,key_index[i]]=F32(source[batch,i])",
  ]), {
    write: {
      kind: "tensor-element",
      expression: "result[batch,key_index[i]]",
      coordinates: ["batch", "key_index[i]"],
      coordinatePrograms: [
        { kind: "symbol", name: "batch" },
        { kind: "indexed-symbol", name: "key_index", indices: [{ kind: "symbol", name: "i" }] },
      ],
    },
    shapeAssertions: [{ kind: "tensor-shape", expression: "result.shape[0]", axis: "0", axisProgram: { kind: "constant", value: 0 } }],
  });
  assert.deepEqual(buildGemma4LiteralConsumerCoordinateNavigation("result", [{
    operationId: "consumer",
    scalarAssignments: ["next[b,i]=result[b,floor((i+1)/2)]"],
  }]), [{
    operationId: "consumer",
    accesses: [{
      kind: "tensor-element",
      expression: "result[b,floor((i+1)/2)]",
      coordinates: ["b", "floor((i+1)/2)"],
      coordinatePrograms: [
        { kind: "symbol", name: "b" },
        {
          kind: "floor-divide",
          left: { kind: "add", left: { kind: "symbol", name: "i" }, right: { kind: "constant", value: 1 } },
          right: { kind: "constant", value: 2 },
        },
      ],
    }],
    scalarUse: "addressed",
  }]);
  assert.deepEqual(extractGemma4LiteralCoordinateAccesses("hidden", [
    "heads[b,h,s,d]=row_major_alias(hidden)[b,s,h*64+d]",
  ]), [{
    kind: "tensor-element", expression: "hidden[b,s,h*64+d]", coordinates: ["b", "s", "h*64+d"],
    coordinatePrograms: [
      { kind: "symbol", name: "b" },
      { kind: "symbol", name: "s" },
      {
        kind: "add",
        left: { kind: "multiply", left: { kind: "symbol", name: "h" }, right: { kind: "constant", value: 64 } },
        right: { kind: "symbol", name: "d" },
      },
    ],
  }]);
  assert.equal(gemma4LiteralCoordinateExpressionLanguage().id, "gemma4-coordinate-expression-v2");
  assert.deepEqual(evaluateGemma4LiteralCoordinateExpression(
    parseGemma4LiteralCoordinateExpression("floor(attention_hidden/256)%4"),
    { symbols: { attention_hidden: 1793 } },
  ), 3);
  assert.deepEqual(evaluateGemma4LiteralCoordinateExpression(
    parseGemma4LiteralCoordinateExpression("0..patches-1"),
    { symbols: { patches: 9 } },
  ), { startInclusive: 0, endInclusive: 8 });
  assert.deepEqual(evaluateGemma4LiteralCoordinateExpression(
    parseGemma4LiteralCoordinateExpression("floor(video_frame/(pixel_values_videos.shape[1]))"),
    { symbols: { video_frame: 7 }, tensorShapes: { pixel_values_videos: [2, 4, 3, 8] } },
  ), 1);
  assert.deepEqual(evaluateGemma4LiteralCoordinateExpression(
    parseGemma4LiteralCoordinateExpression("floor(flattened_channel_feature/(composite_audio_features/audio_subsample_1.shape[1]))"),
    {
      symbols: { flattened_channel_feature: 67 },
      tensorShapes: { "composite_audio_features/audio_subsample_1": [1, 32, 4, 8] },
    },
  ), 2);
  const closedScope = {
    symbols: new Set(["video_frame"]),
    localRoots: new Set<string>(),
    tensors: new Set(["pixel_values_videos"]),
  };
  validateGemma4LiteralCoordinateExpressionBindings(
    parseGemma4LiteralCoordinateExpression("floor(video_frame/(pixel_values_videos.shape[1]))"), closedScope, "video-flatten",
  );
  validateGemma4LiteralCoordinateExpressionBindings(
    parseGemma4LiteralCoordinateExpression("floor(flattened_channel_feature/(composite_audio_features/audio_subsample_1.shape[1]))"),
    {
      symbols: new Set(["flattened_channel_feature"]),
      localRoots: new Set<string>(),
      tensors: new Set(["composite_audio_features/audio_subsample_1"]),
    },
    "audio-unfold",
  );
  assert.throws(() => validateGemma4LiteralCoordinateExpressionBindings(
    parseGemma4LiteralCoordinateExpression("floor(video_frame/frames)"), closedScope, "video-flatten",
  ), /binding livre frames/);
  assert.throws(() => validateGemma4LiteralCoordinateExpressionBindings(
    parseGemma4LiteralCoordinateExpression("undeclared.shape[0]"), closedScope, "video-flatten",
  ), /tensor não declarado undeclared/);
  assert.throws(() => evaluateGemma4LiteralCoordinateExpression(
    parseGemma4LiteralCoordinateExpression("pixel_values_videos.shape[4]"),
    { symbols: {}, tensorShapes: { pixel_values_videos: [2, 4, 3, 8] } },
  ), /não pode resolver shape\[4\]/);
  assert.equal(evaluateGemma4LiteralCoordinateExpression(
    parseGemma4LiteralCoordinateExpression("STABLE_TRUE_PREFIX_RANK(input_ids==258880,batch,sequence)"),
    {
      symbols: { batch: 1, sequence: 1 },
      integerTensors: { input_ids: [[258880, 2], [258880, 258880]] },
    },
  ), 2);
  assert.throws(() => parseGemma4LiteralCoordinateExpression("host_index(x)"), /sem programa incorporado/);
  assert.throws(() => evaluateGemma4LiteralCoordinateExpression(
    parseGemma4LiteralCoordinateExpression("0..patches-1"), { symbols: { patches: 0 } },
  ), /Range de coordenada invertido/);
  assert.throws(() => buildGemma4LiteralOutputCoordinateNavigation("result", ["next[i]=result[i]"]), /uma única escrita tensorial/);
  assert.throws(() => buildGemma4LiteralOutputCoordinateNavigation("result", [
    "result[i]=source[i]", "require result.shape[0]==source.shape[0]",
  ]), /precondição anterior à escrita/);
  assert.throws(() => extractGemma4LiteralCoordinateAccesses("x", ["y=x[batch,key[i]"]), /sem '\]' final/);
});

test("Gemma 4 authoritative traces bind eager inference mode instead of accepting no-grad drift", () => {
  const context = (runtime: string) => ({ runtime, executionMode: "torch.inference_mode", attentionImplementation: "eager" });
  assert.doesNotThrow(() => assertGemma4AuthoritativeRuntime("audio", context(GEMMA4_AUDIO_REFERENCE_RUNTIME)));
  assert.doesNotThrow(() => assertGemma4AuthoritativeRuntime("vision", context(GEMMA4_VISION_REFERENCE_RUNTIME)));
  assert.doesNotThrow(() => assertGemma4AuthoritativeRuntime("composite", context(GEMMA4_COMPOSITE_REFERENCE_RUNTIME)));
  assert.throws(
    () => assertGemma4AuthoritativeRuntime("audio", context("transformers-5.5.0/torch-2.12.1-Gemma4Audio-CPU-eager")),
    /esperado .*inference-mode/,
  );
  assert.throws(
    () => assertGemma4AuthoritativeRuntime("composite", context("transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager")),
    /esperado .*inference-mode/,
  );
  assert.throws(
    () => assertGemma4AuthoritativeRuntime("audio", { ...context(GEMMA4_AUDIO_REFERENCE_RUNTIME), attentionImplementation: "sdpa" }),
    /attentionImplementation='sdpa'.*esperado/,
  );
  assert.throws(
    () => assertGemma4AuthoritativeRuntime("audio", { ...context(GEMMA4_AUDIO_REFERENCE_RUNTIME), executionMode: "torch.no_grad" }),
    /executionMode='torch.no_grad'.*esperado/,
  );
});

const fixtureSourceIdentity = (): Gemma4LiteralSourceIdentity => ({
  modelId: "fixture/tiny-gemma4",
  revision: "a".repeat(40),
  sourceFormat: "safetensors",
  semanticAdapter: "gemma4-composite-v1",
  files: [
    {
      path: "config.json", role: "model-config", bytes: fixtureConfigBytes.length,
      sha256: createHash("sha256").update(fixtureConfigBytes).digest("hex"),
      content: { storage: "embedded-metadata-base64", encoding: "base64", payloadBase64: fixtureConfigBytes.toString("base64"), decode: "base64-to-original-bytes" },
    },
    {
      path: "model.safetensors", role: "weights", bytes: 1, sha256: "b".repeat(64),
      content: { storage: "embedded-tensor-constants", mapping: "safetensors-header-ranges-to-named-constants" },
    },
  ],
});

function referenceF16ToF32Bits(sourceBits: number): number {
  const sign = sourceBits >>> 15;
  const exponent = sourceBits >>> 10 & 0x1f;
  const fraction = sourceBits & 0x03ff;
  if (exponent === 0x1f) return ((sign << 31) | 0x7f800000 | (fraction << 13)) >>> 0;
  if (exponent !== 0) return ((sign << 31) | ((exponent + 112) << 23) | (fraction << 13)) >>> 0;
  if (fraction === 0) return (sign << 31) >>> 0;
  const shiftCount = Math.clz32(fraction) - 21;
  return ((sign << 31) | ((113 - shiftCount) << 23) | (((fraction << shiftCount) & 0x03ff) << 13)) >>> 0;
}

test("dense literal decoders execute serialized row-major addresses and IEEE bit programs", () => {
  const metadata = (storageDtype: "F32" | "F16" | "BF16") => ({
    name: `tensor_${storageDtype.toLowerCase()}`,
    storageDtype,
    storageShape: [2, 3],
    logicalShape: [2, 3],
    layout: "row-major" as const,
    byteOrder: "little-endian" as const,
  });
  const f32 = buildLiteralDenseStorageDecodeAssignment(metadata("F32"));
  assert.deepEqual(f32.address.stridesElements, [3, 1]);
  assert.deepEqual(evaluateLiteralDenseElementAddress(f32, [1, 2]), {
    elementOffset: 5, byteOffset: 20, byteLength: 4,
  });
  const oneF32 = Buffer.alloc(4); oneF32.writeUInt32LE(0x3f800000);
  assert.deepEqual(decodeLiteralDenseElementF32(f32, oneF32), {
    sourceBits: 0x3f800000, sourceBitsHex: "0x3f800000",
    decodedF32Bits: 0x3f800000, decodedF32BitsHex: "0x3f800000", decodedF32: 1,
  });
  const f16 = buildLiteralDenseStorageDecodeAssignment(metadata("F16"));
  const oneF16 = Buffer.alloc(2); oneF16.writeUInt16LE(0x3c00);
  assert.equal(decodeLiteralDenseElementF32(f16, oneF16).decodedF32, 1);
  assert.equal(f16.decode.kind, "ieee-binary16-expand");
  assert.equal(f16.decode.schemaVersion, 2);
  assert.equal(f16.decode.bindings.map((binding) => binding.name).join(","), "sign,exponent,fraction,shiftCount");
  const subnormalF16 = Buffer.alloc(2); subnormalF16.writeUInt16LE(0x0001);
  assert.deepEqual(decodeLiteralDenseElementF32(f16, subnormalF16), {
    sourceBits: 1, sourceBitsHex: "0x0001", decodedF32Bits: 0x33800000,
    decodedF32BitsHex: "0x33800000", decodedF32: 2 ** -24,
  });
  const negativeZeroF16 = Buffer.alloc(2); negativeZeroF16.writeUInt16LE(0x8000);
  const negativeZero = decodeLiteralDenseElementF32(f16, negativeZeroF16);
  assert.ok(Object.is(negativeZero.decodedF32, -0));
  assert.equal(negativeZero.decodedF32BitsHex, "0x80000000");
  const infinityF16 = Buffer.alloc(2); infinityF16.writeUInt16LE(0x7c00);
  assert.equal(decodeLiteralDenseElementF32(f16, infinityF16).decodedF32, Infinity);
  const nanF16 = Buffer.alloc(2); nanF16.writeUInt16LE(0x7e00);
  const nan = decodeLiteralDenseElementF32(f16, nanF16);
  assert.ok(Number.isNaN(nan.decodedF32));
  assert.equal(nan.decodedF32BitsHex, "0x7fc00000");
  const bf16 = buildLiteralDenseStorageDecodeAssignment(metadata("BF16"));
  const oneBF16 = Buffer.alloc(2); oneBF16.writeUInt16LE(0x3f80);
  assert.equal(decodeLiteralDenseElementF32(bf16, oneBF16).decodedF32, 1);
  assert.equal(decodeLiteralDenseElementF32(bf16, oneBF16).decodedF32BitsHex, "0x3f800000");
  assert.equal(bf16.decode.kind, "ieee-bfloat16-expand");
  for (let sourceBits = 0; sourceBits <= 0xffff; sourceBits += 1) {
    const bytes = Buffer.allocUnsafe(2);
    bytes.writeUInt16LE(sourceBits);
    assert.equal(decodeLiteralDenseElementF32(f16, bytes).decodedF32Bits, referenceF16ToF32Bits(sourceBits));
    assert.equal(decodeLiteralDenseElementF32(bf16, bytes).decodedF32Bits, (sourceBits << 16) >>> 0);
  }
  const altered = structuredClone(f16);
  altered.address.stridesElements[0] = 1;
  assert.throws(() => evaluateLiteralDenseElementAddress(altered, [1, 2]), /stride row-major inválido/);
  assert.throws(() => evaluateLiteralDenseElementAddress(f16, [2, 0]), /fora do eixo 0/);
});

test("Gemma linear scalar audits preserve fused products and spell out every ARM reduction region", () => {
  const schedule = {
    kind: "arm-neon-bf16-dot-fma",
    laneCount: 32,
    registerCount: 8,
    lanesPerRegister: 4,
    inputLane: "index-modulo-vector-lane-count",
    horizontalFold: "pairwise",
  } as const;
  assert.equal(
    gemma4LiteralScalarProductFormula(schedule, 7, "x[0,7]", "-0.125"),
    "product[7] = exact_product(x[0,7] * -0.125)",
  );
  const assignments = buildGemma4LiteralLinearReductionAssignments(schedule, 43, "F32");
  const transcript = assignments.join("\n");
  assert.match(transcript, /i=0\.\.31 ascending/);
  assert.match(transcript, /tree_04\[r,l\].*tree_02\[r,l\].*tree_01\[l\]/s);
  assert.match(transcript, /vector_tail\[b\+1,l\].*b=0\.\.0 ascending/s);
  assert.match(transcript, /product\[40\+j\].*j=0\.\.2 ascending/);
  assert.match(transcript, /reduced = tail_acc\[3\]/);
  assert.doesNotMatch(transcript, /weight\[|decode\(|REGISTER_TREE|VECTOR_TAIL|SCALAR_TAIL/);
  assert.equal(
    gemma4LiteralScalarProductFormula({ kind: "ordered-scalar", indexOrder: "ascending" }, 7, "x[0,7]", "-0.125"),
    "product[7] = F32(x[0,7] * -0.125)",
  );
});

test("Gemma normalization scalar audits expose complete cascade and Welford state transitions", () => {
  const cascade = buildGemma4LiteralCascadeSquareReductionAssignments({
    kind: "pytorch-cpu-f32-cascade-sum", vectorLanes: 4, ilpFactor: 4, cascadeLevels: 4,
    minimumLevelStep: 16, registerFold: "ascending", laneFold: "ascending",
  }, 256);
  const cascadeTranscript = cascade.join("\n");
  assert.match(cascadeTranscript, /unit_width = 16; unit_count = 16/);
  assert.match(cascadeTranscript, /cascade\[level,register,lane\] = F32\(0\)/);
  assert.match(cascadeTranscript, /cascade\[level-1,register,lane\]=F32\(0\)/);
  assert.match(cascadeTranscript, /register=1\.\.3 ascending/);
  assert.match(cascadeTranscript, /lane_acc\[lane\+1\].*lane=0\.\.3 ascending/);
  assert.equal(cascade.at(-1), "sum = lane_acc[4]");

  const welford = buildGemma4LiteralWelfordAssignments({
    kind: "pytorch-cpu-bf16-welford", inputVectorLanes: 8, accumulatorVectorLanes: 4,
    chunkVectors: 16, vectorMergeOrder: "low-then-high", laneFold: "ascending",
    secondPass: "x-times-scale-plus-bias-times-gamma",
  }, 32);
  const welfordTranscript = welford.join("\n");
  assert.match(welfordTranscript, /vector_count = 4; merged_lane_count = 8/);
  assert.match(welfordTranscript, /low_delta\[v,lane\].*low_remainder\[v,lane\]/s);
  assert.match(welfordTranscript, /high_delta\[v,lane\].*high_remainder\[v,lane\]/s);
  assert.match(welfordTranscript, /merged_m2\[lane\]/);
  assert.match(welfordTranscript, /fold_count\[lane\+1\]=fold_total\[lane\]/);
  assert.equal(welford.at(-1), "variance = F32(fold_m2[4]/F32(32))");
});

test("Gemma 4 composite trace dispatches every modality through one explicit feature/scatter contract", () => {
  assert.deepEqual(gemma4CompositeTraceProfile("image"), {
    modality: "image", tokenTypeId: 1, featureOutput: "image_features",
    featureOperationId: "composite_image_features", scatterOperationId: "composite_image_scatter",
  });
  assert.deepEqual(gemma4CompositeTraceProfile("video"), {
    modality: "video", tokenTypeId: 2, featureOutput: "video_features",
    featureOperationId: "composite_video_features", scatterOperationId: "composite_video_scatter",
  });
  assert.deepEqual(gemma4CompositeTraceProfile("audio"), {
    modality: "audio", tokenTypeId: 0, featureOutput: "audio_features",
    featureOperationId: "composite_audio_features", scatterOperationId: "composite_audio_scatter",
  });
});

test("Gemma 4 composite trace validates each modality shape and rejects mixed routing before source access", () => {
  const common = { source: "/absent", output: "/tmp/absent", positionIds: [0, 1], maxNewTokens: 1, python: "/python", model: "gemma4", revisionOrChecksum: "revision" };
  const visionValues = Array(9 * 768).fill(0);
  const positions = [Array.from({ length: 9 }, (_, index) => [index % 3, Math.floor(index / 3)])];
  assert.doesNotThrow(() => validateGemma4CompositeTraceOptions({
    ...common, modality: "image", inputTokens: [258880, 2], mmTokenTypeIds: [1, 0],
    pixelValues: { shape: [1, 9, 768], values: visionValues }, imagePositionIds: positions,
  }));
  assert.doesNotThrow(() => validateGemma4CompositeTraceOptions({
    ...common, modality: "video", inputTokens: [258884, 2], mmTokenTypeIds: [2, 0],
    pixelValuesVideos: { shape: [1, 1, 9, 768], values: visionValues }, videoPositionIds: [positions],
  }));
  assert.doesNotThrow(() => validateGemma4CompositeTraceOptions({
    ...common, modality: "audio", inputTokens: [258881, 2], mmTokenTypeIds: [0, 0],
    inputFeatures: { shape: [1, 1, 128], values: Array(128).fill(0) }, inputFeaturesMask: [[true]],
  }));
  assert.throws(() => validateGemma4CompositeTraceOptions({
    ...common, modality: "video", inputTokens: [258884, 2], mmTokenTypeIds: [1, 0],
    pixelValuesVideos: { shape: [1, 1, 9, 768], values: visionValues }, videoPositionIds: [positions],
  }), /mm_token_type_ids incompatível/);
  assert.throws(() => validateGemma4CompositeTraceOptions({
    ...common, modality: "audio", inputTokens: [258881, 2], mmTokenTypeIds: [0, 0],
    pixelValues: { shape: [1, 9, 768], values: visionValues }, imagePositionIds: positions,
    inputFeatures: { shape: [1, 1, 128], values: Array(128).fill(0) }, inputFeaturesMask: [[true]],
  }), /requer somente os inputs da modalidade declarada/);
});

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
  assert.deepEqual(program.assignments.map((assignment) => assignment.id).slice(0, 7), [
    "composite_block_sequence_ids", "composite_full_attention_mask", "composite_sliding_attention_mask",
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
  assert.deepEqual(result.values.get("composite_video_pixels")?.shape, [1, 4, 12]);
  assert.deepEqual(result.values.get("composite_video_position_ids")?.shape, [1, 4, 2]);
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
  }, fixtureSourceIdentity());
  assert.equal(literal.constants.length, catalog.tensors.size);
  assert.equal(literal.storageDecoders.length, catalog.tensors.size);
  assert.ok(literal.storageDecoders.every((decoder) =>
    decoder.address.kind === "row-major-dense-element-address" && decoder.address.schemaVersion === 2 &&
    decoder.address.logicalShape.length === decoder.address.stridesElements.length &&
    decoder.decode.schemaVersion === 2));
  assert.equal(JSON.stringify(literal).includes(catalog.source), false);
  assert.equal(literal.program.textProgram.source.path, "embedded://gemma4-composite-literal");
  assert.equal(literal.schemaVersion, 58);
  assert.equal(literal.payloadIntegrity.length, catalog.tensors.size);
  assert.equal(literal.integrityManifest.sections.length, 24);
  assert.deepEqual(literal.denseDecoderLanguage, buildLiteralDenseDecoderLanguageContract());
  assert.equal(literal.sourceIdentity.modelId, "fixture/tiny-gemma4");
  assert.equal(literal.authoritativeExecution.schemaVersion, 13);
  assert.equal(literal.authoritativeExecution.unresolvedNativeReduction.executableReplay.adapterProgram.sha256,
    GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256);
  assert.deepEqual(literal.authoritativeExecution.unresolvedNativeReduction.executableReplay.runtimeProcessEnvironment,
    GEMMA4_RUNTIME_REDUCTION_PROCESS_ENVIRONMENT);
  assert.deepEqual(literal.authoritativeExecution.unresolvedNativeReduction.executableReplay.runtimeEnvironmentIdentity,
    GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY);
  assert.deepEqual(literal.authoritativeExecution.unresolvedNativeReduction.executableReplay.runtimeExecutionState,
    GEMMA4_RUNTIME_REDUCTION_EXECUTION_STATE);
  assert.equal(literal.authoritativeExecution.unresolvedNativeReduction.executableReplay.executionProtocol.kind,
    "gemma4-runtime-reduction-execution-protocol");
  assert.equal(literal.authoritativeExecution.unresolvedNativeReduction.executableReplay.executionProtocol.schemaVersion, 4);
  assert.deepEqual(literal.authoritativeExecution.unresolvedNativeReduction.executableReplay.executionProtocol.transcriptCommitment, {
    schemaVersion: 1,
    encoding: "utf8",
    hashAlgorithm: "sha256",
    requestSource: "exact request-file bytes",
    responseSource: "exact stdout bytes",
    jsonSerialization: "compact JSON without whitespace or trailing bytes",
    receiptFields: ["requestBytes", "requestSha256", "responseBytes", "responseSha256"],
  });
  assert.deepEqual(literal.authoritativeExecution.unresolvedNativeReduction.executableReplay.executionProtocol.replayCommitment, {
    schemaVersion: 1,
    encoding: "utf8",
    hashAlgorithm: "sha256",
    executionOrder: "provider-append-order",
    receiptSerialization: "ECMAScript JSON.stringify",
    receiptSeparator: "LF after every receipt including the final receipt",
    committedSource: "complete execution receipts in execution order",
    evidenceFields: ["schemaVersion", "contractId", "executionCount", "executionOrder", "operationIds", "receiptCommitment", "executions"],
  });
  assert.deepEqual(literal.authoritativeExecution.unresolvedNativeReduction.executableReplay.executionProtocol.tensorEncoding, {
    kind: "gemma4-runtime-reduction-dense-tensor",
    schemaVersion: 1,
    fields: ["dtype", "bitPattern", "byteOrder", "layout", "shape", "byteLength", "dataBase64"],
    dtype: "F32",
    bitPattern: "IEEE-754 binary32",
    byteOrder: "little-endian",
    layout: "row-major-contiguous",
    indexToByteOffset: "4 * row-major-linear-index(shape, coordinate)",
    payloadEncoding: "RFC4648 canonical base64 with required padding",
    finiteValues: "required",
  });
  assert.match(literal.authoritativeExecution.unresolvedNativeReduction.executableReplay.adapterProgram.sourceUtf8,
    /Execute one pinned Gemma 4 runtime-defined matmul without model access/);
  const invocationPrograms = literal.authoritativeExecution.unresolvedNativeReduction.executableReplay.invocationPrograms;
  assert.equal(invocationPrograms.length, 5);
  assert.deepEqual(invocationPrograms.map((program) => program.id), literal.authoritativeExecution.unresolvedNativeReduction.operationClasses);
  assert.ok(invocationPrograms.every((program) => program.stages.some((stage) => stage.operation === "matmul") &&
    !Object.hasOwn(program, "transform")));
  assert.ok(invocationPrograms.every((program) =>
    program.environment.runtimeDtype.source === "program.runtimeDtype" &&
    program.environment.runtimeDtype.equals === "BF16" &&
    program.environment.towerParameters.every((binding) =>
      binding.source === `program.tower.${binding.name}` && binding.numericDomain === "safe-integer")));
  assert.deepEqual(
    invocationPrograms.find((program) => program.id === "vision-attention-score")!.environment.towerParameters,
    ["attentionHeads", "headDim"].map((name) => ({
      name, source: `program.tower.${name}`, numericDomain: "safe-integer", minimumInclusive: 1,
    })),
  );
  assert.equal(
    invocationPrograms.find((program) => program.id === "audio-attention-value")!.environment.towerParameters
      .find((binding) => binding.name === "attentionContextRight")!.minimumInclusive,
    0,
  );
  assert.equal(literal.fidelityGate.status, "blocked-on-runtime-reduction");
  assert.equal(literal.fidelityGate.schemaVersion, 4);
  assert.equal(literal.fidelityGate.exactReplayClaim, "forbidden");
  assert.equal(literal.fidelityGate.unresolvedNativeReductionCount, literal.fidelityGate.unresolvedNativeReductions.length);
  assert.ok(literal.fidelityGate.unresolvedNativeReductions.every((entry) =>
    entry.scalarCalculationPointer === `/calculationGraph/assignments/${entry.ordinal}/scalarCalculation/reduction` &&
    entry.outputCoordinatePointer === `/calculationGraph/assignments/${entry.ordinal}/outputCoordinate` &&
    entry.invocationProgramId === entry.operationClass &&
    entry.invocationProgramPointer === `/authoritativeExecution/unresolvedNativeReduction/executableReplay/invocationPrograms/${invocationPrograms.findIndex((program) => program.id === entry.operationClass)}` &&
    entry.executionProtocolPointer === "/authoritativeExecution/unresolvedNativeReduction/executableReplay/executionProtocol" &&
    entry.runtimeProcessEnvironmentPointer === "/authoritativeExecution/unresolvedNativeReduction/executableReplay/runtimeProcessEnvironment" &&
    entry.runtimeEnvironmentIdentityPointer === "/authoritativeExecution/unresolvedNativeReduction/executableReplay/runtimeEnvironmentIdentity" &&
    entry.runtimeExecutionStatePointer === "/authoritativeExecution/unresolvedNativeReduction/executableReplay/runtimeExecutionState"));
  assert.equal(literal.formulaLanguage.languageId, literal.scalarCalculations.formulaLanguage);
  assert.equal(literal.formulaLanguage.languageId, literal.generation.scalarCalculations.formulaLanguage);
  assert.equal(literal.formulaLanguage.authority.numericLiteralBits, "/numericLiterals/literals");
  assert.equal(literal.formulaLanguage.schemaVersion, 25);
  assert.equal(literal.formulaLanguage.authority.generationControlProgram, "/generation/controlProgram");
  assert.equal(literal.formulaLanguage.authority.forwardControlProgram, "/forwardControl");
  assert.equal(literal.formulaLanguage.authority.inputContract, "/inputContract");
  assert.equal(literal.formulaLanguage.authority.outputContract, "/outputContract");
  assert.equal(literal.inputContract.kind, "gemma4-literal-input-contract");
  assert.equal(literal.inputContract.vision.pixelFeatures, 12);
  assert.equal(literal.inputContract.audio.projectionInputFeatures, 4);
  assert.deepEqual(literal.inputContract.text.cacheProducers.map((entry) => entry.layer), [0, 1]);
  assert.equal(literal.outputContract.kind, "gemma4-literal-output-contract");
  assert.deepEqual(literal.outputContract.forward.tensors.map((entry) => [entry.role, entry.name]), [
    ["embeddings", "hidden_states_0"],
    ["per-layer-inputs", "ple_inputs"],
    ["logits", literal.outputs.logits],
  ]);
  assert.deepEqual(literal.outputContract.forward.pastKeyValues.producers.map((entry) => entry.layer), [0, 1]);
  assert.equal(literal.outputContract.generation.terminal.pastKeyValues, "forward_state[executed_steps].past_key_values");
  const validInputContractRequest = {
    inputIds: [[1, 99, 97, 98, 2]],
    mmTokenTypeIds: [[0, 1, 2, 3, 0]],
    pixelValues: patterned([1, 4, 12]),
    imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]],
    pixelValuesVideos: patterned([1, 1, 4, 12]),
    videoPositionIds: [[[[0, 0], [1, 0], [0, 1], [1, 1]]]],
    inputFeatures: patterned([1, 4, 16]),
    inputFeaturesMask: [[true, true, true, true]],
  };
  assert.doesNotThrow(() => executeGemma4LiteralInputContract(literal.inputContract, literal.program, validInputContractRequest));
  assert.throws(() => executeGemma4LiteralInputContract(literal.inputContract, literal.program, {
    ...validInputContractRequest,
    inputIds: [[1, 99, 99, 97, 98, 2]],
    mmTokenTypeIds: [[0, 1, 1, 2, 3, 0]],
  }), /image placeholder count=2 não corresponde a features=1/);
  assert.throws(() => executeGemma4LiteralInputContract(literal.inputContract, literal.program, {
    inputIds: [[1, 98, 2]],
    inputFeatures: patterned([1, 4, 9]),
    inputFeaturesMask: [[true, true, true, true]],
  }), /largura de áudio incompatível/);
  assert.equal(literal.forwardControl.kind, "gemma4-literal-forward-control-program");
  const imageSelection = executeGemma4LiteralForwardControlProgram(literal.forwardControl, literal.program, new Set([
    "input_ids", "pixel_values", "image_position_ids", "mm_token_type_ids",
  ]));
  assert.deepEqual(imageSelection.modalities, { image: true, video: false, audio: false });
  assert.equal(imageSelection.visionMasks, true);
  assert.equal(imageSelection.attentionMaskMode, "serialized-vision-block-masks");
  assert.equal(imageSelection.positionIdsMode, "sequence-index-default");
  assert.equal(imageSelection.pastKeyValuesMode, "empty-cache");
  assert.deepEqual(imageSelection.inactiveIdentityAssignments.map((entry) => entry.output), [
    "composite_embeddings_after_video", "hidden_states_0",
  ]);
  assert.ok(imageSelection.activeTopLevelOperationIds.includes("composite_image_features"));
  assert.ok(!imageSelection.activeTopLevelOperationIds.includes("composite_audio_features"));
  assert.throws(() => executeGemma4LiteralForwardControlProgram(literal.forwardControl, literal.program, new Set([
    "input_ids", "pixel_values",
  ])), /pixel_values e image_position_ids juntos/);
  assert.throws(() => executeGemma4LiteralForwardControlProgram(literal.forwardControl, literal.program, new Set([
    "input_ids", "mm_token_type_ids", "attention_mask",
  ])), /não combina mm_token_type_ids com attention_mask/);
  assert.equal(literal.scalarCalculations.schemaVersion, 6);
  assert.equal(literal.formulaLanguage.authority.forwardScalarExecution, "/calculationGraph/assignments/*/scalarCalculation/scalarAssignments");
  assert.equal(literal.formulaLanguage.authority.forwardScalarDataflow, "/calculationGraph/assignments/*/scalarCalculation/statementDataflow");
  assert.equal(literal.formulaLanguage.authority.forwardScalarPrograms, "/calculationGraph/assignments/*/scalarCalculation/statementPrograms");
  assert.equal(literal.formulaLanguage.authority.forwardScalarEnvironment, "/calculationGraph/assignments/*/scalarCalculation/statementEnvironment");
  assert.equal(literal.formulaLanguage.authority.outputCoordinateWrite, "/calculationGraph/assignments/*/outputCoordinate/write");
  assert.equal(literal.formulaLanguage.authority.coordinateExpressionLanguage, "/calculationGraph/coordinateLanguage");
  assert.equal(literal.formulaLanguage.authority.predecessorCoordinateAccesses, "/calculationGraph/assignments/*/predecessors/*/accesses");
  assert.equal(literal.formulaLanguage.authority.consumerCoordinateAccesses, "/calculationGraph/assignments/*/consumerCoordinates/*/accesses");
  assert.ok(literal.scalarCalculations.assignments.every((assignment) =>
    assignment.scalarAssignments.at(-1)?.startsWith(`${assignment.output}[`)));
  const orderedVisionRope = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "vision" && entry.definitionId === "vision_layer_0_q_rope")!;
  assert.deepEqual(orderedVisionRope.scalarAssignments.map((statement) => statement.split("=")[0]?.trim()), [
    "axis", "local_feature", "pair", "paired_feature", "angle", "cosine", "sine",
    "vision_layer_0_q_rotated[batch,head,patch,head_feature]",
  ]);
  const orderedTextAttention = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "text-layer" && entry.definitionId === "layer_0_attention")!;
  assert.deepEqual(orderedTextAttention.scalarAssignments.map((statement) => statement.split("=")[0]?.trim()), [
    "dot[key]", "scaled_dot[key]", "score[key]", "maximum", "exponential[key]", "total", "probability[key]",
    "layer_0_attention_context[batch,sequence,attention_hidden]",
  ]);
  assert.ok(literal.scalarCalculations.assignments.every((assignment) =>
    assignment.statementDataflow.length === assignment.scalarAssignments.length &&
    assignment.statementDataflow.every((statement, ordinal) => statement.ordinal === ordinal)));
  assert.ok(literal.scalarCalculations.assignments.every((assignment) =>
    assignment.statementPrograms.length === assignment.scalarAssignments.length &&
    assignment.statementPrograms.every((statement, ordinal) =>
      statement.ordinal === ordinal && statement.source === assignment.scalarAssignments[ordinal])));
  assert.ok(literal.scalarCalculations.assignments.every((assignment) =>
    assignment.statementEnvironment.schemaVersion === 1 &&
    assignment.statementEnvironment.orderedInputs.map((entry) => entry.name).join("\0") === assignment.orderedInputs.join("\0")));
  const attentionDataflow = orderedTextAttention.statementDataflow;
  assert.deepEqual(attentionDataflow[3]?.reads.map((read) => [read.expression, read.producerStatementOrdinal]), [
    ["score[key]", 2],
  ]);
  assert.deepEqual(attentionDataflow[4]?.reads.map((read) => [read.expression, read.producerStatementOrdinal]), [
    ["score[key]", 2], ["maximum", 3],
  ]);
  assert.deepEqual(attentionDataflow[2]?.consumerStatementOrdinals, [3, 4]);
  assert.deepEqual(attentionDataflow[6]?.consumerStatementOrdinals, [7]);
  assert.deepEqual(attentionDataflow[7]?.writes[0], {
    kind: "indexed", name: "layer_0_attention_context",
    expression: "layer_0_attention_context[batch,sequence,attention_hidden]",
    coordinates: ["batch", "sequence", "attention_hidden"],
    coordinatePrograms: [
      { kind: "symbol", name: "batch" },
      { kind: "symbol", name: "sequence" },
      { kind: "symbol", name: "attention_hidden" },
    ],
    role: "output",
  });
  const ropeDataflow = orderedVisionRope.statementDataflow;
  assert.deepEqual(ropeDataflow[3]?.reads.map((read) => [read.expression, read.producerStatementOrdinal]), [
    ["axis", 0], ["local_feature", 1],
  ]);
  assert.ok(ropeDataflow.at(-1)?.reads.some((read) =>
    read.expression === "paired_feature" && read.producerStatementOrdinal === 3));
  assert.deepEqual(literal.formulaLanguage.reductions.domainLanguage, gemma4LiteralReductionDomainLanguage());
  assert.match(literal.formulaLanguage.evaluation.reductionBinding, /call-site tensor binding/);
  assert.match(literal.formulaLanguage.evaluation.operandClosure, /every source tensor read names one ordered input/);
  assert.match(literal.formulaLanguage.evaluation.dimensionBinding, /safe-integer dimension language/);
  assert.equal(literal.formulaLanguage.indexing.programs.schemaVersion, 3);
  assert.ok(literal.numericLiterals.literals.some((entry) =>
    entry.uses.some((use) => use.section === "cache-transition" && use.definitionId === "layer_0_incremental")));
  assert.deepEqual(
    [-1, 0, 0, -1, 1, 1, -1, 2],
    [0, 1, 2, 0, 2, 1, 0, 1].map((_, sequence) => executeGemma4LiteralContiguousVisionGroupId(
      literal.formulaLanguage.indexing.programs,
      [0, 1, 2, 0, 2, 1, 0, 1],
      sequence,
    )),
  );
  const alteredIndexingPrograms = structuredClone(gemma4LiteralIndexingPrograms());
  alteredIndexingPrograms.contiguousVisionGroupId[1] = "group=I32(0)";
  assert.throws(() => executeGemma4LiteralContiguousVisionGroupId(alteredIndexingPrograms, [1], 0), /ausente ou alterado/);
  assert.throws(() => executeGemma4LiteralContiguousVisionGroupId(gemma4LiteralIndexingPrograms(), [1], 1), /coordenada válida/);
  const stableMask = [[false, true, true], [true, false, true]];
  assert.equal(executeGemma4LiteralStableTrueCount(literal.formulaLanguage.indexing.programs, stableMask), 4);
  assert.equal(executeGemma4LiteralStableTruePrefixRank(literal.formulaLanguage.indexing.programs, stableMask, 0, 1), 0);
  assert.equal(executeGemma4LiteralStableTruePrefixRank(literal.formulaLanguage.indexing.programs, stableMask, 1, 2), 3);
  assert.equal(executeGemma4LiteralStableTruePrefixRank(literal.formulaLanguage.indexing.programs, stableMask, 1, 1), -1);
  assert.deepEqual(executeGemma4LiteralStableTrueCoordinateAtRank(literal.formulaLanguage.indexing.programs, stableMask, 2), [1, 0]);
  assert.throws(() => executeGemma4LiteralStableTrueCoordinateAtRank(literal.formulaLanguage.indexing.programs, stableMask, 4), /rank válido/);
  const alteredStablePrograms = structuredClone(gemma4LiteralIndexingPrograms());
  alteredStablePrograms.stableTruePrefixRank[1] = "rank=I32(1)";
  assert.throws(() => executeGemma4LiteralStableTruePrefixRank(alteredStablePrograms, stableMask, 0, 1), /ausente ou alterado/);
  const poolPositions = [[0, 0], [1, 0], [2, 0], [3, 0], [-1, -1]] as const;
  assert.deepEqual(poolPositions.map((_, patch) => executeGemma4LiteralVisionPoolSlot(
    literal.formulaLanguage.indexing.programs, poolPositions, patch, 2, 2,
  )), [0, 0, 1, 1, -1]);
  assert.equal(executeGemma4LiteralVisionPoolCellHasPatch(literal.formulaLanguage.indexing.programs, poolPositions, 0, 2, 2), true);
  assert.equal(executeGemma4LiteralVisionPoolCellHasPatch(literal.formulaLanguage.indexing.programs, poolPositions, 1, 2, 2), true);
  assert.throws(() => executeGemma4LiteralVisionPoolSlot(literal.formulaLanguage.indexing.programs, [[-1, -1]], 0, 2, 1), /ao menos um patch válido/);
  assert.deepEqual(executeGemma4LiteralAudioRelativeShiftSource(literal.formulaLanguage.indexing.programs, 0, 2, 4, 3), {
    valid: true, queryInBlock: 0, relativeIndex: 2,
  });
  assert.deepEqual(executeGemma4LiteralAudioRelativeShiftSource(literal.formulaLanguage.indexing.programs, 0, 3, 4, 3), {
    valid: false, queryInBlock: 0, relativeIndex: -1,
  });
  const alteredCoordinatePrograms = structuredClone(gemma4LiteralIndexingPrograms());
  alteredCoordinatePrograms.visionPoolSlot[1] = "result=I32(0)";
  assert.throws(() => executeGemma4LiteralVisionPoolSlot(alteredCoordinatePrograms, poolPositions, 0, 2, 2), /ausente ou alterado/);
  assert.throws(() => validateGemma4LiteralFormulaFunctionCoverage(["y=hidden_runtime_helper(x)"]), /helper opaco/);
  assert.equal(literal.formulaLanguage.authority.transcendentalPrograms, "/transcendentalPrograms");
  assert.deepEqual(literal.transcendentalPrograms, buildGemma4LiteralTranscendentalPrograms());
  assert.ok(literal.formulaLanguage.reductions.normalizationPrograms.pytorchCpuF32CascadeSum.length >= 4);
  assert.ok(literal.formulaLanguage.reductions.normalizationPrograms.pytorchCpuBf16Welford.length >= 3);
  assert.deepEqual(literal.formulaLanguage.reductions.softmaxPrograms, gemma4LiteralSoftmaxReductionPrograms());
  assert.equal(executeGemma4LiteralSoftmaxReductionProgram(
    literal.formulaLanguage.reductions.softmaxPrograms,
    "ORDERED_F32_REDUCE_SUM",
    [1, 2, 3, 4],
  ), 10);
  assert.equal(executeGemma4LiteralSoftmaxReductionProgram(
    literal.formulaLanguage.reductions.softmaxPrograms,
    "PYTORCH_F32_VECTOR_REDUCE_SUM",
    [1e20, 1, -1e20, 1, -1e20, 1, 1e20, 1, 3],
  ), 7);
  assert.throws(() => executeGemma4LiteralSoftmaxReductionProgram(
    literal.formulaLanguage.reductions.softmaxPrograms,
    "PYTORCH_F32_VECTOR_REDUCE_SUM",
    [1, 2, 3, 4],
    [true, false, true, true],
  ), /não aceita compactação implícita/);
  assert.match(literal.formulaLanguage.reductions.runtimeDefined, /not executable/);
  assert.ok(literal.formulaLanguage.intrinsics.some((intrinsic) => intrinsic.notation === "decode(role)[indices]"));
  assert.ok(literal.formulaLanguage.intrinsics.some((intrinsic) => intrinsic.notation === "exact_product(a*b)"));
  const expectedDomainCount = literal.assignments.composite.length + literal.assignments.vision.length + literal.assignments.audio.length +
    literal.program.textProgram.prelude.length + literal.program.textProgram.layers.reduce((total, layer) => total + layer.operations.length, 0) + literal.assignments.textEpilogue.length;
  assert.equal(literal.calculationDomains.assignments.length, expectedDomainCount);
  assert.equal(literal.calculationDomains.schemaVersion, 2);
  assert.equal(literal.calculationDomains.dimensionLanguage.id, "gemma4-safe-integer-dimension-expression-v1");
  const dimensionValues = evaluateGemma4LiteralDimensionPrograms(literal.calculationDomains.dimensionPrograms, {
    tensorShapes: {
      input_ids: [2, 3], pixel_values: [2, 8, 12],
      pixel_values_videos: [1, 2, 8, 12], input_features: [2, 5, 7],
    },
    booleanTensors: {
      vision_pool_mask: [true, false, true, true],
      "composite_image_features/vision_pool_mask": [true, true, false, true],
      "composite_video_features/vision_pool_mask": [true, false, true, false],
      audio_output_mask: [true, false, true],
    },
    cacheKeyLengths: [5, 5],
  });
  assert.deepEqual({
    B: dimensionValues.B, S: dimensionValues.S, K: dimensionValues.K,
    VB: dimensionValues.VB, VP: dimensionValues.VP, VPOOL: dimensionValues.VPOOL, VVALID: dimensionValues.VVALID,
    AB: dimensionValues.AB, AT1: dimensionValues.AT1, AF1: dimensionValues.AF1,
    AT2: dimensionValues.AT2, AF2: dimensionValues.AF2, ABLOCKS: dimensionValues.ABLOCKS, AVALID: dimensionValues.AVALID,
  }, { B: 2, S: 3, K: 8, VB: 2, VP: 8, VPOOL: 2, VVALID: 3, AB: 2, AT1: 3, AF1: 4, AT2: 2, AF2: 2, ABLOCKS: 1, AVALID: 2 });
  assert.equal(evaluateGemma4LiteralDimensionProgram("K", literal.calculationDomains.dimensionPrograms, {
    tensorShapes: { input_ids: [1, 3] },
  }), 3);
  assert.throws(() => evaluateGemma4LiteralDimensionProgram("K", literal.calculationDomains.dimensionPrograms, {
    tensorShapes: { input_ids: [1, 3] }, cacheKeyLengths: [5, 6],
  }), /comprimentos de key contraditórios/);
  assert.equal(evaluateGemma4LiteralShapeExpression("VIDEO_BATCH*VIDEO_FRAMES", dimensionValues), 2);
  assert.throws(() => evaluateGemma4LiteralShapeExpression("ceil(VP/4)", dimensionValues), /shape Gemma 4 inválida/);
  assert.throws(() => evaluateGemma4LiteralShapeExpression("9007199254740991*2", dimensionValues), /inteiro seguro/);
  assert.throws(() => evaluateGemma4LiteralDimensionProgram("VPOOL", literal.calculationDomains.dimensionPrograms, {
    tensorShapes: { pixel_values: [1, 7, 12] },
  }), /não é exata/);
  assert.throws(() => evaluateGemma4LiteralDimensionProgram("AVALID", literal.calculationDomains.dimensionPrograms, {
    tensorShapes: {},
  }), /tensor BOOL ausente/);
  assert.ok(literal.calculationDomains.assignments.every((entry) =>
    entry.domain.shape.length > 0 && entry.domain.axes.length === entry.domain.shape.length && entry.domain.layout.length > 0 &&
    entry.domain.dtype.length > 0 && entry.domain.dtypePolicy?.outputDtype === entry.domain.dtype));
  assert.deepEqual(literal.calculationDomains.assignments.find((entry) => entry.definitionId === "token_embedding")?.domain.shape, ["B", "S", "4"]);
  assert.deepEqual(literal.calculationDomains.assignments.find((entry) => entry.definitionId === "vision_layer_0_attention_scores")?.domain.shape, ["VB", "1", "VP", "VP"]);
  assert.deepEqual(literal.calculationDomains.assignments.find((entry) => entry.definitionId === "audio_layer_0_attention_content_scores")?.domain.shape, ["AB", "1", "ABLOCKS", "2", "2"]);
  assert.equal(literal.scalarCalculations.assignments.length, expectedDomainCount);
  const serializedFormulas = literal.scalarCalculations.assignments.map((assignment) => assignment.formula);
  validateGemma4LiteralFormulaFunctionCoverage(serializedFormulas);
  assert.equal(serializedFormulas.filter((formula) => /\b(?:theta_power|rotate_rotate_half|is_declared_modal_token|contiguous_group_id)\s*\(|same_nonnegative_vision_block/.test(formula)).length, 0);
  const proportionalRopeFormula = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "text-layer" && entry.definitionId === "layer_0_q_rope")!.formula;
  assert.match(proportionalRopeFormula, /active_pairs=floor\(F64\(1\)\*F64\(4\)\/F64\(2\)\)/);
  assert.match(proportionalRopeFormula, /paired_feature=head_feature<2 \? head_feature\+2 : head_feature-2/);
  assert.match(proportionalRopeFormula, /pair>=active_pairs \? F32\(0\)/);
  const modalReplacementFormula = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "composite" && entry.definitionId === "composite_pad_substitution")!.formula;
  assert.match(modalReplacementFormula, /input_ids\[batch,sequence\]==99 \|\| input_ids\[batch,sequence\]==97 \|\| input_ids\[batch,sequence\]==98/);
  const slidingMaskFormula = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "composite" && entry.definitionId === "composite_sliding_attention_mask")!.formula;
  assert.match(slidingMaskFormula, /vision_block_sequence_ids\[batch,query\]>=0 && vision_block_sequence_ids\[batch,query\]==vision_block_sequence_ids\[batch,key\]/);
  const positionEmbeddingFormula = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "vision" && entry.definitionId === "vision_position_embedding")!.formula;
  assert.match(positionEmbeddingFormula, /pixel_position_ids\[batch,patch,0\]==-1/);
  assert.doesNotMatch(positionEmbeddingFormula, /padding pair|\[0,x,hidden\]/);
  const scatterFormulas = literal.scalarCalculations.assignments.filter((entry) =>
    entry.operation === "masked-scatter" || entry.operation === "masked-scatter-image-features" || entry.operation === "masked-scatter-audio-features");
  assert.equal(scatterFormulas.length, 5);
  assert.ok(scatterFormulas.every((entry) => entry.formula.includes("STABLE_TRUE_PREFIX_RANK") && entry.formula.includes("STABLE_TRUE_COUNT")));
  assert.match(scatterFormulas.find((entry) => entry.definitionId === "composite_image_scatter")!.formula, /input_ids==99/);
  assert.match(scatterFormulas.find((entry) => entry.definitionId === "composite_video_scatter")!.formula, /input_ids==97/);
  assert.match(scatterFormulas.find((entry) => entry.definitionId === "composite_audio_scatter")!.formula, /input_ids==98/);
  const stripFormulas = literal.scalarCalculations.assignments.filter((entry) => entry.operation === "strip-padding");
  assert.equal(stripFormulas.length, 2);
  assert.ok(stripFormulas.every((entry) => entry.formula.includes("STABLE_TRUE_COORDINATE_AT_RANK") && entry.formula.includes("STABLE_TRUE_COUNT")));
  assert.equal(serializedFormulas.some((formula) => /next_feature_row|placeholder_at|stable_batch_major_true_mask_row/.test(formula)), false);
  const visionPoolFormula = literal.scalarCalculations.assignments.find((entry) => entry.definitionId === "vision_pool")!.formula;
  const visionPoolMaskFormula = literal.scalarCalculations.assignments.find((entry) => entry.definitionId === "vision_pool_mask")!.formula;
  assert.match(visionPoolFormula, /VISION_POOL_SLOT/);
  assert.match(visionPoolFormula, /F32_FMA/);
  assert.match(visionPoolMaskFormula, /VISION_POOL_CELL_HAS_PATCH/);
  const audioCoordinateFormulas = literal.scalarCalculations.assignments.filter((entry) => entry.scope === "audio" &&
    ["chunked-attention-content-matmul", "relative-attention-position-matmul", "relative-attention-shift", "chunked-attention-mask", "chunked-relative-attention-values"].includes(entry.operation));
  assert.equal(audioCoordinateFormulas.length, 5);
  assert.ok(audioCoordinateFormulas.every((entry) => /query_index=|AUDIO_RELATIVE_SHIFT_SOURCE/.test(entry.formula)));
  assert.match(audioCoordinateFormulas.find((entry) => entry.operation === "relative-attention-shift")!.formula, /AUDIO_RELATIVE_SHIFT_SOURCE/);
  assert.equal(serializedFormulas.some((formula) => /mapped to pool_cell|any non-padding patch|after declared pad-flatten-slice-reshape|eager_additive_mask_entry_is_zero_or_block_padding/.test(formula)), false);
  const missingScatterToken = structuredClone(literal.program);
  delete missingScatterToken.assignments.find((entry) => entry.id === "composite_image_scatter")!.placeholderTokenId;
  assert.throws(() => buildGemma4LiteralScalarCalculations(missingScatterToken), /placeholderTokenId autoritativo/);
  assert.ok(literal.scalarCalculations.assignments.every((entry) =>
    entry.outputCoordinates.length > 0 && entry.formula.startsWith(`${entry.output}[`) && entry.orderedInputs.length > 0));
  assert.ok(literal.scalarCalculations.assignments.every((entry) =>
    !/masked_score-max_key|score-max_key|score-max_valid_key|max_k|max_context|sum_k_ascending|sum_context_ascending/.test(entry.formula)));
  const operandClosedOperations = new Set([
    "embedding", "per-layer-embedding", "per_layer_embedding",
    "linear", "clipped-linear", "rms-norm", "rms_norm", "activation", "gelu-tanh",
    "elementwise", "tensor_scale",
    "multidimensional-rope", "attention-score-matmul", "attention-value-matmul",
    "reshape-conv-features", "conv2d-stride2", "layer-norm-channels",
    "split-gated-linear-unit", "causal-depthwise-convolution",
  ]);
  const operandClosed = literal.scalarCalculations.assignments.filter((entry) => operandClosedOperations.has(entry.operation));
  assert.ok(operandClosed.length > 20);
  assert.ok(operandClosed.every((entry) => entry.orderedInputs.every((input) => entry.formula.includes(input))));
  assert.equal(operandClosed.some((entry) => /\+\/-|\bfollows\b|\btuple\s*\(|\.\.\.|\bpadded_input\[/.test(entry.formula)), false);
  assert.equal(operandClosed.filter((entry) => entry.operation === "rms_norm" || entry.operation === "rms-norm")
    .some((entry) => /\binput\b/.test(entry.formula)), false);
  const visionRopes = operandClosed.filter((entry) => entry.operation === "multidimensional-rope");
  assert.equal(visionRopes.length, 2);
  assert.ok(visionRopes.every((entry) =>
    entry.formula.includes("local_feature<") && entry.formula.includes("paired_feature=") &&
    entry.formula.includes(" ? BF16(F32(") && entry.formula.includes(" : BF16(F32(")));
  const visionAttentionCalculation = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "vision" && entry.definitionId === "vision_layer_0_attention_scores")!;
  assert.match(visionAttentionCalculation.formula, /head_feature=0\.\.3/);
  assert.ok(visionAttentionCalculation.orderedInputs.every((input) => visionAttentionCalculation.formula.includes(input)));
  assert.equal(visionAttentionCalculation.reproducibility, "literal");
  assert.equal(visionAttentionCalculation.reduction?.order, "operation-declared");
  assert.deepEqual(visionAttentionCalculation.reduction?.domains, [{
    index: "head_feature", startInclusive: 0,
    endExclusive: { kind: "constant", value: program.visionProgram.tower.headDim }, order: "ascending",
  }]);
  assert.deepEqual(evaluateGemma4LiteralReductionIndexDomains(
    literal.formulaLanguage.reductions.domainLanguage,
    visionAttentionCalculation.reduction!.domains,
    { vision_layer_0_q_rotated: [1, 1, 2, 4] },
  ), [{ index: "head_feature", startInclusive: 0, endExclusive: 4, order: "ascending" }]);
  const visionSoftmaxCalculation = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "vision" && entry.definitionId === "vision_layer_0_attention_weights")!;
  assert.deepEqual(visionSoftmaxCalculation.reductionStages?.map((stage) => [stage.id, stage.program]), [
    ["softmax-maximum", "ORDERED_F32_REDUCE_MAX"],
    ["softmax-exponential-sum", "ORDERED_F32_REDUCE_SUM"],
  ]);
  assert.ok(visionSoftmaxCalculation.reductionStages?.every((stage) =>
    stage.domains[0]?.endExclusive.kind === "tensor-axis" &&
    stage.domains[0].endExclusive.tensor === "vision_layer_0_attention_scores" &&
    stage.domains[0].endExclusive.axis === 3));
  assert.match(visionSoftmaxCalculation.formula, /ORDERED_F32_REDUCE_MAX/);
  const audioSoftmaxCalculation = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "audio" && entry.definitionId === "audio_layer_0_attention_softmax")!;
  assert.deepEqual(audioSoftmaxCalculation.reductionStages?.map((stage) => stage.id), ["softmax-maximum", "softmax-exponential-sum"]);
  assert.ok(audioSoftmaxCalculation.reductionStages?.every((stage) =>
    stage.domains[0]?.endExclusive.kind === "tensor-axis" && stage.domains[0].endExclusive.axis === 4));
  const f32TextAttentionCalculation = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "text-layer" && entry.definitionId === "layer_0_attention")!;
  assert.deepEqual(f32TextAttentionCalculation.reductionStages?.map((stage) => stage.program), [
    "ORDERED_F32_DOT", "ORDERED_F32_REDUCE_MAX", "ORDERED_F32_REDUCE_SUM", "ORDERED_F32_DOT",
  ]);
  const nativeProgram = structuredClone(program);
  const nativeAttention = nativeProgram.textProgram.layers[0]!.operations.find((operation) => operation.op === "scaled_dot_product_attention");
  if (!nativeAttention || nativeAttention.op !== "scaled_dot_product_attention") throw new Error("fixture requires text attention");
  nativeAttention.numericImplementation = structuredClone(GEMMA4_E4B_PYTORCH_BF16_ATTENTION_IMPLEMENTATION);
  const nativeAttentionCalculation = buildGemma4LiteralScalarCalculations(nativeProgram).assignments.find((entry) =>
    entry.scope === "text-layer" && entry.definitionId === nativeAttention.id)!;
  assert.deepEqual(nativeAttentionCalculation.reductionStages?.map((stage) => stage.program), [
    "ARM_NEON_BF16_DOT_F32", "PYTORCH_F32_VECTOR_REDUCE_MAX", "PYTORCH_F32_VECTOR_REDUCE_SUM", "ARM_NEON_BF16_DOT_F32",
  ]);
  assert.match(nativeAttentionCalculation.formula, /reductionStages\[score-dot\]\.schedule/);
  assert.deepEqual(nativeAttentionCalculation.reductionStages?.map((stage) => stage.domains[0]?.endExclusive), [
    { kind: "tensor-axis", tensor: "layer_0_q_rot", axis: 3 },
    { kind: "tensor-axis", tensor: "attention_mask:full_attention", axis: 3 },
    { kind: "tensor-axis", tensor: "attention_mask:full_attention", axis: 3 },
    { kind: "tensor-axis", tensor: "attention_mask:full_attention", axis: 3 },
  ]);
  const audioConvCalculation = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "audio" && entry.definitionId === "audio_subsample_0_conv")!;
  assert.deepEqual(audioConvCalculation.learnedOperandRoles, ["convolution-kernel"]);
  assert.match(audioConvCalculation.formula, /input_channel=0\.\.channels-1,kernel_time=0\.\.2,kernel_feature=0\.\.2/);
  assert.equal(audioConvCalculation.reproducibility, "literal");
  assert.deepEqual(audioConvCalculation.reduction?.domains.map((domain) => domain.endExclusive), [
    { kind: "tensor-axis", tensor: "audio_masked_features_4d", axis: 1 },
    { kind: "constant", value: 3 },
    { kind: "constant", value: 3 },
  ]);
  const audioRmsCalculation = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "audio" && entry.definitionId === "audio_layer_0_ffn1_pre_norm")!;
  assert.match(audioRmsCalculation.formula, /\*decode\(normalization-scale\)\[output_feature\]/);
  assert.doesNotMatch(audioRmsCalculation.formula, /1\+decode\(normalization-scale\)/);
  const audioGluCalculation = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "audio" && entry.definitionId === "audio_layer_0_conv_glu")!;
  assert.match(audioGluCalculation.formula, /audio_layer_0_conv_glu_linear\[batch,frame,hidden\]\/F32\(1\+SLEEF_EXP_F32/);
  const compositeEmbeddingCalculation = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "composite" && entry.definitionId === "composite_text_embedding")!;
  assert.match(compositeEmbeddingCalculation.formula, /decode\(weight\).*F32\(2\)/);
  const textLinearCalculation = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "text-layer" && entry.definitionId === "layer_0_q_proj")!;
  assert.deepEqual(textLinearCalculation.learnedOperandRoles, ["weight"]);
  assert.equal(textLinearCalculation.reduction?.schedule?.kind, "ordered-scalar");
  const audioProjectionOperands = literal.learnedOperands.assignments.find((entry) =>
    entry.scope === "audio" && entry.definitionId === "audio_output_projection")?.operands;
  assert.deepEqual(audioProjectionOperands?.map((operand) => [operand.role, operand.logicalIndices]), [
    ["weight", [
      { kind: "output-coordinate", axis: "output_feature" },
      { kind: "reduction-index", name: "input_feature", minInclusive: 0, endExclusive: 4 },
    ]],
    ["bias", [{ kind: "output-coordinate", axis: "output_feature" }]],
  ]);
  assert.equal(literal.learnedOperands.schemaVersion, 2);
  assert.equal(literal.learnedOperands.indexLanguage.id, "gemma4-learned-index-expression-v1");
  assert.deepEqual(evaluateGemma4LiteralLearnedOperandIndices(audioProjectionOperands![0]!, {
    outputCoordinates: { output_feature: 2 }, reductionIndices: { input_feature: 3 },
  }), [2, 3]);
  assert.throws(() => evaluateGemma4LiteralLearnedOperandIndices(audioProjectionOperands![0]!, {
    outputCoordinates: { output_feature: 2 },
  }), /binding de índice aprendido ausente: input_feature/);
  assert.throws(() => evaluateGemma4LiteralLearnedOperandIndices(audioProjectionOperands![0]!, {
    outputCoordinates: { output_feature: 2 }, reductionIndices: { input_feature: 4 },
  }), /reduction-index: índice 4 fora de 0..3/);
  assert.equal(audioProjectionOperands?.[0]?.decoderId, `decode_${audioProjectionOperands[0]?.tensor.name}`);
  const visionClipOperands = literal.learnedOperands.assignments.find((entry) =>
    entry.scope === "vision" && entry.definitionId === "vision_layer_0_q")?.operands;
  assert.deepEqual(visionClipOperands?.map((operand) => operand.role), ["weight", "input-min", "input-max", "output-min", "output-max"]);
  const expectedLearnedConsumerCount = [...literal.assignments.composite, ...literal.assignments.vision, ...literal.assignments.audio]
    .filter((assignment) => (assignment.tensors?.length ?? 0) > 0).length +
    [...literal.program.textProgram.prelude, ...literal.program.textProgram.layers.flatMap((layer) => layer.operations), ...literal.assignments.textEpilogue]
      .filter((operation) => operation.op === "embedding" || operation.op === "per_layer_embedding" ||
        operation.op === "linear" || operation.op === "tensor_scale" || operation.op === "rms_norm" && operation.weight !== undefined).length;
  assert.equal(literal.learnedOperands.assignments.length, expectedLearnedConsumerCount);
  const halfLiteral = literal.numericLiterals.literals.find((entry) => entry.token === "0.5")!;
  assert.equal(halfLiteral.binary64Hex, "0x3fe0000000000000");
  assert.equal(halfLiteral.binary32Hex, "0x3f000000");
  assert.equal(halfLiteral.bfloat16Hex, "0x3f00");
  assert.ok(halfLiteral.uses.some((use) => use.section === "forward"));
  const geluScaleLiteral = literal.numericLiterals.literals.find((entry) =>
    entry.token === String(Math.sqrt(2 / Math.PI)))!;
  assert.equal(geluScaleLiteral.binary64Hex, "0x3fe9884533d43651");
  assert.ok(geluScaleLiteral.uses.some((use) => use.section === "forward"));
  assert.ok(literal.numericLiterals.literals.some((entry) =>
    entry.uses.some((use) => use.section === "generation" && use.definitionId === "generation_argmax")));
  assert.equal(literal.calculationGraph.assignments.length, 223);
  assert.equal(literal.calculationGraph.schemaVersion, 8);
  assert.deepEqual(literal.calculationGraph.coordinateLanguage, gemma4LiteralCoordinateExpressionLanguage());
  assert.deepEqual(literal.calculationGraph.assignments.map((entry) => entry.ordinal),
    Array.from({ length: literal.calculationGraph.assignments.length }, (_, index) => index));
  assert.equal(literal.calculationGraph.assignments.some((entry) =>
    entry.operation === "vision-feature-program" || entry.operation === "audio-feature-program" || entry.operation === "text-core"), false);
  const instantiatedImageQ = literal.calculationGraph.assignments.find((entry) =>
    entry.operationId === "composite_image_features/vision_layer_0_q")!;
  assert.deepEqual(instantiatedImageQ.orderedInputs, ["composite_image_features/vision_layer_0_attn_norm"]);
  assert.equal(instantiatedImageQ.output, "composite_image_features/vision_layer_0_q_linear");
  assert.deepEqual(instantiatedImageQ.outputCoordinate.write, {
    kind: "tensor-element",
    expression: "composite_image_features/vision_layer_0_q_linear[batch,patch,output_feature]",
    coordinates: ["batch", "patch", "output_feature"],
    coordinatePrograms: [
      { kind: "symbol", name: "batch" },
      { kind: "symbol", name: "patch" },
      { kind: "symbol", name: "output_feature" },
    ],
  });
  assert.equal(instantiatedImageQ.predecessors[0]?.producerOperationId, "composite_image_features/vision_layer_0_input_norm");
  assert.deepEqual(instantiatedImageQ.predecessors[0]?.accesses[0], {
    kind: "tensor-element",
    expression: "composite_image_features/vision_layer_0_attn_norm[batch,patch,input_feature]",
    coordinates: ["batch", "patch", "input_feature"],
    coordinatePrograms: [
      { kind: "symbol", name: "batch" },
      { kind: "symbol", name: "patch" },
      { kind: "symbol", name: "input_feature" },
    ],
  });
  assert.match(instantiatedImageQ.scalarCalculation.formula, /composite_image_features\/vision_layer_0_attn_norm/);
  assert.equal(instantiatedImageQ.scalarCalculation.output, instantiatedImageQ.output);
  const instantiatedVideoPatch = literal.calculationGraph.assignments.find((entry) =>
    entry.operationId === "composite_video_features/vision_patch_projection")!;
  assert.deepEqual(instantiatedVideoPatch.orderedInputs, ["composite_video_features/vision_pixels_standardized"]);
  assert.equal(instantiatedVideoPatch.outputDomain.shape[0], "VIDEO_BATCH*VIDEO_FRAMES");
  const instantiatedTextAttention = literal.calculationGraph.assignments.find((entry) => entry.operationId === "layer_0_attention")!;
  assert.equal(instantiatedTextAttention.orderedInputs.at(-1), "full_attention_mask");
  assert.ok(literal.calculationGraph.assignments.every((entry) => entry.predecessors.every((predecessor) =>
    predecessor.scalarUse === (predecessor.accesses.length === 0 ? "shape-or-control-only" : "addressed"))));
  assert.ok(literal.calculationGraph.assignments.every((entry) => entry.predecessors.every((predecessor) => {
    if (!predecessor.producerOperationId) return true;
    const producer = literal.calculationGraph.assignments.find((candidate) => candidate.operationId === predecessor.producerOperationId)!;
    return producer.ordinal < entry.ordinal && producer.consumers.includes(entry.operationId);
  })));
  assert.ok(literal.calculationGraph.assignments.every((entry) =>
    entry.consumerCoordinates.length === entry.consumers.length &&
    entry.consumerCoordinates.every((consumer) => entry.consumers.includes(consumer.operationId) &&
      consumer.scalarUse === (consumer.accesses.length === 0 ? "shape-or-control-only" : "addressed"))));
  assert.deepEqual(instantiatedImageQ.consumerCoordinates[0], {
    operationId: "composite_image_features/vision_layer_0_q_heads",
    accesses: [{
      kind: "tensor-element",
      expression: "composite_image_features/vision_layer_0_q_linear[batch,patch,head*4+head_feature]",
      coordinates: ["batch", "patch", "head*4+head_feature"],
      coordinatePrograms: [
        { kind: "symbol", name: "batch" },
        { kind: "symbol", name: "patch" },
        {
          kind: "add",
          left: { kind: "multiply", left: { kind: "symbol", name: "head" }, right: { kind: "constant", value: 4 } },
          right: { kind: "symbol", name: "head_feature" },
        },
      ],
    }],
    scalarUse: "addressed",
  });
  assert.deepEqual(literal.inputs.filter((input) => input.usedBy.includes("generation")).map((input) => input.name), [
    "input_ids", "position_ids", "pixel_values", "image_position_ids", "pixel_values_videos", "video_position_ids",
    "input_features", "input_features_mask", "mm_token_type_ids", "max_new_tokens", "eos_token_id",
  ]);
  assert.deepEqual(literal.inputs.find((input) => input.name === "max_new_tokens")?.requiredFor, ["generation"]);
  assert.deepEqual(literal.generation.assignments.map((assignment) => assignment.id), [
    "generation_prefill", "generation_initial_position", "generation_selection_logits", "generation_argmax",
    "generation_token_append", "generation_position_advance", "generation_incremental_inputs",
    "generation_incremental_forward", "generation_logits_append", "generation_cache_append", "generation_eos_stop",
    "generation_terminal_logits", "generation_terminal_cache",
  ]);
  assert.equal(literal.generation.scalarCalculations.assignments.length, literal.generation.assignments.length);
  assert.deepEqual(
    literal.generation.scalarCalculations.assignments.map((calculation) => calculation.definitionId),
    literal.generation.assignments.map((assignment) => assignment.id),
  );
  assert.equal(literal.generation.forwardCalculation.operationOrder.length, 223);
  assert.deepEqual(literal.generation.forwardCalculation.operationOrder,
    literal.calculationGraph.assignments.map((entry) => ({
      operationId: entry.operationId, definitionId: entry.definitionId, scope: entry.scope,
      ...(entry.invocationId ? { invocationId: entry.invocationId } : {}),
    })));
  assert.equal(literal.generation.forwardCalculation.operationOrder[0]?.operationId, "composite_block_sequence_ids");
  assert.equal(literal.generation.forwardCalculation.operationOrder.at(-1)?.operationId, literal.generation.forwardProgram.lastAssignment);
  assert.deepEqual(literal.generation.forwardCalculation.cacheTransitions.map((transition) => transition.layer), [0, 1]);
  assert.equal(literal.generation.forwardCalculation.schemaVersion, 2);
  assert.equal(literal.generation.controlProgram.kind, "gemma4-literal-greedy-control-program");
  assert.equal(literal.generation.controlProgram.schemaVersion, 2);
  assert.equal(literal.generation.controlProgram.loop.logitsSnapshot.source, "forward_state[step+1].logits");
  assert.equal(literal.generation.controlProgram.loop.selection.tokenEndExclusive, 6);
  assert.deepEqual(literal.generation.controlProgram.loop.incrementalForward.omittedInputs, [
    "attention_mask", "mm_token_type_ids", "pixel_values", "image_position_ids", "pixel_values_videos",
    "video_position_ids", "input_features", "input_features_mask",
  ]);
  assert.equal(selectGemma4LiteralGenerationToken(literal.generation.controlProgram, {
    shape: [1, 1, 6], values: Float32Array.from([1, 4, 4, 3, 2, 1]),
  }), 1, "structured argmax must retain the first/lowest exact tie");
  assert.throws(() => selectGemma4LiteralGenerationToken(literal.generation.controlProgram, {
    shape: [1, 1, 6], values: Float32Array.from([1, 2, Number.NaN, 3, 4, 5]),
  }), /rejeita logit não finito/);
  assert.ok(literal.generation.forwardCalculation.cacheTransitions.every((transition) =>
    transition.ownership === "producer" && transition.producerLayer === transition.layer));
  assert.ok(literal.generation.forwardCalculation.cacheTransitions.every((transition) =>
    transition.prefill.mode === "write-producer" && transition.incremental.mode === "write-producer" &&
    transition.prefill.coordinateOrder === "batch,head,sequence,head_feature ascending lexicographic" &&
    transition.incremental.coordinateOrder === "batch,head,sequence,head_feature ascending lexicographic"));
  assert.ok(literal.generation.forwardCalculation.cacheTransitions[0]!.prefill.scalarAssignments.some((formula) =>
    formula.includes("past_key_values[0].key[batch,head,sequence,head_feature]")));
  assert.ok(literal.generation.forwardCalculation.cacheTransitions[0]!.incremental.scalarAssignments.some((formula) =>
    formula.includes("sequence<previous_sequence_length ? previous_past_key_values[0].key")));
  assert.ok(literal.generation.forwardCalculation.cacheTransitions[1]!.incremental.scalarAssignments.some((formula) =>
    formula.includes("sequence<previous_sequence_length ? previous_past_key_values[1].key")));
  const serializedIncrementalForward = literal.generation.scalarCalculations.assignments.find((calculation) =>
    calculation.definitionId === "generation_incremental_forward")!;
  assert.equal(serializedIncrementalForward.forwardExpansionReference, "generation.forwardCalculation.operationOrder");
  assert.equal(serializedIncrementalForward.cacheTransitionReference, "generation.forwardCalculation.cacheTransitions");
  assert.doesNotMatch(serializedIncrementalForward.formula, /declared_cached_incremental_forward|generic_decoder/);
  assert.match(literal.generation.assignments.find((assignment) => assignment.id === "generation_argmax")!.semantics, /lowest token ID/);
  assert.match(literal.generation.assignments.find((assignment) => assignment.id === "generation_incremental_inputs")!.semantics, /Omit attention_mask, mm_token_type_ids/);
  assert.match(literal.generation.assignments.find((assignment) => assignment.id === "generation_eos_stop")!.semantics, /After incremental logits and cache exist/);

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
  assert.equal(generation.stepForwardLogits.length, 2);
  assert.deepEqual(generation.stepForwardLogits[0], generation.selectionLogits[1]);
  assert.deepEqual(generation.stepForwardLogits.at(-1), generation.text.logits);
  const malformedForwardOutput = structuredClone(replay);
  malformedForwardOutput.text.pastKeyValues = new Map([...malformedForwardOutput.text.pastKeyValues].filter(([layer]) => layer !== 0));
  assert.throws(() => executeGemma4LiteralForwardOutputContract(literal.outputContract, literal.program, {
    inputIds: [[1, 99, 2]], mmTokenTypeIds: [[0, 1, 0]],
    pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]],
  }, malformedForwardOutput), /exatamente os caches produtores/);
  const malformedGenerationOutput = structuredClone(generation);
  malformedGenerationOutput.stepPastKeyValues.pop();
  assert.throws(() => executeGemma4LiteralGenerationOutputContract(literal.outputContract, literal.program, {
    inputIds: [[1, 99, 2]], maxNewTokens: 2,
  }, {
    prefill: malformedGenerationOutput.prefill.text,
    generatedTokenIds: malformedGenerationOutput.generatedTokenIds,
    selectionLogits: malformedGenerationOutput.selectionLogits,
    stepForwardLogits: malformedGenerationOutput.stepForwardLogits,
    stepPastKeyValues: malformedGenerationOutput.stepPastKeyValues,
    terminal: malformedGenerationOutput.text,
  }), /cardinalidade greedy/);
  const divergentStateChain = structuredClone(generation);
  divergentStateChain.stepForwardLogits[0] = {
    shape: [...divergentStateChain.stepForwardLogits[0]!.shape],
    values: Float32Array.from(divergentStateChain.stepForwardLogits[0]!.values),
  };
  divergentStateChain.stepForwardLogits[0]!.values[0] = Math.fround(divergentStateChain.stepForwardLogits[0]!.values[0]! + 1);
  assert.throws(() => executeGemma4LiteralGenerationOutputContract(literal.outputContract, literal.program, {
    inputIds: [[1, 99, 2]], maxNewTokens: 2,
  }, {
    prefill: divergentStateChain.prefill.text,
    generatedTokenIds: divergentStateChain.generatedTokenIds,
    selectionLogits: divergentStateChain.selectionLogits,
    stepForwardLogits: divergentStateChain.stepForwardLogits,
    stepPastKeyValues: divergentStateChain.stepPastKeyValues,
    terminal: divergentStateChain.text,
  }), /selection_logits\[1\].*forward_state\[1\]/);
  const divergentTerminalLogits = structuredClone(generation);
  divergentTerminalLogits.text.logits = {
    shape: [...divergentTerminalLogits.text.logits.shape],
    values: Float32Array.from(divergentTerminalLogits.text.logits.values),
  };
  divergentTerminalLogits.text.logits.values[0] = Math.fround(divergentTerminalLogits.text.logits.values[0]! + 1);
  assert.throws(() => executeGemma4LiteralGenerationOutputContract(literal.outputContract, literal.program, {
    inputIds: [[1, 99, 2]], maxNewTokens: 2,
  }, {
    prefill: divergentTerminalLogits.prefill.text,
    generatedTokenIds: divergentTerminalLogits.generatedTokenIds,
    selectionLogits: divergentTerminalLogits.selectionLogits,
    stepForwardLogits: divergentTerminalLogits.stepForwardLogits,
    stepPastKeyValues: divergentTerminalLogits.stepPastKeyValues,
    terminal: divergentTerminalLogits.text,
  }), /logits e cache terminais/);
  const divergentSelectedToken = structuredClone(generation);
  divergentSelectedToken.generatedTokenIds[0] = (divergentSelectedToken.generatedTokenIds[0]! + 1) % literal.program.contract.text.vocabSize;
  assert.throws(() => executeGemma4LiteralGenerationOutputContract(literal.outputContract, literal.program, {
    inputIds: [[1, 99, 2]], maxNewTokens: 2,
  }, {
    prefill: divergentSelectedToken.prefill.text,
    generatedTokenIds: divergentSelectedToken.generatedTokenIds,
    selectionLogits: divergentSelectedToken.selectionLogits,
    stepForwardLogits: divergentSelectedToken.stepForwardLogits,
    stepPastKeyValues: divergentSelectedToken.stepPastKeyValues,
    terminal: divergentSelectedToken.text,
  }), /generated_token_ids\[0\].*argmax/);
  const eosGeneration = generateGemma4CompositeLiteralF32(literal, {
    inputIds: [[1, 99, 2]], mmTokenTypeIds: [[0, 1, 0]],
    pixelValues: patterned([1, 4, 12]), imagePositionIds: [[[0, 0], [1, 0], [0, 1], [1, 1]]],
    maxNewTokens: 2, eosTokenId: expectedGeneration.generatedTokenIds[0]!,
  });
  assert.deepEqual(eosGeneration.generatedTokenIds, expectedGeneration.generatedTokenIds.slice(0, 1));
  assert.equal(eosGeneration.stepPastKeyValues.length, 1);
  assert.equal(eosGeneration.stepForwardLogits.length, 1);

  const missingMask = structuredClone(literal);
  missingMask.assignments.composite = missingMask.assignments.composite.filter((assignment) => assignment.id !== "composite_sliding_attention_mask");
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(missingMask), /omite uma transição de máscara ou cache obrigatória/);
  const dishonestRuntime = structuredClone(literal);
  dishonestRuntime.authoritativeExecution.executionMode = "torch.no_grad" as "torch.inference_mode";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(dishonestRuntime), /contrato autoritativo de execução/);
  const tamperedAdapter = structuredClone(literal);
  tamperedAdapter.authoritativeExecution.unresolvedNativeReduction.executableReplay.adapterProgram.sourceUtf8 += "\n# tampered\n";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(tamperedAdapter), /Adapter de redução Gemma 4 diverge/);
  const tamperedInvocation = structuredClone(literal);
  tamperedInvocation.authoritativeExecution.unresolvedNativeReduction.executableReplay.invocationPrograms[0]!.stages.reverse();
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(tamperedInvocation), /Programas de invocação.*divergentes/);
  const tamperedInvocationEnvironment = structuredClone(literal);
  tamperedInvocationEnvironment.authoritativeExecution.unresolvedNativeReduction.executableReplay
    .invocationPrograms[0]!.environment.towerParameters[0]!.minimumInclusive = 0;
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(tamperedInvocationEnvironment), /Programas de invocação.*divergentes/);
  const tamperedExecutionProtocol = structuredClone(literal);
  tamperedExecutionProtocol.authoritativeExecution.unresolvedNativeReduction.executableReplay
    .executionProtocol.invocation.arguments.reverse();
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(tamperedExecutionProtocol), /contrato autoritativo de execução/);
  const tamperedTensorEncoding = structuredClone(literal);
  tamperedTensorEncoding.authoritativeExecution.unresolvedNativeReduction.executableReplay
    .executionProtocol.tensorEncoding.byteOrder = "big-endian" as "little-endian";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(tamperedTensorEncoding), /contrato autoritativo de execução/);
  const tamperedProcessEnvironment = structuredClone(literal);
  (tamperedProcessEnvironment.authoritativeExecution.unresolvedNativeReduction.executableReplay
    .runtimeProcessEnvironment.variables as unknown as { LANG: string }).LANG = "pt_BR.UTF-8";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(tamperedProcessEnvironment), /contrato autoritativo de execução/);
  const tamperedRuntimeEnvironment = structuredClone(literal);
  tamperedRuntimeEnvironment.authoritativeExecution.unresolvedNativeReduction.executableReplay
    .runtimeEnvironmentIdentity = {
      ...tamperedRuntimeEnvironment.authoritativeExecution.unresolvedNativeReduction.executableReplay.runtimeEnvironmentIdentity,
      operatingSystemBuild: "different-build",
    } as unknown as typeof GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY;
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(tamperedRuntimeEnvironment), /contrato autoritativo de execução/);
  const tamperedRuntimeExecutionState = structuredClone(literal);
  tamperedRuntimeExecutionState.authoritativeExecution.unresolvedNativeReduction.executableReplay
    .runtimeExecutionState = {
      ...tamperedRuntimeExecutionState.authoritativeExecution.unresolvedNativeReduction.executableReplay.runtimeExecutionState,
      intraopThreads: 1,
    } as unknown as typeof GEMMA4_RUNTIME_REDUCTION_EXECUTION_STATE;
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(tamperedRuntimeExecutionState), /contrato autoritativo de execução/);
  const external = structuredClone(literal);
  external.program.textProgram.source.path = "/checkpoint/model.safetensors";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(external), /reteve uma referência de source checkpoint/);
  const hiddenGenerationRule = structuredClone(literal);
  hiddenGenerationRule.generation.assignments = hiddenGenerationRule.generation.assignments.filter((assignment) => assignment.id !== "generation_cache_append");
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(hiddenGenerationRule), /transições de geração greedy incompletas/);
  const hiddenGenerationLogits = structuredClone(literal);
  hiddenGenerationLogits.generation.assignments = hiddenGenerationLogits.generation.assignments.filter((assignment) => assignment.id !== "generation_logits_append");
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(hiddenGenerationLogits), /transições de geração greedy incompletas/);
  const missingGenerationInput = structuredClone(literal);
  missingGenerationInput.inputs = missingGenerationInput.inputs.filter((input) => input.name !== "max_new_tokens");
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(missingGenerationInput), /controles de forward e geração/);
  const missingDomain = structuredClone(literal);
  missingDomain.calculationDomains.assignments = missingDomain.calculationDomains.assignments.filter((entry) => entry.definitionId !== "layer_0_q_proj");
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(missingDomain), /domínios de shape\/dtype\/layout/);
  const cyclicDimension = structuredClone(literal);
  cyclicDimension.calculationDomains.dimensionPrograms.AT1 = { kind: "dimension", name: "AT1" };
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(cyclicDimension), /ciclo de dimensão/);
  const hostDimensionLanguage = structuredClone(literal);
  hostDimensionLanguage.calculationDomains.dimensionLanguage.arithmetic.ceilDivide = "host decides";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(hostDimensionLanguage), /linguagem de dimensões/);
  const hostReductionLanguage = structuredClone(literal);
  hostReductionLanguage.formulaLanguage.reductions.domainLanguage.semantics.tensorAxis = "host decides";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(hostReductionLanguage), /linguagem de fórmulas/);
  const unboundReductionDomain = structuredClone(literal);
  const unboundExtent = unboundReductionDomain.scalarCalculations.assignments.find((entry) =>
    entry.scope === "vision" && entry.definitionId === "vision_layer_0_attention")!.reduction!.domains[0]!.endExclusive;
  if (unboundExtent.kind !== "tensor-axis") throw new Error("fixture requires tensor-axis reduction extent");
  unboundExtent.tensor = "undeclared_attention_input";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(unboundReductionDomain), /fórmulas, casts ou reduções escalares/);
  const visionAttentionValueCalculation = literal.scalarCalculations.assignments.find((entry) =>
    entry.scope === "vision" && entry.definitionId === "vision_layer_0_attention")!;
  assert.throws(() => evaluateGemma4LiteralReductionIndexDomains(
    literal.formulaLanguage.reductions.domainLanguage,
    visionAttentionValueCalculation.reduction!.domains,
    {},
  ), /não pode resolver vision_layer_0_attention_weights\.shape\[3\]/);
  const reversedScalarProgram = structuredClone(literal);
  reversedScalarProgram.scalarCalculations.assignments.find((entry) =>
    entry.scope === "text-layer" && entry.definitionId === "layer_0_attention")!.scalarAssignments.reverse();
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(reversedScalarProgram), /fórmulas, casts ou reduções escalares/);
  const dishonestScalarConsumer = structuredClone(literal);
  dishonestScalarConsumer.scalarCalculations.assignments.find((entry) =>
    entry.scope === "text-layer" && entry.definitionId === "layer_0_attention")!
    .statementDataflow[2]!.consumerStatementOrdinals.pop();
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(dishonestScalarConsumer), /fórmulas, casts ou reduções escalares/);
  const dishonestScalarSyntax = structuredClone(literal);
  dishonestScalarSyntax.scalarCalculations.assignments.find((entry) =>
    entry.scope === "text-layer" && entry.definitionId === "layer_0_attention")!
    .statementPrograms[2]!.source += " ";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(dishonestScalarSyntax), /fórmulas, casts ou reduções escalares/);
  const dishonestScalarEnvironment = structuredClone(literal);
  dishonestScalarEnvironment.scalarCalculations.assignments.find((entry) =>
    entry.scope === "text-layer" && entry.definitionId === "layer_0_attention")!
    .statementEnvironment.intrinsics.pop();
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(dishonestScalarEnvironment), /fórmulas, casts ou reduções escalares/);
  const guessedLearnedRole = structuredClone(literal);
  guessedLearnedRole.learnedOperands.assignments.find((entry) =>
    entry.scope === "vision" && entry.definitionId === "vision_layer_0_q")!.operands[0]!.role = "bias";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(guessedLearnedRole), /papéis ou índices de operandos aprendidos/);
  const hiddenLearnedIndex = structuredClone(literal);
  hiddenLearnedIndex.learnedOperands.assignments.find((entry) =>
    entry.scope === "audio" && entry.definitionId === "audio_subsample_0_conv")!.operands[0]!.logicalIndices.pop();
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(hiddenLearnedIndex), /papéis ou índices de operandos aprendidos/);
  const alteredLearnedDecoder = structuredClone(literal);
  alteredLearnedDecoder.learnedOperands.assignments.find((entry) =>
    entry.scope === "text-layer" && entry.definitionId === "layer_0_q_proj")!.operands[0]!.decoderId = "decode_wrong_tensor";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(alteredLearnedDecoder), /papéis ou índices de operandos aprendidos/);
  const alteredStorageAddress = structuredClone(literal);
  alteredStorageAddress.storageDecoders[0]!.address.stridesElements[0] = 1;
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(alteredStorageAddress), /decoder de storage literal/);
  const alteredStorageDecode = structuredClone(literal);
  alteredStorageDecode.storageDecoders[0]!.decode.resultBits = { op: "literal-u32", value: 0 };
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(alteredStorageDecode), /decoder de storage literal/);
  const alteredDecoderLanguage = structuredClone(literal);
  alteredDecoderLanguage.denseDecoderLanguage.bitSemantics.result = "resultBits is numerically converted to binary32" as never;
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(alteredDecoderLanguage), /linguagem executável de decoder denso/);
  const hiddenScalarFormula = structuredClone(literal);
  hiddenScalarFormula.scalarCalculations.assignments.find((entry) =>
    entry.scope === "audio" && entry.definitionId === "audio_subsample_0_conv")!.formula = "opaque_conv2d(input,weight)";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(hiddenScalarFormula), /fórmulas, casts ou reduções escalares/);
  const alteredReduction = structuredClone(literal);
  const alteredCalculation = alteredReduction.scalarCalculations.assignments.find((entry) =>
    entry.scope === "text-layer" && entry.definitionId === "layer_0_q_proj")!;
  alteredCalculation.reduction!.order = "ascending-lexicographic";
  delete alteredCalculation.reduction!.schedule;
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(alteredReduction), /fórmulas, casts ou reduções escalares/);
  const alteredNumericBits = structuredClone(literal);
  alteredNumericBits.numericLiterals.literals.find((entry) => entry.token === "0.5")!.binary32Hex = "0x00000000";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(alteredNumericBits), /tabela de bits numéricos/);
  const alteredSoftmaxProgram = structuredClone(literal);
  alteredSoftmaxProgram.formulaLanguage.reductions.softmaxPrograms.pytorchF32VectorPairwiseSum[0] = "host reduce";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(alteredSoftmaxProgram), /linguagem de fórmulas/);
  const hiddenInstantiatedInput = structuredClone(literal);
  hiddenInstantiatedInput.calculationGraph.assignments.find((entry) =>
    entry.operationId === "composite_image_features/vision_layer_0_q")!.orderedInputs[0] = "hidden_reader_binding";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(hiddenInstantiatedInput), /grafo instanciado, bindings ou dependências/);
  const opaqueInstantiatedFormula = structuredClone(literal);
  opaqueInstantiatedFormula.calculationGraph.assignments.find((entry) =>
    entry.operationId === "composite_audio_features/audio_subsample_0_conv")!.scalarCalculation.formula = "declared_subprogram(input_features)";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(opaqueInstantiatedFormula), /grafo instanciado, bindings ou dependências/);
  const hiddenPredecessor = structuredClone(literal);
  delete hiddenPredecessor.calculationGraph.assignments.find((entry) =>
    entry.operationId === "composite_audio_features/audio_subsample_0_conv")!.predecessors[0]!.producerOperationId;
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(hiddenPredecessor), /grafo instanciado, bindings ou dependências/);
  const alteredPredecessorCoordinate = structuredClone(literal);
  alteredPredecessorCoordinate.calculationGraph.assignments.find((entry) =>
    entry.operationId === "composite_image_features/vision_layer_0_q")!.predecessors[0]!.accesses[0] = {
      kind: "tensor-element", expression: "wrong[0]", coordinates: ["0"], coordinatePrograms: [{ kind: "constant", value: 0 }],
  };
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(alteredPredecessorCoordinate), /grafo instanciado, bindings ou dependências/);
  const alteredOutputCoordinate = structuredClone(literal);
  alteredOutputCoordinate.calculationGraph.assignments.find((entry) =>
    entry.operationId === "composite_image_features/vision_layer_0_q")!.outputCoordinate.write.coordinates[0] = "wrong";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(alteredOutputCoordinate), /grafo instanciado, bindings ou dependências/);
  const alteredOutputCoordinateProgram = structuredClone(literal);
  alteredOutputCoordinateProgram.calculationGraph.assignments.find((entry) =>
    entry.operationId === "composite_image_features/vision_layer_0_q")!.outputCoordinate.write.coordinatePrograms[0] = {
      kind: "constant", value: 0,
  };
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(alteredOutputCoordinateProgram), /grafo instanciado, bindings ou dependências/);
  const alteredCoordinateLanguage = structuredClone(literal);
  alteredCoordinateLanguage.calculationGraph.coordinateLanguage.arithmetic.multiply = "host arithmetic";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(alteredCoordinateLanguage), /grafo instanciado, bindings ou dependências/);
  const alteredConsumerCoordinate = structuredClone(literal);
  alteredConsumerCoordinate.calculationGraph.assignments.find((entry) =>
    entry.operationId === "composite_image_features/vision_layer_0_q")!.consumerCoordinates[0]!.accesses.pop();
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(alteredConsumerCoordinate), /grafo instanciado, bindings ou dependências/);
  const opaqueGeneration = structuredClone(literal);
  opaqueGeneration.generation.scalarCalculations.assignments.find((calculation) =>
    calculation.definitionId === "generation_prefill")!.formula = "forward_state[0] = generic_decoder(input_ids)";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(opaqueGeneration), /transições de geração greedy incompletas/);
  const hiddenGenerationControl = structuredClone(literal);
  hiddenGenerationControl.generation.controlProgram.loop.incrementalForward.omittedInputs.pop();
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(hiddenGenerationControl), /transições de geração greedy incompletas/);
  const hiddenForwardControl = structuredClone(literal);
  hiddenForwardControl.forwardControl.modalityBranches.find((branch) => branch.modality === "audio")!.inactiveIdentity.input = "host_selected_embedding";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(hiddenForwardControl), /controle forward, roteamento modal ou aliases de ausência/);
  const hiddenInputContract = structuredClone(literal);
  hiddenInputContract.inputContract.vision.pixelFeatures += 1;
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(hiddenInputContract), /contrato de inputs, shapes ou cardinalidade/);
  const hiddenOutputContract = structuredClone(literal);
  hiddenOutputContract.outputContract.forward.tensors.find((entry) => entry.role === "logits")!.shape[2] = "host_vocab_size";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(hiddenOutputContract), /contrato de outputs, cache ou estado terminal/);
  const missingCacheTransition = structuredClone(literal);
  missingCacheTransition.generation.forwardCalculation.cacheTransitions.pop();
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(missingCacheTransition), /transições de geração greedy incompletas/);
  assert.throws(
    () => generateGemma4CompositeLiteralF32(literal, { inputIds: [[1]], maxNewTokens: 1, pastKeyValues: expected.text.pastKeyValues }),
    /começa em prefill sem pastKeyValues/,
  );
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
    }, output, fixtureSourceIdentity());
    const raw = await readFile(output);
    assert.equal(written.artifactSha256, createHash("sha256").update(raw).digest("hex"));
    assert.equal(written.constants, catalog.tensors.size);
    assert.equal(written.embeddedPayloadBytes, [...catalog.tensors.values()].reduce((total, tensor) => total + tensor.logicalShape.reduce((size, dimension) => size * dimension, 1) * 4, 0));
    assert.ok(maxRequestedBytes < written.embeddedPayloadBytes, "writer must request one payload at a time rather than a package buffer");
    const audit = await auditLiteralArtifact(output, { constants: written.constants, embeddedPayloadBytes: written.embeddedPayloadBytes });
    assert.equal(audit.constants, catalog.tensors.size);
    const literal = JSON.parse(raw.toString("utf8"));
    assert.equal(JSON.stringify(literal).includes(catalog.source), false);
    assert.equal(literal.sourceIdentity.modelId, "fixture/tiny-gemma4");
    assert.equal(Buffer.from(literal.sourceIdentity.files.find((file: { path: string }) => file.path === "config.json").content.payloadBase64, "base64").toString("utf8"), "{}");
    assert.equal(literal.payloadIntegrity.length, catalog.tensors.size);
    assert.equal(literal.integrityManifest.sections.length, 24);
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
    }, output, fixtureSourceIdentity());
    const expected = sourceTensors.get("model.language_model.layers.0.self_attn.q_proj.weight")!;
    const expectedBytes = Buffer.alloc(expected.values.length * 4);
    expected.values.forEach((value, index) => expectedBytes.writeFloatLE(value, index * 4));
    sourceTensors.clear();
    await rm(source);

    const artifact = await openGemma4CompositeLiteralArtifact(output);
    try {
      assert.equal(artifact.schemaVersion, 58);
      assert.deepEqual(artifact.denseDecoderLanguage, buildLiteralDenseDecoderLanguageContract());
      assert.equal(artifact.sourceIdentity.revision, "a".repeat(40));
      assert.equal(artifact.authoritativeExecution.unresolvedNativeReduction.executableReplay.adapterProgram.sha256,
        GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256);
      assert.equal(artifact.constants.size, catalog.tensors.size);
      assert.equal(artifact.program.textProgram.source.path, "embedded://gemma4-composite-literal");
      const tensor = catalog.tensors.get("model.language_model.layers.0.self_attn.q_proj.weight")!;
      assert.deepEqual(await artifact.readTensorBytes(tensor), expectedBytes);
      assert.deepEqual(await artifact.readTensorBytesRange(tensor, 5, 23), expectedBytes.subarray(5, 28));
      assert.equal("payloadBase64" in artifact.constants.get(tensor.name)!, false);
      assert.equal(artifact.payloadIntegrity.size, catalog.tensors.size);
      assert.equal(artifact.integrityManifest.sections.length, 24);
      assert.equal(artifact.fidelityGate.exactReplayClaim, "forbidden");
      assert.equal(artifact.fidelityGate.unresolvedNativeReductionCount, artifact.fidelityGate.unresolvedNativeReductions.length);
      assert.equal(artifact.generation.kind, "gemma4-literal-greedy-generation-program");
      assert.equal(artifact.generation.forwardProgram.firstAssignment, "composite_block_sequence_ids");
      assert.equal(artifact.generation.forwardProgram.lastAssignment, "lm_head");
      assert.equal(artifact.generation.outputs.generatedTokenIds, "generated_token_ids");
      assert.equal(artifact.generation.controlProgram.loop.selection.scanOrder, "ascending-token-id");
      assert.equal(artifact.forwardControl.modalityBranches.length, 3);
      assert.equal(artifact.formulaLanguage.authority.forwardControlProgram, "/forwardControl");
      assert.equal(artifact.numericLiterals.literals.find((entry) => entry.token === "0.5")?.binary32Hex, "0x3f000000");
      assert.equal(artifact.formulaLanguage.evaluation.dependencyOrder,
        "evaluate instantiated assignments by ascending ordinal; within each assignment evaluate statementPrograms in ordinal order, where every local and precondition precedes the final output assignment; scalarAssignments is the audit rendering and every predecessor must already exist");
      assert.deepEqual(artifact.transcendentalPrograms, buildGemma4LiteralTranscendentalPrograms());
    } finally {
      await artifact.close();
    }
    const corrupted = path.join(root, "corrupted.gemma4.literal.json");
    const raw = await readFile(output, "utf8");
    await writeFile(corrupted, raw.replace('"semantics":"exact IEEE-754 storage decode; no arithmetic narrowing"', '"semantics":"invalid"'));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(corrupted), /decoder denso do artefato literal/);
    const corruptedNumeric = path.join(root, "corrupted-numeric.gemma4.literal.json");
    const numericOffset = raw.indexOf('"numericLiterals":');
    assert.ok(numericOffset >= 0);
    await writeFile(corruptedNumeric, raw.slice(0, numericOffset) + raw.slice(numericOffset).replace('"binary32Hex":"0x3f000000"', '"binary32Hex":"0x00000000"'));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(corruptedNumeric), /tabela de bits numéricos/);
    const corruptedLanguage = path.join(root, "corrupted-language.gemma4.literal.json");
    await writeFile(corruptedLanguage, raw.replace(
      '"dependencyOrder":"evaluate instantiated assignments by ascending ordinal; within each assignment evaluate statementPrograms in ordinal order, where every local and precondition precedes the final output assignment; scalarAssignments is the audit rendering and every predecessor must already exist"',
      '"dependencyOrder":"host decides"',
    ));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(corruptedLanguage), /linguagem de fórmulas/);
    const corruptedTranscendental = path.join(root, "corrupted-transcendental.gemma4.literal.json");
    await writeFile(corruptedTranscendental, raw.replace('"kernel":"Sleef_expf4_u10advsimd"', '"kernel":"host-exp"'));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(corruptedTranscendental), /transcrição transcendental F32/);
    const corruptedIdentity = path.join(root, "corrupted-identity.gemma4.literal.json");
    await writeFile(corruptedIdentity, raw.replace(`"revision":"${"a".repeat(40)}"`, '"revision":"moving-main"'));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(corruptedIdentity), /identidade de source inválida/);
    const corruptedSemanticCommitment = path.join(root, "corrupted-semantic-commitment.gemma4.literal.json");
    await writeFile(corruptedSemanticCommitment, raw.replace('"modelId":"fixture/tiny-gemma4"', '"modelId":"fixture/tiny-gemma4-mutated"'));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(corruptedSemanticCommitment), /compromisso estrutural/);
    const corruptedFidelityGate = path.join(root, "corrupted-fidelity-gate.gemma4.literal.json");
    await writeFile(corruptedFidelityGate, raw.replace('"exactReplayClaim":"forbidden"', '"exactReplayClaim":"not-certified"'));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(corruptedFidelityGate), /gate de fidelidade/);
    const missingIntegrity = path.join(root, "missing-integrity.gemma4.literal.json");
    await writeFile(missingIntegrity, raw.replace(/,"integrityManifest":\{.*\}\}\n$/s, "}\n"));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(missingIntegrity), /não declara compromisso estrutural/);
    const nonCanonicalMetadata = path.join(root, "non-canonical-metadata.gemma4.literal.json");
    await writeFile(nonCanonicalMetadata, raw.replace('"payloadBase64":"e30="', '"payloadBase64":"e30=\\n"'));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(nonCanonicalMetadata), /bytes incorporados não correspondem/);
    const unorderedIdentity = path.join(root, "unordered-identity.gemma4.literal.json");
    await writeFile(unorderedIdentity, raw.replace('"path":"config.json"', '"path":"z-config.json"'));
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(unorderedIdentity), /ordem determinística/);
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
    for (const tensor of catalog.tensors.values()) tensor.shard = "model.safetensors";
    const program = buildGemma4CompositeProgram(catalog, preview);
    const artifact = path.join(root, "tiny.gemma4.literal.json");
    const sourceIdentity = await buildGemma4LiteralSourceIdentity(catalog, "fixture/tiny-gemma4", "c".repeat(40));
    await writeGemma4CompositeLiteralCalculationProgram(program, catalog, {
      async readTensorBytes(info) { return denseF32Bytes(sourceTensors.get(info.name)!); },
    }, artifact, sourceIdentity);
    assert.equal(sourceIdentity.files.find((file) => file.path === "model.safetensors")?.sha256,
      createHash("sha256").update(await readFile(path.join(source, "model.safetensors"))).digest("hex"));
    const embeddedConfig = sourceIdentity.files.find((file) => file.path === "config.json")!.content;
    assert.equal(embeddedConfig.storage, "embedded-metadata-base64");
    assert.equal(Buffer.from(embeddedConfig.storage === "embedded-metadata-base64" ? embeddedConfig.payloadBase64 : "", "base64").toString("utf8"), JSON.stringify(catalog.config));

    const verified = await verifyGemma4CompositeLiteralPayloadsAgainstCatalog({ artifact, source, maxReadBytes: 13 });
    assert.equal(verified.constants, catalog.tensors.size);
    assert.equal(verified.comparedPayloadBytes, [...sourceTensors.values()].reduce((total, tensor) => total + tensor.values.byteLength, 0));
    assert.equal(verified.sourceStorageSha256, verified.literalStorageSha256);
    assert.equal(verified.modelId, "fixture/tiny-gemma4");
    assert.equal(verified.revision, "c".repeat(40));
    assert.equal(verified.sourceIdentityFiles, 2);

    const corrupt = path.join(root, "corrupt.gemma4.literal.json");
    const raw = await readFile(artifact, "utf8");
    const corruptIdentity = path.join(root, "corrupt-source-identity.gemma4.literal.json");
    const weightSha = sourceIdentity.files.find((file) => file.role === "weights")!.sha256;
    await writeFile(corruptIdentity, raw.replace(weightSha, "d".repeat(64)), "utf8");
    await assert.rejects(
      () => verifyGemma4CompositeLiteralPayloadsAgainstCatalog({ artifact: corruptIdentity, source, maxReadBytes: 13 }),
      /compromisso estrutural/,
    );
    await writeFile(corrupt, raw.replace(/("constants":\[\{"name":[\s\S]*?"payloadBase64":")([A-Za-z0-9])/, (_match, prefix: string, first: string) => `${prefix}${first === "A" ? "B" : "A"}`), "utf8");
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
    }, artifact, fixtureSourceIdentity());
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
    await writeFile(corrupt, raw.replace(/("constants":\[\{"name":[\s\S]*?"payloadBase64":")([A-Za-z0-9])/, (_match, prefix: string, first: string) => `${prefix}${first === "A" ? "B" : "A"}`), "utf8");
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
    }, output, fixtureSourceIdentity());
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
    await assert.rejects(() => openGemma4CompositeLiteralArtifact(legacy), /compromisso estrutural/);

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
    }, blockedTiledOutput, fixtureSourceIdentity());
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
    }, fixtureSourceIdentity());
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
    const pixelValues = patterned([1, 4, 12]), pixelPositionIds = [[[0, 0], [1, 0], [0, 1], [1, 1]]];
    const expectedVision = executeGemma4VisionF32(program.visionProgram, { pixelValues, pixelPositionIds, tensors: sourceTensors });
    const compositeRequest = {
      inputIds: [[1, 99, 97, 98, 2]], mmTokenTypeIds: [[0, 1, 2, 3, 0]], pixelValues, imagePositionIds: pixelPositionIds,
      pixelValuesVideos: patterned([1, 1, 4, 12]), videoPositionIds: [[pixelPositionIds[0]!]],
      inputFeatures: patterned([1, 4, 16]), inputFeaturesMask: [[true, true, true, true]],
    };
    const output = path.join(root, "tiny.gemma4.literal.json");
    await writeGemma4CompositeLiteralCalculationProgram(program, catalog, {
      async readTensorBytes(info) {
        const source = sourceTensors.get(info.name)!;
        const bytes = Buffer.alloc(source.values.length * 4);
        source.values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
        return bytes;
      },
    }, output, fixtureSourceIdentity());
    sourceTensors.clear();
    const artifact = await openGemma4CompositeLiteralArtifact(output);
    try {
      const literalVision = await executeGemma4LiteralVisionF32(artifact, { pixelValues, pixelPositionIds });
      assert.deepEqual(literalVision.imageFeatures, expectedVision.imageFeatures);
      assert.equal(literalVision.values.size, expectedVision.values.size);
      await assert.rejects(() => executeGemma4LiteralCompositeF32(artifact, compositeRequest), /fidelidade numérica não verificada/);
      const literalComposite = await executeGemma4LiteralCompositeF32(artifact, compositeRequest, { allowUnverifiedFidelity: true });
      assert.deepEqual(literalComposite.llmInputIds, [[1, 0, 0, 0, 2]]);
      assert.deepEqual(literalComposite.values.get("image_features"), expectedVision.imageFeatures);
      assert.deepEqual(literalComposite.values.get("video_features"), expectedVision.imageFeatures);
      assert.deepEqual(literalComposite.values.get("audio_features")?.shape, [1, 4]);
      assert.deepEqual(literalComposite.text.logits.shape, [1, 5, 6]);
      assert.ok([...literalComposite.text.logits.values].every(Number.isFinite));
      assert.deepEqual(literalComposite.values.get("ple_inputs")?.shape, [1, 5, 2, 1]);
      const literalGeneration = await generateGemma4LiteralCompositeF32(artifact, { ...compositeRequest, maxNewTokens: 2 }, { allowUnverifiedFidelity: true });
      assert.equal(literalGeneration.generatedTokenIds.length, 2);
      assert.deepEqual(literalGeneration.logits.shape, [1, 1, 6]);
      assert.equal(literalGeneration.pastKeyValues.size, 2);
      assert.equal(literalGeneration.assignmentExecutions.length, 2 + 9 * 2 + 2);
      assert.deepEqual(literalGeneration.stepForwardLogits[0], literalGeneration.selectionLogits[1]);
      assert.deepEqual(literalGeneration.stepForwardLogits.at(-1), literalGeneration.logits);
      assert.deepEqual(literalGeneration.compositePrefill.text.logits, literalComposite.text.logits);
      await assert.rejects(
        () => executeGemma4LiteralCompositeF32(artifact, { inputIds: [[1]], pixelValues }, { allowUnverifiedFidelity: true }),
        /pixel_values e image_position_ids juntos/,
      );
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

      const operations = listGemma4LiteralTextOperations(artifact);
      const qProjection = operations.find((operation) => operation.operationId === "layer_0_q_proj");
      assert.equal(qProjection?.predecessors[0]?.producerOperationId, "layer_0_input_norm");
      assert.equal(qProjection?.nextOperationId, "layer_0_q_heads");

      const scalar = await renderGemma4LiteralScalarView(artifact, {
        operationId: "layer_0_q_proj", outputCoordinate: [0, 0, 1],
      });
      assert.equal(scalar.sourceCheckpointAccessed, false);
      assert.equal(scalar.reduction?.complete, true);
      assert.deepEqual(scalar.reduction?.bounds, { startInclusive: 0, endExclusive: 4 });
      assert.equal(scalar.terms?.length, 4);
      assert.deepEqual(scalar.learnedScalars.map((entry) => entry.indices), [[1, 0], [1, 1], [1, 2], [1, 3]]);
      assert.ok(scalar.learnedScalars.every((entry) => entry.decoderOperation === "ieee-f32-little-endian" && /^0x[0-9a-f]{8}$/.test(entry.storageBitsHex)));
      assert.ok(scalar.terms?.every((term) => term.formula.includes(term.learned.literal) && !term.formula.includes("weight[")));
      assert.deepEqual(scalar.renderedOutputCoordinate.write, {
        kind: "tensor-element", expression: "layer_0_q_linear[0,0,1]", coordinates: ["0", "0", "1"],
        coordinatePrograms: [
          { kind: "constant", value: 0 }, { kind: "constant", value: 0 }, { kind: "constant", value: 1 },
        ],
      });
      assert.equal(scalar.predecessorCoordinates[0]?.producerOperationId, "layer_0_input_norm");
      assert.equal(scalar.predecessorCoordinates[0]?.renderedCoverage, "complete");
      assert.deepEqual(scalar.predecessorCoordinates[0]?.renderedAccesses.map((access) => access.expression), [
        "layer_0_attn_norm[0,0,0]", "layer_0_attn_norm[0,0,1]",
        "layer_0_attn_norm[0,0,2]", "layer_0_attn_norm[0,0,3]",
      ]);
      const tamperedPredecessorNavigation = structuredClone(scalar);
      tamperedPredecessorNavigation.predecessorCoordinates[0]!.renderedAccesses.pop();
      assert.throws(
        () => validateGemma4LiteralScalarView(tamperedPredecessorNavigation),
        /navegação de coordenadas predecessoras ausente ou divergente/,
      );
      const tamperedOutputNavigation = structuredClone(scalar);
      tamperedOutputNavigation.renderedOutputCoordinate.write.coordinates[0] = "1";
      assert.throws(
        () => validateGemma4LiteralScalarView(tamperedOutputNavigation),
        /navegação da coordenada de saída ausente ou divergente/,
      );

      const window = await renderGemma4LiteralScalarView(artifact, {
        operationId: "layer_0_q_proj", outputCoordinate: [0, 0, 1], inputStart: 1, inputCount: 2,
      });
      assert.deepEqual(window.reduction, {
        bounds: { startInclusive: 0, endExclusive: 4 }, schedule: { kind: "ordered-scalar", indexOrder: "ascending" },
        complete: false, renderedWindow: { startInclusive: 1, endExclusive: 3 }, omittedTerms: 2,
      });
      assert.equal(window.predecessorCoordinates[0]?.renderedCoverage, "windowed");
      assert.deepEqual(window.predecessorCoordinates[0]?.renderedAccesses.map((access) => access.expression), [
        "layer_0_attn_norm[0,0,1]", "layer_0_attn_norm[0,0,2]",
      ]);

      const embeddingScalar = await renderGemma4LiteralScalarView(artifact, {
        operationId: "token_embedding", outputCoordinate: [0, 0, 2], tokenId: 1,
      });
      assert.deepEqual(embeddingScalar.learnedScalars[0]?.indices, [1, 2]);
      assert.doesNotMatch(embeddingScalar.formula, /weight\[/);

      const normScalar = await renderGemma4LiteralScalarView(artifact, {
        operationId: "layer_0_input_norm", outputCoordinate: [0, 0, 2],
      });
      assert.deepEqual(normScalar.learnedScalars[0]?.indices, [2]);
      assert.ok(normScalar.scalarAssignments.some((entry) => entry.includes("sum_{i=0..3")));
      const opaqueNorm = structuredClone(normScalar);
      opaqueNorm.scalarAssignments.push("sum = PYTORCH_CPU_F32_CASCADE_SUM(square)");
      assert.throws(() => validateGemma4LiteralScalarView(opaqueNorm), /normalização opaca/);

      const tensorScale = await renderGemma4LiteralScalarView(artifact, {
        operationId: "layer_0_scalar", outputCoordinate: [0, 0, 2],
      });
      assert.equal(tensorScale.learnedScalars[0]?.decodedF32, 1);

      const proportionalRope = await renderGemma4LiteralScalarView(artifact, {
        operationId: "layer_1_q_rope", outputCoordinate: [0, 0, 0, 0],
      });
      assert.ok(proportionalRope.scalarAssignments.some((entry) => entry.startsWith("angle = F32(position_ids[0,0]")));
      assert.ok(proportionalRope.formula.includes("layer_1_q_normalized[0,0,0,2]"));

      const attention = await renderGemma4LiteralScalarView(artifact, {
        operationId: "layer_0_attention", outputCoordinate: [0, 0, 3],
      });
      assert.ok(attention.scalarAssignments.some((entry) => entry.includes("feature=0..3")));
      assert.ok(attention.formula.includes("layer_0_v_normalized[0,0,k,3]"));

      const activation = await renderGemma4LiteralScalarView(artifact, {
        operationId: "layer_0_activation", outputCoordinate: [0, 0, 1],
      });
      assert.ok(activation.formula.includes("0.044715"));

      const allOperations = listGemma4LiteralOperations(artifact);
      const textCore = operations.filter((operation) => operation.scope !== "text-prelude");
      assert.equal(allOperations.length,
        program.assignments.length - 4
        + 2 * (program.visionProgram.assignments.length - 1)
        + (program.audioProgram.assignments.length - 1)
        + textCore.length);
      assert.equal(allOperations.some((operation) => operation.operationId === "token_embedding"), false, "prepared composite text must not rerun standalone embedding");
      assert.equal(allOperations.find((operation) => operation.operationId === "composite_image_features/vision_layer_0_q")?.scope, "vision");
      assert.equal(allOperations.find((operation) => operation.operationId === "composite_audio_features/audio_layer_0_attention")?.predecessors[0]?.producerOperationId, "composite_audio_features/audio_layer_0_attention_softmax");
      assert.equal(allOperations.find((operation) => operation.operationId === "composite_video_features/vision_pixels_affine")?.predecessors[0]?.producerOperationId, "composite_video_pixel_flatten");
      assert.equal(allOperations.find((operation) => operation.operationId === "composite_video_features/vision_position_embedding")?.predecessors[0]?.producerOperationId, "composite_video_position_flatten");
      assert.equal(allOperations.find((operation) => operation.operationId === "composite_image_scatter")?.predecessors[2]?.producerOperationId, "composite_image_features/vision_language_projection");
      const ordinal = new Map(allOperations.map((operation) => [operation.operationId, operation.ordinal]));
      assert.ok(allOperations.every((operation) => operation.predecessors.every((predecessor) => predecessor.producerOperationId === undefined || ordinal.get(predecessor.producerOperationId)! < operation.ordinal)));
      assert.equal(allOperations.at(-1)?.operationId, program.textProgram.epilogue.at(-1)?.id);
      assert.ok(allOperations.every((operation) => operation.outputDomain.shape.length === operation.outputDomain.axes.length));
      assert.deepEqual(allOperations.find((operation) => operation.operationId === "composite_image_features/vision_layer_0_attention_scores")?.outputDomain.shape, ["IMAGE_BATCH", "1", "IMAGE_PATCHES", "IMAGE_PATCHES"]);
      assert.deepEqual(allOperations.find((operation) => operation.operationId === "composite_video_features/vision_pool")?.outputDomain.shape, ["VIDEO_BATCH*VIDEO_FRAMES", "VIDEO_POOL_CELLS", "4"]);
      assert.deepEqual(allOperations.find((operation) => operation.operationId === "layer_0_attention")?.outputDomain.shape, ["B", "S", "4"]);
      assert.deepEqual(allOperations.find((operation) => operation.operationId === "composite_image_features/vision_layer_0_q")?.learnedOperands?.map((operand) => operand.role),
        ["weight", "input-min", "input-max", "output-min", "output-max"]);
      assert.deepEqual(allOperations.find((operation) => operation.operationId === "layer_0_q_proj")?.learnedOperands?.map((operand) => operand.logicalIndices),
        [[
          { kind: "output-coordinate", axis: "output_feature" },
          { kind: "reduction-index", name: "input_feature", minInclusive: 0, endExclusive: 4 },
        ]]);
      assert.ok(allOperations.flatMap((operation) => operation.learnedOperands ?? []).every((operand) =>
        operand.decoderId === `decode_${operand.tensor.name}`));
      assert.equal(program.audioProgram.assignments.find((assignment) => assignment.id === "audio_layer_0_attention")?.tensors, undefined);

      const compositeEmbedding = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_text_embedding", outputCoordinate: [0, 0, 1], tokenId: 1,
      });
      const textEmbedding = await renderGemma4LiteralScalarView(artifact, {
        operationId: "token_embedding", outputCoordinate: [0, 0, 1], tokenId: 1,
      });
      assert.equal(compositeEmbedding.dtypePolicy.outputDtype, textEmbedding.dtypePolicy.outputDtype);
      assert.equal(compositeEmbedding.formula, textEmbedding.formula.replace("hidden_states_0", "composite_text_embeddings"));
      const compositeContext = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_ple_context_projection", outputCoordinate: [0, 0, 0],
      });
      const textContext = await renderGemma4LiteralScalarView(artifact, {
        operationId: "ple_context_projection", outputCoordinate: [0, 0, 0],
      });
      assert.equal(compositeContext.dtypePolicy.outputDtype, textContext.dtypePolicy.outputDtype);
      assert.equal(compositeContext.formula, textContext.formula);

      const visionLinear = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_image_features/vision_layer_0_q", outputCoordinate: [0, 0, 1],
      });
      assert.equal(visionLinear.reduction?.complete, true);
      assert.equal(visionLinear.terms?.length, 4);
      assert.equal(visionLinear.learnedScalars.length, 8, "four weights plus four exact clipping bounds");
      assert.ok(visionLinear.terms?.every((term) => !term.formula.includes("weight[")));
      const symbolicVisionLinear = structuredClone(visionLinear);
      symbolicVisionLinear.scalarAssignments.push("acc[i] = F32_FMA(acc[i-1], x[i], weight[0,i])");
      assert.throws(() => validateGemma4LiteralScalarView(symbolicVisionLinear), /referência aprendida simbólica/);

      const visionScores = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_image_features/vision_layer_0_attention_scores", outputCoordinate: [0, 0, 0, 0],
      });
      assert.ok(visionScores.scalarAssignments.some((formula) => formula.includes("F32_FMA") && formula.includes("d=0..3 ascending")));
      const visionWeights = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_image_features/vision_layer_0_attention_weights", outputCoordinate: [0, 0, 0, 0],
      });
      assert.ok(visionWeights.scalarAssignments.some((formula) => formula.includes("SLEEF_EXP_F32")));
      assert.equal(visionWeights.transcendentalPrograms.programs.SLEEF_EXP_F32.kernel, "Sleef_expf4_u10advsimd");
      assert.equal(visionWeights.formulaLanguage.authority.transcendentalPrograms, "/transcendentalPrograms");
      assert.deepEqual(visionWeights.dimensionLanguage, artifact.calculationDomains.dimensionLanguage);
      assert.deepEqual(visionWeights.dimensionPrograms, artifact.calculationDomains.dimensionPrograms);
      assert.ok(visionWeights.scalarAssignments.some((formula) => formula.includes("ORDERED_F32_REDUCE_MAX")));
      assert.ok(visionWeights.scalarAssignments.some((formula) => formula.includes("ORDERED_F32_REDUCE_SUM")));
      const opaqueVisionSoftmax = structuredClone(visionWeights);
      opaqueVisionSoftmax.scalarAssignments.push("maximum=max_k(masked_score[k])");
      assert.throws(() => validateGemma4LiteralScalarView(opaqueVisionSoftmax), /helper opaco de redução softmax/);
      const visionContext = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_image_features/vision_layer_0_attention", outputCoordinate: [0, 0, 0],
      });
      assert.ok(visionContext.scalarAssignments.some((formula) => formula.includes("F32_FMA") && formula.includes("k=0..patches-1 ascending")));
      const visionPool = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_image_features/vision_pool", outputCoordinate: [0, 0, 0],
      });
      assert.ok(visionPool.scalarAssignments.some((formula) => formula.includes("VISION_POOL_SLOT")));
      assert.ok(visionPool.scalarAssignments.some((formula) => formula.includes("F32_FMA")));
      const visionPoolMask = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_image_features/vision_pool_mask", outputCoordinate: [0, 0],
      });
      assert.match(visionPoolMask.formula, /VISION_POOL_CELL_HAS_PATCH/);
      const visionRope = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_image_features/vision_layer_0_q_rope", outputCoordinate: [0, 0, 0, 0],
      });
      assert.ok(visionRope.scalarAssignments.some((formula) => formula.includes("BF16(SLEEF_COS_F32")));
      assert.ok(visionRope.scalarAssignments.some((formula) => formula.includes("direct=BF16") && formula.includes("rotated=BF16")));
      const visionNorm = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_image_features/vision_layer_0_input_norm", outputCoordinate: [0, 0, 0],
      });
      assert.ok(visionNorm.scalarAssignments.some((formula) => formula.includes("ARM_SQRT_F32")));
      const visionActivation = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_image_features/vision_layer_0_gate_activation", outputCoordinate: [0, 0, 0],
      });
      assert.ok(visionActivation.formula.includes("SLEEF_TANH_F32"));

      const videoLinear = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_video_features/vision_patch_projection", outputCoordinate: [0, 0, 1],
      });
      assert.equal(videoLinear.navigation.definitionId, "vision_patch_projection");
      assert.equal(videoLinear.navigation.invocationId, "composite_video_features");
      assert.ok(videoLinear.terms?.every((term) => term.input.startsWith("composite_video_features/vision_pixels_standardized")));

      const position = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_image_features/vision_position_embedding", outputCoordinate: [0, 0, 2], positionCoordinate: [1, 0],
      });
      assert.deepEqual(position.learnedScalars.map((scalar) => scalar.indices), [[0, 1, 2], [1, 0, 2]]);
      assert.doesNotMatch(position.formula, /position_embedding_table/);

      const convolution = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_audio_features/audio_subsample_0_conv", outputCoordinate: [0, 0, 0, 0],
      });
      assert.equal(convolution.learnedScalars.length, 9);
      assert.equal(convolution.reduction?.complete, true);
      assert.equal(convolution.terms?.length, 9);
      assert.ok(convolution.scalarAssignments.some((formula) => formula.includes("source_in_bounds")));
      assert.ok(convolution.terms?.every((term) =>
        term.formula.includes(`F32(${term.input} * ${term.learned.literal})`)));

      const relativeProjection = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_audio_features/audio_layer_0_relative_k_projection", outputCoordinate: [0, 0, 1],
      });
      assert.equal(relativeProjection.learnedScalars.length, 4);
      const queryScale = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_audio_features/audio_layer_0_q_scale", outputCoordinate: [0, 0, 1],
      });
      assert.equal(queryScale.learnedScalars.length, 1);
      assert.ok(queryScale.formula.includes(queryScale.learnedScalars[0]!.literal));
      assert.equal(queryScale.denseDecoderLanguage.bitLanguageId, "u32-bit-expression-v1");
      assert.equal(queryScale.storageDecoders.length, 1);
      assert.equal(queryScale.storageDecoders[0]!.decode.schemaVersion, 2);
      const depthwise = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_audio_features/audio_layer_0_conv_depthwise", outputCoordinate: [0, 2, 1],
      });
      assert.equal(depthwise.learnedScalars.length, 5);
      assert.equal(depthwise.reduction?.complete, true);
      assert.equal(depthwise.terms?.length, 5);

      await assert.rejects(() => renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_audio_features/audio_layer_0_attention", outputCoordinate: [0, 0, 1],
      }), /redução pytorch-cpu-f32-matmul ainda não possui agenda literal comprovada/);
      await assert.rejects(() => renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_audio_features/audio_layer_0_attention_content_scores", outputCoordinate: [0, 0, 0, 0, 0],
      }), /redução pytorch-cpu-f32-matmul ainda não possui agenda literal comprovada/);
      await assert.rejects(() => renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_audio_features/audio_layer_0_attention_position_scores", outputCoordinate: [0, 0, 0, 0, 0],
      }), /redução pytorch-cpu-f32-matmul ainda não possui agenda literal comprovada/);
      const shiftedAudioPosition = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_audio_features/audio_layer_0_attention_relative_shift", outputCoordinate: [0, 0, 0, 0, 0],
      });
      assert.ok(shiftedAudioPosition.scalarAssignments.some((formula) => formula.includes("AUDIO_RELATIVE_SHIFT_SOURCE")));
      assert.doesNotMatch(shiftedAudioPosition.formula, /composite_audio_features\/composite_audio_features/);
      const audioLogit = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_audio_features/audio_layer_0_attention_logit_add", outputCoordinate: [0, 0, 0, 0, 0],
      });
      assert.ok(audioLogit.formula.includes("attention_ac") && audioLogit.formula.includes("attention_bd"));
      const audioSoftcap = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_audio_features/audio_layer_0_attention_softcap", outputCoordinate: [0, 0, 0, 0, 0],
      });
      assert.ok(audioSoftcap.formula.includes("SLEEF_TANH_F32"));
      const audioMask = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_audio_features/audio_layer_0_attention_mask", outputCoordinate: [0, 0, 0, 0, 0],
      });
      assert.ok(audioMask.scalarAssignments.some((formula) => formula.includes("query_index-key_index")));
      const audioSoftmax = await renderGemma4LiteralMultimodalScalarView(artifact, {
        operationId: "composite_audio_features/audio_layer_0_attention_softmax", outputCoordinate: [0, 0, 0, 0, 0],
      });
      assert.ok(audioSoftmax.scalarAssignments.some((formula) => formula.includes("ORDERED_F32_REDUCE_MAX")));
      assert.ok(audioSoftmax.scalarAssignments.some((formula) => formula.includes("ORDERED_F32_REDUCE_SUM")));

      // The real BF16 package dispatches both vision BMM classes to the same
      // unresolved Apple SGEMM boundary. Promote every compatible fixture
      // instance together so this test covers the full five-class contract
      // without introducing any layer-ID allowlist.
      const visionNativeBmm = artifact.calculationGraph.assignments.filter((assignment) =>
        assignment.scope === "vision" &&
        (assignment.operation === "attention-score-matmul" || assignment.operation === "attention-value-matmul"));
      const originalVisionCalculations = visionNativeBmm.map((assignment) => structuredClone(assignment.scalarCalculation));
      const originalVisionOutputPolicies = visionNativeBmm.map((assignment) => structuredClone(assignment.outputDomain.dtypePolicy));
      try {
        for (const assignment of visionNativeBmm) {
          assignment.scalarCalculation.dtypePolicy = {
            inputDtype: "BF16", computeDtype: "pytorch-native-batched-matmul",
            accumulationDtype: "runtime-defined", outputDtype: "BF16",
          };
          assignment.scalarCalculation.reduction = {
            ...assignment.scalarCalculation.reduction!, order: "runtime-defined",
          };
          delete assignment.scalarCalculation.reduction.schedule;
          assignment.scalarCalculation.reproducibility = "fail-closed-runtime-reduction";
          assignment.outputDomain.dtypePolicy = structuredClone(assignment.scalarCalculation.dtypePolicy);
        }
        const runtimeReductions = listGemma4LiteralRuntimeReductionOperations(artifact);
        assert.deepEqual(new Set(runtimeReductions.map((entry) => entry.operationClass)), new Set([
          "vision-attention-score", "vision-attention-value", "audio-content-attention-score",
          "audio-position-attention-score", "audio-attention-value",
        ]));
        assert.equal(runtimeReductions.length, visionNativeBmm.length + program.audioProgram.tower.layers * 3);
        assert.ok(runtimeReductions.every((entry) => entry.provider === "Apple Accelerate SGEMM" &&
          entry.scalarSchedule === "unpublished-fail-closed" &&
          entry.auditability === "operand-products-addressable-reduction-fail-closed"));
        const fidelityGate = buildGemma4LiteralFidelityGate(artifact.calculationGraph, artifact.authoritativeExecution);
        assert.equal(fidelityGate.status, "blocked-on-runtime-reduction");
        assert.equal(fidelityGate.exactReplayClaim, "forbidden");
        assert.equal(fidelityGate.unresolvedNativeReductionCount, runtimeReductions.length);
        assert.deepEqual(
          fidelityGate.unresolvedNativeReductions.map((entry) => entry.operationId),
          runtimeReductions.map((entry) => entry.operationId),
        );

        const imageScoreAudit = renderGemma4LiteralRuntimeReductionAudit(artifact, {
          operationId: "composite_image_features/vision_layer_0_attention_scores", outputCoordinate: [0, 0, 0, 0],
        });
        assert.equal(imageScoreAudit.status, "fail-closed-runtime-reduction");
        assert.equal(imageScoreAudit.reduction.complete, true);
        assert.equal(imageScoreAudit.terms.length, program.visionProgram.tower.headDim);
        assert.equal(imageScoreAudit.terms[1]?.leftOperand, "composite_image_features/vision_layer_0_q_rotated[0,0,0,1]");
        assert.equal(imageScoreAudit.terms[1]?.rightOperand, "composite_image_features/vision_layer_0_k_rotated[0,0,0,1]");
        assert.deepEqual(imageScoreAudit.renderedOutputCoordinate.write, {
          kind: "tensor-element",
          expression: "composite_image_features/vision_layer_0_attention_scores[0,0,0,0]",
          coordinates: ["0", "0", "0", "0"],
          coordinatePrograms: Array.from({ length: 4 }, () => ({ kind: "constant" as const, value: 0 })),
        });
        assert.deepEqual(imageScoreAudit.predecessorCoordinates.map((entry) => [
          entry.producerOperationId, entry.renderedCoverage, entry.renderedAccesses.filter((access) => access.kind === "tensor-element").length,
        ]), [
          ["composite_image_features/vision_layer_0_q_rope", "complete", program.visionProgram.tower.headDim + 1],
          ["composite_image_features/vision_layer_0_k_rope", "complete", program.visionProgram.tower.headDim + 1],
        ]);
        assert.match(imageScoreAudit.nonExecutableResult, /UNPUBLISHED_REDUCTION/);
        const originalVisionHeadDim = artifact.program.visionProgram.tower.headDim;
        artifact.program.visionProgram.tower.headDim = originalVisionHeadDim + 1;
        try {
          const metadataIndependentScoreAudit = renderGemma4LiteralRuntimeReductionAudit(artifact, {
            operationId: "composite_image_features/vision_layer_0_attention_scores", outputCoordinate: [0, 0, 0, 0],
          });
          assert.deepEqual(metadataIndependentScoreAudit.terms, imageScoreAudit.terms,
            "runtime-reduction audit must execute the serialized scalar program instead of tower metadata");
        } finally {
          artifact.program.visionProgram.tower.headDim = originalVisionHeadDim;
        }

        assert.throws(() => renderGemma4LiteralRuntimeReductionAudit(artifact, {
          operationId: "composite_video_features/vision_layer_0_attention", outputCoordinate: [0, 0, 0],
        }), /redução dinâmica requer inputStart\/inputCount/);
        const videoValueAudit = renderGemma4LiteralRuntimeReductionAudit(artifact, {
          operationId: "composite_video_features/vision_layer_0_attention", outputCoordinate: [0, 0, 0],
          inputStart: 1, inputCount: 2,
        });
        assert.equal(videoValueAudit.operationClass, "vision-attention-value");
        assert.equal(videoValueAudit.reduction.complete, false);
        assert.ok(videoValueAudit.predecessorCoordinates.every((entry) => entry.renderedCoverage === "windowed"));
        assert.deepEqual(videoValueAudit.reduction.renderedWindow, { startInclusive: 1, endExclusive: 3 });
        assert.equal(videoValueAudit.terms[0]?.leftOperand, "composite_video_features/vision_layer_0_attention_weights[0,0,0,1]");

        const audioContentAudit = renderGemma4LiteralRuntimeReductionAudit(artifact, {
          operationId: "composite_audio_features/audio_layer_0_attention_content_scores", outputCoordinate: [0, 0, 0, 0, 0],
        });
        assert.equal(audioContentAudit.operationClass, "audio-content-attention-score");
        assert.ok(audioContentAudit.terms.every((term) => term.predicate?.includes("key_index")));
        const audioPositionAudit = renderGemma4LiteralRuntimeReductionAudit(artifact, {
          operationId: "composite_audio_features/audio_layer_0_attention_position_scores", outputCoordinate: [0, 0, 0, 0, 0],
        });
        assert.equal(audioPositionAudit.operationClass, "audio-position-attention-score");
        assert.ok(audioPositionAudit.terms.every((term) => term.rightOperand.includes("relative_keys")));
        const audioValueAudit = renderGemma4LiteralRuntimeReductionAudit(artifact, {
          operationId: "composite_audio_features/audio_layer_0_attention", outputCoordinate: [0, 0, 0],
        });
        assert.equal(audioValueAudit.operationClass, "audio-attention-value");
        assert.equal(audioValueAudit.reduction.complete, true);
        assert.ok(audioValueAudit.coordinateAssignments.some((entry) => entry.startsWith("key_index=") && entry.includes("key_slot")));
        assert.ok(audioValueAudit.terms.every((term) => term.mathematicalProduct.includes("REAL_PRODUCT")));
        assert.throws(() => renderGemma4LiteralRuntimeReductionAudit(artifact, {
          operationId: "composite_audio_features/audio_layer_0_q_scale", outputCoordinate: [0, 0, 0],
        }), /não possui redução runtime-defined/);
      } finally {
        visionNativeBmm.forEach((assignment, index) => {
          assignment.scalarCalculation = originalVisionCalculations[index]!;
          assignment.outputDomain.dtypePolicy = originalVisionOutputPolicies[index]!;
        });
      }

      const duplicateId = artifact.program.visionProgram.assignments[1]!;
      const originalId = duplicateId.id;
      const serializedIds = listGemma4LiteralOperations(artifact).map((operation) => operation.operationId);
      duplicateId.id = artifact.program.visionProgram.assignments[0]!.id;
      assert.deepEqual(listGemma4LiteralOperations(artifact).map((operation) => operation.operationId), serializedIds,
        "source-removed navigation must consume the serialized graph instead of reconstructing program expansion");
      duplicateId.id = originalId;

      const imageSlice = buildGemma4LiteralCalculationSlice(artifact, "composite_image_scatter");
      assert.equal(imageSlice.sourceCheckpointAccessed, false);
      assert.equal(imageSlice.lastOperationId, "composite_image_scatter");
      assert.equal(imageSlice.operationCount, imageSlice.operations.length);
      assert.ok(imageSlice.operations.some((operation) => operation.operationId === "composite_image_features/vision_patch_projection"));
      assert.ok(imageSlice.operations.every((operation) => operation.invocationId !== "composite_video_features" && operation.scope !== "audio"));
      const sliceOrdinals = new Map(imageSlice.operations.map((operation, index) => [operation.operationId, index]));
      assert.ok(imageSlice.operations.every((operation) => operation.predecessors.every((predecessor) =>
        predecessor.producerOperationId === undefined || sliceOrdinals.get(predecessor.producerOperationId)! < sliceOrdinals.get(operation.operationId)!)));
      assert.ok(imageSlice.externalInputs.some((input) => input.name === "pixel_values"));
      assert.ok(imageSlice.learnedConstants.some((constant) => constant.consumers.some((consumer) =>
        consumer.operationId === "composite_image_features/vision_patch_projection" && consumer.role === "weight")));
      assert.ok(imageSlice.learnedConstants.every((constant) => constant.decoderId === `decode_${constant.tensor.name}`));
      assert.ok(imageSlice.learnedConstants.every((constant) =>
        constant.decoder.id === constant.decoderId && constant.decoder.address.kind === "row-major-dense-element-address" &&
        constant.decoder.decode.kind === "ieee-binary32-bitcast"));
      assert.deepEqual(imageSlice.denseDecoderLanguage, artifact.denseDecoderLanguage);
      assert.deepEqual(imageSlice.transcendentalPrograms, artifact.transcendentalPrograms);
      assert.deepEqual(imageSlice.formulaLanguage, artifact.formulaLanguage);
      assert.deepEqual(imageSlice.dimensionLanguage, artifact.calculationDomains.dimensionLanguage);
      assert.deepEqual(imageSlice.dimensionPrograms, artifact.calculationDomains.dimensionPrograms);
      const boundImageScore = imageSlice.operations.find((operation) =>
        operation.operationId === "composite_image_features/vision_layer_0_attention_scores")!;
      assert.deepEqual(boundImageScore.scalarCalculation.reduction?.domains[0]?.endExclusive, {
        kind: "constant", value: program.visionProgram.tower.headDim,
      });
      assert.ok(imageSlice.numericLiterals.some((literal) => literal.token === "0.5" && /^0x[0-9a-f]{8}$/.test(literal.binary32Hex)));
      assert.equal(imageSlice.reproducibility.status, "literal");
      assert.deepEqual(imageSlice.reproducibility.failClosedOperationIds, []);
      const terminalOperationId = allOperations.at(-1)!.operationId;
      const logitsSlice = buildGemma4LiteralCalculationSlice(artifact, terminalOperationId);
      assert.equal(logitsSlice.lastOperationId, terminalOperationId);
      assert.equal(logitsSlice.operationCount, allOperations.length);
      assert.equal(logitsSlice.reproducibility.status, "fail-closed-runtime-reduction");
      assert.ok(logitsSlice.reproducibility.failClosedOperationIds.some((id) => id.includes("audio_layer_0_attention")));
      assert.throws(() => buildGemma4LiteralCalculationSlice(artifact, "unknown-operation"), /não encontrada para slice/);

      const endToEnd = buildGemma4LiteralEndToEndCalculation(artifact, 2);
      assert.equal(endToEnd.sourceCheckpointAccessed, false);
      assert.equal(endToEnd.maxNewTokens, 2);
      assert.equal(endToEnd.forward.targetOutput, artifact.outputs.logits);
      assert.equal(endToEnd.forward.operationCount, allOperations.length);
      assert.equal(endToEnd.generation.operations.length, 22);
      assert.ok(endToEnd.generation.operations.some((operation) =>
        operation.navigation.operationId === "generation_argmax[1]" &&
        operation.scalarAssignments.some((formula) => formula.includes("candidate[v]"))));
      assert.ok(endToEnd.generation.operations.filter((operation) => operation.forwardExpansion).every((operation) =>
        !("declaredForwardOperations" in operation)));
      assert.equal(endToEnd.generation.cacheTransitions.length, program.textProgram.layers.length);
      assert.ok(endToEnd.declaredInputs.some((input) => input.name === "max_new_tokens" && input.requiredFor.includes("generation")));
      assert.deepEqual(endToEnd.inputContract, artifact.inputContract);
      assert.deepEqual(endToEnd.outputContract, artifact.outputContract);
      assert.equal(endToEnd.numericLiterals.length, artifact.numericLiterals.literals.length);
      assert.equal(endToEnd.storageCoverage.complete, true);
      assert.equal(
        endToEnd.storageCoverage.reachableLearnedConstantCount + endToEnd.storageCoverage.runtimeUnreachableConstantCount,
        endToEnd.storageCoverage.embeddedConstantCount,
      );
      assert.ok(endToEnd.storageCoverage.runtimeUnreachableConstants.every((entry) =>
        entry.decoderId === `decode_${entry.tensor.name}` && entry.decoder.id === entry.decoderId &&
        entry.decoder.address.kind === "row-major-dense-element-address" &&
        entry.reason === "shared-kv-consumer-local-kv-is-runtime-unreachable"));
      assert.equal(endToEnd.reproducibility.generationControl, "literal");
      assert.equal(endToEnd.reproducibility.status, "fail-closed-runtime-reduction");

      const hiddenConstant = endToEnd.storageCoverage.runtimeUnreachableConstants[0]?.tensor.name;
      if (hiddenConstant) {
        const declarationIndex = artifact.unreachableConstants.findIndex((entry) => entry.name === hiddenConstant);
        const declaration = artifact.unreachableConstants.splice(declarationIndex, 1)[0]!;
        assert.throws(() => buildGemma4LiteralEndToEndCalculation(artifact, 0), /não possui declaração runtime-unreachable e decoder/);
        artifact.unreachableConstants.splice(declarationIndex, 0, declaration);
      }
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
    }, output, fixtureSourceIdentity());
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
      assert.deepEqual(generation.assignmentExecutions.map((execution) => execution.assignmentId), [
        "generation_prefill", "generation_initial_position",
        "generation_selection_logits", "generation_argmax", "generation_token_append", "generation_position_advance",
        "generation_incremental_inputs", "generation_incremental_forward", "generation_logits_append", "generation_cache_append", "generation_eos_stop",
        "generation_selection_logits", "generation_argmax", "generation_token_append", "generation_position_advance",
        "generation_incremental_inputs", "generation_incremental_forward", "generation_logits_append", "generation_cache_append", "generation_eos_stop",
        "generation_terminal_logits", "generation_terminal_cache",
      ]);
      assert.deepEqual(generation.assignmentExecutions.map((execution) => execution.output), [
        "forward_state[0]", "position[-1]",
        "selection_logits[0]", "selected_token[0]", "generated_token_ids[0..0]", "position[0]",
        "incremental_inputs[0]", "forward_state[1]", "step_forward_logits[0..0]", "step_past_key_values[0..0]", "stop_after_step[0]",
        "selection_logits[1]", "selected_token[1]", "generated_token_ids[0..1]", "position[1]",
        "incremental_inputs[1]", "forward_state[2]", "step_forward_logits[0..1]", "step_past_key_values[0..1]", "stop_after_step[1]",
        "terminal_logits", "terminal_past_key_values",
      ]);
      assert.deepEqual(generation.stepForwardLogits[0], generation.selectionLogits[1]);
      assert.deepEqual(generation.stepForwardLogits.at(-1), generation.logits);
      assert.equal(generation.assignmentExecutions.find((execution) => execution.output === "selected_token[0]")?.value, generation.generatedTokenIds[0]);
      const eosGeneration = await generateGemma4PagedTextLiteralF32(artifact, {
        inputIds: [[1, 2, 3]], maxNewTokens: 2, eosTokenId: generation.generatedTokenIds[0]!,
      }, { maxReadBytes: 64, allowUnverifiedFidelity: true });
      assert.deepEqual(eosGeneration.generatedTokenIds, generation.generatedTokenIds.slice(0, 1));
      assert.deepEqual(eosGeneration.assignmentExecutions.map((execution) => execution.assignmentId), artifact.generation.assignments.map((assignment) => assignment.id));
      assert.equal(eosGeneration.assignmentExecutions.find((execution) => execution.assignmentId === "generation_eos_stop")?.value, true);
      assert.ok(eosGeneration.assignmentExecutions.findIndex((execution) => execution.assignmentId === "generation_incremental_forward") <
        eosGeneration.assignmentExecutions.findIndex((execution) => execution.assignmentId === "generation_eos_stop"));
      const zeroGeneration = await generateGemma4PagedTextLiteralF32(artifact, { inputIds: [[1, 2, 3]], maxNewTokens: 0 }, { maxReadBytes: 64, allowUnverifiedFidelity: true });
      assert.deepEqual(zeroGeneration.generatedTokenIds, []);
      assert.deepEqual(zeroGeneration.assignmentExecutions.map((execution) => execution.assignmentId), [
        "generation_prefill", "generation_initial_position", "generation_terminal_logits", "generation_terminal_cache",
      ]);
      assert.equal(zeroGeneration.logits, zeroGeneration.prefill.logits);

      const generationPlan = buildGemma4LiteralGenerationNavigation(artifact, 2);
      const generationNavigation = generationPlan.operations;
      assert.equal(generationPlan.sourceCheckpointAccessed, false);
      assert.equal(generationNavigation.length, 22);
      assert.deepEqual(generationNavigation.map((entry) => entry.operationId), [
        "generation_prefill", "generation_initial_position",
        "generation_selection_logits[0]", "generation_argmax[0]", "generation_token_append[0]", "generation_position_advance[0]",
        "generation_incremental_inputs[0]", "generation_incremental_forward[0]", "generation_logits_append[0]", "generation_cache_append[0]", "generation_eos_stop[0]",
        "generation_selection_logits[1]", "generation_argmax[1]", "generation_token_append[1]", "generation_position_advance[1]",
        "generation_incremental_inputs[1]", "generation_incremental_forward[1]", "generation_logits_append[1]", "generation_cache_append[1]", "generation_eos_stop[1]",
        "generation_terminal_logits", "generation_terminal_cache",
      ]);
      assert.deepEqual(
        generationNavigation.find((entry) => entry.operationId === "generation_argmax[1]")?.predecessors[0]?.producerOperationIds,
        ["generation_selection_logits[1]"],
      );
      assert.deepEqual(
        generationNavigation.find((entry) => entry.operationId === "generation_incremental_forward[1]")?.predecessors[0]?.producerOperationIds,
        ["generation_incremental_inputs[1]"],
      );
      assert.deepEqual(
        generationNavigation.find((entry) => entry.operationId === "generation_selection_logits[1]")?.conditionProducerOperationIds,
        ["generation_eos_stop[0]"],
      );
      assert.deepEqual(
        generationNavigation.find((entry) => entry.operationId === "generation_terminal_logits")?.predecessors[0]?.producerOperationIds,
        ["generation_prefill", "generation_incremental_forward[0]", "generation_incremental_forward[1]"],
      );
      const generationOrdinal = new Map(generationNavigation.map((entry) => [entry.operationId, entry.ordinal]));
      assert.ok(generationNavigation.every((entry) => entry.predecessors.every((predecessor) =>
        predecessor.producerOperationIds.every((producer) => generationOrdinal.get(producer)! < entry.ordinal))));
      assert.ok(generationNavigation.every((entry) => entry.conditionProducerOperationIds.every((producer) =>
        generationOrdinal.get(producer)! < entry.ordinal)));
      assert.ok(generationNavigation.every((entry) => entry.consumers.every((consumer) =>
        generationOrdinal.get(consumer)! > entry.ordinal)));
      const prefillNavigation = generationNavigation[0]!;
      assert.equal(prefillNavigation.forwardExpansion?.firstOperationId, "composite_block_sequence_ids");
      assert.equal(prefillNavigation.forwardExpansion?.lastOperationId, program.textProgram.epilogue.at(-1)?.id);
      assert.equal(prefillNavigation.forwardExpansion?.sourceCheckpointAccessed, false);
      assert.equal(prefillNavigation.forwardExpansion?.operationCount, generationPlan.declaredForwardOperations.length);
      assert.ok(generationPlan.declaredForwardOperations.length > program.assignments.length);
      const argmaxView = renderGemma4LiteralGenerationCalculationView(artifact, 2, "generation_argmax[0]");
      assert.equal(argmaxView.sourceCheckpointAccessed, false);
      assert.ok(argmaxView.scalarAssignments.some((formula) => formula.includes("v=0..5")));
      assert.ok(argmaxView.formula.includes("lowest"));
      const incrementalView = renderGemma4LiteralGenerationCalculationView(artifact, 2, "generation_incremental_forward[1]");
      assert.equal(incrementalView.forwardExpansion?.mode, "cached-incremental");
      assert.equal(incrementalView.declaredForwardOperations?.length, generationPlan.declaredForwardOperations.length);
      assert.ok(incrementalView.formula.includes("incremental_past_key_values[1]"));
      assert.ok(incrementalView.scalarAssignments.some((formula) => formula.includes("generation.forwardCalculation.operationOrder")));
      assert.ok(incrementalView.scalarAssignments.some((formula) => formula.includes("generation.forwardCalculation.cacheTransitions")));
      assert.doesNotMatch(incrementalView.formula, /declared_cached_incremental_forward|generic_decoder/);
      const zeroNavigation = buildGemma4LiteralGenerationNavigation(artifact, 0);
      assert.deepEqual(zeroNavigation.operations.map((entry) => entry.operationId), [
        "generation_prefill", "generation_initial_position", "generation_terminal_logits", "generation_terminal_cache",
      ]);
      const zeroTerminal = renderGemma4LiteralGenerationCalculationView(artifact, 0, "generation_terminal_logits");
      assert.equal(zeroTerminal.scalarAssignments[0], "executed_steps = 0 because max_new_tokens = 0");
      assert.throws(() => buildGemma4LiteralGenerationNavigation(artifact, -1), /maxNewTokens inteiro não negativo/);
      assert.throws(() => renderGemma4LiteralGenerationCalculationView(artifact, 1, "generation_argmax[1]"), /não encontrada/);
      await assert.rejects(
        () => generateGemma4PagedTextLiteralF32(artifact, { inputIds: [[1]], maxNewTokens: 1, pastKeyValues: replay.pastKeyValues }, { maxReadBytes: 64, allowUnverifiedFidelity: true }),
        /começa em prefill sem pastKeyValues/,
      );
      await assert.rejects(
        () => generateGemma4PagedTextLiteralF32(artifact, {
          inputIds: [[1]], maxNewTokens: 1, attentionMask: { shape: [1, 1, 1, 1], values: Float32Array.of(0) },
        }, { maxReadBytes: 64, allowUnverifiedFidelity: true }),
        /não aceita máscara externa/,
      );
      await assert.rejects(
        () => generateGemma4PagedTextLiteralF32(artifact, {
          inputIds: [[1]], positionIds: [[Number.MAX_SAFE_INTEGER]], maxNewTokens: 1,
        }, { maxReadBytes: 64, allowUnverifiedFidelity: true }),
        /avanço de posição excede inteiro seguro/,
      );
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

test("paged linear keeps the BFDOT adjacent-pair lane contract distinct from modulo lanes", async () => {
  const tensor: TensorInfo = { name: "embedded://arm-neon-bfdot", storageDtype: "F32", storageShape: [1, 32], logicalShape: [1, 32] };
  const storage = Buffer.alloc(32 * 4);
  const values = new Float32Array(32);
  values[0] = 1e20; values[1] = -1e20; values[4] = 1; values[5] = 1;
  values.forEach((value, index) => storage.writeFloatLE(value, index * 4));
  const matrix = createPagedDenseF32Matrix(tensor, { async readTensorBytesRange(_tensor, offset, byteLength) { return storage.subarray(offset, offset + byteLength); } }, storage.length);
  const input = { shape: [1, 32], values: Float32Array.from({ length: 32 }, () => 1) };
  const bfdot = await pagedLinearF32(input, matrix, {
    accumulationDtype: "F32",
    reduction: { kind: "arm-neon-bf16-bfdot-fma", registerCount: 8, activeRegisterCount: 4, lanesPerRegister: 4, termsPerLane: 2, termsPerInstruction: 8, inputLane: "contiguous-bf16-pairs", horizontalFold: "pairwise" },
  });
  const modulo = await pagedLinearF32(input, matrix, {
    accumulationDtype: "F32",
    reduction: { kind: "arm-neon-bf16-dot-fma", laneCount: 32, registerCount: 8, lanesPerRegister: 4, inputLane: "index-modulo-vector-lane-count", horizontalFold: "pairwise" },
  });
  assert.deepEqual(bfdot.values, Float32Array.from([2]));
  assert.deepEqual(modulo.values, Float32Array.from([0]));
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
    await writeGemma4CompositeLiteralCalculationProgram(program, catalog, reader, artifact, fixtureSourceIdentity());
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
        reductionProbeInput: { kind: "model-forward" } as { kind: "model-forward" } | { kind: "bf16-power-of-two-scale"; factor: number } | { kind: "bf16-scalar-scale"; factorBf16Bits: number },
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
    assert.deepEqual(report.reference.reductionProbeInput, { kind: "model-forward" });
    assert.deepEqual(report.exactProfileIds, ["ordered-f32"]);
    assert.deepEqual(report.candidateSelection, { status: "unique", profileId: "ordered-f32" });
    assert.equal(report.profiles[0]!.mismatchedElements, 0);
    const legacyInput = structuredClone(repeatedPayload);
    legacyInput.captureId = "fixture-capture-legacy-model-forward";
    Reflect.deleteProperty(legacyInput.reference, "reductionProbeInput");
    const legacyTrace = path.join(root, "trace-legacy-model-forward.json");
    await writeFile(legacyTrace, JSON.stringify(legacyInput), "utf8");
    const legacyReport = await probeGemma4LiteralLinearReductionProfiles({
      artifact, traces: [trace, legacyTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
      profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }],
    });
    assert.deepEqual(legacyReport.reference.reductionProbeInput, { kind: "model-forward" });
    const transformedInput = structuredClone(repeatedPayload);
    transformedInput.captureId = "fixture-capture-bf16-scale";
    transformedInput.reference.reductionProbeInput = { kind: "bf16-scalar-scale", factorBf16Bits: 0x3fc0 };
    const transformedTrace = path.join(root, "trace-bf16-scale.json");
    await writeFile(transformedTrace, JSON.stringify(transformedInput), "utf8");
    await assert.rejects(
      () => probeGemma4LiteralLinearReductionProfiles({ artifact, traces: [trace, transformedTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
        profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }], }),
      /contrato de referência diferente/,
    );
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
    assert.deepEqual(campaign.candidateSelection, { status: "unique", profileId: "ordered-f32" });
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
    const none = await probeGemma4LiteralLinearReductionProfiles({
      artifact, traces: [incompatibleFeatureTrace, incompatibleFeatureRepeatTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
      profiles: [{ id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } }],
    });
    assert.deepEqual(none.exactProfileIds, []);
    assert.deepEqual(none.candidateSelection, { status: "none" });
    const ambiguous = await probeGemma4LiteralLinearReductionProfiles({
      artifact, traces: [trace, repeatedTrace], operationId: target.id, maxReadBytes: 1024 * 1024,
      profiles: [
        { id: "ordered-f32-a", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } },
        { id: "ordered-f32-b", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } },
      ],
    });
    assert.deepEqual(ambiguous.exactProfileIds, ["ordered-f32-a", "ordered-f32-b"]);
    assert.deepEqual(ambiguous.candidateSelection, { status: "ambiguous" });
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
  }, fixtureSourceIdentity());
  assert.equal(literal.constants.length, catalog.tensors.size);
  assert.deepEqual(literal.unreachableConstants, [
    { name: "model.language_model.layers.2.self_attn.k_norm.weight", reason: "shared-kv-consumer-local-kv-is-runtime-unreachable", producerLayer: 0 },
    { name: "model.language_model.layers.2.self_attn.k_proj.weight", reason: "shared-kv-consumer-local-kv-is-runtime-unreachable", producerLayer: 0 },
    { name: "model.language_model.layers.2.self_attn.v_proj.weight", reason: "shared-kv-consumer-local-kv-is-runtime-unreachable", producerLayer: 0 },
  ]);
  const sharedTransition = literal.generation.forwardCalculation.cacheTransitions.find((transition) => transition.layer === 2)!;
  assert.equal(sharedTransition.ownership, "reuse-producer");
  assert.equal(sharedTransition.producerLayer, 0);
  assert.equal(sharedTransition.prefill.mode, "reuse-producer");
  assert.equal(sharedTransition.incremental.mode, "reuse-producer");
  if (sharedTransition.prefill.mode === "reuse-producer" && sharedTransition.incremental.mode === "reuse-producer") {
    assert.equal(sharedTransition.prefill.emitsCacheEntry, false);
    assert.equal(sharedTransition.incremental.emitsCacheEntry, false);
    assert.ok(sharedTransition.prefill.scalarAssignments.some((formula) =>
      formula.includes("attention_layer_2.key") && formula.includes("past_key_values[0].key")));
    assert.ok(sharedTransition.incremental.scalarAssignments.some((formula) => formula.includes("cache_entry_present[2]=BOOL(false)")));
  }
  const corruptedTransition = structuredClone(literal);
  corruptedTransition.generation.forwardCalculation.cacheTransitions.find((transition) => transition.layer === 2)!
    .incremental.scalarAssignments[0] = "reuse an unspecified cache";
  assert.throws(() => validateGemma4CompositeLiteralCalculationProgram(corruptedTransition), /transições de geração greedy incompletas/);
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
