import assert from "node:assert/strict";
import test from "node:test";
import { executeGemma4RuntimeReduction, gemma4RuntimeReductionExecutionProtocolSha256,
  type Gemma4RuntimeReductionRequest } from "../../src/gemma4-runtime-reduction-provider.js";
import { GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256, GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY,
  GEMMA4_RUNTIME_REDUCTION_EXECUTION_STATE, GEMMA4_RUNTIME_REDUCTION_PROCESS_ENVIRONMENT,
  gemma4AuthoritativeExecutionContract, loadGemma4RuntimeReductionAdapterProgram } from "../../src/gemma4-authoritative-runtime.js";
import { Gemma4TorchRuntimeReductionProvider } from "../../src/gemma4-torch-runtime-reduction-provider.js";
import { gemma4RuntimeReductionInvocationPrograms } from "../../src/gemma4-runtime-reduction-invocation.js";
import { fixtureProgram, fixtureAudioProgram, sampleTensor, tensorValues } from "../support/gemma4-runtime-reduction-fixture.js";

// Requires the exact runtime identity pinned by the authoritative Gemma contract.
test("Gemma 4 runtime-reduction provider executes the integrity-bound embedded adapter", async () => {
  const request = {
    scope: "vision" as const,
    operationId: "vision_layer_0_attention_scores",
    operation: "attention-score-matmul" as const,
    program: fixtureProgram(),
    operands: [sampleTensor([1, 1, 2, 2]), sampleTensor([1, 1, 2, 2])] as const,
  };
  const replayContract = gemma4AuthoritativeExecutionContract(
    await loadGemma4RuntimeReductionAdapterProgram(),
  ).unresolvedNativeReduction.executableReplay;
  const provider = new Gemma4TorchRuntimeReductionProvider("venv/bin/python", replayContract);
  const previousOmpThreads = process.env.OMP_NUM_THREADS;
  const previousTf32Override = process.env.TORCH_ALLOW_TF32_CUBLAS_OVERRIDE;
  const previousVeclibThreads = process.env.VECLIB_MAXIMUM_THREADS;
  const previousDyldLibraries = process.env.DYLD_INSERT_LIBRARIES;
  process.env.OMP_NUM_THREADS = "1";
  process.env.TORCH_ALLOW_TF32_CUBLAS_OVERRIDE = "1";
  process.env.VECLIB_MAXIMUM_THREADS = "1";
  process.env.DYLD_INSERT_LIBRARIES = "/invalid/parent-only.dylib";
  try {
    assert.deepEqual(executeGemma4RuntimeReduction(provider, request, [1, 1, 2, 2]), sampleTensor([1, 1, 2, 2]));
  } finally {
    if (previousOmpThreads === undefined) delete process.env.OMP_NUM_THREADS;
    else process.env.OMP_NUM_THREADS = previousOmpThreads;
    if (previousTf32Override === undefined) delete process.env.TORCH_ALLOW_TF32_CUBLAS_OVERRIDE;
    else process.env.TORCH_ALLOW_TF32_CUBLAS_OVERRIDE = previousTf32Override;
    if (previousVeclibThreads === undefined) delete process.env.VECLIB_MAXIMUM_THREADS;
    else process.env.VECLIB_MAXIMUM_THREADS = previousVeclibThreads;
    if (previousDyldLibraries === undefined) delete process.env.DYLD_INSERT_LIBRARIES;
    else process.env.DYLD_INSERT_LIBRARIES = previousDyldLibraries;
  }
  assert.equal(provider.executions.length, 1);
  assert.equal(provider.executions[0]!.adapterProgramSha256, GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256);
  assert.equal(provider.executions[0]!.executionProtocolSha256,
    gemma4RuntimeReductionExecutionProtocolSha256(replayContract.executionProtocol));
  assert.deepEqual(provider.executions[0]!.runtimeAttestation.runtimeProcessEnvironment,
    GEMMA4_RUNTIME_REDUCTION_PROCESS_ENVIRONMENT);
  assert.deepEqual(provider.executions[0]!.runtimeAttestation.runtimeEnvironmentIdentity,
    GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY);
  assert.deepEqual(provider.executions[0]!.runtimeAttestation.runtimeExecutionState,
    GEMMA4_RUNTIME_REDUCTION_EXECUTION_STATE);
  assert.equal(provider.executions[0]!.runtimeAttestation.runtimeExecutionState.float32MatmulPrecision, "highest");
  assert.deepEqual(provider.executions[0]!.runtimeAttestation.runtimeExecutionState.subnormalProbe, {
    encoding: "ieee-f32-little-endian",
    inputBits: 1,
    multipliedByOneBits: 1,
  });
});

test("Gemma 4 embedded adapter executes every serialized invocation program without hidden class transforms", async () => {
  const replayContract = gemma4AuthoritativeExecutionContract(
    await loadGemma4RuntimeReductionAdapterProgram(),
  ).unresolvedNativeReduction.executableReplay;
  const provider = new Gemma4TorchRuntimeReductionProvider("venv/bin/python", replayContract);
  const vision = fixtureProgram();
  vision.assignments.push({
    id: "vision_layer_0_attention",
    operation: "attention-value-matmul",
    inputs: ["attention_weights", "value"],
    output: "vision_layer_0_attention",
    dtypePolicy: { inputDtype: "BF16", computeDtype: "pytorch-native-batched-matmul", accumulationDtype: "runtime-defined", outputDtype: "BF16" },
  });
  const audio = fixtureAudioProgram();
  const matrix = tensorValues([1, 1, 2, 2], [1, 2, 3, 4]);
  const other = tensorValues([1, 1, 2, 2], [5, 6, 7, 8]);
  const identity = tensorValues([1, 1, 1, 2, 2], [1, 0, 0, 1]);
  const sequence = tensorValues([1, 2, 2], [5, 6, 7, 8]);
  const cases: Array<{ request: Gemma4RuntimeReductionRequest; shape: number[]; values: number[] }> = [
    { request: { scope: "vision", operationId: "vision_layer_0_attention_scores", operation: "attention-score-matmul", program: vision, operands: [matrix, other] }, shape: [1, 1, 2, 2], values: [17, 23, 39, 53] },
    { request: { scope: "vision", operationId: "vision_layer_0_attention", operation: "attention-value-matmul", program: vision, operands: [matrix, other] }, shape: [1, 2, 2], values: [19, 22, 43, 50] },
    { request: { scope: "audio", operationId: "audio_content", operation: "chunked-attention-content-matmul", program: audio, operands: [tensorValues([1, 2, 2], [1, 2, 3, 4]), sequence] }, shape: [1, 1, 1, 2, 2], values: [17, 23, 39, 53] },
    { request: { scope: "audio", operationId: "audio_position", operation: "relative-attention-position-matmul", program: audio, operands: [tensorValues([1, 2, 2], [1, 2, 3, 4]), tensorValues([1, 3, 2], [1, 0, 0, 1, 1, 1])] }, shape: [1, 1, 1, 2, 3], values: [1, 2, 3, 3, 4, 7] },
    { request: { scope: "audio", operationId: "audio_value", operation: "chunked-relative-attention-values", program: audio, operands: [identity, sequence] }, shape: [1, 2, 2], values: [5, 6, 7, 8] },
  ];
  for (const entry of cases) {
    const result = executeGemma4RuntimeReduction(provider, entry.request, entry.shape);
    assert.deepEqual([...result.values], entry.values);
  }
  assert.deepEqual(provider.executions.map((entry) => entry.invocationProgramId), gemma4RuntimeReductionInvocationPrograms().map((program) => program.id));
  assert.ok(provider.executions.every((entry) => /^[a-f0-9]{64}$/.test(entry.invocationProgramSha256)));
});
