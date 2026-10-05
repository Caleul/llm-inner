import { fixtureProgram, fixtureAudioProgram, sampleTensor, tensorValues } from "./support/gemma4-runtime-reduction-fixture.js";
import assert from "node:assert/strict";
import test from "node:test";
import {
  buildGemma4RuntimeReductionTraceCoverage,
  gemma4RuntimeReductionOperationIds,
  validateGemma4RuntimeReductionTraceCoverage,
} from "../src/gemma4-runtime-reduction-trace-coverage.js";
import {
  buildGemma4RuntimeReductionReplayEvidence,
  executeGemma4RuntimeReduction,
  expectedGemma4RuntimeReductionAttestation,
  gemma4RuntimeReductionExecutionProtocolSha256,
  gemma4RuntimeReductionTensorEvidence,
  validateGemma4RuntimeReductionReplayEvidence,
  type Gemma4RuntimeReductionExecution,
  type Gemma4RuntimeReductionProvider,
  type Gemma4RuntimeReductionRequest,
} from "../src/gemma4-runtime-reduction-provider.js";
import type { DifferentialOperationSample } from "../src/types.js";
import {
  GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256,
  GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY,
  GEMMA4_RUNTIME_REDUCTION_EXECUTION_STATE,
  GEMMA4_RUNTIME_REDUCTION_PROCESS_ENVIRONMENT,
  gemma4AuthoritativeExecutionContract,
  gemma4RuntimeReductionExecutionProtocol,
  loadGemma4RuntimeReductionAdapterProgram,
} from "../src/gemma4-authoritative-runtime.js";
import {
  deserializeGemma4RuntimeReductionTensor,
  Gemma4TorchRuntimeReductionProvider,
  runtimeReductionLaunchContract,
  runtimeReductionSpawnEnvironment,
  serializeGemma4RuntimeReductionTensor,
} from "../src/gemma4-torch-runtime-reduction-provider.js";
import {
  gemma4RuntimeReductionInvocationProgram,
  gemma4RuntimeReductionInvocationProgramSha256,
  gemma4RuntimeReductionInvocationPrograms,
} from "../src/gemma4-runtime-reduction-invocation.js";
import {
  gemma4RuntimeReductionTranscriptEvidence,
  serializeGemma4RuntimeReductionRequestEnvelope,
  serializeGemma4RuntimeReductionResponseEnvelope,
} from "../src/gemma4-runtime-reduction-transport.js";

test("Gemma 4 native-reduction coverage binds both exact operands and native output in program order", () => {
  const program = fixtureProgram();
  const operations = fixtureOperations();
  const coverage = buildGemma4RuntimeReductionTraceCoverage(program, operations);

  assert.deepEqual(coverage.entries, [{
    operationId: "vision_layer_0_attention_scores",
    operation: "attention-score-matmul",
    output: "vision_layer_0_attention_scores",
    outputShape: [1, 1, 2, 2],
    orderedOperands: [
      { input: "vision_layer_0_q_rotated", producerOperationId: "vision_layer_0_q_rope", shape: [1, 1, 2, 2] },
      { input: "vision_layer_0_k_rotated", producerOperationId: "vision_layer_0_k_rope", shape: [1, 1, 2, 2] },
    ],
  }]);
  assert.deepEqual(gemma4RuntimeReductionOperationIds(program), ["vision_layer_0_attention_scores"]);
  validateGemma4RuntimeReductionTraceCoverage(coverage, program, operations);
});

test("Gemma 4 native-reduction coverage fails closed for an earlier or tampered operand checkpoint", () => {
  const program = fixtureProgram();
  const missingPostRope = fixtureOperations().filter((sample) => sample.operationId !== "vision_layer_0_q_rope");
  missingPostRope.push(sample("vision_layer_0_q_norm", "vision_layer_0_q_normalized", [1, 1, 2, 2]));
  assert.throws(
    () => buildGemma4RuntimeReductionTraceCoverage(program, missingPostRope),
    /não contém o operando nativo vision_layer_0_q_rotated/,
  );

  const operations = fixtureOperations();
  const coverage = buildGemma4RuntimeReductionTraceCoverage(program, operations);
  coverage.entries[0]!.orderedOperands[0]!.shape[3] = 3;
  assert.throws(
    () => validateGemma4RuntimeReductionTraceCoverage(coverage, program, operations),
    /cobertura de operandos BMM incompleta ou divergente/,
  );
});

test("Gemma 4 runtime-reduction provider accepts only its pinned contract and exact output shape", () => {
  const request = {
    scope: "vision" as const,
    operationId: "vision_layer_0_attention_scores",
    operation: "attention-score-matmul" as const,
    program: fixtureProgram(),
    operands: [sampleTensor([1, 1, 2, 2]), sampleTensor([1, 1, 2, 2])] as const,
  };
  const provider: Gemma4RuntimeReductionProvider = {
    contractId: "torch-2.12.1-cpu-inference-matmul-v1",
    evidenceContract: fixtureEvidenceContract(),
    executions: [],
    execute(actual) {
      assert.equal(actual, request);
      return execution(actual, sampleTensor([1, 1, 2, 2], 3.5));
    },
  };
  assert.deepEqual(executeGemma4RuntimeReduction(provider, request, [1, 1, 2, 2]), sampleTensor([1, 1, 2, 2], 3.5));

  const wrongShape: Gemma4RuntimeReductionProvider = { ...provider, execute: (actual) => execution(actual, sampleTensor([1, 1, 1, 2])) };
  assert.throws(() => executeGemma4RuntimeReduction(wrongShape, request, [1, 1, 2, 2]), /shape divergente/);
  const nonFinite: Gemma4RuntimeReductionProvider = { ...provider, execute: (actual) => execution(actual, { shape: [1, 1, 2, 2], values: Float32Array.of(0, 0, 0, Infinity) }) };
  assert.throws(() => executeGemma4RuntimeReduction(nonFinite, request, [1, 1, 2, 2]), /tensor.*inválido/);
  const wrongContract = { ...provider, contractId: "wrong" } as unknown as Gemma4RuntimeReductionProvider;
  assert.throws(() => executeGemma4RuntimeReduction(wrongContract, request, [1, 1, 2, 2]), /não corresponde ao contrato/);
  const wrongEvidenceContract = {
    ...provider,
    evidenceContract: { ...provider.evidenceContract, providerContractId: "wrong" },
  } as unknown as Gemma4RuntimeReductionProvider;
  assert.throws(
    () => executeGemma4RuntimeReduction(wrongEvidenceContract, request, [1, 1, 2, 2]),
    /contrato de evidência.*diverge/,
  );
  const wrongProtocolDigest: Gemma4RuntimeReductionProvider = {
    ...provider,
    evidenceContract: { ...provider.evidenceContract, executionProtocolSha256: "0".repeat(64) },
  };
  assert.throws(
    () => executeGemma4RuntimeReduction(wrongProtocolDigest, request, [1, 1, 2, 2]),
    /evidência.*divergente/,
  );
  const wrongTranscript: Gemma4RuntimeReductionProvider = {
    ...provider,
    execute(actual) {
      const result = execution(actual, sampleTensor([1, 1, 2, 2]));
      result.evidence.executionTranscript.requestSha256 = "0".repeat(64);
      return result;
    },
  };
  assert.throws(
    () => executeGemma4RuntimeReduction(wrongTranscript, request, [1, 1, 2, 2]),
    /evidência.*divergente/,
  );

  const wrongAttestation: Gemma4RuntimeReductionProvider = {
    ...provider,
    execute(actual) {
      const result = execution(actual, sampleTensor([1, 1, 2, 2]));
      result.evidence.runtimeAttestation.platform = "Linux-x86_64" as "Darwin-arm64";
      return result;
    },
  };
  assert.throws(() => executeGemma4RuntimeReduction(wrongAttestation, request, [1, 1, 2, 2]), /evidência.*divergente/);

  const wrongEnvironment: Gemma4RuntimeReductionProvider = {
    ...provider,
    execute(actual) {
      const result = execution(actual, sampleTensor([1, 1, 2, 2]));
      result.evidence.runtimeAttestation.runtimeEnvironmentIdentity = {
        ...result.evidence.runtimeAttestation.runtimeEnvironmentIdentity,
        operatingSystemBuild: "different-build",
      } as unknown as typeof GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY;
      return result;
    },
  };
  assert.throws(() => executeGemma4RuntimeReduction(wrongEnvironment, request, [1, 1, 2, 2]), /evidência.*divergente/);

  const wrongProcessEnvironment: Gemma4RuntimeReductionProvider = {
    ...provider,
    execute(actual) {
      const result = execution(actual, sampleTensor([1, 1, 2, 2]));
      result.evidence.runtimeAttestation.runtimeProcessEnvironment = {
        ...result.evidence.runtimeAttestation.runtimeProcessEnvironment,
        inheritance: "parent" as "none",
      };
      return result;
    },
  };
  assert.throws(() => executeGemma4RuntimeReduction(wrongProcessEnvironment, request, [1, 1, 2, 2]), /evidência.*divergente/);

  const wrongRuntimeBinary: Gemma4RuntimeReductionProvider = {
    ...provider,
    execute(actual) {
      const result = execution(actual, sampleTensor([1, 1, 2, 2]));
      const identity = structuredClone(result.evidence.runtimeAttestation.runtimeEnvironmentIdentity);
      const files = identity.runtimeDependencyIdentity.files as unknown as Array<(typeof identity.runtimeDependencyIdentity.files)[number]>;
      files[0] = {
        ...files[0]!,
        sha256: "0".repeat(64),
      } as (typeof identity.runtimeDependencyIdentity.files)[number];
      result.evidence.runtimeAttestation.runtimeEnvironmentIdentity = identity;
      return result;
    },
  };
  assert.throws(() => executeGemma4RuntimeReduction(wrongRuntimeBinary, request, [1, 1, 2, 2]), /evidência.*divergente/);

  const wrongPythonSources: Gemma4RuntimeReductionProvider = {
    ...provider,
    execute(actual) {
      const result = execution(actual, sampleTensor([1, 1, 2, 2]));
      const identity = structuredClone(result.evidence.runtimeAttestation.runtimeEnvironmentIdentity);
      const sourceTrees = identity.runtimeDependencyIdentity.pythonSourceTrees as unknown as Array<(typeof identity.runtimeDependencyIdentity.pythonSourceTrees)[number]>;
      sourceTrees[1] = {
        ...sourceTrees[1]!,
        sha256: "0".repeat(64),
      } as (typeof identity.runtimeDependencyIdentity.pythonSourceTrees)[number];
      result.evidence.runtimeAttestation.runtimeEnvironmentIdentity = identity;
      return result;
    },
  };
  assert.throws(() => executeGemma4RuntimeReduction(wrongPythonSources, request, [1, 1, 2, 2]), /evidência.*divergente/);

  const wrongExecutionState: Gemma4RuntimeReductionProvider = {
    ...provider,
    execute(actual) {
      const result = execution(actual, sampleTensor([1, 1, 2, 2]));
      result.evidence.runtimeAttestation.runtimeExecutionState = {
        ...result.evidence.runtimeAttestation.runtimeExecutionState,
        intraopThreads: 1,
      } as unknown as typeof GEMMA4_RUNTIME_REDUCTION_EXECUTION_STATE;
      return result;
    },
  };
  assert.throws(() => executeGemma4RuntimeReduction(wrongExecutionState, request, [1, 1, 2, 2]), /evidência.*divergente/);

  const wrongNumericState: Gemma4RuntimeReductionProvider = {
    ...provider,
    execute(actual) {
      const result = execution(actual, sampleTensor([1, 1, 2, 2]));
      result.evidence.runtimeAttestation.runtimeExecutionState = {
        ...result.evidence.runtimeAttestation.runtimeExecutionState,
        float32MatmulPrecision: "high",
      } as unknown as typeof GEMMA4_RUNTIME_REDUCTION_EXECUTION_STATE;
      return result;
    },
  };
  assert.throws(() => executeGemma4RuntimeReduction(wrongNumericState, request, [1, 1, 2, 2]), /evidência.*divergente/);

  const mismatchedProgram = fixtureProgram();
  mismatchedProgram.assignments[2]!.operation = "add";
  assert.throws(() => executeGemma4RuntimeReduction(provider, { ...request, program: mismatchedProgram }, [1, 1, 2, 2]), /não corresponde.*BMM/);
});

test("Gemma 4 runtime-reduction transport preserves exact finite IEEE-F32 bits", () => {
  const protocol = gemma4RuntimeReductionExecutionProtocol();
  const input = tensorValues([2, 2], [-0, 2 ** -149, 1, -2.5]);
  const serialized = serializeGemma4RuntimeReductionTensor(input, protocol.tensorEncoding);

  assert.deepEqual(Object.keys(serialized), protocol.tensorEncoding.fields);
  assert.equal(serialized.byteLength, 16);
  assert.equal(Buffer.from(serialized.dataBase64, "base64").toString("hex"), "00000080010000000000803f000020c0");
  const decoded = deserializeGemma4RuntimeReductionTensor(serialized, protocol.tensorEncoding);
  assert.deepEqual(decoded.shape, [2, 2]);
  assert.equal(Object.is(decoded.values[0], -0), true);
  assert.equal(decoded.values[1], 2 ** -149);
  assert.equal(decoded.values[2], 1);
  assert.equal(decoded.values[3], -2.5);

  const wrongLength = structuredClone(serialized);
  wrongLength.byteLength = 12;
  assert.throws(() => deserializeGemma4RuntimeReductionTensor(wrongLength, protocol.tensorEncoding), /shape e byteLength/);
  const nonCanonical = structuredClone(serialized);
  nonCanonical.dataBase64 = `${nonCanonical.dataBase64}\n`;
  assert.throws(() => deserializeGemma4RuntimeReductionTensor(nonCanonical, protocol.tensorEncoding), /envelope F32 incompatível/);
  assert.throws(() => deserializeGemma4RuntimeReductionTensor({
    shape: [2, 2], values: [-0, 2 ** -149, 1, -2.5],
  } as unknown as typeof serialized, protocol.tensorEncoding), /campos não correspondem/);
});

test("Gemma 4 runtime-reduction transcript binds the exact invocation environment", () => {
  const protocol = gemma4RuntimeReductionExecutionProtocol();
  const program = fixtureAudioProgram();
  const request: Gemma4RuntimeReductionRequest = {
    scope: "audio",
    operationId: "audio_content",
    operation: "chunked-attention-content-matmul",
    program,
    operands: [sampleTensor([1, 2, 2]), sampleTensor([1, 2, 2])],
  };
  const invocation = gemma4RuntimeReductionInvocationProgram(
    gemma4RuntimeReductionInvocationPrograms(), request.scope, request.operation,
  );
  const original = JSON.stringify(serializeGemma4RuntimeReductionRequestEnvelope(request, invocation, protocol));
  const changedProgram = structuredClone(program);
  changedProgram.tower.attentionContextRight = 1;
  const changed = JSON.stringify(serializeGemma4RuntimeReductionRequestEnvelope(
    { ...request, program: changedProgram }, invocation, protocol,
  ));
  const response = JSON.stringify(serializeGemma4RuntimeReductionResponseEnvelope(
    request, invocation, expectedGemma4RuntimeReductionAttestation(), sampleTensor([1, 1, 1, 2, 2]), protocol,
  ));

  const originalEvidence = gemma4RuntimeReductionTranscriptEvidence(original, response, protocol);
  const changedEvidence = gemma4RuntimeReductionTranscriptEvidence(changed, response, protocol);
  assert.notEqual(originalEvidence.requestSha256, changedEvidence.requestSha256);
  assert.equal(originalEvidence.responseSha256, changedEvidence.responseSha256);
  assert.throws(
    () => gemma4RuntimeReductionTranscriptEvidence(original, `${response}\n`, protocol),
    /JSON compacto canônico/,
  );
});

test("Gemma 4 runtime-reduction replay commits every receipt in artifact-declared operation order", () => {
  const protocol = gemma4RuntimeReductionExecutionProtocol();
  const request: Gemma4RuntimeReductionRequest = {
    scope: "vision",
    operationId: "vision_layer_0_attention_scores",
    operation: "attention-score-matmul",
    program: fixtureProgram(),
    operands: [sampleTensor([1, 1, 2, 2]), sampleTensor([1, 1, 2, 2])],
  };
  const first = execution(request, sampleTensor([1, 1, 2, 2])).evidence;
  const second = structuredClone(first);
  second.operationId = "vision_layer_1_attention_scores";
  const replay = buildGemma4RuntimeReductionReplayEvidence(
    "torch-2.12.1-cpu-inference-matmul-v1",
    [first, second],
    protocol,
  );

  assert.deepEqual(replay.operationIds, [first.operationId, second.operationId]);
  assert.equal(replay.executionOrder, "provider-append-order");
  assert.ok(replay.receiptCommitment.bytes > 0);
  assert.match(replay.receiptCommitment.sha256, /^[a-f0-9]{64}$/);
  validateGemma4RuntimeReductionReplayEvidence(replay, replay.operationIds, protocol);

  const reordered = structuredClone(replay);
  reordered.executions.reverse();
  assert.throws(
    () => validateGemma4RuntimeReductionReplayEvidence(reordered, replay.operationIds, protocol),
    /ordem completa de reduções/,
  );
  assert.throws(
    () => validateGemma4RuntimeReductionReplayEvidence(replay, [...replay.operationIds].reverse(), protocol),
    /ordem completa de reduções/,
  );
  const divergentProtocol = buildGemma4RuntimeReductionReplayEvidence(
    replay.contractId,
    replay.executions.map((entry, index) => index === 1
      ? { ...entry, executionProtocolSha256: "0".repeat(64) }
      : entry),
    protocol,
  );
  assert.throws(
    () => validateGemma4RuntimeReductionReplayEvidence(divergentProtocol, replay.operationIds, protocol),
    /ordem completa de reduções/,
  );
});

test("Gemma 4 runtime-reduction launch isolates its environment and rejects tampered contracts", async () => {
  const replayContract = gemma4AuthoritativeExecutionContract(
    await loadGemma4RuntimeReductionAdapterProgram(),
  ).unresolvedNativeReduction.executableReplay;
  const provider = new Gemma4TorchRuntimeReductionProvider("venv/bin/python", replayContract);
  assert.deepEqual(runtimeReductionLaunchContract(replayContract), {
    temporaryDirectoryPrefix: "llm-inner-gemma4-runtime-reduction-",
    adapterFileName: "embedded-runtime-reduction.py",
    requestFileName: "request.json",
    arguments: ["adapter-file", "request-file"],
    maxOutputBytes: 134_217_728,
  });
  const tamperedProtocol = structuredClone(replayContract);
  tamperedProtocol.executionProtocol.invocation.arguments.reverse();
  assert.throws(() => runtimeReductionLaunchContract(tamperedProtocol), /Protocolo de execução.*divergente/);
  assert.throws(() => new Gemma4TorchRuntimeReductionProvider("venv/bin/python", tamperedProtocol), /replay executável incompleto ou divergente/);
  assert.deepEqual(runtimeReductionSpawnEnvironment(replayContract), replayContract.runtimeProcessEnvironment.variables);
  const isolatedEnvironment = runtimeReductionSpawnEnvironment(replayContract);
  isolatedEnvironment.LANG = "mutated-after-materialization";
  assert.equal(replayContract.runtimeProcessEnvironment.variables.LANG, "C");
  const isolatedEvidence = provider.evidenceContract;
  (isolatedEvidence.runtimeAttestation.runtimeProcessEnvironment.variables as { LANG: string }).LANG = "mutated-reader-copy";
  assert.equal(provider.evidenceContract.runtimeAttestation.runtimeProcessEnvironment.variables.LANG, "C");
  assert.throws(() => runtimeReductionSpawnEnvironment({
    runtimeProcessEnvironment: {
      ...replayContract.runtimeProcessEnvironment,
      inheritance: "parent",
    } as unknown as typeof replayContract.runtimeProcessEnvironment,
  }), /não é um mapa fechado/);
});

function fixtureOperations(): DifferentialOperationSample[] {
  return [
    sample("vision_layer_0_q_rope", "vision_layer_0_q_rotated", [1, 1, 2, 2]),
    sample("vision_layer_0_k_rope", "vision_layer_0_k_rotated", [1, 1, 2, 2]),
    sample("vision_layer_0_attention_scores", "vision_layer_0_attention_scores", [1, 1, 2, 2]),
  ];
}

function sample(operationId: string, output: string, shape: number[]): DifferentialOperationSample {
  return { operationId, output, tensor: sampleTensor(shape) };
}

function execution(request: Gemma4RuntimeReductionRequest, output: ReturnType<typeof sampleTensor>): Gemma4RuntimeReductionExecution {
  const invocationProgram = gemma4RuntimeReductionInvocationProgram(
    gemma4RuntimeReductionInvocationPrograms(), request.scope, request.operation,
  );
  const executionProtocol = gemma4RuntimeReductionExecutionProtocol();
  const runtimeAttestation = expectedGemma4RuntimeReductionAttestation();
  const requestEnvelope = serializeGemma4RuntimeReductionRequestEnvelope(request, invocationProgram, executionProtocol);
  const responseEnvelope = serializeGemma4RuntimeReductionResponseEnvelope(
    request, invocationProgram, runtimeAttestation, output, executionProtocol,
  );
  return {
    output,
    evidence: {
      schemaVersion: 5,
      contractId: "torch-2.12.1-cpu-inference-matmul-v1",
      scope: request.scope,
      operationId: request.operationId,
      operation: request.operation,
      sourceCheckpointAccessed: false,
      adapterProgramSha256: GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256,
      executionProtocolSha256: gemma4RuntimeReductionExecutionProtocolSha256(
        gemma4RuntimeReductionExecutionProtocol(),
      ),
      invocationProgramId: invocationProgram.id,
      invocationProgramSha256: gemma4RuntimeReductionInvocationProgramSha256(invocationProgram),
      executionTranscript: gemma4RuntimeReductionTranscriptEvidence(
        JSON.stringify(requestEnvelope), JSON.stringify(responseEnvelope), executionProtocol,
      ),
      runtimeAttestation,
      orderedOperands: [
        gemma4RuntimeReductionTensorEvidence(request.operands[0]),
        gemma4RuntimeReductionTensorEvidence(request.operands[1]),
      ],
      output: gemma4RuntimeReductionTensorEvidence(output),
    },
  };
}

function fixtureEvidenceContract() {
  const executionProtocol = gemma4RuntimeReductionExecutionProtocol();
  return {
    providerContractId: "torch-2.12.1-cpu-inference-matmul-v1" as const,
    adapterProgramSha256: GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256,
    executionProtocolSha256: gemma4RuntimeReductionExecutionProtocolSha256(executionProtocol),
    executionProtocol,
    invocationPrograms: gemma4RuntimeReductionInvocationPrograms(),
    runtimeAttestation: expectedGemma4RuntimeReductionAttestation(),
  };
}
