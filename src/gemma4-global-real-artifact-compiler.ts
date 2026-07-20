import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { evaluateLiteralDenseElementAddress } from "./literal.js";
import {
  buildGemma4ExactRealSimplifiedProgram,
  type Gemma4ExactRealSimplifiedProgram,
  type Gemma4GlobalRealOutputRequest,
} from "./gemma4-global-real-program.js";
import type { Gemma4LiteralInstantiatedCalculation } from "./gemma4-literal-calculation-graph.js";
import type { TensorRef } from "./types.js";

export interface Gemma4GlobalRealArtifactCompilation {
  kind: "gemma4-exact-real-source-removed-compilation";
  schemaVersion: 1;
  sourceCheckpointAccessed: false;
  learnedScalarCount: number;
  learnedStorageBytesRead: number;
  program: Gemma4ExactRealSimplifiedProgram;
}

interface LearnedRequest {
  key: string;
  assignment: Gemma4LiteralInstantiatedCalculation;
  role: string;
  coordinates: number[];
}

interface LearnedBits {
  dtype: "BF16" | "F16" | "F32";
  hex: string;
  bytes: 2 | 4;
}

/**
 * Performs a metadata-only discovery pass, reads only the reached learned
 * elements through the artifact's integrity-checked paged reader, then emits
 * the final graph with exact dyadic rationals derived from the storage bits.
 */
export async function buildGemma4ExactRealSimplifiedProgramFromArtifact(
  artifact: OpenGemma4CompositeLiteralArtifact,
  requests: readonly Gemma4GlobalRealOutputRequest[],
  declaredInputs: ReadonlySet<string>,
): Promise<Gemma4GlobalRealArtifactCompilation> {
  const requested = new Map<string, LearnedRequest>();
  buildGemma4ExactRealSimplifiedProgram(artifact.calculationGraph, requests, {
    declaredInputs,
    resolveLearnedElement: (assignment, role, coordinates, _bindings, builder) => {
      const request = learnedRequest(assignment, role, coordinates);
      requested.set(request.key, request);
      // Discovery must not erase a term before every sibling access is visited.
      return builder.rational(1n);
    },
  });
  const { values, bytesRead: learnedStorageBytesRead } = await readLearnedBitsInContiguousRanges(artifact, [...requested.values()]);
  const program = buildGemma4ExactRealSimplifiedProgram(artifact.calculationGraph, requests, {
    declaredInputs,
    resolveLearnedElement: (assignment, role, coordinates, _bindings, builder) => {
      const request = learnedRequest(assignment, role, coordinates);
      const value = values.get(request.key);
      if (!value) throw new Error(`${request.key}: scalar aprendido real não foi autenticado no discovery pass.`);
      return builder.rationalFromBits(value.dtype, value.hex);
    },
  });
  return {
    kind: "gemma4-exact-real-source-removed-compilation",
    schemaVersion: 1,
    sourceCheckpointAccessed: false,
    learnedScalarCount: values.size,
    learnedStorageBytesRead,
    program,
  };
}

async function readLearnedBitsInContiguousRanges(
  artifact: OpenGemma4CompositeLiteralArtifact,
  requests: readonly LearnedRequest[],
): Promise<{ values: Map<string, LearnedBits>; bytesRead: number }> {
  interface Addressed {
    request: LearnedRequest;
    tensor: TensorRef;
    byteOffset: number;
    byteLength: 2 | 4;
    dtype: "BF16" | "F16" | "F32";
  }
  const byTensor = new Map<string, Addressed[]>();
  for (const request of requests) {
    const operand = request.assignment.learnedOperands?.find((candidate) => candidate.role === request.role);
    if (!operand) throw new Error(`${request.assignment.operationId}: papel aprendido real ausente ${request.role}.`);
    const decoder = artifact.storageDecoders.find((candidate) => candidate.output === operand.tensor.name);
    const constant = artifact.constants.get(operand.tensor.name);
    if (!decoder || !constant || (constant.storageDtype !== "BF16" && constant.storageDtype !== "F16" && constant.storageDtype !== "F32")) {
      throw new Error(`${operand.tensor.name}: constante densa real ou decoder ausente.`);
    }
    const address = evaluateLiteralDenseElementAddress(decoder, request.coordinates);
    const entry: Addressed = {
      request,
      tensor: operand.tensor,
      byteOffset: address.byteOffset,
      byteLength: address.byteLength,
      dtype: constant.storageDtype,
    };
    const entries = byTensor.get(operand.tensor.name) ?? [];
    entries.push(entry);
    byTensor.set(operand.tensor.name, entries);
  }
  const values = new Map<string, LearnedBits>();
  let bytesRead = 0;
  for (const entries of byTensor.values()) {
    entries.sort((left, right) => left.byteOffset - right.byteOffset);
    for (let startIndex = 0; startIndex < entries.length;) {
      let endIndex = startIndex + 1;
      let rangeEnd = entries[startIndex]!.byteOffset + entries[startIndex]!.byteLength;
      while (endIndex < entries.length && entries[endIndex]!.byteOffset === rangeEnd) {
        rangeEnd += entries[endIndex]!.byteLength;
        endIndex += 1;
      }
      const first = entries[startIndex]!;
      const range = await artifact.readTensorBytesRangeWithIntegrity(
        { name: first.tensor.name, storageDtype: first.tensor.storageDtype, storageShape: first.tensor.shape, logicalShape: first.tensor.shape },
        first.byteOffset,
        rangeEnd - first.byteOffset,
      );
      bytesRead += range.bytes.length;
      for (let index = startIndex; index < endIndex; index += 1) {
        const entry = entries[index]!;
        const offset = entry.byteOffset - first.byteOffset;
        const bits = entry.byteLength === 2 ? range.bytes.readUInt16LE(offset) : range.bytes.readUInt32LE(offset);
        values.set(entry.request.key, {
          dtype: entry.dtype,
          hex: `0x${bits.toString(16).padStart(entry.byteLength * 2, "0")}`,
          bytes: entry.byteLength,
        });
      }
      startIndex = endIndex;
    }
  }
  return { values, bytesRead };
}

function learnedRequest(
  assignment: Gemma4LiteralInstantiatedCalculation,
  role: string,
  coordinates: readonly number[],
): LearnedRequest {
  const operand = assignment.learnedOperands?.find((candidate) => candidate.role === role);
  if (!operand) throw new Error(`${assignment.operationId}: fórmula real referencia papel aprendido não declarado ${role}.`);
  if (coordinates.length !== operand.tensor.shape.length || coordinates.some((coordinate, axis) =>
    !Number.isSafeInteger(coordinate) || coordinate < 0 || coordinate >= operand.tensor.shape[axis]!)) {
    throw new Error(`${assignment.operationId}:${role}: coordenada aprendida real inválida [${coordinates.join(",")}].`);
  }
  const key = `${operand.tensor.name}[${coordinates.join(",")}]`;
  return { key, assignment, role, coordinates: [...coordinates] };
}
