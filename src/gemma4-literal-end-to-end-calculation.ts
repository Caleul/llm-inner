import type { Gemma4CompositeLiteralInput, Gemma4CompositeUnreachableConstant } from "./gemma4-composite-literal.js";
import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import {
  buildGemma4LiteralCalculationSlice,
  type Gemma4LiteralCalculationSlice,
} from "./gemma4-literal-calculation-slice.js";
import {
  buildGemma4LiteralGenerationCalculationPlan,
  type Gemma4LiteralGenerationCalculationPlan,
} from "./gemma4-literal-generation-navigation.js";
import type { Gemma4LiteralCacheTransition } from "./gemma4-literal-generation-calculations.js";
import type { Gemma4LiteralNumericLiteral } from "./gemma4-literal-numeric-literals.js";
import type { LiteralDenseStorageDecodeAssignment } from "./literal.js";
import type { TensorRef } from "./types.js";
import type { Gemma4LiteralForwardControlProgram } from "./gemma4-literal-forward-control.js";
import type { Gemma4LiteralInputContract } from "./gemma4-literal-input-contract.js";
import type { Gemma4LiteralOutputContract } from "./gemma4-literal-output-contract.js";

export interface Gemma4LiteralRuntimeUnreachableStorage {
  tensor: TensorRef;
  payloadBytes: number;
  decoderId: string;
  decoderOperation: LiteralDenseStorageDecodeAssignment["operation"];
  decoder: LiteralDenseStorageDecodeAssignment;
  reason: Gemma4CompositeUnreachableConstant["reason"];
  producerLayer: number;
}

export interface Gemma4LiteralEndToEndCalculation {
  kind: "gemma4-literal-end-to-end-calculation";
  sourceCheckpointAccessed: false;
  maxNewTokens: number;
  declaredInputs: Gemma4CompositeLiteralInput[];
  inputContract: Gemma4LiteralInputContract;
  outputContract: Gemma4LiteralOutputContract;
  forwardControl: Gemma4LiteralForwardControlProgram;
  forward: Gemma4LiteralCalculationSlice;
  generation: Gemma4LiteralGenerationCalculationPlan & {
    cacheTransitions: Gemma4LiteralCacheTransition[];
  };
  outputs: {
    forwardLogits: string;
    generatedTokenIds: string;
    terminalLogits: string;
    terminalPastKeyValues: string;
  };
  numericLiterals: Gemma4LiteralNumericLiteral[];
  storageCoverage: {
    embeddedConstantCount: number;
    reachableLearnedConstantCount: number;
    runtimeUnreachableConstantCount: number;
    complete: true;
    runtimeUnreachableConstants: Gemma4LiteralRuntimeUnreachableStorage[];
  };
  reproducibility: {
    generationControl: "literal";
    status: Gemma4LiteralCalculationSlice["reproducibility"]["status"];
    failClosedForwardOperationIds: string[];
  };
}

/**
 * Joins the artifact-owned forward graph, learned storage, greedy state
 * machine and cache transitions into one finite navigation boundary. The
 * target is selected by the serialized logits output, never by an operation
 * naming convention.
 */
export function buildGemma4LiteralEndToEndCalculation(
  artifact: OpenGemma4CompositeLiteralArtifact,
  maxNewTokens: number,
): Gemma4LiteralEndToEndCalculation {
  const logitsProducers = artifact.calculationGraph.assignments
    .filter((assignment) => assignment.output === artifact.outputs.logits);
  if (logitsProducers.length !== 1) {
    throw new Error(`Programa Gemma 4 literal requer um único produtor de logits, encontrados ${logitsProducers.length}.`);
  }
  const forward = buildGemma4LiteralCalculationSlice(artifact, logitsProducers[0]!.operationId);
  const generation = buildGemma4LiteralGenerationCalculationPlan(artifact, maxNewTokens);
  const runtimeUnreachableConstants = runtimeUnreachableStorage(artifact, new Set(forward.learnedConstants.map((entry) => entry.tensor.name)));
  const covered = forward.learnedConstants.length + runtimeUnreachableConstants.length;
  if (covered !== artifact.constants.size) {
    throw new Error(`Programa Gemma 4 literal cobre ${covered} de ${artifact.constants.size} constantes entre cálculo e storage runtime-unreachable.`);
  }
  return {
    kind: "gemma4-literal-end-to-end-calculation",
    sourceCheckpointAccessed: false,
    maxNewTokens,
    declaredInputs: structuredClone(artifact.inputs),
    inputContract: structuredClone(artifact.inputContract),
    outputContract: structuredClone(artifact.outputContract),
    forwardControl: structuredClone(artifact.forwardControl),
    forward,
    generation: {
      ...generation,
      cacheTransitions: structuredClone(artifact.generation.forwardCalculation.cacheTransitions),
    },
    outputs: {
      forwardLogits: artifact.outputs.logits,
      generatedTokenIds: artifact.generation.outputs.generatedTokenIds,
      terminalLogits: artifact.generation.outputs.terminalLogits,
      terminalPastKeyValues: artifact.generation.outputs.terminalPastKeyValues,
    },
    numericLiterals: structuredClone(artifact.numericLiterals.literals),
    storageCoverage: {
      embeddedConstantCount: artifact.constants.size,
      reachableLearnedConstantCount: forward.learnedConstants.length,
      runtimeUnreachableConstantCount: runtimeUnreachableConstants.length,
      complete: true,
      runtimeUnreachableConstants,
    },
    reproducibility: {
      generationControl: "literal",
      status: forward.reproducibility.status,
      failClosedForwardOperationIds: [...forward.reproducibility.failClosedOperationIds],
    },
  };
}

function runtimeUnreachableStorage(
  artifact: OpenGemma4CompositeLiteralArtifact,
  reachable: ReadonlySet<string>,
): Gemma4LiteralRuntimeUnreachableStorage[] {
  const declarations = new Map(artifact.unreachableConstants.map((entry) => [entry.name, entry]));
  const decoders = new Map(artifact.storageDecoders.map((entry) => [entry.output, entry]));
  const result: Gemma4LiteralRuntimeUnreachableStorage[] = [];
  for (const [name, constant] of artifact.constants) {
    const declaration = declarations.get(name);
    if (reachable.has(name)) {
      if (declaration) throw new Error(`${name}: constante alcançável também foi declarada runtime-unreachable.`);
      continue;
    }
    const decoder = decoders.get(name);
    if (!declaration || !decoder) {
      throw new Error(`${name}: constante fora do cálculo de logits não possui declaração runtime-unreachable e decoder.`);
    }
    result.push({
      tensor: {
        name,
        storageDtype: constant.storageDtype,
        shape: [...constant.logicalShape],
      },
      payloadBytes: constant.payloadBytes,
      decoderId: decoder.id,
      decoderOperation: decoder.operation,
      decoder: structuredClone(decoder),
      reason: declaration.reason,
      producerLayer: declaration.producerLayer,
    });
    declarations.delete(name);
  }
  if (declarations.size !== 0) {
    throw new Error(`Programa Gemma 4 literal declara storage runtime-unreachable ausente: ${[...declarations.keys()].join(", ")}.`);
  }
  return result.sort((left, right) => left.tensor.name.localeCompare(right.tensor.name, "en"));
}
