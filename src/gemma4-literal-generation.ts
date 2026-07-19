import {
  validateGemma4LiteralGenerationProgram,
  type Gemma4LiteralGenerationAssignment,
  type Gemma4LiteralGreedyGenerationProgram,
} from "./gemma4-composite-literal.js";
import type { Gemma4CompositeProgram } from "./gemma4-composite.js";
import { selectGemma4LiteralGenerationToken } from "./gemma4-literal-generation-control.js";
import type {
  DenseF32Tensor,
  ReferenceF32ExecutionResult,
  ReferenceF32GenerationResult,
  ReferenceF32KeyValueCache,
} from "./types.js";

export interface Gemma4LiteralGenerationInput {
  inputIds: number[][];
  positionIds?: number[][];
  maxNewTokens: number;
  eosTokenId?: number;
}

export interface Gemma4LiteralIncrementalForwardInput {
  inputIds: number[][];
  positionIds: number[][];
  pastKeyValues: ReadonlyMap<number, ReferenceF32KeyValueCache>;
}

export interface Gemma4LiteralForwardExecutor<TPrefill extends Gemma4LiteralGenerationInput> {
  prefill(request: TPrefill): Promise<ReferenceF32ExecutionResult>;
  incremental(request: Gemma4LiteralIncrementalForwardInput): Promise<ReferenceF32ExecutionResult>;
}

export type Gemma4LiteralGenerationValue =
  | number
  | boolean
  | number[]
  | DenseF32Tensor
  | ReferenceF32ExecutionResult
  | Gemma4LiteralIncrementalForwardInput
  | ReadonlyMap<number, ReferenceF32KeyValueCache>
  | ReadonlyArray<ReadonlyMap<number, ReferenceF32KeyValueCache>>;

/** One concrete execution of one serialized generation assignment. */
export interface Gemma4LiteralGenerationAssignmentExecution {
  assignmentId: string;
  operation: Gemma4LiteralGenerationAssignment["operation"];
  /** Present for assignments instantiated inside the declared step loop. */
  step?: number;
  /** Symbolic `step` references have been replaced by this concrete index. */
  output: string;
  value: Gemma4LiteralGenerationValue;
}

export interface Gemma4LiteralGenerationExecutionResult extends ReferenceF32GenerationResult {
  prefill: ReferenceF32ExecutionResult;
  /** Dependency-ordered values produced by the artifact's generation program. */
  assignmentExecutions: Gemma4LiteralGenerationAssignmentExecution[];
}

/**
 * Executes the generation state machine serialized in the literal artifact.
 *
 * Forward computation is injected as a narrow port because the same state
 * machine applies to the paged text executor and the future complete
 * composite executor. The interpreter owns argmax, position/cache transition,
 * EOS timing and terminal-state selection; callers cannot supply a second
 * hidden greedy loop.
 */
export async function executeGemma4LiteralGenerationProgram<TPrefill extends Gemma4LiteralGenerationInput>(
  composite: Gemma4CompositeProgram,
  generation: Gemma4LiteralGreedyGenerationProgram,
  request: TPrefill,
  executor: Gemma4LiteralForwardExecutor<TPrefill>,
): Promise<Gemma4LiteralGenerationExecutionResult> {
  validateGemma4LiteralGenerationProgram(generation, composite);
  validateRequest(request);

  const executions: Gemma4LiteralGenerationAssignmentExecution[] = [];
  const control = generation.controlProgram;
  const prefillAssignment = requiredAssignment(generation, "execute-declared-forward");
  const positionAssignment = requiredAssignment(generation, "initialize-position");
  const selectionAssignment = requiredAssignment(generation, "capture-selection-logits");
  const argmaxAssignment = requiredAssignment(generation, "argmax-lowest-token-id");
  const tokenAppendAssignment = requiredAssignment(generation, "append-token");
  const positionAdvanceAssignment = requiredAssignment(generation, "increment-position");
  const incrementalInputsAssignment = requiredAssignment(generation, "prepare-incremental-forward-inputs");
  const incrementalForwardAssignment = requiredAssignment(generation, "execute-declared-incremental-forward");
  const cacheSnapshotAssignment = requiredAssignment(generation, "append-cache-snapshot");
  const stopAssignment = requiredAssignment(generation, "evaluate-eos-stop");
  const terminalLogitsAssignment = requiredAssignment(generation, "select-terminal-logits");
  const terminalCacheAssignment = requiredAssignment(generation, "select-terminal-cache");
  const prefill = await executor.prefill(request);
  let current = prefill;
  let position = request.positionIds?.[control.initialPosition.whenPositionIdsPresent.batch]?.at(-1)
    ?? request.inputIds[0]!.length - 1;
  if (!Number.isSafeInteger(position) || position < 0) throw new Error(`${positionAssignment.id}: posição inicial inválida.`);
  record(executions, prefillAssignment, undefined, prefill);
  record(executions, positionAssignment, undefined, position);
  const generatedTokenIds: number[] = [];
  const selectionLogits: DenseF32Tensor[] = [];
  const stepPastKeyValues: Array<ReadonlyMap<number, ReferenceF32KeyValueCache>> = [];
  const steps: Array<{ tokenId: number; positionId: number }> = [];

  for (let step: number = control.loop.startInclusive; step < request.maxNewTokens; step += 1) {
    selectionLogits.push(current.logits);
    record(executions, selectionAssignment, step, current.logits);
    const selectedToken = selectGemma4LiteralGenerationToken(control, current.logits);
    record(executions, argmaxAssignment, step, selectedToken);
    generatedTokenIds.push(selectedToken);
    record(executions, tokenAppendAssignment, step, [...generatedTokenIds]);
    if (!Number.isSafeInteger(position + control.loop.positionAdvance.increment)) {
      throw new Error(`${positionAdvanceAssignment.id}: avanço de posição excede inteiro seguro.`);
    }
    position += control.loop.positionAdvance.increment;
    record(executions, positionAdvanceAssignment, step, position);
    const incrementalInputs: Gemma4LiteralIncrementalForwardInput = {
      inputIds: [[selectedToken]], positionIds: [[position]], pastKeyValues: current.pastKeyValues,
    };
    steps.push({ tokenId: selectedToken, positionId: position });
    record(executions, incrementalInputsAssignment, step, incrementalInputs);
    current = await executor.incremental(incrementalInputs);
    record(executions, incrementalForwardAssignment, step, current);
    stepPastKeyValues.push(current.pastKeyValues);
    record(executions, cacheSnapshotAssignment, step, [...stepPastKeyValues]);
    const stop = request.eosTokenId !== undefined && selectedToken === request.eosTokenId;
    record(executions, stopAssignment, step, stop);
    if (stop) break;
  }

  record(executions, terminalLogitsAssignment, undefined, current.logits);
  record(executions, terminalCacheAssignment, undefined, current.pastKeyValues);

  return {
    prefill,
    inputIds: [...request.inputIds[0]!, ...generatedTokenIds],
    generatedTokenIds,
    steps,
    selectionLogits,
    stepPastKeyValues,
    logits: current.logits,
    pastKeyValues: current.pastKeyValues,
    assignmentExecutions: executions,
  };
}

function requiredAssignment(
  generation: Gemma4LiteralGreedyGenerationProgram,
  operation: Gemma4LiteralGenerationAssignment["operation"],
): Gemma4LiteralGenerationAssignment {
  const matches = generation.assignments.filter((assignment) => assignment.operation === operation);
  if (matches.length !== 1) throw new Error(`Programa literal Gemma 4 requer uma atribuição ${operation}; encontrou ${matches.length}.`);
  return matches[0]!;
}

function validateRequest(request: Gemma4LiteralGenerationInput): void {
  const prefill = request as Gemma4LiteralGenerationInput & {
    attentionMask?: DenseF32Tensor;
    attentionMasksByLayer?: ReadonlyMap<number, DenseF32Tensor>;
    pastKeyValues?: ReadonlyMap<number, ReferenceF32KeyValueCache>;
  };
  if (prefill.attentionMask !== undefined || prefill.attentionMasksByLayer !== undefined) {
    throw new Error("Geração literal Gemma 4 não aceita máscara externa; o programa declara as máscaras de prefill e decode.");
  }
  if (prefill.pastKeyValues !== undefined) {
    throw new Error("Geração literal Gemma 4 começa em prefill sem pastKeyValues; estado inicial externo não está declarado.");
  }
  if (request.inputIds.length !== 1 || request.inputIds[0]?.length === 0 || request.inputIds[0]?.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new Error("Geração literal Gemma 4 requer um prompt único não vazio com IDs inteiros não negativos.");
  }
  if (!Number.isSafeInteger(request.maxNewTokens) || request.maxNewTokens < 0) {
    throw new Error("Geração literal Gemma 4 requer maxNewTokens inteiro não negativo.");
  }
  if (request.eosTokenId !== undefined && (!Number.isSafeInteger(request.eosTokenId) || request.eosTokenId < 0)) {
    throw new Error("Geração literal Gemma 4 requer eosTokenId inteiro não negativo.");
  }
  if (request.positionIds !== undefined && (request.positionIds.length !== 1 || request.positionIds[0]?.length !== request.inputIds[0]!.length ||
    request.positionIds[0]?.some((value) => !Number.isSafeInteger(value) || value < 0))) {
    throw new Error("Geração literal Gemma 4 requer positionIds inteiros e alinhados ao prompt.");
  }
}

function record(
  executions: Gemma4LiteralGenerationAssignmentExecution[],
  assignment: Gemma4LiteralGenerationAssignment,
  step: number | undefined,
  value: Gemma4LiteralGenerationValue,
): void {
  executions.push({
    assignmentId: assignment.id,
    operation: assignment.operation,
    ...(step === undefined ? {} : { step }),
    output: instantiateOutput(assignment.output, step),
    value,
  });
}

function instantiateOutput(output: string, step: number | undefined): string {
  if (step === undefined) return output;
  return output
    .replaceAll("[step+1]", `[${step + 1}]`)
    .replaceAll("[step]", `[${step}]`)
    .replaceAll("[0..step]", `[0..${step}]`);
}
