import {
  validateGemma4LiteralGenerationProgram,
  type Gemma4LiteralGenerationAssignment,
  type Gemma4LiteralGreedyGenerationProgram,
} from "./gemma4-composite-literal.js";
import type { Gemma4CompositeProgram } from "./gemma4-composite.js";
import { selectGreedyToken } from "./generation.js";
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
  const leading = generation.assignments.filter((assignment) => assignment.iteration === undefined).slice(0, 2);
  const loop = generation.assignments.filter((assignment) => assignment.iteration !== undefined);
  const terminal = generation.assignments.filter((assignment) => assignment.iteration === undefined).slice(2);
  let prefill: ReferenceF32ExecutionResult | undefined;
  let current: ReferenceF32ExecutionResult | undefined;
  let position: number | undefined;
  let incrementalInputs: Gemma4LiteralIncrementalForwardInput | undefined;
  let selectedToken: number | undefined;
  const generatedTokenIds: number[] = [];
  const selectionLogits: DenseF32Tensor[] = [];
  const stepPastKeyValues: Array<ReadonlyMap<number, ReferenceF32KeyValueCache>> = [];
  const steps: Array<{ tokenId: number; positionId: number }> = [];

  for (const assignment of leading) {
    if (assignment.operation === "execute-declared-forward") {
      prefill = await executor.prefill(request);
      current = prefill;
      record(executions, assignment, undefined, prefill);
    } else if (assignment.operation === "initialize-position") {
      position = request.positionIds?.[0]?.at(-1) ?? request.inputIds[0]!.length - 1;
      record(executions, assignment, undefined, position);
    } else {
      throw new Error(`${assignment.id}: atribuição inicial de geração Gemma 4 não é executável nessa posição.`);
    }
  }
  if (!prefill || !current || position === undefined) throw new Error("Programa literal Gemma 4 não inicializou forward e posição.");

  for (let step = 0; step < request.maxNewTokens; step += 1) {
    let stop = false;
    for (const assignment of loop) {
      switch (assignment.operation) {
        case "capture-selection-logits":
          selectionLogits.push(current.logits);
          record(executions, assignment, step, current.logits);
          break;
        case "argmax-lowest-token-id":
          selectedToken = selectGreedyToken(current.logits);
          record(executions, assignment, step, selectedToken);
          break;
        case "append-token":
          requireSelectedToken(selectedToken, assignment.id);
          generatedTokenIds.push(selectedToken);
          record(executions, assignment, step, [...generatedTokenIds]);
          break;
        case "increment-position":
          if (!Number.isSafeInteger(position + 1)) throw new Error(`${assignment.id}: avanço de posição excede inteiro seguro.`);
          position += 1;
          record(executions, assignment, step, position);
          break;
        case "prepare-incremental-forward-inputs":
          requireSelectedToken(selectedToken, assignment.id);
          incrementalInputs = { inputIds: [[selectedToken]], positionIds: [[position]], pastKeyValues: current.pastKeyValues };
          steps.push({ tokenId: selectedToken, positionId: position });
          record(executions, assignment, step, incrementalInputs);
          break;
        case "execute-declared-incremental-forward":
          if (!incrementalInputs) throw new Error(`${assignment.id}: entradas incrementais ainda não foram produzidas.`);
          current = await executor.incremental(incrementalInputs);
          record(executions, assignment, step, current);
          break;
        case "append-cache-snapshot":
          stepPastKeyValues.push(current.pastKeyValues);
          record(executions, assignment, step, [...stepPastKeyValues]);
          break;
        case "evaluate-eos-stop":
          requireSelectedToken(selectedToken, assignment.id);
          stop = request.eosTokenId !== undefined && selectedToken === request.eosTokenId;
          record(executions, assignment, step, stop);
          break;
        default:
          throw new Error(`${assignment.id}: operação ${assignment.operation} não pertence ao loop de geração Gemma 4.`);
      }
    }
    if (stop) break;
    selectedToken = undefined;
    incrementalInputs = undefined;
  }

  for (const assignment of terminal) {
    if (assignment.operation === "select-terminal-logits") record(executions, assignment, undefined, current.logits);
    else if (assignment.operation === "select-terminal-cache") record(executions, assignment, undefined, current.pastKeyValues);
    else throw new Error(`${assignment.id}: atribuição terminal de geração Gemma 4 não é executável nessa posição.`);
  }

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

function requireSelectedToken(value: number | undefined, assignmentId: string): asserts value is number {
  if (value === undefined) throw new Error(`${assignmentId}: argmax ainda não produziu selected_token.`);
}
