import { isDeepStrictEqual } from "node:util";
import type {
  Gemma4CompositeExecutionRequest,
  Gemma4CompositeExecutionResult,
  Gemma4CompositeGenerationRequest,
  Gemma4CompositeGenerationResult,
  Gemma4CompositeProgram,
} from "./gemma4-composite.js";
import type { DenseF32Tensor } from "./types.js";

export interface Gemma4LiteralGenerationControlProgram {
  kind: "gemma4-literal-greedy-control-program";
  schemaVersion: 1;
  forward: {
    operationOrder: "generation.forwardCalculation.operationOrder";
    cacheTransitions: "generation.forwardCalculation.cacheTransitions";
    logitsOutput: "softcapped_logits" | "logits";
  };
  constraints: {
    batchSize: 1;
    minimumPromptTokens: 1;
    maxNewTokens: "non-negative-safe-integer";
    eosTokenId: "absent-or-non-negative-safe-integer";
    attentionMask: "must-be-absent";
    initialPastKeyValues: "must-be-absent";
  };
  prefill: {
    mode: "execute-forward";
    inputBindings: [
      "input_ids",
      "position_ids?",
      "pixel_values?",
      "image_position_ids?",
      "pixel_values_videos?",
      "video_position_ids?",
      "input_features?",
      "input_features_mask?",
      "mm_token_type_ids?",
    ];
    outputState: "forward_state[0]";
  };
  initialPosition: {
    output: "position[-1]";
    whenPositionIdsPresent: { batch: 0; sequence: "last" };
    whenPositionIdsAbsent: "input_ids.shape[1]-1";
  };
  loop: {
    iterator: "step";
    startInclusive: 0;
    endExclusiveInput: "max_new_tokens";
    selection: {
      logits: "forward_state[step].logits";
      batch: 0;
      sequence: "last";
      tokenStartInclusive: 0;
      tokenEndExclusive: number;
      scanOrder: "ascending-token-id";
      comparator: "strictly-greater";
      tie: "retain-earlier-lowest-token-id";
      nonFinite: "reject";
      output: "selected_token[step]";
    };
    tokenAppend: {
      output: "generated_token_ids";
      source: "selected_token[step]";
      order: "append-after-existing";
    };
    positionAdvance: {
      output: "position[step]";
      source: "position[step-1]";
      increment: 1;
      arithmetic: "exact-safe-integer";
    };
    incrementalForward: {
      mode: "execute-forward";
      inputIds: "[[selected_token[step]]]";
      positionIds: "[[position[step]]]";
      pastKeyValues: "forward_state[step].past_key_values";
      omittedInputs: [
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "image_position_ids",
        "pixel_values_videos",
        "video_position_ids",
        "input_features",
        "input_features_mask",
      ];
      outputState: "forward_state[step+1]";
    };
    cacheSnapshot: {
      output: "step_past_key_values";
      source: "forward_state[step+1].past_key_values";
      order: "append-after-existing";
    };
    stop: {
      evaluateAfter: "forward_state[step+1]";
      predicate: "eos_token_id-is-present-and-selected_token-equals-eos_token_id";
      output: "stop_after_step[step]";
      effect: "break-after-current-step";
    };
  };
  terminal: {
    executedSteps: "zero-if-no-step-else-number-of-completed-steps";
    logits: "forward_state[executed_steps].logits";
    pastKeyValues: "forward_state[executed_steps].past_key_values";
  };
}

export function buildGemma4LiteralGenerationControlProgram(
  program: Gemma4CompositeProgram,
): Gemma4LiteralGenerationControlProgram {
  return {
    kind: "gemma4-literal-greedy-control-program",
    schemaVersion: 1,
    forward: {
      operationOrder: "generation.forwardCalculation.operationOrder",
      cacheTransitions: "generation.forwardCalculation.cacheTransitions",
      logitsOutput: program.outputs.logits,
    },
    constraints: {
      batchSize: 1,
      minimumPromptTokens: 1,
      maxNewTokens: "non-negative-safe-integer",
      eosTokenId: "absent-or-non-negative-safe-integer",
      attentionMask: "must-be-absent",
      initialPastKeyValues: "must-be-absent",
    },
    prefill: {
      mode: "execute-forward",
      inputBindings: [
        "input_ids", "position_ids?", "pixel_values?", "image_position_ids?", "pixel_values_videos?",
        "video_position_ids?", "input_features?", "input_features_mask?", "mm_token_type_ids?",
      ],
      outputState: "forward_state[0]",
    },
    initialPosition: {
      output: "position[-1]",
      whenPositionIdsPresent: { batch: 0, sequence: "last" },
      whenPositionIdsAbsent: "input_ids.shape[1]-1",
    },
    loop: {
      iterator: "step",
      startInclusive: 0,
      endExclusiveInput: "max_new_tokens",
      selection: {
        logits: "forward_state[step].logits",
        batch: 0,
        sequence: "last",
        tokenStartInclusive: 0,
        tokenEndExclusive: program.contract.text.vocabSize,
        scanOrder: "ascending-token-id",
        comparator: "strictly-greater",
        tie: "retain-earlier-lowest-token-id",
        nonFinite: "reject",
        output: "selected_token[step]",
      },
      tokenAppend: { output: "generated_token_ids", source: "selected_token[step]", order: "append-after-existing" },
      positionAdvance: { output: "position[step]", source: "position[step-1]", increment: 1, arithmetic: "exact-safe-integer" },
      incrementalForward: {
        mode: "execute-forward",
        inputIds: "[[selected_token[step]]]",
        positionIds: "[[position[step]]]",
        pastKeyValues: "forward_state[step].past_key_values",
        omittedInputs: [
          "attention_mask", "mm_token_type_ids", "pixel_values", "image_position_ids", "pixel_values_videos",
          "video_position_ids", "input_features", "input_features_mask",
        ],
        outputState: "forward_state[step+1]",
      },
      cacheSnapshot: {
        output: "step_past_key_values",
        source: "forward_state[step+1].past_key_values",
        order: "append-after-existing",
      },
      stop: {
        evaluateAfter: "forward_state[step+1]",
        predicate: "eos_token_id-is-present-and-selected_token-equals-eos_token_id",
        output: "stop_after_step[step]",
        effect: "break-after-current-step",
      },
    },
    terminal: {
      executedSteps: "zero-if-no-step-else-number-of-completed-steps",
      logits: "forward_state[executed_steps].logits",
      pastKeyValues: "forward_state[executed_steps].past_key_values",
    },
  };
}

export function validateGemma4LiteralGenerationControlProgram(
  control: Gemma4LiteralGenerationControlProgram,
  program: Gemma4CompositeProgram,
): void {
  if (!isDeepStrictEqual(control, buildGemma4LiteralGenerationControlProgram(program))) {
    throw new Error("Programa literal Gemma 4 possui controle greedy ausente, opaco ou divergente.");
  }
}

/**
 * Executes the serialized control program. The callback is the artifact's
 * declared forward interpreter; this function owns no model operation and
 * never substitutes a framework generation helper.
 */
export function executeGemma4LiteralGenerationControlProgram(
  control: Gemma4LiteralGenerationControlProgram,
  program: Gemma4CompositeProgram,
  request: Omit<Gemma4CompositeGenerationRequest, "tensors">,
  executeForward: (request: Omit<Gemma4CompositeExecutionRequest, "tensors">) => Gemma4CompositeExecutionResult,
): Gemma4CompositeGenerationResult {
  validateGemma4LiteralGenerationControlProgram(control, program);
  validateRequest(control, request);
  const { maxNewTokens, eosTokenId, ...prefillInputs } = request;
  const prefill = executeForward(prefillInputs);
  let current = prefill;
  let position = request.positionIds?.[control.initialPosition.whenPositionIdsPresent.batch]?.at(-1)
    ?? request.inputIds[0]!.length - 1;
  requireSafePosition(position, control.initialPosition.output);
  const generatedTokenIds: number[] = [];
  const selectionLogits: DenseF32Tensor[] = [];
  const stepPastKeyValues: Gemma4CompositeGenerationResult["stepPastKeyValues"] = [];
  for (let step: number = control.loop.startInclusive; step < maxNewTokens; step += 1) {
    const logits = current.text.logits;
    selectionLogits.push(logits);
    const tokenId = selectGemma4LiteralGenerationToken(control, logits);
    generatedTokenIds.push(tokenId);
    position = requireSafePosition(position + control.loop.positionAdvance.increment, control.loop.positionAdvance.output);
    current = executeForward({
      inputIds: [[tokenId]],
      positionIds: [[position]],
      pastKeyValues: current.text.pastKeyValues,
    });
    stepPastKeyValues.push(current.text.pastKeyValues);
    if (eosTokenId !== undefined && tokenId === eosTokenId) break;
  }
  return { prefill, generatedTokenIds, selectionLogits, stepPastKeyValues, text: current.text };
}

function validateRequest(
  control: Gemma4LiteralGenerationControlProgram,
  request: Omit<Gemma4CompositeGenerationRequest, "tensors">,
): void {
  if (request.inputIds.length !== control.constraints.batchSize ||
    (request.inputIds[0]?.length ?? 0) < control.constraints.minimumPromptTokens) {
    throw new Error("Geração literal Gemma 4 requer um único prompt não vazio sem padding.");
  }
  if (!Number.isSafeInteger(request.maxNewTokens) || request.maxNewTokens < 0) {
    throw new Error("Geração literal Gemma 4 requer maxNewTokens inteiro seguro não negativo.");
  }
  if (request.eosTokenId !== undefined && (!Number.isSafeInteger(request.eosTokenId) || request.eosTokenId < 0)) {
    throw new Error("Geração literal Gemma 4 requer eosTokenId inteiro seguro não negativo.");
  }
  if (request.attentionMask !== undefined) {
    throw new Error("Programa greedy literal Gemma 4 declara attention_mask ausente.");
  }
  if (request.pastKeyValues !== undefined) {
    throw new Error("Programa greedy literal Gemma 4 começa em prefill sem pastKeyValues.");
  }
}

export function selectGemma4LiteralGenerationToken(
  control: Gemma4LiteralGenerationControlProgram,
  logits: DenseF32Tensor,
): number {
  const selection = control.loop.selection;
  if (logits.shape.length !== 3 || logits.shape[0] !== 1 || logits.shape[2] !== selection.tokenEndExclusive ||
    logits.shape[1] === undefined || logits.shape[1] <= 0 || logits.values.length !== logits.shape[1] * selection.tokenEndExclusive) {
    throw new Error("Programa greedy literal Gemma 4 recebeu logits [1,sequence,vocab] incompatíveis.");
  }
  const offset = (logits.shape[1] - 1) * selection.tokenEndExclusive;
  let bestToken: number = selection.tokenStartInclusive;
  let bestLogit = logits.values[offset + bestToken]!;
  if (!Number.isFinite(bestLogit)) throw new Error("Programa greedy literal Gemma 4 rejeita logit não finito.");
  for (let token = bestToken + 1; token < selection.tokenEndExclusive; token += 1) {
    const candidate = logits.values[offset + token]!;
    if (!Number.isFinite(candidate)) throw new Error("Programa greedy literal Gemma 4 rejeita logit não finito.");
    if (candidate > bestLogit) {
      bestLogit = candidate;
      bestToken = token;
    }
  }
  return bestToken;
}

function requireSafePosition(value: number, owner: string): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${owner}: posição incremental deve ser inteiro seguro não negativo.`);
  return value;
}
