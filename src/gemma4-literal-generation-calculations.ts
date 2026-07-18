import type { Gemma4CompositeProgram } from "./gemma4-composite.js";
import {
  buildGemma4LiteralCalculationGraph,
  type Gemma4LiteralCalculationGraph,
} from "./gemma4-literal-calculation-graph.js";

export type Gemma4LiteralGenerationOperation =
  | "execute-declared-forward"
  | "initialize-position"
  | "capture-selection-logits"
  | "argmax-lowest-token-id"
  | "append-token"
  | "increment-position"
  | "prepare-incremental-forward-inputs"
  | "execute-declared-incremental-forward"
  | "append-cache-snapshot"
  | "evaluate-eos-stop"
  | "select-terminal-logits"
  | "select-terminal-cache";

interface Gemma4LiteralGenerationAssignmentContract {
  id: string;
  operation: Gemma4LiteralGenerationOperation;
  inputs: string[];
  output: string;
  dtype: string;
  shape: string;
  iteration?: "step = 0..max_new_tokens-1 while stop_after_step[step-1] is false";
}

export type Gemma4LiteralForwardCalculationScope = "composite" | "vision" | "audio" | "text-layer" | "text-epilogue";

export interface Gemma4LiteralForwardOperationReference {
  operationId: string;
  definitionId: string;
  scope: Gemma4LiteralForwardCalculationScope;
  invocationId?: string;
}

export interface Gemma4LiteralCacheTransition {
  layer: number;
  attentionOperationId: string;
  ownership: "producer" | "reuse-producer";
  producerLayer: number;
  layout: "BHSD";
  query: string;
  key: string;
  value: string;
  prefill: string;
  incremental: string;
}

/**
 * The exact forward expansion referenced by both generation forward states.
 * Operation IDs are instantiated (including both vision invocations and the
 * audio invocation), so a reader never has to replace this list with an
 * opaque decoder call. Cache ownership and append/reuse behavior are data too.
 */
export interface Gemma4LiteralGenerationForwardCalculationContract {
  kind: "gemma4-literal-generation-forward-calculation-contract";
  schemaVersion: 1;
  operationOrder: Gemma4LiteralForwardOperationReference[];
  logitsOutput: "softcapped_logits" | "logits";
  cacheTransitions: Gemma4LiteralCacheTransition[];
}

export interface Gemma4LiteralGenerationScalarCalculation {
  definitionId: string;
  operation: Gemma4LiteralGenerationOperation;
  orderedInputs: string[];
  output: string;
  dtype: string;
  shape: string;
  iteration?: Gemma4LiteralGenerationAssignmentContract["iteration"];
  scalarAssignments: string[];
  formula: string;
  forwardExpansionReference?: "generation.forwardCalculation.operationOrder";
  cacheTransitionReference?: "generation.forwardCalculation.cacheTransitions";
}

export interface Gemma4LiteralGenerationScalarCalculations {
  kind: "gemma4-literal-generation-scalar-calculations";
  schemaVersion: 1;
  formulaLanguage: "indexed-ieee754-expression-v1";
  assignments: Gemma4LiteralGenerationScalarCalculation[];
}

export function buildGemma4LiteralGenerationForwardCalculationContract(
  program: Gemma4CompositeProgram,
  calculationGraph: Gemma4LiteralCalculationGraph = buildGemma4LiteralCalculationGraph(program),
): Gemma4LiteralGenerationForwardCalculationContract {
  return {
    kind: "gemma4-literal-generation-forward-calculation-contract",
    schemaVersion: 1,
    operationOrder: calculationGraph.assignments.map((assignment) => ({
      operationId: assignment.operationId,
      definitionId: assignment.definitionId,
      scope: assignment.scope,
      ...(assignment.invocationId ? { invocationId: assignment.invocationId } : {}),
    })),
    logitsOutput: program.outputs.logits,
    cacheTransitions: cacheTransitions(program),
  };
}

export function buildGemma4LiteralGenerationScalarCalculations(
  assignments: readonly Gemma4LiteralGenerationAssignmentContract[],
  forward: Gemma4LiteralGenerationForwardCalculationContract,
  vocabSize: number,
): Gemma4LiteralGenerationScalarCalculations {
  return {
    kind: "gemma4-literal-generation-scalar-calculations",
    schemaVersion: 1,
    formulaLanguage: "indexed-ieee754-expression-v1",
    assignments: assignments.map((assignment) => scalarCalculation(assignment, forward, vocabSize)),
  };
}

function cacheTransitions(program: Gemma4CompositeProgram): Gemma4LiteralCacheTransition[] {
  const transitions: Gemma4LiteralCacheTransition[] = [];
  for (const layer of program.textProgram.layers) {
    const attention = layer.operations.find((operation) => operation.op === "scaled_dot_product_attention");
    if (!attention || attention.layer === undefined) throw new Error(`Camada textual ${layer.index} não possui atenção cacheável declarada.`);
    const producerLayer = attention.kvSharing?.producerLayer ?? attention.layer;
    const ownership = attention.kvSharing ? "reuse-producer" as const : "producer" as const;
    transitions.push({
      layer: attention.layer,
      attentionOperationId: attention.id,
      ownership,
      producerLayer,
      layout: "BHSD",
      query: attention.query,
      key: attention.key,
      value: attention.value,
      prefill: ownership === "producer"
        ? `past_key_values[${attention.layer}] = {key:${attention.key},value:${attention.value}} in BHSD after RoPE`
        : `attention layer ${attention.layer} reads past_key_values[${producerLayer}] and emits no duplicate cache entry`,
      incremental: ownership === "producer"
        ? `past_key_values[${attention.layer}] = {key:concat(previous_past_key_values[${attention.layer}].key,${attention.key},sequence_axis=2),value:concat(previous_past_key_values[${attention.layer}].value,${attention.value},sequence_axis=2)} in BHSD`
        : `attention layer ${attention.layer} reads the already-appended past_key_values[${producerLayer}] and emits no duplicate cache entry`,
    });
  }
  return transitions;
}

function scalarCalculation(
  assignment: Gemma4LiteralGenerationAssignmentContract,
  forward: Gemma4LiteralGenerationForwardCalculationContract,
  vocabSize: number,
): Gemma4LiteralGenerationScalarCalculation {
  const scalarAssignments = generationFormulas(assignment.operation, forward, vocabSize);
  return {
    definitionId: assignment.id,
    operation: assignment.operation,
    orderedInputs: [...assignment.inputs],
    output: assignment.output,
    dtype: assignment.dtype,
    shape: assignment.shape,
    ...(assignment.iteration ? { iteration: assignment.iteration } : {}),
    scalarAssignments,
    formula: scalarAssignments.at(-1)!,
    ...(assignment.operation === "execute-declared-forward" || assignment.operation === "execute-declared-incremental-forward"
      ? {
        forwardExpansionReference: "generation.forwardCalculation.operationOrder" as const,
        cacheTransitionReference: "generation.forwardCalculation.cacheTransitions" as const,
      }
      : {}),
  };
}

function generationFormulas(
  operation: Gemma4LiteralGenerationOperation,
  forward: Gemma4LiteralGenerationForwardCalculationContract,
  vocabSize: number,
): string[] {
  const orderEnd = forward.operationOrder.length - 1;
  switch (operation) {
    case "execute-declared-forward": return [
      `evaluate generation.forwardCalculation.operationOrder[0..${orderEnd}] in array order using declared prefill inputs and embedded constants`,
      `forward_state[0].logits = ${forward.logitsOutput}`,
      "forward_state[0].past_key_values = apply generation.forwardCalculation.cacheTransitions[*].prefill in layer order",
      `forward_state[0] = STRUCT(logits=${forward.logitsOutput},past_key_values=declared_prefill_cache_outputs)`,
    ];
    case "initialize-position": return ["position[-1] = position_ids supplied ? position_ids[0,input_ids.shape[1]-1] : input_ids.shape[1]-1"];
    case "capture-selection-logits": return ["selection_logits[step] = forward_state[step].logits; exact F32 alias; no cast"];
    case "argmax-lowest-token-id": return [
      `candidate[v] = selection_logits[step][0,current_sequence-1,v], v=0..${vocabSize - 1}; reject if any candidate is non-finite`,
      "best_token[0]=0; best_logit[0]=candidate[0]",
      `best_token[v]=candidate[v] > best_logit[v-1] ? v : best_token[v-1]; best_logit[v]=max(candidate[v],best_logit[v-1]), v=1..${vocabSize - 1} ascending`,
      `selected_token[step] = best_token[${vocabSize - 1}] (equality retains the earlier, therefore lowest, token ID)`,
    ];
    case "append-token": return ["generated_token_ids[0..step] = step==0 ? [selected_token[0]] : concat(generated_token_ids[0..step-1],[selected_token[step]])"];
    case "increment-position": return ["position[step] = exact_safe_integer(position[step-1] + 1)"];
    case "prepare-incremental-forward-inputs": return [
      "incremental_inputs[step].input_ids = [[selected_token[step]]]",
      "incremental_inputs[step].position_ids = [[position[step]]]",
      "incremental_inputs[step].past_key_values = forward_state[step].past_key_values",
      "incremental_inputs[step] omits attention_mask, mm_token_type_ids and every image/video/audio input",
    ];
    case "execute-declared-incremental-forward": return [
      `evaluate generation.forwardCalculation.operationOrder[0..${orderEnd}] in array order with incremental_inputs[step], skipping absent modality branches by their declared optional-input guards`,
      `forward_state[step+1].logits = ${forward.logitsOutput}`,
      "forward_state[step+1].past_key_values = apply generation.forwardCalculation.cacheTransitions[*].incremental in layer order",
      `forward_state[step+1] = STRUCT(logits=${forward.logitsOutput},past_key_values=declared_incremental_cache_outputs)`,
    ];
    case "append-cache-snapshot": return ["step_past_key_values[0..step] = step==0 ? [forward_state[1].past_key_values] : concat(step_past_key_values[0..step-1],[forward_state[step+1].past_key_values])"];
    case "evaluate-eos-stop": return ["stop_after_step[step] = eos_token_id supplied and selected_token[step] == eos_token_id; evaluate after forward_state[step+1] and its cache exist"];
    case "select-terminal-logits": return [
      "executed_steps = max_new_tokens==0 ? 0 : (first s in 0..max_new_tokens-1 with stop_after_step[s]==true mapped to s+1; otherwise max_new_tokens)",
      "terminal_logits = forward_state[executed_steps].logits",
    ];
    case "select-terminal-cache": return [
      "executed_steps = max_new_tokens==0 ? 0 : (first s in 0..max_new_tokens-1 with stop_after_step[s]==true mapped to s+1; otherwise max_new_tokens)",
      "terminal_past_key_values = forward_state[executed_steps].past_key_values",
    ];
  }
}
