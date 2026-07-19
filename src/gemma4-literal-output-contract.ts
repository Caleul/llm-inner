import { isDeepStrictEqual } from "node:util";
import type {
  Gemma4CompositeExecutionRequest,
  Gemma4CompositeExecutionResult,
  Gemma4CompositeProgram,
} from "./gemma4-composite.js";
import { buildGemma4LiteralInputContract } from "./gemma4-literal-input-contract.js";
import type { DenseF32Tensor, ReferenceF32ExecutionResult, ReferenceF32KeyValueCache } from "./types.js";

export interface Gemma4LiteralOutputTensorContract {
  role: "embeddings" | "per-layer-inputs" | "logits";
  name: string;
  dtype: "F32";
  layout: "row-major";
  axes: string[];
  shape: string[];
  producer: {
    activeOperationId: string;
    inactiveAliasPath?: string;
  };
}

export interface Gemma4LiteralOutputCacheProducer {
  layer: number;
  keyValueHeads: number;
  headDim: number;
}

/**
 * Normative public-result boundary for the literal forward and greedy state
 * machine. Calculation definitions already declare every intermediate; this
 * contract makes the values exposed to a caller, their shapes, cache ownership
 * and terminal-state relationships artifact data as well.
 */
export interface Gemma4LiteralOutputContract {
  kind: "gemma4-literal-output-contract";
  schemaVersion: 1;
  forward: {
    llmInputIds: {
      dtype: "I32";
      axes: ["batch", "sequence"];
      shape: ["input_ids.axis(0)", "input_ids.axis(1)"];
      source: "composite_llm_input_ids";
      modalReplacement: "replace-image-video-audio-token-ids-with-pad-token-id";
    };
    tensors: Gemma4LiteralOutputTensorContract[];
    valuesMap: "active-calculation-graph-outputs-including-public-tensors";
    pastKeyValues: {
      output: "past_key_values";
      ownership: "producer-layers-only";
      layout: "BHSD";
      dtype: "F32";
      sequenceLength: "past_key_values.sequence + input_ids.axis(1)";
      transition: "post-rope-key-and-value-state";
      producers: Gemma4LiteralOutputCacheProducer[];
    };
  };
  generation: {
    generatedTokenIds: {
      output: "generated_token_ids";
      dtype: "I32";
      shape: ["executed_steps"];
      domain: "0 <= token < vocab_size";
    };
    selectionLogits: {
      output: "selection_logits";
      dtype: "F32";
      shape: ["executed_steps", "1", "selection_sequence", "vocab_size"];
      source: "forward_state[step].logits";
    };
    stepPastKeyValues: {
      output: "step_past_key_values";
      shape: ["executed_steps", "producer_layers"];
      source: "forward_state[step+1].past_key_values";
      sequenceLength: "prompt_sequence + step + 1";
    };
    terminal: {
      logits: "forward_state[executed_steps].logits";
      pastKeyValues: "forward_state[executed_steps].past_key_values";
      zeroStepState: "forward_state[0]";
      nonZeroStepState: "last-completed-incremental-forward";
    };
  };
}

export interface Gemma4LiteralGenerationOutputState {
  prefill: Pick<ReferenceF32ExecutionResult, "logits" | "pastKeyValues">;
  generatedTokenIds: readonly number[];
  selectionLogits: readonly DenseF32Tensor[];
  stepPastKeyValues: readonly ReadonlyMap<number, ReferenceF32KeyValueCache>[];
  terminal: Pick<ReferenceF32ExecutionResult, "logits" | "pastKeyValues">;
}

export function buildGemma4LiteralOutputContract(program: Gemma4CompositeProgram): Gemma4LiteralOutputContract {
  const terminalOperationId = program.textProgram.epilogue.at(-1)?.id;
  if (terminalOperationId !== "lm_head" && terminalOperationId !== "final_logit_softcap") {
    throw new Error("Contrato de output Gemma 4 requer produtor terminal de logits registrado.");
  }
  const input = buildGemma4LiteralInputContract(program);
  return {
    kind: "gemma4-literal-output-contract",
    schemaVersion: 1,
    forward: {
      llmInputIds: {
        dtype: "I32",
        axes: ["batch", "sequence"],
        shape: ["input_ids.axis(0)", "input_ids.axis(1)"],
        source: "composite_llm_input_ids",
        modalReplacement: "replace-image-video-audio-token-ids-with-pad-token-id",
      },
      tensors: [
        {
          role: "embeddings",
          name: program.outputs.embeddings,
          dtype: "F32",
          layout: "row-major",
          axes: ["batch", "sequence", "hidden"],
          shape: ["input_ids.axis(0)", "input_ids.axis(1)", String(program.contract.text.hiddenSize)],
          producer: {
            activeOperationId: "composite_audio_scatter",
            inactiveAliasPath: "/forwardControl/modalityBranches/audio/inactiveIdentity",
          },
        },
        {
          role: "per-layer-inputs",
          name: program.outputs.perLayerInputs,
          dtype: "F32",
          layout: "row-major",
          axes: ["batch", "sequence", "layer", "per_layer_feature"],
          shape: ["input_ids.axis(0)", "input_ids.axis(1)", String(program.contract.text.layers), String(program.contract.text.perLayerInputSize)],
          producer: { activeOperationId: "composite_ple_combine_scale" },
        },
        {
          role: "logits",
          name: program.outputs.logits,
          dtype: "F32",
          layout: "row-major",
          axes: ["batch", "sequence", "token"],
          shape: ["input_ids.axis(0)", "input_ids.axis(1)", String(program.contract.text.vocabSize)],
          producer: { activeOperationId: terminalOperationId },
        },
      ],
      valuesMap: "active-calculation-graph-outputs-including-public-tensors",
      pastKeyValues: {
        output: "past_key_values",
        ownership: "producer-layers-only",
        layout: "BHSD",
        dtype: "F32",
        sequenceLength: "past_key_values.sequence + input_ids.axis(1)",
        transition: "post-rope-key-and-value-state",
        producers: input.text.cacheProducers.map(({ layer, keyValueHeads, headDim }) => ({ layer, keyValueHeads, headDim })),
      },
    },
    generation: {
      generatedTokenIds: {
        output: "generated_token_ids",
        dtype: "I32",
        shape: ["executed_steps"],
        domain: "0 <= token < vocab_size",
      },
      selectionLogits: {
        output: "selection_logits",
        dtype: "F32",
        shape: ["executed_steps", "1", "selection_sequence", "vocab_size"],
        source: "forward_state[step].logits",
      },
      stepPastKeyValues: {
        output: "step_past_key_values",
        shape: ["executed_steps", "producer_layers"],
        source: "forward_state[step+1].past_key_values",
        sequenceLength: "prompt_sequence + step + 1",
      },
      terminal: {
        logits: "forward_state[executed_steps].logits",
        pastKeyValues: "forward_state[executed_steps].past_key_values",
        zeroStepState: "forward_state[0]",
        nonZeroStepState: "last-completed-incremental-forward",
      },
    },
  };
}

export function validateGemma4LiteralOutputContract(
  contract: Gemma4LiteralOutputContract,
  program: Gemma4CompositeProgram,
): void {
  if (!isDeepStrictEqual(contract, buildGemma4LiteralOutputContract(program))) {
    throw new Error("Programa literal Gemma 4 possui contrato de outputs, cache ou estado terminal divergente.");
  }
}

/** Executes the serialized public forward-result invariants after calculation. */
export function executeGemma4LiteralForwardOutputContract(
  contract: Gemma4LiteralOutputContract,
  program: Gemma4CompositeProgram,
  request: Omit<Gemma4CompositeExecutionRequest, "tensors">,
  result: Gemma4CompositeExecutionResult,
): void {
  validateGemma4LiteralOutputContract(contract, program);
  const batch = request.inputIds.length;
  const sequence = request.inputIds[0]?.length ?? 0;
  validateI32Matrix(result.llmInputIds, batch, sequence, "composite_llm_input_ids");
  const padTokenId = program.textProgram.config.pad_token_id;
  if (!Number.isSafeInteger(padTokenId) || (padTokenId as number) < 0) {
    throw new Error("Contrato de output Gemma 4 requer pad_token_id inteiro não negativo.");
  }
  const expectedIds = request.inputIds.map((row) => row.map((token) => isModalToken(program, token) ? padTokenId as number : token));
  if (!isDeepStrictEqual(result.llmInputIds, expectedIds)) {
    throw new Error("Contrato de output Gemma 4 encontrou llmInputIds divergente da substituição modal declarada.");
  }

  const shapes = new Map<Gemma4LiteralOutputTensorContract["role"], number[]>([
    ["embeddings", [batch, sequence, program.contract.text.hiddenSize]],
    ["per-layer-inputs", [batch, sequence, program.contract.text.layers, program.contract.text.perLayerInputSize]],
    ["logits", [batch, sequence, program.contract.text.vocabSize]],
  ]);
  for (const output of contract.forward.tensors) {
    const tensor = result.values.get(output.name);
    if (!tensor) throw new Error(`Contrato de output Gemma 4 não encontrou tensor público ${output.name}.`);
    validateDenseTensor(tensor, shapes.get(output.role)!, output.name);
    if (output.role === "logits" && !sameDenseTensor(tensor, result.text.logits)) {
      throw new Error("Contrato de output Gemma 4 requer logits públicos idênticos ao estado textual terminal.");
    }
  }
  validateCache(contract, result.text.pastKeyValues, batch, pastSequence(request.pastKeyValues) + sequence, "forward");
}

/** Executes all greedy output/state relationships declared by the artifact. */
export function executeGemma4LiteralGenerationOutputContract(
  contract: Gemma4LiteralOutputContract,
  program: Gemma4CompositeProgram,
  request: { inputIds: number[][]; positionIds?: number[][]; maxNewTokens: number; eosTokenId?: number },
  result: Gemma4LiteralGenerationOutputState,
): void {
  validateGemma4LiteralOutputContract(contract, program);
  const steps = result.generatedTokenIds.length;
  if (steps > request.maxNewTokens || result.selectionLogits.length !== steps || result.stepPastKeyValues.length !== steps) {
    throw new Error("Contrato de output Gemma 4 encontrou cardinalidade greedy divergente de executed_steps.");
  }
  if (steps < request.maxNewTokens && (request.eosTokenId === undefined || result.generatedTokenIds.at(-1) !== request.eosTokenId)) {
    throw new Error("Contrato de output Gemma 4 encontrou parada antecipada sem EOS terminal.");
  }
  if (request.eosTokenId !== undefined && result.generatedTokenIds.slice(0, -1).includes(request.eosTokenId)) {
    throw new Error("Contrato de output Gemma 4 encontrou geração após um EOS já concluído.");
  }
  if (result.generatedTokenIds.some((token) => !Number.isSafeInteger(token) || token < 0 || token >= program.contract.text.vocabSize)) {
    throw new Error("Contrato de output Gemma 4 encontrou token greedy fora do vocabulário.");
  }
  const promptSequence = request.inputIds[0]?.length ?? 0;
  validateDenseTensor(result.prefill.logits, [1, promptSequence, program.contract.text.vocabSize], "forward_state[0].logits");
  validateCache(contract, result.prefill.pastKeyValues, 1, promptSequence, "forward_state[0]");
  for (let step = 0; step < steps; step += 1) {
    const selectionSequence = step === 0 ? promptSequence : 1;
    validateDenseTensor(result.selectionLogits[step]!, [1, selectionSequence, program.contract.text.vocabSize], `selection_logits[${step}]`);
    if (step === 0 && !sameDenseTensor(result.selectionLogits[step]!, result.prefill.logits)) {
      throw new Error("Contrato de output Gemma 4 requer selection_logits[0] idêntico aos logits de prefill.");
    }
    validateCache(contract, result.stepPastKeyValues[step]!, 1, promptSequence + step + 1, `step_past_key_values[${step}]`);
  }
  const terminalSequence = steps === 0 ? promptSequence : 1;
  validateDenseTensor(result.terminal.logits, [1, terminalSequence, program.contract.text.vocabSize], "terminal_logits");
  validateCache(contract, result.terminal.pastKeyValues, 1, promptSequence + steps, "terminal_past_key_values");
  if (steps === 0) {
    if (!sameDenseTensor(result.terminal.logits, result.prefill.logits) || !sameCache(result.terminal.pastKeyValues, result.prefill.pastKeyValues)) {
      throw new Error("Contrato de output Gemma 4 requer estado terminal zero-step idêntico ao prefill.");
    }
  } else if (!sameCache(result.terminal.pastKeyValues, result.stepPastKeyValues.at(-1)!)) {
    throw new Error("Contrato de output Gemma 4 requer cache terminal idêntico ao último forward incremental.");
  }
}

function validateCache(
  contract: Gemma4LiteralOutputContract,
  cache: ReadonlyMap<number, ReferenceF32KeyValueCache>,
  batch: number,
  sequence: number,
  owner: string,
): void {
  if (!(cache instanceof Map) || cache.size !== contract.forward.pastKeyValues.producers.length) {
    throw new Error(`Contrato de output Gemma 4 requer exatamente os caches produtores em ${owner}.`);
  }
  for (const producer of contract.forward.pastKeyValues.producers) {
    const entry = cache.get(producer.layer);
    if (!entry) throw new Error(`Contrato de output Gemma 4 não encontrou cache ${owner} do produtor ${producer.layer}.`);
    validateDenseTensor(entry.key, [batch, producer.keyValueHeads, sequence, producer.headDim], `${owner}.past_key_values[${producer.layer}].key`);
    validateDenseTensor(entry.value, [batch, producer.keyValueHeads, sequence, producer.headDim], `${owner}.past_key_values[${producer.layer}].value`);
  }
  for (const layer of cache.keys()) if (!contract.forward.pastKeyValues.producers.some((producer) => producer.layer === layer)) {
    throw new Error(`Contrato de output Gemma 4 rejeitou cache ${owner} sem produtor no layer ${layer}.`);
  }
}

function validateDenseTensor(tensor: DenseF32Tensor, shape: readonly number[], owner: string): void {
  const values = shape.reduce((total, dimension) => total * dimension, 1);
  if (!(tensor.values instanceof Float32Array) || !isDeepStrictEqual(tensor.shape, shape) || tensor.values.length !== values) {
    throw new Error(`Contrato de output Gemma 4 requer ${owner} F32 row-major com shape [${shape.join(",")}].`);
  }
}

function validateI32Matrix(values: number[][], batch: number, sequence: number, owner: string): void {
  if (values.length !== batch || values.some((row) => row.length !== sequence || row.some((value) => !Number.isSafeInteger(value)))) {
    throw new Error(`Contrato de output Gemma 4 requer ${owner} I32 [${batch},${sequence}].`);
  }
}

function isModalToken(program: Gemma4CompositeProgram, token: number): boolean {
  return token === program.contract.modalities.imageTokenId || token === program.contract.modalities.videoTokenId ||
    token === program.contract.modalities.audioTokenId;
}

function pastSequence(cache: ReadonlyMap<number, ReferenceF32KeyValueCache> | undefined): number {
  return cache?.values().next().value?.key.shape[2] ?? 0;
}

function sameDenseTensor(left: DenseF32Tensor, right: DenseF32Tensor): boolean {
  if (!isDeepStrictEqual(left.shape, right.shape) || left.values.length !== right.values.length) return false;
  return Buffer.from(left.values.buffer, left.values.byteOffset, left.values.byteLength)
    .equals(Buffer.from(right.values.buffer, right.values.byteOffset, right.values.byteLength));
}

function sameCache(
  left: ReadonlyMap<number, ReferenceF32KeyValueCache>,
  right: ReadonlyMap<number, ReferenceF32KeyValueCache>,
): boolean {
  if (left.size !== right.size) return false;
  for (const [layer, entry] of left) {
    const candidate = right.get(layer);
    if (!candidate || !sameDenseTensor(entry.key, candidate.key) || !sameDenseTensor(entry.value, candidate.value)) return false;
  }
  return true;
}
