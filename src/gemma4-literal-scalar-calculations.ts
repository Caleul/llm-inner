import { isDeepStrictEqual } from "node:util";
import type { Gemma4AudioAssignment } from "./gemma4-audio.js";
import type { Gemma4CompositeAssignment, Gemma4CompositeProgram } from "./gemma4-composite.js";
import {
  buildGemma4LiteralCalculationDomains,
  type Gemma4LiteralAssignmentDomain,
  type Gemma4LiteralCalculationScope,
} from "./gemma4-literal-domains.js";
import {
  buildGemma4LiteralLearnedOperandBindings,
  type Gemma4LiteralLearnedOperandRole,
} from "./gemma4-literal-learned-operands.js";
import type { Gemma4VisionAssignment } from "./gemma4-vision.js";
import type { DtypePolicy, Operation, ReductionSchedule } from "./types.js";
import { gemma4LiteralReductionUsesExactProducts } from "./gemma4-literal-linear-reduction-view.js";

type MultimodalAssignment = Gemma4CompositeAssignment | Gemma4VisionAssignment | Gemma4AudioAssignment;

export interface Gemma4LiteralScalarReduction {
  /** Named scalar index and its complete, non-preview domain. */
  indices: string[];
  order: "ascending-lexicographic" | "operation-declared" | "runtime-defined";
  schedule?: ReductionSchedule;
}

/**
 * A coordinate-level formula stored in the artifact itself. The formula may
 * refer only to declared inputs, named predecessors, learned operand roles,
 * exact numeric literals and the serialized reduction schedule beside it.
 */
export interface Gemma4LiteralScalarCalculation {
  scope: Gemma4LiteralCalculationScope;
  definitionId: string;
  operation: string;
  orderedInputs: string[];
  output: string;
  outputCoordinates: string[];
  learnedOperandRoles: Gemma4LiteralLearnedOperandRole[];
  formula: string;
  dtypePolicy: DtypePolicy;
  reduction?: Gemma4LiteralScalarReduction;
  reproducibility: "literal" | "fail-closed-runtime-reduction";
}

export interface Gemma4LiteralScalarCalculations {
  kind: "gemma4-literal-scalar-calculations";
  schemaVersion: 1;
  formulaLanguage: "indexed-ieee754-expression-v1";
  assignments: Gemma4LiteralScalarCalculation[];
}

interface Definition {
  scope: Gemma4LiteralCalculationScope;
  id: string;
  operation: string;
  inputs: string[];
  output: string;
  value: MultimodalAssignment | Operation;
}

export function buildGemma4LiteralScalarCalculations(program: Gemma4CompositeProgram): Gemma4LiteralScalarCalculations {
  const domains = buildGemma4LiteralCalculationDomains(program);
  const learned = buildGemma4LiteralLearnedOperandBindings(program);
  const assignments = definitions(program).map((definition): Gemma4LiteralScalarCalculation => {
    const domain = requiredDomain(domains.assignments, definition);
    const learnedOperandRoles = learned.assignments.find((candidate) =>
      candidate.scope === definition.scope && candidate.definitionId === definition.id)?.operands.map((operand) => operand.role) ?? [];
    const dtypePolicy = domain.domain.dtypePolicy ?? exactPolicy(domain.domain.dtype);
    const reduction = scalarReduction(definition, program, dtypePolicy);
    const reproducibility = dtypePolicy.accumulationDtype === "runtime-defined" || reduction?.order === "runtime-defined"
      ? "fail-closed-runtime-reduction" as const
      : "literal" as const;
    return {
      scope: definition.scope,
      definitionId: definition.id,
      operation: definition.operation,
      orderedInputs: [...definition.inputs],
      output: definition.output,
      outputCoordinates: domain.domain.axes.map((axis) => axis.name),
      learnedOperandRoles,
      formula: scalarFormula(definition, program, domain),
      dtypePolicy: structuredClone(dtypePolicy),
      ...(reduction ? { reduction } : {}),
      reproducibility,
    };
  });
  const keys = new Set<string>();
  for (const assignment of assignments) {
    const key = `${assignment.scope}:${assignment.definitionId}`;
    if (keys.has(key)) throw new Error(`Cálculo escalar literal Gemma 4 duplicado: ${key}.`);
    keys.add(key);
  }
  return {
    kind: "gemma4-literal-scalar-calculations",
    schemaVersion: 1,
    formulaLanguage: "indexed-ieee754-expression-v1",
    assignments,
  };
}

export function validateGemma4LiteralScalarCalculations(
  calculations: Gemma4LiteralScalarCalculations,
  program: Gemma4CompositeProgram,
): void {
  if (calculations.kind !== "gemma4-literal-scalar-calculations" || calculations.schemaVersion !== 1 ||
    calculations.formulaLanguage !== "indexed-ieee754-expression-v1") {
    throw new Error("Programa literal Gemma 4 possui cabeçalho de cálculos escalares inválido.");
  }
  if (!isDeepStrictEqual(calculations, buildGemma4LiteralScalarCalculations(program))) {
    throw new Error("Programa literal Gemma 4 possui fórmulas, casts ou reduções escalares ausentes ou divergentes.");
  }
}

export function requiredGemma4LiteralScalarCalculation(
  calculations: Gemma4LiteralScalarCalculations,
  scope: Gemma4LiteralCalculationScope,
  definitionId: string,
): Gemma4LiteralScalarCalculation {
  const calculation = calculations.assignments.find((candidate) =>
    candidate.scope === scope && candidate.definitionId === definitionId);
  if (!calculation) throw new Error(`${scope}:${definitionId}: cálculo escalar literal não foi declarado.`);
  return structuredClone(calculation);
}

function definitions(program: Gemma4CompositeProgram): Definition[] {
  return [
    ...program.assignments.map((value) => multimodalDefinition("composite", value)),
    ...program.visionProgram.assignments.map((value) => multimodalDefinition("vision", value)),
    ...program.audioProgram.assignments.map((value) => multimodalDefinition("audio", value)),
    ...program.textProgram.prelude.map((value) => textDefinition("text-prelude", value)),
    ...program.textProgram.layers.flatMap((layer) => layer.operations.map((value) => textDefinition("text-layer", value))),
    ...program.textProgram.epilogue.map((value) => textDefinition("text-epilogue", value)),
  ];
}

function multimodalDefinition(scope: "composite" | "vision" | "audio", value: MultimodalAssignment): Definition {
  return { scope, id: value.id, operation: value.operation, inputs: [...value.inputs], output: value.output, value };
}

function textDefinition(scope: "text-prelude" | "text-layer" | "text-epilogue", value: Operation): Definition {
  return { scope, id: value.id, operation: value.op, inputs: textInputs(value), output: value.output, value };
}

function textInputs(operation: Operation): string[] {
  switch (operation.op) {
    case "embedding": case "per_layer_embedding": return [operation.tokenInput];
    case "rms_norm": case "linear": case "reshape_heads": case "reshape_per_layer": case "select_per_layer": case "activation": case "tensor_scale": return [operation.input];
    case "rotary_embedding": return [operation.input, operation.positionInput];
    case "scaled_dot_product_attention": return [operation.query, operation.key, operation.value, operation.maskInput];
    case "elementwise": return [...operation.inputs];
  }
}

function scalarFormula(definition: Definition, program: Gemma4CompositeProgram, domain: Gemma4LiteralAssignmentDomain): string {
  const lhs = indexed(definition.output, domain.domain.axes.map((axis) => axis.name));
  if ("op" in definition.value) return textFormula(definition.value, lhs);
  const assignment = definition.value;
  const input = (index = 0): string => indexed(assignment.inputs[index]!, domain.domain.axes.map((axis) => axis.name));
  const cast = domain.domain.dtype === "BF16" ? "BF16" : domain.domain.dtype === "F32" ? "F32" : domain.domain.dtype;
  switch (assignment.operation) {
    case "placeholder-masks": return `${lhs} = tuple(input_ids==${program.contract.modalities.imageTokenId}, input_ids==${program.contract.modalities.videoTokenId}, input_ids==${program.contract.modalities.audioTokenId})[batch,sequence]`;
    case "vision-block-sequence-ids": return `${lhs} = contiguous_group_id(mm_token_type_ids in {1,2}) else -1`;
    case "causal-attention-mask": return `${lhs} = key<=query ? F32(0) : F32(-Infinity)`;
    case "vision-sliding-attention-mask": return `${lhs} = (key>query-${textSlidingWindow(program)} && (key<=query || same_nonnegative_vision_block)) ? F32(0) : F32(-Infinity)`;
    case "replace-multimodal-ids-with-pad": return `${lhs} = is_declared_modal_token(input_ids[batch,sequence]) ? ${numericConfig(program.textProgram.config.pad_token_id, "pad_token_id")} : input_ids[batch,sequence]`;
    case "embedding": return `${lhs} = F32(decode(weight)[input_ids[batch,sequence],feature]*F32(${Math.sqrt(program.contract.text.hiddenSize)}))`;
    case "per-layer-embedding": return `${lhs} = F32(decode(weight)[input_ids[batch,sequence],layer*per_layer_width+feature]*F32(${Math.sqrt(program.contract.text.perLayerInputSize)}))`;
    case "vision-feature-program": case "audio-feature-program": case "text-core": return `${lhs} = EVALUATE(calculationGraph.assignments where invocationId==${JSON.stringify(assignment.id)} in ordinal order, orderedInputs=[${assignment.inputs.join(",")}]).terminalOutput[${domain.domain.axes.map((axis) => axis.name).join(",")}]`;
    case "video-frame-flatten": return `${lhs} = ${assignment.inputs[0]}[floor(video_frame/frames),video_frame%frames,patch,${domain.domain.axes.at(-1)?.name}]`;
    case "masked-scatter": case "masked-scatter-image-features": case "masked-scatter-audio-features": return `${lhs} = placeholder_mask[batch,sequence] ? next_feature_row[feature] : ${assignment.inputs[0]}[batch,sequence,feature]; feature rows consumed in stable batch-major order`;
    case "linear": return linearFormula(lhs, assignment.inputs[0]!, cast, assignment.tensors?.length === 2, false, domain.domain.dtypePolicy?.reduction);
    case "clipped-linear": return linearFormula(lhs, assignment.inputs[0]!, cast, false, true, domain.domain.dtypePolicy?.reduction);
    case "scale-f32": return `${lhs} = F32(${input()} * F32(${scalarScale(assignment, program)}))`;
    case "reshape-per-layer": return `${lhs} = ${assignment.inputs[0]}[batch,sequence,layer*per_layer_width+feature]`;
    case "rms-norm": return `${lhs} = ${cast}(F32(F32(${input()} * F32(rsqrt(F32(REDUCE(feature=0..width-1,F32(input[...,feature]*input[...,feature]))/width + F32(${rmsEpsilon(definition.scope, program)})))))${assignment.tensors?.length ? "*decode(normalization-scale)[feature]" : ""}))`;
    case "add": case "attention-logit-add": return `${lhs} = ${cast}(F32(${input(0)} + ${input(1)}))`;
    case "pixel-affine": return `${lhs} = F32(2 * F32(${input()} - F32(0.5)))`;
    case "position-embedding-2d": return `${lhs} = ${cast}(F32(decode(position-table)[0,x,hidden] + decode(position-table)[1,y,hidden])); padding pair (-1,-1) yields 0`;
    case "reshape-heads": return `${lhs} = row_major_alias(${assignment.inputs[0]})[batch,patch,head*head_dim+head_feature]`;
    case "multidimensional-rope": return `${lhs} = BF16(F32(BF16(${assignment.inputs[0]}[batch,head,patch,paired_feature]*BF16(SLEEF_COS_F32(F32(position[axis]/F32(${program.visionProgram.tower.ropeTheta}**F32(2*pair/(head_dim/2))))))) +/- BF16(${assignment.inputs[0]}[batch,head,patch,rotated_paired_feature]*BF16(SLEEF_SIN_F32(F32(position[axis]/F32(${program.visionProgram.tower.ropeTheta}**F32(2*pair/(head_dim/2))))))))); axis=floor(head_feature/(head_dim/2)), pair=head_feature%(head_dim/4), sign/order follows rotate_half`;
    case "attention-score-matmul": return `${lhs} = ${cast}(REDUCE(head_feature=0..head_dim-1, F32(q[batch,head,query_patch,head_feature]*k[batch,head,key_patch,head_feature])))`;
    case "masked-softmax": return `${lhs} = ${cast}(padding_key ? F32(0) : F32(SLEEF_EXP_F32(F32(score-max_valid_key(score)))/REDUCE(key_patch=0..patches-1 in ascending order,SLEEF_EXP_F32(F32(score-max_valid_key(score))))))`;
    case "attention-value-matmul": return `${lhs} = ${cast}(REDUCE(key_patch=0..patches-1, F32(probability[batch,head,query_patch,key_patch]*value[batch,head,key_patch,head_feature])))`;
    case "gelu-tanh": return `${lhs} = ${cast}(F32(F32(0.5*x)*F32(1+SLEEF_TANH_F32(F32(sqrt(2/pi)*F32(x+F32(0.044715*F32(x*F32(x*x)))))))))`;
    case "multiply": return `${lhs} = ${cast}(F32(${input(0)} * ${input(1)}))`;
    case "pool-by-position": return `${lhs} = ${cast}(REDUCE(patch in stable ascending order mapped to pool_cell, F32(source[batch,patch,hidden]*F32(1/${program.visionProgram.tower.poolingKernelSize ** 2}))))`;
    case "pool-valid-mask": return `${lhs} = BOOL(any non-padding patch maps to pool_cell)`;
    case "strip-padding": return `${lhs} = ${assignment.inputs[0]}[stable_batch_major_true_mask_row,${domain.domain.axes.at(-1)?.name}]`;
    case "mask-input-features": return `${lhs} = ${assignment.inputs[1]}[batch,frame] ? ${input()} : F32(0)`;
    case "reshape-conv-features": return assignment.id === "audio_input_unsqueeze"
      ? `${lhs} = ${assignment.inputs[0]}[batch,frame,feature] where channel=0`
      : `${lhs} = ${assignment.inputs[0]}[batch,flattened_channel_feature%channels,frame,floor(flattened_channel_feature/channels)]`;
    case "conv2d-stride2": return `${lhs} = ${cast}(REDUCE(input_channel=0..channels-1,kernel_time=0..2,kernel_feature=0..2 in ascending lexicographic order, F32(padded_input[batch,input_channel,2*frame+kernel_time-1,2*feature+kernel_feature-1]*decode(convolution-kernel)[output_channel,input_channel,kernel_time,kernel_feature])))`;
    case "layer-norm-channels": return `${lhs} = ${cast}(F32(F32((x-F32(mean_channels(x)))*rsqrt(F32(variance_channels(x)+F32(${program.audioProgram.rmsNormEpsilon}))))*decode(normalization-scale)[channel]))`;
    case "relu": return `${lhs} = ${cast}(F32(max(0,${input()})))`;
    case "relative-position-encoding": return `${lhs} = BF16(hidden<${program.audioProgram.tower.hiddenSize / 2} ? SLEEF_SIN_F32(BF16(F32((${Math.floor((program.audioProgram.tower.attentionChunkSize + program.audioProgram.tower.attentionContextLeft - 1 + program.audioProgram.tower.attentionContextRight) / 2)}-relative_position)*BF16(SLEEF_EXP_F32(F32(-(hidden%${program.audioProgram.tower.hiddenSize / 2})*F32(log(10000)/F32(${program.audioProgram.tower.hiddenSize / 2 - 1}))))))))) : SLEEF_COS_F32(BF16(F32((${Math.floor((program.audioProgram.tower.attentionChunkSize + program.audioProgram.tower.attentionContextLeft - 1 + program.audioProgram.tower.attentionContextRight) / 2)}-relative_position)*BF16(SLEEF_EXP_F32(F32(-(hidden%${program.audioProgram.tower.hiddenSize / 2})*F32(log(10000)/F32(${program.audioProgram.tower.hiddenSize / 2 - 1}))))))))))`;
    case "clip": return `${lhs} = ${cast}(F32(min(${program.audioProgram.gradientClipping},max(${-program.audioProgram.gradientClipping},${input()}))))`;
    case "silu": return `${lhs} = ${cast}(F32(${input()} / F32(1+SLEEF_EXP_F32(F32(-${input()})))))`;
    case "per-dim-softplus-scale": return `${lhs} = F32(F32(${input()}*F32(head_dim^-0.5/log(2)))*BF16(F32(log1p(SLEEF_EXP_F32(decode(per-dimension-scale)[feature%head_dim])))))`;
    case "split-gated-linear-unit": return `${lhs} = ${cast}(F32(input[...,hidden]/F32(1+SLEEF_EXP_F32(F32(-input[...,hidden+hidden_size])))))`;
    case "causal-depthwise-convolution": return `${lhs} = ${cast}(REDUCE(kernel_index=0..kernel_size-1, F32((frame-kernel_size+1+kernel_index<0 ? F32(0) : input[batch,frame-kernel_size+1+kernel_index,channel])*decode(convolution-kernel)[channel,0,kernel_index])))`;
    case "chunked-attention-content-matmul": return `${lhs} = F32(REDUCE(head_feature=0..head_dim-1, F32(q_scaled[batch,head,block*chunk+query_in_block,head_feature]*context_key[batch,head,block,key_slot,head_feature])))`;
    case "relative-attention-position-matmul": return `${lhs} = F32(REDUCE(head_feature=0..head_dim-1, F32(q_scaled[batch,head,block*chunk+query_in_block,head_feature]*relative_key[relative_position,head,head_feature])))`;
    case "relative-attention-shift": return `${lhs} = source[batch,head,block,query_in_block,query_in_block+context-1-key_slot] after declared pad-flatten-slice-reshape; out-of-range source is F32(0)`;
    case "attention-softcap": return `${lhs} = F32(${program.audioProgram.tower.attentionLogitCap}*SLEEF_TANH_F32(F32(${input()}/${program.audioProgram.tower.attentionLogitCap})))`;
    case "chunked-attention-mask": return `${lhs} = valid_query_and_context_key ? ${input()} : F32(${program.audioProgram.invalidAttentionLogit})`;
    case "chunked-relative-attention-softmax": return `${lhs} = F32(exp(score-max_key(score))/REDUCE(key_slot=0..context-1, exp(score-max_key(score))))`;
    case "chunked-relative-attention-values": return `${lhs} = F32(REDUCE(key_slot=0..context-1, F32(probability[batch,head,block,query_in_block,key_slot]*context_value[batch,head,block,key_slot,head_feature])))`;
    case "cast-bf16": return `${lhs} = BF16(${input()})`;
    case "subsample-mask": return `${lhs} = ${assignment.inputs[0]}[batch,2*frame]`;
  }
}

function textFormula(operation: Operation, lhs: string): string {
  const cast = operation.dtypePolicy?.outputDtype === "BF16" ? "BF16" : "F32";
  switch (operation.op) {
    case "embedding": return `${lhs} = ${cast}(decode(weight)[input_ids[batch,sequence],feature]${operation.scale === undefined ? "" : `*F32(${operation.scale})`})`;
    case "per_layer_embedding": return `${lhs} = ${cast}(decode(weight)[input_ids[batch,sequence],layer*${operation.layerWidth}+feature]${operation.scale === undefined ? "" : `*F32(${operation.scale})`})`;
    case "rms_norm": return `${lhs} = ${cast}(F32(input*rsqrt(F32(REDUCE(feature=0..${operation.reductionSize ?? "width"}-1,F32(input*input))/${operation.reductionSize ?? "width"}+F32(${operation.epsilon}))))${operation.weightTransform === "one_plus_weight" ? "*F32(1+decode(normalization-scale)[feature])" : operation.weightTransform === "direct" ? "*decode(normalization-scale)[feature]" : ""})`;
    case "linear": return linearFormula(lhs, operation.input, cast, operation.bias !== undefined, false, operation.dtypePolicy.reduction);
    case "reshape_heads": return `${lhs} = ${operation.input}[batch,sequence,head*${operation.headDim}+head_feature] as ${operation.layout}`;
    case "reshape_per_layer": return `${lhs} = ${operation.input}[batch,sequence,layer*${operation.layerWidth}+feature]`;
    case "select_per_layer": return `${lhs} = ${operation.input}[batch,sequence,${operation.layerIndex},feature]`;
    case "rotary_embedding": return `${lhs} = ${cast}(F32(${operation.input}[...,head_feature]*${operation.rotaryCasts ? "BF16" : "F32"}(SLEEF_COS_F32(position*theta_power(theta=${operation.theta},rotary_dim=${operation.rotaryDim}))) + rotate_${operation.layout}(${operation.input})*${operation.rotaryCasts ? "BF16" : "F32"}(SLEEF_SIN_F32(position*theta_power(theta=${operation.theta},rotary_dim=${operation.rotaryDim})))))`;
    case "scaled_dot_product_attention": return `${lhs} = ${cast}(REDUCE(key=0..K-1, ${operation.numericImplementation ? "BF16" : "F32"}(F32(exp(F32(masked_score-max_key(masked_score)))/REDUCE(key=0..K-1,exp(F32(masked_score-max_key(masked_score)))))*value)); masked_score=${operation.numericImplementation ? "BF16" : "F32"}(F32(REDUCE(head_feature=0..${operation.headDim - 1},F32(q*k))*F32(${operation.scale}))${operation.scoreSoftcap === undefined ? "" : ` passed through F32(${operation.scoreSoftcap}*SLEEF_TANH_F32(score/${operation.scoreSoftcap}))`} + mask))`;
    case "activation": {
      if (operation.function === "gelu" && operation.approximation === "tanh") return `${lhs} = ${cast}(F32(F32(0.5*input)*F32(1+SLEEF_TANH_F32(F32(sqrt(2/pi)*F32(input+F32(0.044715*F32(input*F32(input*input)))))))))`;
      if (operation.function === "silu" || operation.function === "swish") return `${lhs} = ${cast}(F32(input/F32(1+SLEEF_EXP_F32(F32(-input)))))`;
      return `${lhs} = ${cast}(${operation.function}${operation.approximation ? `[${operation.approximation}]` : ""}(input))`;
    }
    case "elementwise": {
      if (operation.kind === "add") return `${lhs} = ${cast}(F32(${operation.inputs[0]}+${operation.inputs[1]}))`;
      if (operation.kind === "multiply") return `${lhs} = ${cast}(F32(${operation.inputs[0]}*${operation.inputs[1]}))`;
      if (operation.kind === "scale") return `${lhs} = ${cast}(F32(${operation.inputs[0]}*F32(${operation.scalar})))`;
      return `${lhs} = ${cast}(F32(F32(${operation.scalar}*BF16(SLEEF_TANH_F32(BF16(F32(${operation.inputs[0]}/F32(${operation.scalar}))))))))`;
    }
    case "tensor_scale": return `${lhs} = ${cast}(F32(${operation.input}*decode(tensor-scale)[0]))`;
  }
}

function scalarReduction(definition: Definition, program: Gemma4CompositeProgram, policy: DtypePolicy): Gemma4LiteralScalarReduction | undefined {
  const indices = reductionIndices(definition, program);
  if (indices.length === 0) return undefined;
  if (policy.accumulationDtype === "runtime-defined") return { indices, order: "runtime-defined" };
  if (policy.reduction) return { indices, order: "operation-declared", schedule: structuredClone(policy.reduction) };
  return { indices, order: "ascending-lexicographic" };
}

function reductionIndices(definition: Definition, program: Gemma4CompositeProgram): string[] {
  switch (definition.operation) {
    case "linear": case "clipped-linear": return ["input_feature=0..in_features-1"];
    case "rms_norm": case "rms-norm": return ["feature=0..width-1"];
    case "conv2d-stride2": return ["input_channel=0..channels-1", "kernel_time=0..2", "kernel_feature=0..2"];
    case "layer-norm-channels": return ["channel=0..channels-1"];
    case "causal-depthwise-convolution": return ["kernel_index=0..kernel_size-1"];
    case "attention-score-matmul": case "chunked-attention-content-matmul": case "relative-attention-position-matmul": return [`head_feature=0..${definition.scope === "vision" ? program.visionProgram.tower.headDim : program.audioProgram.tower.headDim}-1`];
    case "masked-softmax": return ["key_patch=0..patches-1"];
    case "attention-value-matmul": return ["key_patch=0..patches-1"];
    case "chunked-relative-attention-softmax": case "chunked-relative-attention-values": return ["key_slot=0..context-1"];
    case "pool-by-position": return ["patch=0..patches-1 mapped to pool_cell"];
    case "scaled_dot_product_attention": return ["head_feature=0..head_dim-1", "key=0..K-1"];
    default: return [];
  }
}

function requiredDomain(domains: readonly Gemma4LiteralAssignmentDomain[], definition: Definition): Gemma4LiteralAssignmentDomain {
  const domain = domains.find((candidate) => candidate.scope === definition.scope && candidate.definitionId === definition.id);
  if (!domain) throw new Error(`${definition.scope}:${definition.id}: domínio ausente para cálculo escalar.`);
  return domain;
}

function linearFormula(
  lhs: string,
  input: string,
  cast: string,
  bias: boolean,
  clipped: boolean,
  reduction: ReductionSchedule | undefined,
): string {
  const x = clipped ? `F32(min(decode(input-max),max(decode(input-min),${input}[...,input_feature])))` : `${input}[...,input_feature]`;
  const product = reduction && gemma4LiteralReductionUsesExactProducts(reduction)
    ? `exact_product(${x}*decode(weight)[output_feature,input_feature])`
    : `F32(${x}*decode(weight)[output_feature,input_feature])`;
  const sum = `REDUCE(input_feature=0..in_features-1,${product})`;
  const biased = bias ? `F32(${sum}+decode(bias)[output_feature])` : sum;
  return `${lhs} = ${cast}(${clipped ? `min(decode(output-max),max(decode(output-min),${biased}))` : biased})`;
}

function scalarScale(assignment: MultimodalAssignment, program: Gemma4CompositeProgram): string {
  if (assignment.id === "vision_pool_scale") return String(Math.sqrt(program.visionProgram.tower.hiddenSize));
  if (assignment.semantics?.includes("residual_weight")) return String(program.audioProgram.tower.residualWeight);
  if (assignment.id === "composite_ple_context_scale") return String(program.contract.text.hiddenSize ** -0.5);
  if (assignment.id.includes("k_scale")) return String(Math.log1p(Math.exp(1)) / Math.log(2));
  return String(2 ** -0.5);
}

function rmsEpsilon(scope: Gemma4LiteralCalculationScope, program: Gemma4CompositeProgram): number {
  return scope === "audio" ? program.audioProgram.rmsNormEpsilon : scope === "vision" ? program.visionProgram.rmsNormEpsilon : numericConfig(program.textProgram.config.rms_norm_eps, "rms_norm_eps");
}

function textSlidingWindow(program: Gemma4CompositeProgram): number {
  const windows = new Set(program.textProgram.layers.flatMap((layer) => layer.operations)
    .filter((operation) => operation.op === "scaled_dot_product_attention" && operation.slidingWindow !== undefined)
    .map((operation) => (operation as Extract<Operation, { op: "scaled_dot_product_attention" }>).slidingWindow!));
  if (windows.size !== 1) throw new Error("Programa Gemma 4 não declara uma janela deslizante textual única.");
  return [...windows][0]!;
}

function numericConfig(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${name} inválido para cálculo escalar literal.`);
  return value;
}

function exactPolicy(outputDtype: string): DtypePolicy {
  return { inputDtype: "declared-input", computeDtype: "exact-selection", accumulationDtype: "none", outputDtype };
}

function indexed(name: string, coordinates: readonly string[]): string {
  return `${name}[${coordinates.join(",")}]`;
}
