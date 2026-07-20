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
import {
  constantGemma4LiteralReductionExtent,
  tensorAxisGemma4LiteralReductionExtent,
  validateGemma4LiteralReductionIndexDomains,
  type Gemma4LiteralReductionExtent,
  type Gemma4LiteralReductionIndexDomain,
} from "./gemma4-literal-reduction-domains.js";

type MultimodalAssignment = Gemma4CompositeAssignment | Gemma4VisionAssignment | Gemma4AudioAssignment;

export interface Gemma4LiteralScalarReduction {
  /** Named scalar index and its complete, non-preview domain. */
  indices: string[];
  /** Executable binding for every symbolic bound named by `indices`. */
  domains: Gemma4LiteralReductionIndexDomain[];
  order: "ascending-lexicographic" | "operation-declared" | "runtime-defined";
  schedule?: ReductionSchedule;
}

export type Gemma4LiteralScalarReductionProgram =
  | "ARM_NEON_BF16_DOT_F32"
  | "ORDERED_F32_REDUCE_MAX"
  | "ORDERED_F32_REDUCE_SUM"
  | "ORDERED_F32_DOT"
  | "PYTORCH_F32_VECTOR_REDUCE_MAX"
  | "PYTORCH_F32_VECTOR_REDUCE_SUM";

/**
 * Operations such as attention contain several mathematically distinct
 * reductions. A single assignment-level `reduction` cannot truthfully describe
 * score dots, maximum, exponential sum, and value dots at once, so the artifact
 * records every stage in execution order and binds it to an embedded program.
 */
export interface Gemma4LiteralScalarReductionStage {
  id: "score-dot" | "softmax-maximum" | "softmax-exponential-sum" | "context-dot";
  indices: string[];
  domains: Gemma4LiteralReductionIndexDomain[];
  order: "ascending-lexicographic" | "operation-declared";
  program: Gemma4LiteralScalarReductionProgram;
  identity: "-Infinity" | "F32(0)";
  predicate?: string;
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
  /**
   * Executable scalar statements in dependency order. `formula` remains the
   * compact output-first audit rendering, while this array declares every
   * local coordinate, reduction temporary, precondition, and final output in
   * the order a checkpoint-independent interpreter must evaluate them.
   */
  scalarAssignments: string[];
  formula: string;
  dtypePolicy: DtypePolicy;
  reduction?: Gemma4LiteralScalarReduction;
  reductionStages?: Gemma4LiteralScalarReductionStage[];
  reproducibility: "literal" | "fail-closed-runtime-reduction";
}

export interface Gemma4LiteralScalarCalculations {
  kind: "gemma4-literal-scalar-calculations";
  schemaVersion: 3;
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
    const reductionStages = scalarReductionStages(definition);
    // Staged operations own several incompatible reductions; retaining one
    // assignment-level reduction beside them would falsely override the stage
    // programs (the defect schema v20 is designed to eliminate).
    const reduction = reductionStages.length === 0 ? scalarReduction(definition, program, dtypePolicy, domain) : undefined;
    const reproducibility = dtypePolicy.accumulationDtype === "runtime-defined" || reduction?.order === "runtime-defined"
      ? "fail-closed-runtime-reduction" as const
      : "literal" as const;
    const formula = scalarFormula(definition, program, domain);
    const scalarAssignments = dependencyOrderedScalarAssignments(
      formula,
      definition.output,
      `${definition.scope}:${definition.id}`,
    );
    return {
      scope: definition.scope,
      definitionId: definition.id,
      operation: definition.operation,
      orderedInputs: [...definition.inputs],
      output: definition.output,
      outputCoordinates: domain.domain.axes.map((axis) => axis.name),
      learnedOperandRoles,
      scalarAssignments,
      formula,
      dtypePolicy: structuredClone(dtypePolicy),
      ...(reduction ? { reduction } : {}),
      ...(reductionStages.length > 0 ? { reductionStages } : {}),
      reproducibility,
    };
  });
  const keys = new Set<string>();
  for (const assignment of assignments) {
    const key = `${assignment.scope}:${assignment.definitionId}`;
    if (keys.has(key)) throw new Error(`Cálculo escalar literal Gemma 4 duplicado: ${key}.`);
    keys.add(key);
  }
  validateClosedReductionFormulas(assignments);
  validateExplicitCoordinateFormulas(assignments);
  validateOperandClosedFormulas(assignments);
  return {
    kind: "gemma4-literal-scalar-calculations",
    schemaVersion: 3,
    formulaLanguage: "indexed-ieee754-expression-v1",
    assignments,
  };
}

/**
 * Forward formulas are intentionally rendered output-first for a reader, but
 * multi-statement formulas also declare local values used by that output.
 * Turn that finite presentation into an executable program by evaluating the
 * already dependency-ordered local/precondition tail before the final output.
 * This applies to the complete operation class rather than selected layer IDs.
 */
function dependencyOrderedScalarAssignments(formula: string, output: string, owner: string): string[] {
  const presentation = formula.split(";").map((statement) => statement.trim()).filter(Boolean);
  if (presentation.length === 0 || !new RegExp(`^${escapeRegExp(output)}\\s*\\[`).test(presentation[0]!)) {
    throw new Error(`${owner}: fórmula escalar não inicia pela saída declarada ${output}.`);
  }
  if (presentation.length === 1) return presentation;

  const locals = new Set(presentation.slice(1).flatMap(declaredScalarNames));
  const executable = [...presentation.slice(1), presentation[0]!];
  const available = new Set<string>();
  for (const statement of executable) {
    const declared = declaredScalarNames(statement);
    if (!statement.startsWith("require ") && declared.length === 0) {
      throw new Error(`${owner}: statement escalar não declara destino: ${statement}.`);
    }
    const expression = scalarStatementExpression(statement);
    for (const local of locals) {
      if (containsIdentifier(expression, local) && !available.has(local)) {
        throw new Error(`${owner}: atribuição escalar lê ${local} antes de sua declaração.`);
      }
    }
    for (const local of declared) available.add(local);
  }
  if (!executable.at(-1)?.startsWith(`${output}[`)) {
    throw new Error(`${owner}: programa escalar não termina na saída declarada ${output}.`);
  }
  return executable;
}

function declaredScalarNames(statement: string): string[] {
  if (statement.startsWith("require ")) return [];
  const struct = statement.match(/^STRUCT\(([^)]+)\)\s*=/);
  if (struct) {
    const names = struct[1]!.split(",").map((name) => name.trim());
    if (names.some((name) => !/^[A-Za-z_][A-Za-z0-9_]*$/.test(name))) {
      throw new Error(`Declaração escalar STRUCT inválida: ${statement}.`);
    }
    return names;
  }
  const scalar = statement.match(/^([A-Za-z_][A-Za-z0-9_]*)(?:\[[^=]*\])?\s*=/);
  return scalar ? [scalar[1]!] : [];
}

function scalarStatementExpression(statement: string): string {
  if (statement.startsWith("require ")) return statement.slice("require ".length);
  const equals = statement.indexOf("=");
  if (equals < 0) throw new Error(`Atribuição escalar sem '=': ${statement}.`);
  // Stage IDs such as reductionStages[score-dot] are labels, not reads of the
  // scalar local `score`; the schedule itself is bound beside the assignment.
  return statement.slice(equals + 1).replace(/reductionStages\[[^\]]+\]/g, "reductionStage");
}

function containsIdentifier(value: string, identifier: string): boolean {
  return new RegExp(`(^|[^A-Za-z0-9_])${escapeRegExp(identifier)}([^A-Za-z0-9_]|$)`).test(value);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function validateExplicitCoordinateFormulas(assignments: readonly Gemma4LiteralScalarCalculation[]): void {
  for (const assignment of assignments) {
    const formula = assignment.formula;
    if (assignment.operation === "pool-by-position" && (!formula.includes("VISION_POOL_SLOT") || !formula.includes("F32_FMA"))) {
      throw new Error(`${assignment.scope}:${assignment.definitionId}: pool vision não declara slot e redução FMA por coordenada.`);
    }
    if (assignment.operation === "pool-valid-mask" && !formula.includes("VISION_POOL_CELL_HAS_PATCH")) {
      throw new Error(`${assignment.scope}:${assignment.definitionId}: máscara pool vision não declara o programa existencial de patches.`);
    }
    if (assignment.operation === "relative-attention-shift" && !formula.includes("AUDIO_RELATIVE_SHIFT_SOURCE")) {
      throw new Error(`${assignment.scope}:${assignment.definitionId}: relative shift audio não declara sua coordenada fonte.`);
    }
    if ((assignment.operation === "chunked-attention-content-matmul" || assignment.operation === "chunked-relative-attention-values" ||
      assignment.operation === "chunked-attention-mask") && (!formula.includes("query_index=") || !formula.includes("key_index="))) {
      throw new Error(`${assignment.scope}:${assignment.definitionId}: operação chunked audio não declara índices de query e key.`);
    }
    if (assignment.operation === "relative-attention-position-matmul" && !formula.includes("query_index=")) {
      throw new Error(`${assignment.scope}:${assignment.definitionId}: score posicional audio não declara o índice de query.`);
    }
    if (assignment.operation === "reshape_heads" && !formula.includes("row_major_alias")) {
      throw new Error(`${assignment.scope}:${assignment.definitionId}: reshape de heads não declara alias row-major.`);
    }
  }
}

/**
 * A formula is not literal when a reader still has to infer which declared
 * operand a prose alias such as `input`, `q`, `value`, or `padded_input`
 * denotes. Keep this check operation-class based so a newly added layer cannot
 * silently reintroduce a free operand while an older layer remains explicit.
 */
function validateOperandClosedFormulas(assignments: readonly Gemma4LiteralScalarCalculation[]): void {
  const forbiddenByOperation = new Map<string, RegExp>([
    ["linear", /\.\.\./],
    ["clipped-linear", /\.\.\./],
    ["rms-norm", /\binput\b|\.\.\./],
    ["rms_norm", /\binput\b|\.\.\./],
    ["activation", /\binput\b/],
    ["elementwise", /(?<![\w.])(?:residual|input)(?![\w.])/],
    ["gelu-tanh", /\bx\b/],
    ["multidimensional-rope", /\+\/-|\bfollows\b|\bposition\[/],
    ["attention-score-matmul", /\bq\[|\bk\[/],
    ["attention-value-matmul", /\bprobability\[|\bvalue\[/],
    ["reshape-conv-features", /\bwhere\b/],
    ["conv2d-stride2", /\bpadded_input\[/],
    ["layer-norm-channels", /\binput\[|\bx\b/],
    ["split-gated-linear-unit", /\binput\[/],
    ["causal-depthwise-convolution", /\binput\[/],
  ]);
  const bindEveryOrderedInput = new Set([
    "embedding", "per-layer-embedding", "per_layer_embedding",
    "linear", "clipped-linear", "rms-norm", "rms_norm", "activation", "gelu-tanh",
    "elementwise", "tensor_scale",
    "multidimensional-rope", "attention-score-matmul", "attention-value-matmul",
    "reshape-conv-features", "conv2d-stride2", "layer-norm-channels",
    "split-gated-linear-unit", "causal-depthwise-convolution",
  ]);
  for (const assignment of assignments) {
    const forbidden = forbiddenByOperation.get(assignment.operation);
    if (forbidden?.test(assignment.formula)) {
      throw new Error(`${assignment.scope}:${assignment.definitionId}: fórmula conserva operando livre ou prosa não executável.`);
    }
    for (const input of bindEveryOrderedInput.has(assignment.operation) ? assignment.orderedInputs : []) {
      if (!assignment.formula.includes(input)) {
        throw new Error(`${assignment.scope}:${assignment.definitionId}: fórmula não vincula o operando ordenado ${input}.`);
      }
    }
  }
}

function validateClosedReductionFormulas(assignments: readonly Gemma4LiteralScalarCalculation[]): void {
  const opaqueSoftmax = /\b(?:masked_score-max_key|score-max_key|score-max_valid_key|max_k|max_context|sum_k_ascending|sum_context_ascending)\b/;
  for (const assignment of assignments) {
    if (opaqueSoftmax.test(assignment.formula)) {
      throw new Error(`${assignment.scope}:${assignment.definitionId}: fórmula conserva helper opaco de redução softmax.`);
    }
    const stages = assignment.reductionStages ?? [];
    if (assignment.reduction) validateReductionDomains(assignment.reduction.indices, assignment.reduction.domains, assignment.orderedInputs, `${assignment.scope}:${assignment.definitionId}`);
    for (const stage of stages) validateReductionDomains(stage.indices, stage.domains, assignment.orderedInputs, `${assignment.scope}:${assignment.definitionId}:${stage.id}`);
    const isAttention = assignment.operation === "scaled_dot_product_attention";
    const isSoftmax = assignment.operation === "masked-softmax" || assignment.operation === "chunked-relative-attention-softmax";
    if (isAttention && (stages.length !== 4 || stages.map((stage) => stage.id).join(",") !== "score-dot,softmax-maximum,softmax-exponential-sum,context-dot")) {
      throw new Error(`${assignment.scope}:${assignment.definitionId}: atenção não declara as quatro reduções escalares em ordem.`);
    }
    if (isSoftmax && (stages.length !== 2 || stages.map((stage) => stage.id).join(",") !== "softmax-maximum,softmax-exponential-sum")) {
      throw new Error(`${assignment.scope}:${assignment.definitionId}: softmax não declara máximo e soma exponencial em ordem.`);
    }
  }
}

export function validateGemma4LiteralScalarCalculations(
  calculations: Gemma4LiteralScalarCalculations,
  program: Gemma4CompositeProgram,
): void {
  if (calculations.kind !== "gemma4-literal-scalar-calculations" || calculations.schemaVersion !== 3 ||
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
  if ("op" in definition.value) return textFormula(definition.value, lhs, domain);
  const assignment = definition.value;
  const input = (index = 0): string => indexed(assignment.inputs[index]!, domain.domain.axes.map((axis) => axis.name));
  const cast = domain.domain.dtype === "BF16" ? "BF16" : domain.domain.dtype === "F32" ? "F32" : domain.domain.dtype;
  switch (assignment.operation) {
    case "vision-block-sequence-ids": return `${lhs} = CONTIGUOUS_VISION_GROUP_ID(mm_token_type_ids[batch,0..sequence],sequence)`;
    case "causal-attention-mask": return `${lhs} = key<=query ? F32(0) : F32(-Infinity)`;
    case "vision-sliding-attention-mask": return `${lhs} = (key>query-${textSlidingWindow(program)} && (key<=query || (vision_block_sequence_ids[batch,query]>=0 && vision_block_sequence_ids[batch,query]==vision_block_sequence_ids[batch,key]))) ? F32(0) : F32(-Infinity)`;
    case "replace-multimodal-ids-with-pad": {
      const modalities = program.contract.modalities;
      const token = "input_ids[batch,sequence]";
      const predicate = [`${token}==${modalities.imageTokenId}`, ...(modalities.videoTokenId === undefined ? [] : [`${token}==${modalities.videoTokenId}`]), `${token}==${modalities.audioTokenId}`].join(" || ");
      return `${lhs} = (${predicate}) ? ${numericConfig(program.textProgram.config.pad_token_id, "pad_token_id")} : ${token}`;
    }
    case "embedding": return `${lhs} = F32(decode(weight)[${assignment.inputs[0]}[batch,sequence],hidden]*F32(${Math.sqrt(program.contract.text.hiddenSize)}))`;
    case "per-layer-embedding": return `${lhs} = F32(decode(weight)[${assignment.inputs[0]}[batch,sequence],layer*${program.contract.text.perLayerInputSize}+ple_feature]*F32(${Math.sqrt(program.contract.text.perLayerInputSize)}))`;
    case "vision-feature-program": case "audio-feature-program": case "text-core": return `${lhs} = EVALUATE(calculationGraph.assignments where invocationId==${JSON.stringify(assignment.id)} in ordinal order, orderedInputs=[${assignment.inputs.join(",")}]).terminalOutput[${domain.domain.axes.map((axis) => axis.name).join(",")}]`;
    case "video-frame-flatten": return `${lhs} = ${assignment.inputs[0]}[floor(video_frame/(${assignment.inputs[0]}.shape[1])),video_frame%(${assignment.inputs[0]}.shape[1]),patch,${domain.domain.axes.at(-1)?.name}]`;
    case "masked-scatter": case "masked-scatter-image-features": case "masked-scatter-audio-features": {
      const token = requiredGemma4LiteralPlaceholderTokenId(assignment);
      const prior = assignment.inputs[0]!, inputIds = assignment.inputs[1]!, features = assignment.inputs[2]!;
      const feature = domain.domain.axes.at(-1)?.name;
      if (!feature) throw new Error(`${assignment.id}: masked scatter requer eixo de feature de saída.`);
      const predicate = `${inputIds}[batch,sequence]==${token}`;
      return `${lhs} = ${predicate} ? ${features}[STABLE_TRUE_PREFIX_RANK(${inputIds}==${token},batch,sequence),${feature}] : ${prior}[batch,sequence,${feature}]; ` +
        `require ${features}.shape[0]==STABLE_TRUE_COUNT(${inputIds}==${token})`;
    }
    case "linear": return linearFormula(lhs, assignment.inputs[0]!, inputReductionCoordinates(domain), cast, assignment.tensors?.length === 2, false, domain.domain.dtypePolicy?.reduction);
    case "clipped-linear": return linearFormula(lhs, assignment.inputs[0]!, inputReductionCoordinates(domain), cast, false, true, domain.domain.dtypePolicy?.reduction);
    case "scale-f32": return `${lhs} = F32(${input()} * F32(${scalarScale(assignment, program)}))`;
    case "reshape-per-layer": return `${lhs} = ${assignment.inputs[0]}[batch,sequence,layer*${program.contract.text.perLayerInputSize}+ple_feature]`;
    case "rms-norm": return rmsNormFormula(
      lhs,
      assignment.inputs[0]!,
      domain,
      cast,
      rmsEpsilon(definition.scope, program),
      assignment.tensors?.length ? "direct" : "none",
    );
    case "add": case "attention-logit-add": return `${lhs} = ${cast}(F32(${input(0)} + ${input(1)}))`;
    case "pixel-affine": return `${lhs} = F32(2 * F32(${input()} - F32(0.5)))`;
    case "position-embedding-2d": {
      const positions = assignment.inputs[0]!;
      return `${lhs} = ${positions}[batch,patch,0]==-1 && ${positions}[batch,patch,1]==-1 ? ${cast}(0) : ${cast}(F32(decode(position-table)[0,${positions}[batch,patch,0],hidden] + decode(position-table)[1,${positions}[batch,patch,1],hidden]))`;
    }
    case "reshape-heads": return `${lhs} = row_major_alias(${assignment.inputs[0]})[batch,patch,head*${program.visionProgram.tower.headDim}+head_feature]`;
    case "multidimensional-rope": return visionRotaryFormula(assignment, program, lhs);
    case "attention-score-matmul": return `${lhs} = ${cast}(REDUCE(head_feature=0..${program.visionProgram.tower.headDim - 1}, F32(${assignment.inputs[0]}[batch,head,query_patch,head_feature]*${assignment.inputs[1]}[batch,head,key_patch,head_feature])))`;
    case "masked-softmax": {
      const score = assignment.inputs[0]!, positions = assignment.inputs[1]!;
      const valid = `${positions}[batch,key_patch,0]!=-1 && ${positions}[batch,key_patch,1]!=-1`;
      return `${lhs} = ${cast}(${positions}[batch,key_patch,0]==-1 && ${positions}[batch,key_patch,1]==-1 ? F32(0) : F32(exponential[key_patch]/total)); ` +
        `maximum=ORDERED_F32_REDUCE_MAX(${score}[batch,head,query_patch,key_patch],key_patch=0..${score}.shape[3]-1 where ${valid}); ` +
        `exponential[key_patch]=SLEEF_EXP_F32(F32(${score}[batch,head,query_patch,key_patch]-maximum)); ` +
        `total=ORDERED_F32_REDUCE_SUM(exponential[key_patch],key_patch=0..${score}.shape[3]-1 where ${valid})`;
    }
    case "attention-value-matmul": {
      const headDim = program.visionProgram.tower.headDim;
      return `${lhs} = ${cast}(REDUCE(key_patch=0..${assignment.inputs[0]}.shape[3]-1, F32(${assignment.inputs[0]}[batch,floor(hidden/${headDim}),patch,key_patch]*${assignment.inputs[1]}[batch,floor(hidden/${headDim}),key_patch,hidden%${headDim}])))`;
    }
    case "gelu-tanh": {
      const source = input();
      return `${lhs} = ${cast}(F32(F32(0.5*${source})*F32(1+SLEEF_TANH_F32(F32(${Math.sqrt(2 / Math.PI)}*F32(${source}+F32(0.044715*F32(${source}*F32(${source}*${source})))))))))`;
    }
    case "multiply": return `${lhs} = ${cast}(F32(${input(0)} * ${input(1)}))`;
    case "pool-by-position": {
      const source = assignment.inputs[0]!, positions = assignment.inputs[1]!, kernel = program.visionProgram.tower.poolingKernelSize;
      return `${lhs} = ${cast}(acc[${positions}.shape[1]-1]); pool_slot[patch]=VISION_POOL_SLOT(${positions}[batch,0..${positions}.shape[1]-1],patch,${kernel},floor(${positions}.shape[1]/${kernel ** 2})); ` +
        `acc[-1]=F32(0); acc[patch]=pool_slot[patch]==pool_cell ? F32_FMA(acc[patch-1],${source}[batch,patch,hidden],F32(1/F32(${kernel ** 2}))) : acc[patch-1], patch=0..${positions}.shape[1]-1 ascending`;
    }
    case "pool-valid-mask": {
      const positions = assignment.inputs[0]!, kernel = program.visionProgram.tower.poolingKernelSize;
      return `${lhs} = BOOL(VISION_POOL_CELL_HAS_PATCH(${positions}[batch,0..${positions}.shape[1]-1],pool_cell,${kernel},floor(${positions}.shape[1]/${kernel ** 2})))`;
    }
    case "strip-padding": {
      const source = assignment.inputs[0]!, mask = assignment.inputs[1]!, row = domain.domain.axes[0]?.name, feature = domain.domain.axes[1]?.name;
      if (!row || !feature) throw new Error(`${assignment.id}: strip-padding requer domínio [linha,feature].`);
      return `${lhs} = ${source}[source_coordinate.batch,source_coordinate.sequence,${feature}]; ` +
        `source_coordinate=STABLE_TRUE_COORDINATE_AT_RANK(${mask},${row}); require ${assignment.output}.shape[0]==STABLE_TRUE_COUNT(${mask})`;
    }
    case "mask-input-features": return `${lhs} = ${assignment.inputs[1]}[batch,frame] ? ${input()} : F32(0)`;
    case "reshape-conv-features": return assignment.id === "audio_input_unsqueeze"
      ? `${lhs} = ${assignment.inputs[0]}[batch,frame,feature]; require channel==0`
      : `${lhs} = ${assignment.inputs[0]}[batch,flattened_channel_feature%${assignment.inputs[0]}.shape[1],frame,floor(flattened_channel_feature/${assignment.inputs[0]}.shape[1])]`;
    case "conv2d-stride2": {
      const source = assignment.inputs[0]!;
      return `${lhs} = ${cast}(REDUCE(input_channel=0..channels-1,kernel_time=0..2,kernel_feature=0..2, F32(((2*frame+kernel_time-1<0 || 2*frame+kernel_time-1>=${source}.shape[2] || 2*feature+kernel_feature-1<0 || 2*feature+kernel_feature-1>=${source}.shape[3]) ? F32(0) : ${source}[batch,input_channel,2*frame+kernel_time-1,2*feature+kernel_feature-1])*decode(convolution-kernel)[channel,input_channel,kernel_time,kernel_feature])))`;
    }
    case "layer-norm-channels": {
      const source = assignment.inputs[0]!;
      const current = `${source}[batch,channel,frame,feature]`;
      return domain.domain.dtypePolicy?.reduction?.kind === "pytorch-cpu-bf16-welford"
      ? `${lhs} = ${cast}(F32(F32(F32(${current}*inv_std)+bias)*decode(normalization-scale)[channel])); STRUCT(mean,variance)=REDUCE(channel=0..channels-1,${source}[batch,channel,frame,feature]); inv_std=F32(1/ARM_SQRT_F32(F32(variance+F32(${program.audioProgram.rmsNormEpsilon})))); bias=F32(-inv_std*mean)`
      : `${lhs} = ${cast}(F32(F32(F32(${current}-mean)*inv_std)*decode(normalization-scale)[channel])); mean_sum=REDUCE(channel=0..channels-1,F32(${source}[batch,channel,frame,feature])); mean=F32(mean_sum/F32(channels)); ` +
        `variance_sum=REDUCE(channel=0..channels-1,F32(F32(${source}[batch,channel,frame,feature]-mean)*F32(${source}[batch,channel,frame,feature]-mean))); variance=F32(variance_sum/F32(channels)); ` +
        `inv_std=PYTORCH_POW_NEGATIVE_HALF_F32(F32(variance+F32(${program.audioProgram.rmsNormEpsilon})))`;
    }
    case "relu": return `${lhs} = ${cast}(F32(max(0,${input()})))`;
    case "relative-position-encoding": return `${lhs} = BF16(hidden<${program.audioProgram.tower.hiddenSize / 2} ? SLEEF_SIN_F32(BF16(F32((${Math.floor((program.audioProgram.tower.attentionChunkSize + program.audioProgram.tower.attentionContextLeft - 1 + program.audioProgram.tower.attentionContextRight) / 2)}-relative_position)*BF16(SLEEF_EXP_F32(F32(-(hidden%${program.audioProgram.tower.hiddenSize / 2})*F32(${Math.log(10000) / (program.audioProgram.tower.hiddenSize / 2 - 1)})))))))) : SLEEF_COS_F32(BF16(F32((${Math.floor((program.audioProgram.tower.attentionChunkSize + program.audioProgram.tower.attentionContextLeft - 1 + program.audioProgram.tower.attentionContextRight) / 2)}-relative_position)*BF16(SLEEF_EXP_F32(F32(-(hidden%${program.audioProgram.tower.hiddenSize / 2})*F32(${Math.log(10000) / (program.audioProgram.tower.hiddenSize / 2 - 1)})))))))))`;
    case "clip": return `${lhs} = ${cast}(F32(min(${program.audioProgram.gradientClipping},max(${-program.audioProgram.gradientClipping},${input()}))))`;
    case "silu": return `${lhs} = ${cast}(F32(${input()} / F32(1+SLEEF_EXP_F32(F32(-${input()})))))`;
    case "per-dim-softplus-scale": return `${lhs} = F32(F32(${input()}*F32(${Math.fround(program.audioProgram.tower.headDim ** -0.5 / Math.log(2))}))*BF16(F32(decode(per-dimension-scale)[feature%${program.audioProgram.tower.headDim}]>F32(20) ? decode(per-dimension-scale)[feature%${program.audioProgram.tower.headDim}] : SLEEF_LOG1P_F32(SLEEF_EXP_F32(decode(per-dimension-scale)[feature%${program.audioProgram.tower.headDim}])))))`;
    case "split-gated-linear-unit": return `${lhs} = ${cast}(F32(${assignment.inputs[0]}[batch,frame,hidden]/F32(1+SLEEF_EXP_F32(F32(-${assignment.inputs[0]}[batch,frame,hidden+${program.audioProgram.tower.hiddenSize}])))))`;
    case "causal-depthwise-convolution": {
      const source = assignment.inputs[0]!;
      const kernelSize = assignment.tensors?.[0]?.shape.at(-1);
      if (!Number.isSafeInteger(kernelSize) || kernelSize! <= 0) throw new Error(`${assignment.id}: kernel causal inválido.`);
      return `${lhs} = ${cast}(REDUCE(kernel_index=0..${kernelSize! - 1}, F32((frame-${kernelSize! - 1}+kernel_index<0 ? F32(0) : ${source}[batch,frame-${kernelSize! - 1}+kernel_index,hidden])*decode(convolution-kernel)[hidden,0,kernel_index])))`;
    }
    case "chunked-attention-content-matmul": {
      const query = assignment.inputs[0]!, key = assignment.inputs[1]!, tower = program.audioProgram.tower;
      return `${lhs} = query_index<sequence_length && key_index>=0 && key_index<sequence_length ? F32(REDUCE(head_feature=0..${tower.headDim - 1},F32(${query}[batch,query_index,head*${tower.headDim}+head_feature]*${key}[batch,key_index,head*${tower.headDim}+head_feature]))) : F32(0); ` +
        `query_index=block*${tower.attentionChunkSize}+query_in_block; key_index=block*${tower.attentionChunkSize}-${tower.attentionContextLeft - 1}+key_slot; sequence_length=${query}.shape[1]`;
    }
    case "relative-attention-position-matmul": {
      const query = assignment.inputs[0]!, relative = assignment.inputs[1]!, tower = program.audioProgram.tower, chunk = tower.attentionChunkSize;
      return `${lhs} = query_index<sequence_length ? F32(REDUCE(head_feature=0..${tower.headDim - 1},F32(${query}[batch,query_index,head*${tower.headDim}+head_feature]*${relative}[0,relative_position,head*${tower.headDim}+head_feature]))) : F32(0); ` +
        `query_index=block*${chunk}+query_in_block; sequence_length=${query}.shape[1]`;
    }
    case "relative-attention-shift": {
      const source = assignment.inputs[0]!, context = audioAttentionContext(program), relativeLength = Math.floor(context / 2) + 1;
      return `${lhs} = source_coordinate.valid ? ${source}[batch,head,block,source_coordinate.query_in_block,source_coordinate.relative_index] : F32(0); ` +
        `source_coordinate=AUDIO_RELATIVE_SHIFT_SOURCE(query_in_block,key_slot,${context},${relativeLength})`;
    }
    case "attention-softcap": return `${lhs} = F32(${program.audioProgram.tower.attentionLogitCap}*SLEEF_TANH_F32(F32(${input()}/${program.audioProgram.tower.attentionLogitCap})))`;
    case "chunked-attention-mask": {
      const scores = assignment.inputs[0]!, mask = assignment.inputs[1]!, tower = program.audioProgram.tower;
      return `${lhs} = additive_mask_is_zero || blocked_padding_is_zero ? F32(${program.audioProgram.invalidAttentionLogit}) : ${scores}[batch,head,block,query_in_block,key_slot]; ` +
        `query_index=block*${tower.attentionChunkSize}+query_in_block; key_index=block*${tower.attentionChunkSize}-${tower.attentionContextLeft - 1}+key_slot; sequence_length=${mask}.shape[1]; ` +
        `source_entry_exists=query_index<sequence_length && key_index>=0 && key_index<sequence_length; additive_mask_is_zero=source_entry_exists && ${mask}[batch,key_index] && query_index>=key_index && query_index-key_index<${tower.attentionContextLeft}; blocked_padding_is_zero=!source_entry_exists`;
    }
    case "chunked-relative-attention-softmax": {
      const score = assignment.inputs[0]!;
      return `${lhs} = F32(exponential[key_slot]/total); ` +
        `maximum=ORDERED_F32_REDUCE_MAX(${score}[batch,head,block,query_in_block,key_slot],key_slot=0..context-1); ` +
        `exponential[key_slot]=SLEEF_EXP_F32(F32(${score}[batch,head,block,query_in_block,key_slot]-maximum)); ` +
        "total=ORDERED_F32_REDUCE_SUM(exponential[key_slot],key_slot=0..context-1)";
    }
    case "chunked-relative-attention-values": {
      const probability = assignment.inputs[0]!, value = assignment.inputs[1]!, tower = program.audioProgram.tower;
      return `${lhs} = F32(REDUCE(key_slot=0..context-1,key_index>=0 && key_index<sequence_length ? F32(${probability}[batch,head,block,query_in_block,key_slot]*${value}[batch,key_index,hidden]) : F32(0))); ` +
        `head=floor(hidden/${tower.headDim}); head_feature=hidden%${tower.headDim}; block=floor(frame/${tower.attentionChunkSize}); query_in_block=frame%${tower.attentionChunkSize}; ` +
        `query_index=block*${tower.attentionChunkSize}+query_in_block; key_index=block*${tower.attentionChunkSize}-${tower.attentionContextLeft - 1}+key_slot; sequence_length=${value}.shape[1]`;
    }
    case "cast-bf16": return `${lhs} = BF16(${input()})`;
    case "subsample-mask": return `${lhs} = ${assignment.inputs[0]}[batch,2*frame]`;
  }
}

export function requiredGemma4LiteralPlaceholderTokenId(assignment: MultimodalAssignment): number {
  if (!Number.isSafeInteger(assignment.placeholderTokenId) || assignment.placeholderTokenId! < 0) {
    throw new Error(`${assignment.id}: masked scatter requer placeholderTokenId autoritativo.`);
  }
  return assignment.placeholderTokenId!;
}

function textFormula(operation: Operation, lhs: string, domain: Gemma4LiteralAssignmentDomain): string {
  const cast = operation.dtypePolicy?.outputDtype === "BF16" ? "BF16" : "F32";
  const finalCoordinate = domain.domain.axes.at(-1)?.name;
  if (!finalCoordinate) throw new Error(`${operation.id}: cálculo text requer eixo final de saída.`);
  switch (operation.op) {
    case "embedding": return `${lhs} = ${cast}(decode(weight)[${operation.tokenInput}[batch,sequence],hidden]${operation.scale === undefined ? "" : `*F32(${operation.scale})`})`;
    case "per_layer_embedding": return `${lhs} = ${cast}(decode(weight)[${operation.tokenInput}[batch,sequence],layer*${operation.layerWidth}+ple_feature]${operation.scale === undefined ? "" : `*F32(${operation.scale})`})`;
    case "rms_norm": return rmsNormFormula(lhs, operation.input, domain, cast, operation.epsilon, operation.weightTransform);
    case "linear": return linearFormula(lhs, operation.input, inputReductionCoordinates(domain), cast, operation.bias !== undefined, false, operation.dtypePolicy.reduction);
    case "reshape_heads": return `${lhs} = row_major_alias(${operation.input})[batch,sequence,head*${operation.headDim}+head_feature]`;
    case "reshape_per_layer": return `${lhs} = ${operation.input}[batch,sequence,layer*${operation.layerWidth}+${finalCoordinate}]`;
    case "select_per_layer": return `${lhs} = ${operation.input}[batch,sequence,${operation.layerIndex},${finalCoordinate}]`;
    case "rotary_embedding": return textRotaryFormula(operation, lhs, cast);
    case "scaled_dot_product_attention": return textAttentionFormula(operation, lhs, cast);
    case "activation": {
      const source = indexed(operation.input, domain.domain.axes.map((axis) => axis.name));
      if (operation.function === "gelu" && operation.approximation === "tanh") return `${lhs} = ${cast}(F32(F32(0.5*${source})*F32(1+SLEEF_TANH_F32(F32(${Math.sqrt(2 / Math.PI)}*F32(${source}+F32(0.044715*F32(${source}*F32(${source}*${source})))))))))`;
      if (operation.function === "silu" || operation.function === "swish") return `${lhs} = ${cast}(F32(${source}/F32(1+SLEEF_EXP_F32(F32(-${source})))))`;
      return `${lhs} = ${cast}(${operation.function}${operation.approximation ? `[${operation.approximation}]` : ""}(${source}))`;
    }
    case "elementwise": {
      const coordinates = domain.domain.axes.map((axis) => axis.name);
      const inputs = operation.inputs.map((input) => indexed(input, coordinates));
      if (operation.kind === "add") return `${lhs} = ${cast}(F32(${inputs[0]}+${inputs[1]}))`;
      if (operation.kind === "multiply") return `${lhs} = ${cast}(F32(${inputs[0]}*${inputs[1]}))`;
      if (operation.kind === "scale") return `${lhs} = ${cast}(F32(${inputs[0]}*F32(${operation.scalar})))`;
      return `${lhs} = ${cast}(F32(F32(${operation.scalar}*BF16(SLEEF_TANH_F32(BF16(F32(${inputs[0]}/F32(${operation.scalar}))))))))`;
    }
    case "tensor_scale": return `${lhs} = ${cast}(F32(${indexed(operation.input, domain.domain.axes.map((axis) => axis.name))}*decode(tensor-scale)[0]))`;
  }
}

function textRotaryFormula(operation: Extract<Operation, { op: "rotary_embedding" }>, lhs: string, cast: string): string {
  if (operation.layout !== "rotate_half") throw new Error(`${operation.id}: fórmula literal requer RoPE rotate_half.`);
  const half = operation.rotaryDim / 2;
  const partial = operation.ropeType === "proportional" ? numericScaling(operation, "partial_rotary_factor") : 1;
  const factor = operation.ropeType === "proportional" ? optionalNumericScaling(operation, "factor") ?? 1 : 1;
  const exponentDenominator = operation.ropeType === "proportional" ? "head_dim" : String(operation.rotaryDim);
  const trigCast = operation.rotaryCasts ? "BF16" : "F32";
  const direct = `${trigCast}(F32(${operation.input}[batch,head,sequence,head_feature]*cosine))`;
  const rotated = `${trigCast}(F32(${operation.input}[batch,head,sequence,paired_feature]*sine))`;
  const minus = operation.rotaryCasts ? `BF16(F32(${direct}-${rotated}))` : `${cast}(F32(${direct}-${rotated}))`;
  const plus = operation.rotaryCasts ? `BF16(F32(${direct}+${rotated}))` : `${cast}(F32(${direct}+${rotated}))`;
  return `${lhs} = head_feature>=${operation.rotaryDim} ? ${cast}(${operation.input}[batch,head,sequence,head_feature]) : (head_feature<${half} ? ${minus} : ${plus}); ` +
    `pair=head_feature%${half}; paired_feature=head_feature<${half} ? head_feature+${half} : head_feature-${half}; ` +
    `active_pairs=floor(F64(${partial})*F64(head_dim)/F64(2)); angle=pair>=active_pairs ? F32(0) : F32(${operation.positionInput}[batch,sequence]/F32(F32(${operation.theta}**F64(F64(2*pair)/F64(${exponentDenominator})))*F32(${factor}))); ` +
    `cosine=${trigCast}(SLEEF_COS_F32(angle)); sine=${trigCast}(SLEEF_SIN_F32(angle))`;
}

function numericScaling(operation: Extract<Operation, { op: "rotary_embedding" }>, key: string): number {
  const value = operation.scaling?.[key];
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${operation.id}: ${key} inválido na fórmula RoPE.`);
  return value;
}

function optionalNumericScaling(operation: Extract<Operation, { op: "rotary_embedding" }>, key: string): number | undefined {
  const value = operation.scaling?.[key];
  if (value === undefined) return undefined;
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${operation.id}: ${key} inválido na fórmula RoPE.`);
  return value;
}

function textAttentionFormula(
  operation: Extract<Operation, { op: "scaled_dot_product_attention" }>,
  lhs: string,
  cast: string,
): string {
  const queryHead = `floor(attention_hidden/${operation.headDim})`;
  const outputFeature = `attention_hidden%${operation.headDim}`;
  const kvGroup = operation.numAttentionHeads / operation.numKeyValueHeads;
  if (!Number.isSafeInteger(kvGroup) || kvGroup <= 0) throw new Error(`${operation.id}: GQA inválida para fórmula escalar.`);
  const kvHead = `floor(${queryHead}/${kvGroup})`;
  const q = `${operation.query}[batch,${queryHead},sequence,head_feature]`;
  const k = `${operation.key}[batch,${kvHead},key,head_feature]`;
  const value = `${operation.value}[batch,${kvHead},key,${outputFeature}]`;
  const mask = `${operation.maskInput}[batch,0,sequence,key]`;
  if (operation.numericImplementation) {
    return `${lhs} = BF16(ARM_NEON_BF16_DOT_F32(reductionStages[context-dot].schedule,probability[key],${value},key=0..K-1)); ` +
      `score[key]=BF16(F32(BF16(F32(ARM_NEON_BF16_DOT_F32(reductionStages[score-dot].schedule,${q},${k},head_feature=0..${operation.headDim - 1})*F32(${operation.scale})))+${mask})); ` +
      "maximum=PYTORCH_F32_VECTOR_REDUCE_MAX(score[key],key=0..K-1,lanes=4); " +
      "exponential[key]=SLEEF_EXP_F32(F32(score[key]-maximum)); " +
      "total=PYTORCH_F32_VECTOR_REDUCE_SUM(exponential[key],key=0..K-1,lanes=4); " +
      "probability[key]=BF16(F32(exponential[key]*F32(1/total)))";
  }
  const score = operation.scoreSoftcap === undefined
    ? "scaled_dot[key]"
    : `F32(F32(${operation.scoreSoftcap})*SLEEF_TANH_F32(F32(scaled_dot[key]/F32(${operation.scoreSoftcap}))))`;
  return `${lhs} = ${cast}(ORDERED_F32_DOT(probability[key],${value},key=0..K-1)); ` +
    `dot[key]=ORDERED_F32_DOT(${q},${k},head_feature=0..${operation.headDim - 1}); ` +
    `scaled_dot[key]=F32(dot[key]*F32(${operation.scale})); score[key]=F32(${score}+${mask}); ` +
    "maximum=ORDERED_F32_REDUCE_MAX(score[key],key=0..K-1); " +
    "exponential[key]=SLEEF_EXP_F32(F32(score[key]-maximum)); " +
    "total=ORDERED_F32_REDUCE_SUM(exponential[key],key=0..K-1); probability[key]=F32(exponential[key]/total)";
}

function scalarReductionStages(definition: Definition): Gemma4LiteralScalarReductionStage[] {
  if ("op" in definition.value && definition.value.op === "scaled_dot_product_attention") {
    const operation = definition.value;
    const numeric = operation.numericImplementation;
    return [
      {
        id: "score-dot", indices: [`head_feature=0..${operation.headDim - 1}`],
        domains: [reductionDomain("head_feature", tensorAxisGemma4LiteralReductionExtent(operation.query, 3))],
        order: numeric ? "operation-declared" : "ascending-lexicographic",
        program: numeric ? "ARM_NEON_BF16_DOT_F32" : "ORDERED_F32_DOT", identity: "F32(0)",
        ...(numeric ? { schedule: structuredClone(numeric.scoreReduction) } : {}),
      },
      {
        id: "softmax-maximum", indices: ["key=0..K-1"],
        domains: [reductionDomain("key", tensorAxisGemma4LiteralReductionExtent(operation.maskInput, 3))],
        order: numeric ? "operation-declared" : "ascending-lexicographic",
        program: numeric ? "PYTORCH_F32_VECTOR_REDUCE_MAX" : "ORDERED_F32_REDUCE_MAX", identity: "-Infinity",
      },
      {
        id: "softmax-exponential-sum", indices: ["key=0..K-1"],
        domains: [reductionDomain("key", tensorAxisGemma4LiteralReductionExtent(operation.maskInput, 3))],
        order: numeric ? "operation-declared" : "ascending-lexicographic",
        program: numeric ? "PYTORCH_F32_VECTOR_REDUCE_SUM" : "ORDERED_F32_REDUCE_SUM", identity: "F32(0)",
      },
      {
        id: "context-dot", indices: ["key=0..K-1"],
        domains: [reductionDomain("key", tensorAxisGemma4LiteralReductionExtent(operation.maskInput, 3))],
        order: numeric ? "operation-declared" : "ascending-lexicographic",
        program: numeric ? "ARM_NEON_BF16_DOT_F32" : "ORDERED_F32_DOT", identity: "F32(0)",
        ...(numeric ? { schedule: structuredClone(numeric.contextReduction) } : {}),
      },
    ];
  }
  if (!("op" in definition.value) && definition.value.operation === "masked-softmax") {
    const predicate = "pixel_position_ids[batch,key_patch] != [-1,-1]";
    return [
      { id: "softmax-maximum", indices: ["key_patch=0..patches-1"], domains: [reductionDomain("key_patch", tensorAxisGemma4LiteralReductionExtent(definition.inputs[0]!, 3))], order: "ascending-lexicographic", program: "ORDERED_F32_REDUCE_MAX", identity: "-Infinity", predicate },
      { id: "softmax-exponential-sum", indices: ["key_patch=0..patches-1"], domains: [reductionDomain("key_patch", tensorAxisGemma4LiteralReductionExtent(definition.inputs[0]!, 3))], order: "ascending-lexicographic", program: "ORDERED_F32_REDUCE_SUM", identity: "F32(0)", predicate },
    ];
  }
  if (!("op" in definition.value) && definition.value.operation === "chunked-relative-attention-softmax") {
    return [
      { id: "softmax-maximum", indices: ["key_slot=0..context-1"], domains: [reductionDomain("key_slot", tensorAxisGemma4LiteralReductionExtent(definition.inputs[0]!, 4))], order: "ascending-lexicographic", program: "ORDERED_F32_REDUCE_MAX", identity: "-Infinity" },
      { id: "softmax-exponential-sum", indices: ["key_slot=0..context-1"], domains: [reductionDomain("key_slot", tensorAxisGemma4LiteralReductionExtent(definition.inputs[0]!, 4))], order: "ascending-lexicographic", program: "ORDERED_F32_REDUCE_SUM", identity: "F32(0)" },
    ];
  }
  return [];
}

function scalarReduction(
  definition: Definition,
  program: Gemma4CompositeProgram,
  policy: DtypePolicy,
  domain: Gemma4LiteralAssignmentDomain,
): Gemma4LiteralScalarReduction | undefined {
  const indices = reductionIndices(definition, program);
  const domains = reductionDomains(definition, program, domain);
  if (indices.length === 0 && domains.length === 0) return undefined;
  validateReductionDomains(indices, domains, definition.inputs, `${definition.scope}:${definition.id}`);
  if (policy.accumulationDtype === "runtime-defined") return { indices, domains, order: "runtime-defined" };
  if (policy.reduction) return { indices, domains, order: "operation-declared", schedule: structuredClone(policy.reduction) };
  return { indices, domains, order: "ascending-lexicographic" };
}

function reductionDomains(
  definition: Definition,
  program: Gemma4CompositeProgram,
  domain: Gemma4LiteralAssignmentDomain,
): Gemma4LiteralReductionIndexDomain[] {
  const input = definition.inputs[0]!;
  switch (definition.operation) {
    case "linear": case "clipped-linear": case "rms_norm": case "rms-norm":
      return [reductionDomain(definition.operation.startsWith("rms") ? "reduction_feature" : "input_feature", tensorAxisGemma4LiteralReductionExtent(input, domain.domain.shape.length - 1))];
    case "conv2d-stride2": return [
      reductionDomain("input_channel", tensorAxisGemma4LiteralReductionExtent(input, 1)),
      reductionDomain("kernel_time", constantGemma4LiteralReductionExtent(3)),
      reductionDomain("kernel_feature", constantGemma4LiteralReductionExtent(3)),
    ];
    case "layer-norm-channels": return [reductionDomain("channel", tensorAxisGemma4LiteralReductionExtent(input, 1))];
    case "causal-depthwise-convolution": {
      const kernel = "tensors" in definition.value ? definition.value.tensors?.[0]?.shape.at(-1) : undefined;
      if (!Number.isSafeInteger(kernel) || kernel! <= 0) throw new Error(`${definition.id}: kernel causal sem extent executável.`);
      return [reductionDomain("kernel_index", constantGemma4LiteralReductionExtent(kernel!))];
    }
    case "attention-score-matmul": return [reductionDomain("head_feature", tensorAxisGemma4LiteralReductionExtent(input, 3))];
    case "chunked-attention-content-matmul": case "relative-attention-position-matmul":
      return [reductionDomain("head_feature", constantGemma4LiteralReductionExtent(program.audioProgram.tower.headDim))];
    case "masked-softmax": case "attention-value-matmul":
      return [reductionDomain("key_patch", tensorAxisGemma4LiteralReductionExtent(input, 3))];
    case "chunked-relative-attention-softmax": case "chunked-relative-attention-values":
      return [reductionDomain("key_slot", tensorAxisGemma4LiteralReductionExtent(input, 4))];
    case "pool-by-position": return [reductionDomain("patch", tensorAxisGemma4LiteralReductionExtent(input, 1))];
    case "scaled_dot_product_attention": {
      const operation = definition.value as Extract<Operation, { op: "scaled_dot_product_attention" }>;
      return [
        reductionDomain("head_feature", tensorAxisGemma4LiteralReductionExtent(operation.query, 3)),
        reductionDomain("key", tensorAxisGemma4LiteralReductionExtent(operation.maskInput, 3)),
      ];
    }
    default: return [];
  }
}

function reductionDomain(index: string, endExclusive: Gemma4LiteralReductionExtent): Gemma4LiteralReductionIndexDomain {
  return { index, startInclusive: 0, endExclusive, order: "ascending" };
}

function validateReductionDomains(
  indices: readonly string[],
  domains: readonly Gemma4LiteralReductionIndexDomain[],
  orderedInputs: readonly string[],
  owner: string,
): void {
  validateGemma4LiteralReductionIndexDomains(domains, owner);
  if (indices.length !== domains.length || domains.some((domain, position) => !indices[position]?.startsWith(`${domain.index}=0..`))) {
    throw new Error(`${owner}: índices humanos e domínios executáveis de redução divergem.`);
  }
  const unbound = domains.find((domain) => domain.endExclusive.kind === "tensor-axis" && !orderedInputs.includes(domain.endExclusive.tensor));
  if (unbound?.endExclusive.kind === "tensor-axis") {
    throw new Error(`${owner}: domínio de redução referencia tensor fora de orderedInputs: ${unbound.endExclusive.tensor}.`);
  }
}

function reductionIndices(definition: Definition, program: Gemma4CompositeProgram): string[] {
  switch (definition.operation) {
    case "linear": case "clipped-linear": return ["input_feature=0..in_features-1"];
    case "rms_norm": case "rms-norm": return ["reduction_feature=0..width-1"];
    case "conv2d-stride2": return ["input_channel=0..channels-1", "kernel_time=0..2", "kernel_feature=0..2"];
    case "layer-norm-channels": return ["channel=0..channels-1"];
    case "causal-depthwise-convolution": return ["kernel_index=0..kernel_size-1"];
    case "attention-score-matmul": case "chunked-attention-content-matmul": case "relative-attention-position-matmul": return [`head_feature=0..${definition.scope === "vision" ? program.visionProgram.tower.headDim : program.audioProgram.tower.headDim}-1`];
    case "masked-softmax": return ["key_patch=0..patches-1"];
    case "attention-value-matmul": return ["key_patch=0..patches-1"];
    case "chunked-relative-attention-softmax": case "chunked-relative-attention-values": return ["key_slot=0..context-1"];
    case "pool-by-position": return ["patch=0..patches-1"];
    case "scaled_dot_product_attention": return ["head_feature=0..head_dim-1", "key=0..K-1"];
    default: return [];
  }
}

function requiredDomain(domains: readonly Gemma4LiteralAssignmentDomain[], definition: Definition): Gemma4LiteralAssignmentDomain {
  const domain = domains.find((candidate) => candidate.scope === definition.scope && candidate.definitionId === definition.id);
  if (!domain) throw new Error(`${definition.scope}:${definition.id}: domínio ausente para cálculo escalar.`);
  return domain;
}

function inputReductionCoordinates(domain: Gemma4LiteralAssignmentDomain, reductionIndex = "input_feature"): string[] {
  const coordinates = domain.domain.axes.map((axis) => axis.name);
  if (coordinates.length === 0) throw new Error(`${domain.scope}:${domain.definitionId}: redução requer domínio tensorial não vazio.`);
  return [...coordinates.slice(0, -1), reductionIndex];
}

function rmsNormFormula(
  lhs: string,
  input: string,
  domain: Gemma4LiteralAssignmentDomain,
  cast: string,
  epsilon: number,
  weightTransform: "none" | "direct" | "one_plus_weight",
): string {
  const coordinates = domain.domain.axes.map((axis) => axis.name);
  const feature = coordinates.at(-1);
  const width = domain.domain.axes.at(-1)?.size;
  if (!feature || !width) throw new Error(`${domain.scope}:${domain.definitionId}: RMSNorm requer eixo final explícito.`);
  const current = indexed(input, coordinates);
  const reduced = indexed(input, [...coordinates.slice(0, -1), "reduction_feature"]);
  const normalized = `F32(${current}*PYTORCH_POW_NEGATIVE_HALF_F32(F32(REDUCE(reduction_feature=0..${width}-1,F32(${reduced}*${reduced}))/F32(${width})+F32(${epsilon}))))`;
  const weighted = weightTransform === "direct"
    ? `F32(${normalized}*decode(normalization-scale)[${feature}])`
    : weightTransform === "one_plus_weight"
      ? `F32(${normalized}*F32(1+decode(normalization-scale)[${feature}]))`
      : normalized;
  return `${lhs} = ${cast}(${weighted})`;
}

function visionRotaryFormula(
  assignment: MultimodalAssignment,
  program: Gemma4CompositeProgram,
  lhs: string,
): string {
  const headDim = program.visionProgram.tower.headDim;
  if (!Number.isSafeInteger(headDim) || headDim <= 0 || headDim % 4 !== 0) {
    throw new Error(`${assignment.id}: RoPE vision requer head_dim positivo divisível por quatro.`);
  }
  const axisWidth = headDim / 2;
  const pairWidth = axisWidth / 2;
  const source = assignment.inputs[0]!;
  const positions = assignment.inputs[1]!;
  const theta = program.visionProgram.tower.ropeTheta;
  const direct = `BF16(F32(${source}[batch,head,patch,head_feature]*cosine))`;
  const rotated = `BF16(F32(${source}[batch,head,patch,paired_feature]*sine))`;
  return `${lhs} = local_feature<${pairWidth} ? BF16(F32(${direct}-${rotated})) : BF16(F32(${direct}+${rotated})); ` +
    `axis=floor(head_feature/${axisWidth}); local_feature=head_feature%${axisWidth}; pair=local_feature%${pairWidth}; ` +
    `paired_feature=axis*${axisWidth}+(local_feature<${pairWidth} ? local_feature+${pairWidth} : local_feature-${pairWidth}); ` +
    `angle=F32(${positions}[batch,patch,axis]/F32(${theta}**F32(F32(2*pair)/F32(${axisWidth})))); ` +
    "cosine=BF16(SLEEF_COS_F32(angle)); sine=BF16(SLEEF_SIN_F32(angle))";
}

function linearFormula(
  lhs: string,
  input: string,
  inputCoordinates: readonly string[],
  cast: string,
  bias: boolean,
  clipped: boolean,
  reduction: ReductionSchedule | undefined,
): string {
  const indexedInput = indexed(input, inputCoordinates);
  const x = clipped ? `F32(min(decode(input-max),max(decode(input-min),${indexedInput})))` : indexedInput;
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

function audioAttentionContext(program: Gemma4CompositeProgram): number {
  const tower = program.audioProgram.tower;
  return tower.attentionChunkSize + tower.attentionContextLeft - 1 + tower.attentionContextRight;
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
