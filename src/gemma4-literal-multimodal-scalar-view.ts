import type { Gemma4AudioAssignment } from "./gemma4-audio.js";
import type { Gemma4CompositeAssignment } from "./gemma4-composite.js";
import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import {
  listGemma4LiteralTextOperations,
  readGemma4LiteralLearnedOperandScalar,
  renderGemma4LiteralScalarView,
  validateGemma4LiteralScalarView,
  type Gemma4LiteralLearnedScalar,
  type Gemma4LiteralOperationNavigation,
  type Gemma4LiteralScalarTerm,
  type Gemma4LiteralRenderedScalarView,
  type Gemma4LiteralScalarView,
  type Gemma4LiteralScalarViewBase,
  type Gemma4LiteralScalarViewRequest,
} from "./gemma4-literal-scalar-view.js";
import type { Gemma4VisionAssignment } from "./gemma4-vision.js";
import type { DtypePolicy } from "./types.js";
import {
  gemma4LiteralOutputCoordinateEnvironment,
  optionalGemma4LiteralLearnedOperand,
  requiredGemma4LiteralLearnedOperand,
  type Gemma4LiteralLearnedIndexEnvironment,
  type Gemma4LiteralLearnedOperand,
  type Gemma4LiteralLearnedOperandRole,
} from "./gemma4-literal-learned-operands.js";
import {
  requiredGemma4LiteralScalarCalculation,
  type Gemma4LiteralScalarCalculation,
} from "./gemma4-literal-scalar-calculations.js";
import {
  buildGemma4LiteralLinearReductionAssignments,
  gemma4LiteralScalarProductFormula,
} from "./gemma4-literal-linear-reduction-view.js";
import {
  buildGemma4LiteralCascadeSquareReductionAssignments,
  buildGemma4LiteralWelfordAssignments,
} from "./gemma4-literal-normalization-reduction-view.js";

type Assignment = Gemma4CompositeAssignment | Gemma4VisionAssignment | Gemma4AudioAssignment;
type NonTextScope = "composite" | "vision" | "audio";

interface AssignmentEntry {
  assignment: Assignment;
  scope: NonTextScope;
  operationId: string;
  invocationId?: string;
  bindings: ReadonlyMap<string, string>;
}

export interface Gemma4LiteralMultimodalScalarViewRequest extends Gemma4LiteralScalarViewRequest {
  /** Required when a learned 2-D position table is indexed by caller positions. */
  positionCoordinate?: [number, number];
}

const F32_POLICY: DtypePolicy = { inputDtype: "F32", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" };
const F32_REDUCTION_POLICY: DtypePolicy = {
  ...F32_POLICY,
  reduction: { kind: "ordered-scalar", indexOrder: "ascending" },
};

/**
 * Expands the composite program into its real dependency order. Shared vision
 * definitions are instantiated separately for image and video, audio is
 * inlined at its call site, and prepared text starts at layer 0 rather than
 * incorrectly rerunning the standalone embedding/PLE prelude.
 */
export function listGemma4LiteralOperations(artifact: OpenGemma4CompositeLiteralArtifact): Gemma4LiteralOperationNavigation[] {
  const navigation = artifact.calculationGraph.assignments.map((assignment): Gemma4LiteralOperationNavigation => ({
    operationId: assignment.operationId,
    definitionId: assignment.definitionId,
    operation: assignment.operation,
    scope: assignment.scope,
    ...(assignment.layer === undefined ? {} : { layer: assignment.layer }),
    ...(assignment.invocationId ? { invocationId: assignment.invocationId } : {}),
    ordinal: assignment.ordinal,
    output: assignment.output,
    outputDomain: structuredClone(assignment.outputDomain),
    scalarCalculation: structuredClone(assignment.scalarCalculation),
    ...(assignment.learnedOperands ? { learnedOperands: structuredClone(assignment.learnedOperands) } : {}),
    predecessors: structuredClone(assignment.predecessors),
    consumers: [...assignment.consumers],
  }));
  return navigation.map((entry, ordinal) => {
    return {
    ...entry,
    ...(ordinal === 0 ? {} : { previousOperationId: navigation[ordinal - 1]!.operationId }),
    ...(ordinal + 1 === navigation.length ? {} : { nextOperationId: navigation[ordinal + 1]!.operationId }),
  }; });
}

/**
 * Renders an indexed assignment from any Gemma 4 scope. Assignment families
 * which consume checkpoint tensors must have a registered literal renderer;
 * no generic formula may hide a learned value.
 */
export async function renderGemma4LiteralMultimodalScalarView(
  artifact: OpenGemma4CompositeLiteralArtifact,
  request: Gemma4LiteralMultimodalScalarViewRequest,
): Promise<Gemma4LiteralScalarView> {
  assertCoordinate(request.outputCoordinate);
  const navigation = listGemma4LiteralOperations(artifact).find((candidate) => candidate.operationId === request.operationId);
  if (!navigation) throw new Error(`Atribuição Gemma 4 literal não encontrada: ${request.operationId}.`);
  if (navigation.scope === "text-layer" || navigation.scope === "text-epilogue") {
    const rendered = await renderGemma4LiteralScalarView(artifact, { ...request, operationId: navigation.definitionId ?? navigation.operationId });
    const result = bindScalarView(rendered, navigation, calculationBindings(rendered.navigation.scalarCalculation, navigation.scalarCalculation));
    validateGemma4LiteralScalarView(result);
    return result;
  }
  const entry = assignmentEntries(artifact).find((candidate) => candidate.operationId === request.operationId);
  if (!entry) throw new Error(`Atribuição Gemma 4 literal não possui definição instanciada: ${request.operationId}.`);
  const definitionId = entry.scope === "composite" ? compatibleTextPreludeDefinitionId(artifact, entry.assignment) : undefined;
  if (definitionId) {
    const rendered = await renderGemma4LiteralScalarView(artifact, { ...request, operationId: definitionId });
    const result = bindScalarView(rendered, navigation, new Map([[rendered.navigation.output, entry.assignment.output]]));
    validateGemma4LiteralScalarView(result);
    return result;
  }
  const base = {
    kind: "gemma4-literal-scalar-view" as const,
    sourceCheckpointAccessed: false as const,
    navigation: { ...navigation, output: entry.assignment.output },
    outputCoordinate: [...request.outputCoordinate],
    output: indexed(entry.assignment.output, request.outputCoordinate),
    dtypePolicy: policyFor(entry.assignment),
  };
  const operation = entry.assignment.operation;
  let rendered: Gemma4LiteralRenderedScalarView | undefined;
  if (operation === "linear" || operation === "clipped-linear") rendered = await renderLinear(artifact, entry, request, base);
  else if (operation === "embedding" || operation === "per-layer-embedding") rendered = await renderEmbedding(artifact, entry, request, base);
  else if (operation === "rms-norm") rendered = await renderRmsNorm(artifact, entry, request, base);
  else if (operation === "position-embedding-2d") rendered = await renderPositionEmbedding(artifact, entry, request, base);
  else if (operation === "conv2d-stride2") rendered = await renderConv2d(artifact, entry, request, base);
  else if (operation === "layer-norm-channels") rendered = await renderChannelNorm(artifact, entry, request, base);
  else if (operation === "causal-depthwise-convolution") rendered = await renderDepthwise(artifact, entry, request, base);
  else if (operation === "per-dim-softplus-scale") rendered = await renderPerDimScale(artifact, entry, request, base);
  if (!rendered && (entry.assignment.tensors?.length ?? 0) !== 0) {
    throw new Error(`${entry.assignment.id}: operação ${operation} usa tensores aprendidos sem vista escalar registrada.`);
  }
  if (!rendered && base.dtypePolicy.accumulationDtype === "runtime-defined") {
    throw new Error(`${entry.assignment.id}: vista escalar falha fechada porque a redução ${base.dtypePolicy.computeDtype ?? "nativa"} ainda não possui agenda literal comprovada.`);
  }
  if (!rendered) {
    const formulas = plainFormulas(artifact, entry, request.outputCoordinate);
    rendered = { ...base, formula: formulas.at(-1)!, scalarAssignments: formulas, learnedScalars: [] };
  }
  const result = bindScalarView(attachDenseDecoderEvidence(artifact, rendered, entry.assignment.id), navigation, entry.bindings);
  validateGemma4LiteralScalarView(result);
  return result;
}

function attachDenseDecoderEvidence(
  artifact: OpenGemma4CompositeLiteralArtifact,
  view: Gemma4LiteralRenderedScalarView,
  operationId: string,
): Gemma4LiteralScalarView {
  const decoderIds = new Set(view.learnedScalars.map((scalar) => scalar.decoderId));
  const storageDecoders = artifact.storageDecoders.filter((decoder) => decoderIds.has(decoder.id)).map((decoder) => structuredClone(decoder));
  if (storageDecoders.length !== decoderIds.size) throw new Error(`${operationId}: programa de decoder ausente para valor substituído.`);
  return {
    ...view,
    formulaLanguage: structuredClone(artifact.formulaLanguage),
    transcendentalPrograms: structuredClone(artifact.transcendentalPrograms),
    denseDecoderLanguage: structuredClone(artifact.denseDecoderLanguage),
    storageDecoders,
  };
}

function assignmentEntries(artifact: OpenGemma4CompositeLiteralArtifact): AssignmentEntry[] {
  return artifact.calculationGraph.assignments.flatMap((calculation): AssignmentEntry[] => {
    if (calculation.scope === "text-layer" || calculation.scope === "text-epilogue") return [];
    const definitions = calculation.scope === "composite" ? artifact.program.assignments
      : calculation.scope === "vision" ? artifact.program.visionProgram.assignments
        : artifact.program.audioProgram.assignments;
    const assignment = definitions.find((candidate) => candidate.id === calculation.definitionId);
    if (!assignment) throw new Error(`${calculation.operationId}: definição ${calculation.definitionId} ausente no programa literal.`);
    const definitionCalculation = requiredGemma4LiteralScalarCalculation(artifact.scalarCalculations, calculation.scope, calculation.definitionId);
    return [{
      assignment,
      scope: calculation.scope,
      operationId: calculation.operationId,
      ...(calculation.invocationId ? { invocationId: calculation.invocationId } : {}),
      bindings: calculationBindings(definitionCalculation, calculation.scalarCalculation),
    }];
  });
}

function compatibleTextPreludeDefinitionId(
  artifact: OpenGemma4CompositeLiteralArtifact,
  assignment: Assignment,
): string | undefined {
  const expected = assignment.operation === "embedding" ? "embedding"
    : assignment.operation === "per-layer-embedding" ? "per_layer_embedding"
      : assignment.operation === "linear" ? "linear"
        : assignment.operation === "scale-f32" || assignment.operation === "add" ? "elementwise"
          : assignment.operation === "reshape-per-layer" ? "reshape_per_layer"
            : assignment.operation === "rms-norm" ? "rms_norm"
              : undefined;
  if (!expected) return undefined;
  const candidates = artifact.program.textProgram.prelude.filter((operation) =>
    operation.op === expected && (operation.output === assignment.output ||
      assignment.operation === "embedding" && operation.id === "token_embedding"));
  if (candidates.length !== 1) throw new Error(`${assignment.id}: operação composite compatível não possui definição text-prelude única.`);
  return candidates[0]!.id;
}

function calculationBindings(
  definition: Gemma4LiteralScalarCalculation,
  instantiated: Gemma4LiteralScalarCalculation,
): ReadonlyMap<string, string> {
  if (definition.orderedInputs.length !== instantiated.orderedInputs.length) {
    throw new Error(`${instantiated.definitionId}: cálculo instanciado possui aridade divergente.`);
  }
  const bindings = new Map(definition.orderedInputs.map((input, index) => [input, instantiated.orderedInputs[index]!]));
  bindings.set(definition.output, instantiated.output);
  return bindings;
}

function bindScalarView(
  view: Gemma4LiteralScalarView,
  navigation: Gemma4LiteralOperationNavigation,
  bindings: ReadonlyMap<string, string>,
): Gemma4LiteralScalarView {
  const orderedBindings = [...bindings.entries()].sort(([left], [right]) => right.length - left.length);
  // Replace through sentinels so overlapping names (for example `*_bd` and
  // `*_bd_unshifted`) cannot recursively rewrite a target produced earlier in
  // the same binding pass.
  const placeholders = orderedBindings.map(([source, target], index) => ({ source, target, token: `\u0000binding:${index}\u0000` }));
  const replace = (value: string): string => {
    const tokenized = placeholders.reduce((current, binding) => current.split(binding.source).join(binding.token), value);
    return placeholders.reduce((current, binding) => current.split(binding.token).join(binding.target), tokenized);
  };
  return {
    ...view,
    navigation,
    output: replace(view.output),
    formula: replace(view.formula),
    scalarAssignments: view.scalarAssignments.map(replace),
    learnedScalars: view.learnedScalars.map((scalar) => ({ ...scalar })),
    ...(view.terms ? { terms: view.terms.map((term) => ({ ...term, input: replace(term.input), formula: replace(term.formula) })) } : {}),
  };
}

async function renderLinear(
  artifact: OpenGemma4CompositeLiteralArtifact,
  entry: AssignmentEntry,
  request: Gemma4LiteralMultimodalScalarViewRequest,
  base: ScalarBase,
): Promise<Gemma4LiteralRenderedScalarView> {
  const assignment = entry.assignment;
  if (base.dtypePolicy.accumulationDtype === "runtime-defined") {
    throw new Error(`${assignment.id}: vista escalar falha fechada porque o kernel nativo ${base.dtypePolicy.computeDtype ?? "desconhecido"} ainda não possui agenda de redução literal.`);
  }
  const weight = learnedOperand(artifact, entry, "weight");
  const outputFeature = last(request.outputCoordinate, assignment.id);
  const [outFeatures, inFeatures] = weight.tensor.shape;
  if (outputFeature >= outFeatures!) throw new Error(`${assignment.id}: feature ${outputFeature} excede outFeatures=${outFeatures}.`);
  const prefix = request.outputCoordinate.slice(0, -1);
  const window = reductionWindow(request, inFeatures!, assignment.id);
  const schedule = base.dtypePolicy.reduction ?? { kind: "ordered-scalar", indexOrder: "ascending" } as const;
  if (schedule.kind === "pytorch-cpu-f32-cascade-sum" || schedule.kind === "pytorch-cpu-bf16-welford") {
    throw new Error(`${assignment.id}: agenda ${schedule.kind} não é uma redução linear.`);
  }
  const learnedScalars: Gemma4LiteralLearnedScalar[] = [];
  const terms: Gemma4LiteralScalarTerm[] = [];
  const indexEnvironment = gemma4LiteralOutputCoordinateEnvironment(base.navigation.outputDomain, request.outputCoordinate);
  for (let inputIndex = window.start; inputIndex < window.end; inputIndex += 1) {
    const learned = await readGemma4LiteralLearnedOperandScalar(artifact, weight, {
      ...indexEnvironment,
      reductionIndices: { input_feature: inputIndex },
    });
    learnedScalars.push(learned);
    const input = indexed(assignment.inputs[0]!, [...prefix, inputIndex]);
    terms.push({ inputIndex, input, learned, formula: gemma4LiteralScalarProductFormula(schedule, inputIndex, input, learned.literal) });
  }
  const biasRef = optionalLearnedOperand(artifact, entry, "bias");
  const bias = biasRef ? await readGemma4LiteralLearnedOperandScalar(artifact, biasRef, indexEnvironment) : undefined;
  if (bias) learnedScalars.push(bias);
  const bounds = assignment.operation === "clipped-linear" ? await readClippingBounds(artifact, entry, indexEnvironment) : undefined;
  if (bounds) learnedScalars.push(...bounds.values);
  const inputExpression = (inputIndex: number | string): string => bounds
    ? `F32(min(${bounds.output[1]}, max(${bounds.output[0]}, ${indexed(assignment.inputs[0]!, [...prefix, inputIndex])})))`
    : indexed(assignment.inputs[0]!, [...prefix, inputIndex]);
  if (bounds) {
    terms.forEach((term) => {
      term.input = inputExpression(term.inputIndex);
      term.formula = gemma4LiteralScalarProductFormula(schedule, term.inputIndex, term.input, term.learned.literal);
    });
  }
  const accumulationAssignments = buildGemma4LiteralLinearReductionAssignments(
    schedule,
    inFeatures!,
    base.dtypePolicy.accumulationDtype,
  );
  const biased = bias ? `F32(reduced + ${bias.literal})` : "reduced";
  const castResult = base.dtypePolicy.outputDtype === "BF16" ? `BF16(${biased})` : biased;
  const reduced = bounds ? `${base.dtypePolicy.outputDtype === "BF16" ? "BF16" : "F32"}(min(${bounds.output[3]}, max(${bounds.output[2]}, ${castResult})))` : castResult;
  const formula = `${base.output} = ${reduced}`;
  return {
    ...base,
    formula,
    scalarAssignments: [
      ...terms.map((term) => term.formula),
      ...accumulationAssignments,
      formula,
    ],
    learnedScalars,
    terms,
    reduction: {
      bounds: { startInclusive: 0, endExclusive: inFeatures! },
      schedule: structuredClone(schedule),
      complete: window.start === 0 && window.end === inFeatures,
      renderedWindow: { startInclusive: window.start, endExclusive: window.end },
      omittedTerms: inFeatures! - (window.end - window.start),
    },
  };
}

async function renderEmbedding(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralRenderedScalarView> {
  const weight = learnedOperand(artifact, entry, "weight");
  const tokenId = request.tokenId;
  if (!Number.isSafeInteger(tokenId) || tokenId! < 0 || tokenId! >= weight.tensor.shape[0]!) throw new Error(`${entry.assignment.id}: --token-id válido é obrigatório.`);
  const perLayer = entry.assignment.operation === "per-layer-embedding";
  const expectedRank = perLayer ? 4 : 3;
  if (request.outputCoordinate.length !== expectedRank) throw new Error(`${entry.assignment.id}: embedding requer coordenada rank ${expectedRank}.`);
  const learned = await readGemma4LiteralLearnedOperandScalar(artifact, weight, {
    ...gemma4LiteralOutputCoordinateEnvironment(base.navigation.outputDomain, request.outputCoordinate),
    inputScalars: { token_id: tokenId! },
  });
  const scale = perLayer ? Math.sqrt(artifact.program.contract.text.perLayerInputSize) : Math.sqrt(artifact.program.contract.text.hiddenSize);
  const formula = `${base.output} = F32(${learned.literal} * F32(${literal(scale)}))`;
  return { ...base, formula, scalarAssignments: [formula], learnedScalars: [learned] };
}

async function renderRmsNorm(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralRenderedScalarView> {
  const weight = optionalLearnedOperand(artifact, entry, "normalization-scale");
  const width = weight?.tensor.shape[0] ?? widthForUnscaledNorm(artifact, entry);
  const feature = last(request.outputCoordinate, entry.assignment.id);
  if (feature >= width) throw new Error(`${entry.assignment.id}: feature ${feature} excede width=${width}.`);
  const learned = weight ? await readGemma4LiteralLearnedOperandScalar(
    artifact,
    weight,
    gemma4LiteralOutputCoordinateEnvironment(base.navigation.outputDomain, request.outputCoordinate),
  ) : undefined;
  const epsilon = entry.scope === "vision" ? artifact.program.visionProgram.rmsNormEpsilon
    : entry.scope === "audio" ? artifact.program.audioProgram.rmsNormEpsilon
      : numericConfig(artifact.program.textProgram.config.rms_norm_eps, "text_config.rms_norm_eps");
  const input = indexed(entry.assignment.inputs[0]!, request.outputCoordinate);
  const result = `F32(F32(${input} * inv_rms) * ${learned?.literal ?? "1"})`;
  const formula = `${base.output} = ${base.dtypePolicy.outputDtype === "BF16" ? `BF16(${result})` : result}`;
  const reductionAssignments = base.dtypePolicy.reduction?.kind === "pytorch-cpu-f32-cascade-sum"
    ? buildGemma4LiteralCascadeSquareReductionAssignments(base.dtypePolicy.reduction, width)
    : [`sum = F32(sum_{i=0..${width - 1} in ascending order}(square[i]))`];
  return {
    ...base,
    formula,
    scalarAssignments: [
      `square[i] = F32(${indexed(entry.assignment.inputs[0]!, [...request.outputCoordinate.slice(0, -1), "i"])} * ${indexed(entry.assignment.inputs[0]!, [...request.outputCoordinate.slice(0, -1), "i"])})`,
      ...reductionAssignments,
      `mean_epsilon = F32(F32(sum / F32(${width})) + F32(${literal(epsilon)}))`,
      "sqrt_mean_epsilon = ARM_SQRT_F32(mean_epsilon)",
      "inv_rms = F32(1 / sqrt_mean_epsilon)",
      formula,
    ],
    learnedScalars: learned ? [learned] : [],
  };
}

async function renderPositionEmbedding(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralRenderedScalarView> {
  const position = request.positionCoordinate;
  if (!position || position.some((value) => !Number.isSafeInteger(value) || value < -1)) throw new Error(`${entry.assignment.id}: --position-coordinate x,y é obrigatório.`);
  if ((position[0] === -1) !== (position[1] === -1)) throw new Error(`${entry.assignment.id}: padding exige posição [-1,-1].`);
  if (request.outputCoordinate.length !== 3) throw new Error(`${entry.assignment.id}: posição 2-D requer saída [b,s,d].`);
  if (position[0] === -1) {
    const formula = `${base.output} = F32(0)`;
    return { ...base, formula, scalarAssignments: [formula], learnedScalars: [] };
  }
  const table = learnedOperand(artifact, entry, "position-table");
  if (position[0] >= table.tensor.shape[1]! || position[1] >= table.tensor.shape[1]!) throw new Error(`${entry.assignment.id}: posição fora da tabela.`);
  const outputEnvironment = gemma4LiteralOutputCoordinateEnvironment(base.navigation.outputDomain, request.outputCoordinate);
  const x = await readGemma4LiteralLearnedOperandScalar(artifact, table, {
    ...outputEnvironment, inputScalars: { position_axis: 0, position_index: position[0] },
  });
  const y = await readGemma4LiteralLearnedOperandScalar(artifact, table, {
    ...outputEnvironment, inputScalars: { position_axis: 1, position_index: position[1] },
  });
  const formula = `${base.output} = F32(${x.literal} + ${y.literal})`;
  return { ...base, formula, scalarAssignments: [formula], learnedScalars: [x, y] };
}

async function renderConv2d(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralRenderedScalarView> {
  if (request.outputCoordinate.length !== 4) throw new Error(`${entry.assignment.id}: Conv2d requer [b,out_channel,time,feature].`);
  const [batch, outputChannel, time, feature] = request.outputCoordinate;
  const weight = learnedOperand(artifact, entry, "convolution-kernel"), inputChannels = weight.tensor.shape[1]!;
  if (outputChannel! >= weight.tensor.shape[0]!) throw new Error(`${entry.assignment.id}: canal de saída fora do shape.`);
  const learnedScalars: Gemma4LiteralLearnedScalar[] = [], terms: Gemma4LiteralScalarTerm[] = [];
  const outputEnvironment = gemma4LiteralOutputCoordinateEnvironment(base.navigation.outputDomain, request.outputCoordinate);
  let term = 0;
  for (let inputChannel = 0; inputChannel < inputChannels; inputChannel += 1) for (let kernelTime = 0; kernelTime < 3; kernelTime += 1) for (let kernelFeature = 0; kernelFeature < 3; kernelFeature += 1) {
    const learned = await readGemma4LiteralLearnedOperandScalar(artifact, weight, {
      ...outputEnvironment,
      reductionIndices: { input_channel: inputChannel, kernel_time: kernelTime, kernel_feature: kernelFeature },
    });
    learnedScalars.push(learned);
    const sourceTime = time! * 2 + kernelTime - 1, sourceFeature = feature! * 2 + kernelFeature - 1;
    const input = `source_in_bounds(${sourceTime},${sourceFeature}) ? ${indexed(entry.assignment.inputs[0]!, [batch!, inputChannel, sourceTime, sourceFeature])} : F32(0)`;
    const termFormula = `product[${term}] = F32(${input} * ${learned.literal})`;
    terms.push({ inputIndex: term, input, learned, formula: termFormula });
    term += 1;
  }
  const reduction = requiredProductReduction(base, entry.assignment.id);
  const formula = `${base.output} = ${base.dtypePolicy.outputDtype === "BF16" ? "BF16" : "F32"}(reduced)`;
  return {
    ...base,
    formula,
    scalarAssignments: [...terms.map((candidate) => candidate.formula), ...buildGemma4LiteralLinearReductionAssignments(reduction, term, base.dtypePolicy.accumulationDtype), formula],
    learnedScalars,
    terms,
    reduction: {
      bounds: { startInclusive: 0, endExclusive: term }, schedule: structuredClone(reduction), complete: true,
      renderedWindow: { startInclusive: 0, endExclusive: term }, omittedTerms: 0,
    },
  };
}

async function renderChannelNorm(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralRenderedScalarView> {
  if (request.outputCoordinate.length !== 4) throw new Error(`${entry.assignment.id}: LayerNorm de canais requer [b,c,t,f].`);
  const weight = learnedOperand(artifact, entry, "normalization-scale"), channel = request.outputCoordinate[1]!, channels = weight.tensor.shape[0]!;
  const learned = await readGemma4LiteralLearnedOperandScalar(
    artifact, weight, gemma4LiteralOutputCoordinateEnvironment(base.navigation.outputDomain, request.outputCoordinate),
  );
  const [batch, , time, feature] = request.outputCoordinate, input = entry.assignment.inputs[0]!;
  const epsilon = artifact.program.audioProgram.rmsNormEpsilon;
  const reduction = base.dtypePolicy.reduction;
  if (reduction?.kind === "pytorch-cpu-bf16-welford") {
    const normalized = `F32(F32(${indexed(input, [batch!, channel, time!, feature!])} * inv_std) + bias)`;
    const result = `F32(${normalized} * ${learned.literal})`;
    const formula = `${base.output} = ${base.dtypePolicy.outputDtype === "BF16" ? `BF16(${result})` : result}`;
    return { ...base, formula, scalarAssignments: [
      `x[c] = ${indexed(input, [batch!, "c", time!, feature!])}, c=0..${channels - 1} ascending`,
      ...buildGemma4LiteralWelfordAssignments(reduction, channels),
      `variance_epsilon = F32(variance + F32(${literal(epsilon)}))`,
      "sqrt_variance_epsilon = ARM_SQRT_F32(variance_epsilon)",
      "inv_std = F32(1 / sqrt_variance_epsilon)",
      "bias = F32(-inv_std * mean)",
      formula,
    ], learnedScalars: [learned] };
  }
  if (!reduction || reduction.kind !== "ordered-scalar" || reduction.indexOrder !== "ascending") {
    throw new Error(`${entry.assignment.id}: LayerNorm de canais requer agenda Welford ou escalar ascendente declarada.`);
  }
  const formula = `${base.output} = F32(F32(${indexed(input, [batch!, channel, time!, feature!])} - mean) * inv_std * ${learned.literal})`;
  return { ...base, formula, scalarAssignments: [
    `mean_acc[0]=F32(0); mean_acc[c+1]=F32(mean_acc[c]+${indexed(input, [batch!, "c", time!, feature!])}), c=0..${channels - 1} ascending`,
    `mean=F32(mean_acc[${channels}]/F32(${channels}))`,
    `variance_term[c]=F32((${indexed(input, [batch!, "c", time!, feature!])}-mean)^2), c=0..${channels - 1}`,
    `variance_acc[0]=F32(0); variance_acc[c+1]=F32(variance_acc[c]+variance_term[c]), c=0..${channels - 1} ascending`,
    `variance=F32(variance_acc[${channels}]/F32(${channels}))`,
    `inv_std=F32(1/ARM_SQRT_F32(F32(variance+F32(${literal(epsilon)}))))`, formula,
  ], learnedScalars: [learned] };
}

async function renderDepthwise(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralRenderedScalarView> {
  if (request.outputCoordinate.length !== 3) throw new Error(`${entry.assignment.id}: depthwise requer [b,t,c].`);
  const [batch, time, channel] = request.outputCoordinate, weight = learnedOperand(artifact, entry, "convolution-kernel"), kernel = weight.tensor.shape[2]!;
  const learnedScalars: Gemma4LiteralLearnedScalar[] = [], terms: Gemma4LiteralScalarTerm[] = [];
  const outputEnvironment = gemma4LiteralOutputCoordinateEnvironment(base.navigation.outputDomain, request.outputCoordinate);
  for (let k = 0; k < kernel; k += 1) {
    const learned = await readGemma4LiteralLearnedOperandScalar(artifact, weight, {
      ...outputEnvironment, reductionIndices: { kernel_index: k },
    });
    learnedScalars.push(learned);
    const source = time! - kernel + 1 + k;
    const input = source < 0 ? "F32(0)" : indexed(entry.assignment.inputs[0]!, [batch!, source, channel!]);
    const termFormula = `product[${k}] = F32(${input} * ${learned.literal})`;
    terms.push({ inputIndex: k, input, learned, formula: termFormula });
  }
  const reduction = requiredProductReduction(base, entry.assignment.id);
  const formula = `${base.output} = ${base.dtypePolicy.outputDtype === "BF16" ? "BF16" : "F32"}(reduced)`;
  return {
    ...base,
    formula,
    scalarAssignments: [...terms.map((candidate) => candidate.formula), ...buildGemma4LiteralLinearReductionAssignments(reduction, kernel, base.dtypePolicy.accumulationDtype), formula],
    learnedScalars,
    terms,
    reduction: {
      bounds: { startInclusive: 0, endExclusive: kernel }, schedule: structuredClone(reduction), complete: true,
      renderedWindow: { startInclusive: 0, endExclusive: kernel }, omittedTerms: 0,
    },
  };
}

function requiredProductReduction(base: ScalarBase, operationId: string): Parameters<typeof buildGemma4LiteralLinearReductionAssignments>[0] {
  const reduction = base.dtypePolicy.reduction;
  if (!reduction || reduction.kind === "pytorch-cpu-f32-cascade-sum" || reduction.kind === "pytorch-cpu-bf16-welford") {
    throw new Error(`${operationId}: redução de produtos literal ausente ou incompatível.`);
  }
  return reduction;
}

async function renderPerDimScale(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralRenderedScalarView> {
  const headDim = artifact.program.audioProgram.tower.headDim;
  const learned = await readGemma4LiteralLearnedOperandScalar(
    artifact,
    learnedOperand(artifact, entry, "per-dimension-scale"),
    gemma4LiteralOutputCoordinateEnvironment(base.navigation.outputDomain, request.outputCoordinate),
  );
  const qScale = Math.fround(headDim ** -0.5 / Math.log(2));
  const softplus = `${learned.literal}>F32(20) ? ${learned.literal} : SLEEF_LOG1P_F32(SLEEF_EXP_F32(${learned.literal}))`;
  const formula = `${base.output} = F32(F32(${indexed(entry.assignment.inputs[0]!, request.outputCoordinate)} * ${literal(qScale)}) * BF16(F32(${softplus})))`;
  return { ...base, formula, scalarAssignments: [formula], learnedScalars: [learned] };
}

function plainFormulas(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, coordinate: number[]): string[] {
  const { assignment, scope } = entry, output = indexed(assignment.output, coordinate), inputs = assignment.inputs.map((input) => indexed(input, coordinate));
  switch (assignment.operation) {
    case "add": return [`${output} = F32(${inputs[0]} + ${inputs[1]})`];
    case "attention-logit-add": return [`${output} = F32(${inputs[0]} + ${inputs[1]})`];
    case "cast-bf16": return [`${output} = BF16(${inputs[0]})`];
    case "multiply": return [`${output} = F32(${inputs[0]} * ${inputs[1]})`];
    case "pixel-affine": return [`${output} = F32(2 * F32(${inputs[0]} - 0.5))`];
    case "relu": return [`${output} = F32(max(0, ${inputs[0]}))`];
    case "silu": return [`${output} = ${assignmentOutputDtype(assignment) === "BF16" ? "BF16" : "F32"}(F32(${inputs[0]} / F32(1 + SLEEF_EXP_F32(F32(-${inputs[0]})))))`];
    case "gelu-tanh": return [`${output} = ${assignmentOutputDtype(assignment) === "BF16" ? "BF16" : "F32"}(F32(F32(0.5*${inputs[0]}) * F32(1+SLEEF_TANH_F32(F32(${literal(Math.sqrt(2 / Math.PI))}*F32(${inputs[0]}+F32(0.044715*F32(${inputs[0]}*F32(${inputs[0]}*${inputs[0]})))))))))`];
    case "clip": return [`${output} = F32(min(${literal(artifact.program.audioProgram.gradientClipping)}, max(${literal(-artifact.program.audioProgram.gradientClipping)}, ${inputs[0]})))`];
    case "scale-f32": return [`${output} = F32(${inputs[0]} * F32(${literal(scaleFor(artifact, entry))}))`];
    case "reshape-heads": {
      if (coordinate.length !== 4) throw new Error(`${assignment.id}: reshape-heads requer [b,h,s,d].`);
      const headDim = scope === "vision" ? artifact.program.visionProgram.tower.headDim : artifact.program.audioProgram.tower.headDim;
      return [`${output} = ${indexed(assignment.inputs[0]!, [coordinate[0]!, coordinate[2]!, coordinate[1]! * headDim + coordinate[3]!])}`];
    }
    case "reshape-per-layer": {
      if (coordinate.length !== 4) throw new Error(`${assignment.id}: reshape-per-layer requer [b,s,l,d].`);
      const width = artifact.program.contract.text.perLayerInputSize;
      return [`${output} = ${indexed(assignment.inputs[0]!, [coordinate[0]!, coordinate[1]!, coordinate[2]! * width + coordinate[3]!])}`];
    }
    case "multidimensional-rope": return visionRopeFormula(artifact, assignment, coordinate, output);
    case "attention-score-matmul": return visionAttentionScoreFormula(artifact, assignment, coordinate, output);
    case "masked-softmax": return visionAttentionSoftmaxFormula(assignment, coordinate, output);
    case "attention-value-matmul": return visionAttentionValueFormula(artifact, assignment, coordinate, output);
    case "relative-attention-shift": return audioRelativeShiftFormula(artifact, assignment, coordinate, output);
    case "attention-softcap": return audioAttentionSoftcapFormula(artifact, assignment, coordinate, output);
    case "chunked-attention-mask": return audioAttentionMaskFormula(artifact, assignment, coordinate, output);
    case "chunked-relative-attention-softmax": return audioAttentionSoftmaxFormula(assignment, coordinate, output);
    case "chunked-relative-attention-values": return audioAttentionValueFormula(artifact, assignment, coordinate, output);
    case "split-gated-linear-unit": {
      const half = artifact.program.audioProgram.tower.hiddenSize, d = last(coordinate, assignment.id);
      return [`${output} = ${assignmentOutputDtype(assignment) === "BF16" ? "BF16" : "F32"}(F32(${indexed(assignment.inputs[0]!, [...coordinate.slice(0, -1), d])} / F32(1 + SLEEF_EXP_F32(F32(-${indexed(assignment.inputs[0]!, [...coordinate.slice(0, -1), half + d])})))))`];
    }
    case "relative-position-encoding": {
      if (coordinate.length !== 3 || coordinate[0] !== 0) throw new Error(`${assignment.id}: posição relativa requer [0,position,feature].`);
      const width = artifact.program.audioProgram.tower.hiddenSize, half = width / 2, feature = coordinate[2]!, frequency = feature % half;
      return [
        `increment = F32(${literal(Math.log(10000) / (half - 1))})`,
        `inverse_timescale = BF16(SLEEF_EXP_F32(F32(-${frequency} * increment)))`,
        `scaled_time = BF16(F32(F32(${audioContext(artifact) / 2 - coordinate[1]!}) * inverse_timescale))`,
        `${output} = BF16(SLEEF_${feature < half ? "SIN" : "COS"}_F32(scaled_time))`,
      ];
    }
    case "pool-by-position": return [`${output} = ${assignmentOutputDtype(assignment) === "BF16" ? "BF16" : "F32"}(ordered_F32_FMA_{patches mapped by floor(x/${artifact.program.visionProgram.tower.poolingKernelSize}),floor(y/${artifact.program.visionProgram.tower.poolingKernelSize}) in ascending patch order}(source, F32(1/F32(${artifact.program.visionProgram.tower.poolingKernelSize ** 2}))))`];
    case "pool-valid-mask": return [`${output} = BOOL(any non-padding patch maps to this pooling cell)`];
    case "strip-padding": return [`${output} = ${assignment.inputs[0]}[stable_batch_major_true_mask_row,${coordinate.at(-1)}]`];
    case "mask-input-features": return [`${output} = ${assignment.inputs[1]}[batch,time] ? ${inputs[0]} : F32(0)`];
    case "subsample-mask": return [`${output} = ${assignment.inputs[0]}[batch,2*time]`];
    case "reshape-conv-features": return [`${output} = row_major_alias(${assignment.inputs[0]})[${coordinate.join(",")}]`];
    case "placeholder-masks": return [`${output} = tuple(input_ids == image_token_id, input_ids == video_token_id, input_ids == audio_token_id)[${coordinate.join(",")}]`];
    case "vision-block-sequence-ids": {
      if (coordinate.length !== 2) throw new Error(`${assignment.id}: grupo vision requer [batch,sequence].`);
      return [`${output} = CONTIGUOUS_VISION_GROUP_ID(mm_token_type_ids[${coordinate[0]},0..${coordinate[1]}],${coordinate[1]})`];
    }
    case "causal-attention-mask": return [`${output} = key <= query ? F32(0) : -Infinity`];
    case "vision-sliding-attention-mask": {
      if (coordinate.length !== 4) throw new Error(`${assignment.id}: máscara sliding requer [batch,mask_head,query,key].`);
      const [batch, , query, key] = coordinate;
      return [`${output} = (${key}>${query}-${artifact.program.textProgram.config.sliding_window} && (${key}<=${query} || (vision_block_sequence_ids[${batch},${query}]>=0 && vision_block_sequence_ids[${batch},${query}]==vision_block_sequence_ids[${batch},${key}]))) ? F32(0) : F32(-Infinity)`];
    }
    case "replace-multimodal-ids-with-pad": {
      const token = `input_ids[${coordinate.join(",")}]`, modalities = artifact.program.contract.modalities;
      const predicate = [`${token}==${modalities.imageTokenId}`, ...(modalities.videoTokenId === undefined ? [] : [`${token}==${modalities.videoTokenId}`]), `${token}==${modalities.audioTokenId}`].join(" || ");
      const pad = artifact.program.textProgram.config.pad_token_id;
      if (typeof pad !== "number" || !Number.isSafeInteger(pad)) throw new Error(`${assignment.id}: pad_token_id inválido.`);
      return [`${output} = (${predicate}) ? ${pad} : ${token}`];
    }
    case "vision-feature-program": case "audio-feature-program": case "text-core": return [`${output} = EVALUATE(calculationGraph.assignments where invocationId==${JSON.stringify(assignment.id)} in ordinal order).terminalOutput; every bound assignment is serialized in the artifact`];
    case "video-frame-flatten": return [`${output} = ${assignment.inputs[0]}[floor(${coordinate[0]}/frames),${coordinate[0]} mod frames,${coordinate.slice(1).join(",")}] (row-major alias; no arithmetic)`];
    case "masked-scatter": case "masked-scatter-image-features": case "masked-scatter-audio-features": return [`${output} = placeholder_at(input_ids) ? next_feature_row : prior_embedding; rows consumed in batch-major order with exact cardinality`];
    default: throw new Error(`${assignment.id}: operação ${assignment.operation} sem fórmula escalar registrada.`);
  }
}

function visionRopeFormula(artifact: OpenGemma4CompositeLiteralArtifact, assignment: Assignment, coordinate: number[], output: string): string[] {
  if (coordinate.length !== 4) throw new Error(`${assignment.id}: RoPE vision requer [b,h,s,d].`);
  const dim = artifact.program.visionProgram.tower.headDim, part = dim / 2, half = part / 2, d = coordinate[3]!, axis = Math.floor(d / part), local = d % part, pair = local % half;
  const paired = axis * part + (local < half ? local + half : local - half), sign = local < half ? "-" : "+";
  const input = assignment.inputs[0]!, position = assignment.inputs[1]!, theta = artifact.program.visionProgram.tower.ropeTheta;
  return [
    `angle = F32(${position}[${coordinate[0]},${coordinate[2]},${axis}] / F32(${literal(theta)}^${literal((2 * pair) / part)}))`,
    "cosine=BF16(SLEEF_COS_F32(angle)); sine=BF16(SLEEF_SIN_F32(angle))",
    `direct=BF16(F32(${indexed(input, coordinate)}*cosine)); rotated=BF16(F32(${indexed(input, [...coordinate.slice(0, 3), paired])}*sine))`,
    `${output} = ${assignmentOutputDtype(assignment) === "BF16" ? "BF16" : "F32"}(F32(direct ${sign} rotated))`,
  ];
}

function visionAttentionScoreFormula(artifact: OpenGemma4CompositeLiteralArtifact, assignment: Assignment, coordinate: number[], output: string): string[] {
  if (coordinate.length !== 4) throw new Error(`${assignment.id}: score vision requer [b,h,q,k].`);
  const dim = artifact.program.visionProgram.tower.headDim;
  return [
    "acc[-1]=F32(0)",
    `acc[d]=F32_FMA(acc[d-1],${assignment.inputs[0]}[${coordinate[0]},${coordinate[1]},${coordinate[2]},d],${assignment.inputs[1]}[${coordinate[0]},${coordinate[1]},${coordinate[3]},d]), d=0..${dim - 1} ascending`,
    `${output}=${assignmentOutputDtype(assignment) === "BF16" ? "BF16" : "F32"}(acc[${dim - 1}])`,
  ];
}

function visionAttentionSoftmaxFormula(assignment: Assignment, coordinate: number[], output: string): string[] {
  if (coordinate.length !== 4) throw new Error(`${assignment.id}: softmax vision requer [b,h,q,k].`);
  const prefix = coordinate.slice(0, 3), key = coordinate[3]!;
  const positions = assignment.inputs[1]!;
  return [
    `valid[k]=!(${positions}[${coordinate[0]},k,0]==-1 && ${positions}[${coordinate[0]},k,1]==-1)`,
    `maximum=ORDERED_F32_REDUCE_MAX(${indexed(assignment.inputs[0]!, [...prefix, "k"])},k=0..patches-1 where valid[k])`,
    `exponential[k]=valid[k] ? SLEEF_EXP_F32(F32(${indexed(assignment.inputs[0]!, [...prefix, "k"])}-maximum)) : F32(0)`,
    "denominator=ORDERED_F32_REDUCE_SUM(exponential[k],k=0..patches-1 where valid[k])",
    `${output}=valid[${key}] ? ${assignmentOutputDtype(assignment) === "BF16" ? "BF16" : "F32"}(F32(exponential[${key}]/denominator)) : ${assignmentOutputDtype(assignment) === "BF16" ? "BF16" : "F32"}(0)`,
  ];
}

function visionAttentionValueFormula(artifact: OpenGemma4CompositeLiteralArtifact, assignment: Assignment, coordinate: number[], output: string): string[] {
  if (coordinate.length !== 3) throw new Error(`${assignment.id}: contexto vision requer [b,q,h*headDim+d].`);
  const dim = artifact.program.visionProgram.tower.headDim, merged = coordinate[2]!, head = Math.floor(merged / dim), d = merged % dim;
  return [
    "acc[-1]=F32(0)",
    `acc[k]=F32_FMA(acc[k-1],${assignment.inputs[0]}[${coordinate[0]},${head},${coordinate[1]},k],${assignment.inputs[1]}[${coordinate[0]},${head},k,${d}]), k=0..patches-1 ascending`,
    `${output}=${assignmentOutputDtype(assignment) === "BF16" ? "BF16" : "F32"}(acc[patches-1])`,
  ];
}

function audioRelativeShiftFormula(artifact: OpenGemma4CompositeLiteralArtifact, assignment: Assignment, coordinate: number[], output: string): string[] {
  if (coordinate.length !== 5) throw new Error(`${assignment.id}: relative shift audio requer [b,h,block,query,context].`);
  const context = audioContext(artifact), relativeLength = Math.floor(context / 2) + 1;
  const query = coordinate[3]!, keySlot = coordinate[4]!, source = relativeShiftSourceCoordinate(query, keySlot, context, relativeLength);
  const value = source.relativeIndex === undefined
    ? "F32(0)"
    : indexed(assignment.inputs[0]!, [coordinate[0]!, coordinate[1]!, coordinate[2]!, source.queryInBlock, source.relativeIndex]);
  return [
    `padded_length=${context + 1}; flattened=${query}*${context}+${keySlot}`,
    `source_query=floor(flattened/padded_length)=${source.queryInBlock}; source_relative=flattened mod padded_length=${source.relativeIndex ?? "padding"}`,
    `${output} = ${value}`,
  ];
}

function audioAttentionSoftcapFormula(artifact: OpenGemma4CompositeLiteralArtifact, assignment: Assignment, coordinate: number[], output: string): string[] {
  if (coordinate.length !== 5) throw new Error(`${assignment.id}: softcap audio requer [b,h,block,query,context].`);
  const cap = literal(artifact.program.audioProgram.tower.attentionLogitCap);
  return [`${output} = F32(SLEEF_TANH_F32(F32(${indexed(assignment.inputs[0]!, coordinate)} / ${cap})) * ${cap})`];
}

function audioAttentionMaskFormula(artifact: OpenGemma4CompositeLiteralArtifact, assignment: Assignment, coordinate: number[], output: string): string[] {
  if (coordinate.length !== 5) throw new Error(`${assignment.id}: mask de atenção audio requer [b,h,block,query,context].`);
  const tower = artifact.program.audioProgram.tower, queryIndex = `(${coordinate[2]}*${tower.attentionChunkSize}+${coordinate[3]})`;
  const keyIndex = `(${coordinate[2]}*${tower.attentionChunkSize}-${tower.attentionContextLeft - 1}+${coordinate[4]})`;
  return [
    `query_index=${queryIndex}; key_index=${keyIndex}`,
    `source_entry_exists = query_index < sequence && 0 <= key_index < sequence`,
    `eager_additive_mask_entry_is_zero = source_entry_exists && ${assignment.inputs[1]}[${coordinate[0]},key_index] && query_index >= key_index && query_index-key_index < ${tower.attentionContextLeft}`,
    `blocked_padding_is_zero = !source_entry_exists`,
    `${output} = eager_additive_mask_entry_is_zero || blocked_padding_is_zero ? ${literal(artifact.program.audioProgram.invalidAttentionLogit)} : ${indexed(assignment.inputs[0]!, coordinate)}`,
  ];
}

function relativeShiftSourceCoordinate(query: number, keySlot: number, context: number, relativeLength: number): { queryInBlock: number; relativeIndex?: number } {
  const flattened = query * context + keySlot, paddedLength = context + 1;
  const queryInBlock = Math.floor(flattened / paddedLength), relativeIndex = flattened % paddedLength;
  return relativeIndex < relativeLength ? { queryInBlock, relativeIndex } : { queryInBlock };
}

function audioAttentionSoftmaxFormula(assignment: Assignment, coordinate: number[], output: string): string[] {
  if (coordinate.length !== 5) throw new Error(`${assignment.id}: softmax audio requer [b,h,block,query,context].`);
  const [batch, head, block, query, keySlot] = coordinate;
  return [
    `maximum=ORDERED_F32_REDUCE_MAX(${assignment.inputs[0]}[${batch},${head},${block},${query},k],k=0..context-1)`,
    `exponential[k]=SLEEF_EXP_F32(F32(${assignment.inputs[0]}[${batch},${head},${block},${query},k]-maximum))`,
    "denominator=ORDERED_F32_REDUCE_SUM(exponential[k],k=0..context-1)",
    `${output}=F32(exponential[${keySlot}]/denominator)`,
  ];
}

function audioAttentionValueFormula(artifact: OpenGemma4CompositeLiteralArtifact, assignment: Assignment, coordinate: number[], output: string): string[] {
  if (coordinate.length !== 3) throw new Error(`${assignment.id}: contexto audio requer [b,t,h*headDim+d].`);
  const context = audioContext(artifact), merged = coordinate[2]!;
  return [`${output}=F32(sum_{key_slot=0..${context - 1} ascending and in-bounds}(F32(${assignment.inputs[0]}[b,head,block,query,key_slot]*${assignment.inputs[1]}[b,key_index,${merged}])))`];
}

function policyFor(assignment: Assignment): DtypePolicy {
  if ("dtypePolicy" in assignment && assignment.dtypePolicy) return structuredClone(assignment.dtypePolicy);
  return assignment.operation === "linear" || assignment.operation === "clipped-linear" || assignment.operation === "conv2d-stride2" || assignment.operation === "causal-depthwise-convolution" || assignment.operation === "layer-norm-channels" || assignment.operation === "rms-norm"
    ? structuredClone(F32_REDUCTION_POLICY) : structuredClone(F32_POLICY);
}

function assignmentOutputDtype(assignment: Assignment): string | undefined {
  return "dtypePolicy" in assignment ? assignment.dtypePolicy?.outputDtype : undefined;
}

async function readClippingBounds(
  artifact: OpenGemma4CompositeLiteralArtifact,
  entry: AssignmentEntry,
  environment: Gemma4LiteralLearnedIndexEnvironment,
): Promise<{ values: Gemma4LiteralLearnedScalar[]; output: [string, string, string, string] }> {
  const roles = ["input-min", "input-max", "output-min", "output-max"] as const;
  const scalarRefs = roles.map((role) => learnedOperand(artifact, entry, role));
  const values: Gemma4LiteralLearnedScalar[] = [];
  for (const reference of scalarRefs) {
    values.push(await readGemma4LiteralLearnedOperandScalar(artifact, reference, environment));
  }
  return { values, output: values.map((value) => value.literal) as [string, string, string, string] };
}

function learnedOperand(
  artifact: OpenGemma4CompositeLiteralArtifact,
  entry: AssignmentEntry,
  role: Gemma4LiteralLearnedOperandRole,
): Gemma4LiteralLearnedOperand {
  return requiredGemma4LiteralLearnedOperand(artifact.learnedOperands, entry.scope, entry.assignment.id, role);
}

function optionalLearnedOperand(
  artifact: OpenGemma4CompositeLiteralArtifact,
  entry: AssignmentEntry,
  role: Gemma4LiteralLearnedOperandRole,
): Gemma4LiteralLearnedOperand | undefined {
  return optionalGemma4LiteralLearnedOperand(artifact.learnedOperands, entry.scope, entry.assignment.id, role);
}

function scaleFor(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry): number {
  if (entry.scope === "vision") return Math.sqrt(artifact.program.visionProgram.tower.hiddenSize);
  if (entry.scope === "composite") return entry.assignment.id === "composite_ple_context_scale" ? artifact.program.contract.text.hiddenSize ** -0.5 : 2 ** -0.5;
  if (entry.assignment.id.endsWith("_k_scale")) return Math.fround(Math.log1p(Math.exp(1)) / Math.log(2));
  return artifact.program.audioProgram.tower.residualWeight;
}

function widthForUnscaledNorm(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry): number {
  if (entry.scope === "vision") return entry.assignment.id === "vision_language_projection_norm" ? artifact.program.visionProgram.tower.hiddenSize : artifact.program.visionProgram.tower.headDim;
  if (entry.scope === "audio") return entry.assignment.id === "audio_language_projection_norm" ? artifact.program.audioProgram.tower.outputProjectionSize : artifact.program.audioProgram.tower.hiddenSize;
  throw new Error(`${entry.assignment.id}: RMSNorm sem weight não possui width registrado.`);
}

function reductionWindow(request: Gemma4LiteralScalarViewRequest, width: number, id: string): { start: number; end: number } {
  if ((request.inputStart === undefined) !== (request.inputCount === undefined)) throw new Error(`${id}: inputStart/inputCount devem aparecer juntos.`);
  const start = request.inputStart ?? 0, count = request.inputCount ?? width;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(count) || start < 0 || count <= 0 || start + count > width) throw new Error(`${id}: janela ${start}+${count} fora de 0..${width}.`);
  return { start, end: start + count };
}

function audioContext(artifact: OpenGemma4CompositeLiteralArtifact): number {
  const tower = artifact.program.audioProgram.tower;
  return tower.attentionChunkSize + tower.attentionContextLeft - 1 + tower.attentionContextRight;
}

type ScalarBase = Gemma4LiteralScalarViewBase;
function indexed(name: string, coordinate: ReadonlyArray<number | string>): string { return `${name}[${coordinate.join(",")}]`; }
function literal(value: number): string { return Object.is(value, -0) ? "-0" : Number(value).toString(); }
function last(coordinate: number[], id: string): number { if (coordinate.length === 0) throw new Error(`${id}: coordenada vazia.`); return coordinate.at(-1)!; }
function assertCoordinate(coordinate: number[]): void { if (coordinate.length === 0 || coordinate.some((value) => !Number.isSafeInteger(value) || value < 0)) throw new Error("Vista escalar requer coordenada não negativa."); }
function numericConfig(value: unknown, name: string): number { if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${name} inválido.`); return value; }
