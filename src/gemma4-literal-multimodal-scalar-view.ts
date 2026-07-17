import type { Gemma4AudioAssignment } from "./gemma4-audio.js";
import type { Gemma4CompositeAssignment } from "./gemma4-composite.js";
import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import {
  listGemma4LiteralTextOperations,
  readGemma4LiteralLearnedScalar,
  renderGemma4LiteralScalarView,
  type Gemma4LiteralLearnedScalar,
  type Gemma4LiteralOperationNavigation,
  type Gemma4LiteralScalarTerm,
  type Gemma4LiteralScalarView,
  type Gemma4LiteralScalarViewRequest,
} from "./gemma4-literal-scalar-view.js";
import type { Gemma4VisionAssignment } from "./gemma4-vision.js";
import type { DtypePolicy, TensorRef } from "./types.js";

type Assignment = Gemma4CompositeAssignment | Gemma4VisionAssignment | Gemma4AudioAssignment;
type NonTextScope = "composite" | "vision" | "audio";

interface AssignmentEntry {
  assignment: Assignment;
  scope: NonTextScope;
  operationId: string;
  invocationId?: string;
  bindings: ReadonlyMap<string, string>;
}

interface NavigationSeed {
  operationId: string;
  definitionId?: string;
  operation: string;
  scope: Gemma4LiteralOperationNavigation["scope"];
  layer?: number;
  invocationId?: string;
  inputs: string[];
  output: string;
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
  const seeds = navigationSeeds(artifact);
  const ids = new Set<string>(), producerByOutput = new Map<string, string>();
  for (const seed of seeds) {
    if (ids.has(seed.operationId)) throw new Error(`Plano literal Gemma 4 possui operação instanciada duplicada: ${seed.operationId}.`);
    if (producerByOutput.has(seed.output)) throw new Error(`Plano literal Gemma 4 redeclara saída instanciada: ${seed.output}.`);
    ids.add(seed.operationId);
    producerByOutput.set(seed.output, seed.operationId);
  }
  const consumersByOutput = new Map<string, string[]>();
  for (const seed of seeds) for (const input of seed.inputs) {
    const consumers = consumersByOutput.get(input) ?? [];
    consumers.push(seed.operationId);
    consumersByOutput.set(input, consumers);
  }
  const navigation = seeds.map((seed, ordinal): Gemma4LiteralOperationNavigation => ({
    operationId: seed.operationId,
    ...(seed.definitionId ? { definitionId: seed.definitionId } : {}),
    operation: seed.operation,
    scope: seed.scope,
    ...(seed.layer === undefined ? {} : { layer: seed.layer }),
    ...(seed.invocationId ? { invocationId: seed.invocationId } : {}),
    ordinal,
    output: seed.output,
    predecessors: seed.inputs.map((input) => ({ input, ...(producerByOutput.has(input) ? { producerOperationId: producerByOutput.get(input)! } : {}) })),
    consumers: consumersByOutput.get(seed.output) ?? [],
  }));
  const navigationById = new Map(navigation.map((entry) => [entry.operationId, entry]));
  for (const entry of navigation) for (const predecessor of entry.predecessors) {
    if (!predecessor.producerOperationId) continue;
    const producer = navigationById.get(predecessor.producerOperationId)!;
    if (producer.ordinal >= entry.ordinal) throw new Error(`${entry.operationId}: predecessor ${producer.operationId} não antecede o consumidor no plano literal.`);
  }
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
    return bindScalarView(rendered, navigation, textBindings());
  }
  const entry = assignmentEntries(artifact).find((candidate) => candidate.operationId === request.operationId);
  if (!entry) throw new Error(`Atribuição Gemma 4 literal não possui definição instanciada: ${request.operationId}.`);
  const base = {
    kind: "gemma4-literal-scalar-view" as const,
    sourceCheckpointAccessed: false as const,
    navigation: { ...navigation, output: entry.assignment.output },
    outputCoordinate: [...request.outputCoordinate],
    output: indexed(entry.assignment.output, request.outputCoordinate),
    dtypePolicy: policyFor(entry.assignment),
  };
  const operation = entry.assignment.operation;
  let rendered: Gemma4LiteralScalarView | undefined;
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
  if (!rendered) {
    const formulas = plainFormulas(artifact, entry, request.outputCoordinate);
    rendered = { ...base, formula: formulas.at(-1)!, scalarAssignments: formulas, learnedScalars: [] };
  }
  return bindScalarView(rendered, navigation, entry.bindings);
}

function assignmentEntries(artifact: OpenGemma4CompositeLiteralArtifact): AssignmentEntry[] {
  const entries: AssignmentEntry[] = [];
  for (const assignment of artifact.program.assignments) {
    if (assignment.operation === "vision-feature-program") {
      entries.push(...instantiateAssignments(artifact.program.visionProgram.assignments, "vision", assignment, artifact.program.visionProgram.output));
    } else if (assignment.operation === "audio-feature-program") {
      entries.push(...instantiateAssignments(artifact.program.audioProgram.assignments, "audio", assignment, artifact.program.audioProgram.output));
    } else if (assignment.operation !== "text-core") {
      entries.push({ assignment, scope: "composite", operationId: assignment.id, bindings: new Map() });
    }
  }
  return entries;
}

function navigationSeeds(artifact: OpenGemma4CompositeLiteralArtifact): NavigationSeed[] {
  const entries = assignmentEntries(artifact);
  const seeds: NavigationSeed[] = [];
  for (const assignment of artifact.program.assignments) {
    if (assignment.operation === "vision-feature-program" || assignment.operation === "audio-feature-program") {
      for (const entry of entries.filter((candidate) => candidate.invocationId === assignment.id)) seeds.push(seedFromAssignment(entry));
      continue;
    }
    if (assignment.operation === "text-core") {
      for (const entry of listGemma4LiteralTextOperations(artifact).filter((candidate) => candidate.scope !== "text-prelude")) {
        seeds.push({
          operationId: entry.operationId,
          operation: entry.operation,
          scope: entry.scope,
          ...(entry.layer === undefined ? {} : { layer: entry.layer }),
          invocationId: assignment.id,
          inputs: entry.predecessors.map((predecessor) => bindName(predecessor.input, textBindings())),
          output: bindName(entry.output, textBindings()),
        });
      }
      continue;
    }
    const entry = entries.find((candidate) => candidate.operationId === assignment.id);
    if (!entry) throw new Error(`${assignment.id}: atribuição composite não foi instanciada.`);
    seeds.push(seedFromAssignment(entry));
  }
  return seeds;
}

function seedFromAssignment(entry: AssignmentEntry): NavigationSeed {
  return {
    operationId: entry.operationId,
    ...(entry.invocationId ? { definitionId: entry.assignment.id } : {}),
    operation: entry.assignment.operation,
    scope: entry.scope,
    ...(entry.invocationId ? { invocationId: entry.invocationId } : {}),
    inputs: entry.assignment.inputs.map((input) => bindName(input, entry.bindings)),
    output: bindName(entry.assignment.output, entry.bindings),
  };
}

function instantiateAssignments(
  definitions: Assignment[],
  scope: "vision" | "audio",
  invocation: Gemma4CompositeAssignment,
  terminalOutput: string,
): AssignmentEntry[] {
  const external = scope === "vision"
    ? new Map([["pixel_values", invocation.inputs[0]!], ["pixel_position_ids", invocation.inputs[1]!]])
    : new Map([["input_features", invocation.inputs[0]!], ["input_features_mask", invocation.inputs[1]!]]);
  const usable = definitions.filter((definition) => definition.operation !== "masked-scatter-image-features" && definition.operation !== "masked-scatter-audio-features");
  const bindings = new Map<string, string>(external);
  for (const definition of usable) bindings.set(definition.output, definition.output === terminalOutput ? invocation.output : `${invocation.id}/${definition.output}`);
  return usable.map((assignment) => ({
    assignment,
    scope,
    operationId: `${invocation.id}/${assignment.id}`,
    invocationId: invocation.id,
    bindings,
  }));
}

function textBindings(): ReadonlyMap<string, string> {
  return new Map([["attention_mask:full", "full_attention_mask"], ["attention_mask:sliding", "sliding_attention_mask"]]);
}

function bindScalarView(
  view: Gemma4LiteralScalarView,
  navigation: Gemma4LiteralOperationNavigation,
  bindings: ReadonlyMap<string, string>,
): Gemma4LiteralScalarView {
  const orderedBindings = [...bindings.entries()].sort(([left], [right]) => right.length - left.length);
  const replace = (value: string): string => orderedBindings.reduce((current, [source, target]) => current.split(source).join(target), value);
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

function bindName(value: string, bindings: ReadonlyMap<string, string>): string { return bindings.get(value) ?? value; }

async function renderLinear(
  artifact: OpenGemma4CompositeLiteralArtifact,
  entry: AssignmentEntry,
  request: Gemma4LiteralMultimodalScalarViewRequest,
  base: ScalarBase,
): Promise<Gemma4LiteralScalarView> {
  const assignment = entry.assignment;
  const tensors = assignment.tensors ?? [];
  const weight = tensors.find((tensor) => tensor.shape.length === 2);
  if (!weight) throw new Error(`${assignment.id}: linear literal sem matriz [out,in].`);
  const outputFeature = last(request.outputCoordinate, assignment.id);
  const [outFeatures, inFeatures] = weight.shape;
  if (outputFeature >= outFeatures!) throw new Error(`${assignment.id}: feature ${outputFeature} excede outFeatures=${outFeatures}.`);
  const prefix = request.outputCoordinate.slice(0, -1);
  const window = reductionWindow(request, inFeatures!, assignment.id);
  const learnedScalars: Gemma4LiteralLearnedScalar[] = [];
  const terms: Gemma4LiteralScalarTerm[] = [];
  for (let inputIndex = window.start; inputIndex < window.end; inputIndex += 1) {
    const learned = await readGemma4LiteralLearnedScalar(artifact, weight, [outputFeature, inputIndex]);
    learnedScalars.push(learned);
    const input = indexed(assignment.inputs[0]!, [...prefix, inputIndex]);
    terms.push({ inputIndex, input, learned, formula: `product[${inputIndex}] = F32(${input} * ${learned.literal})` });
  }
  const biasRef = tensors.find((tensor) => tensor.shape.length === 1 && tensor.shape[0] === outFeatures);
  const bias = biasRef ? await readGemma4LiteralLearnedScalar(artifact, biasRef, [outputFeature]) : undefined;
  if (bias) learnedScalars.push(bias);
  const bounds = assignment.operation === "clipped-linear" ? await readClippingBounds(artifact, tensors, weight) : undefined;
  if (bounds) learnedScalars.push(...bounds.values);
  const inputExpression = (inputIndex: number): string => bounds
    ? `F32(min(${bounds.output[1]}, max(${bounds.output[0]}, ${indexed(assignment.inputs[0]!, [...prefix, inputIndex])})))`
    : indexed(assignment.inputs[0]!, [...prefix, inputIndex]);
  if (bounds) {
    terms.forEach((term) => { term.input = inputExpression(term.inputIndex); term.formula = `product[${term.inputIndex}] = F32(${term.input} * ${term.learned.literal})`; });
  }
  const initial = bias ? bias.literal : "0";
  const reduced = bounds ? `F32(min(${bounds.output[3]}, max(${bounds.output[2]}, acc[${inFeatures! - 1}])))` : `acc[${inFeatures! - 1}]`;
  const formula = `${base.output} = ${reduced}`;
  return {
    ...base,
    formula,
    scalarAssignments: [
      ...terms.map((term) => term.formula),
      `acc[-1] = ${initial}`,
      `acc[i] = F32(acc[i-1] + product[i]), i=0..${inFeatures! - 1} in ascending order`,
      formula,
    ],
    learnedScalars,
    terms,
    reduction: {
      bounds: { startInclusive: 0, endExclusive: inFeatures! },
      schedule: { kind: "ordered-scalar", indexOrder: "ascending" },
      complete: window.start === 0 && window.end === inFeatures,
      renderedWindow: { startInclusive: window.start, endExclusive: window.end },
      omittedTerms: inFeatures! - (window.end - window.start),
    },
  };
}

async function renderEmbedding(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralScalarView> {
  const weight = onlyTensor(entry.assignment);
  const tokenId = request.tokenId;
  if (!Number.isSafeInteger(tokenId) || tokenId! < 0 || tokenId! >= weight.shape[0]!) throw new Error(`${entry.assignment.id}: --token-id válido é obrigatório.`);
  const perLayer = entry.assignment.operation === "per-layer-embedding";
  const expectedRank = perLayer ? 4 : 3;
  if (request.outputCoordinate.length !== expectedRank) throw new Error(`${entry.assignment.id}: embedding requer coordenada rank ${expectedRank}.`);
  const feature = perLayer
    ? request.outputCoordinate[2]! * artifact.program.contract.text.perLayerInputSize + request.outputCoordinate[3]!
    : request.outputCoordinate[2]!;
  const learned = await readGemma4LiteralLearnedScalar(artifact, weight, [tokenId!, feature]);
  const scale = perLayer ? Math.sqrt(artifact.program.contract.text.perLayerInputSize) : Math.sqrt(artifact.program.contract.text.hiddenSize);
  const formula = `${base.output} = F32(${learned.literal} * F32(${literal(scale)}))`;
  return { ...base, formula, scalarAssignments: [formula], learnedScalars: [learned] };
}

async function renderRmsNorm(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralScalarView> {
  const weight = entry.assignment.tensors?.[0];
  const width = weight?.shape[0] ?? widthForUnscaledNorm(artifact, entry);
  const feature = last(request.outputCoordinate, entry.assignment.id);
  if (feature >= width) throw new Error(`${entry.assignment.id}: feature ${feature} excede width=${width}.`);
  const learned = weight ? await readGemma4LiteralLearnedScalar(artifact, weight, [feature]) : undefined;
  const epsilon = entry.scope === "vision" ? artifact.program.visionProgram.rmsNormEpsilon
    : entry.scope === "audio" ? artifact.program.audioProgram.rmsNormEpsilon
      : numericConfig(artifact.program.textProgram.config.rms_norm_eps, "text_config.rms_norm_eps");
  const input = indexed(entry.assignment.inputs[0]!, request.outputCoordinate);
  const formula = `${base.output} = F32(F32(${input} * inv_rms) * ${learned?.literal ?? "1"})`;
  return {
    ...base,
    formula,
    scalarAssignments: [
      `square[i] = F32(${indexed(entry.assignment.inputs[0]!, [...request.outputCoordinate.slice(0, -1), "i"])} * ${indexed(entry.assignment.inputs[0]!, [...request.outputCoordinate.slice(0, -1), "i"])})`,
      `sum = F32(sum_{i=0..${width - 1} in ascending order}(square[i]))`,
      `inv_rms = F32(F32(F32(sum / F32(${width})) + F32(${literal(epsilon)}))^-0.5)`,
      formula,
    ],
    learnedScalars: learned ? [learned] : [],
  };
}

async function renderPositionEmbedding(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralScalarView> {
  const position = request.positionCoordinate;
  if (!position || position.some((value) => !Number.isSafeInteger(value) || value < -1)) throw new Error(`${entry.assignment.id}: --position-coordinate x,y é obrigatório.`);
  if ((position[0] === -1) !== (position[1] === -1)) throw new Error(`${entry.assignment.id}: padding exige posição [-1,-1].`);
  if (request.outputCoordinate.length !== 3) throw new Error(`${entry.assignment.id}: posição 2-D requer saída [b,s,d].`);
  if (position[0] === -1) {
    const formula = `${base.output} = F32(0)`;
    return { ...base, formula, scalarAssignments: [formula], learnedScalars: [] };
  }
  const table = onlyTensor(entry.assignment), feature = request.outputCoordinate[2]!;
  if (position[0] >= table.shape[1]! || position[1] >= table.shape[1]!) throw new Error(`${entry.assignment.id}: posição fora da tabela.`);
  const x = await readGemma4LiteralLearnedScalar(artifact, table, [0, position[0], feature]);
  const y = await readGemma4LiteralLearnedScalar(artifact, table, [1, position[1], feature]);
  const formula = `${base.output} = F32(${x.literal} + ${y.literal})`;
  return { ...base, formula, scalarAssignments: [formula], learnedScalars: [x, y] };
}

async function renderConv2d(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralScalarView> {
  if (request.outputCoordinate.length !== 4) throw new Error(`${entry.assignment.id}: Conv2d requer [b,out_channel,time,feature].`);
  const [batch, outputChannel, time, feature] = request.outputCoordinate;
  const weight = onlyTensor(entry.assignment), inputChannels = weight.shape[1]!;
  if (outputChannel! >= weight.shape[0]!) throw new Error(`${entry.assignment.id}: canal de saída fora do shape.`);
  const learnedScalars: Gemma4LiteralLearnedScalar[] = [], assignments: string[] = [];
  let term = 0;
  for (let inputChannel = 0; inputChannel < inputChannels; inputChannel += 1) for (let kernelTime = 0; kernelTime < 3; kernelTime += 1) for (let kernelFeature = 0; kernelFeature < 3; kernelFeature += 1) {
    const learned = await readGemma4LiteralLearnedScalar(artifact, weight, [outputChannel!, inputChannel, kernelTime, kernelFeature]);
    learnedScalars.push(learned);
    const sourceTime = time! * 2 + kernelTime - 1, sourceFeature = feature! * 2 + kernelFeature - 1;
    assignments.push(`product[${term}] = source_in_bounds(${sourceTime},${sourceFeature}) ? F32(${indexed(entry.assignment.inputs[0]!, [batch!, inputChannel, sourceTime, sourceFeature])} * ${learned.literal}) : F32(0)`);
    term += 1;
  }
  const formula = `${base.output} = acc[${term - 1}]`;
  return { ...base, formula, scalarAssignments: [...assignments, "acc[-1] = F32(0)", `acc[i] = F32(acc[i-1] + product[i]), i=0..${term - 1} in channel,kernel_time,kernel_feature order`, formula], learnedScalars };
}

async function renderChannelNorm(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralScalarView> {
  if (request.outputCoordinate.length !== 4) throw new Error(`${entry.assignment.id}: LayerNorm de canais requer [b,c,t,f].`);
  const weight = onlyTensor(entry.assignment), channel = request.outputCoordinate[1]!, channels = weight.shape[0]!;
  const learned = await readGemma4LiteralLearnedScalar(artifact, weight, [channel]);
  const [batch, , time, feature] = request.outputCoordinate, input = entry.assignment.inputs[0]!;
  const epsilon = artifact.program.audioProgram.rmsNormEpsilon;
  const formula = `${base.output} = F32(F32(${indexed(input, [batch!, channel, time!, feature!])} - mean) * inv_std * ${learned.literal})`;
  return { ...base, formula, scalarAssignments: [
    `mean_acc[-1]=F32(0); mean_acc[c]=F32(mean_acc[c-1]+${indexed(input, [batch!, "c", time!, feature!])}), c=0..${channels - 1}`,
    `mean=F32(mean_acc[${channels - 1}]/F32(${channels}))`,
    `variance_term[c]=F32((${indexed(input, [batch!, "c", time!, feature!])}-mean)^2)`,
    `variance=F32(sum_{c=0..${channels - 1} ascending}(variance_term[c])/F32(${channels}))`,
    `inv_std=F32(F32(variance+F32(${literal(epsilon)}))^-0.5)`, formula,
  ], learnedScalars: [learned] };
}

async function renderDepthwise(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralScalarView> {
  if (request.outputCoordinate.length !== 3) throw new Error(`${entry.assignment.id}: depthwise requer [b,t,c].`);
  const [batch, time, channel] = request.outputCoordinate, weight = onlyTensor(entry.assignment), kernel = weight.shape[2]!;
  const learnedScalars: Gemma4LiteralLearnedScalar[] = [], assignments: string[] = [];
  for (let k = 0; k < kernel; k += 1) {
    const learned = await readGemma4LiteralLearnedScalar(artifact, weight, [channel!, 0, k]);
    learnedScalars.push(learned);
    const source = time! - kernel + 1 + k;
    assignments.push(`product[${k}] = ${source < 0 ? "F32(0)" : `F32(${indexed(entry.assignment.inputs[0]!, [batch!, source, channel!])} * ${learned.literal})`}`);
  }
  const formula = `${base.output} = acc[${kernel - 1}]`;
  return { ...base, formula, scalarAssignments: [...assignments, "acc[-1]=F32(0)", `acc[k]=F32(acc[k-1]+product[k]), k=0..${kernel - 1} ascending`, formula], learnedScalars };
}

async function renderPerDimScale(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, request: Gemma4LiteralMultimodalScalarViewRequest, base: ScalarBase): Promise<Gemma4LiteralScalarView> {
  const feature = last(request.outputCoordinate, entry.assignment.id), headDim = artifact.program.audioProgram.tower.headDim, d = feature % headDim;
  const learned = await readGemma4LiteralLearnedScalar(artifact, onlyTensor(entry.assignment), [d]);
  const qScale = Math.fround(headDim ** -0.5 / Math.log(2));
  const formula = `${base.output} = F32(${indexed(entry.assignment.inputs[0]!, request.outputCoordinate)} * F32(${literal(qScale)} * F32(log1p(exp(${learned.literal})))))`;
  return { ...base, formula, scalarAssignments: [formula], learnedScalars: [learned] };
}

function plainFormulas(artifact: OpenGemma4CompositeLiteralArtifact, entry: AssignmentEntry, coordinate: number[]): string[] {
  const { assignment, scope } = entry, output = indexed(assignment.output, coordinate), inputs = assignment.inputs.map((input) => indexed(input, coordinate));
  switch (assignment.operation) {
    case "add": return [`${output} = F32(${inputs[0]} + ${inputs[1]})`];
    case "multiply": return [`${output} = F32(${inputs[0]} * ${inputs[1]})`];
    case "pixel-affine": return [`${output} = F32(2 * F32(${inputs[0]} - 0.5))`];
    case "relu": return [`${output} = F32(max(0, ${inputs[0]}))`];
    case "silu": return [`${output} = F32(${inputs[0]} / F32(1 + exp(F32(-${inputs[0]}))))`];
    case "gelu-tanh": return [`${output} = F32(F32(0.5*${inputs[0]}) * F32(1+F32(tanh(F32(sqrt(2/pi)*F32(${inputs[0]}+F32(0.044715*F32(${inputs[0]}*F32(${inputs[0]}*${inputs[0]})))))))))`];
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
    case "bidirectional-attention": return visionAttentionFormula(artifact, assignment, coordinate, output);
    case "chunked-relative-attention": return audioAttentionFormula(artifact, assignment, coordinate, output);
    case "split-gated-linear-unit": {
      const half = artifact.program.audioProgram.tower.hiddenSize, d = last(coordinate, assignment.id);
      return [`${output} = F32(${indexed(assignment.inputs[0]!, [...coordinate.slice(0, -1), d])} / F32(1 + exp(F32(-${indexed(assignment.inputs[0]!, [...coordinate.slice(0, -1), half + d])}))))`];
    }
    case "relative-position-encoding": return [`${output} = deterministic_sin_cos_relative_position[${coordinate.join(",")}] with context=${audioContext(artifact)} and F32 rounding after scale,sin,cos`];
    case "pool-by-position": return [`${output} = F32(sum_{patches mapped by floor(x/${artifact.program.visionProgram.tower.poolingKernelSize}),floor(y/${artifact.program.visionProgram.tower.poolingKernelSize}) in patch order}(F32(source / F32(${artifact.program.visionProgram.tower.poolingKernelSize ** 2}))))`];
    case "pool-valid-mask": return [`${output} = BOOL(any non-padding patch maps to this pooling cell)`];
    case "strip-padding": return [`${output} = ${assignment.inputs[0]}[stable_batch_major_true_mask_row,${coordinate.at(-1)}]`];
    case "mask-input-features": return [`${output} = ${assignment.inputs[1]}[batch,time] ? ${inputs[0]} : F32(0)`];
    case "subsample-mask": return [`${output} = ${assignment.inputs[0]}[batch,2*time]`];
    case "reshape-conv-features": return [`${output} = row_major_alias(${assignment.inputs[0]})[${coordinate.join(",")}]`];
    case "placeholder-masks": return [`${output} = tuple(input_ids == image_token_id, input_ids == video_token_id, input_ids == audio_token_id)[${coordinate.join(",")}]`];
    case "vision-block-sequence-ids": return [`${output} = contiguous_group_id(mm_token_type_ids in {1,2}) else -1 at [${coordinate.join(",")}]`];
    case "causal-attention-mask": return [`${output} = key <= query ? F32(0) : -Infinity`];
    case "vision-sliding-attention-mask": return [`${output} = key > query-sliding_window and (key<=query or same_nonnegative_vision_block) ? F32(0) : -Infinity`];
    case "replace-multimodal-ids-with-pad": return [`${output} = is_declared_modal_id(input_ids[${coordinate.join(",")}]) ? pad_token_id : input_ids[${coordinate.join(",")}]`];
    case "vision-feature-program": case "audio-feature-program": case "text-core": return [`${output} = declared_subprogram(${assignment.inputs.join(",")}); inspect its named scope assignments for scalar expansion`];
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
    "cosine=F32(cos(angle)); sine=F32(sin(angle))",
    `${output} = F32(F32(${indexed(input, coordinate)}*cosine) ${sign} F32(${indexed(input, [...coordinate.slice(0, 3), paired])}*sine))`,
  ];
}

function visionAttentionFormula(artifact: OpenGemma4CompositeLiteralArtifact, assignment: Assignment, coordinate: number[], output: string): string[] {
  if (coordinate.length !== 3) throw new Error(`${assignment.id}: atenção vision requer [b,s,h*headDim+d].`);
  const dim = artifact.program.visionProgram.tower.headDim, merged = coordinate[2]!, head = Math.floor(merged / dim), d = merged % dim;
  return [
    `score[k]=pixel_position_valid(k) ? F32(sum_{i=0..${dim - 1} ascending}(F32(${assignment.inputs[0]}[${coordinate[0]},${head},${coordinate[1]},i]*${assignment.inputs[1]}[${coordinate[0]},${head},k,i]))) : -Infinity`,
    "probability[k]=F32(exp(F32(score[k]-max(score)))/F32(sum_k_ascending(exp(F32(score[k]-max(score))))))",
    `${output}=F32(sum_{k=0..patches-1 ascending}(F32(probability[k]*${assignment.inputs[2]}[${coordinate[0]},${head},k,${d}])))`,
  ];
}

function audioAttentionFormula(artifact: OpenGemma4CompositeLiteralArtifact, assignment: Assignment, coordinate: number[], output: string): string[] {
  if (coordinate.length !== 3) throw new Error(`${assignment.id}: atenção audio requer [b,t,h*headDim+d].`);
  const tower = artifact.program.audioProgram.tower, dim = tower.headDim, merged = coordinate[2]!, head = Math.floor(merged / dim), d = merged % dim, context = audioContext(artifact);
  return [
    `AC[key_slot]=F32(sum_{i=0..${dim - 1} ascending}(F32(${assignment.inputs[0]}[${coordinate[0]},${coordinate[1]},${head}*${dim}+i]*${assignment.inputs[1]}[${coordinate[0]},key_index,${head}*${dim}+i])))`,
    `BD[key_slot]=F32(sum_{i=0..${dim - 1} ascending}(F32(relative_shift(${assignment.inputs[0]})[${coordinate[0]},${coordinate[1]},${head}*${dim}+i]*${assignment.inputs[3]}[relative_index,${head}*${dim}+i])))`,
    `score[key_slot]=F32(F32(tanh(F32((allowed ? F32(AC+BD) : ${literal(artifact.program.audioProgram.invalidAttentionLogit)})/${literal(tower.attentionLogitCap)})))*${literal(tower.attentionLogitCap)}), key_slot=0..${context - 1}`,
    "probability[key_slot]=F32(exp(F32(score[key_slot]-max(score)))/F32(sum_key_slot_ascending(exp(F32(score[key_slot]-max(score))))))",
    `${output}=F32(sum_{key_slot=0..${context - 1} ascending and in-bounds}(F32(probability[key_slot]*${assignment.inputs[2]}[${coordinate[0]},key_index,${merged}])))`,
  ];
}

function policyFor(assignment: Assignment): DtypePolicy {
  return assignment.operation === "linear" || assignment.operation === "clipped-linear" || assignment.operation === "conv2d-stride2" || assignment.operation === "causal-depthwise-convolution" || assignment.operation === "layer-norm-channels" || assignment.operation === "rms-norm"
    ? structuredClone(F32_REDUCTION_POLICY) : structuredClone(F32_POLICY);
}

async function readClippingBounds(artifact: OpenGemma4CompositeLiteralArtifact, tensors: TensorRef[], weight: TensorRef): Promise<{ values: Gemma4LiteralLearnedScalar[]; output: [string, string, string, string] }> {
  const suffixes = ["input_min", "input_max", "output_min", "output_max"] as const;
  const scalarRefs = suffixes.map((suffix) => {
    const matches = tensors.filter((tensor) => tensor !== weight
      && tensor.name.endsWith(`.${suffix}`)
      && (tensor.shape.length === 0 || (tensor.shape.length === 1 && tensor.shape[0] === 1)));
    if (matches.length !== 1) throw new Error(`${weight.name}: clipped-linear requer exatamente um bound escalar ${suffix}.`);
    return matches[0]!;
  });
  const values: Gemma4LiteralLearnedScalar[] = [];
  for (const reference of scalarRefs) {
    values.push(await readGemma4LiteralLearnedScalar(artifact, reference, reference.shape.length === 0 ? [] : [0]));
  }
  return { values, output: values.map((value) => value.literal) as [string, string, string, string] };
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

function onlyTensor(assignment: Assignment): TensorRef {
  if (assignment.tensors?.length !== 1) throw new Error(`${assignment.id}: esperava exatamente um tensor aprendido.`);
  return assignment.tensors[0]!;
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

type ScalarBase = Omit<Gemma4LiteralScalarView, "formula" | "scalarAssignments" | "learnedScalars">;
function indexed(name: string, coordinate: ReadonlyArray<number | string>): string { return `${name}[${coordinate.join(",")}]`; }
function literal(value: number): string { return Object.is(value, -0) ? "-0" : Number(value).toString(); }
function last(coordinate: number[], id: string): number { if (coordinate.length === 0) throw new Error(`${id}: coordenada vazia.`); return coordinate.at(-1)!; }
function assertCoordinate(coordinate: number[]): void { if (coordinate.length === 0 || coordinate.some((value) => !Number.isSafeInteger(value) || value < 0)) throw new Error("Vista escalar requer coordenada não negativa."); }
function numericConfig(value: unknown, name: string): number { if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${name} inválido.`); return value; }
