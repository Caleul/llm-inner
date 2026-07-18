import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import type { LiteralDenseStorageDecodeAssignment } from "./literal.js";
import type { DtypePolicy, Operation, ReductionSchedule, TensorRef } from "./types.js";
import type { Gemma4LiteralValueDomain } from "./gemma4-literal-domains.js";
import type { Gemma4LiteralLearnedOperand } from "./gemma4-literal-learned-operands.js";
import {
  requiredGemma4LiteralScalarCalculation,
  type Gemma4LiteralScalarCalculation,
} from "./gemma4-literal-scalar-calculations.js";
import { decodeIeeeBF16ToF32, decodeIeeeF16ToF32 } from "./utils.js";

export interface Gemma4LiteralOperationNavigation {
  operationId: string;
  /** Reused tower definition ID; absent when operationId is already the definition ID. */
  definitionId?: string;
  operation: string;
  scope: "composite" | "vision" | "audio" | "text-prelude" | "text-layer" | "text-epilogue";
  layer?: number;
  /** Composite invocation which instantiates a shared vision/audio/text definition. */
  invocationId?: string;
  ordinal: number;
  output: string;
  outputDomain: Gemma4LiteralValueDomain;
  /** Formula template and complete reduction contract serialized in the JSON artifact. */
  scalarCalculation: Gemma4LiteralScalarCalculation;
  /** Exact learned roles and logical index expressions, when this assignment consumes checkpoint storage. */
  learnedOperands?: Gemma4LiteralLearnedOperand[];
  predecessors: Array<{ input: string; producerOperationId?: string }>;
  consumers: string[];
  previousOperationId?: string;
  nextOperationId?: string;
}

export interface Gemma4LiteralLearnedScalar {
  tensor: string;
  indices: number[];
  rowMajorIndex: number;
  storageDtype: "F32" | "F16" | "BF16";
  storageBitsHex: string;
  decodedF32: number;
  literal: string;
  decoderId: string;
  decoderOperation: LiteralDenseStorageDecodeAssignment["operation"];
}

export interface Gemma4LiteralScalarTerm {
  inputIndex: number;
  input: string;
  learned: Gemma4LiteralLearnedScalar;
  formula: string;
}

export interface Gemma4LiteralScalarView {
  kind: "gemma4-literal-scalar-view";
  sourceCheckpointAccessed: false;
  navigation: Gemma4LiteralOperationNavigation;
  outputCoordinate: number[];
  output: string;
  dtypePolicy: DtypePolicy;
  formula: string;
  scalarAssignments: string[];
  learnedScalars: Gemma4LiteralLearnedScalar[];
  terms?: Gemma4LiteralScalarTerm[];
  reduction?: {
    bounds: { startInclusive: number; endExclusive: number };
    schedule: ReductionSchedule;
    complete: boolean;
    renderedWindow: { startInclusive: number; endExclusive: number };
    omittedTerms: number;
    /** Present when authoritative runtime source selected this schedule. */
    provenance?: Extract<Operation, { op: "linear" }>["reductionProvenance"];
  };
}

export interface Gemma4LiteralScalarViewRequest {
  operationId: string;
  outputCoordinate: number[];
  /** Required for embedding rows because input_ids remains a caller variable. */
  tokenId?: number;
  /** Diagnostic window. Omitting both renders the complete reduction. */
  inputStart?: number;
  inputCount?: number;
}

interface OperationEntry {
  operation: Operation;
  scope: Gemma4LiteralOperationNavigation["scope"];
}

/** Lists the exact dependency-ordered Gemma4Text calculation carried by the artifact. */
export function listGemma4LiteralTextOperations(artifact: OpenGemma4CompositeLiteralArtifact): Gemma4LiteralOperationNavigation[] {
  const entries = operationEntries(artifact);
  const producerByOutput = new Map(entries.map((entry) => [entry.operation.output, entry.operation.id]));
  const consumersByOutput = new Map<string, string[]>();
  for (const entry of entries) {
    for (const input of operationInputs(entry.operation)) {
      const consumers = consumersByOutput.get(input) ?? [];
      consumers.push(entry.operation.id);
      consumersByOutput.set(input, consumers);
    }
  }
  return entries.map((entry, ordinal) => ({
    operationId: entry.operation.id,
    operation: entry.operation.op,
    scope: entry.scope,
    ...(entry.operation.layer === undefined ? {} : { layer: entry.operation.layer }),
    ordinal,
    output: entry.operation.output,
    outputDomain: requiredTextDomain(artifact, entry.operation.id),
    scalarCalculation: requiredGemma4LiteralScalarCalculation(artifact.scalarCalculations, entry.scope, entry.operation.id),
    ...learnedOperandsFor(artifact, entry.scope, entry.operation.id),
    predecessors: operationInputs(entry.operation).map((input) => ({
      input,
      ...(producerByOutput.has(input) ? { producerOperationId: producerByOutput.get(input)! } : {}),
    })),
    consumers: consumersByOutput.get(entry.operation.output) ?? [],
    ...(ordinal === 0 ? {} : { previousOperationId: entries[ordinal - 1]!.operation.id }),
    ...(ordinal + 1 === entries.length ? {} : { nextOperationId: entries[ordinal + 1]!.operation.id }),
  }));
}

function learnedOperandsFor(
  artifact: OpenGemma4CompositeLiteralArtifact,
  scope: Gemma4LiteralOperationNavigation["scope"],
  definitionId: string,
): { learnedOperands?: Gemma4LiteralLearnedOperand[] } {
  const binding = artifact.learnedOperands.assignments.find((candidate) =>
    candidate.scope === scope && candidate.definitionId === definitionId);
  return binding ? { learnedOperands: structuredClone(binding.operands) } : {};
}

function requiredTextDomain(artifact: OpenGemma4CompositeLiteralArtifact, definitionId: string): Gemma4LiteralValueDomain {
  const entry = artifact.calculationDomains.assignments.find((candidate) =>
    (candidate.scope === "text-prelude" || candidate.scope === "text-layer" || candidate.scope === "text-epilogue") && candidate.definitionId === definitionId);
  if (!entry) throw new Error(`${definitionId}: domínio literal Gemma4Text ausente.`);
  return structuredClone(entry.domain);
}

/**
 * Renders one indexed scalar assignment directly from embedded storage. Every
 * learned value referenced by the returned formula is decoded from the JSON
 * payload and substituted as an exact round-tripping numeric literal.
 */
export async function renderGemma4LiteralScalarView(
  artifact: OpenGemma4CompositeLiteralArtifact,
  request: Gemma4LiteralScalarViewRequest,
): Promise<Gemma4LiteralScalarView> {
  assertCoordinate(request.outputCoordinate);
  const entries = operationEntries(artifact);
  const navigation = listGemma4LiteralTextOperations(artifact);
  const selected = navigation.find((entry) => entry.operationId === request.operationId);
  const operation = entries.find((entry) => entry.operation.id === request.operationId)?.operation;
  if (!selected || !operation) throw new Error(`Atribuição Gemma4Text literal não encontrada: ${request.operationId}.`);
  const base = {
    kind: "gemma4-literal-scalar-view" as const,
    sourceCheckpointAccessed: false as const,
    navigation: selected,
    outputCoordinate: [...request.outputCoordinate],
    output: indexed(operation.output, request.outputCoordinate),
    dtypePolicy: structuredClone(operation.dtypePolicy),
  };

  switch (operation.op) {
    case "linear": return renderLinear(artifact, operation, request, base);
    case "embedding": return renderEmbedding(artifact, operation, request, base, false);
    case "per_layer_embedding": return renderEmbedding(artifact, operation, request, base, true);
    case "rms_norm": return renderRmsNorm(artifact, operation, request, base);
    case "tensor_scale": return renderTensorScale(artifact, operation, request, base);
    case "reshape_heads": return renderPlain(operation, base,
      `${base.output} = ${outputCast(operation)}(${indexed(operation.input, reshapeHeadsInputCoordinate(operation, request.outputCoordinate))})`,
      ["BHSD[b,h,s,d] aliases input[b,s,h*headDim+d] without arithmetic."]);
    case "reshape_per_layer": return renderPlain(operation, base,
      `${base.output} = ${outputCast(operation)}(${indexed(operation.input, reshapePerLayerInputCoordinate(operation, request.outputCoordinate))})`,
      ["[B,S,L,D] aliases input[b,s,l*layerWidth+d] without reordering."]);
    case "select_per_layer": return renderPlain(operation, base,
      `${base.output} = ${outputCast(operation)}(${indexed(operation.input, selectPerLayerInputCoordinate(operation, request.outputCoordinate))})`,
      [`The selected layer coordinate is the declared constant ${operation.layerIndex}.`]);
    case "activation": return renderPlain(operation, base,
      `${base.output} = ${outputCast(operation)}(${activationFormula(operation, indexed(operation.input, request.outputCoordinate))})`, []);
    case "elementwise": return renderPlain(operation, base,
      elementwiseFormula(operation, request.outputCoordinate), []);
    case "rotary_embedding": return renderRotary(operation, request.outputCoordinate, base);
    case "scaled_dot_product_attention": return renderAttention(operation, request.outputCoordinate, base);
  }
}

async function renderLinear(
  artifact: OpenGemma4CompositeLiteralArtifact,
  operation: Extract<Operation, { op: "linear" }>,
  request: Gemma4LiteralScalarViewRequest,
  base: Omit<Gemma4LiteralScalarView, "formula" | "scalarAssignments" | "learnedScalars">,
): Promise<Gemma4LiteralScalarView> {
  if (!operation.transposeWeight || operation.weight.shape.length !== 2 || operation.weight.shape[0] !== operation.outFeatures || operation.weight.shape[1] !== operation.inFeatures) {
    throw new Error(`${operation.id}: vista escalar requer weight row-major [out,in] transposto explicitamente.`);
  }
  const outputFeature = last(request.outputCoordinate, operation.id);
  if (outputFeature >= operation.outFeatures) throw new Error(`${operation.id}: coordenada de saída ${outputFeature} excede outFeatures=${operation.outFeatures}.`);
  const window = reductionWindow(request, operation.inFeatures, operation.id);
  const prefix = request.outputCoordinate.slice(0, -1);
  const reduction = requireReduction(operation);
  const learnedScalars: Gemma4LiteralLearnedScalar[] = [];
  const terms: Gemma4LiteralScalarTerm[] = [];
  for (let inputIndex = window.start; inputIndex < window.end; inputIndex += 1) {
    const learned = await readGemma4LiteralLearnedScalar(artifact, operation.weight, [outputFeature, inputIndex]);
    learnedScalars.push(learned);
    const input = indexed(operation.input, [...prefix, inputIndex]);
    terms.push({ inputIndex, input, learned, formula: scalarProductFormula(reduction, inputIndex, input, learned.literal) });
  }
  let bias: Gemma4LiteralLearnedScalar | undefined;
  if (operation.bias) {
    bias = await readGemma4LiteralLearnedScalar(artifact, operation.bias, [outputFeature]);
    learnedScalars.push(bias);
  }
  const complete = window.start === 0 && window.end === operation.inFeatures;
  const scalarAssignments = [
    ...terms.map((term) => term.formula),
    ...reductionAssignments(reduction, operation.inFeatures, operation.dtypePolicy.accumulationDtype),
    `${base.output} = ${outputCast(operation)}(${bias ? `F32(reduced + ${bias.literal})` : "reduced"})`,
  ];
  return {
    ...base,
    formula: `${base.output} = ${outputCast(operation)}(reduce_${reduction.kind}(product[0..${operation.inFeatures - 1}])${bias ? ` + ${bias.literal}` : ""})`,
    scalarAssignments,
    learnedScalars,
    terms,
    reduction: {
      bounds: { startInclusive: 0, endExclusive: operation.inFeatures },
      schedule: structuredClone(reduction),
      complete,
      renderedWindow: { startInclusive: window.start, endExclusive: window.end },
      omittedTerms: operation.inFeatures - (window.end - window.start),
      ...(operation.reductionProvenance ? { provenance: structuredClone(operation.reductionProvenance) } : {}),
    },
  };
}

async function renderEmbedding(
  artifact: OpenGemma4CompositeLiteralArtifact,
  operation: Extract<Operation, { op: "embedding" | "per_layer_embedding" }>,
  request: Gemma4LiteralScalarViewRequest,
  base: Omit<Gemma4LiteralScalarView, "formula" | "scalarAssignments" | "learnedScalars">,
  perLayer: boolean,
): Promise<Gemma4LiteralScalarView> {
  const tokenId = request.tokenId;
  if (!Number.isSafeInteger(tokenId) || tokenId! < 0 || tokenId! >= operation.weight.shape[0]!) {
    throw new Error(`${operation.id}: --token-id válido é obrigatório para substituir a linha de embedding.`);
  }
  const expectedRank = perLayer ? 4 : 3;
  if (request.outputCoordinate.length !== expectedRank) throw new Error(`${operation.id}: coordenada de embedding requer rank ${expectedRank}.`);
  const feature = perLayer && operation.op === "per_layer_embedding"
    ? request.outputCoordinate[2]! * operation.layerWidth + request.outputCoordinate[3]!
    : request.outputCoordinate[2]!;
  if (feature < 0 || feature >= operation.weight.shape[1]!) throw new Error(`${operation.id}: feature de embedding ${feature} fora do shape declarado.`);
  const learned = await readGemma4LiteralLearnedScalar(artifact, operation.weight, [tokenId!, feature]);
  const scale = literal(operation.scale ?? 1);
  const formula = `${base.output} = ${outputCast(operation)}(F32(${learned.literal} * ${scale}))`;
  return { ...base, formula, scalarAssignments: [formula], learnedScalars: [learned] };
}

async function renderRmsNorm(
  artifact: OpenGemma4CompositeLiteralArtifact,
  operation: Extract<Operation, { op: "rms_norm" }>,
  request: Gemma4LiteralScalarViewRequest,
  base: Omit<Gemma4LiteralScalarView, "formula" | "scalarAssignments" | "learnedScalars">,
): Promise<Gemma4LiteralScalarView> {
  const feature = last(request.outputCoordinate, operation.id);
  const width = operation.reductionSize ?? operation.weight?.shape[0];
  if (operation.weightTransform !== "none" && (!operation.weight || operation.weight.shape.length !== 1 || width === undefined || feature >= width)) {
    throw new Error(`${operation.id}: RMSNorm ponderado não possui vetor compatível com a coordenada solicitada.`);
  }
  const learned = operation.weight ? await readGemma4LiteralLearnedScalar(artifact, operation.weight, [feature]) : undefined;
  const multiplier = operation.weightTransform === "none" ? "1" : operation.weightTransform === "one_plus_weight"
    ? `F32(1 + ${learned!.literal})` : learned!.literal;
  const input = indexed(operation.input, request.outputCoordinate);
  const domain = width ?? "last_dimension";
  const formula = `${base.output} = ${outputCast(operation)}(F32(F32(${input} * inv_rms) * ${multiplier}))`;
  return {
    ...base,
    formula,
    scalarAssignments: [
      `square[i] = F32(${indexed(operation.input, [...request.outputCoordinate.slice(0, -1), "i"])} * ${indexed(operation.input, [...request.outputCoordinate.slice(0, -1), "i"])})`,
      ...(operation.dtypePolicy.reduction?.kind === "pytorch-cpu-f32-cascade-sum"
        ? reductionAssignments(operation.dtypePolicy.reduction, typeof domain === "number" ? domain : 0, operation.dtypePolicy.accumulationDtype)
        : [`sum = ${operation.dtypePolicy.accumulationDtype === "F64" ? "F64" : "F32"}(sum_{i=0..${typeof domain === "number" ? domain - 1 : domain} in ascending order}(square[i]))`]),
      `mean = F32(sum / F32(${String(domain)}))`,
      `inv_rms = F32(1 / F32(sqrt(F32(mean + ${literal(operation.epsilon)}))))`,
      formula,
    ],
    learnedScalars: learned ? [learned] : [],
  };
}

async function renderTensorScale(
  artifact: OpenGemma4CompositeLiteralArtifact,
  operation: Extract<Operation, { op: "tensor_scale" }>,
  request: Gemma4LiteralScalarViewRequest,
  base: Omit<Gemma4LiteralScalarView, "formula" | "scalarAssignments" | "learnedScalars">,
): Promise<Gemma4LiteralScalarView> {
  const learned = await readGemma4LiteralLearnedScalar(artifact, operation.scalar, [0]);
  const formula = `${base.output} = ${outputCast(operation)}(F32(${indexed(operation.input, request.outputCoordinate)} * ${learned.literal}))`;
  return { ...base, formula, scalarAssignments: [formula], learnedScalars: [learned] };
}

function renderPlain(
  _operation: Operation,
  base: Omit<Gemma4LiteralScalarView, "formula" | "scalarAssignments" | "learnedScalars">,
  formula: string,
  prologue: string[],
): Gemma4LiteralScalarView {
  return { ...base, formula, scalarAssignments: [...prologue, formula], learnedScalars: [] };
}

export async function readGemma4LiteralLearnedScalar(
  artifact: OpenGemma4CompositeLiteralArtifact,
  reference: TensorRef,
  indices: number[],
): Promise<Gemma4LiteralLearnedScalar> {
  const constant = artifact.constants.get(reference.name);
  const decoder = artifact.storageDecoders.find((entry) => entry.input === `${reference.name}:storage`);
  if (!constant || !decoder) throw new Error(`${reference.name}: constante ou decoder não encontrado no artefato literal.`);
  if (decoder.operation !== "ieee-f32-little-endian" && decoder.operation !== "ieee-f16-to-f32" && decoder.operation !== "ieee-bf16-to-f32") {
    throw new Error(`${reference.name}: vista escalar densa não suporta decoder ${decoder.operation}.`);
  }
  if (constant.quantization || constant.layout !== "row-major" || !sameShape(constant.logicalShape, reference.shape) || !sameShape(constant.storageShape, reference.shape) || indices.length !== reference.shape.length) {
    throw new Error(`${reference.name}: vista escalar exige storage denso row-major com shape idêntico.`);
  }
  let rowMajorIndex = 0;
  for (let axis = 0; axis < indices.length; axis += 1) {
    const index = indices[axis]!, dimension = reference.shape[axis]!;
    if (!Number.isSafeInteger(index) || index < 0 || index >= dimension) throw new Error(`${reference.name}: índice ${index} fora do eixo ${axis} de tamanho ${dimension}.`);
    rowMajorIndex = rowMajorIndex * dimension + index;
  }
  const byteWidth = constant.storageDtype === "F32" ? 4 : constant.storageDtype === "F16" || constant.storageDtype === "BF16" ? 2 : 0;
  if (byteWidth === 0) throw new Error(`${reference.name}: dtype denso não suportado na vista escalar: ${constant.storageDtype}.`);
  const bytes = await artifact.readTensorBytesRange({
    name: constant.name,
    storageDtype: constant.storageDtype as "F32" | "F16" | "BF16",
    storageShape: constant.storageShape,
    logicalShape: constant.logicalShape,
  }, rowMajorIndex * byteWidth, byteWidth);
  if (bytes.length !== byteWidth) throw new Error(`${reference.name}: leitura escalar retornou ${bytes.length} bytes; esperados ${byteWidth}.`);
  const bits = byteWidth === 4 ? bytes.readUInt32LE(0) : bytes.readUInt16LE(0);
  const decodedF32 = constant.storageDtype === "F32" ? bytes.readFloatLE(0)
    : constant.storageDtype === "F16" ? decodeIeeeF16ToF32(bits) : decodeIeeeBF16ToF32(bits);
  if (!Number.isFinite(decodedF32)) throw new Error(`${reference.name}: scalar não finito em [${indices.join(",")}].`);
  return {
    tensor: reference.name,
    indices: [...indices],
    rowMajorIndex,
    storageDtype: constant.storageDtype as "F32" | "F16" | "BF16",
    storageBitsHex: `0x${bits.toString(16).padStart(byteWidth * 2, "0")}`,
    decodedF32,
    literal: literal(decodedF32),
    decoderId: decoder.id,
    decoderOperation: decoder.operation,
  };
}

function operationEntries(artifact: OpenGemma4CompositeLiteralArtifact): OperationEntry[] {
  return [
    ...artifact.program.textProgram.prelude.map((operation) => ({ operation, scope: "text-prelude" as const })),
    ...artifact.program.textProgram.layers.flatMap((layer) => layer.operations.map((operation) => ({ operation, scope: "text-layer" as const }))),
    ...artifact.program.textProgram.epilogue.map((operation) => ({ operation, scope: "text-epilogue" as const })),
  ];
}

function operationInputs(operation: Operation): string[] {
  switch (operation.op) {
    case "embedding": case "per_layer_embedding": return [operation.tokenInput];
    case "rms_norm": case "linear": case "reshape_heads": case "reshape_per_layer": case "select_per_layer": case "activation": case "tensor_scale": return [operation.input];
    case "rotary_embedding": return [operation.input, operation.positionInput];
    case "scaled_dot_product_attention": return [operation.query, operation.key, operation.value, operation.maskInput];
    case "elementwise": return operation.inputs;
  }
}

function requireReduction(operation: Extract<Operation, { op: "linear" }>): ReductionSchedule {
  const reduction = operation.dtypePolicy.reduction;
  if (!reduction) throw new Error(`${operation.id}: vista escalar exige agenda de redução declarada.`);
  return reduction;
}

function reductionAssignments(schedule: ReductionSchedule, width: number, accumulationDtype: string | undefined): string[] {
  switch (schedule.kind) {
    case "pytorch-cpu-f32-cascade-sum": return [
      `four ADVSIMD lanes across four ILP registers consume ${width} contiguous square terms in 16-coordinate units`,
      `four-level F32 cascade uses levelStep=max(${schedule.minimumLevelStep}, 2^(ceil(log2(${width}/16))/4)); registers then lanes fold ${schedule.registerFold}/${schedule.laneFold}`,
    ];
    case "pytorch-cpu-bf16-welford": return [
      `${schedule.inputVectorLanes} BF16 lanes widen into two ${schedule.accumulatorVectorLanes}-lane F32 vectors; each performs Welford updates over at most ${schedule.chunkVectors} input vectors`,
      `low/high vectors merge ${schedule.vectorMergeOrder}; ${schedule.accumulatorVectorLanes} moment lanes fold ${schedule.laneFold}; second pass is ${schedule.secondPass}`,
    ];
    case "ordered-scalar": {
      const accumulator = accumulationDtype === "F64" ? "F64" : accumulationDtype === "F32" ? "F32" : undefined;
      if (!accumulator) throw new Error(`ordered-scalar requer accumulationDtype F32 ou F64; recebeu ${accumulationDtype ?? "missing"}.`);
      return [
      "acc[-1] = 0",
      `acc[i] = ${accumulator}(acc[i-1] + product[i]), i=0..${width - 1} in ascending order`,
      `reduced = acc[${width - 1}]`,
      ];
    }
    case "ordered-fma": return [
      "acc[-1] = 0",
      `acc[i] = F32(acc[i-1] + product[i]), i=0..${width - 1} in ascending order; product[i] remains exact until this boundary`,
      `reduced = acc[${width - 1}]`,
    ];
    case "arm-neon-bf16-dot-fma": return [
      `register[r,lane] = 0 for r=0..7, lane=0..${schedule.lanesPerRegister - 1}`,
      `for i=0..${width - 1}: r=floor((i mod ${schedule.laneCount})/${schedule.lanesPerRegister}); lane=i mod ${schedule.lanesPerRegister}; register[r,lane]=F32(register[r,lane] + product[i]); product[i] remains exact until this boundary`,
      "register[0]+=register[4]; register[1]+=register[5]; register[2]+=register[6]; register[3]+=register[7]; register[0]+=register[2]; register[1]+=register[3]; register[0]+=register[1], each lane addition rounded F32",
      `reduced = horizontal_${schedule.horizontalFold}_F32(register[0,0..${schedule.lanesPerRegister - 1}])`,
    ];
    case "arm-neon-bf16-bfdot-fma": return [
      "register[r,lane] = 0 for r=0..7, lane=0..3",
      "for each adjacent pair (i0,i1): pair_acc=F32(register[r,lane] + product[i0]); register[r,lane]=F32(pair_acc + product[i1]); each product remains exact until its declared add",
      "register tree: 0+=4; 1+=5; 2+=6; 3+=7; 0+=2; 1+=3; 0+=1, each lane addition rounded F32",
      `reduced = horizontal_${schedule.horizontalFold}_F32(register[0,0..3]) after all ${width} terms`,
    ];
    case "blocked-f32-terms": return [
      `partial[b] = ${schedule.productBoundary === "fused-fma" ? "F32_FMA chain" : "ordered F32 products/adds"} over contiguous ${schedule.termsPerBlock}-term block in ${schedule.termOrder} order`,
      "reduced = F32 sum of partial[b] in ascending block order",
    ];
    case "interleaved-f32-lanes": case "interleaved-fma-lanes": return [
      `lane[l]=0 for l=0..${schedule.laneCount - 1}; input i updates lane[i mod ${schedule.laneCount}] with ${schedule.kind === "interleaved-fma-lanes" ? "F32_FMA" : "F32 add of an F32 product"}`,
      `reduced = ${schedule.laneReductionOrder} F32 fold of all lanes`,
    ];
    case "tiled-f32-lanes": case "tiled-fma-lanes": return [
      `lane groups consume ${schedule.termsPerLane} contiguous terms per lane across ${schedule.laneCount} persistent lanes; updates use ${schedule.kind === "tiled-fma-lanes" ? "F32_FMA" : "F32 products/adds"}`,
      `reduced = ${schedule.laneReductionOrder} F32 fold of all lanes`,
    ];
    case "blocked-tiled-f32-lanes": return [
      `each block resets ${schedule.laneCount} lanes and consumes ${schedule.termsPerLane} contiguous terms per lane with ${schedule.productBoundary}`,
      `each block uses a ${schedule.laneReductionOrder} F32 lane fold; reduced is the ascending F32 fold of block results`,
    ];
  }
}

function scalarProductFormula(schedule: ReductionSchedule, inputIndex: number, input: string, learnedLiteral: string): string {
  const fused = schedule.kind === "ordered-fma" || schedule.kind === "interleaved-fma-lanes" || schedule.kind === "tiled-fma-lanes" ||
    schedule.kind === "arm-neon-bf16-dot-fma" || schedule.kind === "arm-neon-bf16-bfdot-fma" ||
    (schedule.kind === "blocked-f32-terms" || schedule.kind === "blocked-tiled-f32-lanes") && schedule.productBoundary === "fused-fma";
  return `product[${inputIndex}] = ${fused ? "exact_product" : "F32"}(${input} * ${learnedLiteral})`;
}

function reductionWindow(request: Gemma4LiteralScalarViewRequest, width: number, operationId: string): { start: number; end: number } {
  if ((request.inputStart === undefined) !== (request.inputCount === undefined)) throw new Error(`${operationId}: inputStart e inputCount devem ser fornecidos juntos.`);
  const start = request.inputStart ?? 0, count = request.inputCount ?? width;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(count) || start < 0 || count <= 0 || start + count > width) {
    throw new Error(`${operationId}: janela de redução ${start}+${count} fora de 0..${width}.`);
  }
  return { start, end: start + count };
}

function reshapeHeadsInputCoordinate(operation: Extract<Operation, { op: "reshape_heads" }>, coordinate: number[]): number[] {
  if (coordinate.length !== 4) throw new Error(`${operation.id}: BHSD requer coordenada [b,h,s,d].`);
  const [b, h, s, d] = coordinate;
  if (h! >= operation.numHeads || d! >= operation.headDim) throw new Error(`${operation.id}: coordenada BHSD fora dos heads declarados.`);
  return [b!, s!, h! * operation.headDim + d!];
}

function reshapePerLayerInputCoordinate(operation: Extract<Operation, { op: "reshape_per_layer" }>, coordinate: number[]): number[] {
  if (coordinate.length !== 4) throw new Error(`${operation.id}: reshape PLE requer coordenada [b,s,l,d].`);
  const [b, s, l, d] = coordinate;
  if (l! >= operation.numLayers || d! >= operation.layerWidth) throw new Error(`${operation.id}: coordenada PLE fora do shape declarado.`);
  return [b!, s!, l! * operation.layerWidth + d!];
}

function selectPerLayerInputCoordinate(operation: Extract<Operation, { op: "select_per_layer" }>, coordinate: number[]): number[] {
  if (coordinate.length !== 3 || coordinate[2]! >= operation.layerWidth) throw new Error(`${operation.id}: select PLE requer coordenada [b,s,d].`);
  return [coordinate[0]!, coordinate[1]!, operation.layerIndex, coordinate[2]!];
}

function activationFormula(operation: Extract<Operation, { op: "activation" }>, input: string): string {
  if (operation.function === "gelu" && operation.approximation === "tanh") {
    return `F32(F32(0.5 * ${input}) * F32(1 + F32(tanh(F32(sqrt(2/pi) * F32(${input} + F32(0.044715 * F32(${input} * F32(${input} * ${input})))))))))`;
  }
  throw new Error(`${operation.id}: ativação ${operation.function}/${operation.approximation ?? "exact"} sem vista escalar registrada.`);
}

function elementwiseFormula(operation: Extract<Operation, { op: "elementwise" }>, coordinate: number[]): string {
  const inputs = operation.inputs.map((input) => indexed(input, coordinate));
  if (operation.kind === "add") return `${indexed(operation.output, coordinate)} = ${outputCast(operation)}(F32(${inputs[0]} + ${inputs[1]}))`;
  if (operation.kind === "multiply") return `${indexed(operation.output, coordinate)} = ${outputCast(operation)}(F32(${inputs[0]} * ${inputs[1]}))`;
  if (operation.kind === "scale") return `${indexed(operation.output, coordinate)} = ${outputCast(operation)}(F32(${inputs[0]} * ${literal(operation.scalar!)}))`;
  if (operation.tanhSoftcapCasts) {
    return `${indexed(operation.output, coordinate)} = BF16(F32(${literal(operation.scalar!)} * BF16(SLEEF_TANH_F32(BF16(F32(${inputs[0]} / ${literal(operation.scalar!)}))))))`;
  }
  return `${indexed(operation.output, coordinate)} = ${outputCast(operation)}(F32(${literal(operation.scalar!)} * tanh(F32(${inputs[0]} / ${literal(operation.scalar!)}))))`;
}

function renderRotary(
  operation: Extract<Operation, { op: "rotary_embedding" }>,
  coordinate: number[],
  base: Omit<Gemma4LiteralScalarView, "formula" | "scalarAssignments" | "learnedScalars">,
): Gemma4LiteralScalarView {
  if (coordinate.length !== 4 || operation.layout !== "rotate_half") throw new Error(`${operation.id}: RoPE rotate_half requer coordenada [b,h,s,d].`);
  const d = coordinate[3]!;
  if (d >= operation.rotaryDim) return renderPlain(operation, base, `${base.output} = ${outputCast(operation)}(${indexed(operation.input, coordinate)})`, []);
  const half = operation.rotaryDim / 2, pair = d % half;
  const partial = operation.ropeType === "proportional" ? operation.scaling?.partial_rotary_factor : 1;
  const factor = operation.ropeType === "proportional" ? operation.scaling?.factor ?? 1 : 1;
  if (typeof partial !== "number" || !Number.isFinite(partial) || partial <= 0 || partial > 1 || typeof factor !== "number" || !Number.isFinite(factor) || factor <= 0) {
    throw new Error(`${operation.id}: RoPE proporcional não possui partial_rotary_factor/factor escalar válido.`);
  }
  const activePairs = Math.floor(partial * operation.rotaryDim / 2);
  const paired = d < half ? d + half : d - half;
  const sign = d < half ? "-" : "+";
  const angle = pair >= activePairs ? "0" : `F32(position_ids[${coordinate[0]},${coordinate[2]}] / F32(F32(${literal(operation.theta)}^${literal((2 * pair) / operation.rotaryDim)}) * F32(${literal(factor)})))`;
  const formula = operation.rotaryCasts
    ? `${base.output} = BF16(F32(BF16(F32(${indexed(operation.input, coordinate)} * cosine)) ${sign} BF16(F32(${indexed(operation.input, [...coordinate.slice(0, 3), paired])} * sine))))`
    : `${base.output} = ${outputCast(operation)}(F32(F32(${indexed(operation.input, coordinate)} * cosine) ${sign} F32(${indexed(operation.input, [...coordinate.slice(0, 3), paired])} * sine)))`;
  return renderPlain(operation, base, formula, [
    `angle = ${angle}`,
    ...(operation.trigImplementation ? [
      `range_reduction = abs(angle) < ${operation.trigImplementation.argumentReduction.fastRangeMaxExclusive} ? SLEEF_CODY_WAITE_F32(angle) : SLEEF_REMPIF_F32(angle, inline_f32_le_table_sha256=${operation.trigImplementation.argumentReduction.tablePayloadSha256})`,
      `trig_polynomial = ${operation.trigImplementation.reducedPolynomial.evaluation}; coefficients_ascending=[${operation.trigImplementation.reducedPolynomial.coefficientsAscending.map(literal).join(",")}]`,
    ] : []),
    operation.trigImplementation ? "cosine = BF16(SLEEF_COS_F32(angle))" : "cosine = F32(cos(angle))",
    operation.trigImplementation ? "sine = BF16(SLEEF_SIN_F32(angle))" : "sine = F32(sin(angle))",
  ]);
}

function renderAttention(
  operation: Extract<Operation, { op: "scaled_dot_product_attention" }>,
  coordinate: number[],
  base: Omit<Gemma4LiteralScalarView, "formula" | "scalarAssignments" | "learnedScalars">,
): Gemma4LiteralScalarView {
  if (coordinate.length !== 3) throw new Error(`${operation.id}: saída de atenção requer coordenada compactada [b,q,h*headDim+d].`);
  const [b, q, merged] = coordinate;
  if (merged! >= operation.numAttentionHeads * operation.headDim) throw new Error(`${operation.id}: feature compactada de atenção fora do shape declarado.`);
  const h = Math.floor(merged! / operation.headDim), d = merged! % operation.headDim;
  const kv = Math.floor(h / (operation.numAttentionHeads / operation.numKeyValueHeads));
  const topology = operation.slidingWindow === undefined
    ? `k=0..min(past_length+${q},key_length-1)`
    : `k=max(0,past_length+${q}-${operation.slidingWindow - 1})..min(past_length+${q},key_length-1)`;
  const softcap = operation.scoreSoftcap === undefined ? "scaled_dot[k]" : `F32(${literal(operation.scoreSoftcap)} * F32(tanh(F32(scaled_dot[k] / F32(${literal(operation.scoreSoftcap)})))))`;
  if (operation.numericImplementation) {
    const implementation = operation.numericImplementation;
    const allKeys = "k=0..key_length-1";
    const formula = `${base.output} = BF16_RNE(ARM_NEON_BF16_DOT_F32_${implementation.contextReduction.horizontalFold}(probability[k], ${operation.value}[${b},${kv},k,${d}], ${allKeys}))`;
    return renderPlain(operation, base, formula, [
      `dot[k] = BF16_RNE(ARM_NEON_BF16_DOT_F32_${implementation.scoreReduction.horizontalFold}(${operation.query}[${b},${h},${q},feature], ${operation.key}[${b},${kv},k,feature], feature=0..${operation.headDim - 1}))`,
      `scaled_dot[k] = BF16_RNE(F32(dot[k] * F32(${literal(operation.scale)})))`,
      `topology_mask[k] = 0 when ${topology}; otherwise -Infinity`,
      `declared_mask[k] = ${operation.maskInput}[${b},${h},${q},k]`,
      "effective_mask[k] = declared_mask[k] when it already defines topology; otherwise F32(topology_mask[k] + declared_mask[k])",
      "score[k] = BF16_RNE(F32(scaled_dot[k] + effective_mask[k]))",
      `maximum = PYTORCH_F32_VECTOR_MAX_PAIRWISE(score[k], ${allKeys}, lanes=${implementation.softmaxVectorLanes})`,
      `exp_score[k] = SLEEF_EXP_F32(F32(score[k] - maximum))`,
      `total = PYTORCH_F32_VECTOR_SUM_PAIRWISE(exp_score[k], ${allKeys}, lanes=${implementation.softmaxVectorLanes})`,
      "probability[k] = BF16_RNE(F32(exp_score[k] * F32(1 / total)))",
    ]);
  }
  const formula = `${base.output} = ${outputCast(operation)}(sum_{${topology} in ascending order}(F32(probability[k] * ${operation.value}[${b},${kv},k,${d}])))`;
  return renderPlain(operation, base, formula, [
    `dot[k] = F32(sum_{feature=0..${operation.headDim - 1} in ascending order}(F32(${operation.query}[${b},${h},${q},feature] * ${operation.key}[${b},${kv},k,feature])))`,
    `scaled_dot[k] = F32(dot[k] * F32(${literal(operation.scale)}))`,
    `unmasked[k] = ${softcap}`,
    `score[k] = F32(unmasked[k] + ${operation.maskInput}[${b},${h},${q},k])`,
    `exp_score[k] = F32(exp(F32(score[k] - F32(max_{${topology}}(score)))))`,
    `total = F32(sum_{${topology} in ascending order}(exp_score[k]))`,
    "probability[k] = F32(exp_score[k] / total)",
  ]);
}

function outputCast(operation: Operation): string {
  if (operation.dtypePolicy.outputDtype === "BF16") return "BF16_RNE";
  if (operation.dtypePolicy.outputDtype === "F32") return "F32";
  throw new Error(`${operation.id}: outputDtype '${operation.dtypePolicy.outputDtype ?? "missing"}' sem cast escalar registrado.`);
}

function indexed(name: string, coordinate: ReadonlyArray<number | string>): string { return `${name}[${coordinate.join(",")}]`; }
function literal(value: number): string { return Object.is(value, -0) ? "-0" : Number(value).toString(); }
function last(coordinate: number[], operationId: string): number { if (coordinate.length === 0) throw new Error(`${operationId}: coordenada de saída não pode ser vazia.`); return coordinate.at(-1)!; }
function assertCoordinate(coordinate: number[]): void { if (coordinate.length === 0 || coordinate.some((value) => !Number.isSafeInteger(value) || value < 0)) throw new Error("Vista escalar requer coordenada de saída não vazia com inteiros não negativos."); }
function sameShape(left: readonly number[], right: readonly number[]): boolean { return left.length === right.length && left.every((value, index) => value === right[index]); }
