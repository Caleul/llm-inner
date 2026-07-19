import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { listGemma4LiteralOperations } from "./gemma4-literal-multimodal-scalar-view.js";
import type { Gemma4LiteralScalarCalculation } from "./gemma4-literal-scalar-calculations.js";
import type {
  Gemma4LiteralOperationNavigation,
  Gemma4LiteralScalarViewRequest,
} from "./gemma4-literal-scalar-view.js";

export type Gemma4LiteralRuntimeReductionOperationClass =
  | "vision-attention-score"
  | "vision-attention-value"
  | "audio-content-attention-score"
  | "audio-position-attention-score"
  | "audio-attention-value";

export interface Gemma4LiteralRuntimeReductionOperation {
  operationId: string;
  definitionId: string;
  invocationId?: string;
  operation: string;
  operationClass: Gemma4LiteralRuntimeReductionOperationClass;
  output: string;
  outputDomain: Gemma4LiteralOperationNavigation["outputDomain"];
  reductionDomain: NonNullable<Gemma4LiteralScalarCalculation["reduction"]>["domains"][number];
  provider: "Apple Accelerate SGEMM";
  scalarSchedule: "unpublished-fail-closed";
  auditability: "operand-products-addressable-reduction-fail-closed";
}

export interface Gemma4LiteralRuntimeReductionProductTerm {
  reductionIndex: number;
  leftOperand: string;
  rightOperand: string;
  predicate?: string;
  mathematicalProduct: string;
}

/**
 * A deliberately non-executable audit of one address in a native BMM.
 * It exposes every operand address and padding predicate without inventing the
 * provider's unpublished product/accumulation rounding tree.
 */
export interface Gemma4LiteralRuntimeReductionAudit {
  kind: "gemma4-literal-runtime-reduction-product-audit";
  schemaVersion: 1;
  sourceCheckpointAccessed: false;
  navigation: Gemma4LiteralOperationNavigation;
  operationClass: Gemma4LiteralRuntimeReductionOperationClass;
  outputCoordinate: number[];
  output: string;
  status: "fail-closed-runtime-reduction";
  coordinateAssignments: string[];
  termTemplate: {
    reductionIndex: string;
    leftOperand: string;
    rightOperand: string;
    predicate?: string;
    mathematicalProduct: string;
  };
  terms: Gemma4LiteralRuntimeReductionProductTerm[];
  reduction: {
    index: string;
    domain: NonNullable<Gemma4LiteralScalarCalculation["reduction"]>["domains"][number];
    renderedWindow: { startInclusive: number; endExclusive: number };
    complete: boolean;
    omittedTerms?: number;
    provider: "Apple Accelerate SGEMM";
    scalarSchedule: "unpublished-fail-closed";
    productRounding: "unpublished-provider-boundary";
    accumulationOrder: "unpublished-provider-boundary";
    outputCast: string;
  };
  nonExecutableResult: string;
}

interface RuntimeReductionOperands {
  index: "head_feature" | "key_patch" | "key_slot";
  /** Undefined when the extent depends on a caller-provided tensor axis. */
  extent?: number;
  coordinateAssignments: string[];
  left: (index: number | string) => string;
  right: (index: number | string) => string;
  predicate?: (index: number | string) => string;
}

/** Lists the complete operation-class-dispatched native BMM boundary. */
export function listGemma4LiteralRuntimeReductionOperations(
  artifact: OpenGemma4CompositeLiteralArtifact,
): Gemma4LiteralRuntimeReductionOperation[] {
  const declaredClasses = new Set(artifact.authoritativeExecution.unresolvedNativeReduction.operationClasses);
  const operations = listGemma4LiteralOperations(artifact).flatMap((navigation): Gemma4LiteralRuntimeReductionOperation[] => {
    const reduction = navigation.scalarCalculation.reduction;
    if (reduction?.order !== "runtime-defined") return [];
    const operationClass = runtimeReductionOperationClass(navigation.scope, navigation.operation);
    if (!declaredClasses.has(operationClass)) {
      throw new Error(`${navigation.operationId}: classe BMM ${operationClass} ausente do contrato autoritativo.`);
    }
    if (reduction.domains.length !== 1) {
      throw new Error(`${navigation.operationId}: BMM runtime-defined requer exatamente um domínio de redução.`);
    }
    return [{
      operationId: navigation.operationId,
      definitionId: navigation.definitionId ?? navigation.operationId,
      ...(navigation.invocationId ? { invocationId: navigation.invocationId } : {}),
      operation: navigation.operation,
      operationClass,
      output: navigation.output,
      outputDomain: structuredClone(navigation.outputDomain),
      reductionDomain: structuredClone(reduction.domains[0]!),
      provider: artifact.authoritativeExecution.unresolvedNativeReduction.provider,
      scalarSchedule: artifact.authoritativeExecution.unresolvedNativeReduction.scalarSchedule,
      auditability: "operand-products-addressable-reduction-fail-closed",
    }];
  });
  const encounteredClasses = new Set(operations.map((operation) => operation.operationClass));
  for (const operationClass of declaredClasses) {
    if (!encounteredClasses.has(operationClass)) {
      throw new Error(`Contrato autoritativo declara ${operationClass}, mas nenhuma atribuição runtime-defined compatível foi encontrada.`);
    }
  }
  return operations;
}

/**
 * Renders a concrete operand window for any of the five compatible Gemma 4
 * native-BMM classes. Strict scalar rendering still rejects the same operation:
 * this view ends before the unknown provider reduction and cannot yield output.
 */
export function renderGemma4LiteralRuntimeReductionAudit(
  artifact: OpenGemma4CompositeLiteralArtifact,
  request: Gemma4LiteralScalarViewRequest,
): Gemma4LiteralRuntimeReductionAudit {
  assertCoordinate(request.outputCoordinate);
  const navigation = listGemma4LiteralOperations(artifact).find((candidate) => candidate.operationId === request.operationId);
  if (!navigation) throw new Error(`Atribuição Gemma 4 literal não encontrada: ${request.operationId}.`);
  const reduction = navigation.scalarCalculation.reduction;
  if (reduction?.order !== "runtime-defined" || navigation.scalarCalculation.reproducibility !== "fail-closed-runtime-reduction") {
    throw new Error(`${request.operationId}: operação não possui redução runtime-defined para auditoria fail-closed.`);
  }
  if (reduction.domains.length !== 1) throw new Error(`${request.operationId}: auditoria BMM requer um único domínio de redução.`);
  const operationClass = runtimeReductionOperationClass(navigation.scope, navigation.operation);
  if (!artifact.authoritativeExecution.unresolvedNativeReduction.operationClasses.includes(operationClass)) {
    throw new Error(`${request.operationId}: classe ${operationClass} não pertence à fronteira autoritativa incorporada.`);
  }
  const rendered = runtimeReductionOperands(artifact, navigation, request.outputCoordinate);
  const dynamicExtent = rendered.extent === undefined;
  if (dynamicExtent && (request.inputStart === undefined || request.inputCount === undefined)) {
    throw new Error(`${request.operationId}: redução dinâmica requer inputStart/inputCount para selecionar uma janela auditável.`);
  }
  const window = dynamicExtent
    ? explicitDynamicReductionWindow(request, request.operationId)
    : reductionWindow(request, rendered.extent!, request.operationId);
  const terms = Array.from({ length: window.end - window.start }, (_, offset) => {
    const reductionIndex = window.start + offset;
    const leftOperand = rendered.left(reductionIndex), rightOperand = rendered.right(reductionIndex);
    const predicate = rendered.predicate?.(reductionIndex);
    return {
      reductionIndex,
      leftOperand,
      rightOperand,
      ...(predicate ? { predicate } : {}),
      mathematicalProduct: predicate
        ? `term[${reductionIndex}] = ${predicate} ? REAL_PRODUCT(${leftOperand} * ${rightOperand}) : REAL(0)`
        : `term[${reductionIndex}] = REAL_PRODUCT(${leftOperand} * ${rightOperand})`,
    };
  });
  const output = indexed(navigation.output, request.outputCoordinate);
  const outputCast = navigation.scalarCalculation.dtypePolicy.outputDtype ?? "operation-declared";
  const templateLeft = rendered.left(rendered.index), templateRight = rendered.right(rendered.index);
  const templatePredicate = rendered.predicate?.(rendered.index);
  return {
    kind: "gemma4-literal-runtime-reduction-product-audit",
    schemaVersion: 1,
    sourceCheckpointAccessed: false,
    navigation,
    operationClass,
    outputCoordinate: [...request.outputCoordinate],
    output,
    status: "fail-closed-runtime-reduction",
    coordinateAssignments: rendered.coordinateAssignments,
    termTemplate: {
      reductionIndex: rendered.index,
      leftOperand: templateLeft,
      rightOperand: templateRight,
      ...(templatePredicate ? { predicate: templatePredicate } : {}),
      mathematicalProduct: templatePredicate
        ? `term[${rendered.index}] = ${templatePredicate} ? REAL_PRODUCT(${templateLeft} * ${templateRight}) : REAL(0)`
        : `term[${rendered.index}] = REAL_PRODUCT(${templateLeft} * ${templateRight})`,
    },
    terms,
    reduction: {
      index: rendered.index,
      domain: structuredClone(reduction.domains[0]!),
      renderedWindow: { startInclusive: window.start, endExclusive: window.end },
      complete: !dynamicExtent && window.start === 0 && window.end === rendered.extent,
      ...(!dynamicExtent ? { omittedTerms: rendered.extent! - terms.length } : {}),
      provider: artifact.authoritativeExecution.unresolvedNativeReduction.provider,
      scalarSchedule: artifact.authoritativeExecution.unresolvedNativeReduction.scalarSchedule,
      productRounding: "unpublished-provider-boundary",
      accumulationOrder: "unpublished-provider-boundary",
      outputCast,
    },
    nonExecutableResult: `${output} = ${outputCast}(APPLE_ACCELERATE_SGEMM_UNPUBLISHED_REDUCTION(term[complete declared domain])); intentionally unavailable until one authoritative class-wide scalar schedule is proven`,
  };
}

function runtimeReductionOperationClass(
  scope: Gemma4LiteralOperationNavigation["scope"],
  operation: string,
): Gemma4LiteralRuntimeReductionOperationClass {
  if (scope === "vision" && operation === "attention-score-matmul") return "vision-attention-score";
  if (scope === "vision" && operation === "attention-value-matmul") return "vision-attention-value";
  if (scope === "audio" && operation === "chunked-attention-content-matmul") return "audio-content-attention-score";
  if (scope === "audio" && operation === "relative-attention-position-matmul") return "audio-position-attention-score";
  if (scope === "audio" && operation === "chunked-relative-attention-values") return "audio-attention-value";
  throw new Error(`Redução runtime-defined inesperada em ${scope}:${operation}; nenhuma semântica BMM pode ser inferida.`);
}

function runtimeReductionOperands(
  artifact: OpenGemma4CompositeLiteralArtifact,
  navigation: Gemma4LiteralOperationNavigation,
  coordinate: number[],
): RuntimeReductionOperands {
  const inputs = navigation.scalarCalculation.orderedInputs;
  if (inputs.length !== 2) throw new Error(`${navigation.operationId}: BMM requer exatamente dois orderedInputs.`);
  const [left, right] = inputs as [string, string];
  const tower = artifact.program.audioProgram.tower;
  switch (runtimeReductionOperationClass(navigation.scope, navigation.operation)) {
    case "vision-attention-score": {
      requireCoordinateRank(coordinate, 4, navigation.operationId, "[batch,head,query_patch,key_patch]");
      const [batch, head, query, key] = coordinate;
      if (head! >= artifact.program.visionProgram.tower.attentionHeads) throw new Error(`${navigation.operationId}: head ${head} fora do domínio.`);
      return {
        index: "head_feature", extent: artifact.program.visionProgram.tower.headDim, coordinateAssignments: [],
        left: (feature) => indexed(left, [batch!, head!, query!, feature]),
        right: (feature) => indexed(right, [batch!, head!, key!, feature]),
      };
    }
    case "vision-attention-value": {
      requireCoordinateRank(coordinate, 3, navigation.operationId, "[batch,query_patch,hidden]");
      const [batch, query, hidden] = coordinate, headDim = artifact.program.visionProgram.tower.headDim;
      if (hidden! >= artifact.program.visionProgram.tower.hiddenSize) throw new Error(`${navigation.operationId}: hidden ${hidden} fora do domínio.`);
      const head = Math.floor(hidden! / headDim), headFeature = hidden! % headDim;
      return {
        index: "key_patch",
        coordinateAssignments: [`head=floor(${hidden}/${headDim})=${head}`, `head_feature=${hidden}%${headDim}=${headFeature}`],
        left: (key) => indexed(left, [batch!, head, query!, key]),
        right: (key) => indexed(right, [batch!, head, key, headFeature]),
      };
    }
    case "audio-content-attention-score": {
      requireCoordinateRank(coordinate, 5, navigation.operationId, "[batch,head,block,query_in_block,key_slot]");
      const [batch, head, block, query, keySlot] = coordinate;
      assertAudioAttentionCoordinate(artifact, navigation.operationId, head!, query!, keySlot!, audioContext(artifact));
      const queryIndex = block! * tower.attentionChunkSize + query!;
      const keyIndex = block! * tower.attentionChunkSize - (tower.attentionContextLeft - 1) + keySlot!;
      const predicate = `query_index<${left}.shape[1] && 0<=key_index && key_index<${right}.shape[1]`;
      return {
        index: "head_feature", extent: tower.headDim,
        coordinateAssignments: [`query_index=${queryIndex}`, `key_index=${keyIndex}`],
        left: (feature) => indexed(left, [batch!, queryIndex, `${head}*${tower.headDim}+${feature}`]),
        right: (feature) => indexed(right, [batch!, keyIndex, `${head}*${tower.headDim}+${feature}`]),
        predicate: () => predicate,
      };
    }
    case "audio-position-attention-score": {
      requireCoordinateRank(coordinate, 5, navigation.operationId, "[batch,head,block,query_in_block,relative_position]");
      const [batch, head, block, query, relative] = coordinate, relativeLength = Math.floor(audioContext(artifact) / 2) + 1;
      if (head! >= tower.attentionHeads || query! >= tower.attentionChunkSize || relative! >= relativeLength) {
        throw new Error(`${navigation.operationId}: coordenada de score posicional fora do domínio.`);
      }
      const queryIndex = block! * tower.attentionChunkSize + query!, predicate = `query_index<${left}.shape[1]`;
      return {
        index: "head_feature", extent: tower.headDim, coordinateAssignments: [`query_index=${queryIndex}`],
        left: (feature) => indexed(left, [batch!, queryIndex, `${head}*${tower.headDim}+${feature}`]),
        right: (feature) => indexed(right, [0, relative!, `${head}*${tower.headDim}+${feature}`]),
        predicate: () => predicate,
      };
    }
    case "audio-attention-value": {
      requireCoordinateRank(coordinate, 3, navigation.operationId, "[batch,frame,hidden]");
      const [batch, frame, hidden] = coordinate;
      if (hidden! >= tower.hiddenSize) throw new Error(`${navigation.operationId}: hidden ${hidden} fora do domínio.`);
      const head = Math.floor(hidden! / tower.headDim), headFeature = hidden! % tower.headDim;
      const block = Math.floor(frame! / tower.attentionChunkSize), query = frame! % tower.attentionChunkSize;
      return {
        index: "key_slot", extent: audioContext(artifact),
        coordinateAssignments: [
          `head=floor(${hidden}/${tower.headDim})=${head}`,
          `head_feature=${hidden}%${tower.headDim}=${headFeature}`,
          `block=floor(${frame}/${tower.attentionChunkSize})=${block}`,
          `query_in_block=${frame}%${tower.attentionChunkSize}=${query}`,
          `key_index(key_slot)=${block}*${tower.attentionChunkSize}-${tower.attentionContextLeft - 1}+key_slot`,
        ],
        left: (keySlot) => indexed(left, [batch!, head, block, query, keySlot]),
        right: (keySlot) => indexed(right, [batch!, `${block}*${tower.attentionChunkSize}-${tower.attentionContextLeft - 1}+${keySlot}`, hidden!]),
        predicate: (keySlot) => `0<=${block}*${tower.attentionChunkSize}-${tower.attentionContextLeft - 1}+${keySlot} && ${block}*${tower.attentionChunkSize}-${tower.attentionContextLeft - 1}+${keySlot}<${right}.shape[1]`,
      };
    }
  }
}

function explicitDynamicReductionWindow(request: Gemma4LiteralScalarViewRequest, id: string): { start: number; end: number } {
  const start = request.inputStart!, count = request.inputCount!;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(count) || start < 0 || count <= 0) {
    throw new Error(`${id}: janela dinâmica ${start}+${count} requer inteiros não negativos e count positivo.`);
  }
  return { start, end: start + count };
}

function reductionWindow(request: Gemma4LiteralScalarViewRequest, width: number, id: string): { start: number; end: number } {
  if ((request.inputStart === undefined) !== (request.inputCount === undefined)) throw new Error(`${id}: inputStart/inputCount devem aparecer juntos.`);
  const start = request.inputStart ?? 0, count = request.inputCount ?? width;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(count) || start < 0 || count <= 0 || start + count > width) {
    throw new Error(`${id}: janela ${start}+${count} fora de 0..${width}.`);
  }
  return { start, end: start + count };
}

function requireCoordinateRank(coordinate: number[], rank: number, id: string, axes: string): void {
  if (coordinate.length !== rank) throw new Error(`${id}: auditoria BMM requer coordenada ${axes}.`);
}

function assertAudioAttentionCoordinate(
  artifact: OpenGemma4CompositeLiteralArtifact,
  id: string,
  head: number,
  query: number,
  keySlot: number,
  keyExtent: number,
): void {
  const tower = artifact.program.audioProgram.tower;
  if (head >= tower.attentionHeads || query >= tower.attentionChunkSize || keySlot >= keyExtent) {
    throw new Error(`${id}: coordenada de atenção audio fora do domínio.`);
  }
}

function audioContext(artifact: OpenGemma4CompositeLiteralArtifact): number {
  const tower = artifact.program.audioProgram.tower;
  return tower.attentionChunkSize + tower.attentionContextLeft - 1 + tower.attentionContextRight;
}

function indexed(name: string, coordinate: ReadonlyArray<number | string>): string {
  return `${name}[${coordinate.join(",")}]`;
}

function assertCoordinate(coordinate: number[]): void {
  if (coordinate.length === 0 || coordinate.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new Error("Auditoria BMM requer coordenada não negativa.");
  }
}
