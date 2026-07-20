import { createHash } from "node:crypto";
import { access, writeFile } from "node:fs/promises";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { verifyGemma4CompositeLiteralEmbeddedPayloadIntegrity } from "./gemma4-composite-literal-payload-verification.js";
import { buildGemma4LiteralGenerationNavigation, renderGemma4LiteralGenerationCalculationView } from "./gemma4-literal-generation-navigation.js";
import { buildGemma4LiteralCalculationSlice } from "./gemma4-literal-calculation-slice.js";
import { buildGemma4LiteralEndToEndCalculation } from "./gemma4-literal-end-to-end-calculation.js";
import {
  listGemma4LiteralOperations,
  renderGemma4LiteralMultimodalScalarView,
} from "./gemma4-literal-multimodal-scalar-view.js";
import {
  listGemma4LiteralRuntimeReductionOperations,
  renderGemma4LiteralRuntimeReductionAudit,
} from "./gemma4-literal-runtime-reduction-audit.js";

interface Arguments {
  artifact: string;
  tensor?: string;
  offset: number;
  byteLength: number;
  verifyPayloads: boolean;
  assertSourceUnavailable?: string;
  output?: string;
  listOperations: boolean;
  listRuntimeReductions: boolean;
  runtimeReductionAudit: boolean;
  showGenerationProgram: boolean;
  listGenerationOperations: boolean;
  generationOperationId?: string;
  generationMaxNewTokens?: number;
  operationId?: string;
  outputCoordinate?: number[];
  tokenId?: number;
  positionCoordinate?: [number, number];
  inputStart?: number;
  inputCount?: number;
  numericLiteral?: string;
  calculationSliceOperationId?: string;
  endToEndCalculation: boolean;
}

const args = parseArguments(process.argv.slice(2));
if (args.assertSourceUnavailable) await assertUnavailable(args.assertSourceUnavailable);
const artifact = await openGemma4CompositeLiteralArtifact(args.artifact);
try {
  const selected = args.tensor ? artifact.constants.get(args.tensor) : undefined;
  if (args.tensor && !selected) throw new Error(`Tensor literal não encontrado: ${args.tensor}.`);
  const result: Record<string, unknown> = {
    schemaVersion: artifact.schemaVersion,
    artifact: artifact.artifact,
    artifactBytes: artifact.artifactBytes,
    sourceIdentity: artifact.sourceIdentity,
    authoritativeExecution: artifact.authoritativeExecution,
    constants: artifact.constants.size,
    storageDecoders: artifact.storageDecoders.length,
    embeddedTextSource: artifact.program.textProgram.source.path,
    sourceFormat: "safetensors",
    payloadIntegrityCommitted: artifact.payloadIntegrity !== undefined,
    numericLiterals: artifact.numericLiterals.literals.length,
    formulaLanguage: artifact.formulaLanguage,
    denseDecoderLanguage: artifact.denseDecoderLanguage,
    inputContract: artifact.inputContract,
    outputContract: artifact.outputContract,
    forwardControl: artifact.forwardControl,
    calculationGraph: {
      assignments: artifact.calculationGraph.assignments.length,
      coordinateLanguage: artifact.calculationGraph.coordinateLanguage,
      firstOperation: artifact.calculationGraph.assignments[0]?.operationId,
      lastOperation: artifact.calculationGraph.assignments.at(-1)?.operationId,
      explicitPredecessorEdges: artifact.calculationGraph.assignments.reduce((total, assignment) =>
        total + assignment.predecessors.filter((predecessor) => predecessor.producerOperationId !== undefined).length, 0),
      instantiatedInvocations: [...new Set(artifact.calculationGraph.assignments.flatMap((assignment) => assignment.invocationId ? [assignment.invocationId] : []))],
    },
    coordinateNavigation: coordinateNavigationSummary(artifact.calculationGraph.assignments),
    scalarExecution: scalarExecutionSummary(artifact.calculationGraph.assignments),
    reductionDomains: reductionDomainSummary(artifact.calculationGraph.assignments),
    ...(args.assertSourceUnavailable ? { assertedUnavailableSource: args.assertSourceUnavailable, sourceCheckpointAccessed: false } : {}),
  };
  if (args.numericLiteral) {
    const literal = artifact.numericLiterals.literals.find((entry) => entry.token === args.numericLiteral);
    if (!literal) throw new Error(`Literal numérico não encontrado: ${args.numericLiteral}.`);
    result.numericLiteral = literal;
  }
  if (args.calculationSliceOperationId) {
    result.calculationSlice = buildGemma4LiteralCalculationSlice(artifact, args.calculationSliceOperationId);
  }
  if (args.endToEndCalculation) {
    result.endToEndCalculation = buildGemma4LiteralEndToEndCalculation(artifact, args.generationMaxNewTokens!);
  }
  if (selected) {
    const tensor = { name: selected.name, storageDtype: selected.storageDtype, storageShape: selected.storageShape, logicalShape: selected.logicalShape };
    const bytes = await artifact.readTensorBytesRange(tensor, args.offset, args.byteLength);
    result.selectedTensor = {
      name: selected.name,
      storageDtype: selected.storageDtype,
      storageShape: selected.storageShape,
      payloadBytes: selected.payloadBytes,
      offset: args.offset,
      byteLength: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex"),
    };
  }
  if (args.listOperations) result.operations = listGemma4LiteralOperations(artifact);
  if (args.listRuntimeReductions) result.runtimeReductionOperations = listGemma4LiteralRuntimeReductionOperations(artifact);
  if (args.showGenerationProgram) result.generationProgram = artifact.generation;
  if (args.listGenerationOperations) result.generationNavigation = buildGemma4LiteralGenerationNavigation(artifact, args.generationMaxNewTokens!);
  if (args.generationOperationId) {
    result.generationCalculationView = renderGemma4LiteralGenerationCalculationView(
      artifact, args.generationMaxNewTokens!, args.generationOperationId,
    );
  }
  if (args.operationId) {
    const request = {
      operationId: args.operationId,
      outputCoordinate: args.outputCoordinate!,
      ...(args.tokenId === undefined ? {} : { tokenId: args.tokenId }),
      ...(args.positionCoordinate === undefined ? {} : { positionCoordinate: args.positionCoordinate }),
      ...(args.inputStart === undefined ? {} : { inputStart: args.inputStart, inputCount: args.inputCount! }),
    };
    if (args.runtimeReductionAudit) result.runtimeReductionAudit = renderGemma4LiteralRuntimeReductionAudit(artifact, request);
    else result.scalarView = await renderGemma4LiteralMultimodalScalarView(artifact, request);
  }
  if (args.verifyPayloads) {
    result.payloadIntegrityVerification = await verifyGemma4CompositeLiteralEmbeddedPayloadIntegrity({
      artifact: args.artifact,
      ...(args.assertSourceUnavailable ? { assertSourceUnavailable: args.assertSourceUnavailable } : {}),
    });
  }
  const json = `${JSON.stringify(result, null, 2)}\n`;
  if (args.output) await writeFile(args.output, json);
  else process.stdout.write(json);
} finally {
  await artifact.close();
}

function reductionDomainSummary(assignments: OpenedCalculationAssignments): Record<string, number> {
  let reductions = 0, stages = 0, domains = 0, runtimeDefined = 0;
  for (const assignment of assignments) {
    if (assignment.scalarCalculation.reduction) {
      reductions += 1;
      domains += assignment.scalarCalculation.reduction.domains.length;
      if (assignment.scalarCalculation.reduction.order === "runtime-defined") runtimeDefined += 1;
    }
    for (const stage of assignment.scalarCalculation.reductionStages ?? []) {
      stages += 1;
      domains += stage.domains.length;
    }
  }
  return { reductions, stages, domains, runtimeDefined };
}

type OpenedCalculationAssignments = Awaited<ReturnType<typeof openGemma4CompositeLiteralArtifact>>["calculationGraph"]["assignments"];

function scalarExecutionSummary(assignments: OpenedCalculationAssignments): Record<string, number> {
  let statements = 0, localStatements = 0, preconditions = 0, multiStatementAssignments = 0;
  let statementDataflowEntries = 0, localWrites = 0, outputWrites = 0, localReads = 0;
  let producerEdges = 0, reverseConsumerEdges = 0, indexedLocalAccesses = 0, localCoordinatePrograms = 0;
  let statementPrograms = 0, expressionNodes = 0, reductionCalls = 0, orderedLoops = 0, filteredDomains = 0, evaluateInvocations = 0;
  let statementEnvironments = 0, orderedInputBindings = 0, localBindings = 0, reductionBindings = 0, intrinsicBindings = 0, memberAccessBindings = 0;
  const countExpression = (node: unknown): void => {
    if (!node || typeof node !== "object") return;
    const kind = (node as { kind?: string }).kind;
    if (kind) expressionNodes += 1;
    if (kind === "ordered-loop") orderedLoops += 1;
    if (kind === "filtered-domain") filteredDomains += 1;
    if (kind === "evaluate-invocation") evaluateInvocations += 1;
    if (kind === "call") {
      const callee = (node as { callee?: { kind?: string; name?: string } }).callee;
      if (callee?.kind === "identifier" && /(?:REDUCE|DOT|FMA)/.test(callee.name ?? "")) reductionCalls += 1;
    }
    for (const [key, value] of Object.entries(node)) {
      if (key === "kind") continue;
      if (Array.isArray(value)) value.forEach(countExpression);
      else countExpression(value);
    }
  };
  for (const assignment of assignments) {
    const scalarAssignments = assignment.scalarCalculation.scalarAssignments;
    const dataflow = assignment.scalarCalculation.statementDataflow;
    statements += scalarAssignments.length;
    if (scalarAssignments.length > 1) multiStatementAssignments += 1;
    localStatements += Math.max(0, scalarAssignments.length - 1);
    preconditions += scalarAssignments.filter((statement) => statement.startsWith("require ")).length;
    statementPrograms += assignment.scalarCalculation.statementPrograms.length;
    const environment = assignment.scalarCalculation.statementEnvironment;
    statementEnvironments += 1;
    orderedInputBindings += environment.orderedInputs.length;
    localBindings += environment.locals.length;
    reductionBindings += environment.reductions.length;
    intrinsicBindings += environment.intrinsics.length;
    memberAccessBindings += environment.memberAccesses.length;
    assignment.scalarCalculation.statementPrograms.forEach((statement) => {
      statement.targets.forEach((target) => target.coordinates.forEach(countExpression));
      countExpression(statement.expression);
    });
    statementDataflowEntries += dataflow.length;
    for (const statement of dataflow) {
      localWrites += statement.writes.filter((write) => write.role === "local").length;
      outputWrites += statement.writes.filter((write) => write.role === "output").length;
      localReads += statement.reads.length;
      producerEdges += statement.reads.length;
      reverseConsumerEdges += statement.consumerStatementOrdinals.length;
      for (const access of [...statement.writes, ...statement.reads]) {
        if (access.kind === "indexed") indexedLocalAccesses += 1;
        localCoordinatePrograms += access.coordinatePrograms?.length ?? 0;
      }
    }
  }
  return {
    assignments: assignments.length,
    statements,
    localStatements,
    preconditions,
    multiStatementAssignments,
    statementDataflowEntries,
    localWrites,
    outputWrites,
    localReads,
    producerEdges,
    reverseConsumerEdges,
    indexedLocalAccesses,
    localCoordinatePrograms,
    statementPrograms,
    statementEnvironments,
    orderedInputBindings,
    localBindings,
    reductionBindings,
    intrinsicBindings,
    memberAccessBindings,
    expressionNodes,
    reductionCalls,
    orderedLoops,
    filteredDomains,
    evaluateInvocations,
  };
}

function coordinateNavigationSummary(assignments: OpenedCalculationAssignments): Record<string, number> {
  let predecessors = 0, addressed = 0, shapeOrControlOnly = 0;
  let tensorElementAccesses = 0, tensorShapeAccesses = 0, wholeValueAccesses = 0;
  let outputWrites = 0, outputShapeAssertions = 0;
  let consumers = 0, addressedConsumers = 0, shapeOrControlOnlyConsumers = 0, consumerAccesses = 0;
  let consumerTensorElementAccesses = 0, consumerTensorShapeAccesses = 0, consumerWholeValueAccesses = 0;
  let coordinatePrograms = 0, tensorAxisPrograms = 0, rangePrograms = 0, stablePrefixPrograms = 0;
  const countProgram = (program: { kind: string }): void => {
    coordinatePrograms += 1;
    if (program.kind === "inclusive-range") rangePrograms += 1;
    if (program.kind === "stable-true-prefix-rank") stablePrefixPrograms += 1;
    const visit = (node: unknown): void => {
      if (!node || typeof node !== "object") return;
      if ((node as { kind?: string }).kind === "tensor-axis") tensorAxisPrograms += 1;
      for (const value of Object.values(node)) {
        if (Array.isArray(value)) value.forEach(visit);
        else visit(value);
      }
    };
    visit(program);
  };
  for (const assignment of assignments) {
    outputWrites += 1;
    outputShapeAssertions += assignment.outputCoordinate.shapeAssertions.length;
    assignment.outputCoordinate.write.coordinatePrograms.forEach(countProgram);
    assignment.outputCoordinate.shapeAssertions.forEach((access) => countProgram(access.axisProgram));
    for (const consumer of assignment.consumerCoordinates) {
      consumers += 1;
      consumerAccesses += consumer.accesses.length;
      if (consumer.scalarUse === "addressed") addressedConsumers += 1;
      else shapeOrControlOnlyConsumers += 1;
      for (const access of consumer.accesses) {
        if (access.kind === "tensor-element") {
          consumerTensorElementAccesses += 1;
          access.coordinatePrograms.forEach(countProgram);
        } else if (access.kind === "tensor-shape") {
          consumerTensorShapeAccesses += 1;
          countProgram(access.axisProgram);
        }
        else consumerWholeValueAccesses += 1;
      }
    }
  }
  for (const assignment of assignments) for (const predecessor of assignment.predecessors) {
    predecessors += 1;
    if (predecessor.scalarUse === "addressed") addressed += 1;
    else shapeOrControlOnly += 1;
    for (const access of predecessor.accesses) {
      if (access.kind === "tensor-element") {
        tensorElementAccesses += 1;
        access.coordinatePrograms.forEach(countProgram);
      } else if (access.kind === "tensor-shape") {
        tensorShapeAccesses += 1;
        countProgram(access.axisProgram);
      }
      else wholeValueAccesses += 1;
    }
  }
  return {
    outputWrites,
    outputShapeAssertions,
    predecessors,
    addressed,
    shapeOrControlOnly,
    tensorElementAccesses,
    tensorShapeAccesses,
    wholeValueAccesses,
    consumers,
    addressedConsumers,
    shapeOrControlOnlyConsumers,
    consumerAccesses,
    consumerTensorElementAccesses,
    consumerTensorShapeAccesses,
    consumerWholeValueAccesses,
    coordinatePrograms,
    tensorAxisPrograms,
    rangePrograms,
    stablePrefixPrograms,
  };
}

function parseArguments(argv: string[]): Arguments {
  let artifact: string | undefined, tensor: string | undefined, output: string | undefined, assertSourceUnavailable: string | undefined, operationId: string | undefined, generationOperationId: string | undefined, numericLiteral: string | undefined, calculationSliceOperationId: string | undefined;
  let outputCoordinate: number[] | undefined, tokenId: number | undefined, positionCoordinate: [number, number] | undefined, inputStart: number | undefined, inputCount: number | undefined;
  let generationMaxNewTokens: number | undefined;
  let offset = 0, byteLength = 4096, verifyPayloads = false, listOperations = false, listRuntimeReductions = false, runtimeReductionAudit = false, showGenerationProgram = false, listGenerationOperations = false, endToEndCalculation = false;
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    const next = argv[index + 1];
    if (value === "--artifact") { artifact = next; index += 1; }
    else if (value === "--tensor") { tensor = next; index += 1; }
    else if (value === "--offset") { offset = parseInteger(next, "--offset"); index += 1; }
    else if (value === "--byte-length") { byteLength = parseInteger(next, "--byte-length"); index += 1; }
    else if (value === "--verify-payloads") { verifyPayloads = true; }
    else if (value === "--list-operations") { listOperations = true; }
    else if (value === "--list-runtime-reductions") { listRuntimeReductions = true; }
    else if (value === "--runtime-reduction-audit") { runtimeReductionAudit = true; }
    else if (value === "--show-generation-program") { showGenerationProgram = true; }
    else if (value === "--list-generation-operations") { listGenerationOperations = true; }
    else if (value === "--generation-operation") { generationOperationId = requiredValue(next, "--generation-operation"); index += 1; }
    else if (value === "--generation-max-new-tokens") { generationMaxNewTokens = parseInteger(next, "--generation-max-new-tokens"); index += 1; }
    else if (value === "--operation") { operationId = requiredValue(next, "--operation"); index += 1; }
    else if (value === "--output-coordinate") { outputCoordinate = parseCoordinate(requiredValue(next, "--output-coordinate")); index += 1; }
    else if (value === "--token-id") { tokenId = parseInteger(next, "--token-id"); index += 1; }
    else if (value === "--position-coordinate") { positionCoordinate = parsePositionCoordinate(requiredValue(next, "--position-coordinate")); index += 1; }
    else if (value === "--input-start") { inputStart = parseInteger(next, "--input-start"); index += 1; }
    else if (value === "--input-count") { inputCount = parseInteger(next, "--input-count"); index += 1; }
    else if (value === "--numeric-literal") { numericLiteral = requiredValue(next, "--numeric-literal"); index += 1; }
    else if (value === "--calculation-slice") { calculationSliceOperationId = requiredValue(next, "--calculation-slice"); index += 1; }
    else if (value === "--end-to-end-calculation") { endToEndCalculation = true; }
    else if (value === "--assert-source-unavailable") {
      if (!next || next.startsWith("--")) throw new Error("--assert-source-unavailable requer um caminho de checkpoint.");
      assertSourceUnavailable = next;
      index += 1;
    }
    else if (value === "--output") { output = next; index += 1; }
    else throw new Error(`Argumento desconhecido: ${value}.`);
  }
  if (!artifact) throw new Error("Uso: --artifact <literal.json> [--numeric-literal <token>] [--calculation-slice <operation-id>] [--end-to-end-calculation --generation-max-new-tokens <n>] [--list-operations] [--list-runtime-reductions] [--show-generation-program] [--list-generation-operations --generation-max-new-tokens <n>] [--generation-operation <id> --generation-max-new-tokens <n>] [--operation <id> --output-coordinate <i,j,...> [--runtime-reduction-audit] [--token-id <id>] [--position-coordinate <x,y>] [--input-start <i> --input-count <n>]] [--tensor <nome> --offset <bytes> --byte-length <bytes>] [--verify-payloads --assert-source-unavailable <checkpoint>] [--output <report.json>].");
  if (assertSourceUnavailable !== undefined && !verifyPayloads && !operationId && !numericLiteral && !calculationSliceOperationId && !endToEndCalculation && !listOperations && !listRuntimeReductions && !showGenerationProgram && !listGenerationOperations && !generationOperationId) throw new Error("--assert-source-unavailable requer uma operação de inspeção.");
  if ((tensor === undefined && (offset !== 0 || byteLength !== 4096)) || (tensor !== undefined && (!Number.isSafeInteger(offset) || !Number.isSafeInteger(byteLength) || offset < 0 || byteLength <= 0))) {
    throw new Error("--offset e --byte-length requerem --tensor e valores inteiros positivos.");
  }
  if ((operationId === undefined) !== (outputCoordinate === undefined)) throw new Error("--operation e --output-coordinate devem ser fornecidos juntos.");
  if (runtimeReductionAudit && operationId === undefined) throw new Error("--runtime-reduction-audit requer --operation e --output-coordinate.");
  if ((inputStart === undefined) !== (inputCount === undefined) || (inputStart !== undefined && operationId === undefined)) throw new Error("--input-start e --input-count requerem --operation e devem ser fornecidos juntos.");
  if (tokenId !== undefined && operationId === undefined) throw new Error("--token-id requer --operation.");
  if (positionCoordinate !== undefined && operationId === undefined) throw new Error("--position-coordinate requer --operation.");
  if ((listGenerationOperations || generationOperationId !== undefined || endToEndCalculation) !== (generationMaxNewTokens !== undefined)) throw new Error("Navegação de geração requer --generation-max-new-tokens e uma operação/listagem de geração.");
  if (generationMaxNewTokens !== undefined && generationMaxNewTokens < 0) throw new Error("--generation-max-new-tokens requer inteiro não negativo.");
  return {
    artifact, ...(tensor ? { tensor } : {}), offset, byteLength, verifyPayloads, listOperations, listRuntimeReductions, runtimeReductionAudit, showGenerationProgram, listGenerationOperations, endToEndCalculation,
    ...(assertSourceUnavailable ? { assertSourceUnavailable } : {}), ...(output ? { output } : {}),
    ...(operationId ? { operationId, outputCoordinate: outputCoordinate! } : {}), ...(tokenId === undefined ? {} : { tokenId }),
    ...(positionCoordinate === undefined ? {} : { positionCoordinate }),
    ...(inputStart === undefined ? {} : { inputStart, inputCount: inputCount! }),
    ...(generationOperationId ? { generationOperationId } : {}),
    ...(generationMaxNewTokens === undefined ? {} : { generationMaxNewTokens }),
    ...(numericLiteral ? { numericLiteral } : {}),
    ...(calculationSliceOperationId ? { calculationSliceOperationId } : {}),
  };
}

function parseInteger(value: string | undefined, flag: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`${flag} requer inteiro seguro.`);
  return parsed;
}

function requiredValue(value: string | undefined, flag: string): string {
  if (!value || value.startsWith("--")) throw new Error(`${flag} requer um valor.`);
  return value;
}

function parseCoordinate(value: string): number[] {
  const coordinate = value.split(",").map((entry) => Number(entry));
  if (coordinate.length === 0 || coordinate.some((entry) => !Number.isSafeInteger(entry) || entry < 0)) {
    throw new Error("--output-coordinate requer inteiros não negativos separados por vírgula.");
  }
  return coordinate;
}

function parsePositionCoordinate(value: string): [number, number] {
  const coordinate = value.split(",").map((entry) => Number(entry));
  if (coordinate.length !== 2 || coordinate.some((entry) => !Number.isSafeInteger(entry) || entry < -1) || ((coordinate[0] === -1) !== (coordinate[1] === -1))) {
    throw new Error("--position-coordinate requer x,y inteiros não negativos ou -1,-1 para padding.");
  }
  return coordinate as [number, number];
}

async function assertUnavailable(source: string): Promise<void> {
  try {
    await access(source);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  throw new Error(`Inspeção literal Gemma 4 requer source indisponível, mas '${source}' ainda existe.`);
}
