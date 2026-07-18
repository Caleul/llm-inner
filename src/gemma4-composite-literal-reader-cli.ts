import { createHash } from "node:crypto";
import { access, writeFile } from "node:fs/promises";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { verifyGemma4CompositeLiteralEmbeddedPayloadIntegrity } from "./gemma4-composite-literal-payload-verification.js";
import { buildGemma4LiteralGenerationNavigation, renderGemma4LiteralGenerationCalculationView } from "./gemma4-literal-generation-navigation.js";
import { buildGemma4LiteralCalculationSlice } from "./gemma4-literal-calculation-slice.js";
import { buildGemma4LiteralEndToEndCalculation } from "./gemma4-literal-end-to-end-calculation.js";
import { listGemma4LiteralOperations, renderGemma4LiteralMultimodalScalarView } from "./gemma4-literal-multimodal-scalar-view.js";

interface Arguments {
  artifact: string;
  tensor?: string;
  offset: number;
  byteLength: number;
  verifyPayloads: boolean;
  assertSourceUnavailable?: string;
  output?: string;
  listOperations: boolean;
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
    calculationGraph: {
      assignments: artifact.calculationGraph.assignments.length,
      firstOperation: artifact.calculationGraph.assignments[0]?.operationId,
      lastOperation: artifact.calculationGraph.assignments.at(-1)?.operationId,
      explicitPredecessorEdges: artifact.calculationGraph.assignments.reduce((total, assignment) =>
        total + assignment.predecessors.filter((predecessor) => predecessor.producerOperationId !== undefined).length, 0),
      instantiatedInvocations: [...new Set(artifact.calculationGraph.assignments.flatMap((assignment) => assignment.invocationId ? [assignment.invocationId] : []))],
    },
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
  if (args.showGenerationProgram) result.generationProgram = artifact.generation;
  if (args.listGenerationOperations) result.generationNavigation = buildGemma4LiteralGenerationNavigation(artifact, args.generationMaxNewTokens!);
  if (args.generationOperationId) {
    result.generationCalculationView = renderGemma4LiteralGenerationCalculationView(
      artifact, args.generationMaxNewTokens!, args.generationOperationId,
    );
  }
  if (args.operationId) {
    result.scalarView = await renderGemma4LiteralMultimodalScalarView(artifact, {
      operationId: args.operationId,
      outputCoordinate: args.outputCoordinate!,
      ...(args.tokenId === undefined ? {} : { tokenId: args.tokenId }),
      ...(args.positionCoordinate === undefined ? {} : { positionCoordinate: args.positionCoordinate }),
      ...(args.inputStart === undefined ? {} : { inputStart: args.inputStart, inputCount: args.inputCount! }),
    });
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

function parseArguments(argv: string[]): Arguments {
  let artifact: string | undefined, tensor: string | undefined, output: string | undefined, assertSourceUnavailable: string | undefined, operationId: string | undefined, generationOperationId: string | undefined, numericLiteral: string | undefined, calculationSliceOperationId: string | undefined;
  let outputCoordinate: number[] | undefined, tokenId: number | undefined, positionCoordinate: [number, number] | undefined, inputStart: number | undefined, inputCount: number | undefined;
  let generationMaxNewTokens: number | undefined;
  let offset = 0, byteLength = 4096, verifyPayloads = false, listOperations = false, showGenerationProgram = false, listGenerationOperations = false, endToEndCalculation = false;
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    const next = argv[index + 1];
    if (value === "--artifact") { artifact = next; index += 1; }
    else if (value === "--tensor") { tensor = next; index += 1; }
    else if (value === "--offset") { offset = parseInteger(next, "--offset"); index += 1; }
    else if (value === "--byte-length") { byteLength = parseInteger(next, "--byte-length"); index += 1; }
    else if (value === "--verify-payloads") { verifyPayloads = true; }
    else if (value === "--list-operations") { listOperations = true; }
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
  if (!artifact) throw new Error("Uso: --artifact <literal.json> [--numeric-literal <token>] [--calculation-slice <operation-id>] [--end-to-end-calculation --generation-max-new-tokens <n>] [--list-operations] [--show-generation-program] [--list-generation-operations --generation-max-new-tokens <n>] [--generation-operation <id> --generation-max-new-tokens <n>] [--operation <id> --output-coordinate <i,j,...> [--token-id <id>] [--position-coordinate <x,y>] [--input-start <i> --input-count <n>]] [--tensor <nome> --offset <bytes> --byte-length <bytes>] [--verify-payloads --assert-source-unavailable <checkpoint>] [--output <report.json>].");
  if (assertSourceUnavailable !== undefined && !verifyPayloads && !operationId && !numericLiteral && !calculationSliceOperationId && !endToEndCalculation && !listOperations && !showGenerationProgram && !listGenerationOperations && !generationOperationId) throw new Error("--assert-source-unavailable requer uma operação de inspeção.");
  if ((tensor === undefined && (offset !== 0 || byteLength !== 4096)) || (tensor !== undefined && (!Number.isSafeInteger(offset) || !Number.isSafeInteger(byteLength) || offset < 0 || byteLength <= 0))) {
    throw new Error("--offset e --byte-length requerem --tensor e valores inteiros positivos.");
  }
  if ((operationId === undefined) !== (outputCoordinate === undefined)) throw new Error("--operation e --output-coordinate devem ser fornecidos juntos.");
  if ((inputStart === undefined) !== (inputCount === undefined) || (inputStart !== undefined && operationId === undefined)) throw new Error("--input-start e --input-count requerem --operation e devem ser fornecidos juntos.");
  if (tokenId !== undefined && operationId === undefined) throw new Error("--token-id requer --operation.");
  if (positionCoordinate !== undefined && operationId === undefined) throw new Error("--position-coordinate requer --operation.");
  if ((listGenerationOperations || generationOperationId !== undefined || endToEndCalculation) !== (generationMaxNewTokens !== undefined)) throw new Error("Navegação de geração requer --generation-max-new-tokens e uma operação/listagem de geração.");
  if (generationMaxNewTokens !== undefined && generationMaxNewTokens < 0) throw new Error("--generation-max-new-tokens requer inteiro não negativo.");
  return {
    artifact, ...(tensor ? { tensor } : {}), offset, byteLength, verifyPayloads, listOperations, showGenerationProgram, listGenerationOperations, endToEndCalculation,
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
