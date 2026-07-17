import { createHash } from "node:crypto";
import { access, writeFile } from "node:fs/promises";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { verifyGemma4CompositeLiteralEmbeddedPayloadIntegrity } from "./gemma4-composite-literal-payload-verification.js";
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
  operationId?: string;
  outputCoordinate?: number[];
  tokenId?: number;
  positionCoordinate?: [number, number];
  inputStart?: number;
  inputCount?: number;
}

const args = parseArguments(process.argv.slice(2));
if (args.assertSourceUnavailable) await assertUnavailable(args.assertSourceUnavailable);
const artifact = await openGemma4CompositeLiteralArtifact(args.artifact);
try {
  const selected = args.tensor ? artifact.constants.get(args.tensor) : undefined;
  if (args.tensor && !selected) throw new Error(`Tensor literal não encontrado: ${args.tensor}.`);
  const result: Record<string, unknown> = {
    artifact: artifact.artifact,
    artifactBytes: artifact.artifactBytes,
    constants: artifact.constants.size,
    storageDecoders: artifact.storageDecoders.length,
    embeddedTextSource: artifact.program.textProgram.source.path,
    sourceFormat: "safetensors",
    payloadIntegrityCommitted: artifact.payloadIntegrity !== undefined,
    ...(args.assertSourceUnavailable ? { assertedUnavailableSource: args.assertSourceUnavailable, sourceCheckpointAccessed: false } : {}),
  };
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

function parseArguments(argv: string[]): Arguments {
  let artifact: string | undefined, tensor: string | undefined, output: string | undefined, assertSourceUnavailable: string | undefined, operationId: string | undefined;
  let outputCoordinate: number[] | undefined, tokenId: number | undefined, positionCoordinate: [number, number] | undefined, inputStart: number | undefined, inputCount: number | undefined;
  let offset = 0, byteLength = 4096, verifyPayloads = false, listOperations = false;
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    const next = argv[index + 1];
    if (value === "--artifact") { artifact = next; index += 1; }
    else if (value === "--tensor") { tensor = next; index += 1; }
    else if (value === "--offset") { offset = parseInteger(next, "--offset"); index += 1; }
    else if (value === "--byte-length") { byteLength = parseInteger(next, "--byte-length"); index += 1; }
    else if (value === "--verify-payloads") { verifyPayloads = true; }
    else if (value === "--list-operations") { listOperations = true; }
    else if (value === "--operation") { operationId = requiredValue(next, "--operation"); index += 1; }
    else if (value === "--output-coordinate") { outputCoordinate = parseCoordinate(requiredValue(next, "--output-coordinate")); index += 1; }
    else if (value === "--token-id") { tokenId = parseInteger(next, "--token-id"); index += 1; }
    else if (value === "--position-coordinate") { positionCoordinate = parsePositionCoordinate(requiredValue(next, "--position-coordinate")); index += 1; }
    else if (value === "--input-start") { inputStart = parseInteger(next, "--input-start"); index += 1; }
    else if (value === "--input-count") { inputCount = parseInteger(next, "--input-count"); index += 1; }
    else if (value === "--assert-source-unavailable") {
      if (!next || next.startsWith("--")) throw new Error("--assert-source-unavailable requer um caminho de checkpoint.");
      assertSourceUnavailable = next;
      index += 1;
    }
    else if (value === "--output") { output = next; index += 1; }
    else throw new Error(`Argumento desconhecido: ${value}.`);
  }
  if (!artifact) throw new Error("Uso: --artifact <literal.json> [--list-operations] [--operation <id> --output-coordinate <i,j,...> [--token-id <id>] [--position-coordinate <x,y>] [--input-start <i> --input-count <n>]] [--tensor <nome> --offset <bytes> --byte-length <bytes>] [--verify-payloads --assert-source-unavailable <checkpoint>] [--output <report.json>].");
  if (assertSourceUnavailable !== undefined && !verifyPayloads && !operationId && !listOperations) throw new Error("--assert-source-unavailable requer --verify-payloads, --operation ou --list-operations.");
  if ((tensor === undefined && (offset !== 0 || byteLength !== 4096)) || (tensor !== undefined && (!Number.isSafeInteger(offset) || !Number.isSafeInteger(byteLength) || offset < 0 || byteLength <= 0))) {
    throw new Error("--offset e --byte-length requerem --tensor e valores inteiros positivos.");
  }
  if ((operationId === undefined) !== (outputCoordinate === undefined)) throw new Error("--operation e --output-coordinate devem ser fornecidos juntos.");
  if ((inputStart === undefined) !== (inputCount === undefined) || (inputStart !== undefined && operationId === undefined)) throw new Error("--input-start e --input-count requerem --operation e devem ser fornecidos juntos.");
  if (tokenId !== undefined && operationId === undefined) throw new Error("--token-id requer --operation.");
  if (positionCoordinate !== undefined && operationId === undefined) throw new Error("--position-coordinate requer --operation.");
  return {
    artifact, ...(tensor ? { tensor } : {}), offset, byteLength, verifyPayloads, listOperations,
    ...(assertSourceUnavailable ? { assertSourceUnavailable } : {}), ...(output ? { output } : {}),
    ...(operationId ? { operationId, outputCoordinate: outputCoordinate! } : {}), ...(tokenId === undefined ? {} : { tokenId }),
    ...(positionCoordinate === undefined ? {} : { positionCoordinate }),
    ...(inputStart === undefined ? {} : { inputStart, inputCount: inputCount! }),
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
