import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { verifyGemma4CompositeLiteralEmbeddedPayloadIntegrity } from "./gemma4-composite-literal-payload-verification.js";

interface Arguments {
  artifact: string;
  tensor?: string;
  offset: number;
  byteLength: number;
  verifyPayloads: boolean;
  assertSourceUnavailable?: string;
  output?: string;
}

const args = parseArguments(process.argv.slice(2));
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
  let artifact: string | undefined, tensor: string | undefined, output: string | undefined, assertSourceUnavailable: string | undefined;
  let offset = 0, byteLength = 4096, verifyPayloads = false;
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    const next = argv[index + 1];
    if (value === "--artifact") { artifact = next; index += 1; }
    else if (value === "--tensor") { tensor = next; index += 1; }
    else if (value === "--offset") { offset = parseInteger(next, "--offset"); index += 1; }
    else if (value === "--byte-length") { byteLength = parseInteger(next, "--byte-length"); index += 1; }
    else if (value === "--verify-payloads") { verifyPayloads = true; }
    else if (value === "--assert-source-unavailable") {
      if (!next || next.startsWith("--")) throw new Error("--assert-source-unavailable requer um caminho de checkpoint.");
      assertSourceUnavailable = next;
      index += 1;
    }
    else if (value === "--output") { output = next; index += 1; }
    else throw new Error(`Argumento desconhecido: ${value}.`);
  }
  if (!artifact) throw new Error("Uso: --artifact <literal.json> [--tensor <nome> --offset <bytes> --byte-length <bytes>] [--verify-payloads --assert-source-unavailable <checkpoint>] [--output <report.json>].");
  if (assertSourceUnavailable !== undefined && (!verifyPayloads || !assertSourceUnavailable)) throw new Error("--assert-source-unavailable requer --verify-payloads e um caminho de checkpoint.");
  if ((tensor === undefined && (offset !== 0 || byteLength !== 4096)) || (tensor !== undefined && (!Number.isSafeInteger(offset) || !Number.isSafeInteger(byteLength) || offset < 0 || byteLength <= 0))) {
    throw new Error("--offset e --byte-length requerem --tensor e valores inteiros positivos.");
  }
  return { artifact, ...(tensor ? { tensor } : {}), offset, byteLength, verifyPayloads, ...(assertSourceUnavailable ? { assertSourceUnavailable } : {}), ...(output ? { output } : {}) };
}

function parseInteger(value: string | undefined, flag: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`${flag} requer inteiro seguro.`);
  return parsed;
}
