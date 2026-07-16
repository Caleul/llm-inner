import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { executeGemma4PagedTextLiteralF32, generateGemma4PagedTextLiteralF32 } from "./gemma4-paged-text.js";

interface Arguments {
  artifact: string;
  inputIds: number[];
  maxNewTokens: number;
  eosTokenId?: number;
  maxReadBytes: number;
  output?: string;
}

const args = parseArguments(process.argv.slice(2));
const artifact = await openGemma4CompositeLiteralArtifact(args.artifact);
const started = performance.now();
const rssBefore = process.memoryUsage().rss;
try {
  const request = { inputIds: [args.inputIds] };
  const result = args.maxNewTokens === 0
    ? await executeGemma4PagedTextLiteralF32(artifact, request, { maxReadBytes: args.maxReadBytes })
    : await generateGemma4PagedTextLiteralF32(artifact, { ...request, maxNewTokens: args.maxNewTokens, ...(args.eosTokenId === undefined ? {} : { eosTokenId: args.eosTokenId }) }, { maxReadBytes: args.maxReadBytes });
  const logits = result.logits;
  const hash = createHash("sha256").update(Buffer.from(logits.values.buffer, logits.values.byteOffset, logits.values.byteLength)).digest("hex");
  const report = {
    kind: "gemma4-paged-text-literal-replay",
    artifact: artifact.artifact,
    artifactBytes: artifact.artifactBytes,
    sourceCheckpointAccessed: false,
    executionScope: "text-only; image/video/audio inputs are intentionally unsupported by this command",
    inputIds: args.inputIds,
    maxReadBytes: args.maxReadBytes,
    maxNewTokens: args.maxNewTokens,
    ...("generatedTokenIds" in result ? { generatedTokenIds: result.generatedTokenIds } : {}),
    logitsShape: logits.shape,
    logitsSha256: hash,
    kvProducerLayers: [...result.pastKeyValues.keys()],
    elapsedMilliseconds: Math.round((performance.now() - started) * 1000) / 1000,
    rssBefore,
    rssAfter: process.memoryUsage().rss,
    // Node normalizes ru_maxrss to KiB. Unlike rssAfter, this preserves the
    // process high-water mark for a real paged replay report.
    maxRssKiB: process.resourceUsage().maxRSS,
  };
  const json = `${JSON.stringify(report, null, 2)}\n`;
  if (args.output) await writeFile(args.output, json);
  else process.stdout.write(json);
} finally {
  await artifact.close();
}

function parseArguments(argv: string[]): Arguments {
  let artifact: string | undefined, inputIds: number[] | undefined, output: string | undefined, eosTokenId: number | undefined;
  let maxNewTokens = 0, maxReadMiB = 16;
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index], next = argv[index + 1];
    if (value === "--artifact") { artifact = next; index += 1; }
    else if (value === "--input-ids") { inputIds = parseIds(next); index += 1; }
    else if (value === "--max-new-tokens") { maxNewTokens = parseInteger(next, value); index += 1; }
    else if (value === "--eos-token-id") { eosTokenId = parseInteger(next, value); index += 1; }
    else if (value === "--max-read-mib") { maxReadMiB = parseInteger(next, value); index += 1; }
    else if (value === "--output") { output = next; index += 1; }
    else throw new Error(`Argumento desconhecido: ${value}.`);
  }
  if (!artifact || !inputIds || inputIds.length === 0 || maxNewTokens < 0 || maxReadMiB <= 0) {
    throw new Error("Uso: --artifact <literal.json> --input-ids <id,id,...> [--max-new-tokens N] [--eos-token-id N] [--max-read-mib N] [--output report.json].");
  }
  const maxReadBytes = maxReadMiB * 1024 * 1024;
  if (!Number.isSafeInteger(maxReadBytes)) throw new Error("--max-read-mib excede limite seguro.");
  return { artifact, inputIds, maxNewTokens, ...(eosTokenId === undefined ? {} : { eosTokenId }), maxReadBytes, ...(output ? { output } : {}) };
}

function parseIds(value: string | undefined): number[] {
  if (!value) throw new Error("--input-ids requer ao menos um ID.");
  const ids = value.split(",").map((entry) => parseInteger(entry, "--input-ids"));
  if (ids.some((id) => id < 0)) throw new Error("--input-ids requer IDs não negativos.");
  return ids;
}

function parseInteger(value: string | undefined, flag: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`${flag} requer inteiro seguro.`);
  return parsed;
}
