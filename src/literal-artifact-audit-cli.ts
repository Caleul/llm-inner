import { mkdir, writeFile } from "node:fs/promises";
import * as path from "node:path";
import { auditLiteralArtifactAgainstCatalog } from "./literal-artifact-audit.js";
import { verifyGemma4CompositeLiteralPayloadsAgainstCatalog } from "./gemma4-composite-literal-payload-verification.js";

function value(argv: string[], name: string): string {
  const index = argv.indexOf(name);
  const result = index === -1 ? undefined : argv[index + 1];
  if (!result || result.startsWith("--")) throw new Error(`Uso: --artifact <literal.json> --source <checkpoint> --output <audit.json>; ${name} ausente.`);
  return path.resolve(result);
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const artifact = value(argv, "--artifact");
  const source = value(argv, "--source");
  const output = value(argv, "--output");
  const audit = await auditLiteralArtifactAgainstCatalog(artifact, source);
  const verification = argv.includes("--verify-gemma4-payloads")
    ? await verifyGemma4CompositeLiteralPayloadsAgainstCatalog({ artifact, source })
    : undefined;
  await mkdir(path.dirname(output), { recursive: true });
  const result = verification ? { ...audit, payloadVerification: verification } : audit;
  await writeFile(output, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(result));
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
