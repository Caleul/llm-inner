import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readdir, stat, writeFile } from "node:fs/promises";
import * as path from "node:path";
import { inspectGemma4PackageContract } from "./gemma4-contract.js";
import { SafetensorsCatalogReader } from "./safetensors.js";

async function sha256(file: string): Promise<string> {
  return await new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    createReadStream(file)
      .on("error", reject)
      .on("data", (chunk: string | Buffer) => { hash.update(chunk); })
      .on("end", () => resolve(hash.digest("hex")));
  });
}

function option(argv: string[], name: string): string {
  const index = argv.indexOf(name);
  const value = index >= 0 ? argv[index + 1] : undefined;
  if (!value || value.startsWith("--")) throw new Error(`Uso: --source <Gemma4-dir> --output <audit.json>; valor ausente para ${name}.`);
  return path.resolve(value);
}

async function main(): Promise<void> {
  const source = option(process.argv.slice(2), "--source");
  const output = option(process.argv.slice(2), "--output");
  const reader = new SafetensorsCatalogReader(source);
  try {
    const catalog = await reader.inspect();
    const contract = inspectGemma4PackageContract(catalog);
    const files = (await readdir(source)).filter((file) => /^(config|generation_config|tokenizer|tokenizer_config)\.json$|^model\.safetensors$/.test(file)).sort();
    const checksums = await Promise.all(files.map(async (file) => ({ file, bytes: (await stat(path.join(source, file)).catch(() => ({ size: 0 }))).size, sha256: await sha256(path.join(source, file)) })));
    await mkdir(path.dirname(output), { recursive: true });
    await writeFile(output, `${JSON.stringify({ schemaVersion: 1, kind: "gemma4-package-contract-audit", source, contract, checksums }, null, 2)}\n`, "utf8");
    console.log(`Auditoria Gemma 4 escrita em ${output}.`);
  } finally {
    await reader.close();
  }
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
