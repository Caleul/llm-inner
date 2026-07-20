import { resolve } from "node:path";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { writeGemma4RealSsaExport } from "./gemma4-real-ssa-export.js";

const values = new Map<string, string>();
for (let index = 2; index < process.argv.length; index += 2) {
  const flag = process.argv[index], value = process.argv[index + 1];
  if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`);
  values.set(flag, value);
}
for (const flag of values.keys()) if (flag !== "--artifact" && flag !== "--output") throw new Error(`Flag desconhecida: ${flag}.`);
const artifactPath = resolve(values.get("--artifact") ?? "artifacts/gemma4-e4b-dense.literal.json");
const outputPath = resolve(values.get("--output") ?? "artifacts/gemma4-e4b-dense.real-simplified.ssa.json");
const artifact = await openGemma4CompositeLiteralArtifact(artifactPath);
try { process.stdout.write(`${JSON.stringify({ output: outputPath, ...(await writeGemma4RealSsaExport(artifact, outputPath)) }, null, 2)}\n`); }
finally { await artifact.close(); }
