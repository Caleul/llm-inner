import { writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { bindGemma4VectorizedRealLoweringPlan } from "./gemma4-compiled-bundle.js";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { buildGemma4VectorizedRealLoweringPlan } from "./gemma4-vectorized-real-lowering.js";

const values = new Map<string, string>();
for (let index = 2; index < process.argv.length; index += 2) {
  const flag = process.argv[index], value = process.argv[index + 1];
  if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`);
  values.set(flag, value);
}
for (const flag of values.keys()) if (flag !== "--artifact" && flag !== "--output" && flag !== "--bind-bundle") throw new Error(`Flag desconhecida: ${flag}.`);
const required = (flag: string): string => { const value = values.get(flag); if (!value) throw new Error(`${flag} é obrigatório.`); return resolve(value); };
const artifactPath = required("--artifact"), bundle = values.get("--bind-bundle") ? resolve(values.get("--bind-bundle")!) : undefined;
const output = values.get("--output") ? resolve(values.get("--output")!) : bundle ? join(bundle, "vectorized-real-lowering.json") : undefined;
if (!output) throw new Error("--output ou --bind-bundle é obrigatório.");
const artifact = await openGemma4CompositeLiteralArtifact(artifactPath);
try {
  const plan = buildGemma4VectorizedRealLoweringPlan(artifact.realSimplifiedProgram, artifact.calculationGraph, artifact.integrityManifest);
  await writeFile(output, `${JSON.stringify(plan)}\n`, { encoding: "utf8", flag: "wx" });
  const manifest = bundle ? await bindGemma4VectorizedRealLoweringPlan(bundle, output) : undefined;
  process.stdout.write(`${JSON.stringify({ output, bytes: Buffer.byteLength(JSON.stringify(plan), "utf8") + 1, contract: plan.contract, ...(manifest ? { bundle, manifestSchemaVersion: manifest.schemaVersion } : {}) }, null, 2)}\n`);
} finally {
  await artifact.close();
}
