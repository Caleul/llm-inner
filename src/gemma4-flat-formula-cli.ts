import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { generateGemma4FlatFormulaObject } from "./gemma4-flat-formula-generator.js";
import { gemma4ArtifactLearnedProvider } from "./gemma4-parametric-real-evaluator.js";

const values = new Map<string, string>();
for (let index = 2; index < process.argv.length; index += 2) {
  const flag = process.argv[index], value = process.argv[index + 1];
  if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`);
  values.set(flag, value);
}
const known = new Set(["--artifact", "--output", "--family", "--dimension", "--batch", "--sequence", "--max-characters", "--max-reduction-terms"]);
for (const flag of values.keys()) if (!known.has(flag)) throw new Error(`Flag desconhecida: ${flag}.`);
const artifactPath = resolve(values.get("--artifact") ?? "artifacts/gemma4-e4b-dense.literal.json");
const outputPath = resolve(values.get("--output") ?? "artifacts/gemma4-e4b-dense.real-simplified.flat.json");
const family = values.get("--family") ?? "terminal_logit";
const dimensionToken = values.get("--dimension");
const dimension = dimensionToken === undefined ? undefined : Number(dimensionToken);
if (dimension !== undefined && (!Number.isSafeInteger(dimension) || dimension < 0)) throw new Error("--dimension requer inteiro não negativo.");
const maxCharacters = BigInt(values.get("--max-characters") ?? "1000000000");
const maxReductionTerms = Number(values.get("--max-reduction-terms") ?? "1000000");
const batch = Number(values.get("--batch") ?? "0"), sequence = Number(values.get("--sequence"));
if (maxCharacters < 1n || !Number.isSafeInteger(maxReductionTerms) || maxReductionTerms < 1 || !Number.isSafeInteger(batch) || batch < 0 || !Number.isSafeInteger(sequence) || sequence < 0) throw new Error("Limites ou coordenadas da fórmula plana inválidos; --sequence é obrigatório.");

const artifact = await openGemma4CompositeLiteralArtifact(artifactPath);
try {
  const outputs = artifact.realSimplifiedProgram.outputFunctions.filter((entry) => entry.name === family && (dimension === undefined || entry.fixedDimension === dimension));
  if (!outputs.length) throw new Error(`${family}${dimension === undefined ? "" : `[${dimension}]`}: saída ausente.`);
  const learned = gemma4ArtifactLearnedProvider(artifact);
  const result = await generateGemma4FlatFormulaObject(artifact.realSimplifiedProgram, outputs, {
    maxCharacters, maxUnrolledReductionTerms: maxReductionTerms, outputParameters: { batch, sequence },
    inputVariable: (tensor, coordinates) => {
      if (tensor !== "hidden_states_0") throw new Error(`Boundary livre não substituída: ${tensor}.`);
      if (coordinates.length !== 3) throw new Error(`hidden_states_0 requer três coordenadas, recebeu ${coordinates.length}.`);
      return `x[((${coordinates[1]})*2560)+(${coordinates[2]})]`;
    },
    learnedLiteral: async (tensor, storageDtype, coordinates, decoderId) => String(await learned.element(tensor, storageDtype, coordinates, decoderId)),
  });
  await writeFile(outputPath, `${JSON.stringify(result.formulas)}\n`, { encoding: "utf8", flag: "wx" });
  process.stdout.write(`${JSON.stringify({ output: outputPath, formulas: outputs.length, freeVariables: result.freeVariables.length, characters: result.characters }, null, 2)}\n`);
} finally { await artifact.close(); }
