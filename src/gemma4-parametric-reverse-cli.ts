import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { composeGemma4ParametricReverse } from "./gemma4-parametric-reverse-composer.js";

const values = new Map<string, string>(); for (let index = 2; index < process.argv.length; index += 2) { const flag = process.argv[index], value = process.argv[index + 1]; if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`); values.set(flag, value); }
const known = new Set(["--artifact", "--family", "--dimension", "--batch", "--sequence", "--steps", "--output"]); for (const flag of values.keys()) if (!known.has(flag)) throw new Error(`Flag desconhecida: ${flag}.`);
const artifact = await openGemma4CompositeLiteralArtifact(resolve(values.get("--artifact") ?? "artifacts/gemma4-e4b-dense.literal.json"));
try {
  const family = values.get("--family") ?? "terminal_logit", dimension = Number(values.get("--dimension") ?? "0"), batch = Number(values.get("--batch") ?? "0"), sequence = Number(values.get("--sequence") ?? "0"), steps = Number(values.get("--steps") ?? "8");
  if (![dimension, batch, sequence, steps].every((value) => Number.isSafeInteger(value) && value >= 0)) throw new Error("Coordenadas ou steps inválidos.");
  const output = artifact.realSimplifiedProgram.outputFunctions.find((entry) => entry.name === family && entry.fixedDimension === dimension); if (!output) throw new Error(`${family}[${dimension}]: output ausente.`);
  const result = composeGemma4ParametricReverse(artifact.realSimplifiedProgram, output, { batch, sequence }, steps);
  const outputPath = values.get("--output"); if (outputPath) await writeFile(resolve(outputPath), `${JSON.stringify(result)}\n`, { encoding: "utf8", flag: "wx" });
  process.stdout.write(`${JSON.stringify({ output: result.output, root: result.root, expressionNodes: result.graph.nodes.length, steps: result.steps, remainingFunctionCalls: result.remainingFunctionCalls.length, ...(outputPath ? { file: resolve(outputPath) } : {}) }, null, 2)}\n`);
} finally { await artifact.close(); }
