import { createWriteStream } from "node:fs";
import { open } from "node:fs/promises";
import { resolve } from "node:path";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { closeGemma4ParametricReverseTextInputs, composeGemma4ParametricReverse, type Gemma4ParametricReverseComposition } from "./gemma4-parametric-reverse-composer.js";

const values = new Map<string, string>(); for (let index = 2; index < process.argv.length; index += 2) { const flag = process.argv[index], value = process.argv[index + 1]; if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`); values.set(flag, value); }
const known = new Set(["--artifact", "--family", "--dimension", "--batch", "--sequence", "--steps", "--output", "--close-text-inputs", "--sequence-length", "--hidden-size", "--layers", "--ple-features"]); for (const flag of values.keys()) if (!known.has(flag)) throw new Error(`Flag desconhecida: ${flag}.`);
const artifact = await openGemma4CompositeLiteralArtifact(resolve(values.get("--artifact") ?? "artifacts/gemma4-e4b-dense.literal.json"));
try {
  const family = values.get("--family") ?? "terminal_logit", dimension = Number(values.get("--dimension") ?? "0"), batch = Number(values.get("--batch") ?? "0"), sequence = Number(values.get("--sequence") ?? "0"), steps = Number(values.get("--steps") ?? "8");
  if (![dimension, batch, sequence, steps].every((value) => Number.isSafeInteger(value) && value >= 0)) throw new Error("Coordenadas ou steps inválidos.");
  const output = artifact.realSimplifiedProgram.outputFunctions.find((entry) => entry.name === family && entry.fixedDimension === dimension); if (!output) throw new Error(`${family}[${dimension}]: output ausente.`);
  let result = composeGemma4ParametricReverse(artifact.realSimplifiedProgram, output, { batch, sequence }, steps);
  if (values.get("--close-text-inputs") === "true") {
    const dimensions = ["--sequence-length", "--hidden-size", "--layers", "--ple-features"].map((flag) => Number(values.get(flag)));
    if (!dimensions.every((value) => Number.isSafeInteger(value) && value > 0)) throw new Error("Fechamento text-inputs requer --sequence-length, --hidden-size, --layers e --ple-features.");
    result = closeGemma4ParametricReverseTextInputs(result, { sequenceLength: dimensions[0]!, hiddenSize: dimensions[1]!, layers: dimensions[2]!, pleFeatures: dimensions[3]! });
  } else if (values.has("--close-text-inputs") && values.get("--close-text-inputs") !== "false") throw new Error("--close-text-inputs aceita true ou false.");
  const outputPath = values.get("--output"); if (outputPath) await writeComposition(resolve(outputPath), result);
  process.stdout.write(`${JSON.stringify({ output: result.output, root: result.root, expressionNodes: result.graph.nodes.length, steps: result.steps, remainingFunctionCalls: result.remainingFunctionCalls.length, ...("inputVector" in result ? { inputVector: result.inputVector } : {}), ...(outputPath ? { file: resolve(outputPath) } : {}) }, null, 2)}\n`);
} finally { await artifact.close(); }

async function writeComposition(path: string, result: Gemma4ParametricReverseComposition & { inputVector?: unknown }): Promise<void> {
  const handle = await open(path, "wx"); await handle.close();
  const stream = createWriteStream(path, { flags: "w" });
  const write = async (text: string): Promise<void> => { if (!stream.write(text)) await new Promise<void>((accept) => stream.once("drain", accept)); };
  await write(`{"kind":${JSON.stringify(result.kind)},"schemaVersion":${result.schemaVersion},"output":${JSON.stringify(result.output)},"graph":{"kind":${JSON.stringify(result.graph.kind)},"schemaVersion":${result.graph.schemaVersion},"semantics":${JSON.stringify(result.graph.semantics)},"nodeOrder":${JSON.stringify(result.graph.nodeOrder)},"nodes":[`);
  for (let index = 0; index < result.graph.nodes.length; index += 1) { if (index) await write(","); await write(JSON.stringify(result.graph.nodes[index])); }
  await write(`]},"root":${JSON.stringify(result.root)},"steps":${JSON.stringify(result.steps)},"remainingFunctionCalls":${JSON.stringify(result.remainingFunctionCalls)}${result.inputVector === undefined ? "" : `,"inputVector":${JSON.stringify(result.inputVector)}`}}\n`);
  await new Promise<void>((accept, reject) => { stream.once("error", reject); stream.end(accept); });
}
