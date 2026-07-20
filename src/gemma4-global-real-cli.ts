import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { buildGemma4ExactRealSimplifiedProgramFromArtifact } from "./gemma4-global-real-artifact-compiler.js";
import { validateGemma4ExactRealSimplifiedProgram, type Gemma4GlobalRealOutputFunction } from "./gemma4-global-real-program.js";

interface CliOptions {
  artifact: string;
  operation: string;
  coordinate: number[];
  name: string;
  finalQuantization: Gemma4GlobalRealOutputFunction["finalQuantization"];
  inputBoundaries: Set<string>;
  output?: string;
}

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2));
  const artifact = await openGemma4CompositeLiteralArtifact(options.artifact);
  try {
    const result = await buildGemma4ExactRealSimplifiedProgramFromArtifact(artifact, [{
      name: options.name,
      operationId: options.operation,
      coordinate: options.coordinate,
      finalQuantization: options.finalQuantization,
    }], new Set([...artifact.inputs.map((input) => input.name), ...options.inputBoundaries]));
    validateGemma4ExactRealSimplifiedProgram(result.program);
    const json = `${JSON.stringify(result)}\n`;
    if (options.output) {
      await writeFile(options.output, json, "utf8");
      process.stdout.write(`${JSON.stringify({
        kind: result.kind,
        output: options.output,
        sourceCheckpointAccessed: false,
        learnedScalarCount: result.learnedScalarCount,
        learnedStorageBytesRead: result.learnedStorageBytesRead,
        expressionNodes: result.program.expressionGraph.nodes.length,
        coverage: result.program.coverage,
      }, null, 2)}\n`);
    } else process.stdout.write(json);
  } finally {
    await artifact.close();
  }
}

function parseOptions(arguments_: readonly string[]): CliOptions {
  const values = new Map<string, string[]>();
  for (let index = 0; index < arguments_.length; index += 2) {
    const flag = arguments_[index], value = arguments_[index + 1];
    if (!flag?.startsWith("--") || value === undefined) throw new Error(`Argumento real Gemma 4 inválido em ${flag ?? "fim"}.`);
    const current = values.get(flag) ?? [];
    current.push(value);
    values.set(flag, current);
  }
  const one = (flag: string, required = true): string | undefined => {
    const entries = values.get(flag) ?? [];
    if (entries.length > 1 || (required && entries.length !== 1)) throw new Error(`${flag} requer ${required ? "exatamente um valor" : "no máximo um valor"}.`);
    return entries[0];
  };
  const artifact = one("--artifact")!, operation = one("--operation")!, coordinateToken = one("--coordinate")!;
  const coordinate = coordinateToken.split(",").map((token) => Number(token));
  if (coordinate.length === 0 || coordinate.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new Error("--coordinate requer inteiros não negativos separados por vírgula.");
  }
  const quantization = one("--final-quantization", false) ?? "F32-round-to-nearest-ties-to-even";
  if (quantization !== "F32-round-to-nearest-ties-to-even" && quantization !== "BF16-round-to-nearest-ties-to-even" && quantization !== "none") {
    throw new Error(`--final-quantization inválido: ${quantization}.`);
  }
  const known = new Set(["--artifact", "--operation", "--coordinate", "--name", "--final-quantization", "--input-boundary", "--output"]);
  for (const flag of values.keys()) if (!known.has(flag)) throw new Error(`Flag real Gemma 4 desconhecida: ${flag}.`);
  return {
    artifact: resolve(artifact),
    operation,
    coordinate,
    name: one("--name", false) ?? operation,
    finalQuantization: quantization,
    inputBoundaries: new Set(values.get("--input-boundary") ?? ["hidden_states_0"]),
    ...(one("--output", false) ? { output: resolve(one("--output", false)!) } : {}),
  };
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
