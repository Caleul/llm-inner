import { mkdir, writeFile } from "node:fs/promises";
import * as path from "node:path";
import { openCatalog } from "./catalog.js";
import { buildModelIR } from "./architecture.js";
import { buildLiteralCalculationProgram } from "./literal.js";
import type { LiteralTensorReader } from "./literal.js";
import { buildGemma4CompositeProgram } from "./gemma4-composite.js";
import { buildGemma4CompositeLiteralCalculationProgram } from "./gemma4-composite-literal.js";
import { renderEquations } from "./render.js";
import type { PreviewOptions } from "./types.js";

export interface CompileOptions {
  source: string;
  output: string;
  equationsOutput?: string;
  preview: PreviewOptions;
  /** Emit a source-independent dense, MLX-affine, or established GGUF Q8_0 calculation program. */
  literal?: boolean;
}

export async function compileModel(options: CompileOptions): Promise<void> {
  const opened = await openCatalog(options.source, options.preview.includeWeights);
  try {
    const ir = await buildModelIR(opened.catalog, options.preview, opened.bridge);
    await mkdir(path.dirname(options.output), { recursive: true });
    const artifact = options.literal
      ? await buildLiteralCalculationProgram(ir, opened.catalog, literalReader(opened.reader))
      : ir;
    await writeFile(options.output, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
    if (options.equationsOutput) {
      await mkdir(path.dirname(options.equationsOutput), { recursive: true });
      await writeFile(options.equationsOutput, renderEquations(ir), "utf8");
    }
  } finally {
    await opened.close();
  }
}

/**
 * Explicit route for the registered outer Gemma 4 package. `compileModel`
 * intentionally refuses composite packages rather than silently compiling
 * text_config alone, so this route always emits the full literal program.
 */
export async function compileGemma4CompositeLiteralModel(options: Omit<CompileOptions, "literal">): Promise<void> {
  const opened = await openCatalog(options.source, false);
  try {
    const composite = buildGemma4CompositeProgram(opened.catalog, options.preview);
    const artifact = await buildGemma4CompositeLiteralCalculationProgram(composite, opened.catalog, literalReader(opened.reader));
    await mkdir(path.dirname(options.output), { recursive: true });
    await writeFile(options.output, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  } finally {
    await opened.close();
  }
}

function literalReader(reader: unknown): LiteralTensorReader {
  if (
    typeof reader !== "object" || reader === null ||
    !("readTensorBytes" in reader) || typeof reader.readTensorBytes !== "function"
  ) {
    throw new Error("O contêiner selecionado não expõe ranges brutos para exportação literal autocontida.");
  }
  return reader as LiteralTensorReader;
}
