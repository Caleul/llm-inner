import { mkdir, writeFile } from "node:fs/promises";
import * as path from "node:path";
import { openCatalog } from "./catalog.js";
import { buildModelIR } from "./architecture.js";
import { buildDenseSafetensorsLiteralProgram } from "./literal.js";
import type { LiteralTensorReader } from "./literal.js";
import { renderEquations } from "./render.js";
import type { PreviewOptions } from "./types.js";

export interface CompileOptions {
  source: string;
  output: string;
  equationsOutput?: string;
  preview: PreviewOptions;
  /** Emit a source-independent dense or established MLX-affine Safetensors calculation program. */
  literal?: boolean;
}

export async function compileModel(options: CompileOptions): Promise<void> {
  const opened = await openCatalog(options.source, options.preview.includeWeights);
  try {
    const ir = await buildModelIR(opened.catalog, options.preview, opened.bridge);
    await mkdir(path.dirname(options.output), { recursive: true });
    const artifact = options.literal
      ? await buildDenseSafetensorsLiteralProgram(ir, opened.catalog, literalReader(opened.reader))
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

function literalReader(reader: unknown): LiteralTensorReader {
  if (
    typeof reader !== "object" || reader === null ||
    !("readTensorBytes" in reader) || typeof reader.readTensorBytes !== "function"
  ) {
    throw new Error("O contêiner selecionado não expõe ranges brutos para exportação literal autocontida.");
  }
  return reader as LiteralTensorReader;
}
