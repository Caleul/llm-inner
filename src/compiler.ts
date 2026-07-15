import { mkdir, writeFile } from "node:fs/promises";
import * as path from "node:path";
import { openCatalog } from "./catalog.js";
import { buildModelIR } from "./architecture.js";
import { renderEquations } from "./render.js";
import type { PreviewOptions } from "./types.js";

export interface CompileOptions {
  source: string;
  output: string;
  equationsOutput?: string;
  preview: PreviewOptions;
}

export async function compileModel(options: CompileOptions): Promise<void> {
  const opened = await openCatalog(options.source, options.preview.includeWeights || options.source.endsWith(".gguf"));
  try {
    const ir = await buildModelIR(opened.catalog, options.preview, opened.bridge);
    await mkdir(path.dirname(options.output), { recursive: true });
    await writeFile(options.output, `${JSON.stringify(ir, null, 2)}\n`, "utf8");
    if (options.equationsOutput) {
      await mkdir(path.dirname(options.equationsOutput), { recursive: true });
      await writeFile(options.equationsOutput, renderEquations(ir), "utf8");
    }
  } finally {
    await opened.close();
  }
}
