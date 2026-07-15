import { stat } from "node:fs/promises";
import * as path from "node:path";
import type { ModelCatalog } from "./types.js";
import { SafetensorsCatalogReader } from "./safetensors.js";
import { GgufCatalogReader } from "./gguf.js";
import { TensorBridge } from "./bridge.js";
import type { DenseF32Reader } from "./materialize.js";

export interface OpenCatalogResult {
  catalog: ModelCatalog;
  /** The live container reader used for verified range materialization. */
  reader: DenseF32Reader;
  close(): Promise<void>;
  bridge?: TensorBridge;
}

export async function openCatalog(source: string, includeBridge: boolean): Promise<OpenCatalogResult> {
  const info = await stat(source);
  if (info.isFile() && path.extname(source).toLowerCase() === ".gguf") {
    const reader = new GgufCatalogReader(source);
    try {
      return {
        catalog: await reader.inspect(),
        reader,
        close: () => reader.close(),
      };
    } catch (error) {
      await reader.close();
      throw error;
    }
  }

  if (info.isDirectory()) {
    const reader = new SafetensorsCatalogReader(source);
    try {
      const catalog = await reader.inspect();
      const bridge = includeBridge ? new TensorBridge(source) : undefined;
      return {
        catalog,
        reader,
        ...(bridge ? { bridge } : {}),
        close: async () => {
          await reader.close();
          if (bridge) await bridge.close();
        },
      };
    } catch (error) {
      await reader.close();
      throw error;
    }
  }

  throw new Error(`Fonte não suportada: ${source}`);
}
