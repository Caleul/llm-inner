import type { ModelCatalog, TensorInfo } from "./types.js";

/**
 * GGUF stores ggml tensor dimensions with the fastest-moving dimension first,
 * whereas the IR's linear tensors are row-major [out, in].  This adapter is
 * intentionally limited to the documented Llama GGUF names and is selected
 * only after `general.architecture=llama`; it is not a name-suffix fallback.
 */
export function adaptGgufLlamaCatalog(catalog: ModelCatalog): ModelCatalog {
  if (catalog.format !== "gguf") return catalog;
  if (catalog.rawMetadata["general.architecture"] !== "llama") return catalog;

  const tensors = new Map<string, TensorInfo>();
  for (const [name, tensor] of catalog.tensors) {
    tensors.set(name, isLlamaMatrixName(name) ? withReversedLogicalShape(tensor) : { ...tensor, storageShape: [...tensor.storageShape], logicalShape: [...tensor.logicalShape] });
  }
  return { ...catalog, config: { ...catalog.config }, rawMetadata: { ...catalog.rawMetadata }, tensors };
}

function withReversedLogicalShape(tensor: TensorInfo): TensorInfo {
  if (tensor.storageShape.length !== 2) {
    throw new Error(`${tensor.name}: tensor matricial Llama GGUF deve declarar exatamente duas dimensões GGML.`);
  }
  const columns = tensor.storageShape[0]!;
  const rows = tensor.storageShape[1]!;
  if (!Number.isSafeInteger(columns) || !Number.isSafeInteger(rows) || columns <= 0 || rows <= 0) {
    throw new Error(`${tensor.name}: dimensões GGML matriciais inválidas.`);
  }
  return { ...tensor, storageShape: [...tensor.storageShape], logicalShape: [rows, columns] };
}

function isLlamaMatrixName(name: string): boolean {
  return name === "token_embd.weight" || name === "output.weight" ||
    /^blk\.\d+\.(?:attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$/.test(name);
}
