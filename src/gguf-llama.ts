import type { ModelCatalog, TensorInfo } from "./types.js";

interface GgufDecoderLayout {
  readonly label: string;
  readonly matrixName: (name: string) => boolean;
}

const STANDARD_DECODER_MATRIX_NAMES: GgufDecoderLayout["matrixName"] = (name) =>
  name === "token_embd.weight" || name === "output.weight" ||
  /^blk\.\d+\.(?:attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$/.test(name);

/**
 * These architectures use the documented GGUF decoder tensor registry: the
 * matrices below are stored as GGML [in, out], while the IR represents a
 * row-major linear weight as [out, in].  Selection is exclusively by the
 * authoritative `general.architecture` value; a matching tensor name is
 * never enough to assign a layout to an unknown architecture.
 */
const DECODER_LAYOUTS: ReadonlyMap<string, GgufDecoderLayout> = new Map([
  ["llama", { label: "Llama", matrixName: STANDARD_DECODER_MATRIX_NAMES }],
  ["qwen2", { label: "Qwen 2", matrixName: STANDARD_DECODER_MATRIX_NAMES }],
  ["qwen3", { label: "Qwen 3", matrixName: STANDARD_DECODER_MATRIX_NAMES }],
]);

/**
 * Adapts only registered GGUF decoder layouts before architecture lowering or
 * materialization.  The catalog reader intentionally retains raw GGML shapes;
 * this explicit architecture adapter is the only place that assigns the IR
 * matrix orientation.
 */
export function adaptGgufDecoderCatalog(catalog: ModelCatalog): ModelCatalog {
  if (catalog.format !== "gguf") return catalog;
  const architecture = catalog.rawMetadata["general.architecture"];
  if (typeof architecture !== "string") return catalog;
  const layout = DECODER_LAYOUTS.get(architecture);
  if (!layout) return catalog;

  const tensors = new Map<string, TensorInfo>();
  for (const [name, tensor] of catalog.tensors) {
    tensors.set(name, layout.matrixName(name) ? withReversedLogicalShape(tensor, layout.label) : { ...tensor, storageShape: [...tensor.storageShape], logicalShape: [...tensor.logicalShape] });
  }
  return { ...catalog, config: { ...catalog.config }, rawMetadata: { ...catalog.rawMetadata }, tensors };
}

/** @deprecated Use adaptGgufDecoderCatalog; retained for callers of the original Llama-only API. */
export const adaptGgufLlamaCatalog = adaptGgufDecoderCatalog;

function withReversedLogicalShape(tensor: TensorInfo, architecture: string): TensorInfo {
  if (tensor.storageShape.length !== 2) {
    throw new Error(`${tensor.name}: tensor matricial ${architecture} GGUF deve declarar exatamente duas dimensões GGML.`);
  }
  const columns = tensor.storageShape[0]!;
  const rows = tensor.storageShape[1]!;
  if (!Number.isSafeInteger(columns) || !Number.isSafeInteger(rows) || columns <= 0 || rows <= 0) {
    throw new Error(`${tensor.name}: dimensões GGML matriciais inválidas.`);
  }
  return { ...tensor, storageShape: [...tensor.storageShape], logicalShape: [rows, columns] };
}
