import type { DenseF32Tensor, DenseTensor } from "./types.js";

/**
 * Canonical greedy decoding policy shared by execution and trace validation.
 * Selection is from the final sequence coordinate; equal logits keep the
 * lowest token ID.  A capture with a non-finite score has no portable,
 * deterministic greedy interpretation and is rejected rather than guessed.
 */
export function selectGreedyToken(logits: DenseTensor | DenseF32Tensor): number {
  if (logits.shape.length !== 3 || logits.shape[0] !== 1 || logits.shape[1] === undefined || logits.shape[1] <= 0 || logits.shape[2] === undefined || logits.shape[2] <= 0) {
    throw new Error(`Logits de geração devem ter shape [1, sequence, vocab] positivo, recebeu [${logits.shape.join(", ")}].`);
  }
  const sequence = logits.shape[1];
  const vocab = logits.shape[2];
  const offset = (sequence - 1) * vocab;
  let result = 0;
  for (let index = 0; index < vocab; index += 1) {
    if (!Number.isFinite(logits.values[offset + index])) {
      throw new Error("Logits de geração devem ser finitos para argmax determinístico.");
    }
    if (index > 0 && logits.values[offset + index]! > logits.values[offset + result]!) result = index;
  }
  return result;
}
