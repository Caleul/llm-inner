import type { DenseF32Tensor } from "./types.js";

/** Stable top-k over the final sequence coordinate; ties select the lower token id. */
export function rankGemma4TerminalLogits(logits: DenseF32Tensor, count = 10): Array<{ tokenId: number; logit: number }> {
  if (logits.shape.length !== 3 || logits.shape[0] !== 1 || (logits.shape[2] ?? 0) <= 0 || !Number.isSafeInteger(count) || count < 1) {
    throw new Error("Logits Gemma 4 devem possuir shape [1,sequence,vocab] e count positivo.");
  }
  const vocab = logits.shape[2]!, offset = logits.values.length - vocab;
  const top: Array<{ tokenId: number; logit: number }> = [];
  for (let tokenId = 0; tokenId < vocab; tokenId += 1) {
    const logit = logits.values[offset + tokenId]!;
    if (!Number.isFinite(logit)) throw new Error(`Logit terminal não finito no token ${tokenId}.`);
    if (top.length === count && (logit < top.at(-1)!.logit || (logit === top.at(-1)!.logit && tokenId > top.at(-1)!.tokenId))) continue;
    const index = top.findIndex((entry) => logit > entry.logit || (logit === entry.logit && tokenId < entry.tokenId));
    top.splice(index < 0 ? top.length : index, 0, { tokenId, logit });
    if (top.length > count) top.pop();
  }
  return top;
}
