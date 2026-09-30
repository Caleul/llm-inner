import { f16BitsToDyadic } from "./fixed-f16-projection.js";
import { prepareFixedF16CachedScalarSource, type FixedF16CachedScalarSource } from "./fixed-f16-parametric-formulas.js";

export interface FixedF16GreedyChoice { tokenId: number; logitBits: number; logitValue: number }

/** Select the first maximum at the last valid position, as torch.argmax does. */
export function chooseFixedF16GreedyToken(logitBits: readonly number[]): FixedF16GreedyChoice {
  if (logitBits.length === 0) throw new RangeError("Saída sem logits.");
  let tokenId = 0;
  let bits = logitBits[0]!;
  let dyadic = f16BitsToDyadic(bits);
  let value = Number(dyadic.coefficient) * 2 ** dyadic.exponent;
  for (let dimension = 1; dimension < logitBits.length; dimension++) {
    const candidateBits = logitBits[dimension]!;
    dyadic = f16BitsToDyadic(candidateBits);
    const candidate = Number(dyadic.coefficient) * 2 ** dyadic.exponent;
    if (candidate > value) {
      tokenId = dimension;
      bits = candidateBits;
      value = candidate;
    }
  }
  return { tokenId, logitBits: bits, logitValue: value };
}

/** Current checkpoint-independent scalar source; used to validate the choice rule. */
export function prepareFixedF16GreedyChoiceSource(program: FixedF16CachedScalarSource):
  (embeddings: readonly (readonly number[])[]) => FixedF16GreedyChoice {
  const forward = prepareFixedF16CachedScalarSource(program);
  return (embeddings) => chooseFixedF16GreedyToken(forward(embeddings).at(-1)!);
}
