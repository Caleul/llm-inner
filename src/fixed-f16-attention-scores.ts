import {
  addDyadic, f16BitsToDyadic, f32BitsToDyadic, multiplyDyadic,
  roundDyadicToF16IfElse, roundDyadicToF32IfElse,
} from "./fixed-f16-projection.js";

export interface FixedAttentionScores {
  qRotated: number[][][];
  kRotated: number[][][];
  score: number[][][];
  scaled: number[][][];
  masked: number[][][];
}

function addF16(left: number, right: number): number {
  return roundDyadicToF16IfElse(addDyadic(f16BitsToDyadic(left), f16BitsToDyadic(right)));
}
function multiplyF16(left: number, right: number): number {
  return roundDyadicToF16IfElse(multiplyDyadic(f16BitsToDyadic(left), f16BitsToDyadic(right)));
}
export function dotF16(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length) throw new Error("Dimensões Q/K incompatíveis.");
  let accumulator = f32BitsToDyadic(0);
  for (let dimension = 0; dimension < left.length; dimension++) {
    const product = multiplyDyadic(f16BitsToDyadic(left[dimension]!), f16BitsToDyadic(right[dimension]!));
    const productF32 = f32BitsToDyadic(roundDyadicToF32IfElse(product));
    accumulator = f32BitsToDyadic(roundDyadicToF32IfElse(addDyadic(accumulator, productF32)));
  }
  return roundDyadicToF16IfElse(accumulator);
}

/** Applies the two attention probabilities to V and concatenates four heads. */
export function evaluateFixedTwoTokenAttentionValues(
  probabilities: readonly (readonly (readonly number[])[])[],
  value: readonly (readonly number[])[],
): number[][] {
  if (probabilities.length !== 4 || value.length !== 2 || value.some((row) => row.length !== 16)) {
    throw new Error("Probabilidades ou valores fora do contrato da atenção fixa.");
  }
  return Array.from({ length: 2 }, (_, token) => Array.from({ length: 4 }, (_, head) =>
    Array.from({ length: 4 }, (_, dimension) => dotF16(
      probabilities[head]![token]!,
      [value[0]![head * 4 + dimension]!, value[1]![head * 4 + dimension]!],
    ))).flat());
}

/** Fixed two-token, four-head Llama attention through the masked score boundary. */
export function evaluateFixedTwoTokenAttentionScores(
  query: readonly (readonly number[])[], key: readonly (readonly number[])[],
  cos: readonly (readonly number[])[], sin: readonly (readonly number[])[],
): FixedAttentionScores {
  if (query.length !== 2 || key.length !== 2 || cos.length !== 2 || sin.length !== 2 ||
    query.some((row) => row.length !== 16) || key.some((row) => row.length !== 16) ||
    cos.some((row) => row.length !== 4) || sin.some((row) => row.length !== 4)) {
    throw new Error("Atenção fixa requer dois tokens, quatro heads e head_dim=4.");
  }
  const rotate = (tensor: readonly (readonly number[])[]) => Array.from({ length: 4 }, (_, head) =>
    Array.from({ length: 2 }, (_, token) => Array.from({ length: 4 }, (_, dimension) => {
      const value = tensor[token]![head * 4 + dimension]!;
      const partner = tensor[token]![head * 4 + (dimension + 2) % 4]!;
      const rotated = dimension < 2 ? partner ^ 0x8000 : partner;
      return addF16(multiplyF16(value, cos[token]![dimension]!), multiplyF16(rotated, sin[token]![dimension]!));
    })));
  const qRotated = rotate(query), kRotated = rotate(key);
  const score = Array.from({ length: 4 }, (_, head) => Array.from({ length: 2 }, (_, token) =>
    Array.from({ length: 2 }, (_, keyToken) => dotF16(qRotated[head]![token]!, kRotated[head]![keyToken]!))));
  const scaled = score.map((head) => head.map((row) => row.map((value) => multiplyF16(value, 0x3800))));
  const masked = scaled.map((head) => head.map((row, token) => row.map((value, keyToken) =>
    addF16(value, keyToken > token ? 0xfbff : 0))));
  return { qRotated, kRotated, score, scaled, masked };
}
