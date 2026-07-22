import assert from "node:assert/strict";
import test from "node:test";
import { selectGemma4PagedTerminalHidden } from "../src/gemma4-paged-text.js";

test("epílogo greedy projeta somente o hidden terminal de cada batch", () => {
  const selected = selectGemma4PagedTerminalHidden(
    [[10, 11, 12], [20, 21, 22]],
    { shape: [2, 3, 2], values: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) },
  );
  assert.deepEqual(selected.inputIds, [[12], [22]]);
  assert.deepEqual(selected.hidden.shape, [2, 1, 2]);
  assert.deepEqual([...selected.hidden.values], [5, 6, 11, 12]);
  assert.equal(selected.positionsAvoided, 4);
});

test("seleção terminal rejeita shape e payload incompatíveis", () => {
  assert.throws(() => selectGemma4PagedTerminalHidden([[1, 2]], { shape: [1, 1, 2], values: Float32Array.of(1, 2) }), /shape incompatível/);
  assert.throws(() => selectGemma4PagedTerminalHidden([[1, 2]], { shape: [1, 2, 2], values: Float32Array.of(1, 2) }), /payload incompatível/);
});
