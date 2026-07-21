import assert from "node:assert/strict";
import test from "node:test";
import { rankGemma4TerminalLogits } from "../src/gemma4-terminal-logits.js";

test("ranqueia logits do último passo com desempate greedy pelo menor token", () => {
  const ranked = rankGemma4TerminalLogits({ shape: [1, 2, 4], values: Float32Array.from([99, 98, 97, 96, 3, 7, 7, -1]) }, 3);
  assert.deepEqual(ranked, [{ tokenId: 1, logit: 7 }, { tokenId: 2, logit: 7 }, { tokenId: 0, logit: 3 }]);
  assert.throws(() => rankGemma4TerminalLogits({ shape: [1, 1, 2], values: Float32Array.from([0, NaN]) }), /não finito/);
});
