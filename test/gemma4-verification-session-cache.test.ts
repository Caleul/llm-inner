import assert from "node:assert/strict";
import test from "node:test";
import { Gemma4VerificationSessionCache } from "../src/gemma4-verification-session-cache.js";

test("cache exato reutiliza somente um prefixo token a token da mesma sessão", () => {
  const cache = new Gemma4VerificationSessionCache<{ id: string }>();
  cache.update(7, [1, 2, 3], [3], { id: "exact" }, 128);
  assert.deepEqual(cache.resolve(7, [1, 2, 3, 4, 5]), { state: { id: "exact" }, currentInputIds: [3], suffixTokenIds: [4, 5], prefixTokensReused: 3, cachedContextTokens: 3 });
  assert.equal(cache.residentBytes, 128);
  assert.equal(cache.resolve(8, [1, 2, 3]), undefined);
  assert.equal(cache.resolve(7, [1, 9, 3]), undefined);
  assert.equal(cache.sessions, 0);
});

test("cache exato limita sessões por LRU e valida o estado residente", () => {
  const cache = new Gemma4VerificationSessionCache<number>(2);
  cache.update(1, [1], [1], 1, 10);
  cache.update(2, [2], [2], 2, 20);
  assert.equal(cache.resolve(1, [1])?.state, 1);
  cache.update(3, [3], [3], 3, 30);
  assert.equal(cache.resolve(2, [2]), undefined);
  assert.equal(cache.resolve(1, [1])?.state, 1);
  assert.equal(cache.resolve(3, [3])?.state, 3);
  assert.equal(cache.residentBytes, 40);
  assert.throws(() => cache.update(4, [4], [4, 5], 4, 1), /Estado residente/);
});
