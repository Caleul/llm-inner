export interface Gemma4VerificationSessionCacheHit<State> {
  state: State;
  currentInputIds: number[];
  suffixTokenIds: number[];
  prefixTokensReused: number;
  cachedContextTokens: number;
  cacheScope: "session" | "shared-prefix";
  sourceSessionId: number;
}

interface Gemma4VerificationSessionCacheEntry<State> {
  tokenIds: number[];
  currentInputIds: number[];
  state: State;
  residentBytes: number;
}

export class Gemma4VerificationSessionCache<State> {
  readonly #entries = new Map<number, Gemma4VerificationSessionCacheEntry<State>>();
  readonly #maxSessions: number;

  constructor(maxSessions = 8) {
    if (!Number.isSafeInteger(maxSessions) || maxSessions < 1) throw new Error("Cache de verificação requer limite positivo de sessões.");
    this.#maxSessions = maxSessions;
  }

  resolve(sessionId: number, inputIds: readonly number[], truncateState?: (state: State, prefixTokens: number) => State): Gemma4VerificationSessionCacheHit<State> | undefined {
    validateSessionId(sessionId); validateTokenIds(inputIds, "Entrada da sessão");
    const ownEntry = this.#entries.get(sessionId);
    if (ownEntry && ownEntry.tokenIds.length <= inputIds.length && ownEntry.tokenIds.every((token, index) => inputIds[index] === token)) {
      this.#touch(sessionId, ownEntry);
      return this.#hit(sessionId, ownEntry, inputIds, ownEntry.tokenIds.length, ownEntry.state, "session");
    }

    let best: { sourceSessionId: number; entry: Gemma4VerificationSessionCacheEntry<State>; prefixTokens: number } | undefined;
    for (const [sourceSessionId, entry] of this.#entries) {
      let prefixTokens = commonPrefixLength(entry.tokenIds, inputIds);
      if (prefixTokens === inputIds.length && prefixTokens < entry.tokenIds.length) prefixTokens -= 1;
      if (prefixTokens < 1 || (prefixTokens < entry.tokenIds.length && !truncateState)) continue;
      if (!best || prefixTokens > best.prefixTokens || (prefixTokens === best.prefixTokens && (sourceSessionId === sessionId || best.sourceSessionId !== sessionId))) best = { sourceSessionId, entry, prefixTokens };
    }
    if (!best) { if (ownEntry) this.#entries.delete(sessionId); return undefined; }
    const state = best.prefixTokens === best.entry.tokenIds.length ? best.entry.state : truncateState!(best.entry.state, best.prefixTokens);
    if (ownEntry && best.sourceSessionId !== sessionId) this.#entries.delete(sessionId);
    if (this.#entries.has(best.sourceSessionId)) this.#touch(best.sourceSessionId, best.entry);
    return this.#hit(best.sourceSessionId, best.entry, inputIds, best.prefixTokens, state, best.sourceSessionId === sessionId ? "session" : "shared-prefix");
  }

  #hit(sourceSessionId: number, entry: Gemma4VerificationSessionCacheEntry<State>, inputIds: readonly number[], prefixTokens: number, state: State, cacheScope: "session" | "shared-prefix"): Gemma4VerificationSessionCacheHit<State> {
    return {
      state,
      currentInputIds: prefixTokens === entry.tokenIds.length ? [...entry.currentInputIds] : [],
      suffixTokenIds: inputIds.slice(prefixTokens),
      prefixTokensReused: prefixTokens,
      cachedContextTokens: prefixTokens,
      cacheScope,
      sourceSessionId,
    };
  }

  #touch(sessionId: number, entry: Gemma4VerificationSessionCacheEntry<State>): void {
    this.#entries.delete(sessionId); this.#entries.set(sessionId, entry);
  }

  update(sessionId: number, tokenIds: readonly number[], currentInputIds: readonly number[], state: State, residentBytes: number): void {
    validateSessionId(sessionId); validateTokenIds(tokenIds, "Contexto da sessão"); validateTokenIds(currentInputIds, "Chunk oculto da sessão");
    if (currentInputIds.length > tokenIds.length || !Number.isSafeInteger(residentBytes) || residentBytes < 0) throw new Error("Estado residente da sessão de verificação é inválido.");
    this.#entries.delete(sessionId);
    this.#entries.set(sessionId, { tokenIds: [...tokenIds], currentInputIds: [...currentInputIds], state, residentBytes });
    while (this.#entries.size > this.#maxSessions) this.#entries.delete(this.#entries.keys().next().value!);
  }

  get sessions(): number { return this.#entries.size; }
  get residentBytes(): number { return [...this.#entries.values()].reduce((total, entry) => total + entry.residentBytes, 0); }
}

function validateSessionId(value: number): void {
  if (!Number.isSafeInteger(value) || value < 1 || value > 0xffff_ffff) throw new Error("ID da sessão de verificação é inválido.");
}

function validateTokenIds(values: readonly number[], label: string): void {
  if (values.length < 1 || values.some((token) => !Number.isSafeInteger(token) || token < 0)) throw new Error(`${label} contém tokens inválidos.`);
}

function commonPrefixLength(left: readonly number[], right: readonly number[]): number {
  const limit = Math.min(left.length, right.length);
  let index = 0;
  while (index < limit && left[index] === right[index]) index += 1;
  return index;
}
