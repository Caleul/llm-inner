export interface Gemma4VerificationSessionCacheHit<State> {
  state: State;
  currentInputIds: number[];
  suffixTokenIds: number[];
  prefixTokensReused: number;
  cachedContextTokens: number;
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

  resolve(sessionId: number, inputIds: readonly number[]): Gemma4VerificationSessionCacheHit<State> | undefined {
    validateSessionId(sessionId); validateTokenIds(inputIds, "Entrada da sessão");
    const entry = this.#entries.get(sessionId);
    if (!entry) return undefined;
    const matches = entry.tokenIds.length <= inputIds.length && entry.tokenIds.every((token, index) => inputIds[index] === token);
    if (!matches) { this.#entries.delete(sessionId); return undefined; }
    this.#entries.delete(sessionId); this.#entries.set(sessionId, entry);
    return {
      state: entry.state,
      currentInputIds: [...entry.currentInputIds],
      suffixTokenIds: inputIds.slice(entry.tokenIds.length),
      prefixTokensReused: entry.tokenIds.length,
      cachedContextTokens: entry.tokenIds.length,
    };
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
