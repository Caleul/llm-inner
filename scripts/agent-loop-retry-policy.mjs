const LIMIT_PATTERNS = [
  { kind: "usage_limit", pattern: /(?:^|\n)(?:ERROR:|Error:|[^\n]*"(?:error|message)"\s*:)[^\n]*(?:usage limit|quota exceeded|insufficient credits|credits? (?:are )?exhausted|(?:purchase|requires?) more credits|can only afford|upgrade (?:your plan|to a paid account)|\bstatusCode["']?\s*:\s*402\b)/i },
  { kind: "rate_limit", pattern: /(?:^|\n)(?:ERROR:|Error:|[^\n]*"(?:error|message)"\s*:)[^\n]*(?:rate limit(?:ed| exceeded)?|too many requests|\bHTTP\s*429\b|\bstatus(?:Code)?["']?\s*[:=]\s*429\b)/i },
  { kind: "model_capacity", pattern: /(?:^|\n)(?:ERROR:|Error:|[^\n]*"(?:error|message)"\s*:)[^\n]*(?:model is at capacity|selected model is at capacity|service overloaded|temporarily unavailable)/i },
  { kind: "provider_unavailable", pattern: /(?:^|\n)(?:ERROR:|Error:|[^\n]*"(?:error|message)"\s*:)[^\n]*(?:not logged in|authentication required|failed to authenticate|connection refused|network is unreachable|could not resolve host)/i },
];

export function classifyTransientLimit(output) {
  for (const candidate of LIMIT_PATTERNS) {
    if (candidate.pattern.test(output)) return candidate.kind;
  }
  return null;
}

export function exponentialRetrySeconds(attempt, initialSeconds, maximumSeconds) {
  if (!Number.isInteger(attempt) || attempt < 1) throw new Error("Retry attempt must be a positive integer.");
  if (!Number.isInteger(initialSeconds) || initialSeconds < 1) throw new Error("Initial retry seconds must be positive.");
  if (!Number.isInteger(maximumSeconds) || maximumSeconds < initialSeconds) {
    throw new Error("Maximum retry seconds must be at least the initial retry seconds.");
  }
  return Math.min(maximumSeconds, initialSeconds * (2 ** Math.min(attempt - 1, 30)));
}

export function stateAfterTransientLimit(state, options) {
  const legacyRetry = !state.providerRetries
    && (!state.limitRetry?.providerId || state.limitRetry.providerId === options.providerId)
    ? state.limitRetry
    : undefined;
  const previous = state.providerRetries?.[options.providerId] ?? legacyRetry;
  const attempts = (previous?.attempts ?? 0) + 1;
  const retryAfterSeconds = exponentialRetrySeconds(attempts, options.initialSeconds, options.maximumSeconds);
  const nextRetryAt = new Date(Date.parse(options.detectedAt) + retryAfterSeconds * 1000).toISOString();
  return {
    ...state,
    status: "waiting_limit",
    // A rejected transport attempt is not an autonomous loop. Reuse the same
    // sequence once the provider accepts work again.
    currentSequence: options.sequence - 1,
    consecutiveFailures: state.consecutiveFailures,
    lastCompletedAt: options.detectedAt,
    providerRetries: {
      ...(state.providerRetries ?? {}),
      [options.providerId]: {
        providerId: options.providerId,
        kind: options.kind,
        attempts,
        runId: options.runId,
        attemptedSequence: options.sequence,
        detectedAt: options.detectedAt,
        retryAfterSeconds,
        nextRetryAt,
      },
    },
    limitRetry: {
      providerId: options.providerId,
      kind: options.kind,
      attempts,
      runId: options.runId,
      attemptedSequence: options.sequence,
      detectedAt: options.detectedAt,
      retryAfterSeconds,
      nextRetryAt,
    },
  };
}

export function clearProviderRetry(state, providerId) {
  if (!state.providerRetries?.[providerId]) return state;
  const providerRetries = { ...state.providerRetries };
  delete providerRetries[providerId];
  return { ...state, providerRetries };
}
