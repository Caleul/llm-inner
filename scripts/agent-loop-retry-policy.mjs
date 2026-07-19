const LIMIT_PATTERNS = [
  { kind: "usage_limit", pattern: /(?:^|\n)ERROR:\s*(?:you(?:'|’)ve hit your usage limit|[^\n]*purchase more credits)/i },
  { kind: "rate_limit", pattern: /(?:^|\n)ERROR:\s*[^\n]*(?:rate limit(?:ed| exceeded)?|too many requests|\bHTTP\s*429\b)/i },
  { kind: "model_capacity", pattern: /(?:^|\n)ERROR:\s*selected model is at capacity/i },
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
  const attempts = (state.limitRetry?.attempts ?? 0) + 1;
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
    limitRetry: {
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
