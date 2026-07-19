import assert from "node:assert/strict";
import test from "node:test";
import {
  classifyTransientLimit,
  exponentialRetrySeconds,
  stateAfterTransientLimit,
} from "./agent-loop-retry-policy.mjs";

test("classifies quota, rate and capacity limits without classifying ordinary failures", () => {
  assert.equal(classifyTransientLimit("ERROR: You've hit your usage limit. Purchase more credits."), "usage_limit");
  assert.equal(classifyTransientLimit("ERROR: HTTP 429: too many requests"), "rate_limit");
  assert.equal(classifyTransientLimit("ERROR: Selected model is at capacity. Please try again."), "model_capacity");
  assert.equal(classifyTransientLimit("Expected exactly one handoff, found 0"), null);
});

test("does not classify historical limit text echoed inside the runner prompt", () => {
  assert.equal(classifyTransientLimit(JSON.stringify({ reason: "You've hit your usage limit" })), null);
  assert.equal(classifyTransientLimit("Prior error was: Selected model is at capacity"), null);
});

test("uses capped exponential retry delays", () => {
  assert.deepEqual([1, 2, 3, 4, 7].map((attempt) => exponentialRetrySeconds(attempt, 60, 3600)), [60, 120, 240, 480, 3600]);
});

test("a transient limit consumes neither a loop sequence nor a consecutive failure", () => {
  const state = stateAfterTransientLimit({
    currentSequence: 92,
    completedLoops: 81,
    consecutiveFailures: 4,
  }, {
    kind: "usage_limit",
    runId: "run-0092-example",
    sequence: 92,
    detectedAt: "2026-07-19T20:00:00.000Z",
    initialSeconds: 60,
    maximumSeconds: 3600,
  });
  assert.equal(state.currentSequence, 91);
  assert.equal(state.completedLoops, 81);
  assert.equal(state.consecutiveFailures, 4);
  assert.equal(state.limitRetry.attempts, 1);
  assert.equal(state.limitRetry.nextRetryAt, "2026-07-19T20:01:00.000Z");
});

test("repeated limits increase the persisted attempt and delay", () => {
  const first = stateAfterTransientLimit({ currentSequence: 92, completedLoops: 81, consecutiveFailures: 0 }, {
    kind: "usage_limit", runId: "run-a", sequence: 92, detectedAt: "2026-07-19T20:00:00.000Z", initialSeconds: 60, maximumSeconds: 3600,
  });
  const second = stateAfterTransientLimit({ ...first, currentSequence: 92 }, {
    kind: "usage_limit", runId: "run-b", sequence: 92, detectedAt: "2026-07-19T20:01:00.000Z", initialSeconds: 60, maximumSeconds: 3600,
  });
  assert.equal(second.currentSequence, 91);
  assert.equal(second.limitRetry.attempts, 2);
  assert.equal(second.limitRetry.retryAfterSeconds, 120);
});
