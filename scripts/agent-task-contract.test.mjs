import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  OUTCOME,
  buildAgentTaskPrompt,
  buildContinuityRecord,
  classifyObjectiveOutcome,
  resolveObjectiveStatus,
} from "./agent-task-contract.mjs";

test("builds an isolated engineering task without controller context", () => {
  const prompt = buildAgentTaskPrompt("# Task\nImplement Gemma.", "Produce the Gemma artifact.");
  assert.match(prompt, /Implement Gemma/);
  assert.match(prompt, /Produce the Gemma artifact/);
  assert.doesNotMatch(prompt, /loop|round|cycle|handoff|sequence|runId/i);
});

test("keeps a configured task fixture and repository guidance free of controller context", async () => {
  const fixtureRoot = new URL("./fixtures/agent-task/", import.meta.url);
  const config = JSON.parse(await readFile(new URL("config.json", fixtureRoot), "utf8"));
  const [master, guidance] = await Promise.all([
    readFile(new URL(config.promptFile, fixtureRoot), "utf8"),
    readFile(new URL("../AGENTS.md", import.meta.url), "utf8"),
  ]);
  const prompt = buildAgentTaskPrompt(master, config.missionGoal);
  assert.match(prompt, /Safetensors/);
  for (const text of [prompt, guidance]) {
    assert.doesNotMatch(text, /autonomous loop|next (?:agent|round|cycle)|previous (?:agent|handoff)|runId/i);
  }
  assert.doesNotMatch(prompt, /\.agent-loop|currentSequence|completedLoops|HANDOFF-/i);
});

test("requires exactly one explicit objective result", () => {
  assert.equal(classifyObjectiveOutcome("OBJECTIVE_COMPLETE: already proven"), OUTCOME.complete);
  assert.equal(classifyObjectiveOutcome("notes\nOBJECTIVE_ADVANCED: implemented"), OUTCOME.advanced);
  assert.equal(classifyObjectiveOutcome("OBJECTIVE_BLOCKED: source unavailable"), OUTCOME.blocked);
  assert.equal(classifyObjectiveOutcome("done"), OUTCOME.invalid);
  assert.equal(classifyObjectiveOutcome("OBJECTIVE_COMPLETE: x\nOBJECTIVE_ADVANCED: y"), OUTCOME.invalid);
});

test("prevents an implementation session from certifying its own changes", () => {
  assert.equal(resolveObjectiveStatus({ outcome: OUTCOME.complete, repositoryChanged: false }), OUTCOME.complete);
  assert.equal(resolveObjectiveStatus({ outcome: OUTCOME.complete, repositoryChanged: true }), OUTCOME.advanced);
  assert.equal(resolveObjectiveStatus({ outcome: OUTCOME.advanced, repositoryChanged: true }), OUTCOME.advanced);
  assert.equal(resolveObjectiveStatus({ outcome: OUTCOME.advanced, repositoryChanged: false }), OUTCOME.invalid);
  assert.equal(resolveObjectiveStatus({ outcome: OUTCOME.blocked, repositoryChanged: false }), OUTCOME.blocked);
});

test("creates internal continuity records without assigning future work", () => {
  const record = buildContinuityRecord({
    sequence: 1,
    runId: "run-0001",
    createdAt: "2026-07-19T00:00:00.000Z",
    status: OUTCOME.advanced,
    summary: "OBJECTIVE_ADVANCED: implemented navigation",
    endingCommit: "abc123",
    validation: [{ command: "npm test", status: "passed" }],
    missionGateIds: ["gemma"],
  });
  assert.equal(record.missionStatus, "continue");
  assert.equal(record.endingState.gitCommit, "abc123");
  assert.ok(!("nextSteps" in record));
  assert.ok(!("nextRecommendedMilestone" in record));
});
