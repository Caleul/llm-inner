export const OUTCOME = Object.freeze({
  complete: "complete",
  advanced: "advanced",
  blocked: "blocked",
  invalid: "invalid",
});

export function buildAgentTaskPrompt(masterPrompt, missionGoal) {
  return `${masterPrompt.trim()}\n\n## Exact objective\n\n${missionGoal.trim()}\n`;
}

export function classifyObjectiveOutcome(message) {
  const matches = [...message.matchAll(/(?:^|\n)OBJECTIVE_(COMPLETE|ADVANCED|BLOCKED):/g)];
  if (matches.length !== 1) return OUTCOME.invalid;
  return matches[0][1].toLowerCase();
}

export function resolveObjectiveStatus({ outcome, repositoryChanged }) {
  if (outcome === OUTCOME.complete && !repositoryChanged) return OUTCOME.complete;
  if (outcome === OUTCOME.blocked && !repositoryChanged) return OUTCOME.blocked;
  if (repositoryChanged && (outcome === OUTCOME.advanced || outcome === OUTCOME.complete)) {
    return OUTCOME.advanced;
  }
  return OUTCOME.invalid;
}

export function buildContinuityRecord({
  sequence,
  runId,
  createdAt,
  status,
  summary,
  endingCommit,
  validation,
  missionGateIds,
}) {
  const complete = status === OUTCOME.complete;
  const blocked = status === OUTCOME.blocked;
  return {
    schemaVersion: 1,
    sequence,
    runId,
    createdAt,
    missionStatus: complete ? "complete" : blocked ? "blocked" : "continue",
    loopStatus: "success",
    title: complete ? "Gemma objective independently verified" : blocked ? "Gemma objective externally blocked" : "Gemma objective advanced",
    summary: summary.trim().slice(0, 12_000),
    validation: { commands: validation },
    missionGates: {
      satisfied: complete ? missionGateIds : [],
      unsatisfied: complete ? [] : missionGateIds,
    },
    endingState: { gitCommit: endingCommit, dirty: false },
    knownLimitations: [],
    blockedBy: blocked ? [summary.trim().slice(0, 2_000)] : [],
    bottlenecks: [],
  };
}
