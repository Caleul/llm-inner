# llm-inner agent instructions

## Autonomous loop boundary

This repository is advanced by one Codex process at a time. The external
runner owns process creation, sequencing, state transitions and stop handling.
An agent must never start another agent or invoke the runner.

## Every loop

1. Read `agent-loop.config.json`, `.agent-loop/state.json`, the master prompt,
   the most recent accepted handoff and the current repository state.
2. Select one bounded, highest-impact unfinished milestone from live evidence.
3. Implement a coherent vertical slice; do not manufacture broad scaffolding.
4. Run the configured validation commands and record their real results.
5. Update documentation when behavior or support changes.
6. Create one local Git commit for the loop.
7. Write exactly one valid handoff atomically to
   `.agent-loop/handoffs/completed/` as the final repository-changing action.

## Safety and fidelity

- Preserve fail-closed behavior for unsupported model semantics.
- Never claim validation, equivalence or a commit that did not occur.
- Do not modify `.agent-loop/state.json`, lock files or the runner's run logs.
- Do not use network resources, credentials or destructive commands unless the
  current milestone makes that necessary and the repository instructions allow it.
- If `.agent-loop/STOP` appears, finish only the coherent in-progress work,
  create the final handoff with `missionStatus: "blocked"`, then exit.
