# llm-inner agent instructions

## Autonomous loop boundary

This repository is advanced by one Codex process at a time. The external
runner owns process creation, sequencing, state transitions and stop handling.
An agent must never start another agent or invoke the runner.

## Every loop

1. Read `agent-loop.config.json`, `.agent-loop/state.json`, the master prompt,
   the most recent accepted handoff and the current repository state.
2. Select one substantial, highest-impact unfinished milestone from live evidence.
3. Use the available loop budget for a complete multi-layer vertical slice when
   the evidence supports it. Do not stop after a cosmetic, isolated or
   mechanical micro-change merely to produce a handoff.
4. Run the configured validation commands and record their real results.
5. Update documentation when behavior or support changes.
6. Create one local Git commit for the loop.
7. Write exactly one concise valid handoff atomically to
   `.agent-loop/handoffs/completed/` as the final repository-changing action.

## Safety and fidelity

- Preserve fail-closed behavior for unsupported model semantics.
- Never claim validation, equivalence or a commit that did not occur.
- A handoff is continuity context, not a work diary: retain only the completed
  outcome, real validation evidence, unresolved bottlenecks and the next
  highest-impact milestone with acceptance criteria.
- Act as an owner of the mission, not as a ticket executor. Reassess the whole
  system and connect adjacent layers when that removes a material fidelity or
  validation gap. A passing narrow test alone is not a reason to end a cycle
  when a larger coherent acceptance boundary remains reachable.
- Do not modify `.agent-loop/state.json`, lock files or the runner's run logs.
- Do not use network resources, credentials or destructive commands unless the
  current milestone makes that necessary and the repository instructions allow it.
- If `.agent-loop/STOP` appears, finish only the coherent in-progress work,
  create the final handoff with `missionStatus: "blocked"`, then exit.
