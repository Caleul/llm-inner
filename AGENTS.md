# llm-inner autonomous ownership contract

## Autonomous loop boundary

This repository is advanced by one Codex process at a time. The external
runner owns process creation, sequencing, state transitions and stop handling.
An agent must never start another agent or invoke the runner.

## Operating posture

Each invocation is a fresh senior-engineering work session, not a queue worker
consuming the previous handoff. The agent owns the outcome, diagnoses the
current system independently, forms an internal high-level plan, makes the
necessary decisions from repository evidence, and advances the mission as far
as the available context permits. It proposes the next strategic move rather
than merely reporting the next local task.

The loop is deliberately configured for `gpt-5.6-terra` at high reasoning
effort. Use that reasoning capacity on architecture, semantic boundaries,
validation strategy, tradeoffs and long-term maintainability — not on slicing a
single coherent problem into artificial microtasks.

## Work-session contract

1. Read `agent-loop.config.json`, `.agent-loop/state.json`, the master prompt,
   the most recent accepted handoff and the current repository state.
2. Independently map mission gates, architectural risks and validation gaps;
   do not inherit a narrow task from the previous handoff without re-justifying it.
3. Select the largest coherent strategic boundary reachable from that evidence,
   and execute across all adjacent layers needed to close it.
4. Keep working through related implementation, tests, reports and documentation
   until the boundary is closed or a genuine higher-level blocker is proven.
5. Run the configured validation commands and record their real results.
6. Perform an architectural self-review: cohesion, duplication, dependency
   direction, naming, error boundaries, testability and extension seams.
7. Update documentation when behavior, support or architecture changes.
8. Create one local Git commit for the loop.
9. Write exactly one concise decision brief atomically to
  `.agent-loop/handoffs/completed/` as the final repository-changing action.

## Primary deliverable: Safetensors-to-literal-calculation JSON decompilation

The central product is a decompiler: it receives a `.safetensors` model package
(and its authoritative metadata/architecture evidence) and emits a lossless
calculation JSON artifact. This is not merely a tensor catalog, diagram,
benchmark, trace or differential report. For every supported package, a later
reader must reproduce the forward pass and generation from only declared input
variables and that JSON, without reopening the source checkpoint or inferring
hidden behavior from source code.

The artifact must declare an ordered assignment graph: named inputs such as
`x[i]`, token IDs and positions; every named intermediate such as
`x_embedding[i]`, `x_med[i]`, `x_alguma_variacao[i]` and
`x_para_proxima_camada[i]`; its operation and ordered inputs; exact
tensor/constant values; shapes/layouts/dtypes; casts; accumulation order;
masks; cache transitions; and final outputs. Names may be improved for the
actual domain, but every value must have a stable declaration and provenance.
A downstream assignment, including the input to layer 2, must explicitly
reference the named result calculated by the preceding layer — never an
implicit layer shortcut or a hidden generic-decoder invocation.

All weights required for replay must be embedded losslessly in the JSON
artifact. For large or quantized tensors, an exact binary payload encoded in
JSON is allowed only when dtype, endianness, shape, layout, packing, scale,
zero/bias/codebook metadata and a deterministic decoding assignment are also
declared. The artifact still may not require the original checkpoint.

“Model-type agnostic” means the pipeline has a generic Safetensors container
path and can add explicit semantic adapters or authoritative runtime-graph
extraction for new architectures. It never means guessing architecture from
raw weights. If package semantics cannot be established from metadata, a
registered adapter or an authoritative graph, export must fail closed and name
the missing semantic contract. Completion requires a generic path that can
lower every established Safetensors semantic contract into this literal JSON,
not a collection of architecture-specific prose or external references.

## Mandatory checkpoint roadmap

The next product checkpoint is not a generic feature count. It is a real,
unquantized Gemma 4 Safetensors package transformed into and replayed from a
lossless literal calculation JSON. Until that checkpoint is proven, it outranks
new work on unrelated families, quantization variants, container breadth and
cosmetic infrastructure. The agent has authority to download the required
official/public model files, `config.json`, tokenizer/configuration metadata,
and authoritative runtime/source evidence needed to establish its semantics.

Gemma 4 is complete only when the selected immutable dense package has a
documented source/revision/checksum; every model semantic is explicitly
lowered; every original weight is embedded losslessly; forward and generation
can replay after the source checkpoint is unavailable; and an authoritative
runtime comparison establishes the claimed fidelity. On success, create and
commit `.agent-loop/checkpoints/gemma4-dense-lossless/` containing a concise
`manifest.json` and `CHECKPOINT.md` with source identity, artifact hash/path,
validation commands/results, fidelity evidence, and known limits. Do not create
this success folder for a partial adapter, synthetic fixture, catalog, or
source-dependent IR.

Only after that Gemma 4 checkpoint exists may GLM 5.2 become the primary
target. Apply the same rules and create and commit
`.agent-loop/checkpoints/glm-5.2-dense-lossless/` only after a real dense GLM
5.2 Safetensors package is losslessly exported and independently replayed.
If a requested family/version/package cannot be acquired or its semantics are
not authoritatively established, record the exact acquisition/semantic blocker
and continue the highest-leverage work that unblocks that checkpoint; never
substitute a smaller, older, quantized, synthetic, or merely similar model and
call the checkpoint reached.

## Safety and fidelity

- Preserve fail-closed behavior for unsupported model semantics.
- Never call an export complete when it contains external tensor references,
  preview-only weights, omitted intermediates, implicit configuration, or a
  mathematical dimension truncated for presentation.
- Never claim validation, equivalence or a commit that did not occur.
- A handoff is continuity context, not a work diary: retain only the completed
  outcome, real validation evidence, unresolved bottlenecks and the next
  highest-impact milestone with acceptance criteria.
- Act as an owner of the mission, not as a ticket executor. Reassess the whole
  system and connect adjacent layers when that removes a material fidelity or
  validation gap. A passing narrow test alone is not a reason to end a cycle
  when a larger coherent acceptance boundary remains reachable.
- Follow Clean Code and SOLID as engineering tools, not ceremony: keep each
  module focused on one responsibility, make format/architecture/runtime
  extensions additive behind explicit interfaces, depend on contracts rather
  than incidental concrete implementations, preserve clear domain names and
  fail-closed error boundaries, and remove duplication that can drift across
  supported model paths.
- Do not hand off because a local edit is complete. Hand off only after either
  a material mission boundary is closed, the available context has exposed a
  genuine external/semantic blocker, or the remaining work is a distinct
  higher-level boundary. Use the context budget to improve the system around
  the problem rather than reproducing a pattern of adjacent microtasks.
- A single helper, fixture, assertion, decoder variant, test, or narrow commit
  is normally an intermediate step. It becomes a handoff boundary only when it
  closes a material mission gate or exposes a new external/semantic decision.
- Do not modify `.agent-loop/state.json`, lock files or the runner's run logs.
- Do not use network resources, credentials or destructive commands unless the
  current milestone makes that necessary and the repository instructions allow it.
- If `.agent-loop/STOP` appears, finish only the coherent in-progress work,
  create the final handoff with `missionStatus: "blocked"`, then exit.
