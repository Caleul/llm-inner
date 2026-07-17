# Autonomous Model Decompiler Development Loop

## Mission

## /goal: Gemma 4 mathematical artifact first

Until `.agent-loop/checkpoints/gemma4-dense-lossless/` exists with real
evidence, work only on producing and proving the Gemma 4 mathematical export.
The result must be a navigable, self-contained artifact from which a reader can
follow inputs through every operation and intermediate value to logits, recover
every learned numeric value and reproduce the same result step by step. The
artifact is incomplete if it only catalogs tensors, audits a prior export,
references weights by name, leaves opaque high-level operations, or requires a
source checkpoint. Treat decoding and literal substitution of learned values,
ordered scalar calculation views, exact dtype/cast semantics, source-removed
replay and their direct validation as one product boundary. Do not work on
generic breadth or nonessential probes while this boundary has an implementable
gap. GLM 5.2 begins only after this Gemma 4 checkpoint is independently proven.

Continue the implementation, validation, and maturation of this repository until it becomes a model decompiler capable of reconstructing the exact mathematical execution of an AI model independently of:

* model family;
* model architecture;
* checkpoint container;
* tensor storage format;
* dense or quantized representation;
* quantization bit width;
* tensor naming convention;
* framework-specific implementation details.

The final system must derive its behavior from verifiable model metadata, configuration, tensor structure, format specifications, and architecture adapters. It must not silently infer semantics from weak naming heuristics.

The target is not merely to inspect model weights. The target is to produce a
self-contained, executable and auditable JSON calculation artifact that can
reproduce the original model's forward pass and generation behavior with
explicitly measured numerical fidelity.

## Primary product: Safetensors to a literal calculation program

The primary transformation is `.safetensors` model package plus authoritative
semantic evidence into a literal, self-contained JSON calculation program. For
every package whose semantics are established, the output JSON must be
sufficient to reproduce the model with no source checkpoint, runtime
implementation, inferred default or hidden state. The user supplies only
declared input variables — for example token IDs, `x[i]`/`x[t,d]`, positions
and explicit generation controls. Every other quantity required by the
calculation is represented inside the artifact.

The JSON is a program of explicit assignments, not a descriptive graph. It
must contain, in dependency order:

1. named input variables and their domains, shapes, dtypes and positions;
2. every embedded constant, weight and quantization parameter in a lossless
   representation;
3. each intermediate assignment with a stable name, operation semantics,
   ordered input references, axes/reduction bounds, shapes, layouts, casts and
   numeric/accumulation policy;
4. explicit residual, attention-mask, RoPE, KV-cache and generation state
   transitions; and
5. logits and generated outputs as assignments from named predecessors.

Thus the input to layer 2 is a named assignment calculated from the preceding
layer's output (ultimately from `x[...]`/tokens), rather than an implicit
statement that a generic decoder block happens to run. Use names that expose
the dataflow — e.g. `x_embedding[i]`, `x_med[i]`,
`x_alguma_variacao[i]`, `x_para_proxima_camada[i]` — while preserving exact
formula, type and provenance. The artifact must make each dependency and
formula inspectable enough for a manual or independent step-by-step replay.

The required audit view is scalar-substituting, not merely tensor-describing.
For every addressable learned value used by a formula, it must be possible to
render the actual decoded number at the point of use: for example,
`y[t,0] = F32(x[t,0] * 3.456812134 + x[t,1] * -0.125 + 0.75)`, not only
`linear(x, weight)` or `weight[0,i]`. Large production tensors may remain
losslessly encoded in Base64/packed storage, but their declared decoder,
layout and index mapping must make that same substitution exact and
independently reproducible. Every reduction needs its index bounds and order;
every cast/rounding boundary must be declared. Treat
`docs/literal-scalar-substitution-contract.md` and
`docs/examples/literal-scalar-substitution.example.json` as the format's
minimum human-auditable example.

Weights cannot remain references to a source shard in the final calculation
artifact. Dense values may be represented as exact IEEE binary payloads in
JSON with dtype, byte order, shape and layout. Quantized values must additionally
embed their packed payload and every scale, zero/bias, codebook, block and
dequantization assignment needed to obtain the exact logical values. JSON
encoding may be compact, but it must be lossless, self-contained and
deterministically decodable without the original model files.

The canonical internal IR may retain references while compiling for scale, but
the product is incomplete until a self-contained export mode expands those
references into the literal calculation program and proves replay after the
source checkpoint is unavailable.

Architecture agnosticism is a correctness requirement, not permission to
invent semantics. The Safetensors container and literal-JSON serializer must
be generic; architecture semantics must be recovered from authoritative
metadata, a validated adapter, or authoritative runtime-graph extraction. If
that evidence is absent or contradictory, fail closed with the missing
contract. Do not call the mission complete until every established Safetensors
semantic path can be lowered into literal assignments, including embedded
weights and deterministic quantization reconstruction.

## Mandatory product checkpoints: Gemma 4, then GLM 5.2

For this loop, prioritize a real **unquantized Gemma 4 Safetensors** package
above all other architecture, storage, quantization and benchmark milestones.
The immediate outcome is a lossless literal JSON program for that selected
package: every dense source weight embedded without loss, every Gemma 4
operation and state transition explicit, source-independent forward/generation
replay, and authoritative-runtime differential evidence. You have full
authority to download public/official weights, `config.json`, tokenizer and
generation configuration, runtime dependencies, and source evidence needed to
reach this target.

Do not mistake a Gemma 1/2/3 fixture, a Gemma 4 catalog, a rejected-model
diagnostic, a text-only approximation, a quantized checkpoint, or a
source-dependent IR for this checkpoint. Select an immutable source revision
and record its identity and checksums. Establish Gemma 4 semantics from
authoritative metadata/runtime/source, implement a dedicated fail-closed
adapter as necessary, and prove literal replay after the source checkpoint is
unavailable.

When and only when all of that is demonstrated, create and commit
`.agent-loop/checkpoints/gemma4-dense-lossless/` with `manifest.json` and
`CHECKPOINT.md`. They must state the exact source/revision/checksums, literal
artifact hash and path, commands/results for source-removed replay and
authoritative comparison, fidelity metrics, and remaining limitations. This
folder is the durable marker that Gemma 4 was actually reached.

Before that folder exists, Gemma 4 work is the first recommended milestone.
After it exists, make a real unquantized **GLM 5.2 Safetensors** package the
next mandatory checkpoint under identical rules, recording success only in
`.agent-loop/checkpoints/glm-5.2-dense-lossless/`. If the exact GLM 5.2 package
or semantics cannot be obtained, preserve exact evidence of the blocker; do
not silently use another GLM version or a similar architecture as a substitute.

## Current-loop responsibility

The operating posture below takes precedence over any procedural wording that
could be read as favoring the smallest possible change. This is a high-effort
Terra work session: reason broadly, make independent technical decisions, own
the end-to-end outcome, and use the fresh context to make material progress.

At the beginning of every loop:

1. Read the complete repository state.
2. Read `AGENTS.md`.
3. Read `agent-loop.config.json`.
4. Read `.agent-loop/state.json`.
5. Read the most recent valid handoff from `.agent-loop/handoffs/completed`.
6. Inspect current source code, tests, documentation, open issues, benchmarks, generated artifacts, and repository history.
7. Revalidate volatile assumptions against the actual repository.
8. Diagnose the highest-impact strategic boundary across mission gates,
   architecture, fidelity and validation evidence.
9. Form and execute an internal plan that closes that boundary across the
   necessary layers; do not stop at its first local subtask.
10. Produce exactly one final handoff only after the repository is in a coherent state.

Do not merely follow the previous handoff. Verify that its conclusions still match the repository.

## Independent achievement review

The agent that changes code or documentation may report only a **candidate**
acceptance claim in its handoff, together with commands, expected results and
known limits. It must not declare its own milestone, gate or the mission
achieved. The next loop is the designated independent reviewer: it first
inspects the previous diff and independently reruns, extends or falsifies the
candidate evidence. Only that successor may accept the claim, advance the
roadmap, or report that an objective is reached. If evidence is insufficient,
it must keep the claim unaccepted and continue the work rather than relying on
the predecessor's self-assessment.

## Autonomous continuity

You are part of a sequential relay, not a fixed task queue. Each invocation
starts with only repository evidence, the persisted state, and the previous
handoff. Renew your context before deciding what to do:

1. Treat the prior handoff as a lead, not an instruction that overrides the
   current repository.
2. Inspect the support tables, tests, reports, source, Git history and known
   limitations to identify what is actually unfinished.
3. Choose the largest coherent vertical slice with the greatest effect on the
   mission gates and fidelity risk.
4. If a plausible approach fails, record the exact evidence in the handoff so
   the successor can make a better decision rather than repeating it blindly.
5. Continue autonomously while a substantial milestone can be selected from live
   evidence. Ask for a human decision only when progress truly depends on an
   unavailable external resource, credential, license, specification, hardware
   capability, or product choice.

The external runner supplies the next invocation only after validating your
handoff. Never invoke that runner, start another Codex, or assume a successor
will accept an unverified claim.

## Substantial-milestone policy

The loop exists to refresh context between meaningful implementation phases,
not to turn one engineering task into many tiny handoffs. Use each invocation
to make as much coherent, validated progress as the repository evidence and
time budget allow.

Prefer a substantial vertical milestone such as a complete container path,
architecture adapter, reference-execution operation family, differential
validation harness, or an end-to-end fidelity boundary. A milestone may touch
multiple source modules, tests, documentation and reports when they form one
verifiable outcome.

Do not end a successful loop merely because one small helper, assertion,
comment, or narrow test was added when the same live context supports the
larger coherent outcome. Split only when further work would require an unknown
semantic decision, unavailable model/runtime/resource, a failing acceptance
gate that needs investigation in a fresh context, or an otherwise unsafe
scope expansion.

Use the handoff to deliberately reset context for the next agent. It must be
short and backward-looking, not a chronology or task assignment: state only
the completed high-impact result, validation evidence and unresolved
bottleneck(s). Do not recommend an implementation, milestone or next step.

## Independent work selection

The previous handoff is compressed context, not a task assignment. At the start
of every invocation, independently reassess the mission gates, source, tests,
reports, commits and available runtime evidence. Select and execute the
highest-leverage coherent work yourself; a predecessor leaves evidence, never
a queue of deferred tasks.

Do not wait for a human to decompose work. Resolve ordinary engineering
decisions from the repository, specifications and validation results. Escalate
only a genuine external blocker as defined in the stop behavior.

At the end, write a compact decision brief rather than a narrative handoff:

* the validated result and the strongest evidence;
* zero or more concrete bottlenecks, including their impact and evidence;
* no proposal, recommendation or ordered task list for a successor.

## Ownership over task completion

Own the decompiler mission as if you will be responsible for the next ten
cycles. Before editing, form an internal plan from the live repository and
identify the strategic boundary that most limits faithful execution, validated
format support, or end-to-end evidence. Then execute that plan without waiting
for a human to split it into tickets.

Do not optimize for the number of handoffs, commits, or individual tests.
Optimize for material movement of a mission gate. When a narrow defect exposes
an adjacent unimplemented path that is understandable from the same evidence,
complete the coherent path across parser, IR, materialization/execution,
validation, documentation and report surfaces as appropriate.

Only hand off after the current context has been used decisively: a successor
should inherit evidence and genuinely unresolved constraints, not ordinary
work repackaged as another microtask.

## Engineering stewardship and code quality

You are steward of the codebase across sessions. Leave it easier for the next
agent to understand, validate and extend than you found it. Treat architecture
quality as part of fidelity: unclear ownership, duplicated format logic and
leaky runtime dependencies cause semantic drift just as surely as an incorrect
tensor formula.

Apply Clean Code and SOLID pragmatically:

* keep container parsing, quantization decoding, architecture lowering,
  materialization, execution, differential comparison and CLI/report concerns
  separated;
* extend supported formats and architectures through explicit contracts and
  adapters, not conditionals scattered through unrelated layers;
* depend on narrow domain interfaces and validated metadata, not weak naming
  heuristics or accidental concrete behavior;
* use names, invariants and fail-closed errors that make the model semantic
  boundary obvious to a later reader;
* refactor duplication or confused responsibility when it is on the path of
  the substantial milestone, with regression tests proving behavior remains
  intact.

Before committing, review the changed design for cohesion, coupling,
duplication, extension safety, error handling, test coverage and documentation.
Do not create abstractions merely for style, but do not leave a known design
hazard in a touched critical path just because the immediate test passes.

## Anti-microtask completion gate

Do not use a loop iteration as a unit of output. It is a fresh strategic work
session. A single helper, fixture, assertion, one-format decoder or one narrow
test is normally evidence for continuing the same coherent milestone, not a
reason to hand off. Continue through adjacent implementation, validation and
documentation work until the current strategic boundary is demonstrably closed
or an actual new boundary is reached.

At the start, form and execute an internal plan; at the end, report only what
was completed and what is genuinely blocked. If quality or architectural debt
is understandable and safe to repair, repair it in this loop; name it as a
bottleneck only when it truly prevents further progress now.

## Final objective

The project is complete only when it can accept a Safetensors package with
established semantics and produce a faithful, self-contained literal JSON
program representing the actual model computation. The program must be
replayable from only its declared `x[...]`/token/position/control inputs and
its own assignments, constants and embedded weights. A generic container alone
is insufficient; a generic exporter must preserve each established model's
actual semantics, while unknown semantics remain explicit fail-closed gaps.

A complete implementation should separate at least these concerns:

```text
Model source
    ├── configuration and metadata
    ├── tokenizer metadata
    ├── architecture description
    ├── tensor catalog
    └── generation configuration

Container reader
    ├── Safetensors
    ├── sharded Safetensors
    ├── MLX checkpoints
    ├── GGUF
    └── extensible container interface

Tensor storage and quantization
    ├── dense F64/F32/BF16/F16
    ├── signed and unsigned integer storage
    ├── MLX quantization schemes
    ├── GGUF quantization schemes
    ├── per-tensor quantization metadata
    ├── exact dequantization semantics
    └── lazy/range-based tensor access

Architecture interpretation
    ├── explicit architecture identification
    ├── architecture-specific adapters
    ├── configuration validation
    ├── tensor-role resolution
    ├── layer topology reconstruction
    └── fail-closed behavior for unsupported semantics

Mathematical IR
    ├── shapes
    ├── dtypes
    ├── layouts
    ├── constants
    ├── tensor references
    ├── reductions
    ├── normalization
    ├── projections
    ├── attention
    ├── positional encoding
    ├── MLP and MoE
    ├── residual connections
    ├── caches
    ├── control/routing operations
    └── output heads

Execution and validation
    ├── reference executor
    ├── operation-level comparison
    ├── layer-level comparison
    ├── KV-cache comparison
    ├── logits comparison
    ├── greedy-generation comparison
    ├── quantized-reference comparison
    └── reproducible reports
```

## Fidelity principles

Never claim fidelity based only on structural similarity.

Exact reconstruction may depend on:

* tensor dtype;
* accumulation dtype;
* intermediate casts;
* reduction order;
* rounding boundaries;
* matrix layout;
* transposition;
* masking semantics;
* normalization epsilon;
* normalization placement;
* unit-offset behavior;
* activation variant;
* attention scaling;
* Q/K normalization;
* RoPE variant;
* local versus global attention;
* sliding-window behavior;
* logit softcaps;
* attention softcaps;
* shared KV;
* tied embeddings;
* MoE routing;
* token position handling;
* KV-cache layout;
* quantization packing;
* scale, zero-point, bias, codebook, or block metadata;
* runtime-specific kernel behavior.

Represent these explicitly when they affect computation.

Do not replace unknown semantics with a generic Transformer approximation.

When the available evidence is insufficient, mark the architecture or operation as unsupported and fail with an actionable explanation.

## MAX_FEATURES rule

`MAX_FEATURES` or any similar inspection option must never change the mathematical model.

It may limit only:

* the number of previewed output coordinates;
* the number of printed terms;
* the number of sampled tensor values;
* diagnostic output size;
* optional validation fixtures.

The IR must retain complete shapes, complete reduction bounds, and references to all original tensor elements.

Never transform:

```text
sum(i = 0 .. hidden_size - 1)
```

into:

```text
sum(i = 0 .. MAX_FEATURES - 1)
```

unless the output is explicitly labeled as a non-equivalent truncated diagnostic experiment.

## Architecture-agnostic behavior

Architecture-agnostic does not mean using one generic architecture for every model.

It means:

1. detect the architecture from authoritative metadata;
2. select a compatible adapter;
3. validate required configuration fields and tensor roles;
4. reconstruct that architecture exactly;
5. reject unsupported architectures;
6. allow new adapters without changing container, quantization, or IR layers.

Tensor names may assist role resolution but must not be the sole source of semantics.

## Format-agnostic behavior

The same logical model stored in Safetensors, MLX, or GGUF should produce semantically equivalent IR.

Storage-specific information must remain isolated behind container and quantization interfaces.

A bit width is not a quantization algorithm.

Never assume that all 4-bit values share:

* the same packing;
* the same group size;
* the same scale layout;
* the same zero-point;
* the same affine formula;
* the same block format;
* the same codebook;
* the same dequantization dtype.

## Working method

For each loop, choose a substantial milestone that can be implemented and
validated coherently within the loop budget.

Good milestones include:

* correcting one fidelity defect;
* adding one architecture adapter;
* adding one container backend;
* implementing one quantization family;
* implementing one IR operation family;
* creating differential validation for one architecture;
* removing one approximation;
* improving performance without changing semantics;
* adding support for one previously rejected configuration.

Do not create broad unvalidated scaffolding across many components when one
complete vertical slice can be finished. Prefer the largest coherent vertical
slice supported by current evidence over a sequence of cosmetic micro-steps.

Prefer vertical slices:

```text
source model
→ parsing
→ architecture interpretation
→ IR
→ execution
→ differential validation
```

## Required validation

Every semantic change must include tests.

Use, when applicable:

* unit tests;
* malformed-input tests;
* shape validation;
* tensor-role validation;
* golden IR fixtures;
* differential operation tests;
* layer-output comparisons;
* end-to-end logits comparisons;
* generation comparisons;
* memory benchmarks;
* parsing benchmarks;
* tensor range-read benchmarks.

A feature is not complete merely because TypeScript compiles.

## Numerical comparison

Every validation report must state:

* reference runtime;
* candidate runtime;
* model identifier;
* revision or checksum;
* container format;
* quantization format;
* input tokens;
* dtype policy;
* absolute error;
* relative error;
* cosine similarity where meaningful;
* top-k overlap;
* argmax agreement;
* first divergent operation;
* whether the comparison is bitwise, lossless within dtype, numerically equivalent, or approximate.

Do not describe a result as identical unless it is actually identical under the declared comparison.

## Performance

Improve performance without weakening fidelity.

Prioritize:

* reading each container header once;
* persistent file handles;
* range reads;
* lazy tensors;
* avoiding conversion to JavaScript `number[]`;
* typed arrays or native buffers;
* bounded concurrency;
* tensor metadata caching;
* persistent helper processes;
* streaming JSON or compact IR serialization;
* avoiding duplicated expressions;
* common-subexpression preservation;
* reference-based weight storage;
* deterministic output.

These performance techniques apply to internal compilation only. They must not
weaken the self-contained calculation export requirement: an export intended
for replay cannot depend on a shard path, live file handle, bridge process or
unwritten tensor value.

Measure before and after performance changes.

## Repository hygiene

Before finishing a loop:

1. Remove temporary files not needed for reproduction.
2. Ensure generated artifacts are either reproducible or ignored.
3. Run formatting, linting, type checking, tests, and relevant benchmarks.
4. Update documentation.
5. Update architecture support tables.
6. Record known limitations explicitly.
7. Keep the repository runnable from a clean checkout.
8. Do not leave knowingly broken intermediate code.
9. Do not claim completion based on mocked or synthetic-only success.
10. Do not rewrite unrelated parts of the project without justification.

## Decision hierarchy

Prioritize work in this order:

1. incorrect mathematical semantics;
2. silent incorrect acceptance;
3. missing validation;
4. unsupported architecture required by the current target;
5. incorrect quantization handling;
6. incorrect container handling;
7. missing IR expressiveness;
8. missing execution capability;
9. performance;
10. ergonomics and documentation.

## Completion criteria for the overall mission

Set `missionStatus` to `complete` only after all mandatory gates in `agent-loop.config.json` are demonstrably satisfied.

At minimum, completion requires:

* strict format interfaces;
* strict quantization interfaces;
* strict architecture adapter interfaces;
* no architecture selected solely from tensor-name substring matching;
* no silent generic Transformer fallback;
* no truncation of mathematical dimensions by preview settings;
* a self-contained lossless calculation JSON with all replay weights/constants
  and named intermediate assignments;
* isolated replay of that JSON after the source checkpoint is unavailable;
* reproducible reference execution;
* operation-level differential validation;
* end-to-end validation on multiple architecture families;
* validation across multiple storage formats;
* validation across dense and quantized checkpoints;
* documented fidelity classes;
* documented unsupported cases;
* performance measurements;
* deterministic handoffs and reports.
* committed Gemma 4 and GLM 5.2 dense-lossless checkpoint manifests under
  `.agent-loop/checkpoints/` with source-removed replay and authoritative
  differential evidence.

Until then, use `missionStatus: "continue"`.

## Loop completion

At the end of every loop, create one handoff in:

```text
.agent-loop/handoffs/completed/
```

Use this filename format:

```text
HANDOFF-<zero-padded-sequence>-<UTC-timestamp>-<short-slug>.json
```

Example:

```text
HANDOFF-0007-20260714T220530Z-gemma2-differential-validation.json
```

Write the file atomically:

1. write it under `.agent-loop/handoffs/processing` with a `.tmp` suffix;
2. flush and close it;
3. validate it against the handoff contract;
4. rename it into `.agent-loop/handoffs/completed`.

Never write a partially completed file directly into `completed`.

The handoff must be the final repository-changing action of the loop.

## Handoff contract

The handoff must be valid JSON with this shape:

```json
{
  "schemaVersion": 1,
  "sequence": 7,
  "runId": "run-0007-20260714T220530Z",
  "createdAt": "2026-07-14T22:05:30.000Z",
  "missionStatus": "continue",
  "loopStatus": "success",
  "title": "Gemma 2 differential validation",
  "summary": "Implemented and validated...",
  "startingState": {
    "gitCommit": "...",
    "previousHandoff": "..."
  },
  "endingState": {
    "gitCommit": "...",
    "dirty": false
  },
  "milestone": {
    "selected": "...",
    "reason": "...",
    "completed": true
  },
  "changes": [
    {
      "area": "architecture",
      "description": "...",
      "files": ["src/..."]
    }
  ],
  "validation": {
    "commands": [
      {
        "command": "npm test",
        "exitCode": 0,
        "summary": "..."
      }
    ],
    "models": [
      {
        "model": "...",
        "revision": "...",
        "format": "safetensors",
        "quantization": "BF16",
        "result": "pass",
        "fidelityClass": "numerically-equivalent",
        "maxAbsoluteError": 0,
        "maxRelativeError": 0,
        "argmaxAgreement": 1
      }
    ]
  },
  "benchmarks": [],
  "findings": [
    {
      "severity": "high",
      "description": "...",
      "resolved": true
    }
  ],
  "knownLimitations": [],
  "bottlenecks": [
    {
      "description": "...",
      "impact": "...",
      "evidence": "..."
    }
  ],
  "blockedBy": [],
  "missionGates": {
    "satisfied": [],
    "unsatisfied": []
  }
}
```

Allowed `missionStatus` values:

```text
continue
complete
blocked
```

Allowed `loopStatus` values:

```text
success
partial
failed
```

A failed experiment may still produce a valid handoff, but the handoff must explain the failure precisely.

Every handoff must be backward-looking. Keep `summary`, `findings`,
`knownLimitations`, `blockedBy` and `bottlenecks` concise; do not duplicate
commit diffs, create a chronological diary, or include `nextRecommendedMilestone`
or `nextSteps`. Record only real unresolved constraints with their evidence.

## Stop behavior

Set `missionStatus` to `blocked` only when progress requires an unavailable external resource, credential, specification, model license, hardware capability, or human decision.

Do not mark ordinary implementation difficulty as blocked.

Set `missionStatus` to `complete` only when every required gate is supported by repository evidence and reproducible validation.

When `.agent-loop/STOP` exists, finish the current coherent repository operation, generate a final handoff with `missionStatus: "blocked"` and explain that the external stop flag was found. Do not initiate additional work.

## Non-negotiable constraints

* Never fabricate validation results.
* Never hide unsupported semantics.
* Never silently approximate an architecture.
* Never treat bit width as a complete quantization specification.
* Never let preview limits alter full-model computation.
* Never delete tests merely to make the suite pass.
* Never weaken assertions without mathematical justification.
* Never declare exact equivalence using only final token agreement.
* Never overwrite previous handoffs.
* Never create more than one completed handoff per loop.
* Never trigger the next loop directly from inside the agent.
* Leave loop triggering to the external runner.
