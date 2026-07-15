# Autonomous Model Decompiler Development Loop

## Mission

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

The target is not merely to inspect model weights. The target is to produce an executable, auditable intermediate representation that can reproduce the original model's forward pass and generation behavior with explicitly measured numerical fidelity.

## Current-loop responsibility

At the beginning of every loop:

1. Read the complete repository state.
2. Read `AGENTS.md`.
3. Read `agent-loop.config.json`.
4. Read `.agent-loop/state.json`.
5. Read the most recent valid handoff from `.agent-loop/handoffs/completed`.
6. Inspect current source code, tests, documentation, open issues, benchmarks, generated artifacts, and repository history.
7. Revalidate volatile assumptions against the actual repository.
8. Determine the highest-impact unfinished milestone.
9. Implement, test, benchmark, document, and review that milestone.
10. Produce exactly one final handoff only after the repository is in a coherent state.

Do not merely follow the previous handoff. Verify that its conclusions still match the repository.

## Autonomous continuity

You are part of a sequential relay, not a fixed task queue. Each invocation
starts with only repository evidence, the persisted state, and the previous
handoff. Renew your context before deciding what to do:

1. Treat the prior handoff as a lead, not an instruction that overrides the
   current repository.
2. Inspect the support tables, tests, reports, source, Git history and known
   limitations to identify what is actually unfinished.
3. Choose the smallest complete vertical slice with the greatest effect on the
   mission gates and fidelity risk.
4. If a plausible approach fails, record the exact evidence in the handoff so
   the successor can make a better decision rather than repeating it blindly.
5. Continue autonomously while a bounded milestone can be selected from live
   evidence. Ask for a human decision only when progress truly depends on an
   unavailable external resource, credential, license, specification, hardware
   capability, or product choice.

The external runner supplies the next invocation only after validating your
handoff. Never invoke that runner, start another Codex, or assume a successor
will accept an unverified claim.

## Final objective

The project is complete only when it can accept a supported model package and produce a faithful executable IR representing the actual model computation.

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

For each loop, choose a bounded milestone that can be implemented and validated coherently.

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

Do not create broad unvalidated scaffolding across many components when one complete vertical slice can be finished.

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
* reproducible reference execution;
* operation-level differential validation;
* end-to-end validation on multiple architecture families;
* validation across multiple storage formats;
* validation across dense and quantized checkpoints;
* documented fidelity classes;
* documented unsupported cases;
* performance measurements;
* deterministic handoffs and reports.

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
  "nextRecommendedMilestone": {
    "title": "...",
    "reason": "...",
    "acceptanceCriteria": [
      "..."
    ]
  },
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
