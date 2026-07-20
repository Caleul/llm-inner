# Gemma 4 E4B: artifact-owned native replay — 2026-07-20

## Completed boundary

The schema-54 artifact already serialized the complete native-reduction
`executableReplay` contract, but the TypeScript provider still launched Python
with `GEMMA4_RUNTIME_REDUCTION_PROCESS_ENVIRONMENT` imported from the checkout.
The evidence validator likewise rebuilt the adapter digest, invocation program
and expected runtime attestation from checkout constants. Those values happened
to equal the JSON, but the replay path had two authorities.

The provider now receives the complete `executableReplay` object opened from
the artifact and derives from it:

- the embedded adapter source and SHA-256;
- all five declarative invocation programs;
- the closed `inheritance=none` process environment;
- runtime, Torch build, device, platform, BLAS and build-setting identity;
- dependency and execution-state attestations; and
- the evidence contract used to validate every operand/output receipt.

`Gemma4RuntimeReductionProvider` exposes that narrow evidence contract as an
isolated copy. Receipt validation uses the provider's artifact-derived adapter
digest, invocation program and attestation rather than regenerating them from
the checkout. A provider/evidence contract mismatch fails before execution.
The spawn-environment materializer accepts only schema 1 with
`inheritance=none`, returns a copy and never merges the parent environment.

## Live artifact boundary

The existing real artifact remains schema 54 because every newly consumed
field was already integrity-bound JSON data; no payload or schema regeneration
was required:

- path: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes: `21,409,381,135`;
- artifact SHA-256:
  `f56cd4f125b87aa094ffa2e9b57cd1f0cffef10ba1f9a360f9538b39ce2d7585`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded learned bytes: `15,992,314,836`; and
- unresolved native reductions: `100`, with
  `exactReplayClaim=forbidden`.

The live streaming reader opened that 21 GB file through
`Gemma4TorchRuntimeReductionProvider.fromArtifact` and returned contract ID
`torch-2.12.1-cpu-inference-matmul-v1`, adapter SHA-256
`5fa0286427c92a557a4893219779f6eb11b7229e5f63d49863ed11ad18c74950`,
all five expected invocation IDs, `inheritance=none`, the exact six-variable
map, operating-system build `25F84` and execution-state schema 2.

The same read was repeated while both `./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were moved to explicit private
temporary paths under an `EXIT`/`INT`/`TERM` restoration trap. The provider
reported five invocation programs and `inheritance=none` while both declared
paths were absent. Postconditions verified both paths restored and both
temporary paths removed.

This change closes reader-side authority duplication. It does not publish the
Apple Accelerate product-rounding, lane, tile or fold schedule, so all 100 BMM
scalar reductions remain correctly fail-closed.

## Validation

The focused native integration executes the embedded adapter on the pinned
`venv/bin/python` runtime. It covers exact operand/output evidence, hostile
parent-environment isolation and all five serialized invocation classes.

```text
npm run build
node --test dist/test/gemma4-runtime-reduction-trace-coverage.test.js
# 5 pass, 0 fail

npm run typecheck
npm test
git diff --check
```

Fresh validation passed `255` tests with `0` failures. Three optional,
non-Gemma Transformers integrations were skipped because their external
runtime opt-in was not set. The pinned Gemma 4 embedded-adapter integrations
ran and passed.
