# Gemma literal-export engineering contract

## Product objective

Build a production-quality decompiler for the real dense, unquantized Gemma 4
Safetensors package. Its primary output is one navigable, self-contained JSON
program from which a reader can reproduce the model's forward pass and greedy
generation step by step without reopening the checkpoint.

This repository is focused exclusively on Gemma 4. Do not broaden the product
roadmap to other model families, containers, quantization schemes, benchmarks,
or generic abstractions unless the change is an immediate prerequisite for the
Gemma artifact and is exercised by it.

## Required artifact

The JSON must contain or losslessly encode every learned weight, bias, scale,
and constant needed for replay. A compact Base64 payload is valid only when the
artifact also declares dtype, bit pattern, endianness, shape, layout, exact
index-to-value decoding, casts, and accumulation semantics. It must never need
the original Safetensors files after export.

The program must expose:

- declared inputs such as token IDs, positions, modality inputs, and cache;
- all dependency-ordered operations and named intermediate values;
- stable producer and consumer navigation between layers and operations;
- masks, RoPE, normalization, attention, MLP, cache transitions, logits, and
  greedy-generation state changes;
- scalar audit views that substitute addressable learned operands with their
  decoded numeric literals instead of unexplained placeholders such as
  `weight[o,i]`;
- dtype conversions, rounding points, reduction domains, reduction order, and
  runtime semantics sufficient to reproduce the same outputs;
- immutable source identity, artifact integrity evidence, source-removed
  replay, and differential comparison with an authoritative Gemma runtime.

Unknown semantics must fail closed. Never call the artifact complete while it
contains external tensor references, truncated dimensions, preview-only
weights, implicit configuration, guessed behavior, hidden generic-decoder
steps, or formulas whose learned values cannot be decoded and substituted.

## Work contract

Start by auditing the live repository, generated artifact, tests, and current
Gemma evidence against the complete product objective. Work from source and
runtime evidence rather than assumptions.

If any material gap is present, implement the largest coherent safe boundary
now. Generalize repeated problems by operation, shape, dtype, layout, or source
contract; do not hard-code one layer or assignment at a time. Continue through
implementation, tests, artifact regeneration, validation, documentation, and
architectural review until that boundary is complete or a genuine external
blocker is proven. Do not write that a future implementation should perform
ordinary work that current evidence makes possible.

Apply Clean Code and SOLID pragmatically: keep responsibilities cohesive,
dependency direction explicit, contracts narrow, extension points additive,
names domain-specific, errors fail-closed, and tests independent of the
implementation under test. Remove relevant duplication and update docs when
behavior or architecture changes.

Do not inspect or modify hidden orchestration state, process-control scripts,
execution history, or operational records. They are not product evidence and
must not influence engineering decisions.

Before finishing, run `npm run typecheck` and `npm test`. If you changed the
repository, create exactly one focused local Git commit and leave the worktree
clean.

## Final response protocol

Begin the final response with exactly one of these markers:

- `OBJECTIVE_COMPLETE:` only when the repository already satisfied every
  acceptance criterion before this session made any change, no commit was
  necessary, the worktree is clean, and fresh validation independently proves
  it. Completion certification is external to any implementation session.
- `OBJECTIVE_ADVANCED:` whenever this session changed or committed anything,
  even if the resulting state appears to satisfy the entire objective. Report
  the completed outcome and exact validation, not proposed future work.
- `OBJECTIVE_BLOCKED:` only for a genuine external impossibility that prevents
  both implementation and meaningful validation. Include concrete evidence.

Never claim completion for work implemented by the same session.
