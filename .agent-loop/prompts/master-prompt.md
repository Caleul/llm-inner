# Engineering task: complete the Gemma 4 literal calculation artifact

Independently audit and advance this repository toward one objective: transform
the real dense, unquantized Gemma 4 Safetensors package into a navigable,
self-contained mathematical JSON program that reproduces the same forward pass
and greedy generation without access to the source checkpoint.

The artifact must embed every required learned weight, bias, scale, and
constant losslessly. Base64 is permitted for size only with exact dtype bits,
endianness, shape, layout, index-to-value decoding, casts, and reduction
semantics. Its navigable scalar audit view must replace addressable learned
operands with their decoded real numeric literals; unexplained expressions such
as `weight[o,i]` are insufficient.

It must declare all inputs, dependency-ordered operations, intermediate values,
producer/consumer coordinates, layer transitions, masks, RoPE, normalization,
attention, MLP, cache updates, logits, and greedy-generation state. It must
preserve mathematical dimensions, operation order, rounding boundaries, and
fail-closed semantics. Completion requires immutable source identity, artifact
integrity, successful replay while the source is physically unavailable, and
differential validation against an authoritative Gemma runtime.

Use only live repository, artifact, test, and runtime evidence. Do not inspect
hidden orchestration state, process-control scripts, execution history, or
operational records. Do not broaden the objective to other model families.

If the objective is incomplete, implement the largest coherent Gemma gap you
can prove now. Solve repeated problems through a general operation-, shape-,
dtype-, layout-, or source-level contract instead of one assignment at a time.
Carry the work through code, tests, artifact regeneration when required,
validation, documentation, and architectural review. Apply Clean Code, SOLID,
clear dependency direction, cohesive modules, explicit contracts, independent
tests, and fail-closed errors. Do not defer implementable improvements or write
recommendations for future implementation.

Run `npm run typecheck` and `npm test`. If you make any repository change,
create exactly one focused local Git commit and leave the worktree clean.

Begin the final response with exactly one marker:

- `OBJECTIVE_COMPLETE:` only if every acceptance criterion was already met at
  the start of this session, you made no change and no commit, and fresh
  validation proves the existing state complete.
- `OBJECTIVE_ADVANCED:` if you changed or committed anything, even if the final
  state now appears complete. Describe completed results and validation only.
- `OBJECTIVE_BLOCKED:` only when a concrete external impossibility prevents
  both useful implementation and validation.

An implementation session cannot certify its own changes as complete.
