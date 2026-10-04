# Coherent branch distribution

The compiler now distributes stabilized decisions with shared facts across
their occurrences. Each arm owns its branch context; each dispatch predicate
retains the context preceding that decision. Conditions preserve lazy ordered
evaluation. Definitions are compiler-only and are fully substituted at emission.
Every selected dependency, guard, arm and final combination passes the existing
certified SymPy stabilization sequence. Candidate costs use the selected path's
definitions, with the previous callback restored even after failure.

`pre-coordinate.expr` is the actual, fully expanded first input-normalization
coordinate of the checkpoint. It contains one flat selector and only the original
inputs, with no compiler aliases. The native test re-reads this file, compiles it
with contraction disabled and compares 888,832 finite Half boundary pairs with
native F32/F16 reference operations: zero bit mismatches. This is a prefix,
not a complete Llama output coordinate.

`coordinate-run.json` records fresh compilation of dimension 2, position zero,
then attempts coherent flat emission with a 1 MiB artifact limit. Composition
completes 25 producers; the first complete path has 13 decisions, a
49,281,475,763-character body and 26,942,282,636 characters of guards.
The writer refuses before expanding that arm to bytes. No complete output
artifact or final-coordinate parity is claimed. These are costs of the first
path, not the total number or aggregate size of all paths. Flattening by itself
does not solve the excessive numerical dependency expansion.

The run records growth for every producer and for the dependencies reachable
in that selected body. For example, the final inverse grows from a
61,609,342,376-character mean to 799,001,340,704 characters before path
distribution. In the selected path its inverse dependency still expands to
11,864,058,920 characters. These observations locate remaining repeated
numerical substitution; they do not justify changing the reference arithmetic.

Reproduce:

```sh
LLM_INNER_DIRECT_JSON_CHECKPOINT=docs/evidence/direct-sympy-test-checkpoint \
LLM_INNER_DIRECT_FLAT_PREFIX_OUTPUT=docs/evidence/direct-sympy-coherent-paths/pre-coordinate.expr \
/private/tmp/llm-inner-pytorch/bin/python helpers/direct_sympy_coherent_paths_test.py

/private/tmp/llm-inner-pytorch/bin/python helpers/direct_sympy_coherent_run.py \
  docs/evidence/direct-sympy-test-checkpoint \
  /private/tmp/llm-inner-coherent-coordinate.expr \
  docs/evidence/direct-sympy-coherent-paths/coordinate-run.json \
  --max-characters 1048576 --max-paths 4 --max-seconds 90
```

The bounded coordinate run exits 1 with an explicit artifact-budget failure;
an existing destination survives unchanged. Legacy saved states are not loaded.
The adapter remains position-zero-only. Neither multi-token last-output parity
nor the other coordinates are covered by this prefix validation. No new Colab
or CUDA speedup measurement is asserted by these local changes.
