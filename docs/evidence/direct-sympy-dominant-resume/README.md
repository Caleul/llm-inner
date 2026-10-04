# Compatible continuation after dominant-square RMS simplification

The previous source identity's 28 completed domains were used only as geometry. `recompile.py` compiled each numerical expression afresh with the current checkpoint/compiler, validated it against the native checkpoint reference, and admitted only its pending intersections into the new state. Previous numerical bodies and certificates were not reused. `record.py` proves that every previously completed domain is now fully covered by compatible completed leaves, with no pending intersection.

Coverage is **1,414,004,736 / 4,030,726,144 finite Half input-vector patterns (35.0806%)**. This includes **61,288,448 additional patterns** beyond the previous source's 1,352,716,288. **2,616,721,408 remain pending.** Both signs of zero count separately. Exact partition audit preserves the complete admitted root; resource limits do not omit any input domain.

The new source-compatible tree contains **43 completed leaves and eight distinct effective expression files**. All 28 old-domain recompilations produced valid fresh evidence (7,294 native comparisons, zero mismatches). A subsequent controller run made 24 attempts with a five-second worker limit: 12 completed and 12 stopped for subdivision. A final independent native pass over all 43 leaves made **11,198 comparisons, zero mismatches**. These samples cover numerical boundaries and signed zeros; they are not exhaustive enumeration of all input patterns.

The ordinary continuation used:

```sh
/private/tmp/llm-inner-pytorch/bin/python helpers/direct_sympy_partition_run.py docs/evidence/direct-sympy-test-checkpoint artifacts/direct-sympy-input-partitions/dominant-square-state --resume --max-attempts 24 --region-seconds 5 --max-paths 128 --max-characters 1048576 --cas-characters 8388608
```

`recompile.log` and `continuation.log` record effective coverage after each admission/attempt. Numerical source hashes, checkpoint/backend identity, artifact bytes, old-domain recovery and complete root coverage are checked by `record.py`. [artifact-map.json](artifact-map.json) maps every completed leaf to an actual input-only expression; repeated bodies have one evidence file per hash. [frontier.json](frontier.json) is the resumable compilation-state snapshot, not a complete-coordinate artifact.

No compiler sources changed after the prior build and 25 passed, zero skipped integration tests in `../direct-sympy-dominant-square/test.log`; the current identity matches that gate's recorded identity. That gate was not rerun for this evidence-only continuation. The test map retains earlier evidence and adds `dominantResumeValidation`.

Colab session inventory through Access Broker operation `34deb47c-d26f-401f-9fea-ff3486fdc80d` returned successfully with no active runtime. [colab-inventory.json](colab-inventory.json) records that observation. This run demonstrates neither GPU execution nor parallel speedup. Compilation and validation were local; no performance comparison is inferred from concurrent or budget-limited timings.

The full coordinate, multiple-token compilation/parity, output vector and final Rust function remain unfinished. Input scope here is coordinate 2, position 0, one-token arbitrary finite Half embedding coordinates X1/X2. Expressions are mathematical strings; JSON files contain saved compiler state/evidence.
