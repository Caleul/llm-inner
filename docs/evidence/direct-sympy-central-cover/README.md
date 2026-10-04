# Fresh broad-region compilation and pending-only admission

Two regions were compiled afresh using the current checkpoint and numerical compiler. Their expressions passed independent native comparisons before their exact intersections with pending leaves were admitted. Existing completed leaves and their artifact records were preserved unchanged. Shared artifact files avoid compilation-time copies; they are not runtime activation caches or response lookups.

| Input domain | Effective expression | Newly covered input patterns | Native comparisons | Mismatches |
| --- | --- | ---: | ---: | ---: |
| X1 in [-65504,-32], X2 in [-2^-7,2^-7] | `1309/131072` | 184,571,904 | 8,200 | 0 |
| X1 in [32,65504], X2 in [-2^-7,2^-7] | `-1309/131072` | 184,571,904 | 8,200 | 0 |

The expressions are outputs of the actual compiler, which incorporates weights and eliminates producers under each region's numerical conditions. They are not obtained by recording sampled responses. Native comparisons check emitted expressions against the checkpoint reference, including signed-zero inputs and boundaries; they are not exhaustive enumeration of all patterns.

Coverage increased from **614,428,672 (15.2436%)** to **983,572,480 (24.4019%)**, with **3,047,153,664 patterns still pending**. There are 26 completed leaves and six distinct expression files. The 23 previously completed leaves are unchanged. The original 48 ordinary attempts are retained separately from the two fresh promotions. An additional native pass over all 26 leaves made **6,766 comparisons with zero mismatches**.

`helpers/direct_sympy_cover_regions.py` uses the controller's writer lock, exact Half-rank geometry, full numerical source/backend/checkpoint identity, artifact hashes, fresh compilation and native parity before atomic manifest publication. Incompatible state, compilation budget exhaustion and parity failure do not mutate the manifest. Numerical compiler sources have not changed. The geometry/admission helper's own hash is recorded in each promotion. Every admitted subregion is contained in the broad compiled domain; coverage counts exclude previous completed intersections.

The three helper tests cover exhaustive small-domain partition geometry (including both zero signs), incompatible state and failed candidates, and real compilation with compatible controller resume. `npm run build` passed. All **25 direct-string integrations passed, zero skipped**; prior integrations remain in the gate. See [validation.json](validation.json) and [artifact-map.json](artifact-map.json).

Reproduce the two promotions against a compatible pre-promotion tree:

```sh
/private/tmp/llm-inner-pytorch/bin/python helpers/direct_sympy_cover_regions.py docs/evidence/direct-sympy-test-checkpoint artifacts/direct-sympy-input-partitions/update-cell-state /tmp/negative-cover.json --domains '{"X1":[-31743,-20480],"X2":[-8192,8192]}' --max-seconds 30 --random-cases 8192
/private/tmp/llm-inner-pytorch/bin/python helpers/direct_sympy_cover_regions.py docs/evidence/direct-sympy-test-checkpoint artifacts/direct-sympy-input-partitions/update-cell-state /tmp/positive-cover.json --domains '{"X1":[20480,31743],"X2":[-8192,8192]}' --max-seconds 30 --random-cases 8192
```

`record.py` audits the live manifest, numerical identity, old leaves, promotion domains, native reports and test gate, then snapshots the effective files. Expressions use mathematical strings; JSON files here contain compilation state and evidence.

This evidence covers position 0, coordinate 2 and one-token arbitrary finite Half embedding inputs X1/X2. The full coordinate, multiple-token parity, complete output vector and final Rust function remain unfinished. No compilation speedup is claimed: broad-region coverage and artifact size are not timing benchmarks. Colab/CUDA execution is not demonstrated by this evidence.
