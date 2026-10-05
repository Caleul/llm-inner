# Prove numerical guards before distributing their paths

Expressions remain mathematical strings; JSON records compiler identity and
evidence. The final format instruction remains SymPy-compatible strings.

The decision search previously used branch-local numerical bounds only to
skip singleton nonzero producers. It did not consult them to establish the
truth of the next comparison. It could therefore distribute an impossible
arm and only discard it after substituting its dependencies.

`condition_truth` now combines original numerical certificates with bounds
already proved by the current prefix. It checks strict/non-strict comparisons
using conservative, finite enclosures, including unsigned magnitude guards.
It never assumes the comparison it is proving. Overlapping or unsupported
comparisons remain undecided; equality is deliberately not inferred. Logical
conditions retain their existing lazy traversal. Prefix sessions are isolated
from sibling and global contexts, and earlier emitted guards keep their own
frozen context. Selected bodies still pass factor/simplify to stability.

The full-domain diagnostic now proves 30 comparisons in the visited prefixes.
Before its artifact-budget stop, its first completed path changes as follows:

| Measure | Previous compiler | Current compiler |
| --- | ---: | ---: |
| Decisions distributed | 14 | 6 |
| Guard characters after substitution | 3,314,096,435 | 299,107,294 |
| Body characters after substitution | 5,980,067,411 | 5,980,067,411 |
| Diagnostic seconds | 40.5370 | 10.6887 |

Guard expansion drops 90.97%. This does not shrink the body's dominant
normalization expansion. The 32 MiB publication budget still rejects the
full-domain coordinate; no path or input is silently omitted. Expansion
measurements are not an emitted final artifact or a parity claim.

Thirteen coherent-path tests pass, including a numeric impossible branch
which now creates no split, overlapping comparisons, unknown division,
sibling context isolation, lazy dispatch and both signed zeros. Native
checkpoint-normalization parity covers 888,832 cases with zero mismatches;
selected arithmetic propagation covers 30,722 cases with zero mismatches.

`run.py` compares commit d5dd566's guard compiler with current code in separate
processes using identical checkpoint, backend, region and budgets. Each
actual emitted expression is compiled and compared with a fresh Torch CPU
forward for all 66,049 Half pairs in X1 and X2 [0.03125,0.0390625], coordinate
2, position zero, one token. Both have zero bit mismatches. This region has
no newly inferred numerical guards: both artifacts remain 326,227 characters
and two paths. Do not attribute the earlier root-scale size reduction to this
change, or claim its observed timing difference as a new speedup benchmark.
The regional artifact validates actual output under the new source identity;
it does not validate a complete coordinate, multiple tokens or full vector.

The 34 integration tests pass. The general suite remains 628 tests: 570 pass,
15 failures at the same test locations, 43 skips. The missing agent-loop
configuration/prompt files can change which concurrent file read reports
ENOENT in that existing failing test; its failure remains outside these
compiler changes. Historical test-map entries are preserved and extended.

The main remaining obstacle is the body's repeated numerical refinement,
particularly constant propagation and the final normalization's expanded
mean. This change used CPU. CUDA and a sequential/parallel speedup comparison
remain unproved, as do full-domain and multiple-token output artifacts.
