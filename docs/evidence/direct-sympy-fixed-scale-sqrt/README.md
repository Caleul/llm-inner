# Incorporate a proved square-root scale into the substituted expression

Expressions remain mathematical strings. The compiler still applies
parentheses, substitution, factor and simplify to stability. JSON files in
this directory record state, identity and evidence rather than expression trees.

The existing root uses five ordered partial fractions with a certified final
F32 boundary. The current change preserves those coefficients and that order.
If the actual finite F32 source enclosure lies wholly inside one binary scale
range, its exponent and parity are constants. Incorporate the fixed multiplier
and power-of-two scale before substituting its producer. The generic route
remains when a source crosses a scale boundary. This introduces no runtime
branch or fundamental variable. No intermediate survives final emission.

The completed root uses five source occurrences instead of seven in those
contexts. Both square-root helper tests pass. The existing generic native test
covers 25,167,601 mantissa/parity/subnormal/exponent cases; the new native test
covers 58,720,257 cases across seven complete normal F32 mantissa ranges,
positive/negative exponent parities, extreme scales and the smallest subnormal.
Both report zero bit mismatches against widened native F32 square root.

`run.py` creates an isolated baseline using the root source from commit
048de2f, and runs baseline then current version in separate processes with
the same region and resource budgets. The region is X1 and X2 in
[0.03125,0.0390625], coordinate 2, position zero, one token. Every one of its
66,049 Half input pairs is compared with a fresh Torch CPU checkpoint forward
using each emitted file's actual bytes. Both files have zero bit mismatches.
Neither generated expression looks up answers. The verifier enumerates inputs.

Observed single-run results:

| Measure | Baseline | Current |
| --- | ---: | ---: |
| Emitted characters | 554,081 | 326,227 |
| Completed paths | 2 | 2 |
| Compilation seconds | 6.9767 | 4.3847 |
| Worker wall seconds | 7.8176 | 5.1482 |
| Worker peak RSS bytes | 399,015,936 | 328,908,800 |
| Exhaustive output pairs | 66,049 | 66,049 |
| Bit mismatches | 0 | 0 |

The actual file is 41.12% smaller. Compilation time is 37.15% lower and worker
peak resident memory is 17.57% lower in this observation. RSS is macOS
RUSAGE_CHILDREN.ru_maxrss, captured immediately after the compilation worker
returns and before native verification. These are single measurements;
concurrent unrelated verification may affect timings. This is a compiler
change comparison, not a sequential-versus-parallel or CUDA benchmark.

Full-domain recompilation is attempted under the same 32 MiB artifact budget.
It still cannot emit a complete coordinate: its first path body remains
5,980,067,411 characters and its guards 3,314,096,435. Four more selected
numeric recipes admit a smaller representation (50 versus 46), but that does
not shrink this path's dominant body. The full-domain report records the
failure and producer-by-producer expansion; no partial file is admitted.

`validation.json` extends the historical test map after current integration
and general regression checks. Source/backend/checkpoint identities are checked
before/after actual compilation and parity. Old regional numerical results
are not silently carried forward. The full input domain, multiple tokens and
full output vector remain unfinished. This change uses CPU and does not claim
GPU acceleration or completion based on expansion estimates.
