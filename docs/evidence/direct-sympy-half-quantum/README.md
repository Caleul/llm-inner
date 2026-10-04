# Interval-specific Half precision

Expressions remain mathematical strings processed by SymPy. JSON reports are diagnostics, not expression representations.

Half conversion bounds now use the minimum precision of their reached interval rather than always using the format-wide subnormal grid. Zero crossings retain the conservative grid; a coarser source dyadic grid is preserved. Half rounding-cell radii stop shrinking below the subnormal spacing.

The three focused tests cover all 63,488 finite Half encodings, positive/negative ranges, zero crossings, coarse source grids and subnormal cells. A saved and re-read sum expression was compiled natively: 1,050,625 Half pairs in [1,2], zero bit mismatches. Its text shrank from 294 to 233 characters against the same compiler with conservative old grid metadata. This is a bounded numerical lemma, not a model artifact.

A fresh dimension-two, position-zero coordinate run completed 25 producers, without loading saved state. Fully substituted size remained exactly 13,773,588,190,880 characters. Emission stopped at the configured 1 MiB budget; no complete artifact or final parity exists. The first selected arm still has 49,281,257,959 body characters and 26,942,163,578 guard characters. No full-coordinate reduction or acceleration was demonstrated.

The numerical-grid correction is valid but does not address the dominant dependency expansion. Continue investigating repeated expressions and branch-specific simplification. The resource cap never authorizes dropping branches or narrowing the model input domain.

Build, focused tests, regression suite and fresh-coordinate diagnostics are stored beside this document. Timings are observations; the regression suite overlapped the coordinate run. Old savepoint source identities change and must remain rejected.

Validation: build succeeded; all 16 named integrations passed, with zero failures and zero skips (112.08 s). This includes real fresh-state resume, strict identity rejection and existing numerical regression suites.
