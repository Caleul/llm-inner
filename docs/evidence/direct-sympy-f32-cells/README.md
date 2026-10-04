# Finite F32 arithmetic proofs

The numerical compiler now removes F32 casts on proved dyadic grids with
at most 24 significant bits. It preserves the original F64 arithmetic order
and signed zeros. It also removes an F32 update only when both operands are
proved finite F32 values and the update lies strictly inside the smaller
rounding cell of the nonzero center, including binade/subnormal boundaries.
Midpoints, zero-crossing center ranges and unknown types cannot prove this
elision. These rules run before elementary conversion expansion.

Constant elementary reinterpretations are evaluated during compilation.
Nonfinite floating payloads remain explicitly encoded. SymPy's canonical
integer scalar zero is admitted as positive zero by the savepoint sign
validator; nonzero integers and booleans remain rejected. The activation
frontier fixture uses a conservative grid so it still exercises selector
synchronization after the tighter grid can independently eliminate its F32
frontier. Its original native Half-input parity coverage remains intact.

Validation:

- `integration-tests.log`: 15 named integrations passed, zero failed/skipped.
- `final-cell-tests.log`: four new tests passed, with 268,216 native cell
  comparisons, 263,173 native exact-grid comparisons (including signed zeros)
  and 69,634 constant reinterpretation cases, all without bit mismatches.
- `canonical-zero-regression.log` preserves the first regression gate: the
  sign validator rejected canonical positive zero during restore, and a
  synchronization fixture no longer reached its intended frontier. Both were
  corrected and revalidated, including the real checkpoint resume test.

`coordinate-run.json` is a fresh bounded dimension-2, position-zero attempt.
The first path still exceeds the 1 MiB artifact budget. Its logical size has
not materially improved; no complete artifact, final-coordinate parity or
speedup is claimed. Existing source identities change, so old savepoints
must be rejected before restore; this run reused none.

Next investigation: `ConversionSession.bounds(R16(X1))` for a source limited
to `[1, 2]` still reports quantum `-24`. Half values in that interval have a
coarser `-10` grid. The overly conservative grid prevents exact F32 proofs
for subsequent arithmetic. Correct interval-specific Half grid propagation
and then remeasure the complete coordinate; do not silently restrict input
domains to make the artifact smaller.

The complete coordinate, variable-length last-token semantics and remaining
output coordinates remain pending. Expressions remain mathematical strings;
JSON files in this directory are diagnostic metadata.
