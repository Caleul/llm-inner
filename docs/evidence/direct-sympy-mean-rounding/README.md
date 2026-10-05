# One-source F32 mean conversion with a rounding stability proof

The composed RMS mean repeats its arithmetic expression to recover the retained
parity for F32 rounding. Its subsequent square-root refinement multiplies those
copies. Existing sqrt/reciprocal midpoint proofs do not remove this mean copy.

The new certificate applies only to a nonnegative certified Half/F32 operand
plus a positive finite constant, with a proved finite normal F32 output range
and a known source grid. It checks every possible source binade. Small binades
with at most 42 significant bits admit the existing exact-integer-word route.
Otherwise the operand is aligned to the binade's F32 grid (or its predecessor's
half-grid at a binade crossing). The constant determines the aligned remainder.

Converting a positive UInt64 word to F64 can lose at most 512 integer units.
A subsequent offset operation rounds on the 2**29 grid. The compiler excludes
all nonexact F32 midpoint neighborhoods of radius 1024 units before admitting
this composition. Exact midpoints stay exactly represented. Fraction arithmetic
computes the constant's rounded displacement and tests its residues against
all possible midpoint locations. This is a rounding stability certificate; it
does not pretend the initial UInt64-to-F64 conversion is exact.

The kernel contains one occurrence of the original sum and preserves its F64
addition and final F32 storage result. Subnormal, overflow, cancellation,
unknown types/grids and unsafe midpoint neighborhoods retain the original
kernel. The new flag applies only to R32 normal-domain conversion. It does not
weaken Half/tandem conversion admission or reassociate model arithmetic.

## Native and checkpoint validation

`mean-proof.log` exhaustively tests every nonnegative F32 operand on the stated
2**-49 grid through 2**32, including both input zeros and the upper endpoint:
494,927,874 cases and zero bit mismatches against native F32 storage of the
original sum with the checkpoint's F32 epsilon. Tests also reject unknown
dtypes, subtraction, and constants immediately on either side of an unsafe
midpoint; an exact midpoint remains admissible.

All three regional coordinate expressions are freshly compiled and have exact
checkpoint parity over 66,317 pairs. They contain no compiler aliases. Their
sizes are unchanged in these regions (23,898; 326,051; 2,913,052 characters):
those regions already admitted one-source conversions through tighter proofs.
This result covers coordinate 2, position 0, one token and those regions only.

The full-domain 32/96 MiB attempts both reach eight completed candidate paths,
then reject publication. The large candidate body falls from 3,571,931,313 to
714,123,437 characters (about 80% smaller); its guards still occupy 432,141,590.
The first candidate's guard decreases from 6,932 characters in the older
prefix investigation to 6,618. The full worker attempt takes about 43.1 seconds
and peaks at 281,657,344 child RSS bytes. These timings are recorded executions,
not a controlled parallel or CUDA speedup comparison.

`record.py` verifies source/checkpoint/backend identity, actual artifact hashes,
native evidence and the failure map. The focused suite passes all 37 tests.
The general suite reports 631 tests: 570 pass, the same 15 locations fail and
46 skip. The new optional Python test accounts for the additional skip when
the explicit backend environment is absent. Historical evidence remains in
the existing test map; no previous numerical savepoint is relabeled compatible.

## Remaining work

There is no complete full-domain coordinate file, multiple-token parity or
complete output vector yet. The output budget rejects a candidate before
allocating its giant expression and preserves all remaining input paths.
The next investigation must simplify the surviving composition and its frozen
conditions, which still multiply large residual expressions. Emitting only the
admitted prefix, preserving a runtime intermediate cache or increasing the
budget does not satisfy the final compilation requirement. Expressions remain
mathematical strings, following the latest explicit user format instruction;
JSON here contains diagnostics and validation metadata only.

The first focused run exposed a recipe-specific assertion in the checkpoint
continuation test: the reduced compiler persisted 11 dependencies under the
same budget, while the test expected exactly 10. The test now compares restored
counts with the actual persisted count and still verifies immutable rejection,
record-prefix preservation, byte counts and further progress. The initial
failure is retained in `focused-before-resume-fix.log`; final validation uses
the rerun. The change does not accept incompatible saved state or relax parity.

The next rerun also showed that the old 4 MiB resume budget no longer admits
the next producer: its rejected numerical envelope is 4,454,001 characters.
The test now derives the resume budget from the measured rejected expansion,
then requires real progress. `resume-proof.log` shows 11 restored dependencies
and 12 persisted dependencies, with identity rejection before mutation. The
second obsolete-budget failure is retained in `focused-before-budget-fix.log`.
This adapts a test's resource budget, not the final artifact admission budget.
