# Signed-zero updates retain formats without redundant casts

A finite Half/F32 scalar plus or minus a producer proved to have zero magnitude
remains representable in that same format. The operation itself is retained:
its result may be positive or negative zero. Removing the redundant surrounding
R32/R16 conversions therefore preserves the exact value and zero sign.

A literal negative-zero addition is also an exact identity for a finite scalar
under the compiler's nearest-even numerical contract. Positive-zero addition
is kept when the other operand may be negative zero. Unknown/nonfinite operands
remain barriers. No sum is reassociated and no actual nonzero update is removed.

The new three-test suite compares actual saved mathematical strings compiled
natively: **656,300 comparisons, zero mismatches**, covering every finite Half
word, both signed-zero updates, all finite F32 exponent classes, representative
mantissas at binade/subnormal boundaries, addition and both subtraction orders.
Tests also verify branch-local zero bounds do not escape into sibling paths.
The final integration gate passes **20 tests, zero failures or skips**; build passes.

The authoritative fresh whole-domain position-zero/dimension-two run completed
25 producers but failed final emission. Its fully substituted logical size is
still **11,756,851,964,536 characters**, with 34,979 resident definition characters.
Neither count decreased versus the prior constant-cell correction. This change
is a localized semantic reduction, not a solution to the main normalization
expansion or evidence of complete model compilation.

For X1/X2 in [-2^-23,2^-23], logical size changed from 7,780,503 to 7,779,511
characters. The 1MiB atomic emission limit is still exceeded; no final region
artifact or full-coordinate parity is claimed. The comparison script also keeps
a historical pre-constant-cell baseline at 10,303,598,463 characters, distinct
from the immediate prior version used to evaluate this change.

Both source hashes changed; strict saved-state rejection was verified. All runs
use fresh producer proofs. Existing emitted artifacts are retained as historical
SHA/parity evidence, not silently imported as compatible proof tables.

`validation.json` and the central test map retain the exact scope and unresolved
requirements. Timing/RSS observations are not a controlled speedup measurement.
Expressions remain mathematical strings; JSON files contain metadata only.
