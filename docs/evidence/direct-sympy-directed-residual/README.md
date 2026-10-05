# Signed attention bounds eliminate irrelevant MLP dependencies

Expressions remain mathematical strings under the user's latest instruction.
JSON files in this directory record compiler state and validation evidence.
This is progress toward the first output coordinate, not a completed model.

The earlier mixed-sign compilation retained a third normalization vector when
the lower input magnitude was reduced from 0.0625 to 0.03125. Its absolute
attention bound could not establish the distance to the residual's rounding
boundary, so it retained the MLP and exceeded the 16 MiB artifact budget.

`composed_dot_intervals()` now encloses the signed attention update using the
concrete V and O weights. Exact rational coefficient composition is used only
for the proof; error bounds retain both original F32 reductions and the first
Half storage. The final Half storage is applied monotonically to outward
endpoints. Generated numerical operations keep their original order.

`directed_residual_radius()` propagates this enclosure through the original
F32-to-Half residual addition. The resulting minimum cell radius is accepted
only for the actual residual producer in that layer. If the entire MLP update
is strictly smaller than the radius, storing the updated residual gives the
same value. The compiler can eliminate those dependencies before expanding
them. Unsupported shapes, nonfinite data, overflow and exhausted proof budgets
produce no certificate and retain the dependencies.

For the tested mixed-sign region, the radius is 0.0000152587890625 for both
coordinates; certified MLP magnitudes are 0.000006794929504394531 and
0.000008046627044677734. Both updates can therefore be eliminated. A native
test checks the original ordered attention projections and actual residual
cells for every pair in both regions: 62,924,800 pairs, zero violations.

The actual `mixed.expr` contains 10,269,774 characters, five paths and no
compiler aliases. Its only fundamental variables are the original X1/X2.
Each substitution continues through CPU SymPy factor/simplify to stability;
branch combinations are distributed after simplification. The file is executed
against the original Torch CPU forward over every finite Half pair in:

| X1 | X2 | Pairs |
| --- | --- | ---: |
| [-2, -0.03125] | [0.031280517578125, 1] | 31,462,400 |
| [0.03125, 2] | [-1, -0.031280517578125] | 31,462,400 |

Both exhaustive reports have zero bit mismatches. This checks the emitted
artifact separately from the proof of its dependency elimination. The wider
artifact is slightly larger than its predecessor; the improvement is admitting
additional input regions without retaining the third normalization vector.

The saved frontier resets every old numerical leaf after the source change,
then recompiles and validates before atomic admission. Earlier geometry alone
is retained. During recovery, the shared temporary Python environment lost
its venv configuration and a Transformers source file. The interrupted region
was not admitted. A new task-owned environment reinstalls Python 3.10.13,
Torch 2.12.1, Transformers 5.5.0, SymPy 1.14.0, NumPy 2.2.6 and Safetensors
0.8.0. The reference imports correctly and the complete compiler identity
matches before recovery resumes. The native proof and integration suite are
repeated in that environment. Failure and recovery logs are retained separately.

`record.py` audits coverage without overlap, fresh parity, artifact hashes and
the existing test map before saving evidence. It appends a new validation entry
and refuses to overwrite differing historical evidence. The complete-coordinate
artifact, full-coordinate parity and multiple-token parity remain unfinished.
No other coordinate is advanced on the strength of these regional results.

The finished recovery admits 82 freshly compiled regions, with 672,118
reference comparisons and zero mismatches. Disjoint coverage is 2,379,732,998
of 4,030,726,144 finite-Half input pairs (59.0398%); 1,650,993,146 remain
pending. This adds 20,963,326 pairs beyond the previous source version.
That coverage comes from certified regions and their recorded validation;
it is not an exhaustive reference comparison of all 2.38 billion pairs.
The separate exhaustive artifact runs cover the 62,924,800 pairs listed above.

The rebuilt project passes all 32 direct-string integration tests without
skips. The whole-project run has 626 tests, 570 passes, 15 failures and 41
skips. All 15 failure locations match the preceding evidence: missing/stale
Gemma calibration artifacts, pinned runtime identity assertions and two absent
contract/example files. No new failure location is introduced by this change.
