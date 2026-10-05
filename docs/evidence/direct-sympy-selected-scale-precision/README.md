# Propagate selected scale precision and prove exact Half membership

After an equivalent arithmetic replacement is admitted, the compiler now
intersects its computed enclosure with the original operation's certificate.
The stronger dyadic grid survives instead of being overwritten by a coarse
format-wide proof. Selected producers likewise transfer the enclosure of
their actual selected expression, rather than doing so only for constants.
This information belongs only to the current branch context.

Finite values proved to lie on a grid with at most eleven significant bits,
within Half's finite/subnormal limits, are already exactly representable in
Half. Both signed zeros also fit. The format-membership check can therefore
remove an identity cast, including for an F32 producer. It does not change
the dtype or order of any preceding arithmetic operation. Wider grids,
underflowing values and overflow remain barriers.

The fused R32/sqrt boundary is retained until its numerical expansion.
Format membership alone must not expose a standalone unfinished sqrt;
the signed-zero domain has a specific regression for this interaction.

Seventeen coherent helper tests pass. A new regression gives a selected
scale a deliberately weak original Half certificate, then verifies that
its stronger precision reaches a following F32 subnormal conversion. The
actual emitted expression is under 128 characters and preserves the four
source values, including both signed zeros. Six F32-cell helper tests pass;
a native emitted-expression check covers all 253 grid/zero inputs for the
new Half-membership proof. Existing 1,080 unit-scale, 263,173 F32-grid,
268,216 F32-cell, 69,634 constant-cast, 888,832 normalization and 30,722
selected-propagation cases retain zero mismatches.

Against a freshly reconstructed `e499215` baseline, `near-zero.expr` shrinks
from 24,337 to 23,910 characters. Both are fully substituted one-token
coordinate 2, position 0, for inputs in [-2^-24,2^-24]. Each actual file
agrees bit for bit with Torch CPU on all sixteen IEEE input pairs. The wider
[1/32,5/128] file `current.expr` remains 326,227 characters with zero differences
on all 66,049 pairs. Neither region proves the unrestricted coordinate.

The same first six RMS decisions have a symbolic body of 212,175 characters,
down from 217,133, with 321,876 guard characters instead of 329,374. Bounded
full-domain diagnostics still complete thirteen paths under 32 MiB and
fourteen under 96 MiB, then reject publication at the artifact budget. No
complete coordinate, multiple-token result or full vector is admitted.
Single-run timings do not establish acceleration.

`run.py` and `near_compare.py` reconstruct all three changed helpers in an
isolated baseline directory. `near_zero.py` and `run_one.py` verify actual
emitted files; `diagnose.py` records path/dependency growth; `full_run.py`
attempts complete publication in a separate worker. `record.py` checks live
source identities, hashes, parity and regression logs before appending
`selectedScalePrecisionValidation` to the test map. Run from the repository
root with the compatible Python runtime. JSON files are metadata; expressions
remain mathematical strings. No CUDA execution or parallel speedup is claimed.

Next investigation: the source guards of a later RMS stage restrict outputs
of earlier operations, but do not yet propagate those restrictions back
through their correlated affine dependencies. An inverse enclosure including
all original F32/Half errors may prove additional input restrictions or
unreachable paths. Such a proof is not implemented by this change.
