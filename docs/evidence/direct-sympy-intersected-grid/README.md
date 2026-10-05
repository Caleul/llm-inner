# Preserve exact precision when intersecting branch proofs

The branch compiler previously intersected two dyadic precision guarantees
for the same value using their finer grid. That discarded an independently
proved coarser grid and prevented exact algebraic factoring. For example,
an original F32 enclosure can use 2^-149, while the selected recipe proves
that every result is a multiple of 2^-23. Their intersection retains 2^-23.

`intersection_quantum` now preserves the stronger known guarantee in guard
recipes, RMS/source proof intersections and selected recipes. An unknown
precision cannot erase an independent known guarantee. Adding two different
operands still uses their finer grid; this change applies only to intersecting
proofs about the same value. It changes no numerical operation or rounding.

The regression proves that a selected ordered R32 producer inherits the
stronger precision and can be factored exactly. Its before/after expressions
match on all 30,722 Half inputs in [-1,1], including both signed zeros. It
also checks unknown precision and the distinct rule for adding operands.
Sixteen coherent helper tests pass, retaining 888,832 native normalization,
30,722 prefix dispatch and 30,722 selected propagation cases without mismatch.

`current.expr` is the actual one-token coordinate 2, position 0, for both
inputs in [1/32,5/128]. It contains 326,227 characters and two paths, identical
to the independently rebuilt `c381663` baseline. Exhaustive comparison with
Torch CPU covers all 66,049 pairs with zero bit differences. This regional
artifact does not prove the unrestricted coordinate.

The same first six RMS decisions now produce a symbolic body of 2,061,331
characters instead of 2,438,851 (15.48% smaller). Guard size falls from
1,838,558 to 1,554,142. The 32 MiB diagnostic still completes three paths;
96 MiB completes eleven rather than nine under the previous compiler. Its
later endpoint is a different path and must not be compared as identical.
Both diagnostics stop at the artifact budget and publish no partial coordinate.

`full_run.py` records a separate bounded worker attempt; `run.py` rebuilds
baseline/current regional strings and compares their actual bytes against
the reference; `diagnose.py` records per-dependency/path growth. Run drivers
from the repository root with the compatible Python environment. `record.py`
checks source identities, expression hashes, parity and test logs before
appending `intersectedGridValidation` to the existing test map.

JSON files are evidence metadata, not expression representations. Full-domain,
multiple-token and full-vector parity remain unproved. No CUDA execution or
parallel speedup is claimed; one timing sample cannot establish acceleration.
