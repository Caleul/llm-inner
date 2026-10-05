# Prefix recipe bounds before branch distribution

Before deciding a conditional, the compiler now propagates the current
prefix's input/RMS bounds through the original numerical recipes in backward
producer order. A producer's stored global enclosure remains a safety bound;
only a valid intersection with its recipe's local enclosure can narrow it.
Forward/cyclic dependencies and unknown operations provide no certificate.
No truth of the candidate condition is assumed. These are compiler-only
proofs; the emitted arithmetic and its rounding order do not change.

The regression chains two dependencies and shows that a later comparison
is false only under its preceding input condition. The overlapping/global
cases remain undecided. The actual emitted expression agrees bit for bit
on all 30,722 finite Half inputs in [-1,1], including both signed zeros.
The helper suite has 15 tests and retains the native 888,832 normalization
and 30,722 selected propagation cases with zero mismatches.

`current.expr` remains an actual expression for coordinate 2, position 0,
one token, with both inputs in [1/32,5/128]. Exhaustive Torch CPU comparison
covers all 66,049 pairs. Its size and two paths are unchanged from the
independently rebuilt `d90b7fc` baseline. A single timing observation is not
evidence of a speedup.

Full-domain compilation remains incomplete. The 32 MiB diagnostic completes
three paths; the 96 MiB diagnostic completes nine rather than the previous
eight under that same budget. The first path's body and guards are unchanged.
The final measured 96 MiB arm has six decisions and 6,000,690 guard characters;
the previous final arm had nine decisions and 6,113,559 guard characters.
Those are different traversal endpoints, so their sizes are not a paired
comparison of the same expression. Both attempts reject publication at the
artifact budget. No complete coordinate is admitted and no path is omitted.

`run.py` reconstructs the prior compiler in an isolated helper directory,
then compiles and verifies each actual regional expression. `diagnose.py`
records path/dependency growth under explicit limits. `full_run.py` performs
a separate worker attempt and records its peak RSS (macOS bytes). None of
these drivers claims full-domain, multiple-token or full-vector parity.

`record.py` checks fresh compiler/checkpoint identities, hashes, exact regional
parity and test logs before appending `prefixRecipeValidation` to the existing
test map. JSON files are evidence metadata; expressions remain mathematical
strings. CUDA and parallel speedup are not claimed.
