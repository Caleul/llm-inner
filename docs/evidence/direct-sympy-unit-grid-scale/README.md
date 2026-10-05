# Ordered rounded scale over a certified single dyadic unit

The compiler now reduces a homogeneous conversion when its varying source
is proved to be a multiple of 2^q inside [-2^q,2^q]. This entire domain contains
only the two nonzero endpoints and both signed zeros. The numerical recipe
is restricted to multiplication, constant division, unary signs and R16/R32
casts. Constants include audited pure producers with nonzero singleton
bounds. Every operation retains its original order.

The compiler evaluates that scalar recipe at the positive endpoint, derives
a scale coefficient, and checks all four floating payloads against the
proposed multiplication. Only then may the cast tree be replaced. Overflow,
unknown precision/purity, multiple varying occurrences, wider source domains,
addition and variable division remain barriers. This is a generic numerical
identity over the certified grid, not a lookup of checkpoint outputs.

Five F32-cell helper tests pass. The new test compiles the actual reduced
strings to native C++ and compares 1,080 cases across extreme F64/F32/Half
scales, underflow, intermediate casts, divisions and both signs of zero.
It also proves selected-producer propagation, branch isolation and refusal
to discard an unknown call. Existing 263,173 grid, 268,216 cell and 69,634
constant-cast cases retain zero mismatches.

`near-zero.expr` is an actual fully substituted coordinate 2, position 0,
one token, for both inputs in [-2^-24,2^-24]. All sixteen IEEE input pairs
match Torch CPU bit for bit. Against a freshly rebuilt `79d1437` baseline,
the expression shrinks from 58,417 to 24,337 characters (58.34%), retaining
two paths and no compiler aliases. `near_compare.py` reproduces the baseline;
`near_zero.py` emits and exhaustively verifies each version.

The wider [1/32,5/128] regional file `current.expr` remains 326,227 characters,
two paths, and zero bit differences on all 66,049 pairs. `run.py` rebuilds
both changed helpers in an isolated baseline directory. Single-run timings
do not establish a speedup.

For the unrestricted finite-Half inputs, the same first six RMS decisions
now have a symbolic body of 217,133 characters rather than 2,061,331 (89.47%
smaller), with 329,374 guard characters rather than 1,554,142. Under 32 MiB,
the diagnostic advances from three to thirteen paths; under 96 MiB, eleven
to fourteen. The later endpoint is a different path; its size cannot be
treated as a paired comparison. Both attempts hit the artifact budget and
publish no partial coordinate. Full-domain, multiple-token and full-vector
parity remain unproved.

`diagnose.py` records dependency/path growth and `full_run.py` attempts complete
publication in a separate bounded worker. `record.py` checks fresh compiler
identities, expression hashes, native parity and regression logs before
appending `unitGridScaleValidation` to the existing test map. Drivers run from
the repository root using the compatible Python environment. JSON files are
evidence metadata; expressions remain mathematical strings. No CUDA execution
or parallel speedup is claimed.
