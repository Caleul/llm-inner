# Selected dependency simplification

The compiler previously left a closed producer expanded when its numerical
recipe referred to a dependency whose marker had already been embedded in
the expression. The selected-tree traversal did not visit that marker, so
the consumer silently retained the old payload instead of simplifying it
under the current branch context.

The correction selects and stabilizes missing, already-decided backward
dependencies before re-closing a consumer recipe. Selected audited scalar
producers also carry their purity certificate. Undecided dependencies remain
barriers; each branch retains its own conditions and numerical order. An
intersection whose minimum magnitude exceeds every value in its enclosure
now rejects the impossible path rather than creating an invalid certificate.

`proof.log` includes a regression which embeds a producer directly into its
consumer's closed expression, while retaining the original numerical recipe.
The emitted expression must simplify below 128 characters and match the
ordered F32 reference, including both signed zeros. Fourteen helper tests,
888,832 native normalization cases and 30,722 propagation cases pass.

`current.expr` is an actual mathematical string for coordinate 2, position 0,
one token, with both inputs in the finite-Half region [1/32, 5/128]. All
66,049 input pairs match Torch CPU bit for bit. The file contains 326,227
characters, two paths and no compiler aliases. This region was already
simplified: its size is unchanged from the independently rebuilt `9877cef`
baseline. Single-run timing observations do not establish a speedup.

The unrestricted finite-Half coordinate remains incomplete. On the same first
six RMS decisions, the symbolic body decreases from 5,980,067,411 to 2,438,851
characters, and its guards from 299,107,294 to 1,838,558. These are measured
symbolic sizes, not a complete emitted file. The 32 MiB diagnostic completes
three paths; 96 MiB completes eight, rejecting two impossible paths. Both stop
at their artifact budget and publish no partial coordinate. Later path sizes
must not be compared as if they were the same first path.

`full-run.json` records the separate bounded worker attempt: 33.28 seconds
inside the compiler and 273,711,104 bytes peak worker RSS on macOS. No complete
coordinate, full-domain parity, multiple-token parity, CUDA execution or
parallel speedup is claimed.

`integration.log` records 34 passing integration tests. `project-test.log`
records 628 tests: 570 pass, 15 fail, 43 skip. The failing test locations match
the preceding evidence exactly: Gemma calibration evidence, the generic
Python/pinned backend, a missing legacy JSON example and missing agent-loop
configuration/prompt. `record.py` verifies the source/checkpoint identities,
artifact hashes, parity counts and regression map before appending
`selectedProducerValidation` to `docs/direct-string-validation.json`.

Run the evidence drivers from the repository root using the compatible
Python environment. `run.py` rebuilds baseline and current regional files;
`diagnose.py` measures unrestricted path growth; `full_run.py` attempts
publication under its explicit limits. `record.py` reconciles their outputs.
Expressions are strings; JSON files contain metadata and evidence only.
