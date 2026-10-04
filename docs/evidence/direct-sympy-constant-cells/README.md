# Whole-producer constant cells and compound conditions

The compiler proves a finite producer constant when rounding both interval
endpoints yields the same IEEE payload. Monotonicity then covers every enclosed
value, including tie-to-even boundaries. Numeric equality alone is insufficient:
zero sign ambiguity, overflow and unknown/impure calls block elimination.

The proof runs before traversing a producer and again inside its own branch
context. Pure completed literals can participate without restoring their large
bodies. Purity certificates are isolated from sibling branches.

Compound `And` conditions now refine every component on their selected path;
false `Or` conditions refine each component's complement. Unrepresentable
unions retain a conservative interval. Negated comparison handling remains
unchanged: an attempted extension failed the existing NaN regression and was
removed. The original test assertion and native NaN corpus were preserved.

A second correction compares the *fully substituted* costs of both candidates,
instead of comparing logical size to resident size. Allocation limits are checked
separately. When a completed producer's magnitude is proved zero, the smaller
complete sign-preserving expression can replace the producer itself. This retains
negative zero and does not replace a nonzero value by its sign.

Validation:

- Build passed; final gate: **19 integrations, zero failures or skips**.
- New four-test suite: **317,440** constant-cell and **63,488** signed-zero native
  comparisons, zero mismatches, including every finite Half input pattern.
- Existing 34 conversion tests pass, including **196,608** native branch-fact
  comparisons spanning all Half words and NaN patterns.
- Strict saved-state rejection verified; compiler source hashes changed.

The fresh whole-domain position-zero/dimension-two coordinate composition has
25 completed producers. Resident expression definitions dropped from 846,999
to **34,979** characters. Fully substituted logical size dropped from
13,773,588,190,880 to **11,756,851,964,536** characters. The selected first path
still requires 32,860,461,015 body characters and 18,028,559,449 guard characters.
It cannot be emitted under the configured limit; composition is not completion.

The isolated region comparison uses X1/X2 in [-2^-23,2^-23], including both
zeros: six encodings per input and 36 vectors. Logical expansion dropped from
10,303,598,463 to **7,780,503** characters. It still failed atomic emission with
both the 1MiB and 16MiB limits after branch distribution. No region artifact or
parity is claimed for this attempt. Inputs were not enumerated during compilation;
no response lookup was constructed.

Timing and RSS in the original reports are observations, not a controlled
performance claim. Concurrent validation and differing compilation forms affect
those measurements. The actual improvement proven here is reduced expression
storage/expansion and native primitive parity. The full coordinate, last-token
semantics and output vector remain incomplete.

`validation.json` and the central test map distinguish final-source evidence
from the failed intermediate guard experiment. Metadata remains JSON; expressions
remain mathematical strings. Scripts retain the exact execution and size budgets.
