# Prove guard outcomes before expanding dispatch

Before substituting and freezing a guard, the compiler now checks its two
prospective outcomes against the original ordered numerical recipes and the
existing prefix constraints. If one outcome is proved unreachable, only the
other context continues and no guard expression is expanded. If neither is
reachable the prefix is discarded; if both remain possible the original guard
is still frozen under its prior context, never under its own assumed truth.
Unsupported operations and resource-limited proofs keep possible outcomes.
Only actual constraint contradictions authorize removal. Input domains and
original calculation/rounding order are unchanged.

The focused integration suite passes all 38 tests. The general suite retains
the same 15 failing locations, 570 passes and 47 skips out of 632 tests.

The coherent suite passes 22 tests. A new test proves that two tiny-magnitude
conditions cannot both hold across an original Half storage and offset, removes
the compound guard before literal expansion, and compares both signed zeros
and all four central input values on each axis. Existing dispatch tests cover
prefix ownership, laziness, sibling isolation, atomic publication and native
rounding. Normalization retains zero mismatches across 888,832 original cases.

Fresh complete regional mathematical-string artifacts contain no compiler
aliases and match the checkpoint bit for bit over all 66,317 regional pairs.
Those regions do not establish full-domain, multiple-token or full-vector
parity. `record.py` verifies artifact hashes, current compiler/backend/checkpoint
identity, and the regression failure map before appending the test-map entry.
Old numerical savepoints are incompatible with the changed compiler identity.

For both 32 and 96 MiB publication budgets the compiler eliminates 26 prospective
guards before expansion and resolves 14 candidate paths before refusing the
next large candidate. That candidate has 172,480,895 calculation characters,
86,741,265 condition characters and six surviving guards. The previous blocker
had the same calculation size but different dispatch, so this is not a proven
same-prefix size comparison. No truncated prefix or partial artifact is emitted.

The isolated 96 MiB full-coordinate attempt takes about 49 seconds and peaks at
289,144,832 child RSS bytes. These are CPU measurements, not evidence of CUDA or
parallel acceleration. The complete coordinate and its full-domain parity
remain unachieved. The dominant surviving calculation still needs simplification.

The next candidate strategy remains explicit necessary-input screening: derive
an input box B implied by an expensive guard, emit B first, and simplify the
original guard only inside that box. Outside B must retain the false branch.
A guard's assumed truth must never directly rewrite its own frozen dispatch.
This screening strategy is not implemented here.
