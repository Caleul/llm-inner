# Prefix numerical proofs inside logical guards

`condition_truth` now proves `And`, `Or` and `Not` using the same numerical
prefix bounds as scalar comparisons. Logical operands are checked in order;
an unknown operand stops numerical reasoning before later lazy operands.
Dependency search and selected substitution use these proofs consistently.
No predicate receives its own assumed truth or a later leaf's constraints.

The 21 coherent-path tests pass, including all 30,722 signed finite Half patterns
in [-1,1] for a logical dispatch that reduces to its input. The original rounded
checkpoint normalization test covers 888,832 cases with zero mismatches.
The focused integration suite passes 38/38. The general suite retains the same
15 failing locations, 570 passes and 47 skips out of 632 tests.

Fresh regional artifacts have zero mismatches over all 66,317 regional pairs.
They contain no compiler aliases. These regions do not establish full-domain,
multiple-token or full-vector parity. Old numerical savepoints are incompatible
because the compiler source identity changed.

The full-domain diagnostics still reject candidate 14: 172,480,895 calculation
characters plus 110,263,941 condition characters, under both 32 and 96 MiB
budgets. Logical proof propagation fixes a real gap but does not reduce that
blocking candidate. No complete coordinate was emitted and no speedup is claimed.

An isolated next experiment checked each prospective guard outcome through the
existing backward constraint fixed point before freezing its expression. It
removed 26 decisions before expansion and retained 14 completed candidate paths;
the blocker calculation remained 172,480,895 characters, while its conditions
fell to 87,524,626 characters. This experiment is not the production validation
record in this directory; integration must preserve prefix dispatch and prove
that every removed branch is unreachable over the unchanged input domain.
