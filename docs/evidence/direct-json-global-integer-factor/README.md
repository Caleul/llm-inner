# Structural JSON global integer factoring

The updated project instructions make typed structural JSON the expression
source of truth. This change extends the existing JSON lowerer rather than
converting a completed mathematical string into JSON. Every incremental
substitution already reaches `simplifyJsonFixedPoint` before its consumer;
the new global rule participates in that same fixed point and in each branch's
existing condition context. Float constants retain their exact bit strings.

`factorJsonIntegerSum` traverses nested unsigned sums/subtractions, distributes
constant coefficients for analysis, combines structurally identical atoms,
cancels modular coefficients and extracts common coefficient factors. Arithmetic
is in Z/(2^width), for u16/u32/u64. Nonlinear, bitwise and conditional subtrees
remain structural atoms. It changes neither floating arithmetic order nor
rounding boundaries. Atoms must have proved total evaluation, so expressions
with possible invalid integer division/shift remain visible.

A proposal is admitted only when its serialized expression is smaller. The
comparison computes expanded sizes from shared compiler syntax using BigInt;
it does not allocate the repeated copies. Analysis limits of 256 sum visits and
4,096 unique measurement nodes decline the whole rewrite without dropping any
term. A regression fixture has over one billion predicted serialized bytes and
checks factoring without materializing that expansion. Compiler structural IDs
and sharing do not enter the emitted JSON.

Validation covers 65,536 u16 inputs with a deterministic second coordinate and
6,144 overflow cases across u16/u32/u64: 71,680 exact comparisons, not the entire
Cartesian u16 pair domain. It also covers cancellation, common factors, partial
operations, IEEE barriers, fixed points and declining oversized searches. The
source-discovered checkpoint coordinate is checked in memory against fresh
PyTorch outputs for 60 cases, with token lengths 1,2,3,4,8, zero bit mismatches.
That test compares position zero. It does not prove the variable-length final
token or a serialized final artifact.

The interrupted parallel-controller change was completed separately. A whole
worker wave's eventual expressions and input/branch conditions are charged
before numerical validation or publication. Over-budget waves preserve the
previous manifest and remove candidates. Saved verification policies cannot be
downgraded, and new parity must bind the candidate hash. Eight native controller
tests passed; their one/two-worker benchmark retains identical regional domains
and artifact hashes. It remains a regional string-reference benchmark, not a
benchmark of the structural JSON compiler or proof of a complete coordinate.

Actual JSON compiler attempts before/after used the same checkpoint, position 0,
dimension 2, 512 MiB output cap and numerical profiles. The baseline source was
`9e6f3ad`'s JSON simplifier, rebuilt in an isolated copy with the same project
compiler. No historical numerical result was reused. Both attempts predict
56,393,457,973,004,369,801 serialized bytes after condition stabilization and
reject before expression emission. All growth records are identical. Thus this
rule adds proved general algebra but demonstrates no size reduction or speedup
for this checkpoint. The oversized prediction is not an emitted artifact.
`baseline-growth.jsonl` and `current-growth.jsonl` retain each substitution and
dependency's unique nodes, expanded occurrences, decisions and predicted bytes.
The remaining duplication in the lowered numerical conversions still needs
structural/context simplification. More GPU resources do not resolve it.

The TypeScript build passed; structural tests passed 36 with one optional test
skipped; both configured checkpoint/native integrations passed. The portable
suite passed 575/624 with 49 optional integrations skipped and zero failures.
`record.py` validates the evidence and appends a distinct entry to the existing
test maps. No complete model JSON/Rust artifact, full-vector parity or
variable-length last-token parity is claimed. Those remain the active goal.
