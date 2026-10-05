# Backward numerical constraints before the next substitution

The compiler previously propagated most scalar bounds only toward consumers.
An output normalization branch could restrict a residual, without carrying
that restriction back through the stored residual addition and its MLP input.
This retained contradictory branch combinations and expensive dependencies.

`direct_sympy_recipe_bounds.py` now intersects original numerical certificates
with the current prefix and iterates forward evaluation and backward constraints
before the next substitution. It follows strictly backward producer references.
It inverts rounding using conservative adjacent-value midpoint cells; it carries
constraints through signs, sums, differences and products whose other factor is
proved nonzero. A factor that spans zero requires an existing positive magnitude
bound. Unknown operations remain barriers. Recognized magnitude predicates use
only assumptions already belonging to the prefix, never the decision's own truth
before its guard has been frozen.

The emitted numerical operations do not change. Backward sum/product bounds
include conservative F64 error using exact rational arithmetic and directional
endpoint conversion. Half/F32 ties and both signed zeros remain admissible.
Integer-grid intersections refine certificates, without replacing stored zeros
by unsigned constants. A proved empty intersection rejects a path. Node/pass
limits retain valid constraints and do not authorize removing inputs or paths.
The new helper participates in savepoint source identity; previous numerical
savepoints must be rejected, rather than inheriting these new proofs silently.
Expression artifacts remain mathematical strings, following the latest explicit
user format instruction. JSON files here contain evidence and source identity.

## Fresh validation

Use `/private/tmp/llm-inner-directed-residual-20261005/bin/python` from the
repository root. `bounds-proof.log` records four tests including two native
exhaustive certificates over 10,242 Half values each: all four values selected
by the residual output constraint and all 132 selected by the varying inverse
constraint remain enclosed. These are constraint soundness checks, not complete
checkpoint parity. `coherent-proof.log` records 19 existing helper tests.

Three freshly generated regional coordinate artifacts preserve exact checkpoint
parity in all 66,317 tested pairs (16 near-zero, 66,049 balanced, 252 mixed).
Their sizes and hashes match the previous source version: this change improves
prefix constraint propagation, not their emitted numerical bodies. They contain
no compiler aliases. They cover coordinate 2, position 0, one token, and only
the recorded input regions. They do not establish full-domain or multitoken
parity.

`diagnose.py` tries full-domain publication under 32 and 96 MiB budgets, without
accepting a partial expression. It reaches six and eight completed candidate
paths respectively, then rejects publication. At 32 MiB the last candidate has
6,053,765 body characters and 6,579,693 guard characters, with eight decisions.
At 96 MiB a later candidate still grows to 3,577,798,761 body characters and
2,107,762,736 guard characters. The full-domain attempt takes about 60 seconds
and peaks at 280,969,216 child RSS bytes, then fails its artifact budget. No
complete full-domain file is admitted. The remaining large candidate must be
investigated before increasing resources or advancing to other coordinates.

`record.py` checks actual files/hashes against live compiler identity, verifies
all three regional parities and records test counts and failure locations. It
appends `bidirectionalRecipeBoundsValidation` to the existing test map without
rewriting its historical records. CPU tests do not prove CUDA compatibility or
parallel speedup. The final coordinate, multiple tokens and full output vector
remain pending.

## Regression map and next dependency

The focused suite passes all 35 tests. The general suite reports 629 tests:
570 pass, 15 fail and 44 skip. `record.py` verifies that the 15 failing test
locations exactly match the previous validation: missing Gemma calibration
artifacts, generic Python backend failures and missing legacy example/prompt
files. This is not a green general suite. The new optional Python test accounts
for the additional skip when the explicit Python environment is absent.

In the large last candidate, the recorded dependency growth isolates the next
problem: residual components 29/31 expand to about 30.32 million characters;
producer 34 (`R32(1.0 / R32(sqrt(CompileValue33())))`) expands to 848,969,157.
The final projection 39 reaches 3,577,798,761. These are compiler diagnostic
references, not intermediates permitted in a final artifact. Investigation must
track the selected second-normalization expression and its certified domain,
then reduce repeated source occurrences without changing sqrt, reciprocal,
storage or reduction semantics. Raising publication budgets does not resolve
this dependency expansion.
