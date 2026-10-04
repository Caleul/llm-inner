# Tighter update proofs before dependency expansion

The compiler's dependency-elision bounds multiplied every projection's
absolute sum by two, then doubled the gated Half product again. Across the
attention/MLP chains this compounded unnecessary slack. The new proof uses
exact Fraction sums of absolute Half products and the F32 reduction bound
`sum / (1 - (n + 3) * 2**-24)`. Four zero-initialized lanes and their final
three additions fit this budget; it also covers other F32 binary reduction
orders. Half products are exact F32, and their sums cannot underflow F32.
The admitted term-count/operand limits keep intermediate sums far below
F32 overflow. Outward F64 conversion and monotone final Half storage avoid
rounding the proof down. Nonfinite, non-Half or overflowing bounds fail closed.

The gate and activation remain stored Half, with the existing
`abs(silu(x)) <= abs(x)` contract. Their product is exact F32, so an extra
factor of two is unnecessary. **No model operation, numerical order or
rounding boundary was replaced by the bound.** A dependency is skipped only
when its entire update stays strictly inside the original Half cell.

For this checkpoint the conservative update-cell cuts change from **32/16
to 8/4**. A matched region with X1 in [0.125,1], X2 in [-65504,-64] required
25 producers before and 17 after: post-normalization/gate dependencies can
be excluded before constructing them. Both independently compiled versions
emit the **same 171,977-character expression and SHA256**. There is no final
file-size improvement claimed for that comparison, and its ordered timings
are not a controlled speed benchmark. The effective expression is
[tightened.expr](tightened.expr); 8,196 fresh checkpoint/native comparisons
have zero mismatches in [witness-parity.json](witness-parity.json).

The bound tests check 3,555,328 ordered projection reductions and 888,832
actual native layer-update cases with zero enclosure violations. The build
and **all 27 direct-string integration tests pass, zero skipped**, including
the existing conversion, branch, source-identity, coverage and native-parity
gates. The four-box sequential/parallel test now covers 763,363,328 patterns
with matching expression hashes under the wider, proved cuts.

Changing this proof changes numerical compiler identity. An actual attempt
to resume the prior state was rejected before compilation and preserved its
manifest: [state-compatibility.json](state-compatibility.json). The new state
imports only audited geometry; all 89 formerly completed domains have fresh
coverage. It contains **100 completed leaves / 20 distinct expressions**,
covering **1,867,358,208 patterns (46.328%)**, an increase of **138,412,032**
over the old state. Eighty-six fresh broad compilations yielded **704,878
checkpoint/native comparisons, zero mismatches**. Comparisons may overlap;
they are not an exhaustive enumeration of the covered input domain. Exact
partition audits establish coverage separately from sampled parity.

[artifact-map.json](artifact-map.json) links every completed leaf to actual
expression bytes. Nineteen distinct files already exist in prior evidence
and are referenced instead of copied. Their fresh proofs and checkpoint
comparisons belong to the new source identity. One new expression file is
stored here. [record.py](record.py) audits identities, hashes, old-domain
coverage, reports and test logs, then adds `tightUpdateBoundValidation` to
the existing test map without overwriting earlier evidence.

The compatible live state is
`artifacts/direct-sympy-input-partitions/tight-update-state/frontier.json`.
The `dominant-square-state` retains the preceding source identity and must
not supply cached expressions to the current compiler. The first coordinate
is still incomplete: **2,163,367,936 admitted patterns remain**. No complete
coordinate/vector or multiple-token parity is claimed. Expressions use the
latest requested SymPy string syntax; JSON files contain evidence/state.

Next reduction candidate: on the already-certified Half SiLU polynomial
domain `abs(x) <= B <= 1/16`, its ordered quartic has coefficient in [0,1/4].
Its magnitude is therefore bounded by `B * (1/2 + B/4)`, followed by the
original F32 and Half rounding. The endpoint operations are exact F64 on
this dyadic Half domain. This could tighten the activation bound further;
it is **not implemented or admitted by this change** and requires its own
primitive/enclosure tests before invalidating/recompiling saved states.
