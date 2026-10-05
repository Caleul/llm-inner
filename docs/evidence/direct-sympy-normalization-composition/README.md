# Preserve magnitude exclusions across the residual

A selected prefix can prove that a scalar lies outside a central interval around
zero, while its enclosing min/max range still spans both signs. The numerical
bounds engine previously lost that information in addition, subtraction,
products, storage and squares. The next normalization then used weaker source
bounds, retaining avoidable conversion cases and less precise grids.

The change propagates a magnitude floor through the original F64 arithmetic.
For sums and differences it uses the reverse triangle inequality; for products
and supported divisions it uses the operand floors and denominator maximum.
It includes conservative F64 rounding loss with exact rational arithmetic and
rounds proof endpoints downward. This changes certificates, never the ordered
calculation. Half/F32 storage maps the floor through the actual monotone cast;
a floor that rounds to zero is dropped. A square of the same certified Half
scalar has a positive lower bound when its central interval is excluded.
Unsigned bounds do not erase negative zero or replace signed stored values.
Unsupported operations and zero-crossing divisors remain barriers.

## Actual evidence

`gap-proof.log` tests the residual and its original F32/Half storage, then its
square, against native arithmetic for all 147,456 admitted combinations of
finite Half inputs in the declared test domain. It reports zero violations.
The test also retains cancellation and drops a floor rounded to zero. Existing
bidirectional constraints and coherent-path helper tests are rerun. The new
integration test is included in the existing test map.

The three fresh mathematical-string artifacts are smaller and retain exact
checkpoint parity in all their input regions:

| Artifact | Characters | Exhaustive pairs | Mismatches |
| --- | ---: | ---: | ---: |
| near-zero.expr | 23,898 | 16 | 0 |
| current.expr | 326,051 | 66,049 | 0 |
| mixed.expr | 2,913,052 | 252 | 0 |

They contain no compiler aliases. These are coordinate 2, position 0, one-token
regional artifacts, not the complete coordinate across the whole input domain.
`record.py` checks fresh compiler identity, on-disk hashes and actual parity.
The changed conversion source participates in savepoint identity, so older
numerical savepoints cannot silently inherit its new certificates.

`capture_candidate.py` traces dependency growth without allocating the giant
expression. `baseline-candidate.json` records the previous compiler and
`candidate.json` the current compiler. Their JSON is diagnostic metadata with
mathematical-string text, not a JSON expression representation or a final
runtime artifact. The eighth candidate decreases from 3,577,798,761 to
3,571,931,313 characters, approximately 0.164%. This is a small reduction; it
does not solve the expansion. Its second inverse normalization still contains
14 occurrences of each residual component. The actual complete-domain attempt
rejects publication under 96 MiB after about 66.6 seconds. There is no full.expr,
no claimed speedup, and no omitted input domain.

The focused suite passes all 36 tests. The general suite has 630 tests:
570 pass, the same 15 locations fail and 45 skip. The additional skip is the
new optional Python test without the explicit backend environment. Existing
failures concern calibration artifacts, backend compatibility and missing
legacy example/prompt files. This is not a green general suite.

## Pending final requirement

The main remaining expansion is the composed mean, square-root refinement and
its conversion boundaries. The trace shows that the mean's parity calculation
repeats its source and that subsequent refinement propagates those copies.
Existing sqrt/reciprocal midpoint certificates already remove their own parity
copies; applying that optimization again would not fix the mean. Future work
must simplify that composition with its original storage semantics, not merely
remove rounding or emit an intermediate runtime cache. Complete-coordinate,
multiple-token and full-vector parity remain unproved. CUDA/Colab execution and
parallel speedup are also not established by these CPU results.
