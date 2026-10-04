# Certified dominant-square RMS simplification

The compiler now proves, before constructing the generic square-root expansion, whether an original scalar RMS mean equals one Half component squared and divided by its discovered width. For widths 1 and 2, Half squares and division by that width are exact normal F32. Exact rational bounds and strict smaller-neighbor rounding cells certify that the other square and the configured F32 epsilon cannot change the original rounded mean. Zero-crossing, unknown, tie and unsupported-width cases retain the existing implementation.

For a certified dominant component x, the F32-rounded square root is emitted as the rounded absolute Half input times 1 or `0.7071067811865476`, respectively. The absolute value is lowered to elementary operations. The original reciprocal, multiplication, F32/Half boundaries and weight multiplication remain in their original order. No generic square-root or rounding primitive survives in the admitted expression. Every substitution/producer still passes the existing SymPy stabilization. This changes numerical compilation, not scheduling or response storage.

The exact F32 root identity was checked exhaustively for both widths on all finite nonzero Half values, including both signs and all subnormals: **126,972 comparisons, zero mismatches**. An actual compiled normalization was compared on **90,112 inputs, zero mismatches**, including signed zero on the other component. Certificate tests reject equality at a half-cell, absent bounds, crossing-zero dominance and unproved widths.

A matching complete checkpoint-coordinate region demonstrates the reduction:

| Measurement | Previous compiler at b9721b2 | Current compiler |
| --- | ---: | ---: |
| Input region | X1 [-0.125,0.125], X2 [-65504,-1024] | same |
| Logical expanded characters before flat dispatch | 99,188 | 10,613 |
| Stored compilation characters | 8,841 | 3,906 |
| Actual complete artifact characters | 121,889 | 17,115 |
| Flat paths | 4 | 4 |
| Native comparisons against checkpoint | 8,200 | 8,200 |
| Mismatches | 0 | 0 |

The actual artifact shrank **85.96%**. The baseline was compiled using a temporary copy of the committed helpers from b9721b2, not by reverting the shared checkout. Its body is [baseline.expr](baseline.expr); the improved body is [dominated.expr](dominated.expr). Both retain only fundamental X1/X2 runtime inputs and elementary operations. This is a regional numerical simplification result; no end-to-end timing speedup is claimed.

A wider fresh region, X1 [-0.125,0.125], X2 [-65504,-512], did not satisfy the dominance certificate. It followed the previous path and produced the same 200,303-character body, freshly checked on 8,200 cases with zero mismatches. This region was admitted to `artifacts/direct-sympy-input-partitions/dominant-square-state`; the optimized subregion overlaps it and is not counted as additional coverage.

Numerical source compatibility changed. A new state was created by auditing and copying only the full-domain partition geometry from the prior state, resetting every numerical result to pending. Freshly compiled compatible results are recorded in [frontier.json](frontier.json); previous-source coverage is historical evidence and has not been adopted under the new source identity. The prior live state remains available unchanged.

The build passed. All **25 direct-string integrations passed, zero skipped**, including the now-five correlated-RMS tests and all prior regression gates. The test map is preserved in `docs/direct-string-validation.json` with a new `dominantSquareValidation` entry. [record.py](record.py) checks current identity, source hashes, actual artifact hashes, native parity, complete-path counts, audited new-state coverage and integration logs.

The Half-rank conversion in the earlier X2-cover explanation was corrected: rank 12288 means 0.125, not 0.0625. The compiled domains, counts, files and comparisons were already recorded using the correct ranks; only the prose label changed.

The complete coordinate, multiple-token adapter/parity, full output vector and final Rust function remain unfinished. Expressions are mathematical strings, per the latest representation instruction; JSON contains saved compilation state and evidence. CUDA acceleration is not demonstrated by this change.
