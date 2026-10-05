# Next dependency reduction to prove

The signed residual certificate resolves the previously rejected mixed-sign
region. It does not complete the full coordinate or eliminate remaining
dependencies in every region.

The current MLP bound uses independent component magnitudes for its gate/up
projections. The existing `norm_linear_bound()` instead accounts for the shared
Euclidean bound of an actual RMS vector, including its original rounding and
learned gamma storage. Applying it to the concrete projection coefficients,
then adding the original F32 reduction error and Half storage, gives these
diagnostic candidates for the test checkpoint:

| Projection/update | Current bound | Candidate bound |
| --- | ---: | ---: |
| gate | 0.021270751953125 | 0.0206298828125 |
| up | 0.0400390625 | 0.032135009765625 |
| MLP coordinate 0 | 0.000006794929504394531 | 0.000005304813385009766 |
| MLP coordinate 1 | 0.000008046627044677734 | 0.000006258487701416016 |

These values are not implemented or admitted as compiler certificates.
The next change must prove that the bound is attached to the actual named
normalization, preserve both original projection storage boundaries, reject
unsupported configurations, and validate the generated expression against the
reference. In particular, its smaller second-coordinate update might fall
inside residual cells where the current bound fails. This is a reason to test
the next reduction, not permission to widen an existing artifact's domain.
