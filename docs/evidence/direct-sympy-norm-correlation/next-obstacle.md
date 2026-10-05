# Remaining dependency near the central boundary

The rejected larger region retains a post-attention MLP normalization. Do not
assume that dependency is invisible or reuse the two-normalization expression.
The full coordinate, other coordinates and multiple-token output remain pending.

The current attention update bounds are absolute and symmetric. For this
fixture, exact proof-only composition of O and V rows gives these coefficients
on the pre-attention normalization vector:

| Row | X-normalized coefficient | Y-normalized coefficient |
| --- | ---: | ---: |
| O0 | 0.0006119238096289337 | -0.00024831853806972504 |
| O1 | -0.00007747116615064442 | 0.0002179201110266149 |

In the mixed negative/positive quadrant, their ideal directions point away
from zero. Symmetric bounds lose that direction and may allow the residual
enclosure to cross a Half cell boundary unnecessarily. This is a hypothesis,
not permission to eliminate the MLP. The next proof must use signed composed
intervals, retain first-projection F32 reduction and Half storage errors,
retain second-projection reduction and storage errors, and propagate the
actual F32/Half residual endpoints before checking the existing cell-radius
criterion. Unsupported or sign-uncertain cases must retain the dependency.

A separate ideal gate/up joint-product bound improves their independent L2
product bound by only about 1.7% for this fixture (ratio 0.9831497917). That
alone is not evidence that the problematic update becomes invisible. The
signed attention/residual enclosure is the more direct issue to investigate.

Any numerical source change invalidates the saved numerical identity. Reuse
only audited geometry, then recompile and validate actual files afresh. Keep
the present 58.5197204606% coverage record as history, not a substitute for
emitting and validating a full coordinate.
