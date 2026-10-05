# Additional subnormal conversion paths cause the broad-region growth

The [0.03125, 8] mixed-sign compilation has no subnormal normalization
components: the certified smallest stored magnitude is 0.005523681640625.
Extending the maximum to 65504 lowers that enclosure to
0.0000006556510925292969, below the smallest normal Half, 0.00006103515625.
The compiler correctly retains the additional conversion branch.

The first normalized component's stored expression grows from 101 to 363
characters. Its conversion now contains a Piecewise with a subnormal arm
and a normal fallback, guarded by the source-square/mean threshold. The
inverse normalization dependency occurs once in the smaller representation
and three times across the larger one. Substitution then compounds that
growth through later numerical boundaries:

| Producer | Maximum 8 | Maximum 65504 |
| --- | ---: | ---: |
| pre-normalization component 0 | 2,987 | 6,589 |
| V projection 1 | 6,103 | 53,240 |
| residual component 0 | 122,798 | 426,674 |
| final normalization component 0 | 3,564,183 | 27,313,837 |
| output coordinate 2 | 28,514,048 | 218,511,376 |

These are expanded working-expression counts, not final artifact sizes.
The broad final artifact was not emitted. `growth.json` and
`component-enclosures.json` preserve this diagnosis.

The next proof should connect conversion decisions for different components
of the same RMS vector, before expanding combinations. For example, the
source squares jointly constrain the shared rounded mean; claiming every
component is tiny can contradict that relation and the certified input norm
floor. Any exclusion must include the original F32 reduction, division and
epsilon addition, all storage boundaries, and the branch's frozen context.
It must preserve genuinely reachable subnormal paths. Treating these
decisions as independent or discarding their conversions is not a correction.

There is also a concrete early-elimination opportunity inside a subnormal
arm. Its observed guard is `X1**2 < mean * 2**(-28)`. In this declared region
`abs(X1) >= 0.03125`, so that guard implies the actual shared rounded mean
exceeds 262144. This sharply reduces the relative epsilon contribution and
forces the other normalization component close to sqrt(2). The next proof
can test whether all original F32/Half errors still leave that component in
one Half cell. If so, propagate its constant through V/O and check their
storage cells before expanding the tiny component. This would eliminate
whole dependencies using the branch's own context, rather than merely
factoring copies of a fully expanded conversion. No such constant is adopted
in the present implementation.

This investigation is not implemented as a new exclusion. The current
compiler retains all unproved paths, and the failed broad region admits zero
input pairs. Full-coordinate and multiple-token artifacts remain unfinished.
