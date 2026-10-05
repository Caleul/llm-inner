# Backward bounds through an entire stored RMS vector

A projection guard can bound every normalized component to a small magnitude.
Those bounds previously propagated through individual products using a very
wide inverse interval. They did not invert the RMS relation of the entire
vector, retaining large input ranges and expensive branch expressions.

The new rule uses the adapter's existing finite-Half RMS certificate. If the
stored output vector has norm at most H and its original rounding error is
bounded by E, the real RMS norm is at most H+E. With K=(H+E)^2<n, the original
relation n*T/(T+n*epsilon)<=K implies T<=n*epsilon*K/(n-K). The compiler encloses
the source magnitudes using exact rational arithmetic and outward square-root
bounds, then intersects them with the prior prefix and exact source grid.

Only certified finite Half sources, matching contexts, all available components
and exact +/-1 gamma are admitted. Unknown components, foreign contexts and
non-unit gamma retain their previous bounds. The rule changes certificates,
not the numerical mean, sqrt, reciprocal, multiplication or storage. It returns
an independent magnitude cap; the caller owns intersections and contradictions,
including an older nonzero source gap. Both zero signs remain admissible.
The rule runs in the existing forward/backward fixed point before selecting the
next dependency. A frozen guard never inherits its own or a later assumption.

## Fresh proof and parity

The native RMS preimage test covers 37,773,316 central finite-Half input pairs.
Of these, 552,300 satisfy both stated output caps; none violates the inferred
source bounds. Its forward preserves original F32 reduction/division/epsilon,
sqrt/reciprocal and Half storage. It includes both signed input zeros and a
negative unit gamma. Tests also preserve source gaps, prove incompatible prefix
constraints unreachable, and reject unsupported gamma/context or large output
norms. Four existing bidirectional bound tests also pass.

The three fresh coordinate artifacts retain exact checkpoint parity for all
66,317 regional pairs, with no compiler aliases. Their sizes remain 23,898,
326,051 and 2,913,052 characters. They cover coordinate 2, position 0, one token
in those regions, not the complete domain or variable-length last-token output.

Under 32 and 96 MiB publication budgets, the full-domain compiler now resolves
14 candidate paths before rejection, compared with eight before this rule.
The new blocking candidate is a different prefix, so its 172,480,895 body and
110,263,941 guard characters are not a same-path size comparison against the
previous blocking candidate. No full.expr or bounded prefix is published as a
complete coordinate. The actual full worker attempt stops after about 63 seconds
and peaks at 289,783,808 child RSS bytes. This is not a parallel speedup claim.

The focused suite passes all 38 tests. The general suite reports 632 tests:
570 pass, the same 15 test locations fail and 47 skip. The new optional Python
test accounts for the additional skip without the explicit backend environment.
`record.py` verifies source identity, real files/hashes, native results and the
failure map before appending the new entry to the existing test map. Both new
proof sources already participate in compiler savepoint identity; old numerical
states are incompatible. Geometry-only reuse is distinct from numerical reuse.

## Remaining artifact

The complete coordinate, multiple-token parity and complete output vector remain
pending. The new capture records compact mathematical-string diagnostics only;
JSON files are evidence metadata, not an expression representation or runtime
executor. The initial literal-guard investigation found no direct original-
definition matches for the expensive guard sources. Therefore the implemented
change follows the verified whole-vector relation rather than assuming word
payloads or intermediate numerical identities were interchangeable. Surviving
prefixes and frozen guard expansions still need further simplification before
final publication. CPU evidence does not establish CUDA/Colab acceleration.

One possible next reduction is explicit necessary-input screening for an
expensive guard. A temporary true-branch proof may derive an input box B such
that guard => B. The equivalent ordered condition is then B AND the original
guard simplified inside B. Outside B must remain the false branch. This would
permit narrower guard expressions only after the cheap input screening is
actually emitted; applying a guard's own assumed truth directly to its frozen
dispatch remains invalid. This strategy is not implemented or validated here.
