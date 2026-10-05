# Branch-certified regional square-root reduction

The 512 MiB attempts on local CPU and A100 reached the same numerical expansion.
This change reduces that expansion at the original F32 square-root boundary,
without changing the model's reduction order, its dtype transitions or branches.
The compiler chooses a smaller rational expression only when the complete
branch enclosure fits one certified normalized interval and binary exponent.
The universal 5/5 expression remains for all unsupported ranges. There is no
runtime kernel selector, new branch, response lookup or residual sqrt call.

Four new intervals admit 2, 3 or 4 source occurrences instead of 5. The previous
2-source narrow certificate remains. Coefficients are reproduced independently
from high-precision Lobatto interpolation. The emitted expressions passed
67,895,912 comparisons across normal/subnormal scales and both scale parities.
Every admitted normalized mantissa/parity was checked: 19,398,662 comparisons,
zero mismatches and zero midpoint hits. The original universal and narrow tests
also passed. Wider failed candidates remain investigation data and are not
included in the compiler. Initial emitted-test endpoints were corrected inward
because rounding a subnormal upper bound could include a value outside the
certified interval; the production certificate was not weakened.

## Actual checkpoint evidence

Fresh compilation of coordinate 2 on the same input region [1/32,5/128]^2
produced a complete input-only mathematical string. Baseline and new artifacts
both passed all 66,049 Half input pairs against the checkpoint forward. The
artifact fell from 326,051 to 268,753 characters (17.57%). Both contain two flat
arms and zero compiler aliases. Only direct_sympy_sqrt.py differs in numerical
source identity; no historical state was reused. These are regional validation
artifacts, not the requested whole-domain/multiple-token output.

The first 27 stabilized branch candidates were measured independently in both
versions. Every ordered guard skeleton matched. Candidate 24's body fell from
168,126,471 to 96,061,439 characters; its guards fell from 109,003,811 to
65,564,183. Candidate 26's body fell from 117,367,675 to 100,045,755. Measurements
cover the same branches and include actual post-substitution dependency costs;
they do not substitute for final artifact parity.

The fresh whole-domain 512 MiB attempt resolved 25 producers and generated 27
candidates, including the final rejected one, versus 26 before this change. It
still hit the flat artifact cap and published no coordinate; other logits were
not advanced. The attempted normal controller initially admitted no worker
because its conservative 16*CAS RAM reservation exceeded available RAM. A
separate single-coordinate harness reused the actual worker and shared artifact
ledger, sampling parent/descendant RSS every 200 ms with an independent 80%-of-
available RAM limit and wall-clock cancellation. Its peak observed RSS was
499,712,000 bytes. Production scheduling and numerical semantics were unchanged.
Concurrent test load makes these run timings unsuitable as a speed benchmark.

Two additional regional comparisons hit their explicit 32 MiB or 180-second
limits. They emitted no final coordinate. Temporary unvalidated files left by
the timed-out workers were inspected, recorded and removed. No timed-out proof
state or incomplete candidate was admitted as a compiled result.

## Validation and remaining goal

All 40 focused native/SymPy integrations passed after rebuilding the project.
The portable suite passed 571 tests with zero failures and 49 explicit skips
(620 tests total). These skips remain the existing optional/historical gates. The test
map retains earlier entries and adds this change's proof, smoke and expansion
measurements. The full coordinate, variable-token compilation, full output
vector and final Rust parity remain unfinished. Input expressions remain strings
as requested; JSON here is evidence and metadata only.
