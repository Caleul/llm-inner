# Propagate normalization source conditions within each selected branch

The current expression format is a SymPy-compatible mathematical string.
JSON files here contain evidence and compiler identity, not expression trees.
This change advances coordinate 2 at position zero for one token; it does
not complete that coordinate over the entire input domain or multiple tokens.

A normalization with inputs admitting zero still has useful finite-source
metadata. The compiler now publishes it even when its norm lower bound is
zero. Existing exclusions still require a positive lower bound; they do not
claim that a small normalized vector is impossible when its inputs can be zero.

If every original Half source has a tiny-value guard relative to the same
original rounded F32 mean, sum those inequalities. With T=sum(x_i^2),
C=sum(c_i), u=2^-23 and K=(1+u)^(n+3), the original ordered operations give
m <= K*T/n+(1+u)*epsilon. Therefore T < C*(1+u)*epsilon/(1-C*K/n).
Only exact matching source/mean predicates in their certified context admit
this refinement. For the fixture's two 2^-28 guards the Half source enclosure
is [-2^-24,2^-24]. The compiler retains both signs of zero, narrows only the
selected leaf, and never changes the global input domain or earlier guards.
The proof currently supports width two; other widths retain their domains.

The compiler also recomputes numerical recipe enclosures from selected
operands before falling back to broader global bounds. These local enclosures
propagate to the next producer. Numeric operation order and dtype are unchanged;
all substituted strings still pass factor/simplify to stability.

Four RMS helper tests cover identity/context rejection, complementary arms,
source-domain isolation and native normalization. Native checks cover 462,422,016
positive Half pairs (44,294,796 tiny-component cases) with zero cell violations,
and 3,047,424 normalization output cases with zero bit mismatches, including
signed zeros, subnormals and both input placements. The latter also checks the
joint-source enclosure whenever both tiny guards hold. These tests establish
the scoped numerical proofs, not full output-coordinate parity.

`small.expr` is an actual fully substituted artifact for both inputs in
[-2^-24,2^-24], with 58,417 characters, two paths, concrete weights and only
original input variables. It has no producer aliases or pending rounding,
normalization or activation primitives. `run.py` recompiles it under a frozen
source/backend/checkpoint identity, compiles its actual bytes to native code,
and compares all 16 IEEE input pairs in that region with a fresh Torch CPU
checkpoint forward. There are zero bit mismatches. This enumerates inputs in
a verifier; the emitted expression performs arithmetic and does not look up
responses. This regional file is not the requested full-domain artifact.

The full-domain diagnostic still stops at the 32 MiB publication budget.
Its first completed path has a 5,980,067,411-character body. Recomputing local
bounds reduces that path's serialized guards from 3,513,846,635 to
3,314,096,435 characters, but does not reduce the body. No complete full-domain
file is emitted. The next obstacle is repeated expansion of the normalization
refinement: the post-normalization mean reaches 1,451,927 characters and the
final mean 1,395,349,047. The pending domain remains pending; no path is omitted
or considered complete on the basis of these expansion measurements.

`validation.json` and the existing test map record fresh test results and the
unchanged general-suite failures. Historical artifacts remain historical after
source changes; they are not silently reused as current numerical evidence.
This work used CPU. It makes no CUDA or parallel speedup claim.
