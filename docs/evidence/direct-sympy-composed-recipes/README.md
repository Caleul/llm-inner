# Preserve arithmetic recipes before numerical composition

The streaming registry captured the result of an eager numerical composition
as its recipe. In the checkpoint's gated MLP product, the recorded recipe was
a 636-character word-conversion expression. Subsequent branch-local bounds
could no longer see the original Half/F32 product boundaries directly.

The scoped streaming wrapper now retains the typed mathematical composition
before closure when its operands are compact backward aliases. Every
substitution passes through the existing factor/simplify fixed point. The
normal emitted producer remains closed; only its compiler-only recipe changes.
The recipe is attached only when the composition is the entire producer.
Embedded compositions cannot replace the enclosing operation's recipe.
Nested producers have separate captures, and all wrappers are restored on exit.
No operand payload is copied into these recipes or retained in final output.

The actual checkpoint product recipe is now
`R16(R32(CompileValue24() * CompileValue25()))`. These temporary aliases are
fully substituted during emission. `audit_compare.py` reconstructs `9d227df`
in an isolated directory and audits both versions with their source identities.
The first selected body shrinks from 212,175 to 6,379 characters; its guards
shrink from 321,876 to 6,944. Both first paths have six decisions. These are
growth diagnostics, not a complete-coordinate parity claim.

Fresh full-domain attempts still stop at the artifact budget. The 32 MiB run
completes six paths and the 96 MiB run eight, with thirteen decisions in their
last paths. Previously those budgets reached thirteen/fourteen paths in a
different traversal. The changed decision expansion prevents interpreting
these counts or single-run times as acceleration. No full coordinate is emitted.

Fresh actual files remain unchanged in two fixed input regions: `near-zero.expr`
has 23,910 characters and matches Torch CPU on all sixteen signed-zero/subnormal
pairs; `current.expr` has 326,227 characters and matches all 66,049 pairs in
[1/32,5/128] on both input axes. These are one-token coordinate 2, position 0,
and do not cover the full finite input domain or multiple tokens.

The new regression checks original product boundaries, backward dependencies,
whole-producer ownership and restoration. Eight streaming helper tests pass,
including 30,722 native emitted-literal cases, 888,832 inverse cases and sixty
compiler/reference composition cases. These complement the file parity above.
The focused integration and whole-project results are reconciled by `record.py`;
historical evidence remains in the test map rather than being relabeled current.
JSON files store evidence metadata; mathematical strings remain expressions.

Remaining investigation: follow the newly exposed branch decisions and the
large guard dependencies, then prove correlated bounds before expanding them.
The complete first coordinate, multiple-token parity and full vector remain
required. No CUDA execution or parallel speedup is established here.
