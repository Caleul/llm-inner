# Selected numerical simplification

Selected dependencies now pass through exact elementary-operation reduction
before the next dependency. A separate compiler owns only the roots and
payloads of that selected path; annotations for the original, unselected
expression cannot restore discarded branches. SymPy factor/simplify remains
mandatory before and after an admitted replacement. Arithmetic order and
floating boundaries are unchanged.

Unsigned zero/one-mask certificates remove redundant masks and repeated
constant unions. Certificates follow explicit typed operations, including
modular addition/multiplication overflow; they do not inspect or reassociate
floating arithmetic. Completed payload certificates are context-specific.

Exhaustive selectors whose stabilized bodies are structurally identical now
disappear before path distribution. Equality compares IEEE-sensitive AST
signatures, so signed zeros stay distinct. Missing fallbacks and unknown guard
calls are preserved. Pending R16/R32/root/activation boundaries also retain
their selector until closure: those guards supply the domains needed by
arm-specific numerical simplification.

`pre-closure-context-regression.log` records the two integrations that exposed
premature selector elimination. The conversion tests observed lost arm
certificates and the path-conversion tests observed missing narrowing facts.
The closure requirement fixes that regression; the final integration log
records revalidation of those cases and their native numerical comparisons.

Validation:

- `integration-tests.log`: 14 named integrations passed, none failed/skipped.
- `final-path-tests.log`: all eight branch tests passed after the final
  processing-admission adjustment; the emitted normalization prefix was re-read
  and compared natively in 888,832 cases with zero bit mismatches.
- The four new mask tests cover 278,544 comparisons, including arbitrary
  64-bit words and overflow, conservative mask assertions, mandatory CAS
  processing, numeric context isolation and refusal of unproved operands.
- The 32 string tests include exhaustive identical-body removal, distinct
  signed zeros, missing fallback, unknown guard calls and pending conversion
  contexts.
- Strict old-state rejection remains covered by the savepoint tests. No
  legacy state was restored by this fresh coordinate run.

`pre-coordinate.expr` remains only the first input-normalization coordinate.
It is fully expanded and input-only; it is not the full Llama output.

`coordinate-run.json` records a bounded fresh attempt at output dimension 2,
position zero. No final coordinate was emitted: its first path still exceeded
the 1 MiB limit. `validation.json` compares logical expansion with the preceding
coherent-path run. The reductions are small relative to the remaining tens of
billions of characters. This local observation is not a controlled performance
benchmark and demonstrates no CPU, parallel or CUDA speedup. The complete
artifact, its parity, variable-length last-token semantics and other coordinates
remain pending.
