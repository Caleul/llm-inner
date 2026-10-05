# Simplify selected projection decisions before substituting dependencies

For a linear projection of one stored RMS vector, the real positive inverse
cannot change the sign of its weighted sources. Original Half/F32 rounding can:
this implementation removes the inverse from a sign-only decision only when
that branch's magnitude exceeds an outward error bound including RMS storage,
original ordered projection arithmetic, and the replacement F64 summation.
A matching pre-storage projection can establish the margin with both independent
error bounds. Magnitudes, original reduction order and zero/cancellation paths
retain their original expressions. Unsupported gamma, foreign contexts, forward
sources, bias and insufficient margins keep the original producer.

The branch-local certificate is applied to the already substituted expression,
including closed sign fields that previously retained the expensive producer.
No normalization value is replaced by its sign. SymPy still processes selected
expressions and dependencies; compiler sharing never becomes a runtime alias.

The native proof checks 37,838,852 input pairs with zero mismatches among
36,904,348 certified cases. An actual checkpoint integration compiles its input
RMS and gate projection, preserving both admitted and non-admitted contexts.
The admitted decision body has 107 characters; its fallback has 8,836. The
input-only `selected.expr` contains 62,059 characters including original dispatch
conditions. Native execution finds zero bit mismatches over 135,184 varied
inputs, including signed zeros and cancellation; 117,379 enter the decision.
This fragment is not a complete Llama output coordinate.

Fresh near-zero, positive and mixed regional coordinate files match the baseline
byte for byte, with zero mismatches over all 66,317 regional input pairs. They
show no size improvement. Full-domain publication reaches 15 generated candidates
under 32 MiB and 23 under 96 MiB, versus 14 and 22 previously. These counts include
the last rejected candidate, and changed prefixes are not equal-path comparisons.
The isolated 96 MiB attempt stops at the artifact budget after 73.73 seconds,
with 301,858,816 bytes peak child RSS; its final artifact is not published.
Candidate 24 still has a 168,126,471-character body, unchanged from the baseline.
Timings of incomplete runs do not establish a full compilation speedup.

All 39 focused integrations pass. Concurrent commit 5ecf74e separated historical
and pinned native tests from portable regression tests. The new portable run has
619 tests: 571 pass, zero fail, 48 explicitly skip. Historical and pinned native
prerequisites remain those recorded in `docs/test-suite-map.json`. The selected
integration was repeated after adding its final assertions. `record.py` checks
source/backend/checkpoint identities, actual file hashes and numerical evidence
before appending to both existing test maps without deleting historical entries.
Old numerical savepoints are incompatible; regional compilation was fresh.

Complete-domain output-coordinate parity, multiple-token semantics and the full
output vector remain pending. This evidence uses local CPU, not CUDA or a new
Colab session. The subsequently requested H100 / 512 MiB execution is separate.
