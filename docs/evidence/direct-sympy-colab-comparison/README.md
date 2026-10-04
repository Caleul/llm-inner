# Actual emitted coordinate: serial versus two CPU workers on Colab

Access Broker's official `colab_cli` created a new A100 session after verified
readback showed the previous session was gone. The comparison is terminal
`COMPLETED`; both compiler processes exited zero. Source/archive hashes and
original preparation/measurement scripts accompany the results.

The actual emitted function is `coordinate-region.expr`, using mathematical
strings. It is fully substituted for **position zero, output dimension two,
one token, and finite Half X1/X2 in [32,65504]**. This is a restricted input
region, not a complete-domain coordinate or the final compiled model.

| Measurement | Sequential | Two workers |
| --- | ---: | ---: |
| Compile time (s) | 1.843999 | 1.452304 |
| Process wall time incl. imports (s) | 6.579587 | 3.693359 |
| Producers completed | 9 | 9 |
| Coherent paths | 2 | 2 |
| Emitted characters | 28,282 | 28,282 |
| Aggregate RSS sampled every 50ms (bytes) | 680,857,600 | 680,738,816 |

Both files are byte-identical, SHA256
`da7e5dbc78eca35f19eb04f1e3a5ccc7b8706687a5ee8021fe3fe006ee40fc60`.
The downloaded expression was natively compiled and compared with fresh CPU
checkpoint execution: **1,028 comparisons, zero bit mismatches**. No reference
outputs or checkpoint data are used by the emitted expression.

The two-worker backend completed four jobs, two blocks and zero pair merges.
The two-dimensional checkpoint supplies one block to each occupied reduction
lane here; this run does not exercise the pairwise merge phase. Its order
preservation remains covered by the existing parallel test map. Factor and
simplify execute on CPU; the prior CUDA evidence applies to compatible numeric
primitives, not this symbolic expression pipeline.

The backend observed 1,221,746,688 bytes summed RSS during its 46ms parallel
event. The external 50ms sampler missed that short-lived peak. Summed RSS may
also count shared pages repeatedly; neither measure means unique physical RAM.
Concurrency was capped at two workers and an 8GiB configured memory budget.

The observed compile-time reduction was approximately 21%. This single trial
ran sequential first; startup warming and normal variation limit the result.
It does not prove a repeatable speedup or justify extrapolating full-domain
completion time. Coverage did not advance beyond the already completed region.
The full coordinate, variable-length last-token semantics and complete output
vector remain unfinished.
