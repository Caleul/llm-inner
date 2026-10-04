# Four-worker continuation and deadline retry correction

A four-worker continuation exposed unnecessary subdivision: a region stopped at a ten-second wall limit, but its same numerical domain produced a complete **638,482-character, two-path expression** when rerun alone with a larger deadline. Compilation took 9.298 seconds inside the worker and 10.092 seconds including startup. Its pre-flat expansion was 609,041 characters with 23,053 stored compiler characters; producer growth is recorded in `diagnosis.json`. This was a deadline limitation, not proof that the expression exceeded the one-MiB artifact limit.

The auxiliary parallel controller now retries the **same domain** once with a larger deadline before subdividing a wall-clock failure. Defaults are `--timeout-retries 1 --retry-factor 3`: a ten-second attempt is followed by thirty seconds. Actual size/path-budget failures continue to subdivide immediately; an exhausted timeout retry also permits subdivision. Every attempt records its effective deadline. Deferred regions remain pending in the audited complete-domain tree. Numerical worker sources, arithmetic order, branch contexts, dtype boundaries and memory limits are unchanged.

Tests prove that the first timeout retains exactly the same input domain without children, that the next attempt receives thirty seconds, and that a genuine size failure then subdivides without losing coverage. In the real continuation, two regions stopped at ten seconds, retained their domains and completed on the following attempts. Both completed artifacts passed native reference parity before admission. The recorded controller hashes distinguish the earlier controller from the corrected policy.

Three local stages ran with four requested workers and a 6 GiB budget:

| Stage | Attempts | Complete regions | Added patterns |
| --- | ---: | ---: | ---: |
| Prior controller, ten-second deadline | 32 | 7 | 5,324,800 |
| Prior controller, thirty-second deadline | 16 | 12 | 491,520 |
| Corrected controller, ten/thirty-second retry | 8 | 3 | 245,760 |

Coverage advanced from **1,721,819,136 (42.7173%)** to **1,727,881,216 (42.8677%)**, adding **6,062,080 patterns**. **2,302,844,928 remain pending.** There are **76 completed leaves and 21 distinct effective files**. All 54 prior completed leaves are unchanged. Peak observed compilation RSS across the stages was **2,718,269,440 bytes**, below the 6,442,450,944-byte limit. These runs overlapped diagnostics and regression work; no four-worker speedup or throughput comparison is claimed.

Admission made 5,720 fresh native comparisons, zero mismatches. Final independent revalidation of all 76 leaves made **19,782 comparisons, zero mismatches**. The standalone [probe.expr](probe.expr) passed 8,196 sampled comparisons and an additional **exhaustive comparison of all 40,960 Half input-vector patterns in its exact rectangle**, zero mismatches. `exhaustive-probe.py` enumerates validation inputs only; the emitted function contains arithmetic and conditions, not a response table. This proves the standalone probe's complete regional parity, not complete-coordinate parity. Its region is fully covered in the current tree; it is not double-counted as new coverage.

The build and all **26 direct-string integration tests passed, zero skipped**, including four parallel-controller helper tests. The prior test map remains intact, with a new `fourWorkerDeadlineValidation` entry. `record.py` checks numerical/controller identity, old-leaf preservation, exact root geometry, actual expression hashes, matching domains on retries, native reports and the exhaustive corpus cardinality. [artifact-map.json](artifact-map.json) maps all completed leaves to real files; [frontier.json](frontier.json) remains an incomplete saved state. Unchanged expression bytes are referenced from the prior evidence folder rather than copied again.

Resume the corrected policy using:

```sh
/private/tmp/llm-inner-pytorch/bin/python helpers/direct_sympy_partition_parallel.py docs/evidence/direct-sympy-test-checkpoint artifacts/direct-sympy-input-partitions/dominant-square-state /tmp/continuation.json --workers 4 --memory-bytes 6442450944 --max-attempts 16 --region-seconds 10 --timeout-retries 1 --retry-factor 3
```

The current scope remains coordinate 2, position 0 and one-token arbitrary finite Half embedding inputs X1/X2. The full coordinate, multiple-token adapter/parity, output vector and final Rust function remain unfinished. Expressions are mathematical strings; JSON holds saved state and evidence. CUDA execution is not demonstrated here.
