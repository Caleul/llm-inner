# X2-dominant regions and producer expansion diagnosis

The current compiler produced two complete input-only expressions for coordinate 2, position 0: X1 in [-2^-7,2^-7] and X2 in either [-65504,-32] or [32,65504]. Every weight and intermediate is substituted in each emitted flat arm. Each artifact has four paths and no compiler aliases. The negative artifact has 200,303 characters; the positive artifact has 201,903 characters. Their native reference comparisons total 16,400 cases, zero mismatches, including signed-zero inputs and boundaries.

Admission preserves all 26 previously completed leaves and covers 369,143,808 new input patterns without overlaps. The live tree now covers **1,352,716,288 / 4,030,726,144 patterns (33.5601%)**, with **2,678,009,856 pending**. All 28 completed leaves were independently revalidated: **7,294 cases, zero mismatches**. The compiler/checkpoint/backend identity is unchanged. Native sample parity is not exhaustive enumeration of the covered domain.

Two wider candidates were rejected without manifest mutation:

| X1 Half-rank interval | Logical expanded characters | Stored compiler characters | Outcome |
| --- | ---: | ---: | --- |
| [-8192,8192] | 170,419 | 9,033 | Complete flat artifact, 200,303 characters after path guards |
| [-12288,12288] | 24,240,079 | 50,273 | Flat artifact budget exceeded |
| [-19455,19455] | 1,013,336,912 | 52,705 | Flat artifact budget exceeded |

X2 is [-31743,-20480] in this table. Logical expansion estimates are pre-flat-dispatch measurements; they are not final artifact sizes. The second X1 interval is [-0.0625,0.0625]; the third ends just below ±16.

`trace-growth.py` recompiles these domains with the current numerical sources and records actual producer events and direct literal occurrences without constructing a billion-character string. `growth.json` locates the wide-domain final inverse at 67,432,073 expanded characters, final component 0 at 185,901,923, and the output at 1,013,336,912. In that output definition, both final components occur four times. In component 0, earlier definitions occur 34, 70 and 70 times. These counts identify where substitution amplifies earlier expressions; they do not prove every repeated occurrence is eliminable. Branch-specific numerical conditions and rounding boundaries must be proved before removing or factoring copies. The next compiler change should address these producers before raising resource budgets.

`record.py` checks live source compatibility, artifact hashes, unchanged old leaves, exact audited coverage, fresh reports and failed-candidate non-admission; it snapshots the current tree and two effective files. See [artifact-map.json](artifact-map.json), [validation.json](validation.json), and [partition-parity.json](partition-parity.json).

No numerical/compiler source was changed in this run. The existing integration gate remains the 25 passed, zero skipped tests recorded in `../direct-sympy-central-cover/test.log`; it was not rerun for evidence-only additions. No timing speedup, CUDA acceleration, complete-coordinate parity or multiple-token parity is claimed. Expressions remain mathematical strings; JSON files hold state and evidence. The full coordinate, all-token-length adapter, complete output vector and final function remain unfinished.
