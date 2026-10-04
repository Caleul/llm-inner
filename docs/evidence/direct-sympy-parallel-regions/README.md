# Memory-admitted parallel region continuation

`helpers/direct_sympy_partition_parallel.py` compiles independent pending regions using isolated processes and the unchanged numerical worker protocol. Each worker owns its checkpoint scalar reads, substitutions, SymPy fixed points and branch facts. Parent admission follows original descriptor order after the entire wave is reaped and all completed candidates pass native parity. No reduction order is changed; this is parallelism across independent regions, complementing the existing ordered continuation-block implementation.

The controller verifies the full numerical source/checkpoint/backend identity, complete input root and old artifact hashes before starting. It holds the controller writer lock. Source identity and manifest bytes are checked again after compilation and parity. Candidates are published atomically; pending splits preserve complete input coverage. Content-addressed files share identical completed bodies. No runtime activation cache, response lookup or compiler alias is added.

Memory admission reserves 768 MiB plus 16 times the CAS character budget for every live worker, together with parent RSS. It measures current parent/worker RSS through `ps` on macOS/Linux while compilation runs, reduces admitted concurrency when slots do not fit, and kills/reaps the wave if sampled RSS exceeds the limit. An unmeasurable live worker is an error. Temporary worker outputs are removed on failure. Reported peaks are sampled RSS during compilation, not total-machine memory or a bound on the later native compiler's allocation.

A matching-work benchmark ran the same four checkpoint-coordinate corner regions with one and two workers. The regions, covered domain and artifact hashes were identical:

| Measurement | One worker | Two workers |
| --- | ---: | ---: |
| End-to-end compilation/admission/native validation | 10.347 s | 5.366 s |
| Peak observed compilation RSS | 597,311,488 bytes | 842,317,824 bytes |
| Covered input patterns | 553,648,128 | 553,648,128 |
| Native comparisons | 272 | 272 |
| Mismatches | 0 | 0 |

The measured ratio is **1.93×** for this one matching-work run. This is neither a full-coordinate speedup nor a repeated statistical benchmark. The later live continuation ran alongside the regression gate and is not used for a performance comparison.

The real compatible state then processed 16 attempts with two requested workers, a 3 GiB limit and a ten-second worker limit. Eleven regions completed, each with native parity before admission; five resource-limited regions were subdivided without omitting inputs. Coverage advanced from **1,414,004,736 (35.0806%)** to **1,721,819,136 (42.7173%)**, adding **307,814,400 patterns**. **2,308,907,008 remain pending.** The peak observed compilation RSS was **1,045,135,360 bytes**, below 3,221,225,472 bytes.

All 43 prior completed leaves remain identical. The live tree has **54 complete leaves and 13 distinct actual expressions**. Admission made 2,864 native comparisons without mismatches; the final independent pass over all 54 leaves made **14,062 comparisons without mismatches**. These are sampled reference comparisons with boundaries and signed zeros, not exhaustive enumeration of the covered domain.

Three new helper tests prove rejection before worker startup for an insufficient memory budget, unchanged manifest on incompatible source or insufficient memory, and identical numerical files/full-root geometry for one/two workers. The build and all **26 direct-string integrations passed, zero skipped**. Previous test-map entries are preserved, with a new `parallelRegionValidation` entry. `record.py` audits benchmark/compiler/controller identity, sampled memory limits, artifact hashes, native reports, old-leaf preservation and exact root coverage.

Resume through:

```sh
/private/tmp/llm-inner-pytorch/bin/python helpers/direct_sympy_partition_parallel.py docs/evidence/direct-sympy-test-checkpoint artifacts/direct-sympy-input-partitions/dominant-square-state /tmp/parallel-continuation.json --workers 2 --memory-bytes 3221225472 --max-attempts 16 --region-seconds 10
```

The auxiliary controller's hash is recorded separately; numerical compiler sources were unchanged, so already compatible numerical state was retained. [artifact-map.json](artifact-map.json) maps all completed leaves to real expression files. [frontier.json](frontier.json) is an incomplete saved state, not a full-coordinate artifact.

The scope remains coordinate 2, position 0 and one-token finite Half embedding inputs X1/X2. Full-coordinate parity, multiple-token compilation/parity, complete output vector and final Rust function are unfinished. CUDA execution is not claimed here. Expressions are mathematical strings; JSON contains state and evidence.
