# Actual A100 run through Access Broker

The initial A100 allocation with high-memory shape failed with Colab Service
Unavailable. An inventory confirmed zero sessions; one fresh allocation without
that shape succeeded. Actual resource readback confirmed NVIDIA A100-SXM4-40GB,
12 CPUs, 89,629,196,288 bytes total RAM and CUDA 13.0. Sources are the deterministic
archive identified by source-archive.json, numerical baseline 351aa1656ab7eb0b.
No historical numerical states were reused. Mathematical expressions remain
strings; JSON files here contain evidence and execution metadata only.

The original checkpoint was also attempted locally with the same 512 MiB
accumulated expression/condition cap. It reached 25 completed producers, generated
26 branch candidates including the rejected final candidate, and stopped at the
artifact cap after 114.233 seconds. Peak observed aggregate RAM was 594,640,896
bytes. No coordinate or vector was published, and no further logits dispatched.
Thus artifact growth, not RAM admission or GPU availability, prevented completion.

All 27 Colab preflight tests passed: 22 coherent-path tests, 3 shared-budget tests
and 2 projection-sign tests. The projection certificate compared 37,838,852 cases
without mismatches. This verifies those transformations, not a complete logit.

The A100 numerical batch compared 888,832 half products and 19,458 cases each of
small-domain SiLU and quadratic guards, with zero bit mismatches. Median product
times were CPU 2.058 ms, resident CUDA 0.07875 ms, and CUDA including transfers
2.489 ms. Resident batches were about 26 times faster, but transfer-inclusive
execution was slower than CPU for this size. This does not establish acceleration
of the symbolic compiler. SymPy factor/simplify remain on CPU.

The remote controller then dispatched the actual checkpoint's first coordinate
with the same 512 MiB cap and separately measured RAM. Remaining independent
logits are eligible for four-worker memory admission only after this coordinate
is emitted and passes exact native sampled parity and integrity checks. Actual
terminal results are bound by record.py. Complete-coordinate/vector and
multiple-token parity remain unmet until demonstrated by final artifacts.

## Terminal result

The actual A100 attempt reached exactly the same final branch statistics as the
local attempt: 25 completed producers, 26 generated candidates including the
rejected final candidate, 26,439 CAS passes, and identical dependency-growth
sizes. The flat expression/condition cap stopped both before a complete output.
Remote worker time was 288.319 seconds versus 114.233 locally (2.52 times longer).
Remote peak aggregate RSS was 1,616,392,192 bytes including the earlier CUDA
validation process; local peak was 594,640,896 bytes. These are full observed run
peaks, not isolated SymPy-only memory comparisons. Machines, architectures,
Python and Torch versions differ, so this is an operational comparison rather
than a controlled CPU/GPU compilation benchmark. The GPU numerical timing above
is the distinct controlled batch comparison within the Colab runtime.

Neither attempt dispatched remaining logits or produced a final coordinate or
vector. Therefore complete-coordinate parity and multiple-token parity were not
executed. No numerical invariant was relaxed to advance. Evidence is captured in
terminal-observation.json and local-run.json; record.py validates hashes, preflight
logs and CUDA bit counts and adds a new test-map entry without replacing any
historical result. The temporary A100 session was stopped after terminal readback.
