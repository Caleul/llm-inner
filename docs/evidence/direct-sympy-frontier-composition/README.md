# Certified producer frontier composition

After substitution and SymPy stabilization, expose one certified producer frontier, close each arm with its own interval, dtype and conditions, then stabilize again. Preserve numerical operation order. Admit a candidate only when its fully restored mathematical string is smaller than the already synchronized baseline. Compiler aliases are restored before emission; they are not runtime variables.

Savepoints now use schema 3, persisting validated arm bounds alongside exact source/checkpoint/domain/backend identities. Incompatible states are rejected before model mutation.

## Validation

94 Python tests and 8 Node tests passed; TypeScript build passed. The broad suite has 599 tests: 581 passed, the same 15 baseline failures, 3 skipped. The new native composition check covers 147,472 input pairs with zero bit mismatches against both reference and previous formulation. Continuous/resumed savepoint output remains identical.

## Actual Colab comparison

All runs used the same A100/12-CPU machine, CPU SymPy, fresh states, 9-GiB string budget and 96-GiB memory budget. CUDA-compatible numerical checks passed; factor/simplify remained on CPU. Baseline commit is f374234325a6a7fa48be4f049f50cc02a675bd3d. Source manifests are included.

Baseline sequential: 144.26 s, sampled aggregate RSS 11,977,453,568 bytes. New sequential: 128.23 s, 9,283,489,792 bytes. New two-worker parallel: 133.05 s, 21,158,248,448 bytes. Fork-shared memory may be counted multiple times in aggregate RSS. Sequential/parallel saved records are identical.

Each run completed 15 dependencies, through model.layers.0.up:0, and evaluated the actual saved producers on 60 input matrices: 300 comparisons per run, zero mismatches. This proves partial producer parity at position zero, not the full output coordinate or last-token output for varying lengths.

The activation is physically emitted and downloaded: 4,279,835,121 characters versus 6,913,579,521 previously (38% smaller). Its SHA256 is 1e1b4e2f94825fa694ea0f31816e51924413e7fccb8265ce2f175470e040b275. Local artifact paths and compressed size are in local-artifact.json; large expressions are excluded from Git.

The next envelope still estimates 20,082,302,882 characters and exceeds the 9,663,676,416-character budget before allocation. No coordinate artifact was admitted. More composition/simplification work remains; increased hardware has not completed the model.
