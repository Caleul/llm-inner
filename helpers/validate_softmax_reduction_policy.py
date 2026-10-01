"""Check the discovered CPU reduction correction; not final-model acceptance."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

torch.set_num_threads(1)
generator = np.random.default_rng(97)
rows = []
for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17]:
    for active in sorted({1, min(n, 2), min(n, 3), n}):
        x = generator.uniform(-0.1, 0, (1000, n)).astype(np.float32)
        x[:, active:] = -65504
        expected = torch.softmax(torch.from_numpy(x), -1).numpy()
        exponents = torch.exp(torch.from_numpy(x - x.max(-1, keepdims=True))).numpy()
        lanes = np.zeros((1000, 4), dtype=np.float32)
        for i in range(n):
            lanes[:, i % 4] = lanes[:, i % 4] + exponents[:, i]
        sequential = np.float32(np.float32(np.float32(lanes[:, 0] + lanes[:, 1]) + lanes[:, 2]) + lanes[:, 3])
        paired = np.float32(np.float32(lanes[:, 0] + lanes[:, 2]) + np.float32(lanes[:, 1] + lanes[:, 3]))
        total = sequential if n < 4 else paired
        corrected = np.float32(exponents * np.float32(1 / total)[:, None])
        old = np.float32(exponents * np.float32(1 / sequential)[:, None])
        mismatch = np.count_nonzero(corrected.view(np.uint32) != expected.view(np.uint32))
        if mismatch:
            raise ValueError(f"Corrected softmax mismatch: n={n}, active={active}, count={mismatch}")
        rows.append({"length": n, "active": active, "rows": 1000,
                     "oldMismatches": int(np.count_nonzero(old.view(np.uint32) != expected.view(np.uint32))),
                     "correctedMismatches": int(mismatch)})
report = {"torch": torch.__version__, "backend": "cpu-arm64", "rows": rows,
          "exactBits": True, "completeModelParity": False}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"cases": len(rows) * 1000, "oldMismatches": sum(r["oldMismatches"] for r in rows),
                  "correctedMismatches": 0, "completeModelParity": False}))
