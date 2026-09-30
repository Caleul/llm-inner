"""Capture every F16 SiLU result from the declared PyTorch CPU backend."""

import pathlib
import sys

import numpy as np
import torch


if torch.__version__ != "2.12.1":
    raise RuntimeError(f"Expected PyTorch 2.12.1; got {torch.__version__}")

bits = np.arange(65536, dtype=np.uint16)
source = torch.from_numpy(bits.view(np.float16))
result = torch.nn.functional.silu(source).numpy().view(np.uint16)
target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "numeric-profiles/pytorch-2.12.1-cpu-f16-silu.bin")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(result.astype("<u2", copy=False).tobytes())
print(f"{target}: {result.size} results, PyTorch {torch.__version__}, CPU")
