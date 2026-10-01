"""Validate the direct affine root rule; this is not final-model acceptance."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)


def check(bits):
    inputs = bits.view(np.float32)
    x = inputs.astype(np.float64)
    _, exponent = np.frexp(x)
    e = (exponent - 1) // 2
    scale = np.ldexp(np.ones(len(x)), 2 * e)
    m = x / scale
    density = np.where(m < 2, 1024, 512)
    lo = np.floor(m * density) / density
    hi = lo + 1 / density
    y0 = torch.sqrt(torch.from_numpy(lo.astype(np.float32))).numpy().astype(np.float64)
    y1 = torch.sqrt(torch.from_numpy(hi.astype(np.float32))).numpy().astype(np.float64)
    slope = (y1 - y0) * density
    offset = y0 - slope * lo
    # These separate operations are also separate operations in the emitted
    # Rust. The affine interpolation is exact binary64 on this input domain.
    candidate = (((slope * m + offset) / (2.0 ** -23) + 2.0 ** 52) - 2.0 ** 52)
    unit = np.ldexp(np.ones(len(x)), e - 23)
    candidate = candidate * unit
    below = candidate - unit / 2
    above = candidate + unit / 2
    actual = np.where(x < below * below, candidate - unit,
                      np.where(x > above * above, candidate + unit, candidate))
    expected = torch.sqrt(torch.from_numpy(inputs)).numpy().astype(np.float64)
    mismatch = np.flatnonzero(actual.view(np.uint64) != expected.view(np.uint64))
    if len(mismatch):
        i = int(mismatch[0])
        raise ValueError(f"Root mismatch input bits={int(bits[i])}: {actual[i]} != {expected[i]}")
    return len(bits)


count = 0
for start in range(0x3f800000, 0x40800001, 65536):
    count += check(np.arange(start, min(start + 65536, 0x40800001), dtype=np.uint32))
generator = np.random.default_rng(97319)
random_count = 0
for _ in range(16):
    random_count += check(generator.integers(1, 0x7f800000, 65536, dtype=np.uint32))
report = {"normalizedF32Values": count, "randomPositiveF32Values": random_count,
          "normalizedAffineCells": 2048,
          "reference": f"torch.sqrt CPU {torch.__version__}", "exactBits": True,
          "completeModelParity": False}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
