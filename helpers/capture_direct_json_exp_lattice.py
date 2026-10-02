"""Diagnostic F32 exp profile only; never embedded in generated JSON."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("points", type=int)
parser.add_argument("output")
args = parser.parse_args()
if not 1 <= args.points <= 5_704_255:
    raise ValueError("Invalid bounded half-difference lattice size")
torch.set_num_threads(1)
# Every integer is below 2^24, so arange and this dyadic product are exact F32.
x = -torch.arange(args.points, dtype=torch.float32) * (2.0 ** -24)
with torch.inference_mode():
    result = torch.exp(x).contiguous().numpy().astype("<f4", copy=False).tobytes()
Path(args.output).write_bytes(result)
Path(args.output + ".meta.json").write_text(json.dumps({
    "torch": torch.__version__, "architecture": platform.machine(), "threads": 1,
    "points": args.points, "sha256": hashlib.sha256(result).hexdigest(),
    "kernel": "torch.exp F32 CPU; not a claim about softmax-internal dispatch"
}) + "\n")
