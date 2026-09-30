"""Compare the final chosen logit, every requested position, with a Rust executable.

Input lists are JSON command-line data. No patched model, intermediate boundary,
argmax or tolerance is used. F16/F32 values widen exactly to binary64, so equality
of the widened representation includes signed zero and every source mantissa bit.
"""
import argparse
import json
import struct
import subprocess
import torch
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("executable")
parser.add_argument("dimension", type=int)
parser.add_argument("cases", help="JSON list of nonempty token sequences")
args = parser.parse_args()
torch.set_num_threads(1)
model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float16,
                                            attn_implementation="eager").eval()
count = 0
with torch.no_grad():
    for tokens in json.loads(args.cases):
        expected = model(torch.tensor([tokens]), use_cache=False).logits[0, :, args.dimension]
        expected_bits = [str(struct.unpack("<Q", struct.pack("<d", float(x)))[0]) for x in expected]
        result = subprocess.run([args.executable, *map(str, tokens)], check=True, capture_output=True, text=True)
        actual_bits = result.stdout.splitlines()
        if actual_bits != expected_bits:
            first = next((i for i, (a, b) in enumerate(zip(actual_bits, expected_bits)) if a != b),
                         min(len(actual_bits), len(expected_bits)))
            raise AssertionError(json.dumps({"tokens": tokens, "dimension": args.dimension,
                "firstPosition": first, "expectedBits": expected_bits, "actualBits": actual_bits}))
        count += len(tokens)
print(json.dumps({"dimension": args.dimension, "positionsCompared": count,
                  "exactFinalLogitParity": True, "torch": torch.__version__}))
