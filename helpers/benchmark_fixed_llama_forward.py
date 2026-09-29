"""Time the unmodified PyTorch CPU eager forward after loading and warmup."""
import json
import statistics
import sys
import time

import torch
from transformers import LlamaForCausalLM


model = LlamaForCausalLM.from_pretrained(
    "artifacts/tiny-random-llama", local_files_only=True,
    attn_implementation="eager", dtype=torch.float16,
).eval()
inputs = torch.tensor([[1, 2]])
with torch.no_grad():
    for _ in range(3):
        model(input_ids=inputs, use_cache=False)
    times = []
    for _ in range(20):
        start = time.perf_counter_ns()
        output = model(input_ids=inputs, use_cache=False)
        times.append((time.perf_counter_ns() - start) / 1e6)
json.dump({"runtime": "PyTorch CPU eager", "torch": torch.__version__,
           "samples": len(times), "median_ms": statistics.median(times),
           "min_ms": min(times), "next_token": int(output.logits[0, -1].argmax())}, sys.stdout)
