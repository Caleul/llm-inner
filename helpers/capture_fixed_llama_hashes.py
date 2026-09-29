"""Independent whole-logit hashes over deterministic two-token CPU prompts."""
import hashlib
import json
import random
import sys

import torch
from transformers import LlamaForCausalLM


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


model = LlamaForCausalLM.from_pretrained(
    "artifacts/tiny-random-llama", local_files_only=True,
    attn_implementation="eager", dtype=torch.float16,
).eval()
rng = random.Random(1907)
prompts = [[0, 0], [1, 2], [31999, 0], [31999, 31999]]
prompts += [[rng.randrange(32000), rng.randrange(32000)] for _ in range(28)]
cases = []
with torch.no_grad():
    for ids in prompts:
        output = model(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False)
        cases.append({"ids": ids, "layer0_sha256": digest(output.hidden_states[1]),
                      "final_hidden_sha256": digest(output.hidden_states[2]),
                      "logits_sha256": digest(output.logits.to(torch.float16)),
                      "next_token": int(output.logits[0, -1].argmax())})
json.dump({"torch": torch.__version__, "cases": cases}, sys.stdout, separators=(",", ":"))
