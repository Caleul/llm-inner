"""Diagnostic only: verify certified critical RMS pairs in the actual CPU module."""
import json, sys, torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm
with open(sys.argv[1]) as f:
    certificate = json.load(f)
torch.set_num_threads(1)
module = LlamaRMSNorm(certificate['width'], eps=certificate['epsilon']).half().eval()
count = 0
mismatches = []
with torch.no_grad():
    for pair in certificate['cases']:
        for signs in range(4):
            inputs = [b | (0x8000 if signs & (1 << i) else 0) for i, b in enumerate(pair['inputBits'])]
            expected = [b | (0x8000 if signs & (1 << i) else 0) for i, b in enumerate(pair['referenceBits'])]
            for length in (1, 8):
                tensor = torch.tensor(inputs, dtype=torch.uint16).view(torch.float16).reshape(1, 1, 2).repeat(1, length, 1)
                actual = module(tensor).view(torch.uint16).tolist()
                count += 1
                if actual != [[expected] * length]:
                    mismatches.append({'inputBits': inputs, 'length': length, 'actual': actual, 'expected': expected})
with open(sys.argv[2], 'w') as f:
    json.dump({'torch': torch.__version__, 'cases': count, 'mismatches': mismatches}, f)
