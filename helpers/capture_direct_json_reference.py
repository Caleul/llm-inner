"""Diagnostic corpus only: authoritative complete-vector eager CPU outputs.

The compiler does not consume these outputs. Matrices cover signed zero,
subnormals, checkpoint embeddings and independently generated finite F16 values.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.float16, attn_implementation="eager"
    ).eval()
    embedding = model.get_input_embeddings()
    width = embedding.embedding_dim
    context = int(model.config.max_position_embeddings)
    cases = []
    generator = torch.Generator().manual_seed(8675309)
    for length in sorted({1, min(2, context), min(3, context), min(4, context), context}):
        matrices = [("embedding", embedding(torch.randint(0, embedding.num_embeddings, (length,), generator=generator)).detach())]
        zeros = torch.zeros(length, width, dtype=torch.float16)
        zeros.view(torch.int16).reshape(-1)[::2] = -32768
        matrices.append(("signed-zeros", zeros))
        for scale in [2.0 ** -24, 0.03125, 1.0, 32.0]:
            matrices.append((f"random-{scale}", (torch.rand(length, width, generator=generator) * (2 * scale) - scale).half()))
        for label, matrix in matrices:
            with torch.inference_mode():
                logits = model(inputs_embeds=matrix.unsqueeze(0), use_cache=False).logits[0]
            bits = logits.double().contiguous().view(torch.int64).tolist()
            cases.append({"label": label, "inputBits": [[int(x) & 0xffff for x in row]
                          for row in matrix.contiguous().view(torch.int16).tolist()],
                          "logitF64Bits": [[f"0x{int(x) & 0xffffffffffffffff:016x}" for x in row] for row in bits]})
    Path(args.output).write_text(json.dumps({"torch": torch.__version__, "backend": "cpu-arm64-eager",
        "width": width, "vocab": embedding.num_embeddings, "context": context, "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
