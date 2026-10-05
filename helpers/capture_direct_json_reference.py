"""Diagnostic corpus only: authoritative complete-vector eager CPU outputs.

The compiler does not consume these outputs. Matrices cover signed zero,
subnormals, checkpoint embeddings and independently generated finite F16 values.
"""
import argparse
import json
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama import modeling_llama


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--all-lengths",action="store_true",help="Cover every discovered causal sequence length")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.float16, attn_implementation="eager"
    ).eval()
    # Diagnostic only: observe the exact pre-mask half scores used by the
    # installed eager attention implementation, then call it unchanged.
    score_peaks = []
    original_eager = modeling_llama.eager_attention_forward

    def observe_scores(module, query, key, value, attention_mask, scaling, **kwargs):
        repeated = modeling_llama.repeat_kv(key, module.num_key_value_groups)
        scores = torch.matmul(query, repeated.transpose(2, 3)) * scaling
        score_peaks.append({"layer": int(module.layer_idx),
                            "byHeadMaximumMagnitude": scores.abs().amax(dim=(-2, -1))[0].tolist()})
        return original_eager(module, query, key, value, attention_mask, scaling, **kwargs)

    modeling_llama.eager_attention_forward = observe_scores
    embedding = model.get_input_embeddings()
    width = embedding.embedding_dim
    context = int(model.config.max_position_embeddings)
    cases = []
    generator = torch.Generator().manual_seed(8675309)
    lengths=range(1,context+1) if args.all_lengths else sorted({1,min(2,context),min(3,context),min(4,context),context})
    for length in lengths:
        matrices = [("embedding", embedding(torch.randint(0, embedding.num_embeddings, (length,), generator=generator)).detach())]
        zeros = torch.zeros(length, width, dtype=torch.float16)
        zeros.view(torch.int16).reshape(-1)[::2] = -32768
        matrices.append(("signed-zeros", zeros))
        matrices.append(("maximum-finite", torch.full((length, width), 65504.0, dtype=torch.float16)))
        matrices.append(("minimum-finite", torch.full((length, width), -65504.0, dtype=torch.float16)))
        alternating = torch.full((length, width), 65504.0, dtype=torch.float16)
        alternating.reshape(-1)[::2] *= -1
        matrices.append(("alternating-extremes", alternating))
        for label, magnitude in [("smallest-normal", 2.0 ** -14),
                                 ("largest-subnormal", 2.0 ** -14 - 2.0 ** -24)]:
            boundary = torch.full((length, width), magnitude, dtype=torch.float16)
            boundary.reshape(-1)[::2] *= -1
            matrices.append((label, boundary))
        mixed = torch.full((length, width), 2.0 ** -24, dtype=torch.float16)
        mixed.reshape(-1)[::2] = -65504.0
        matrices.append(("mixed-magnitudes", mixed))
        for scale in [2.0 ** -24, 0.03125, 1.0, 32.0]:
            matrices.append((f"random-{scale}", (torch.rand(length, width, generator=generator) * (2 * scale) - scale).half()))
        for label, matrix in matrices:
            score_peaks.clear()
            with torch.inference_mode():
                logits = model(inputs_embeds=matrix.unsqueeze(0), use_cache=False).logits[0]
            bits = logits.double().contiguous().view(torch.int64).tolist()
            cases.append({"label": label, "inputBits": [[int(x) & 0xffff for x in row]
                          for row in matrix.contiguous().view(torch.int16).tolist()],
                          "scorePeaks": list(score_peaks),
                          "logitF64Bits": [[f"0x{int(x) & 0xffffffffffffffff:016x}" for x in row] for row in bits]})
    machine=platform.machine().lower()
    architecture='arm64' if machine in ('arm64','aarch64') else machine
    Path(args.output).write_text(json.dumps({"torch": torch.__version__, "backend": f"cpu-{architecture}-eager",
        "machine": platform.machine(), "system": platform.system(),
        "width": width, "vocab": embedding.num_embeddings, "context": context, "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
