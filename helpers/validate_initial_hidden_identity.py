"""Validate the proved zero-correction rule; this is not final-logit acceptance."""
import argparse
import json
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    comparisons = []
    for width in [2, 4, 8]:
        config = LlamaConfig(hidden_size=width, intermediate_size=width,
                             num_hidden_layers=1, num_attention_heads=1,
                             num_key_value_heads=1, head_dim=width, vocab_size=4,
                             max_position_embeddings=3)
        config._attn_implementation = "eager"
        model = LlamaModel(config).half().eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.fill_(1 if "layernorm" in name or name == "norm.weight" else 1 / 1024)
        reached = []

        def capture(_module, _arguments, output):
            reached.append(output[0] if isinstance(output, tuple) else output)

        hook = model.layers[0].register_forward_hook(capture)
        choices = torch.tensor([-2 ** -24, -0., 0., 2 ** -24], dtype=torch.float16)
        for length in [1, 2, 3]:
            cases = torch.stack([choices[(torch.arange(length * width).reshape(length, width) + i) % 4]
                                 for i in range(16)])
            with torch.no_grad():
                model(inputs_embeds=cases, use_cache=False)
            actual = reached.pop()
            expected = cases + torch.zeros_like(cases)
            differences = actual.view(torch.int16) != expected.view(torch.int16)
            comparisons.append({"width": width, "length": length, "scalars": actual.numel(),
                                "bitDifferences": int(differences.sum())})
        hook.remove()
    report = {"rule": "hidden = embedding + positive zero when both corrections are proved zero",
              "torch": torch.__version__, "backend": "cpu-eager", "dtype": "float16",
              "linearWeights": 1 / 1024, "normalizationWeights": 1,
              "comparisons": comparisons, "completeModelParity": False}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    assert not any(c["bitDifferences"] for c in comparisons), report
    print(json.dumps(report))


if __name__ == "__main__":
    main()
