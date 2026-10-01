"""Validate residual identity rules; this is not final-logit acceptance."""
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
        corrections = {"attention": [], "mlp": []}
        correction_hooks = [
            model.layers[0].self_attn.o_proj.register_forward_hook(
                lambda _m, _a, output: corrections["attention"].append(output)),
            model.layers[0].mlp.down_proj.register_forward_hook(
                lambda _m, _a, output: corrections["mlp"].append(output)),
        ]
        choices = torch.tensor([-2 ** -24, -0., 0., 2 ** -24], dtype=torch.float16)
        for length in [1, 2, 3]:
            cases = torch.stack([choices[(torch.arange(length * width).reshape(length, width) + i) % 4]
                                 for i in range(16)])
            with torch.no_grad():
                model(inputs_embeds=cases, use_cache=False)
            actual = reached.pop()
            corrections["attention"].clear()
            corrections["mlp"].clear()
            expected = cases + torch.zeros_like(cases)
            differences = actual.view(torch.int16) != expected.view(torch.int16)
            comparisons.append({"rule": "zero corrections", "width": width, "length": length, "scalars": actual.numel(),
                                "bitDifferences": int(differences.sum())})
        choices = torch.tensor([-65504., -32., -4., 4., 32., 65504.], dtype=torch.float16)
        for length in [1, 2, 3]:
            cases = torch.stack([choices[i % 6].expand(length, width) if i < 6 else
                                 choices[(torch.arange(length * width).reshape(length, width) + i) % 6]
                                 for i in range(16)])
            with torch.no_grad():
                model(inputs_embeds=cases, use_cache=False)
            actual = reached.pop()
            differences = actual.view(torch.int16) != cases.view(torch.int16)
            comparisons.append({"rule": "absorbed nonzero corrections", "width": width,
                                "length": length, "scalars": actual.numel(),
                                "nonzeroAttentionScalars": int((corrections["attention"].pop() != 0).sum()),
                                "nonzeroMlpScalars": int((corrections["mlp"].pop() != 0).sum()),
                                "bitDifferences": int(differences.sum())})
        hook.remove()
        for correction_hook in correction_hooks:
            correction_hook.remove()
    report = {"rule": "hidden = embedding + positive zero for proved zero or absorbed corrections",
              "torch": torch.__version__, "backend": "cpu-eager", "dtype": "float16",
              "linearWeights": 1 / 1024, "normalizationWeights": 1,
              "comparisons": comparisons, "completeModelParity": False}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    assert not any(c["bitDifferences"] for c in comparisons), report
    assert sum(c.get("nonzeroAttentionScalars", 0) for c in comparisons) > 0, report
    assert sum(c.get("nonzeroMlpScalars", 0) for c in comparisons) > 0, report
    print(json.dumps(report))


if __name__ == "__main__":
    main()
