"""Capture bit-exact two-token CPU checkpoints for the pinned local Llama fixture."""
import json
import sys

import torch
from transformers import LlamaForCausalLM


def bits(tensor):
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.float16:
        return value.view(torch.int16).reshape(-1).to(torch.int32).bitwise_and(65535).tolist()
    if value.dtype == torch.float32:
        return value.view(torch.int32).reshape(-1).to(torch.int64).bitwise_and(0xFFFFFFFF).tolist()
    raise ValueError(value.dtype)


def capture(ids):
    model = LlamaForCausalLM.from_pretrained(
        "artifacts/tiny-random-llama", local_files_only=True,
        attn_implementation="eager", dtype=torch.float16,
    ).eval()
    records = {}
    handles = []
    names = ["model.embed_tokens", "model.norm", "lm_head"]
    for layer in range(2):
        base = f"model.layers.{layer}"
        names.extend(f"{base}.{suffix}" for suffix in (
            "input_layernorm", "self_attn", "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.o_proj", "post_attention_layernorm",
            "mlp", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ))

    for name in names:
        module = model.get_submodule(name)

        def hook(_, inputs, output, name=name):
            result = output[0] if isinstance(output, tuple) else output
            source = inputs[0] if inputs else None
            records[name] = {"input": [] if source is None else source.reshape(-1).tolist() if source.dtype == torch.int64 else bits(source),
                             "output": bits(result)}

        handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        output = model(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False)
    for handle in handles:
        handle.remove()
    records["hidden_states"] = [bits(value) for value in output.hidden_states]
    records["argmax"] = output.logits.argmax(-1).reshape(-1).tolist()
    return records


if __name__ == "__main__":
    json.dump({"torch": torch.__version__, "transformers": __import__("transformers").__version__,
               "cases": {"1,2": capture([1, 2]), "0,1": capture([0, 1]), "5,7": capture([5, 7]),
                         "22144,30091": capture([22144, 30091])}}, sys.stdout,
              separators=(",", ":"))
