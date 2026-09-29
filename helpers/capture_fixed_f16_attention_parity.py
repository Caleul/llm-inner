"""Capture eager attention scalar inputs/outputs at variable sequence lengths."""
import json
import sys

import torch
from transformers import LlamaForCausalLM


def bits(tensor):
    return tensor.detach().cpu().contiguous().view(torch.int16).to(torch.int32).bitwise_and(65535).tolist()


torch.manual_seed(2109)
model = LlamaForCausalLM.from_pretrained(
    "artifacts/tiny-random-llama", local_files_only=True,
    attn_implementation="eager", dtype=torch.float16,
).eval()
inputs = torch.randn(8, model.config.hidden_size, dtype=torch.float32).to(torch.float16)
cases = {}
for length in range(1, 9):
    traces = {}
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        def pre_hook(_, args, kwargs, layer_index=layer_index):
            traces.setdefault(str(layer_index), {})["layer_input"] = bits(args[0] if args else kwargs["hidden_states"])
        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
        def post_norm_hook(_, args, layer_index=layer_index):
            traces[str(layer_index)]["post_norm_input"] = bits(args[0])
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(post_norm_hook))
        def hook(_, args, kwargs, output, layer_index=layer_index):
            traces[str(layer_index)].update({"input": bits(kwargs["hidden_states"]), "output": bits(output[0])})
        handles.append(layer.self_attn.register_forward_hook(hook, with_kwargs=True))
    with torch.no_grad():
        model(inputs_embeds=inputs[:length].unsqueeze(0), use_cache=False)
    for handle in handles:
        handle.remove()
    cases[str(length)] = traces
stress = {}
for scale in (10, 100, 1000):
    hidden = (inputs[:4].float() * scale).half().unsqueeze(0)
    positions = torch.arange(4).unsqueeze(0)
    rotary = model.model.rotary_emb(hidden, positions)
    mask = torch.zeros((1, 1, 4, 4), dtype=torch.float16)
    mask.masked_fill_(torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1), torch.finfo(torch.float16).min)
    stress[str(scale)] = {}
    with torch.no_grad():
        for layer_index, layer in enumerate(model.model.layers):
            output, _ = layer.self_attn(hidden_states=hidden, position_embeddings=rotary, attention_mask=mask)
            stress[str(scale)][str(layer_index)] = {"input": bits(hidden), "output": bits(output)}
json.dump({"torch": torch.__version__, "cases": cases, "stress": stress}, sys.stdout, separators=(",", ":"))
