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
        def layer_hook(_, args, kwargs, output, layer_index=layer_index):
            traces[str(layer_index)]["layer_output"] = bits(output)
        handles.append(layer.register_forward_hook(layer_hook, with_kwargs=True))
        def post_norm_hook(_, args, layer_index=layer_index):
            traces[str(layer_index)]["post_norm_input"] = bits(args[0])
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(post_norm_hook))
        def hook(_, args, kwargs, output, layer_index=layer_index):
            traces[str(layer_index)].update({"input": bits(kwargs["hidden_states"]), "output": bits(output[0])})
        handles.append(layer.self_attn.register_forward_hook(hook, with_kwargs=True))
    with torch.no_grad():
        output = model(inputs_embeds=inputs[:length].unsqueeze(0), use_cache=False)
    for handle in handles:
        handle.remove()
    cases[str(length)] = {"layers": traces, "logits": bits(output.logits)}
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
token_ids = [1, 17, 109, 31999, 5, 23, 407, 25000]
token_cases = {}
with torch.no_grad():
    for length in (1, 4, 8):
        output = model(input_ids=torch.tensor([token_ids[:length]]), use_cache=False)
        token_cases[str(length)] = {"ids": token_ids[:length], "logits": bits(output.logits)}
random_token_cases = {}
with torch.no_grad():
    for seed, length in ((42, 3), (2026, 7)):
        generator = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, model.config.vocab_size, (length,), generator=generator).tolist()
        output = model(input_ids=torch.tensor([ids]), use_cache=False)
        random_token_cases[str(seed)] = {"ids": ids, "logits": bits(output.logits)}
json.dump({"torch": torch.__version__, "cases": cases, "stress": stress, "token_cases": token_cases,
           "random_token_cases": random_token_cases},
          sys.stdout, separators=(",", ":"))
