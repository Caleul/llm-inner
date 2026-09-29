"""Reference F16 linear values for variable sequence lengths on the CPU backend."""
import json
import sys

import torch
from transformers import LlamaForCausalLM


torch.manual_seed(1907)
model = LlamaForCausalLM.from_pretrained(
    "artifacts/tiny-random-llama", local_files_only=True,
    attn_implementation="eager", dtype=torch.float16,
).eval()
names = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.1.self_attn.v_proj",
    "model.layers.0.mlp.down_proj",
]
norm_names = ["model.layers.0.input_layernorm", "model.layers.1.post_attention_layernorm"]
cases = {}
composites = {}
mlps = {}
with torch.no_grad():
    for name in names:
        module = model.get_submodule(name)
        inputs = torch.randn(4, module.in_features, dtype=torch.float32).to(torch.float16)
        cases[name] = {
            "inputs": inputs.view(torch.int16).to(torch.int32).bitwise_and(65535).tolist(),
            "outputs": {
                str(length): module(inputs[:length]).view(torch.int16).to(torch.int32).bitwise_and(65535).tolist()
                for length in range(1, 5)
            },
        }
    for name in norm_names:
        module = model.get_submodule(name)
        inputs = torch.randn(4, module.weight.numel(), dtype=torch.float32).to(torch.float16)
        cases[name] = {
            "inputs": inputs.view(torch.int16).to(torch.int32).bitwise_and(65535).tolist(),
            "outputs": {
                str(length): module(inputs[:length]).view(torch.int16).to(torch.int32).bitwise_and(65535).tolist()
                for length in range(1, 5)
            },
        }
    layer = model.model.layers[0]
    composite_inputs = torch.randn(4, 16, dtype=torch.float32).to(torch.float16)
    composites["layer0_norm_q"] = {
        "inputs": composite_inputs.view(torch.int16).to(torch.int32).bitwise_and(65535).tolist(),
        "outputs": {
            str(length): layer.self_attn.q_proj(layer.input_layernorm(composite_inputs[:length])).view(torch.int16).to(torch.int32).bitwise_and(65535).tolist()
            for length in range(1, 5)
        },
    }
    post_norm_inputs = torch.randn(4, 16, dtype=torch.float32).to(torch.float16)
    composites["layer0_post_norm_mlp"] = {
        "inputs": post_norm_inputs.view(torch.int16).to(torch.int32).bitwise_and(65535).tolist(),
        "outputs": {
            str(length): layer.mlp(layer.post_attention_layernorm(post_norm_inputs[:length])).view(torch.int16).to(torch.int32).bitwise_and(65535).tolist()
            for length in range(1, 5)
        },
    }
    for layer_index in range(2):
        mlp = model.model.layers[layer_index].mlp
        mlp_inputs = torch.randn(4, mlp.gate_proj.in_features, dtype=torch.float32).to(torch.float16)
        mlps[str(layer_index)] = {
            "inputs": mlp_inputs.view(torch.int16).to(torch.int32).bitwise_and(65535).tolist(),
            "outputs": {
                str(length): mlp(mlp_inputs[:length]).view(torch.int16).to(torch.int32).bitwise_and(65535).tolist()
                for length in range(1, 5)
            },
        }
json.dump({"torch": torch.__version__, "cases": cases, "composites": composites, "mlps": mlps}, sys.stdout, separators=(",", ":"))
