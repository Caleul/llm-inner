"""Discover attention projections and geometry from installed forward source."""
import ast
import inspect
import json
import sys
import textwrap

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from discover_direct_output import attribute_path, find_final_norm, find_logit_projection, single_assignment


def nested_linear_call(node, module):
    found = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call):
            continue
        try:
            path = attribute_path(candidate.func)
            target = module.get_submodule(path)
            if isinstance(target, torch.nn.Linear) and target.bias is None:
                found.append((path, target))
        except (ValueError, AttributeError):
            continue
    if len(found) != 1:
        raise ValueError("Attention projection is not a unique unbiased Linear call")
    return found[0]


def discover_attention(model, layer_path):
    layer = model.get_submodule(layer_path)
    layer_source = inspect.getsource(type(layer).forward)
    layer_forward = next(node for node in ast.walk(ast.parse(textwrap.dedent(layer_source)))
                         if isinstance(node, ast.FunctionDef) and node.name == "forward")
    candidates = []
    for node in ast.walk(layer_forward):
        if not isinstance(node, ast.Call):
            continue
        try:
            path = attribute_path(node.func)
            module = layer.get_submodule(path)
            if "attention_interface(" in inspect.getsource(type(module).forward):
                candidates.append((path, module))
        except (ValueError, AttributeError):
            continue
    if len(candidates) != 1:
        raise ValueError("Attention module is not uniquely discoverable")
    relative, module = candidates[0]
    source = inspect.getsource(type(module).forward)
    forward = next(node for node in ast.walk(ast.parse(textwrap.dedent(source)))
                   if isinstance(node, ast.FunctionDef) and node.name == "forward")
    calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "attention_interface"]
    if len(calls) != 1 or len(calls[0].args) < 4:
        raise ValueError("Attention interface call is not unique")
    result = {}
    for role, arg in zip(("q", "k", "v"), calls[0].args[1:4]):
        if not isinstance(arg, ast.Name):
            raise ValueError("Attention operand has no source name")
        path, projection = nested_linear_call(single_assignment(forward, arg.id), module)
        result[role] = {"weight": f"{layer_path}.{relative}.{path}.weight",
                        "shape": [projection.out_features, projection.in_features]}
    outputs = [node.value for node in ast.walk(forward) if isinstance(node, ast.Return)]
    if len(outputs) != 1 or not isinstance(outputs[0], ast.Tuple) or not isinstance(outputs[0].elts[0], ast.Name):
        raise ValueError("Attention output is not a named tuple")
    output_projections = []
    for node in ast.walk(forward):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not any(isinstance(target, ast.Name) and target.id == outputs[0].elts[0].id for target in node.targets):
            continue
        try:
            output_projections.append(nested_linear_call(node.value, module))
        except ValueError:
            continue
    if len(output_projections) != 1:
        raise ValueError("Output projection is not unique")
    path, projection = output_projections[0]
    result["o"] = {"weight": f"{layer_path}.{relative}.{path}.weight",
                   "shape": [projection.out_features, projection.in_features]}
    config = module.config
    if config.rope_parameters.get("rope_type") != "default":
        raise ValueError("Only default source-confirmed rotary encoding is supported")
    return {"projections": result, "heads": config.num_attention_heads,
            "kvHeads": config.num_key_value_heads, "headDim": module.head_dim,
            "scaling": module.scaling, "ropeTheta": config.rope_parameters["rope_theta"],
            "maxPosition": config.max_position_embeddings}


def main(directory, layer_index):
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    _, _, _, forward, logit_call = find_logit_projection(model)
    _, _, _, layers = find_final_norm(model, forward, logit_call)
    index = int(layer_index)
    if index < 0 or index >= len(layers):
        raise ValueError("Layer index outside discovered decoder")
    print(json.dumps({"layer": layers[index], "attention": discover_attention(model, layers[index])}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
