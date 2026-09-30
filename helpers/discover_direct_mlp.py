"""Discover an MLP's learned projections from its installed forward source."""
import ast
import inspect
import json
import sys
import textwrap

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from discover_direct_output import attribute_path, find_final_norm, find_logit_projection


def module_calls(function):
    for node in function.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            try:
                yield attribute_path(node.value.func)
            except ValueError:
                continue


def discover_mlp(model, layer_path):
    layer = model.get_submodule(layer_path)
    source = inspect.getsource(type(layer).forward)
    forward = next(node for node in ast.walk(ast.parse(textwrap.dedent(source)))
                   if isinstance(node, ast.FunctionDef) and node.name == "forward")
    candidates = []
    for relative in module_calls(forward):
        module = layer.get_submodule(relative)
        body = inspect.getsource(type(module).forward)
        if "self.act_fn(" in body and " * " in body:
            candidates.append((relative, module, body))
    if len(candidates) != 1:
        raise ValueError("Expected one source-discovered gated MLP")
    relative, mlp, body = candidates[0]
    mlp_forward = next(node for node in ast.walk(ast.parse(textwrap.dedent(body)))
                       if isinstance(node, ast.FunctionDef) and node.name == "forward")
    returns = [node.value for node in ast.walk(mlp_forward) if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0], ast.Name):
        raise ValueError("MLP return is not a named scalar producer")
    assignments = [node.value for node in ast.walk(mlp_forward) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == returns[0].id for target in node.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0], ast.Call):
        raise ValueError("MLP output is not a direct projection")
    down = attribute_path(assignments[0].func)
    product = assignments[0].args[0]
    if not isinstance(product, ast.BinOp) or not isinstance(product.op, ast.Mult):
        raise ValueError("MLP producer is not a product")
    def unwrap_gate(node):
        if not isinstance(node, ast.Call) or attribute_path(node.func) != "act_fn":
            return None
        nested = node.args[0]
        return attribute_path(nested.func) if isinstance(nested, ast.Call) else None
    gate = unwrap_gate(product.left) or unwrap_gate(product.right)
    other = product.right if unwrap_gate(product.left) else product.left
    up = attribute_path(other.func) if isinstance(other, ast.Call) else None
    if not gate or not up:
        raise ValueError("MLP gate and up projections are not direct module calls")
    activation = inspect.getsource(type(mlp.act_fn).forward)
    if "silu" not in activation.lower():
        raise ValueError("Only source-confirmed SiLU is numerically defined")
    result = {}
    for role, path in (("gate", gate), ("up", up), ("down", down)):
        module = mlp.get_submodule(path)
        if not isinstance(module, torch.nn.Linear) or module.bias is not None:
            raise ValueError(f"{role} must be an unbiased Linear module")
        result[role] = {"weight": f"{layer_path}.{relative}.{path}.weight",
                        "shape": [module.out_features, module.in_features]}
    if result["gate"]["shape"] != result["up"]["shape"] or \
            result["down"]["shape"] != [result["gate"]["shape"][1], result["gate"]["shape"][0]]:
        raise ValueError("Gated MLP projection shapes disagree")
    norms = []
    for norm_relative in module_calls(forward):
        candidate = layer.get_submodule(norm_relative)
        norm_source = inspect.getsource(type(candidate).forward)
        if ".pow(2).mean(" not in norm_source or "torch.rsqrt(" not in norm_source:
            continue
        norm_tree = ast.parse(textwrap.dedent(norm_source))
        eps_names = [node.attr for node in ast.walk(norm_tree) if isinstance(node, ast.Attribute)
                     and isinstance(node.value, ast.Name) and node.value.id == "self"
                     and node.attr != "weight" and isinstance(getattr(candidate, node.attr, None), (int, float))]
        if len(set(eps_names)) != 1 or not hasattr(candidate, "weight"):
            raise ValueError("Layer normalization source is unsupported")
        norms.append({"weight": f"{layer_path}.{norm_relative}.weight",
                      "epsilon": float(getattr(candidate, eps_names[0]))})
    if len(norms) != 2:
        raise ValueError("Expected two source-discovered decoder normalizations")
    return result, norms


def main(directory, layer_index):
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    _, _, _, forward, logit_call = find_logit_projection(model)
    _, _, _, layers = find_final_norm(model, forward, logit_call)
    index = int(layer_index)
    if index < 0 or index >= len(layers):
        raise ValueError("Layer index outside discovered decoder")
    mlp, norms = discover_mlp(model, layers[index])
    print(json.dumps({"layer": layers[index], "mlp": mlp, "normalizations": norms}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
