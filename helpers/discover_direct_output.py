"""Find a direct linear logit projection from the installed model source.

This reports the first backward substitution boundary. It reads config and
Safetensors, never constructs or consumes the project's model IR.
Unsupported forward source shapes fail explicitly.
"""
import ast
import hashlib
import inspect
import json
import sys
import textwrap
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


def attribute_path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id != "self":
        raise ValueError("Logit projection is not a direct self module call")
    return ".".join(reversed(parts))


def find_logit_projection(model):
    source = inspect.getsource(type(model).forward)
    tree = ast.parse(textwrap.dedent(source))
    forward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "forward")
    returned = []
    for node in ast.walk(forward):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            returned.extend(keyword.value for keyword in node.value.keywords if keyword.arg == "logits")
    if len(returned) != 1 or not isinstance(returned[0], ast.Name):
        raise ValueError("Forward must return exactly one named logits value")
    logit_name = returned[0].id
    assignments = [node.value for node in ast.walk(forward) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == logit_name for target in node.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0], ast.Call):
        raise ValueError("Logits must have one directly called producer")
    module_path = attribute_path(assignments[0].func)
    module = model.get_submodule(module_path)
    if not isinstance(module, torch.nn.Linear) or module.bias is not None:
        raise ValueError("Direct unbiased Linear logit producer required")
    return module_path, module, hashlib.sha256(source.encode()).hexdigest(), forward, assignments[0]


def single_assignment(function, name):
    matches = [node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    matches.extend(node.value for node in ast.walk(function) if isinstance(node, ast.AnnAssign)
                   and isinstance(node.target, ast.Name) and node.target.id == name)
    if len(matches) != 1:
        raise ValueError(f"Expected one source assignment for {name}")
    return matches[0]


def find_final_norm(model, causal_forward, logit_call):
    argument = logit_call.args[0]
    while isinstance(argument, ast.Subscript):
        argument = argument.value
    if not isinstance(argument, ast.Name):
        raise ValueError("Logit input must be a named hidden tensor")
    hidden = single_assignment(causal_forward, argument.id)
    if not isinstance(hidden, ast.Attribute) or not isinstance(hidden.value, ast.Name):
        raise ValueError("Logit input must come from the model output")
    model_call = single_assignment(causal_forward, hidden.value.id)
    if not isinstance(model_call, ast.Call):
        raise ValueError("Model output must come from a module call")
    model_path = attribute_path(model_call.func)
    inner = model.get_submodule(model_path)
    source = inspect.getsource(type(inner).forward)
    tree = ast.parse(textwrap.dedent(source))
    forward = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "forward")
    outputs = [keyword.value for node in ast.walk(forward) if isinstance(node, ast.Return)
               and isinstance(node.value, ast.Call) for keyword in node.value.keywords
               if keyword.arg == hidden.attr]
    if len(outputs) != 1 or not isinstance(outputs[0], ast.Name):
        raise ValueError("Model output has no unique hidden tensor")
    norm_calls = [node.value for node in forward.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == outputs[0].id for target in node.targets)
                  and isinstance(node.value, ast.Call)]
    if len(norm_calls) != 1:
        raise ValueError("Final hidden normalization is not a direct module call")
    norm_path = f"{model_path}.{attribute_path(norm_calls[0].func)}"
    norm = model.get_submodule(norm_path)
    norm_source = inspect.getsource(type(norm).forward)
    if ".pow(2).mean(" not in norm_source or "torch.rsqrt(" not in norm_source:
        raise ValueError("Final normalization semantics unsupported")
    norm_tree = ast.parse(textwrap.dedent(norm_source))
    eps_names = [node.attr for node in ast.walk(norm_tree) if isinstance(node, ast.Attribute)
                 and isinstance(node.value, ast.Name) and node.value.id == "self"
                 and node.attr != "weight" and isinstance(getattr(norm, node.attr, None), (int, float))]
    if len(set(eps_names)) != 1:
        raise ValueError("Final normalization epsilon not unambiguous")
    epsilon = float(getattr(norm, eps_names[0]))
    if not hasattr(norm, "weight") or norm.weight.ndim != 1:
        raise ValueError("Final normalization weight must be a vector")
    loops = [node for node in ast.walk(forward) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name)
             and any(isinstance(statement, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == outputs[0].id
                             for target in statement.targets)
                     and isinstance(statement.value, ast.Call)
                     and isinstance(statement.value.func, ast.Name)
                     and statement.value.func.id == node.target.id
                     for statement in node.body)]
    if len(loops) != 1:
        raise ValueError("Decoder layer iteration is not discoverable from forward source")
    iterator = loops[0].iter
    if isinstance(iterator, ast.Subscript):
        layer_path = attribute_path(iterator.value)
    else:
        layer_path = attribute_path(iterator)
    layers = inner.get_submodule(layer_path)
    if not isinstance(layers, torch.nn.ModuleList):
        raise ValueError("Decoder source loop does not select a ModuleList")
    selected = len(layers)
    if isinstance(iterator, ast.Subscript) and isinstance(iterator.slice, ast.Slice):
        upper = iterator.slice.upper
        if isinstance(upper, ast.Attribute) and isinstance(upper.value, ast.Attribute) \
                and isinstance(upper.value.value, ast.Name) and upper.value.value.id == "self" \
                and upper.value.attr == "config":
            selected = int(getattr(inner.config, upper.attr))
        else:
            raise ValueError("Decoder layer bound is not a config attribute")
    if selected < 0 or selected > len(layers):
        raise ValueError("Decoder layer bound outside ModuleList")
    layer_paths = [f"{model_path}.{layer_path}.{index}" for index in range(selected)]
    return norm_path, epsilon, hashlib.sha256(norm_source.encode()).hexdigest(), layer_paths


def main(directory):
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    module_path, module, source_hash, causal_forward, logit_call = find_logit_projection(model)
    norm_path, epsilon, norm_hash, layer_paths = find_final_norm(model, causal_forward, logit_call)
    weight_name = f"{module_path}.weight"
    files = sorted(Path(directory).glob("*.safetensors"))
    matches = []
    for path in files:
        with safe_open(path, framework="pt") as reader:
            if weight_name in reader.keys():
                tensor = reader.get_slice(weight_name)
                matches.append((str(path), tensor.get_shape(), tensor.get_dtype()))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Safetensors tensor for {weight_name}")
    path, shape, dtype = matches[0]
    if shape != [module.out_features, module.in_features]:
        raise ValueError("Logit projection shape disagrees with the installed model source")
    norm_weight = f"{norm_path}.weight"
    norm_matches = []
    for candidate in files:
        with safe_open(candidate, framework="pt") as reader:
            if norm_weight in reader.keys():
                norm_tensor = reader.get_slice(norm_weight)
                norm_matches.append((norm_tensor.get_shape(), norm_tensor.get_dtype()))
    if norm_matches != [([module.in_features], "F16")]:
        raise ValueError("Final normalization tensor does not match the logit input")
    inner = model.get_submodule(".".join(norm_path.split(".")[:-1]))
    inner_tree = ast.parse(textwrap.dedent(inspect.getsource(type(inner).forward)))
    embeddings = []
    for node in ast.walk(inner_tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
            try:
                relative = attribute_path(node.value.func)
                candidate = inner.get_submodule(relative)
                if isinstance(candidate, torch.nn.Embedding):
                    embeddings.append((relative, candidate))
            except (ValueError, AttributeError):
                continue
    if len(embeddings) != 1:
        raise ValueError("Input token embedding is not uniquely discoverable")
    embedding_path, embedding = embeddings[0]
    embedding_weight = ".".join(norm_path.split(".")[:-1]) + "." + embedding_path + ".weight"
    embedding_matches = []
    for candidate in files:
        with safe_open(candidate, framework="pt") as reader:
            if embedding_weight in reader.keys():
                value = reader.get_slice(embedding_weight)
                embedding_matches.append((value.get_shape(), value.get_dtype()))
    if embedding_matches != [([embedding.num_embeddings, embedding.embedding_dim], "F16")]:
        raise ValueError("Input embedding weight does not match source")
    print(json.dumps({"architecture": type(model).__name__, "forwardSourceSha256": source_hash,
                      "weight": weight_name, "safetensors": path, "shape": shape, "dtype": dtype,
                      "finalNormWeight": norm_weight, "finalNormEpsilon": epsilon,
                      "finalNormSourceSha256": norm_hash,
                      "decoderLayers": layer_paths, "embeddingWeight": embedding_weight,
                      "embeddingShape": [embedding.num_embeddings, embedding.embedding_dim],
                      "maxPosition": config.max_position_embeddings,
                      "transformers": __import__("transformers").__version__, "torch": torch.__version__}))


if __name__ == "__main__":
    main(sys.argv[1])
