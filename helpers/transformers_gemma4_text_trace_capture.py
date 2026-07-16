#!/usr/bin/env python3
"""Pinned native BF16 Transformers capture for Gemma4Text generation.

This is deliberately an authoritative-runtime adapter, not a reimplementation
of the calculation program.  It captures only text-token prefill and cached
greedy decode from Gemma4ForConditionalGeneration; image, video and audio
inputs are rejected by the TypeScript boundary that invokes it.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForImageTextToText


SUPPORTED_TRANSFORMERS = "5.5.0"
SUPPORTED_TORCH = "2.12.1"


def tensor_payload(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    array = value.numpy()
    return {
        "dtype": "F32",
        "shape": list(array.shape),
        "valuesBase64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def cache_payload(cache: Any) -> list[dict[str, Any]]:
    if not hasattr(cache, "layers"):
        raise ValueError(f"Gemma 4 capture expected DynamicCache.layers, received {type(cache).__name__}.")
    result: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(cache.layers):
        key, value = layer.keys, layer.values
        if key is None or value is None:
            continue
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(f"Gemma 4 DynamicCache layer {layer_index} is not BHSD.")
        result.append({"layer": layer_index, "key": tensor_payload(key), "value": tensor_payload(value)})
    if not result:
        raise ValueError("Gemma 4 DynamicCache contains no producer-owned KV tensors.")
    return result


def greedy(logits: torch.Tensor) -> int:
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1:
        raise ValueError(f"Gemma 4 logits have invalid shape {tuple(logits.shape)}.")
    return int(torch.argmax(logits[0, -1], dim=-1).item())


def validate(model: Any) -> None:
    if transformers.__version__ != SUPPORTED_TRANSFORMERS or torch.__version__.split("+")[0] != SUPPORTED_TORCH:
        raise ValueError(
            "Gemma 4 native capture is version-pinned and requires "
            f"transformers=={SUPPORTED_TRANSFORMERS}, torch=={SUPPORTED_TORCH}; received "
            f"transformers=={transformers.__version__}, torch=={torch.__version__}."
        )
    config = model.config
    if config.model_type != "gemma4" or model.__class__.__name__ != "Gemma4ForConditionalGeneration":
        raise ValueError(
            "Gemma 4 native capture requires Gemma4ForConditionalGeneration/model_type=gemma4; received "
            f"{model.__class__.__name__}/{config.model_type}."
        )
    text = config.text_config
    if text.dtype not in ("bfloat16", torch.bfloat16) or text.attention_bias is not False or text.enable_moe_block is not False:
        raise ValueError("Gemma 4 native text capture requires the registered dense BF16, bias-free, non-MoE contract.")
    if getattr(text, "_attn_implementation", None) != "eager":
        raise ValueError(f"Gemma 4 native capture requires eager attention; received {text._attn_implementation}.")
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ValueError("Gemma 4 native capture requires every loaded parameter to remain BF16.")


def forward(model: Any, tokens: list[int], positions: list[int], cache: Any | None = None):
    if not tokens or len(tokens) != len(positions):
        raise ValueError("Gemma 4 tokens and positions must be non-empty vectors of equal length.")
    input_ids = torch.tensor([tokens], dtype=torch.long)
    position_ids = torch.tensor([positions], dtype=torch.long)
    with torch.inference_mode():
        return model(input_ids=input_ids, position_ids=position_ids, past_key_values=cache, use_cache=True)


def operation_checkpoints(model: Any, tokens: list[int], positions: list[int]) -> dict[str, Any]:
    """Capture native module boundaries without sending the candidate IR to Python.

    The stable operation IDs are owned by the registered TypeScript Gemma4Text
    adapter.  We deliberately capture only boundaries that are native modules
    (linear, norm, embedding, complete attention and layer output); residuals,
    activations and rotations still need an authoritative fine-grained trace.
    """
    text = model.model.language_model
    checkpoints: dict[str, dict[str, Any]] = {}

    def save(operation_id: str, output: str, transform=None):
        def hook(_module, _inputs, value):
            if isinstance(value, tuple):
                value = value[0]
            if transform is not None:
                value = transform(value)
            checkpoints[operation_id] = {"operationId": operation_id, "output": output, "tensor": tensor_payload(value)}
        return hook

    hooks = []
    try:
        hooks.append(text.embed_tokens.register_forward_hook(save("token_embedding", "hidden_states_0")))
        if getattr(text, "hidden_size_per_layer_input", None):
            hooks.append(text.embed_tokens_per_layer.register_forward_hook(save(
                "ple_token_identity", "ple_token_identity",
                lambda value: value.reshape(*value.shape[:2], text.config.num_hidden_layers, text.config.hidden_size_per_layer_input),
            )))
            hooks.append(text.per_layer_model_projection.register_forward_hook(save("ple_context_projection", "ple_context_packed")))
            hooks.append(text.per_layer_projection_norm.register_forward_hook(save("ple_context_norm", "ple_context_normalized")))
        for layer_index, layer in enumerate(text.layers):
            prefix = f"layer_{layer_index}"
            hooks.extend([
                layer.input_layernorm.register_forward_hook(save(f"{prefix}_input_norm", f"{prefix}_attn_norm")),
                layer.self_attn.q_proj.register_forward_hook(save(f"{prefix}_q_proj", f"{prefix}_q_linear")),
                layer.self_attn.q_norm.register_forward_hook(save(f"{prefix}_q_norm", f"{prefix}_q_normalized", lambda value: value.transpose(1, 2))),
                layer.self_attn.register_forward_hook(save(f"{prefix}_o_proj", f"{prefix}_attention_projected")),
                layer.post_attention_layernorm.register_forward_hook(save(f"{prefix}_post_attention_norm", f"{prefix}_post_attention_normalized")),
                layer.pre_feedforward_layernorm.register_forward_hook(save(f"{prefix}_pre_ffn_norm", f"{prefix}_ffn_norm")),
                layer.mlp.gate_proj.register_forward_hook(save(f"{prefix}_gate_proj", f"{prefix}_gate")),
                layer.mlp.up_proj.register_forward_hook(save(f"{prefix}_up_proj", f"{prefix}_up")),
                layer.mlp.down_proj.register_forward_hook(save(f"{prefix}_down_proj", f"{prefix}_mlp_output")),
                layer.post_feedforward_layernorm.register_forward_hook(save(f"{prefix}_post_ffn_norm", f"{prefix}_post_ffn_normalized")),
                layer.register_forward_hook(save(f"{prefix}_scalar", f"hidden_states_{layer_index + 1}")),
            ])
            if getattr(layer, "hidden_size_per_layer_input", None):
                hooks.extend([
                    layer.per_layer_input_gate.register_forward_hook(save(f"{prefix}_ple_gate", f"{prefix}_ple_gate_linear")),
                    layer.per_layer_projection.register_forward_hook(save(f"{prefix}_ple_project", f"{prefix}_ple_projected")),
                    layer.post_per_layer_input_norm.register_forward_hook(save(f"{prefix}_post_ple_norm", f"{prefix}_post_ple_normalized")),
                ])
            if not layer.self_attn.is_kv_shared_layer:
                hooks.extend([
                    layer.self_attn.k_proj.register_forward_hook(save(f"{prefix}_k_proj", f"{prefix}_k_linear")),
                    layer.self_attn.k_norm.register_forward_hook(save(f"{prefix}_k_norm", f"{prefix}_k_normalized", lambda value: value.transpose(1, 2))),
                    layer.self_attn.v_norm.register_forward_hook(save(f"{prefix}_v_norm", f"{prefix}_v_normalized", lambda value: value.transpose(1, 2))),
                ])
                if layer.self_attn.v_proj is not None:
                    hooks.append(layer.self_attn.v_proj.register_forward_hook(save(f"{prefix}_v_proj", f"{prefix}_v_linear")))
        hooks.append(text.norm.register_forward_hook(save("final_norm", "final_hidden_states")))
        hooks.append(model.lm_head.register_forward_hook(save("lm_head", "logits")))
        native = forward(model, tokens, positions)
        if model.config.text_config.final_logit_softcapping is not None:
            cap = model.config.text_config.final_logit_softcapping
            checkpoints["final_logit_softcap"] = {
                "operationId": "final_logit_softcap", "output": "softcapped_logits",
                "tensor": tensor_payload(native.logits),
            }
        return {
            "runtime": f"PyTorch {torch.__version__.split('+')[0]} / Transformers {transformers.__version__} Gemma4ForConditionalGeneration eager BF16 native module checkpoints",
            # Hook invocation preserves actual native execution order, which is
            # required for a meaningful first-divergence diagnostic.
            "operations": list(checkpoints.values()),
            "pastKeyValues": cache_payload(native.past_key_values),
        }
    finally:
        for hook in hooks:
            hook.remove()


def main(request: dict[str, Any]) -> dict[str, Any]:
    source = Path(request["source"])
    tokens = request["inputTokens"]
    positions = request["positionIds"]
    if not isinstance(tokens, list) or not all(isinstance(token, int) and token >= 0 for token in tokens):
        raise ValueError("inputTokens must be non-negative integer IDs.")
    if not isinstance(positions, list) or not all(isinstance(position, int) and position >= 0 for position in positions):
        raise ValueError("positionIds must be non-negative integer IDs.")
    model = AutoModelForImageTextToText.from_pretrained(
        source, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()
    validate(model)
    if request.get("mode") == "operation-checkpoints":
        return operation_checkpoints(model, tokens, positions)
    max_new_tokens = request.get("maxNewTokens")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("maxNewTokens must be a non-negative integer.")
    current = forward(model, tokens, positions)
    generated: list[int] = []
    selection_logits: list[dict[str, Any]] = []
    steps: list[dict[str, int]] = []
    step_cache: list[list[dict[str, Any]]] = []
    next_position = positions[-1] + 1
    for _ in range(max_new_tokens):
        selection_logits.append(tensor_payload(current.logits))
        token = greedy(current.logits)
        generated.append(token)
        steps.append({"tokenId": token, "positionId": next_position})
        current = forward(model, [token], [next_position], current.past_key_values)
        step_cache.append(cache_payload(current.past_key_values))
        next_position += 1
    return {
        "runtime": f"PyTorch {torch.__version__.split('+')[0]} / Transformers {transformers.__version__} Gemma4ForConditionalGeneration eager BF16 native capture",
        "generatedTokenIds": generated,
        "steps": steps,
        "selectionLogits": selection_logits,
        "stepPastKeyValues": step_cache,
        "logits": tensor_payload(current.logits),
        "pastKeyValues": cache_payload(current.past_key_values),
    }


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("usage: transformers_gemma4_text_trace_capture.py <request.json>")
        print(json.dumps(main(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))), flush=True)
    except Exception as error:
        print(f"Gemma 4 Transformers capture failed: {error}", file=sys.stderr)
        raise
