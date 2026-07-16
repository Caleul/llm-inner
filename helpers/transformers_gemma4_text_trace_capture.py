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


def main(request: dict[str, Any]) -> dict[str, Any]:
    source = Path(request["source"])
    tokens = request["inputTokens"]
    positions = request["positionIds"]
    max_new_tokens = request["maxNewTokens"]
    if not isinstance(tokens, list) or not all(isinstance(token, int) and token >= 0 for token in tokens):
        raise ValueError("inputTokens must be non-negative integer IDs.")
    if not isinstance(positions, list) or not all(isinstance(position, int) and position >= 0 for position in positions):
        raise ValueError("positionIds must be non-negative integer IDs.")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("maxNewTokens must be a non-negative integer.")

    model = AutoModelForImageTextToText.from_pretrained(
        source, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.eval()
    validate(model)
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
