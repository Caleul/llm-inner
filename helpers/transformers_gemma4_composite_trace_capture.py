#!/usr/bin/env python3
"""Pinned authoritative image/video/audio prefill and greedy decode for Gemma 4."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText

from transformers_gemma4_text_trace_capture import (
    cache_payload,
    execution_device_metadata,
    greedy,
    tensor_payload,
    validate,
)


def dense(payload: dict[str, Any], device: str) -> torch.Tensor:
    shape = payload.get("shape")
    values = payload.get("values")
    if (
        not isinstance(shape, list)
        or not shape
        or any(not isinstance(dimension, int) or dimension <= 0 for dimension in shape)
        or not isinstance(values, list)
    ):
        raise ValueError("Gemma 4 composite capture requires dense shape/values input.")
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    if tensor.numel() != torch.tensor(shape).prod().item():
        raise ValueError("Gemma 4 composite dense input does not match its shape.")
    return tensor.reshape(shape)


def cache_equal(left: Any, right: Any) -> bool:
    if not hasattr(left, "layers") or not hasattr(right, "layers") or len(left.layers) != len(right.layers):
        return False
    for left_layer, right_layer in zip(left.layers, right.layers):
        if not torch.equal(left_layer.keys, right_layer.keys) or not torch.equal(left_layer.values, right_layer.values):
            return False
    return True


def operation(operation_id: str, output: str, value: torch.Tensor) -> dict[str, Any]:
    return {"operationId": operation_id, "output": output, "tensor": tensor_payload(value)}


def run(request: dict[str, Any]) -> dict[str, Any]:
    source = Path(request["source"])
    device = request.get("executionDevice", "cpu")
    if device != "cpu":
        raise ValueError("Gemma 4 composite capture is currently pinned to CPU eager execution.")
    model = AutoModelForImageTextToText.from_pretrained(
        source,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    validate(model)
    if model.config._attn_implementation != "eager":
        raise ValueError("Gemma 4 composite capture requires eager attention mask construction.")

    modality = request.get("modality")
    profiles = {
        "image": (model.config.image_token_id, 1, "get_image_features", "image_features", "composite_image_features", "composite_image_scatter"),
        "video": (model.config.video_token_id, 2, "get_video_features", "video_features", "composite_video_features", "composite_video_scatter"),
        "audio": (model.config.audio_token_id, 0, "get_audio_features", "audio_features", "composite_audio_features", "composite_audio_scatter"),
    }
    if modality not in profiles:
        raise ValueError("Gemma 4 composite modality must be image, video, or audio.")
    token_id, token_type_id, modal_method_name, feature_output, feature_operation, scatter_operation = profiles[modality]
    tokens = request["inputTokens"]
    positions = request["positionIds"]
    mm_types = request["mmTokenTypeIds"]
    if not isinstance(tokens, list) or not tokens or len(tokens) != len(positions) or len(tokens) != len(mm_types):
        raise ValueError("Gemma 4 composite tokens, positions, and mm token types must align.")
    if tokens.count(token_id) != 1 or (token_type_id != 0 and mm_types.count(token_type_id) != 1) or (token_type_id == 0 and any(mm_types)):
        raise ValueError(f"Gemma 4 composite capture requires exactly one {modality} placeholder and compatible token types.")
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    position_ids = torch.tensor([positions], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor([mm_types], dtype=torch.long, device=device)
    forward_kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "mm_token_type_ids": mm_token_type_ids,
        "use_cache": True,
    }
    if modality == "image":
        forward_kwargs["pixel_values"] = dense(request["pixelValues"], device)
        forward_kwargs["image_position_ids"] = torch.tensor(request["imagePositionIds"], dtype=torch.long, device=device)
    elif modality == "video":
        forward_kwargs["pixel_values_videos"] = dense(request["pixelValuesVideos"], device)
        forward_kwargs["video_position_ids"] = torch.tensor(request["videoPositionIds"], dtype=torch.long, device=device)
    else:
        forward_kwargs["input_features"] = dense(request["inputFeatures"], device)
        forward_kwargs["input_features_mask"] = torch.tensor(request["inputFeaturesMask"], dtype=torch.bool, device=device)
    with torch.inference_mode():
        baseline = model(**forward_kwargs)

    language_model = model.model.language_model
    captured: dict[str, torch.Tensor] = {}
    original_project = language_model.project_per_layer_inputs
    original_modal = getattr(model.model, modal_method_name)

    def language_pre_hook(_module, _args, kwargs):
        if "hidden_states_0" not in captured:
            captured["hidden_states_0"] = kwargs["inputs_embeds"].detach()

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs=None):
        value = original_project(inputs_embeds, per_layer_inputs)
        if "ple_inputs" not in captured:
            captured["ple_inputs"] = value.detach()
        return value

    def modal_features(self, *args, **kwargs):
        output = original_modal(*args, **kwargs)
        features = output.pooler_output
        if modality == "audio":
            features = features[output.attention_mask]
        captured[feature_output] = features.detach()
        return output

    handle = language_model.register_forward_pre_hook(language_pre_hook, with_kwargs=True)
    language_model.project_per_layer_inputs = types.MethodType(project_per_layer_inputs, language_model)
    setattr(model.model, modal_method_name, types.MethodType(modal_features, model.model))
    try:
        with torch.inference_mode():
            prefill = model(**forward_kwargs)
    finally:
        handle.remove()
        language_model.project_per_layer_inputs = original_project
        setattr(model.model, modal_method_name, original_modal)

    if not torch.equal(baseline.logits, prefill.logits) or not cache_equal(baseline.past_key_values, prefill.past_key_values):
        raise ValueError("Gemma 4 composite instrumentation changed authoritative logits or cache.")
    if feature_output not in captured or "hidden_states_0" not in captured or "ple_inputs" not in captured:
        raise ValueError("Gemma 4 composite capture did not observe every required prefill boundary.")

    llm_ids = input_ids.clone()
    llm_ids[(input_ids == model.config.image_token_id) | (input_ids == model.config.video_token_id) | (input_ids == model.config.audio_token_id)] = model.config.text_config.pad_token_id
    with torch.inference_mode():
        text_embeddings = model.model.get_input_embeddings()(llm_ids)

    prefill_operations = [
        operation("composite_text_embedding", "composite_text_embeddings", text_embeddings),
        operation(feature_operation, feature_output, captured[feature_output]),
        operation(scatter_operation, "hidden_states_0", captured["hidden_states_0"]),
        operation("composite_ple_combine_scale", "ple_inputs", captured["ple_inputs"]),
        operation("final_logit_softcap", "softcapped_logits", prefill.logits),
    ]

    max_new_tokens = request["maxNewTokens"]
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("Gemma 4 composite maxNewTokens must be a non-negative integer.")
    current = prefill
    current_position = positions[-1]
    generated: list[int] = []
    steps: list[dict[str, int]] = []
    selection_logits: list[dict[str, Any]] = []
    step_caches: list[list[dict[str, Any]]] = []
    for _ in range(max_new_tokens):
        selection_logits.append(tensor_payload(current.logits))
        token = greedy(current.logits)
        generated.append(token)
        current_position += 1
        steps.append({"tokenId": token, "positionId": current_position})
        with torch.inference_mode():
            current = model(
                input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
                position_ids=torch.tensor([[current_position]], dtype=torch.long, device=device),
                past_key_values=current.past_key_values,
                use_cache=True,
            )
        step_caches.append(cache_payload(current.past_key_values))

    metadata = execution_device_metadata(model, device)
    return {
        "modality": modality,
        "runtime": "transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode",
        "executionMode": "torch.inference_mode",
        "attentionImplementation": model.config._attn_implementation,
        **metadata,
        "generatedTokenIds": generated,
        "steps": steps,
        "selectionLogits": selection_logits,
        "stepPastKeyValues": step_caches,
        "logits": tensor_payload(current.logits),
        "pastKeyValues": cache_payload(current.past_key_values),
        "prefillOperations": prefill_operations,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("usage: transformers_gemma4_composite_trace_capture.py REQUEST.json")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(json.dumps(run(request), separators=(",", ":")))


if __name__ == "__main__":
    main()
