#!/usr/bin/env python3
"""Authoritative real Gemma 4 image/video tower-boundary capture."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import Gemma4ForConditionalGeneration


SUPPORTED_TRANSFORMERS = "5.5.0"
SUPPORTED_TORCH = "2.12.1"


def tensor_payload(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def validate(model: Any) -> None:
    if transformers.__version__ != SUPPORTED_TRANSFORMERS or torch.__version__.split("+")[0] != SUPPORTED_TORCH:
        raise ValueError(
            f"Gemma 4 vision capture requires transformers=={SUPPORTED_TRANSFORMERS}, torch=={SUPPORTED_TORCH}; "
            f"received transformers=={transformers.__version__}, torch=={torch.__version__}."
        )
    config = model.config
    if config.model_type != "gemma4" or model.__class__.__name__ != "Gemma4ForConditionalGeneration":
        raise ValueError("Gemma 4 vision capture requires Gemma4ForConditionalGeneration/model_type=gemma4.")
    if config.vision_config.dtype not in ("bfloat16", torch.bfloat16) or config.vision_config.use_clipped_linears is not True:
        raise ValueError("Gemma 4 vision capture requires the registered dense BF16 clipped-linear contract.")
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ValueError("Gemma 4 vision capture found a non-BF16 model parameter.")


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if request.get("executionDevice") != "cpu" or request.get("invocation") not in ("image", "video"):
        raise ValueError("Gemma 4 vision capture pins CPU eager and image/video invocation.")
    shape = request["pixelValues"]["shape"]
    values = request["pixelValues"]["values"]
    expected_rank = 3 if request["invocation"] == "image" else 4
    if len(shape) != expected_rank or shape[-1] != 768 or len(values) != torch.tensor(shape).prod().item():
        raise ValueError("Gemma 4 pixel_values payload is malformed.")
    pixel_values = torch.tensor(values, dtype=torch.float32).reshape(shape)
    positions = torch.tensor(request["pixelPositionIds"], dtype=torch.long)
    if list(positions.shape) != shape[:-1] + [2]:
        raise ValueError("Gemma 4 pixel_position_ids do not match pixel_values.")

    model = Gemma4ForConditionalGeneration.from_pretrained(
        request["source"], local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    validate(model)
    checkpoints: list[dict[str, Any]] = []
    handles: list[Any] = []

    def capture(module: Any, operation_id: str, output: str, selector=lambda result: result) -> None:
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            checkpoints.append({"operationId": operation_id, "output": output, "tensor": tensor_payload(selector(result))})
        handles.append(module.register_forward_hook(hook))

    vision = model.model.vision_tower
    capture(vision.patch_embedder, "vision_patch_embeddings", "vision_hidden_0")
    for index, layer in enumerate(vision.encoder.layers):
        stem = f"vision_layer_{index}"
        capture(layer.input_layernorm, f"{stem}_input_norm", f"{stem}_attn_norm")
        capture(layer.self_attn.q_proj, f"{stem}_q", f"{stem}_q_linear")
        capture(layer.self_attn.q_norm, f"{stem}_q_norm", f"{stem}_q_normalized", lambda result: result.permute(0, 2, 1, 3).contiguous())
        capture(layer.self_attn.k_proj, f"{stem}_k", f"{stem}_k_linear")
        capture(layer.self_attn.k_norm, f"{stem}_k_norm", f"{stem}_k_normalized", lambda result: result.permute(0, 2, 1, 3).contiguous())
        capture(layer.self_attn.v_proj, f"{stem}_v", f"{stem}_v_linear")
        capture(layer.self_attn.v_norm, f"{stem}_v_norm", f"{stem}_v_normalized", lambda result: result.permute(0, 2, 1, 3).contiguous())
        capture(layer.self_attn, f"{stem}_o", f"{stem}_attention_projected", lambda result: result[0])
        capture(layer.post_attention_layernorm, f"{stem}_post_attention_norm", f"{stem}_post_attention_normalized")
        capture(layer.pre_feedforward_layernorm, f"{stem}_pre_ffn_norm", f"{stem}_ffn_norm")
        capture(layer.mlp.gate_proj, f"{stem}_gate", f"{stem}_gate_linear")
        capture(layer.mlp.up_proj, f"{stem}_up", f"{stem}_up_linear")
        capture(layer.mlp.down_proj, f"{stem}_down", f"{stem}_mlp_output")
        capture(layer.post_feedforward_layernorm, f"{stem}_post_ffn_norm", f"{stem}_post_ffn_normalized")
        capture(layer, f"vision_layer_{index}_ffn_residual", f"vision_hidden_{index + 1}")
    capture(vision.pooler, "vision_pool_scale", "vision_pooled_scaled", lambda result: result[0])
    capture(vision, "vision_strip_padding", "vision_soft_tokens", lambda result: result.last_hidden_state)
    capture(model.model.embed_vision.embedding_pre_projection_norm, "vision_language_projection_norm", "vision_soft_tokens_normalized")
    capture(model.model.embed_vision.embedding_projection, "vision_language_projection", "image_features")

    with torch.no_grad():
        if request["invocation"] == "image":
            model.model.get_image_features(pixel_values, positions, return_dict=True)
        else:
            model.model.get_video_features(pixel_values, positions, return_dict=True)
    for handle in handles:
        handle.remove()
    expected = 1 + 15 * len(vision.encoder.layers) + 4
    if len(checkpoints) != expected:
        raise ValueError(f"Gemma 4 vision capture emitted {len(checkpoints)} checkpoints; expected {expected}.")
    result = {
        "runtime": f"transformers-{transformers.__version__}/torch-{torch.__version__}-Gemma4Vision-CPU-eager",
        "executionDevice": "cpu",
        "dtypePolicy": "native BF16 modules with source-visible F32 RoPE, softmax and pooling promotions; serialized as F32",
        "pixelValues": {"shape": shape, "values": values},
        "pixelPositionIds": request["pixelPositionIds"],
        "operations": checkpoints,
    }
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
