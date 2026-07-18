#!/usr/bin/env python3
"""Authoritative real Gemma 4 audio module-boundary capture."""
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


def tensor_payload(value: torch.Tensor, squeeze_batch: bool = False) -> dict[str, Any]:
    value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if squeeze_batch and value.ndim == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def validate(model: Any) -> None:
    if transformers.__version__ != SUPPORTED_TRANSFORMERS or torch.__version__.split("+")[0] != SUPPORTED_TORCH:
        raise ValueError(
            f"Gemma 4 audio capture requires transformers=={SUPPORTED_TRANSFORMERS}, torch=={SUPPORTED_TORCH}; "
            f"received transformers=={transformers.__version__}, torch=={torch.__version__}."
        )
    config = model.config
    if config.model_type != "gemma4" or model.__class__.__name__ != "Gemma4ForConditionalGeneration":
        raise ValueError("Gemma 4 audio capture requires Gemma4ForConditionalGeneration/model_type=gemma4.")
    if config.audio_config.dtype not in ("bfloat16", torch.bfloat16) or config.audio_config.use_clipped_linears is not True:
        raise ValueError("Gemma 4 audio capture requires the registered dense BF16 clipped-linear contract.")
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ValueError("Gemma 4 audio capture found a non-BF16 model parameter.")


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if request.get("executionDevice") != "cpu":
        raise ValueError("Gemma 4 audio capture currently pins the CPU eager runtime.")
    shape = request["inputFeatures"]["shape"]
    values = request["inputFeatures"]["values"]
    if len(shape) != 3 or len(values) != shape[0] * shape[1] * shape[2]:
        raise ValueError("Gemma 4 audio input_features payload is malformed.")
    input_features = torch.tensor(values, dtype=torch.float32).reshape(shape)
    input_mask = torch.tensor(request["inputFeaturesMask"], dtype=torch.bool)
    if list(input_mask.shape) != shape[:2]:
        raise ValueError("Gemma 4 audio input_features_mask does not match input_features.")

    model = Gemma4ForConditionalGeneration.from_pretrained(
        request["source"], local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    validate(model)
    checkpoints: list[dict[str, Any]] = []
    handles: list[Any] = []

    def capture(module: Any, operation_id: str, output: str, squeeze_batch: bool = False) -> None:
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            tensor = result[0] if isinstance(result, tuple) else result
            checkpoints.append({"operationId": operation_id, "output": output, "tensor": tensor_payload(tensor, squeeze_batch)})
        handles.append(module.register_forward_hook(hook))

    audio = model.model.audio_tower
    capture(audio.subsample_conv_projection.layer0, "audio_subsample_0_relu", "audio_subsample_0")
    capture(audio.subsample_conv_projection.layer1, "audio_subsample_1_relu", "audio_subsample_1")
    capture(audio.subsample_conv_projection.input_proj_linear, "audio_input_projection", "audio_hidden_0")
    capture(audio.rel_pos_enc, "audio_relative_positions", "audio_relative_positions", squeeze_batch=True)
    for index, layer in enumerate(audio.layers):
        capture(layer, f"audio_layer_{index}_out_norm", f"audio_hidden_{index + 1}")
    capture(audio.output_proj, "audio_output_projection", "audio_output_projected")
    capture(model.model.embed_audio.embedding_pre_projection_norm, "audio_language_projection_norm", "audio_output_normalized")
    capture(model.model.embed_audio.embedding_projection, "audio_language_projection", "audio_projected_features")

    with torch.no_grad():
        output = model.model.get_audio_features(input_features, input_mask, return_dict=True)
    for handle in handles:
        handle.remove()
    stripped = output.pooler_output[output.attention_mask]
    checkpoints.append({"operationId": "audio_strip_padding", "output": "audio_features", "tensor": tensor_payload(stripped)})
    expected = 2 + 1 + 1 + len(audio.layers) + 3 + 1
    if len(checkpoints) != expected:
        raise ValueError(f"Gemma 4 audio capture emitted {len(checkpoints)} checkpoints; expected {expected}.")
    result = {
        "runtime": f"transformers-{transformers.__version__}/torch-{torch.__version__}-Gemma4Audio-CPU-eager",
        "executionDevice": "cpu",
        "dtypePolicy": "native BF16 modules with source-visible F32 RMSNorm and attention promotions; serialized as F32",
        "inputFeatures": {"shape": shape, "values": values},
        "inputFeaturesMask": request["inputFeaturesMask"],
        "operations": checkpoints,
    }
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
