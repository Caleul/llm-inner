#!/usr/bin/env python3
"""Authoritative real Gemma 4 audio module-boundary capture."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Callable

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
    with torch.no_grad():
        baseline = model.model.get_audio_features(input_features, input_mask, return_dict=True)
    checkpoints: list[dict[str, Any]] = []
    handles: list[Any] = []

    def capture(
        module: Any,
        operation_id: str,
        output: str,
        squeeze_batch: bool = False,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        tuple_index: int = 0,
    ) -> None:
        def hook(_module: Any, _inputs: Any, result: Any) -> None:
            tensor = result[tuple_index] if isinstance(result, tuple) else result
            if transform is not None:
                tensor = transform(tensor)
            checkpoints.append({"operationId": operation_id, "output": output, "tensor": tensor_payload(tensor, squeeze_batch)})
        handles.append(module.register_forward_hook(hook))

    def capture_input(module: Any, operation_id: str, output: str) -> None:
        def hook(_module: Any, inputs: Any) -> None:
            tensor = inputs[0]
            checkpoints.append({"operationId": operation_id, "output": output, "tensor": tensor_payload(tensor)})
        handles.append(module.register_forward_pre_hook(hook))

    audio = model.model.audio_tower

    def instrument_attention(module: Any, layer_index: int) -> None:
        def forward(
            self: Any,
            hidden_states: torch.Tensor,
            position_embeddings: torch.Tensor,
            attention_mask: torch.BoolTensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            batch_size, seq_length, _ = hidden_states.shape
            hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)
            query_states = self.q_proj(hidden_states).float().view(hidden_shape)
            key_states = self.k_proj(hidden_states).float().view(hidden_shape)
            value_states = self.v_proj(hidden_states).float().view(hidden_shape)
            query_states = query_states * self.q_scale * torch.nn.functional.softplus(self.per_dim_scale)
            key_states = key_states * self.k_scale
            checkpoints.append({
                "operationId": f"audio_layer_{layer_index}_q_scale",
                "output": f"audio_layer_{layer_index}_q_scaled",
                "tensor": tensor_payload(query_states.reshape(batch_size, seq_length, -1)),
            })
            checkpoints.append({
                "operationId": f"audio_layer_{layer_index}_k_scale",
                "output": f"audio_layer_{layer_index}_k_scaled",
                "tensor": tensor_payload(key_states.reshape(batch_size, seq_length, -1)),
            })
            query_states = self._convert_to_block(query_states)
            key_states = self._extract_block_context(key_states)
            value_states = self._extract_block_context(value_states)
            num_blocks = query_states.shape[1]
            relative_key_states = self.relative_k_proj(position_embeddings)
            relative_key_states = relative_key_states.view(-1, self.num_heads, self.head_dim)
            relative_key_states = relative_key_states.to(dtype=query_states.dtype)
            queries = query_states.permute(0, 3, 1, 2, 4)
            matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)
            queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
            matrix_bd_unshifted = queries_flat @ relative_key_states.permute(1, 2, 0)
            matrix_bd_unshifted = matrix_bd_unshifted.reshape(
                batch_size, self.num_heads, num_blocks, self.chunk_size, -1
            )
            matrix_bd = matrix_bd_unshifted
            matrix_bd = self._rel_shift(matrix_bd)
            attention_logits = matrix_ac + matrix_bd
            attention_softcapped = torch.tanh(attention_logits / self.softcap) * self.softcap
            attn_weights = attention_softcapped
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    attention_mask.logical_not(), self.config.attention_invalid_logits_value
                )
            for operation_id, output_name, tensor in (
                ("attention_content_scores", "attention_ac", matrix_ac),
                ("attention_position_scores", "attention_bd_unshifted", matrix_bd_unshifted),
                ("attention_relative_shift", "attention_bd", matrix_bd),
                ("attention_logit_add", "attention_logits", attention_logits),
                ("attention_softcap", "attention_softcapped", attention_softcapped),
            ):
                checkpoints.append({
                    "operationId": f"audio_layer_{layer_index}_{operation_id}",
                    "output": f"audio_layer_{layer_index}_{output_name}",
                    "tensor": tensor_payload(tensor),
                })
            checkpoints.append({
                "operationId": f"audio_layer_{layer_index}_attention_mask",
                "output": f"audio_layer_{layer_index}_attention_scores",
                "tensor": tensor_payload(attn_weights),
            })
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
            attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
            attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, num_blocks * self.chunk_size, -1)
            attn_output = attn_output[:, :seq_length].contiguous()
            checkpoints.append({
                "operationId": f"audio_layer_{layer_index}_attention",
                "output": f"audio_layer_{layer_index}_attention_context",
                "tensor": tensor_payload(attn_output),
            })
            attn_output = self.post(attn_output.to(dtype=self.post.linear.weight.dtype))
            return attn_output, attn_weights

        module.forward = types.MethodType(forward, module)
    channels_first = lambda tensor: tensor.permute(0, 3, 1, 2).contiguous()
    for index, subsampler in enumerate((audio.subsample_conv_projection.layer0, audio.subsample_conv_projection.layer1)):
        capture(subsampler.conv, f"audio_subsample_{index}_conv", f"audio_subsample_{index}_conv")
        capture(subsampler.norm, f"audio_subsample_{index}_norm", f"audio_subsample_{index}_norm", transform=channels_first)
        capture(subsampler.act, f"audio_subsample_{index}_relu", f"audio_subsample_{index}")
    capture(audio.subsample_conv_projection.input_proj_linear, "audio_input_projection", "audio_hidden_0")
    capture(audio.rel_pos_enc, "audio_relative_positions", "audio_relative_positions")
    def capture_feed_forward(module: Any, layer_index: int, index: int) -> None:
        stem = f"audio_layer_{layer_index}_ffn{index}"
        capture(module.pre_layer_norm, f"{stem}_pre_norm", f"{stem}_norm")
        capture(module.ffw_layer_1, f"{stem}_linear_1", f"{stem}_linear_1")
        capture(module.ffw_layer_2, f"{stem}_linear_2", f"{stem}_linear_2")
        capture(module.post_layer_norm, f"{stem}_post_norm", f"{stem}_normalized")
        capture(module, f"{stem}_residual", f"{stem}_output")

    for index, layer in enumerate(audio.layers):
        instrument_attention(layer.self_attn, index)
        capture_feed_forward(layer.feed_forward1, index, 1)
        capture(layer.norm_pre_attn, f"audio_layer_{index}_attn_pre_norm", f"audio_layer_{index}_attn_norm")
        capture(layer.self_attn.q_proj, f"audio_layer_{index}_self_attn.q_proj", f"audio_layer_{index}_q")
        capture(layer.self_attn.k_proj, f"audio_layer_{index}_self_attn.k_proj", f"audio_layer_{index}_k")
        capture(layer.self_attn.v_proj, f"audio_layer_{index}_self_attn.v_proj", f"audio_layer_{index}_v")
        capture(layer.self_attn.relative_k_proj, f"audio_layer_{index}_relative_k_projection", f"audio_layer_{index}_relative_keys")
        capture_input(layer.self_attn.post, f"audio_layer_{index}_attention_context_cast", f"audio_layer_{index}_attention_context_bf16")
        capture(layer.self_attn.post, f"audio_layer_{index}_self_attn.post", f"audio_layer_{index}_attention_projected")
        capture(layer.self_attn, f"audio_layer_{index}_attention_softmax", f"audio_layer_{index}_attention_weights", tuple_index=1)
        capture(layer.norm_post_attn, f"audio_layer_{index}_attn_post_norm", f"audio_layer_{index}_after_attention_norm")
        capture_input(layer.lconv1d.pre_layer_norm, f"audio_layer_{index}_attn_residual", f"audio_layer_{index}_after_attention")
        capture(layer.lconv1d.pre_layer_norm, f"audio_layer_{index}_conv_pre_norm", f"audio_layer_{index}_conv_norm_in")
        capture(layer.lconv1d.linear_start, f"audio_layer_{index}_lconv1d.linear_start", f"audio_layer_{index}_conv_glu_linear")
        capture(
            layer.lconv1d.depthwise_conv1d,
            f"audio_layer_{index}_conv_depthwise",
            f"audio_layer_{index}_conv_depthwise",
            transform=lambda tensor: tensor.transpose(1, 2).contiguous(),
        )
        capture(layer.lconv1d.conv_norm, f"audio_layer_{index}_conv_post_norm", f"audio_layer_{index}_conv_normalized")
        capture(layer.lconv1d.linear_end, f"audio_layer_{index}_lconv1d.linear_end", f"audio_layer_{index}_conv_projected")
        capture(layer.lconv1d, f"audio_layer_{index}_conv_residual", f"audio_layer_{index}_after_conv")
        capture_feed_forward(layer.feed_forward2, index, 2)
        capture(layer, f"audio_layer_{index}_out_norm", f"audio_hidden_{index + 1}")
    capture(audio.output_proj, "audio_output_projection", "audio_output_projected")
    capture(model.model.embed_audio.embedding_pre_projection_norm, "audio_language_projection_norm", "audio_output_normalized")
    capture(model.model.embed_audio.embedding_projection, "audio_language_projection", "audio_projected_features")

    with torch.no_grad():
        output = model.model.get_audio_features(input_features, input_mask, return_dict=True)
    if not torch.equal(output.pooler_output, baseline.pooler_output) or not torch.equal(output.attention_mask, baseline.attention_mask):
        raise ValueError("Instrumented Gemma 4 audio attention changed authoritative model output.")
    for handle in handles:
        handle.remove()
    stripped = output.pooler_output[output.attention_mask]
    checkpoints.append({"operationId": "audio_strip_padding", "output": "audio_features", "tensor": tensor_payload(stripped)})
    expected = 6 + 1 + 1 + 36 * len(audio.layers) + 3 + 1
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
