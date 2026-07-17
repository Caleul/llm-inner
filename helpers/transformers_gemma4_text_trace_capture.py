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
import types
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForImageTextToText
from transformers.models.gemma4 import modeling_gemma4


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
    """Capture every declared Gemma4Text assignment from the native forward.

    This is instrumentation of the pinned Transformers 5.5.0 eager-BF16
    implementation, not an independent candidate executor.  The wrappers use
    the exact registered modules and native tensor operations, record the
    adapter-owned assignment IDs at their execution boundaries, and are
    checked against an uninstrumented forward before results are emitted.
    """
    text = model.model.language_model
    checkpoints: dict[str, dict[str, Any]] = {}

    def save(operation_id: str, output: str, value: torch.Tensor) -> torch.Tensor:
        if operation_id in checkpoints:
            raise ValueError(f"Gemma 4 native operation trace captured duplicate operation {operation_id}.")
        checkpoints[operation_id] = {"operationId": operation_id, "output": output, "tensor": tensor_payload(value)}
        return value

    def equal_cache(left: Any, right: Any) -> bool:
        if not hasattr(left, "layers") or not hasattr(right, "layers") or len(left.layers) != len(right.layers):
            return False
        for left_layer, right_layer in zip(left.layers, right.layers):
            if not torch.equal(left_layer.keys, right_layer.keys) or not torch.equal(left_layer.values, right_layer.values):
                return False
        return True

    # First prove that the unmodified authoritative path is stable for this
    # exact request.  The instrumented pass below is compared to this result
    # before any trace is accepted.
    baseline = forward(model, tokens, positions)

    original_project = text.project_per_layer_inputs
    original_layers = [(layer.forward, layer.self_attn.forward, layer.mlp.forward) for layer in text.layers]
    try:
        def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs=None):
            projection = self.per_layer_model_projection(inputs_embeds)
            save("ple_context_projection", "ple_context_packed", projection)
            projection = projection * self.per_layer_model_projection_scale
            save("ple_context_scale", "ple_context_scaled", projection)
            projection = projection.reshape(*inputs_embeds.shape[:-1], self.config.num_hidden_layers, self.hidden_size_per_layer_input)
            save("ple_context_reshape", "ple_context_reshaped", projection)
            projection = self.per_layer_projection_norm(projection)
            save("ple_context_norm", "ple_context_normalized", projection)
            if per_layer_inputs is None:
                return projection
            combined = projection + per_layer_inputs
            save("ple_combine", "ple_combined", combined)
            result = combined * self.per_layer_input_scale
            return save("ple_combine_scale", "ple_inputs", result)

        text.project_per_layer_inputs = types.MethodType(project_per_layer_inputs, text)

        def make_mlp_forward(layer_index: int):
            def mlp_forward(self, x):
                prefix = f"layer_{layer_index}"
                gate = self.gate_proj(x)
                save(f"{prefix}_gate_proj", f"{prefix}_gate", gate)
                up = self.up_proj(x)
                save(f"{prefix}_up_proj", f"{prefix}_up", up)
                activated = self.act_fn(gate)
                save(f"{prefix}_activation", f"{prefix}_gate_activated", activated)
                gated = activated * up
                save(f"{prefix}_gated_mlp", f"{prefix}_gated_mlp", gated)
                output = self.down_proj(gated)
                return save(f"{prefix}_down_proj", f"{prefix}_mlp_output", output)
            return mlp_forward

        def make_attention_forward(layer_index: int):
            def attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
                prefix = f"layer_{layer_index}"
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)
                cos, sin = position_embeddings
                query_states = self.q_proj(hidden_states).view(hidden_shape)
                save(f"{prefix}_q_proj", f"{prefix}_q_linear", query_states.reshape(*input_shape, -1))
                save(f"{prefix}_q_heads", f"{prefix}_q_heads", query_states.transpose(1, 2))
                query_states = self.q_norm(query_states)
                save(f"{prefix}_q_norm", f"{prefix}_q_normalized", query_states.transpose(1, 2))
                query_states = modeling_gemma4.apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
                query_states = query_states.transpose(1, 2)
                save(f"{prefix}_q_rope", f"{prefix}_q_rot", query_states)
                if self.is_kv_shared_layer and past_key_values is not None:
                    key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
                    key_states = key_states.to(query_states.device)
                    value_states = value_states.to(query_states.device)
                else:
                    key_states = self.k_proj(hidden_states).view(hidden_shape)
                    save(f"{prefix}_k_proj", f"{prefix}_k_linear", key_states.reshape(*input_shape, -1))
                    save(f"{prefix}_k_heads", f"{prefix}_k_heads", key_states.transpose(1, 2))
                    value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
                    if self.v_proj is not None:
                        save(f"{prefix}_v_proj", f"{prefix}_v_linear", value_states.reshape(*input_shape, -1))
                        save(f"{prefix}_v_heads", f"{prefix}_v_heads", value_states.transpose(1, 2))
                    key_states = self.k_norm(key_states)
                    save(f"{prefix}_k_norm", f"{prefix}_k_normalized", key_states.transpose(1, 2))
                    key_states = modeling_gemma4.apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
                    key_states = key_states.transpose(1, 2)
                    save(f"{prefix}_k_rope", f"{prefix}_k_rot", key_states)
                    value_states = self.v_norm(value_states)
                    value_states = value_states.transpose(1, 2)
                    save(f"{prefix}_v_norm", f"{prefix}_v_normalized", value_states)
                if past_key_values is not None:
                    if not self.is_kv_shared_layer:
                        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
                    if self.store_full_length_kv:
                        if not hasattr(past_key_values, "shared_layers"):
                            past_key_values.shared_layers = {}
                        past_key_values.shared_layers[self.layer_idx] = key_states, value_states
                attention_interface = modeling_gemma4.eager_attention_forward
                if self.config._attn_implementation != "eager":
                    attention_interface = modeling_gemma4.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
                output, weights = attention_interface(
                    self, query_states, key_states, value_states, attention_mask,
                    dropout=self.attention_dropout if self.training else 0.0,
                    scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
                )
                output = output.reshape(*input_shape, -1).contiguous()
                save(f"{prefix}_attention", f"{prefix}_attention_context", output)
                output = self.o_proj(output)
                return save(f"{prefix}_o_proj", f"{prefix}_attention_projected", output), weights
            return attention_forward

        def make_layer_forward(layer_index: int):
            def layer_forward(self, hidden_states, per_layer_input=None, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
                prefix = f"layer_{layer_index}"
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                save(f"{prefix}_input_norm", f"{prefix}_attn_norm", hidden_states)
                hidden_states, _ = self.self_attn(hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)
                hidden_states = self.post_attention_layernorm(hidden_states)
                save(f"{prefix}_post_attention_norm", f"{prefix}_post_attention_normalized", hidden_states)
                hidden_states = residual + hidden_states
                save(f"{prefix}_attention_residual", f"{prefix}_after_attention", hidden_states)
                residual = hidden_states
                hidden_states = self.pre_feedforward_layernorm(hidden_states)
                save(f"{prefix}_pre_ffn_norm", f"{prefix}_ffn_norm", hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = self.post_feedforward_layernorm(hidden_states)
                save(f"{prefix}_post_ffn_norm", f"{prefix}_post_ffn_normalized", hidden_states)
                hidden_states = residual + hidden_states
                save(f"{prefix}_mlp_residual", f"{prefix}_after_mlp", hidden_states)
                if self.hidden_size_per_layer_input:
                    if per_layer_input is None:
                        raise ValueError("Gemma 4 dense PLE trace requires per_layer_input.")
                    save(f"{prefix}_ple_select", f"{prefix}_ple_input", per_layer_input)
                    residual = hidden_states
                    hidden_states = self.per_layer_input_gate(hidden_states)
                    save(f"{prefix}_ple_gate", f"{prefix}_ple_gate_linear", hidden_states)
                    hidden_states = self.act_fn(hidden_states)
                    save(f"{prefix}_ple_activation", f"{prefix}_ple_gate_activated", hidden_states)
                    hidden_states = hidden_states * per_layer_input
                    save(f"{prefix}_ple_gated_multiply", f"{prefix}_ple_gated", hidden_states)
                    hidden_states = self.per_layer_projection(hidden_states)
                    save(f"{prefix}_ple_project", f"{prefix}_ple_projected", hidden_states)
                    hidden_states = self.post_per_layer_input_norm(hidden_states)
                    save(f"{prefix}_post_ple_norm", f"{prefix}_post_ple_normalized", hidden_states)
                    hidden_states = residual + hidden_states
                    save(f"{prefix}_ple_residual", f"{prefix}_before_scalar", hidden_states)
                hidden_states *= self.layer_scalar
                return save(f"{prefix}_scalar", f"hidden_states_{layer_index + 1}", hidden_states)
            return layer_forward

        for layer_index, layer in enumerate(text.layers):
            layer.mlp.forward = types.MethodType(make_mlp_forward(layer_index), layer.mlp)
            layer.self_attn.forward = types.MethodType(make_attention_forward(layer_index), layer.self_attn)
            layer.forward = types.MethodType(make_layer_forward(layer_index), layer)

        embedding_hook = text.embed_tokens.register_forward_hook(lambda _module, _inputs, value: save("token_embedding", "hidden_states_0", value))
        ple_embedding_hook = text.embed_tokens_per_layer.register_forward_hook(lambda _module, _inputs, value: save("ple_token_identity", "ple_token_identity", value.reshape(*value.shape[:2], text.config.num_hidden_layers, text.config.hidden_size_per_layer_input))) if getattr(text, "hidden_size_per_layer_input", None) else None
        final_norm_hook = text.norm.register_forward_hook(lambda _module, _inputs, value: save("final_norm", "final_hidden_states", value))
        lm_head_hook = model.lm_head.register_forward_hook(lambda _module, _inputs, value: save("lm_head", "logits", value))
        native = forward(model, tokens, positions)
        if model.config.text_config.final_logit_softcapping is not None:
            save("final_logit_softcap", "softcapped_logits", native.logits)
        if not torch.equal(baseline.logits, native.logits) or not equal_cache(baseline.past_key_values, native.past_key_values):
            raise ValueError("Gemma 4 native operation instrumentation changed the authoritative forward result or KV cache.")
        return {
            "runtime": f"PyTorch {torch.__version__.split('+')[0]} / Transformers {transformers.__version__} Gemma4ForConditionalGeneration eager BF16 full assignment trace",
            "operations": list(checkpoints.values()),
            "pastKeyValues": cache_payload(native.past_key_values),
        }
    finally:
        text.project_per_layer_inputs = original_project
        for layer, originals in zip(text.layers, original_layers):
            layer.forward, layer.self_attn.forward, layer.mlp.forward = originals
        for hook in (locals().get("embedding_hook"), locals().get("ple_embedding_hook"), locals().get("final_norm_hook"), locals().get("lm_head_hook")):
            if hook is not None:
                hook.remove()


def linear_reduction_checkpoint(model: Any, tokens: list[int], positions: list[int], request: dict[str, Any]) -> dict[str, Any]:
    """Capture a single registered dense MLP projection without graph rewriting.

    A forward pre-hook observes the exact input passed to the native Linear
    module and a forward hook observes its result.  Both hooks return None, so
    the second forward remains bitwise-identical to the uninstrumented eager
    BF16 reference before the small evidence bundle is accepted.
    """
    layer_index = request.get("layerIndex")
    projection = request.get("projection")
    operation_id = request.get("operationId")
    producer_operation_id = request.get("producerOperationId")
    producer_output = request.get("producerOutput")
    output = request.get("output")
    if not isinstance(layer_index, int) or layer_index < 0 or layer_index >= len(model.model.language_model.layers):
        raise ValueError("Gemma 4 bounded linear capture requires a valid layerIndex.")
    if projection not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError("Gemma 4 bounded linear capture requires gate_proj, up_proj, or down_proj.")
    if not all(isinstance(value, str) and value for value in (operation_id, producer_operation_id, producer_output, output)):
        raise ValueError("Gemma 4 bounded linear capture requires declared operation and output IDs.")
    expected_operation = f"layer_{layer_index}_{projection}"
    if operation_id != expected_operation:
        raise ValueError(f"Gemma 4 bounded linear capture received {operation_id}; expected {expected_operation}.")
    expected_producer = f"layer_{layer_index}_pre_ffn_norm" if projection in ("gate_proj", "up_proj") else f"layer_{layer_index}_gated_mlp"
    expected_producer_output = f"layer_{layer_index}_ffn_norm" if projection in ("gate_proj", "up_proj") else f"layer_{layer_index}_gated_mlp"
    expected_output = f"layer_{layer_index}_{projection.removesuffix('_proj')}" if projection != "down_proj" else f"layer_{layer_index}_mlp_output"
    if producer_operation_id != expected_producer or producer_output != expected_producer_output or output != expected_output:
        raise ValueError("Gemma 4 bounded linear capture IDs do not match the registered Gemma4Text MLP contract.")
    target = getattr(model.model.language_model.layers[layer_index].mlp, projection)
    captured: dict[str, torch.Tensor] = {}

    def before(_module, inputs):
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise ValueError("Gemma 4 bounded linear capture received invalid native Linear input.")
        captured["input"] = inputs[0]

    def after(_module, _inputs, value):
        if not isinstance(value, torch.Tensor):
            raise ValueError("Gemma 4 bounded linear capture received invalid native Linear output.")
        captured["output"] = value

    baseline = forward(model, tokens, positions)
    before_hook = target.register_forward_pre_hook(before)
    after_hook = target.register_forward_hook(after)
    try:
        native = forward(model, tokens, positions)
    finally:
        before_hook.remove()
        after_hook.remove()
    if "input" not in captured or "output" not in captured:
        raise ValueError("Gemma 4 bounded linear capture did not observe its registered projection.")
    if not torch.equal(baseline.logits, native.logits):
        raise ValueError("Gemma 4 bounded linear hooks changed authoritative logits.")
    if not hasattr(baseline.past_key_values, "layers") or not hasattr(native.past_key_values, "layers"):
        raise ValueError("Gemma 4 bounded linear capture requires DynamicCache layers.")
    for left, right in zip(baseline.past_key_values.layers, native.past_key_values.layers):
        if not torch.equal(left.keys, right.keys) or not torch.equal(left.values, right.values):
            raise ValueError("Gemma 4 bounded linear hooks changed authoritative KV cache.")
    return {
        "runtime": f"PyTorch {torch.__version__.split('+')[0]} / Transformers {transformers.__version__} Gemma4ForConditionalGeneration eager BF16 bounded MLP projection trace",
        "operations": [
            {"operationId": producer_operation_id, "output": producer_output, "tensor": tensor_payload(captured["input"])},
            {"operationId": operation_id, "output": output, "tensor": tensor_payload(captured["output"])},
        ],
        "pastKeyValues": cache_payload(native.past_key_values),
    }


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
    if request.get("mode") == "linear-reduction-checkpoint":
        return linear_reduction_checkpoint(model, tokens, positions, request)
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
