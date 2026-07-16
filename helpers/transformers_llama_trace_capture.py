#!/usr/bin/env python3
"""Native, version-pinned Transformers capture for the supported Llama path.

This is deliberately a reference-runtime adapter, not another IR executor.
It receives only the package path, tokens and positions; operation names are
the explicit stable mapping for Transformers' LlamaForCausalLM eager forward
implementation.  The helper first proves that its hooks leave logits and
canonical DynamicCache tensors unchanged, then returns those native values as
an integrity-bound trace consumed by the TypeScript candidate comparison.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.llama import modeling_llama


SUPPORTED_TRANSFORMERS = "4.57.1"
SUPPORTED_TORCH = "2.7.1"


def tensor_payload(value: torch.Tensor) -> dict[str, Any]:
    if value.dtype != torch.float32:
        raise ValueError(f"Capture Transformers exige tensor F32, recebeu {value.dtype}.")
    array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return {
        "dtype": "F32",
        "shape": list(array.shape),
        "valuesBase64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def canonical_cache(cache: DynamicCache, layers: int) -> list[dict[str, Any]]:
    if len(cache.layers) != layers:
        raise ValueError(f"DynamicCache contém {len(cache.layers)} camadas, esperado {layers}.")
    result: list[dict[str, Any]] = []
    for index, layer in enumerate(cache.layers):
        key, value = layer.keys, layer.values
        if key is None or value is None:
            raise ValueError(f"DynamicCache não contém K/V para camada {index}.")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(f"DynamicCache camada {index} não está no layout BHSD 4D.")
        result.append({"layer": index, "key": tensor_payload(key), "value": tensor_payload(value)})
    return result


def exact_tensor(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape) or not torch.equal(left, right):
        delta = float((left.float() - right.float()).abs().max().item()) if tuple(left.shape) == tuple(right.shape) else float("inf")
        raise ValueError(f"Instrumentação Transformers alterou {label}; maxAbsoluteError={delta}.")


def exact_cache(left: Any, right: Any, layers: int, label: str) -> None:
    if len(left.layers) != layers or len(right.layers) != layers:
        raise ValueError(f"Instrumentação Transformers alterou contagem de cache {label}.")
    for index, (left_layer, right_layer) in enumerate(zip(left.layers, right.layers, strict=True)):
        if left_layer.keys is None or left_layer.values is None or right_layer.keys is None or right_layer.values is None:
            raise ValueError(f"Cache {label} incompleto na camada {index}.")
        exact_tensor(left_layer.keys, right_layer.keys, f"{label} key layer {index}")
        exact_tensor(left_layer.values, right_layer.values, f"{label} value layer {index}")


def greedy(logits: torch.Tensor) -> int:
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1:
        raise ValueError(f"Logits Transformers inválidos para geração: {tuple(logits.shape)}.")
    return int(torch.argmax(logits[0, -1], dim=-1).item())


class LlamaCapture:
    def __init__(self, source: Path):
        if transformers.__version__ != SUPPORTED_TRANSFORMERS or torch.__version__.split("+")[0] != SUPPORTED_TORCH:
            raise ValueError(
                "Adapter Transformers Llama é version-pinned e requer "
                f"transformers=={SUPPORTED_TRANSFORMERS} e torch=={SUPPORTED_TORCH}; "
                f"recebeu transformers=={transformers.__version__}, torch=={torch.__version__}."
            )
        self.source = source
        self.model = AutoModelForCausalLM.from_pretrained(
            source, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"
        )
        self.model.eval()
        self._validate_model()
        self.layers = self.model.config.num_hidden_layers
        self.operations: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        self.original_rope: Callable[..., Any] | None = None
        self.original_attention: Callable[..., Any] | None = None

    @property
    def runtime(self) -> str:
        return (
            f"PyTorch {torch.__version__.split('+')[0]} / Transformers {transformers.__version__} "
            "LlamaForCausalLM eager native capture"
        )

    def _validate_model(self) -> None:
        config = self.model.config
        if config.model_type != "llama" or self.model.__class__.__name__ != "LlamaForCausalLM":
            raise ValueError(
                "Transformers capture requer exatamente LlamaForCausalLM com model_type=llama; "
                f"recebeu {self.model.__class__.__name__}/{config.model_type}."
            )
        if getattr(config, "hidden_act", None) != "silu":
            raise ValueError(f"Transformers Llama capture requer hidden_act=silu; recebeu {getattr(config, 'hidden_act', None)}.")
        if getattr(config, "attention_bias", False) or getattr(config, "mlp_bias", False):
            raise ValueError("Transformers Llama capture não aceita projeções com bias; implemente o contrato explícito antes de capturar.")
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling not in (None, {}):
            raise ValueError("Transformers Llama capture requer RoPE default sem rope_scaling.")
        if getattr(config, "_attn_implementation", None) != "eager":
            raise ValueError(f"Transformers Llama capture requer attention eager; recebeu {getattr(config, '_attn_implementation', None)}.")
        for name, parameter in self.model.named_parameters():
            if parameter.dtype != torch.float32:
                raise ValueError(f"{name}: Transformers Llama capture requer parâmetro F32; recebeu {parameter.dtype}.")

    def _forward(self, tokens: list[int], positions: list[int], cache: DynamicCache | None = None):
        if len(tokens) != len(positions) or not tokens:
            raise ValueError("Tokens e posições Transformers devem ser vetores não vazios de mesmo comprimento.")
        input_ids = torch.tensor([tokens], dtype=torch.long)
        position_ids = torch.tensor([positions], dtype=torch.long)
        with torch.inference_mode():
            return self.model(input_ids=input_ids, position_ids=position_ids, past_key_values=cache, use_cache=True)

    def _record(self, operation_id: str, tensor: torch.Tensor) -> None:
        if operation_id in self.operations:
            raise ValueError(f"Capture Transformers registrou operação duplicada: {operation_id}.")
        self.operations[operation_id] = tensor.detach()

    def _hook_output(self, operation_id: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            self._record(operation_id, output[0] if isinstance(output, tuple) else output)
        return hook

    def _hook_input(self, operation_id: str):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            self._record(operation_id, inputs[0])
        return hook

    def _install(self) -> None:
        if self.handles or self.original_rope is not None:
            raise RuntimeError("Hooks Transformers Llama já instalados.")
        self.operations = {}
        self.handles.append(self.model.model.embed_tokens.register_forward_hook(self._hook_output("token_embedding")))
        for index, layer in enumerate(self.model.model.layers):
            prefix = f"layer_{index}"
            self.handles.extend([
                layer.input_layernorm.register_forward_hook(self._hook_output(f"{prefix}_input_norm")),
                layer.self_attn.q_proj.register_forward_hook(self._hook_output(f"{prefix}_q_proj")),
                layer.self_attn.k_proj.register_forward_hook(self._hook_output(f"{prefix}_k_proj")),
                layer.self_attn.v_proj.register_forward_hook(self._hook_output(f"{prefix}_v_proj")),
                layer.self_attn.v_proj.register_forward_hook(self._v_heads_hook(index)),
                layer.self_attn.o_proj.register_forward_hook(self._hook_output(f"{prefix}_o_proj")),
                # Its input is the native residual+attention addition, before
                # the norm consumes it, so this observes the IR boundary.
                layer.post_attention_layernorm.register_forward_pre_hook(self._hook_input(f"{prefix}_attention_residual")),
                layer.post_attention_layernorm.register_forward_hook(self._hook_output(f"{prefix}_pre_ffn_norm")),
                layer.mlp.gate_proj.register_forward_hook(self._hook_output(f"{prefix}_gate_proj")),
                layer.mlp.up_proj.register_forward_hook(self._hook_output(f"{prefix}_up_proj")),
                layer.mlp.act_fn.register_forward_hook(self._hook_output(f"{prefix}_activation")),
                # Its input is exactly activation(gate) * up in native code.
                layer.mlp.down_proj.register_forward_pre_hook(self._hook_input(f"{prefix}_gated_multiply")),
                layer.mlp.down_proj.register_forward_hook(self._hook_output(f"{prefix}_down_proj")),
                layer.register_forward_hook(self._hook_output(f"{prefix}_mlp_residual")),
            ])
        self.handles.append(self.model.model.norm.register_forward_hook(self._hook_output("final_norm")))
        self.handles.append(self.model.lm_head.register_forward_hook(self._hook_output("lm_head")))

        self.original_rope = modeling_llama.apply_rotary_pos_emb
        self.original_attention = modeling_llama.eager_attention_forward

        def rope(q: torch.Tensor, k: torch.Tensor, *args: Any, **kwargs: Any):
            index = self._next_rope_layer()
            self._record(f"layer_{index}_q_heads", q)
            self._record(f"layer_{index}_k_heads", k)
            result_q, result_k = self.original_rope(q, k, *args, **kwargs)
            self._record(f"layer_{index}_q_rope", result_q)
            self._record(f"layer_{index}_k_rope", result_k)
            return result_q, result_k

        def attention(module: Any, *args: Any, **kwargs: Any):
            result, weights = self.original_attention(module, *args, **kwargs)
            index = int(module.layer_idx)
            # Transformers returns BSHD here; the candidate attention op's
            # declared result is the subsequent native contiguous flatten.
            self._record(f"layer_{index}_attention", result.reshape(result.shape[0], result.shape[1], -1))
            return result, weights

        modeling_llama.apply_rotary_pos_emb = rope
        modeling_llama.eager_attention_forward = attention

    def _v_heads_hook(self, layer: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            batch, sequence, _ = output.shape
            heads = self.model.config.num_key_value_heads
            dim = self.model.config.head_dim
            self._record(f"layer_{layer}_v_heads", output.view(batch, sequence, heads, dim).transpose(1, 2))
        return hook

    def _next_rope_layer(self) -> int:
        for index in range(self.layers):
            if f"layer_{index}_q_heads" not in self.operations:
                return index
        raise ValueError("Capture Transformers observou mais aplicações RoPE do que camadas Llama declaradas.")

    def _remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.original_rope is not None:
            modeling_llama.apply_rotary_pos_emb = self.original_rope
            self.original_rope = None
        if self.original_attention is not None:
            modeling_llama.eager_attention_forward = self.original_attention
            self.original_attention = None

    def _captured_forward(self, tokens: list[int], positions: list[int], cache: DynamicCache | None = None):
        self._install()
        try:
            output = self._forward(tokens, positions, cache)
            expected = self.expected_operation_ids()
            missing = [operation for operation in expected if operation not in self.operations]
            unexpected = sorted(set(self.operations).difference(expected))
            if missing or unexpected:
                raise ValueError(f"Hooks Transformers não cobriram exatamente as operações Llama: ausentes={missing}, extras={unexpected}.")
            return output, self.operations
        finally:
            self._remove()

    def expected_operation_ids(self) -> list[str]:
        result = ["token_embedding"]
        for index in range(self.layers):
            prefix = f"layer_{index}"
            result.extend([
                f"{prefix}_input_norm", f"{prefix}_q_proj", f"{prefix}_k_proj", f"{prefix}_v_proj",
                f"{prefix}_q_heads", f"{prefix}_k_heads", f"{prefix}_v_heads", f"{prefix}_q_rope", f"{prefix}_k_rope",
                f"{prefix}_attention", f"{prefix}_o_proj", f"{prefix}_attention_residual", f"{prefix}_pre_ffn_norm",
                f"{prefix}_gate_proj", f"{prefix}_up_proj", f"{prefix}_activation", f"{prefix}_gated_multiply",
                f"{prefix}_down_proj", f"{prefix}_mlp_residual",
            ])
        return [*result, "final_norm", "lm_head"]

    def execution(self, tokens: list[int], positions: list[int]) -> dict[str, Any]:
        plain = self._forward(tokens, positions)
        captured, operations = self._captured_forward(tokens, positions)
        exact_tensor(plain.logits, captured.logits, "execution logits")
        exact_cache(plain.past_key_values, captured.past_key_values, self.layers, "execution")
        return {
            "runtime": self.runtime,
            "operations": [
                {"operationId": operation_id, "output": operation_output(operation_id), "tensor": tensor_payload(operations[operation_id])}
                for operation_id in self.expected_operation_ids()
            ],
            "pastKeyValues": canonical_cache(captured.past_key_values, self.layers),
        }

    def generation(self, tokens: list[int], positions: list[int], limit: int) -> dict[str, Any]:
        plain = self._generate(tokens, positions, limit, capture=False)
        captured = self._generate(tokens, positions, limit, capture=True)
        if plain["generatedTokenIds"] != captured["generatedTokenIds"] or plain["steps"] != captured["steps"]:
            raise ValueError("Instrumentação Transformers alterou tokens ou posições da geração.")
        for index, (left, right) in enumerate(zip(plain["selection"], captured["selection"], strict=True)):
            exact_tensor(left, right, f"generation selection logits {index}")
        exact_tensor(plain["logits"], captured["logits"], "generation terminal logits")
        for index, (left, right) in enumerate(zip(plain["snapshots"], captured["snapshots"], strict=True)):
            exact_cache(left, right, self.layers, f"generation snapshot {index}")
        exact_cache(plain["cache"], captured["cache"], self.layers, "generation terminal cache")
        return {
            "runtime": self.runtime,
            "generatedTokenIds": captured["generatedTokenIds"],
            "steps": captured["steps"],
            "selectionLogits": [tensor_payload(value) for value in captured["selection"]],
            "stepPastKeyValues": [canonical_cache(cache, self.layers) for cache in captured["snapshots"]],
            "logits": tensor_payload(captured["logits"]),
            "pastKeyValues": canonical_cache(captured["cache"], self.layers),
        }

    def _generate(self, tokens: list[int], positions: list[int], limit: int, capture: bool) -> dict[str, Any]:
        output = self._captured_forward(tokens, positions)[0] if capture else self._forward(tokens, positions)
        cache = output.past_key_values
        logits = output.logits
        generated: list[int] = []
        steps: list[dict[str, int]] = []
        selection: list[torch.Tensor] = []
        snapshots: list[DynamicCache] = []
        for index in range(limit):
            token = greedy(logits)
            generated.append(token)
            position = positions[-1] + 1 + index
            steps.append({"tokenId": token, "positionId": position})
            selection.append(logits.detach())
            output = self._captured_forward([token], [position], cache)[0] if capture else self._forward([token], [position], cache)
            cache = output.past_key_values
            snapshots.append(clone_cache(cache, self.layers))
            logits = output.logits
        return {"generatedTokenIds": generated, "steps": steps, "selection": selection, "snapshots": snapshots, "logits": logits.detach(), "cache": cache}


class CacheSnapshot:
    def __init__(self) -> None:
        self.layers: list[Any] = []


def clone_cache(cache: DynamicCache, layers: int) -> CacheSnapshot:
    # DynamicCache constructors differ across Transformers releases; retaining
    # the native tensors in this narrow immutable snapshot avoids a private
    # cache-copy API while preserving exactly the canonical cache fields.
    result = CacheSnapshot()
    for layer in cache.layers[:layers]:
        copy = type(layer)()
        copy.keys = layer.keys.detach().clone()
        copy.values = layer.values.detach().clone()
        result.layers.append(copy)
    return result


def operation_output(operation_id: str) -> str:
    if operation_id == "token_embedding": return "hidden_states_0"
    if operation_id == "final_norm": return "final_hidden_states"
    if operation_id == "lm_head": return "logits"
    parts = operation_id.split("_")
    layer = parts[1]
    suffix = "_".join(parts[2:])
    return {
        "input_norm": f"layer_{layer}_attn_norm", "q_proj": f"layer_{layer}_q_linear", "k_proj": f"layer_{layer}_k_linear",
        "v_proj": f"layer_{layer}_v_linear", "q_heads": f"layer_{layer}_q_heads", "k_heads": f"layer_{layer}_k_heads",
        "v_heads": f"layer_{layer}_v_heads", "q_rope": f"layer_{layer}_q_rot", "k_rope": f"layer_{layer}_k_rot",
        "attention": f"layer_{layer}_attention_context", "o_proj": f"layer_{layer}_attention_projected",
        "attention_residual": f"layer_{layer}_after_attention", "pre_ffn_norm": f"layer_{layer}_ffn_norm",
        "gate_proj": f"layer_{layer}_gate", "up_proj": f"layer_{layer}_up", "activation": f"layer_{layer}_gate_activated",
        "gated_multiply": f"layer_{layer}_gated_mlp", "down_proj": f"layer_{layer}_mlp_output", "mlp_residual": f"hidden_states_{int(layer) + 1}",
    }[suffix]


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text("utf8"))
    if request.get("kind") not in {"execution", "generation"}:
        raise ValueError("Capture Transformers requer kind execution ou generation.")
    source = Path(request["source"])
    if not source.is_dir():
        raise ValueError("Capture Transformers requer diretório de pacote local.")
    tokens, positions = request["inputTokens"], request["positionIds"]
    if not isinstance(tokens, list) or not isinstance(positions, list) or any(not isinstance(value, int) or value < 0 for value in [*tokens, *positions]):
        raise ValueError("Capture Transformers requer tokens e posições inteiros não negativos.")
    capture = LlamaCapture(source)
    result = capture.execution(tokens, positions) if request["kind"] == "execution" else capture.generation(tokens, positions, request["maxNewTokens"])
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
