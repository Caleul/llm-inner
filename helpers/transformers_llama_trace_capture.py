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
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.gemma2 import modeling_gemma2


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


class TransformersDecoderCapture:
    """Pinned native capture with explicit Llama, Qwen 2, and Gemma 2 contracts.

    The filename is retained for backwards-compatible runner paths; adapter
    selection is explicit request data and every contract below is validated
    before a model forward or hook installation can happen.
    """

    def __init__(self, source: Path, adapter: str):
        if transformers.__version__ != SUPPORTED_TRANSFORMERS or torch.__version__.split("+")[0] != SUPPORTED_TORCH:
            raise ValueError(
                "Adapter Transformers Llama é version-pinned e requer "
                f"transformers=={SUPPORTED_TRANSFORMERS} e torch=={SUPPORTED_TORCH}; "
                f"recebeu transformers=={transformers.__version__}, torch=={torch.__version__}."
            )
        self.source = source
        self.adapter = adapter
        contracts = {
            "llama": {"model_type": "llama", "model_class": "LlamaForCausalLM", "label": "Llama", "module": modeling_llama, "attention_bias": False},
            "qwen2": {"model_type": "qwen2", "model_class": "Qwen2ForCausalLM", "label": "Qwen 2", "module": modeling_qwen2, "attention_bias": None},
            "gemma2": {"model_type": "gemma2", "model_class": "Gemma2ForCausalLM", "label": "Gemma 2", "module": modeling_gemma2, "attention_bias": False},
        }
        if adapter not in contracts:
            raise ValueError(f"Adapter Transformers desconhecido: {adapter}.")
        self.contract = contracts[adapter]
        self.modeling = self.contract["module"]
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
            f"{self.contract['model_class']} eager native capture"
        )

    def _validate_model(self) -> None:
        config = self.model.config
        if config.model_type != self.contract["model_type"] or self.model.__class__.__name__ != self.contract["model_class"]:
            raise ValueError(
                f"Transformers capture requer exatamente {self.contract['model_class']} com model_type={self.contract['model_type']}; "
                f"recebeu {self.model.__class__.__name__}/{config.model_type}."
            )
        expected_activation = "gelu_pytorch_tanh" if self.adapter == "gemma2" else "silu"
        actual_activation = getattr(config, "hidden_activation", getattr(config, "hidden_act", None))
        if actual_activation != expected_activation:
            raise ValueError(
                f"Transformers {self.contract['label']} capture requer ativação {expected_activation}; recebeu {actual_activation}."
            )
        configured_attention_bias = bool(getattr(config, "attention_bias", False))
        required_attention_bias = self.contract["attention_bias"]
        if (required_attention_bias is not None and configured_attention_bias is not required_attention_bias) or getattr(config, "mlp_bias", False):
            raise ValueError(
                f"Transformers {self.contract['label']} capture requer attention_bias={required_attention_bias} "
                "e mlp_bias=false; implemente o contrato explícito antes de capturar."
            )
        attention = self.model.model.layers[0].self_attn
        expected_qkv_bias = True if self.adapter == "qwen2" else configured_attention_bias
        for projection in ("q_proj", "k_proj", "v_proj"):
            if (getattr(attention, projection).bias is not None) is not expected_qkv_bias:
                raise ValueError(
                    f"Transformers {self.contract['label']} capture encontrou {projection}.bias incompatível com o contrato esperado={expected_qkv_bias}."
                )
        if attention.o_proj.bias is not None:
            raise ValueError(f"Transformers {self.contract['label']} capture requer o_proj sem bias no contrato declarado.")
        if self.adapter == "qwen2" and any(name.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")) for name, _ in self.model.named_parameters()):
            raise ValueError("Transformers Qwen 2 capture rejeita Q/K head norms: elas pertencem a um contrato arquitetural diferente.")
        if self.adapter == "gemma2":
            if any(name.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")) for name, _ in self.model.named_parameters()):
                raise ValueError("Transformers Gemma 2 4.57.1 capture rejeita Q/K head norms: seu Gemma2Attention eager não declara esses módulos.")
            if not all(hasattr(layer, name) for layer in self.model.model.layers for name in (
                "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
            )):
                raise ValueError("Transformers Gemma 2 capture requer as quatro RMSNorms declaradas por camada.")
            if getattr(config, "query_pre_attn_scalar", None) is None or getattr(config, "attn_logit_softcapping", None) is None or getattr(config, "final_logit_softcapping", None) is None:
                raise ValueError("Transformers Gemma 2 capture requer query_pre_attn_scalar e ambos os softcaps declarados.")
            if not isinstance(getattr(config, "layer_types", None), list) or len(config.layer_types) != config.num_hidden_layers:
                raise ValueError("Transformers Gemma 2 capture requer layer_types explícito para selecionar atenção local/global.")
            if config.tie_word_embeddings and self.model.lm_head.weight is not self.model.model.embed_tokens.weight:
                raise ValueError("Transformers Gemma 2 capture encontrou tie_word_embeddings=true sem lm_head amarrado ao embedding.")
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling not in (None, {}):
            raise ValueError(f"Transformers {self.contract['label']} capture requer RoPE default sem rope_scaling.")
        if getattr(config, "_attn_implementation", None) != "eager":
            raise ValueError(f"Transformers {self.contract['label']} capture requer attention eager; recebeu {getattr(config, '_attn_implementation', None)}.")
        for name, parameter in self.model.named_parameters():
            if parameter.dtype != torch.float32:
                raise ValueError(f"{name}: Transformers {self.contract['label']} capture requer parâmetro F32; recebeu {parameter.dtype}.")

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
        if self.adapter == "gemma2":
            # Gemma2Model multiplies embeddings by sqrt(hidden_size) after
            # embed_tokens. The IR boundary is post-scale, observed by the
            # first layer norm's input rather than embed_tokens' output.
            self.handles.append(self.model.model.layers[0].input_layernorm.register_forward_pre_hook(self._hook_input("token_embedding")))
        else:
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
                *([] if self.adapter == "gemma2" else [
                    # In Llama/Qwen2 this norm consumes the native
                    # residual+attention addition and is the IR pre-FFN norm.
                    layer.post_attention_layernorm.register_forward_pre_hook(self._hook_input(f"{prefix}_attention_residual")),
                    layer.post_attention_layernorm.register_forward_hook(self._hook_output(f"{prefix}_pre_ffn_norm")),
                ]),
                *([
                    layer.post_attention_layernorm.register_forward_hook(self._hook_output(f"{prefix}_post_attention_norm")),
                    # Gemma 2 applies the residual after post-attention norm;
                    # pre_feedforward_layernorm observes that exact IR add.
                    layer.pre_feedforward_layernorm.register_forward_pre_hook(self._hook_input(f"{prefix}_attention_residual")),
                    layer.pre_feedforward_layernorm.register_forward_hook(self._hook_output(f"{prefix}_pre_ffn_norm")),
                ] if self.adapter == "gemma2" else []),
                layer.mlp.gate_proj.register_forward_hook(self._hook_output(f"{prefix}_gate_proj")),
                layer.mlp.up_proj.register_forward_hook(self._hook_output(f"{prefix}_up_proj")),
                layer.mlp.act_fn.register_forward_hook(self._hook_output(f"{prefix}_activation")),
                # Its input is exactly activation(gate) * up in native code.
                layer.mlp.down_proj.register_forward_pre_hook(self._hook_input(f"{prefix}_gated_multiply")),
                layer.mlp.down_proj.register_forward_hook(self._hook_output(f"{prefix}_down_proj")),
                *([
                    layer.post_feedforward_layernorm.register_forward_hook(self._hook_output(f"{prefix}_post_ffn_norm")),
                ] if self.adapter == "gemma2" else []),
                layer.register_forward_hook(self._hook_output(f"{prefix}_mlp_residual")),
            ])
        self.handles.append(self.model.model.norm.register_forward_hook(self._hook_output("final_norm")))
        self.handles.append(self.model.lm_head.register_forward_hook(self._hook_output("lm_head")))
        if self.adapter == "gemma2":
            self.handles.append(self.model.register_forward_hook(lambda _module, _inputs, output: self._record("final_logit_softcap", output.logits)))

        self.original_rope = self.modeling.apply_rotary_pos_emb
        self.original_attention = self.modeling.eager_attention_forward

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

        self.modeling.apply_rotary_pos_emb = rope
        self.modeling.eager_attention_forward = attention

    def _v_heads_hook(self, layer: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            batch, sequence, _ = output.shape
            heads = self.model.config.num_key_value_heads
            # Qwen 2 materializes head_dim inside Qwen2Attention when its
            # public config omits that optional derived field.
            dim = self.model.model.layers[layer].self_attn.head_dim
            self._record(f"layer_{layer}_v_heads", output.view(batch, sequence, heads, dim).transpose(1, 2))
        return hook

    def _next_rope_layer(self) -> int:
        for index in range(self.layers):
            if f"layer_{index}_q_heads" not in self.operations:
                return index
        raise ValueError(f"Capture Transformers observou mais aplicações RoPE do que camadas {self.contract['label']} declaradas.")

    def _remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.original_rope is not None:
            self.modeling.apply_rotary_pos_emb = self.original_rope
            self.original_rope = None
        if self.original_attention is not None:
            self.modeling.eager_attention_forward = self.original_attention
            self.original_attention = None

    def _captured_forward(self, tokens: list[int], positions: list[int], cache: DynamicCache | None = None):
        self._install()
        try:
            output = self._forward(tokens, positions, cache)
            expected = self.expected_operation_ids()
            missing = [operation for operation in expected if operation not in self.operations]
            unexpected = sorted(set(self.operations).difference(expected))
            if missing or unexpected:
                raise ValueError(f"Hooks Transformers não cobriram exatamente as operações {self.contract['label']}: ausentes={missing}, extras={unexpected}.")
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
                f"{prefix}_attention", f"{prefix}_o_proj",
                *([f"{prefix}_post_attention_norm"] if self.adapter == "gemma2" else []),
                f"{prefix}_attention_residual", f"{prefix}_pre_ffn_norm",
                f"{prefix}_gate_proj", f"{prefix}_up_proj", f"{prefix}_activation", f"{prefix}_gated_multiply",
                f"{prefix}_down_proj",
                *([f"{prefix}_post_ffn_norm"] if self.adapter == "gemma2" else []),
                f"{prefix}_mlp_residual",
            ])
        return [*result, "final_norm", "lm_head", *(["final_logit_softcap"] if self.adapter == "gemma2" else [])]

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


class CacheLayerSnapshot:
    """Tensor-only cache layer snapshot, independent of runtime constructors."""

    def __init__(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.keys = key.detach().clone()
        self.values = value.detach().clone()


def clone_cache(cache: DynamicCache, layers: int) -> CacheSnapshot:
    # DynamicCache constructors differ across Transformers releases; retaining
    # the native tensors in this narrow immutable snapshot avoids a private
    # cache-copy API while preserving exactly the canonical cache fields.
    result = CacheSnapshot()
    for layer in cache.layers[:layers]:
        if layer.keys is None or layer.values is None:
            raise ValueError("Não é possível capturar snapshot de camada KV incompleta.")
        result.layers.append(CacheLayerSnapshot(layer.keys, layer.values))
    return result


def operation_output(operation_id: str) -> str:
    if operation_id == "token_embedding": return "hidden_states_0"
    if operation_id == "final_norm": return "final_hidden_states"
    if operation_id == "lm_head": return "logits"
    if operation_id == "final_logit_softcap": return "softcapped_logits"
    parts = operation_id.split("_")
    layer = parts[1]
    suffix = "_".join(parts[2:])
    return {
        "input_norm": f"layer_{layer}_attn_norm", "q_proj": f"layer_{layer}_q_linear", "k_proj": f"layer_{layer}_k_linear",
        "v_proj": f"layer_{layer}_v_linear", "q_heads": f"layer_{layer}_q_heads", "k_heads": f"layer_{layer}_k_heads",
        "v_heads": f"layer_{layer}_v_heads", "q_rope": f"layer_{layer}_q_rot", "k_rope": f"layer_{layer}_k_rot",
        "attention": f"layer_{layer}_attention_context", "o_proj": f"layer_{layer}_attention_projected",
        "post_attention_norm": f"layer_{layer}_post_attention_norm",
        "attention_residual": f"layer_{layer}_after_attention", "pre_ffn_norm": f"layer_{layer}_ffn_norm",
        "gate_proj": f"layer_{layer}_gate", "up_proj": f"layer_{layer}_up", "activation": f"layer_{layer}_gate_activated",
        "gated_multiply": f"layer_{layer}_gated_mlp", "down_proj": f"layer_{layer}_mlp_output",
        "post_ffn_norm": f"layer_{layer}_post_ffn_norm", "mlp_residual": f"hidden_states_{int(layer) + 1}",
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
    adapter = request.get("adapter", "llama")
    if not isinstance(adapter, str):
        raise ValueError("Request adapter deve ser string.")
    capture = TransformersDecoderCapture(source, adapter)
    result = capture.execution(tokens, positions) if request["kind"] == "execution" else capture.generation(tokens, positions, request["maxNewTokens"])
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
