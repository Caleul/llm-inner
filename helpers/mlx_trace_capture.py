#!/usr/bin/env python3
"""Independent MLX-kernel capture for explicitly lowered dense F32 IR.

This is intentionally not a model-family detector or a fallback executor.  The
Node caller has already selected and validated an adapter, and this helper
refuses every storage or operation contract outside the narrow, declared MLX
capture boundary.  Its output is consumed as an integrity-bound trace, never
as candidate execution input.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file


def tensor_payload(value: mx.array) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float32)
    return {"dtype": "F32", "shape": list(array.shape), "valuesBase64": base64.b64encode(array.tobytes(order="C")).decode("ascii")}


class Capture:
    def __init__(self, request: dict[str, Any]):
        self.ir = request["ir"]
        self.source = Path(request["source"])
        if self.ir["source"]["format"] != "safetensors":
            raise ValueError("MLX capture suporta somente Safetensors denso F32.")
        model_type = self.ir["architecture"]["modelType"]
        if model_type not in {"llama", "gemma", "qwen3"}:
            raise ValueError(f"MLX capture não possui contrato independente para {model_type}; suportados: llama, gemma, qwen3.")
        self.weights = self._weights()

    def _weights(self) -> dict[str, mx.array]:
        index = self.source / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text("utf8"))["weight_map"]
            shards = sorted(set(weight_map.values()))
        else:
            shards = [file.name for file in self.source.glob("*.safetensors")]
        result: dict[str, mx.array] = {}
        for shard in shards:
            for name, value in load_file(str(self.source / shard)).items():
                if value.dtype != np.float32:
                    raise ValueError(f"{name}: MLX capture requer storage F32, recebeu {value.dtype}.")
                if name in result:
                    raise ValueError(f"Tensor duplicado: {name}")
                result[name] = mx.array(value, dtype=mx.float32)
        return result

    def forward(self, input_ids: list[int], positions: list[int], past: dict[int, tuple[mx.array, mx.array]] | None = None) -> tuple[dict[str, mx.array], dict[int, tuple[mx.array, mx.array]]]:
        if not input_ids or len(input_ids) != len(positions):
            raise ValueError("input_ids e positions devem ser vetores não vazios de mesmo tamanho.")
        values: dict[str, mx.array] = {"input_ids": mx.array([input_ids], dtype=mx.int32)}
        values["position_ids"] = mx.array([positions], dtype=mx.int32)
        caches: dict[int, tuple[mx.array, mx.array]] = {}
        operations = [*self.ir["prelude"], *[op for layer in self.ir["layers"] for op in layer["operations"]], *self.ir["epilogue"]]
        for op in operations:
            kind = op["op"]
            if kind == "embedding":
                value = self.weights[op["weight"]["name"]][values[op["tokenInput"]]]
                if "scale" in op:
                    value = value * np.float32(op["scale"])
            elif kind == "rms_norm":
                source = values[op["input"]]
                weight = self.weights[op["weight"]["name"]]
                if op["weightTransform"] == "one_plus_weight": weight = weight + np.float32(1)
                value = source * mx.rsqrt(mx.mean(source * source, axis=-1, keepdims=True) + np.float32(op["epsilon"])) * weight
            elif kind == "linear":
                value = mx.matmul(values[op["input"]], mx.transpose(self.weights[op["weight"]["name"]]))
                if op.get("bias") is not None: value = value + self.weights[op["bias"]["name"]]
            elif kind == "reshape_heads":
                source = values[op["input"]]
                batch, sequence, _ = source.shape
                value = mx.transpose(mx.reshape(source, (batch, sequence, op["numHeads"], op["headDim"])), (0, 2, 1, 3))
            elif kind == "rotary_embedding":
                value = self._rope(values[op["input"]], positions, op)
            elif kind == "scaled_dot_product_attention":
                query = values[op["query"]]
                if op.get("kvSharing", {}).get("enabled"):
                    producer = op["kvSharing"].get("producerLayer")
                    if producer not in caches: raise ValueError(f"{op['id']}: producer KV ausente")
                    key, val = caches[producer]
                    past_length = key.shape[2] - query.shape[2]
                else:
                    key, val = values[op["key"]], values[op["value"]]
                    prior = past.get(op["layer"]) if past else None
                    past_length = 0 if prior is None else prior[0].shape[2]
                    if prior is not None:
                        key, val = mx.concatenate([prior[0], key], axis=2), mx.concatenate([prior[1], val], axis=2)
                    caches[op["layer"]] = (key, val)
                value = self._attention(query, key, val, op, past_length)
            elif kind == "activation":
                source = values[op["input"]]
                if op["function"] == "silu": value = source * mx.sigmoid(source)
                elif op["function"] == "gelu" and op.get("approximation") == "tanh": value = 0.5 * source * (1 + mx.tanh(np.sqrt(2 / np.pi) * (source + np.float32(0.044715) * source * source * source)))
                else: raise ValueError(f"{op['id']}: activation não suportada")
            elif kind == "elementwise":
                inputs = [values[name] for name in op["inputs"]]
                if op["kind"] == "add": value = inputs[0] + inputs[1]
                elif op["kind"] == "multiply": value = inputs[0] * inputs[1]
                elif op["kind"] == "scale": value = inputs[0] * np.float32(op["scalar"])
                elif op["kind"] == "tanh_softcap": value = np.float32(op["scalar"]) * mx.tanh(inputs[0] / np.float32(op["scalar"]))
                else: raise ValueError(f"{op['id']}: elementwise não suportado")
            else:
                raise ValueError(f"{op['id']}: op não suportada pelo capture MLX: {kind}")
            values[op["output"]] = mx.astype(value, mx.float32)
        mx.eval(*values.values(), *[item for pair in caches.values() for item in pair])
        return values, caches

    def _rope(self, source: mx.array, positions: list[int], op: dict[str, Any]) -> mx.array:
        if op["ropeType"] != "default" or op["layout"] != "rotate_half" or op.get("scaling") is not None:
            raise ValueError(f"{op['id']}: RoPE fora do contrato default/rotate_half")
        _, _, sequence, dim = source.shape
        rotary = op["rotaryDim"]
        half = rotary // 2
        angles = np.array(positions, dtype=np.float32)[:, None] / np.power(np.float32(op["theta"]), np.arange(half, dtype=np.float32) * np.float32(2 / rotary))[None, :]
        cos = mx.reshape(mx.array(np.cos(angles), dtype=mx.float32), (1, 1, sequence, half))
        sin = mx.reshape(mx.array(np.sin(angles), dtype=mx.float32), (1, 1, sequence, half))
        first, second = source[..., :half], source[..., half:rotary]
        rotated = mx.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1)
        return rotated if rotary == dim else mx.concatenate([rotated, source[..., rotary:]], axis=-1)

    def _attention(self, query: mx.array, key: mx.array, value: mx.array, op: dict[str, Any], past_length: int) -> mx.array:
        group = op["numAttentionHeads"] // op["numKeyValueHeads"]
        key = mx.repeat(key, group, axis=1)
        value = mx.repeat(value, group, axis=1)
        scores = mx.matmul(query, mx.transpose(key, (0, 1, 3, 2))) * np.float32(op["scale"])
        if op.get("scoreSoftcap") is not None:
            cap = np.float32(op["scoreSoftcap"]); scores = cap * mx.tanh(scores / cap)
        query_sequence, key_sequence = query.shape[2], key.shape[2]
        mask = np.full((query_sequence, key_sequence), -np.inf, dtype=np.float32)
        for row in range(query_sequence):
            absolute = past_length + row
            start = 0 if op.get("slidingWindow") is None else max(0, absolute - op["slidingWindow"] + 1)
            end = min(absolute, key_sequence - 1) if op["causal"] else key_sequence - 1
            mask[row, start:end + 1] = 0
        probabilities = mx.softmax(scores + mx.array(mask, dtype=mx.float32), axis=-1)
        output = mx.matmul(probabilities, value)
        return mx.reshape(mx.transpose(output, (0, 2, 1, 3)), (query.shape[0], query_sequence, -1))


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text("utf8"))
    capture = Capture(request)
    kind = request["kind"]
    if kind == "execution":
        values, caches = capture.forward(request["inputTokens"], request["positionIds"])
        operations = [*capture.ir["prelude"], *[op for layer in capture.ir["layers"] for op in layer["operations"]], *capture.ir["epilogue"]]
        result = {"operations": [{"operationId": op["id"], "output": op["output"], "tensor": tensor_payload(values[op["output"]])} for op in operations], "pastKeyValues": [{"layer": layer, "key": tensor_payload(pair[0]), "value": tensor_payload(pair[1])} for layer, pair in sorted(caches.items())]}
    elif kind == "generation":
        prompt, positions, limit = request["inputTokens"], request["positionIds"], request["maxNewTokens"]
        values, cache = capture.forward(prompt, positions)
        tokens, steps, logits, snapshots = [], [], [], []
        current_logits = values["softcapped_logits"] if "softcapped_logits" in values else values["logits"]
        for index in range(limit):
            selection = current_logits[:, -1:, :]
            token = int(np.asarray(mx.argmax(selection, axis=-1)).reshape(-1)[0])
            tokens.append(token); steps.append({"tokenId": token, "positionId": positions[-1] + 1 + index}); logits.append(tensor_payload(selection))
            values, cache = capture.forward([token], [steps[-1]["positionId"]], cache)
            snapshots.append([{"layer": layer, "key": tensor_payload(pair[0]), "value": tensor_payload(pair[1])} for layer, pair in sorted(cache.items())])
            current_logits = values["softcapped_logits"] if "softcapped_logits" in values else values["logits"]
        result = {"generatedTokenIds": tokens, "steps": steps, "selectionLogits": logits, "stepPastKeyValues": snapshots, "logits": tensor_payload(current_logits), "pastKeyValues": [{"layer": layer, "key": tensor_payload(pair[0]), "value": tensor_payload(pair[1])} for layer, pair in sorted(cache.items())]}
    else:
        raise ValueError("kind deve ser execution ou generation")
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
