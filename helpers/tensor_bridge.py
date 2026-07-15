#!/usr/bin/env python3
"""Backend de referência para leitura/dequantização.

- Safetensors densos: safetensors.numpy
- MLX quantizado: mlx.core.dequantize, usando config.json
- GGUF: pacote oficial gguf/gguf-py

O processo fica vivo e recebe JSON-RPC por stdin para evitar spawn por tensor.
"""
from __future__ import annotations

import json
import base64
import sys
import traceback
from pathlib import Path
from typing import Any

SOURCE = Path(sys.argv[1]).resolve()


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


class Backend:
    def inspect(self) -> dict[str, Any]:
        raise NotImplementedError

    def read_rows(self, tensor: str, row_start: int, row_end: int, col_start: int, col_end: int) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return


class SafeTensorsBackend(Backend):
    def __init__(self, root: Path):
        self.root = root
        self.config = json.loads((root / "config.json").read_text("utf-8"))
        self.weight_map = self._weight_map()
        self._safe_handles: dict[str, tuple[Any, Any]] = {}
        self._mlx_shards: dict[str, dict[str, Any]] = {}

    def _weight_map(self) -> dict[str, str]:
        indexes = list(self.root.glob("*.safetensors.index.json"))
        if indexes:
            return json.loads(indexes[0].read_text("utf-8"))["weight_map"]
        result: dict[str, str] = {}
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise RuntimeError("Instale 'safetensors' para usar o bridge: pip install safetensors") from exc
        for shard in self.root.glob("*.safetensors"):
            with safe_open(str(shard), framework="numpy") as handle:
                for key in handle.keys():
                    if key in result:
                        raise RuntimeError(f"Tensor duplicado: {key}")
                    result[key] = shard.name
        return result

    def _quant_root(self) -> dict[str, Any] | None:
        value = self.config.get("quantization") or self.config.get("quantization_config")
        return value if isinstance(value, dict) else None

    def _quant_spec(self, tensor_name: str) -> dict[str, Any] | None:
        if not tensor_name.endswith(".weight"):
            return None
        root = self._quant_root()
        if not root:
            return None
        module = tensor_name[: -len(".weight")]
        override = root.get(module)
        if override is False:
            return None
        local = override if isinstance(override, dict) else {}
        bits = local.get("bits", root.get("bits"))
        group_size = local.get("group_size", root.get("group_size"))
        mode = local.get("mode", root.get("mode"))
        if bits is None or group_size is None or not isinstance(mode, str) or not mode:
            return None
        return {"bits": int(bits), "group_size": int(group_size), "mode": str(mode), "module": module}

    def _safe_handle(self, shard: str):
        if shard not in self._safe_handles:
            try:
                from safetensors import safe_open
            except Exception as exc:
                raise RuntimeError("Instale 'safetensors': pip install safetensors") from exc
            context = safe_open(str(self.root / shard), framework="numpy")
            handle = context.__enter__()
            self._safe_handles[shard] = (context, handle)
        return self._safe_handles[shard][1]

    def _dense(self, name: str):
        shard = self.weight_map[name]
        return self._safe_handle(shard).get_tensor(name)

    def _mlx_shard(self, shard: str) -> dict[str, Any]:
        if shard not in self._mlx_shards:
            try:
                import mlx.core as mx
            except Exception as exc:
                raise RuntimeError("Modelo MLX quantizado requer macOS/Apple Silicon e 'pip install mlx'.") from exc
            self._mlx_shards[shard] = mx.load(str(self.root / shard))
        return self._mlx_shards[shard]

    def _mlx_tensor(self, name: str):
        shard = self.weight_map[name]
        values = self._mlx_shard(shard)
        if name not in values:
            raise KeyError(f"Tensor {name} ausente em {shard}")
        return values[name]

    def inspect(self) -> dict[str, Any]:
        # A inspeção principal do Safetensors é feita em TypeScript; este método existe por simetria.
        return {"config": self.config, "rawMetadata": {}, "tensors": []}

    def read_rows(self, tensor: str, row_start: int, row_end: int, col_start: int, col_end: int) -> dict[str, Any]:
        quant = self._quant_spec(tensor)
        if quant is None:
            array = self._dense(tensor)
            if array.ndim != 2:
                raise ValueError(f"{tensor} não é matriz 2D: {array.shape}")
            return {"values": array[row_start:row_end, col_start:col_end].astype("float64").tolist()}

        try:
            import mlx.core as mx
        except Exception as exc:
            raise RuntimeError("Dequantização MLX exata requer 'pip install mlx'.") from exc

        module = quant["module"]
        weight = self._mlx_tensor(tensor)[row_start:row_end]
        if weight.dtype != mx.uint32:
            raise ValueError(f"{tensor}: config MLX não pode reinterpretar storage {weight.dtype} como peso U32 embalado")
        scales_name = f"{module}.scales"
        biases_name = f"{module}.biases"
        global_scale_name = f"{module}.global_scale"
        scales = self._mlx_tensor(scales_name)[row_start:row_end]
        biases = self._mlx_tensor(biases_name)[row_start:row_end] if biases_name in self.weight_map else None
        kwargs: dict[str, Any] = {
            "group_size": quant["group_size"],
            "bits": quant["bits"],
            "mode": quant["mode"],
        }
        if global_scale_name in self.weight_map:
            kwargs["global_scale"] = self._mlx_tensor(global_scale_name)
        dequantized = mx.dequantize(weight, scales, biases, **kwargs)
        sliced = dequantized[:, col_start:col_end]
        return {"values": mx.array(sliced, dtype=mx.float32).tolist()}

    def read_tensor_f32(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return a complete dequantized F32 tensor as bytes, never JSON floats."""
        tensor = str(request.get("tensor", ""))
        expected = request.get("quantization")
        if not isinstance(expected, dict):
            raise ValueError("read_tensor_f32 requer quantization declarada pelo catálogo")
        quant = self._quant_spec(tensor)
        if quant is None:
            raise ValueError(f"{tensor}: não possui especificação MLX verificável")
        if request.get("storage_dtype") != "U32":
            raise ValueError(f"{tensor}: execução MLX requer storage_dtype U32")
        if expected.get("family") != "mlx" or any(expected.get(key) != quant.get(key) for key in ("mode", "bits", "group_size")):
            raise ValueError(f"{tensor}: contrato de quantização do RPC diverge do config.json")
        module = quant["module"]
        expected_scales = f"{module}.scales"
        if expected.get("scale_tensor") != expected_scales:
            raise ValueError(f"{tensor}: scales declaradas pelo RPC não correspondem ao módulo MLX")
        expected_biases = f"{module}.biases"
        expected_global_scale = f"{module}.global_scale"
        if expected.get("bias_tensor") not in (None, expected_biases) or expected.get("global_scale_tensor") not in (None, expected_global_scale):
            raise ValueError(f"{tensor}: metadados auxiliares MLX divergem do módulo")
        storage_shape = request.get("storage_shape")
        logical_shape = request.get("logical_shape")
        if not (isinstance(storage_shape, list) and isinstance(logical_shape, list) and len(storage_shape) == 2 and len(logical_shape) == 2):
            raise ValueError(f"{tensor}: execução MLX requer shapes 2D declarados")
        weight = self._mlx_tensor(tensor)
        scales = self._mlx_tensor(expected_scales)
        try:
            import mlx.core as mx
        except Exception as exc:
            raise RuntimeError("Dequantização MLX exata requer 'pip install mlx'.") from exc
        if weight.dtype != mx.uint32:
            raise ValueError(f"{tensor}: storage carregado não é U32 embalado")
        if list(weight.shape) != storage_shape or len(scales.shape) != 2 or scales.shape[0] != weight.shape[0]:
            raise ValueError(f"{tensor}: storage/scales carregados não correspondem ao catálogo")
        if int(scales.shape[1]) * int(quant["group_size"]) != int(logical_shape[1]) or int(weight.shape[0]) != int(logical_shape[0]):
            raise ValueError(f"{tensor}: shape lógico não corresponde a scales e group_size")
        biases = self._mlx_tensor(expected_biases) if expected_biases in self.weight_map else None
        if biases is not None and list(biases.shape) != list(scales.shape):
            raise ValueError(f"{tensor}: biases MLX devem ter o mesmo shape de scales")
        kwargs: dict[str, Any] = {"group_size": quant["group_size"], "bits": quant["bits"], "mode": quant["mode"]}
        if expected_global_scale in self.weight_map:
            kwargs["global_scale"] = self._mlx_tensor(expected_global_scale)
        dequantized = mx.dequantize(weight, scales, biases, **kwargs).astype(mx.float32)
        if list(dequantized.shape) != logical_shape:
            raise ValueError(f"{tensor}: mlx.core.dequantize retornou shape {list(dequantized.shape)}, esperado {logical_shape}")
        raw = dequantized.tobytes()
        if len(raw) != int(logical_shape[0]) * int(logical_shape[1]) * 4:
            raise ValueError(f"{tensor}: mlx.core.dequantize retornou tamanho F32 inválido")
        return {"shape": logical_shape, "f32leBase64": base64.b64encode(raw).decode("ascii")}

    def close(self) -> None:
        for context, _handle in self._safe_handles.values():
            try:
                context.__exit__(None, None, None)
            except Exception:
                pass
        self._safe_handles.clear()
        self._mlx_shards.clear()


class GgufBackend(Backend):
    def __init__(self, file: Path):
        try:
            import gguf
        except Exception as exc:
            raise RuntimeError("GGUF requer o pacote oficial gguf-py: pip install gguf") from exc
        self.gguf = gguf
        self.reader = gguf.GGUFReader(str(file))
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        self.metadata = self._metadata()
        self.config = self._canonical_config()

    def _field_value(self, field: Any) -> Any:
        # API do gguf-py: field.parts contém views numpy; field.data aponta os índices relevantes.
        try:
            values = [field.parts[index] for index in field.data]
            decoded = []
            for value in values:
                item = value.tolist() if hasattr(value, "tolist") else value
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                decoded.append(item)
            return decoded[0] if len(decoded) == 1 else decoded
        except Exception:
            return str(field)

    def _metadata(self) -> dict[str, Any]:
        return {name: self._field_value(field) for name, field in self.reader.fields.items()}

    def _canonical_config(self) -> dict[str, Any]:
        arch = str(self.metadata.get("general.architecture", ""))
        prefix = f"{arch}." if arch else ""
        def first(*keys: str, default: Any = None) -> Any:
            for key in keys:
                value = self.metadata.get(key)
                if value is not None:
                    return value
            return default
        return {
            "model_type": arch,
            "num_hidden_layers": first(prefix + "block_count"),
            "hidden_size": first(prefix + "embedding_length"),
            "intermediate_size": first(prefix + "feed_forward_length"),
            "num_attention_heads": first(prefix + "attention.head_count"),
            "num_key_value_heads": first(prefix + "attention.head_count_kv", default=first(prefix + "attention.head_count")),
            "head_dim": first(prefix + "attention.key_length"),
            "rope_dimension_count": first(prefix + "rope.dimension_count"),
            "rope_theta": first(prefix + "rope.freq_base", default=10000.0),
            "rms_norm_eps": first(prefix + "attention.layer_norm_rms_epsilon", default=1e-6),
            "vocab_size": len(self.metadata.get("tokenizer.ggml.tokens", [])) if isinstance(self.metadata.get("tokenizer.ggml.tokens"), list) else None,
        }

    def inspect(self) -> dict[str, Any]:
        tensors = []
        for tensor in self.reader.tensors:
            shape = [int(value) for value in tensor.shape.tolist()]
            # gguf-py expõe shape lógico; tensor.data mantém a representação quantizada.
            tensors.append({
                "name": tensor.name,
                "storageDtype": str(tensor.tensor_type.name),
                "storageShape": list(tensor.data.shape),
                "logicalShape": shape,
                "quantization": {
                    "family": "gguf",
                    "mode": str(tensor.tensor_type.name),
                    "tensorType": str(tensor.tensor_type.name),
                },
            })
        return {"config": self.config, "rawMetadata": self.metadata, "tensors": tensors}

    def read_rows(self, tensor: str, row_start: int, row_end: int, col_start: int, col_end: int) -> dict[str, Any]:
        import numpy as np
        value = self.tensors[tensor]
        data = value.data
        # Para matrizes, gguf-py organiza a última dimensão como bytes/blocos da linha.
        # Dequantizar apenas as linhas pedidas evita materializar o tensor inteiro.
        if len(value.shape) != 2:
            raise ValueError(f"{tensor} não é matriz 2D: {value.shape}")
        selected = data[row_start:row_end]
        dequantized = self.gguf.dequantize(selected, value.tensor_type)
        matrix = np.asarray(dequantized, dtype=np.float32).reshape((row_end - row_start, -1))
        return {"values": matrix[:, col_start:col_end].astype(np.float64).tolist()}


def open_backend(source: Path) -> Backend:
    if source.is_file() and source.suffix.lower() == ".gguf":
        return GgufBackend(source)
    if source.is_dir():
        return SafeTensorsBackend(source)
    raise ValueError(f"Fonte não suportada: {source}")


backend = open_backend(SOURCE)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request: dict[str, Any] = json.loads(line)
    request_id = int(request["id"])
    method = request["method"]
    params = request.get("params", {})
    try:
        if method == "inspect":
            result = backend.inspect()
        elif method == "read_rows":
            result = backend.read_rows(
                str(params["tensor"]),
                int(params["row_start"]),
                int(params["row_end"]),
                int(params["col_start"]),
                int(params["col_end"]),
            )
        elif method == "read_tensor_f32":
            if not isinstance(backend, SafeTensorsBackend):
                raise ValueError("read_tensor_f32 é exclusivo para checkpoints MLX Safetensors")
            result = backend.read_tensor_f32(params)
        elif method == "close":
            backend.close()
            emit({"id": request_id, "result": True})
            break
        else:
            raise ValueError(f"Método desconhecido: {method}")
        emit({"id": request_id, "result": result})
    except Exception as exc:
        emit({
            "id": request_id,
            "error": {"message": str(exc), "stack": traceback.format_exc()},
        })
