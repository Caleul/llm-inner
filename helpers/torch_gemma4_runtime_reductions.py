#!/usr/bin/env python3
"""Execute one pinned Gemma 4 runtime-defined matmul without model access."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


CONTRACT_ID = "torch-2.12.1-cpu-inference-matmul-v1"
SUPPORTED_TORCH = "2.12.1"


def tensor(raw: Any, label: str) -> torch.Tensor:
    if not isinstance(raw, dict) or not isinstance(raw.get("shape"), list) or not isinstance(raw.get("values"), list):
        raise ValueError(f"{label} must be a serialized dense tensor.")
    shape = raw["shape"]
    if any(not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError(f"{label} has an invalid shape.")
    result = torch.tensor(raw["values"], dtype=torch.float32)
    if result.numel() != torch.tensor(shape).prod().item() or not torch.isfinite(result).all():
        raise ValueError(f"{label} has an invalid payload.")
    return result.reshape(shape)


def vision(request: dict[str, Any], left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    tower = request["tower"]
    if tower != {
        "attentionHeads": tower.get("attentionHeads"),
        "headDim": tower.get("headDim"),
        "runtimeDtype": "BF16",
    } or tower["attentionHeads"] <= 0 or tower["headDim"] <= 0:
        raise ValueError("Vision runtime-reduction tower contract is invalid.")
    if request["operation"] == "attention-score-matmul":
        if left.ndim != 4 or list(left.shape) != list(right.shape):
            raise ValueError("Vision score operands must be matching [B,H,Q,D] tensors.")
        return torch.matmul(left.to(torch.bfloat16), right.to(torch.bfloat16).transpose(2, 3)).float()
    if request["operation"] == "attention-value-matmul":
        if left.ndim != 4 or right.ndim != 4 or left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1] or left.shape[3] != right.shape[2]:
            raise ValueError("Vision value operands must be [B,H,Q,K] and [B,H,K,D].")
        context = torch.matmul(left.to(torch.bfloat16), right.to(torch.bfloat16))
        return context.transpose(1, 2).contiguous().reshape(left.shape[0], left.shape[2], -1).float()
    raise ValueError("Unknown vision runtime reduction.")


def audio_context(value: torch.Tensor, chunk: int, past: int, future: int) -> torch.Tensor:
    value = F.pad(value, (0, 0, 0, 0, past, future + chunk - 1))
    return torch.movedim(value.unfold(1, chunk + past + future, chunk), -1, 2).contiguous()


def audio(request: dict[str, Any], left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    tower = request["tower"]
    heads, dim = tower.get("attentionHeads"), tower.get("headDim")
    chunk, left_context, right_context = tower.get("attentionChunkSize"), tower.get("attentionContextLeft"), tower.get("attentionContextRight")
    if tower.get("runtimeDtype") != "BF16" or any(not isinstance(value, int) or value <= 0 for value in (heads, dim, chunk, left_context)) or not isinstance(right_context, int) or right_context < 0:
        raise ValueError("Audio runtime-reduction tower contract is invalid.")
    if request["operation"] == "chunked-relative-attention-values":
        if right.ndim != 3 or right.shape[2] != heads * dim:
            raise ValueError("Audio V operand must be [B,S,H*D].")
        batch, sequence, _ = right.shape
        blocks = (sequence + chunk - 1) // chunk
        context = chunk + left_context - 1 + right_context
        if list(left.shape) != [batch, heads, blocks, chunk, context]:
            raise ValueError("Audio attention-weight operand is incompatible.")
        values = audio_context(right.reshape(batch, sequence, heads, dim), chunk, left_context - 1, right_context)
        result = torch.matmul(left, values.permute(0, 3, 1, 2, 4))
        return result.permute(0, 2, 3, 1, 4).reshape(batch, blocks * chunk, -1)[:, :sequence].contiguous()
    if left.ndim != 3 or left.shape[2] != heads * dim:
        raise ValueError("Audio Q operand must be [B,S,H*D].")
    batch, sequence, _ = left.shape
    blocks = (sequence + chunk - 1) // chunk
    padded = F.pad(left.reshape(batch, sequence, heads, dim), (0, 0, 0, 0, 0, blocks * chunk - sequence))
    queries = padded.reshape(batch, blocks, chunk, heads, dim).contiguous().permute(0, 3, 1, 2, 4)
    if request["operation"] == "chunked-attention-content-matmul":
        if list(right.shape) != [batch, sequence, heads * dim]:
            raise ValueError("Audio content K operand is incompatible.")
        keys = audio_context(right.reshape(batch, sequence, heads, dim), chunk, left_context - 1, right_context)
        return torch.matmul(queries, keys.permute(0, 3, 1, 4, 2))
    if request["operation"] == "relative-attention-position-matmul":
        if right.ndim != 3 or right.shape[0] != 1 or right.shape[2] != heads * dim:
            raise ValueError("Audio relative-key operand is incompatible.")
        relative = right.reshape(-1, heads, dim)
        result = torch.matmul(queries.reshape(batch, heads, -1, dim), relative.permute(1, 2, 0))
        return result.reshape(batch, heads, blocks, chunk, -1)
    raise ValueError("Unknown audio runtime reduction.")


def main() -> None:
    if torch.__version__.split("+")[0] != SUPPORTED_TORCH:
        raise ValueError(f"Runtime reduction requires torch=={SUPPORTED_TORCH}; received {torch.__version__}.")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if request.get("schemaVersion") != 1 or request.get("contractId") != CONTRACT_ID or request.get("scope") not in ("vision", "audio"):
        raise ValueError("Runtime-reduction request contract is invalid.")
    if not isinstance(request.get("operationId"), str) or not request["operationId"] or not isinstance(request.get("operands"), list) or len(request["operands"]) != 2:
        raise ValueError("Runtime-reduction operation and operands are invalid.")
    left, right = (tensor(value, f"operand {index}") for index, value in enumerate(request["operands"]))
    with torch.inference_mode():
        output = vision(request, left, right) if request["scope"] == "vision" else audio(request, left, right)
    result = output.detach().to(device="cpu", dtype=torch.float32).contiguous()
    print(json.dumps({
        "schemaVersion": 1,
        "contractId": CONTRACT_ID,
        "sourceCheckpointAccessed": False,
        "output": {"shape": list(result.shape), "values": result.reshape(-1).tolist()},
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
