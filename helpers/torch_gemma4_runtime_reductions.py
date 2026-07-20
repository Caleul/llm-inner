#!/usr/bin/env python3
"""Execute one pinned Gemma 4 runtime-defined matmul without model access."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch


CONTRACT_ID = "torch-2.12.1-cpu-inference-matmul-v1"
SUPPORTED_TORCH = "2.12.1"
SUPPORTED_TORCH_COMMIT = "7269437d655783a26cba32aa88195b741ff496aa"
SUPPORTED_PLATFORM = "Darwin-arm64"
SUPPORTED_BLAS_SETTING = "BLAS_INFO=accelerate"


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


def exact_object(value: Any, required: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label} has an invalid closed contract.")
    return value


def dimension(expression: Any, values: dict[str, torch.Tensor], tower: dict[str, Any]) -> int:
    if not isinstance(expression, dict) or not isinstance(expression.get("kind"), str):
        raise ValueError("Invocation dimension is invalid.")
    kind = expression["kind"]
    if kind == "constant":
        exact_object(expression, {"kind", "value"}, "constant dimension")
        result = expression["value"]
    elif kind == "tensor-axis":
        exact_object(expression, {"kind", "tensor", "axis"}, "tensor-axis dimension")
        source = values.get(expression["tensor"])
        axis = expression["axis"]
        if source is None or not isinstance(axis, int) or axis < 0 or axis >= source.ndim:
            raise ValueError("Invocation tensor-axis is invalid.")
        result = source.shape[axis]
    elif kind == "tower-parameter":
        exact_object(expression, {"kind", "name"}, "tower dimension")
        result = tower.get(expression["name"])
    elif kind in ("add", "subtract", "multiply", "ceil-divide", "exact-divide"):
        exact_object(expression, {"kind", "left", "right"}, "arithmetic dimension")
        left, right = dimension(expression["left"], values, tower), dimension(expression["right"], values, tower)
        if kind == "add":
            result = left + right
        elif kind == "subtract":
            result = left - right
        elif kind == "multiply":
            result = left * right
        elif kind == "ceil-divide":
            if right <= 0:
                raise ValueError("Invocation ceil-divide divisor is invalid.")
            result = (left + right - 1) // right
        else:
            if right <= 0 or left % right:
                raise ValueError("Invocation exact-divide is not exact.")
            result = left // right
    else:
        raise ValueError(f"Invocation dimension kind {kind!r} is unsupported.")
    if not isinstance(result, int) or result < 0:
        raise ValueError("Invocation dimension did not produce a non-negative integer.")
    return result


def require_value(values: dict[str, torch.Tensor], name: Any) -> torch.Tensor:
    if not isinstance(name, str) or name not in values:
        raise ValueError(f"Invocation value {name!r} is unavailable.")
    return values[name]


def execute_stage(stage: Any, values: dict[str, torch.Tensor], tower: dict[str, Any]) -> None:
    if not isinstance(stage, dict) or not isinstance(stage.get("id"), str) or not isinstance(stage.get("operation"), str):
        raise ValueError("Invocation stage is invalid.")
    operation = stage["operation"]
    common = {"id", "operation", "input", "output"}
    if operation == "cast":
        exact_object(stage, common | {"dtype"}, stage["id"])
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32}.get(stage["dtype"])
        if dtype is None:
            raise ValueError("Invocation cast dtype is unsupported.")
        result = require_value(values, stage["input"]).to(dtype)
    elif operation == "reshape":
        exact_object(stage, common | {"shape"}, stage["id"])
        if not isinstance(stage["shape"], list):
            raise ValueError("Invocation reshape shape is invalid.")
        shape = [dimension(item, values, tower) for item in stage["shape"]]
        if any(item <= 0 for item in shape):
            raise ValueError("Invocation reshape requires positive dimensions.")
        result = require_value(values, stage["input"]).reshape(shape)
    elif operation == "permute":
        exact_object(stage, common | {"axes"}, stage["id"])
        source, axes = require_value(values, stage["input"]), stage["axes"]
        if not isinstance(axes, list) or sorted(axes) != list(range(source.ndim)):
            raise ValueError("Invocation permutation is invalid.")
        result = source.permute(*axes)
    elif operation == "pad-axis-zero":
        exact_object(stage, common | {"axis", "before", "after"}, stage["id"])
        source, axis = require_value(values, stage["input"]), stage["axis"]
        if not isinstance(axis, int) or axis < 0 or axis >= source.ndim:
            raise ValueError("Invocation pad axis is invalid.")
        before, after = dimension(stage["before"], values, tower), dimension(stage["after"], values, tower)
        parts = []
        for width in (before, source.shape[axis], after):
            if width == source.shape[axis]:
                parts.append(source)
            elif width:
                shape = list(source.shape)
                shape[axis] = width
                parts.append(torch.zeros(shape, dtype=source.dtype, device=source.device))
        result = torch.cat(parts, dim=axis)
    elif operation == "unfold":
        exact_object(stage, common | {"axis", "size", "step"}, stage["id"])
        source, axis = require_value(values, stage["input"]), stage["axis"]
        size, step = dimension(stage["size"], values, tower), dimension(stage["step"], values, tower)
        if not isinstance(axis, int) or axis < 0 or axis >= source.ndim or size <= 0 or step <= 0 or size > source.shape[axis]:
            raise ValueError("Invocation unfold contract is invalid.")
        result = source.unfold(axis, size, step)
    elif operation == "move-axis":
        exact_object(stage, common | {"source", "destination"}, stage["id"])
        source = require_value(values, stage["input"])
        if any(not isinstance(stage[key], int) or stage[key] < 0 or stage[key] >= source.ndim for key in ("source", "destination")):
            raise ValueError("Invocation move-axis contract is invalid.")
        result = torch.movedim(source, stage["source"], stage["destination"])
    elif operation == "contiguous":
        exact_object(stage, common, stage["id"])
        result = require_value(values, stage["input"]).contiguous()
    elif operation == "matmul":
        exact_object(stage, {"id", "operation", "left", "right", "output"}, stage["id"])
        result = torch.matmul(require_value(values, stage["left"]), require_value(values, stage["right"]))
    elif operation == "slice-axis":
        exact_object(stage, common | {"axis", "start", "endExclusive"}, stage["id"])
        source, axis = require_value(values, stage["input"]), stage["axis"]
        if not isinstance(axis, int) or axis < 0 or axis >= source.ndim:
            raise ValueError("Invocation slice axis is invalid.")
        start, end = dimension(stage["start"], values, tower), dimension(stage["endExclusive"], values, tower)
        if start > end or end > source.shape[axis]:
            raise ValueError("Invocation slice range is invalid.")
        slices = [slice(None)] * source.ndim
        slices[axis] = slice(start, end)
        result = source[tuple(slices)].contiguous()
    else:
        raise ValueError(f"Invocation stage operation {operation!r} is unsupported.")
    output = stage.get("output")
    if not isinstance(output, str) or not output or output in values:
        raise ValueError("Invocation stage output is invalid or duplicated.")
    values[output] = result


def execute_invocation(request: dict[str, Any], operands: list[torch.Tensor]) -> tuple[str, torch.Tensor]:
    program = request.get("invocationProgram")
    exact_object(program, {"kind", "schemaVersion", "id", "scope", "graphOperation", "orderedOperands", "environment", "stages", "output"}, "invocation program")
    if program["kind"] != "gemma4-runtime-reduction-invocation-program" or program["schemaVersion"] != 2 or program["scope"] != request["scope"] or program["graphOperation"] != request["operation"]:
        raise ValueError("Invocation program identity is incompatible.")
    ordered = program["orderedOperands"]
    if not isinstance(ordered, list) or len(ordered) != 2:
        raise ValueError("Invocation ordered operands are invalid.")
    values: dict[str, torch.Tensor] = {}
    for index, (declaration, operand) in enumerate(zip(ordered, operands)):
        exact_object(declaration, {"ordinal", "name", "role", "dtype", "layout"}, f"operand {index}")
        if declaration["ordinal"] != index or declaration["name"] != f"operand_{index}" or declaration["dtype"] not in ("BF16", "F32") or not isinstance(declaration["layout"], str) or operand.ndim != len(declaration["layout"].split(",")):
            raise ValueError("Invocation operand declaration is incompatible.")
        values[declaration["name"]] = operand
    environment = exact_object(program["environment"], {"runtimeDtype", "towerParameters"}, "invocation environment")
    runtime_dtype = exact_object(environment["runtimeDtype"], {"source", "equals"}, "runtime dtype binding")
    if runtime_dtype != {"source": "program.runtimeDtype", "equals": "BF16"}:
        raise ValueError("Invocation runtime dtype binding is invalid.")
    tower = request.get("tower")
    parameters = environment["towerParameters"]
    if not isinstance(tower, dict) or not isinstance(parameters, list):
        raise ValueError("Invocation tower parameters are invalid.")
    names: list[str] = []
    for parameter in parameters:
        exact_object(parameter, {"name", "source", "numericDomain", "minimumInclusive"}, "tower parameter binding")
        name = parameter["name"]
        minimum = parameter["minimumInclusive"]
        if not isinstance(name, str) or parameter["source"] != f"program.tower.{name}" or parameter["numericDomain"] != "safe-integer" or minimum not in (0, 1):
            raise ValueError("Invocation tower parameter binding is invalid.")
        names.append(name)
        value = tower.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum or value > (1 << 53) - 1:
            raise ValueError("Invocation tower parameter value is invalid.")
    if len(set(names)) != len(names) or set(tower) != set(names) | {"runtimeDtype"} or tower.get("runtimeDtype") != runtime_dtype["equals"]:
        raise ValueError("Invocation tower environment is incomplete or divergent.")
    stages = program["stages"]
    if not isinstance(stages, list) or not stages or len({stage.get("id") for stage in stages if isinstance(stage, dict)}) != len(stages):
        raise ValueError("Invocation stages are invalid or duplicated.")
    for stage in stages:
        execute_stage(stage, values, tower)
    output = exact_object(program["output"], {"value", "dtype", "layout"}, "invocation output")
    result = require_value(values, output["value"])
    if output["dtype"] != "F32" or result.dtype != torch.float32 or not isinstance(output["layout"], str) or result.ndim != len(output["layout"].split(",")):
        raise ValueError("Invocation output contract is incompatible.")
    return program["id"], result


def main() -> None:
    if torch.__version__.split("+")[0] != SUPPORTED_TORCH:
        raise ValueError(f"Runtime reduction requires torch=={SUPPORTED_TORCH}; received {torch.__version__}.")
    build_config = torch.__config__.show()
    actual_platform = f"{platform.system()}-{platform.machine()}"
    if actual_platform != SUPPORTED_PLATFORM or SUPPORTED_BLAS_SETTING not in build_config:
        raise ValueError(
            f"Runtime reduction requires {SUPPORTED_PLATFORM} with {SUPPORTED_BLAS_SETTING}; "
            f"received {actual_platform}."
        )
    if torch.version.git_version != SUPPORTED_TORCH_COMMIT:
        raise ValueError(
            f"Runtime reduction requires torch commit {SUPPORTED_TORCH_COMMIT}; "
            f"received {torch.version.git_version}."
        )
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if request.get("schemaVersion") != 1 or request.get("contractId") != CONTRACT_ID or request.get("scope") not in ("vision", "audio"):
        raise ValueError("Runtime-reduction request contract is invalid.")
    if not isinstance(request.get("operationId"), str) or not request["operationId"] or not isinstance(request.get("operands"), list) or len(request["operands"]) != 2:
        raise ValueError("Runtime-reduction operation and operands are invalid.")
    operands = [tensor(value, f"operand {index}") for index, value in enumerate(request["operands"])]
    with torch.inference_mode():
        invocation_program_id, output = execute_invocation(request, operands)
    result = output.detach().to(device="cpu", dtype=torch.float32).contiguous()
    print(json.dumps({
        "schemaVersion": 1,
        "contractId": CONTRACT_ID,
        "operationId": request["operationId"],
        "scope": request["scope"],
        "operation": request["operation"],
        "invocationProgramId": invocation_program_id,
        "sourceCheckpointAccessed": False,
        "runtimeAttestation": {
            "runtime": f"torch-{SUPPORTED_TORCH}",
            "torchBuildCommit": torch.version.git_version,
            "executionMode": "torch.inference_mode",
            "device": "cpu",
            "platform": actual_platform,
            "backend": "Apple Accelerate SGEMM",
            "blasBuildSetting": SUPPORTED_BLAS_SETTING,
        },
        "output": {"shape": list(result.shape), "values": result.reshape(-1).tolist()},
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
