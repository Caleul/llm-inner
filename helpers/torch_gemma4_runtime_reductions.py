#!/usr/bin/env python3
"""Execute one pinned Gemma 4 runtime-defined matmul without model access."""
from __future__ import annotations

import ctypes
import json
import hashlib
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Capture the complete launch environment before importing Torch.  The provider
# deliberately starts this program without inheriting its parent's environment;
# Torch may add cache variables during import, which are not launch inputs.
RUNTIME_PROCESS_ENVIRONMENT_AT_START = dict(os.environ)

import torch


CONTRACT_ID = "torch-2.12.1-cpu-inference-matmul-v1"
SUPPORTED_TORCH = "2.12.1"
SUPPORTED_TORCH_COMMIT = "7269437d655783a26cba32aa88195b741ff496aa"
SUPPORTED_PLATFORM = "Darwin-arm64"
SUPPORTED_BLAS_SETTING = "BLAS_INFO=accelerate"
SUPPORTED_RUNTIME_PROCESS_ENVIRONMENT = {
    "schemaVersion": 1,
    "inheritance": "none",
    "variables": {
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x47",
    },
}
SUPPORTED_RUNTIME_ENVIRONMENT = {
    "pythonImplementation": "CPython",
    "pythonVersion": "3.14.3",
    "operatingSystem": "macOS",
    "operatingSystemVersion": "26.5.2",
    "operatingSystemBuild": "25F84",
    "kernelRelease": "25.5.0",
    "machineModel": "Mac15,10",
    "cpuBrand": "Apple M3 Max",
    "torchBuildConfigSha256": "606e3853213dea3faabc6d58b66ed7e419ee4452a6d53c2b27495a2ecc4e07a7",
    "runtimeDependencyIdentity": {
        "schemaVersion": 1,
        "files": [
            {"role": "cpython-runtime", "locator": "sys.base_prefix/Python", "bytes": 5438400, "sha256": "e5728c35bdc26dee85e45b3fb94780afc1c9f97ced6b0af64d54e4eab3422e0a"},
            {"role": "torch-python-extension", "locator": "torch._C.__file__", "bytes": 50232, "sha256": "c48ade47e58bf4d28f4f41bd59b5be4b37e4931b36a1a90c44e9d8f5cb6ee434"},
            {"role": "torch-python-library", "locator": "torch.package/lib/libtorch_python.dylib", "bytes": 29929032, "sha256": "cb0f00560a29f0ff82cc125013c4fe5dfd544fba9cc728aa922efa56cd047557"},
            {"role": "torch-cpu-kernel-library", "locator": "torch.package/lib/libtorch_cpu.dylib", "bytes": 248507328, "sha256": "791f549846676c37c778a6bb043b6fafae54d3d32112a782a388be4ba6e6d52f"},
            {"role": "torch-tensor-runtime-library", "locator": "torch.package/lib/libc10.dylib", "bytes": 1072704, "sha256": "935940fedf52ad9d3aa40f1570ec4e6529b6be7d860bf0daaced344927d7e657"},
            {"role": "torch-core-library", "locator": "torch.package/lib/libtorch.dylib", "bytes": 16752, "sha256": "4eab0bfaef1b14044359cefebb0030f1dcc5ad5f757d514a2aa7fff6dd08b032"},
            {"role": "torch-shared-memory-library", "locator": "torch.package/lib/libshm.dylib", "bytes": 64016, "sha256": "43e8d43211fdbc270a2a5a6d580ed0344f74f5f15cf34f8abe5da6f958323dfe"},
            {"role": "openmp-runtime-library", "locator": "torch.package/lib/libomp.dylib", "bytes": 856096, "sha256": "6256bee09e93c28d71c65711cc69224d69994c6965648b628b70a22772fe98d4"},
            {"role": "torch-global-dependencies-library", "locator": "torch.package/lib/libtorch_global_deps.dylib", "bytes": 16760, "sha256": "63504e19a4f955eb4abe956a3f90578e343b033be75f26aa168aa4991db437fd"},
        ],
        "pythonSourceTrees": [
            {"role": "cpython-standard-library", "locator": "os.__file__/..", "includeSuffixes": [".py", ".pyi"], "excludePathParts": ["__pycache__", "site-packages"], "canonicalLeafEncoding": "relative-posix-path\\0byte-count\\0sha256-hex\\n", "files": 1848, "bytes": 35754693, "sha256": "3179ebdc3d1f5bbb1f3612d64fd0feb137af43688523c8c6eb0ff12eb9b4254d"},
            {"role": "torch-python-package", "locator": "torch.__file__/..", "includeSuffixes": [".py", ".pyi"], "excludePathParts": ["__pycache__"], "canonicalLeafEncoding": "relative-posix-path\\0byte-count\\0sha256-hex\\n", "files": 2230, "bytes": 46148427, "sha256": "31caad9097d0c18d1ea1067589d973544965f4e91d27f223c865608ffbc9c8e7"},
        ],
        "sharedCacheImages": [
            {"role": "accelerate-blas", "installName": "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib", "architecture": "arm64e", "machoUuid": "F078C775-D8DC-3C4D-879F-A9BB228DBE06"},
        ],
    },
}
SUPPORTED_RUNTIME_EXECUTION_STATE = {
    "schemaVersion": 2,
    "intraopThreads": 10,
    "interopThreads": 14,
    "deterministicAlgorithms": False,
    "deterministicAlgorithmsWarnOnly": False,
    "float32MatmulPrecision": "highest",
    "defaultDtype": "torch.float32",
    "defaultDevice": "cpu",
    "cpuCapability": "DEFAULT",
    "flushDenormal": False,
    "subnormalProbe": {
        "encoding": "ieee-f32-little-endian",
        "inputBits": 1,
        "multipliedByOneBits": 1,
    },
    "cFloatingPointRoundingMode": "FE_TONEAREST",
    "mkldnnAvailable": False,
    "mkldnnEnabled": True,
}

FE_TONEAREST = 0x00000000


def file_identity(role: str, locator: str, path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return {
        "role": role,
        "locator": locator,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def source_tree_identity(
    role: str,
    locator: str,
    root: Path,
    include_suffixes: tuple[str, ...],
    exclude_path_parts: tuple[str, ...],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    candidates = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in candidates:
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path.suffix not in include_suffixes
            or any(part in exclude_path_parts for part in relative.parts)
        ):
            continue
        payload = path.read_bytes()
        leaf_sha256 = hashlib.sha256(payload).hexdigest()
        digest.update(
            f"{relative.as_posix()}\0{len(payload)}\0{leaf_sha256}\n".encode("utf-8")
        )
        files += 1
        total_bytes += len(payload)
    return {
        "role": role,
        "locator": locator,
        "includeSuffixes": list(include_suffixes),
        "excludePathParts": list(exclude_path_parts),
        "canonicalLeafEncoding": "relative-posix-path\\0byte-count\\0sha256-hex\\n",
        "files": files,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def runtime_dependency_identity() -> dict[str, Any]:
    torch_root = Path(torch.__file__).resolve().parent
    blas_install_name = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib"
    uuid_output = subprocess.check_output(
        ["/usr/bin/dyld_info", "-uuid", blas_install_name], text=True
    )
    uuid_match = re.search(r"[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}", uuid_output)
    architecture_match = re.search(r"\[([^\]]+)\]", uuid_output)
    if uuid_match is None or architecture_match is None:
        raise ValueError("Accelerate libBLAS Mach-O identity is unavailable.")
    return {
        "schemaVersion": 1,
        "files": [
            file_identity("cpython-runtime", "sys.base_prefix/Python", Path(sys.base_prefix).resolve() / "Python"),
            file_identity("torch-python-extension", "torch._C.__file__", Path(torch._C.__file__).resolve()),
            file_identity("torch-python-library", "torch.package/lib/libtorch_python.dylib", torch_root / "lib" / "libtorch_python.dylib"),
            file_identity("torch-cpu-kernel-library", "torch.package/lib/libtorch_cpu.dylib", torch_root / "lib" / "libtorch_cpu.dylib"),
            file_identity("torch-tensor-runtime-library", "torch.package/lib/libc10.dylib", torch_root / "lib" / "libc10.dylib"),
            file_identity("torch-core-library", "torch.package/lib/libtorch.dylib", torch_root / "lib" / "libtorch.dylib"),
            file_identity("torch-shared-memory-library", "torch.package/lib/libshm.dylib", torch_root / "lib" / "libshm.dylib"),
            file_identity("openmp-runtime-library", "torch.package/lib/libomp.dylib", torch_root / "lib" / "libomp.dylib"),
            file_identity("torch-global-dependencies-library", "torch.package/lib/libtorch_global_deps.dylib", torch_root / "lib" / "libtorch_global_deps.dylib"),
        ],
        "pythonSourceTrees": [
            source_tree_identity(
                "cpython-standard-library", "os.__file__/..", Path(os.__file__).resolve().parent,
                (".py", ".pyi"), ("__pycache__", "site-packages"),
            ),
            source_tree_identity(
                "torch-python-package", "torch.__file__/..", torch_root,
                (".py", ".pyi"), ("__pycache__",),
            ),
        ],
        "sharedCacheImages": [
            {
                "role": "accelerate-blas",
                "installName": blas_install_name,
                "architecture": architecture_match.group(1),
                "machoUuid": uuid_match.group(0),
            },
        ],
    }


def runtime_environment_identity(build_config: str) -> dict[str, Any]:
    return {
        "pythonImplementation": platform.python_implementation(),
        "pythonVersion": platform.python_version(),
        "operatingSystem": "macOS" if platform.system() == "Darwin" else platform.system(),
        "operatingSystemVersion": subprocess.check_output(
            ["/usr/bin/sw_vers", "-productVersion"], text=True
        ).strip(),
        "operatingSystemBuild": subprocess.check_output(
            ["/usr/bin/sw_vers", "-buildVersion"], text=True
        ).strip(),
        "kernelRelease": platform.release(),
        "machineModel": subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "hw.model"], text=True
        ).strip(),
        "cpuBrand": subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "torchBuildConfigSha256": hashlib.sha256(build_config.encode("utf-8")).hexdigest(),
        "runtimeDependencyIdentity": runtime_dependency_identity(),
    }


def configure_runtime_execution_state() -> None:
    torch.set_num_threads(SUPPORTED_RUNTIME_EXECUTION_STATE["intraopThreads"])
    torch.set_num_interop_threads(SUPPORTED_RUNTIME_EXECUTION_STATE["interopThreads"])
    torch.use_deterministic_algorithms(
        SUPPORTED_RUNTIME_EXECUTION_STATE["deterministicAlgorithms"],
        warn_only=SUPPORTED_RUNTIME_EXECUTION_STATE["deterministicAlgorithmsWarnOnly"],
    )
    torch.set_float32_matmul_precision(
        SUPPORTED_RUNTIME_EXECUTION_STATE["float32MatmulPrecision"]
    )
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    if not torch.set_flush_denormal(SUPPORTED_RUNTIME_EXECUTION_STATE["flushDenormal"]):
        raise ValueError("Runtime does not support configuring denormal handling.")
    process = ctypes.CDLL(None)
    if process.fesetround(FE_TONEAREST) != 0:
        raise ValueError("Runtime does not support configuring FE_TONEAREST.")
    torch.backends.mkldnn.enabled = SUPPORTED_RUNTIME_EXECUTION_STATE["mkldnnEnabled"]


def subnormal_probe() -> dict[str, Any]:
    minimum_positive = torch.tensor([2 ** -149], dtype=torch.float32)
    multiplied = minimum_positive * torch.tensor([1.0], dtype=torch.float32)
    return {
        "encoding": "ieee-f32-little-endian",
        "inputBits": minimum_positive.view(torch.int32).item(),
        "multipliedByOneBits": multiplied.view(torch.int32).item(),
    }


def c_floating_point_rounding_mode() -> str:
    mode = ctypes.CDLL(None).fegetround()
    if mode != FE_TONEAREST:
        return f"unknown-{mode}"
    return "FE_TONEAREST"


def runtime_execution_state() -> dict[str, Any]:
    return {
        "schemaVersion": 2,
        "intraopThreads": torch.get_num_threads(),
        "interopThreads": torch.get_num_interop_threads(),
        "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministicAlgorithmsWarnOnly": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32MatmulPrecision": torch.get_float32_matmul_precision(),
        "defaultDtype": str(torch.get_default_dtype()),
        "defaultDevice": str(torch.get_default_device()),
        "cpuCapability": torch.backends.cpu.get_cpu_capability(),
        "flushDenormal": False,
        "subnormalProbe": subnormal_probe(),
        "cFloatingPointRoundingMode": c_floating_point_rounding_mode(),
        "mkldnnAvailable": torch.backends.mkldnn.is_available(),
        "mkldnnEnabled": torch.backends.mkldnn.enabled,
    }


def runtime_process_environment() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "inheritance": "none",
        "variables": dict(sorted(RUNTIME_PROCESS_ENVIRONMENT_AT_START.items())),
    }


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
    process_environment = runtime_process_environment()
    if process_environment != SUPPORTED_RUNTIME_PROCESS_ENVIRONMENT:
        raise ValueError(
            "Runtime reduction process environment diverges from the pinned contract: "
            f"expected {SUPPORTED_RUNTIME_PROCESS_ENVIRONMENT!r}; received {process_environment!r}."
        )
    configure_runtime_execution_state()
    execution_state = runtime_execution_state()
    if execution_state != SUPPORTED_RUNTIME_EXECUTION_STATE:
        raise ValueError(
            "Runtime reduction execution state diverges from the pinned contract: "
            f"expected {SUPPORTED_RUNTIME_EXECUTION_STATE!r}; received {execution_state!r}."
        )
    runtime_environment = runtime_environment_identity(build_config)
    if runtime_environment != SUPPORTED_RUNTIME_ENVIRONMENT:
        raise ValueError(
            "Runtime reduction environment identity diverges from the pinned contract: "
            f"expected {SUPPORTED_RUNTIME_ENVIRONMENT!r}; received {runtime_environment!r}."
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
            "runtimeProcessEnvironment": process_environment,
            "runtimeEnvironmentIdentity": runtime_environment,
            "runtimeExecutionState": execution_state,
        },
        "output": {"shape": list(result.shape), "values": result.reshape(-1).tolist()},
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
