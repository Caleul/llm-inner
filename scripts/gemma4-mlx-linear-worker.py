#!/usr/bin/env python3
"""Persistent MLX Metal worker for F32 inputs and dense F32/F16/BF16 weights."""

import argparse
import math
from pathlib import Path
import struct
import sys

import mlx.core as mx
import numpy as np


def read_exact(size):
    chunks = bytearray(size)
    view = memoryview(chunks)
    offset = 0
    while offset < size:
        count = sys.stdin.buffer.readinto(view[offset:])
        if not count:
            if offset == 0:
                return None
            raise EOFError(f"truncated tile: {offset}/{size}")
        offset += count
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--binary-pool", required=True)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    pool = Path(args.binary_pool).resolve()
    if not pool.is_dir():
        raise ValueError("--binary-pool must be a directory")
    shards = {}
    while True:
        header = read_exact(16)
        if header is None:
            return
        rows, outputs, features, encoded_dtype = struct.unpack("<IIII", header)
        referenced = bool(encoded_dtype & 0x80000000)
        if not rows or not outputs or not features or not referenced:
            raise ValueError("MLX worker requires a positive referenced tile")
        batched = bool(encoded_dtype & 0x40000000)
        fused_mlp = bool(encoded_dtype & 0x20000000)
        native_bf16 = bool(encoded_dtype & 0x10000000)
        dtype_code = encoded_dtype & 0x0fffffff
        input_bytes = read_exact(rows * features * 4)
        if native_bf16:
            raise ValueError("native BF16 GEMM is available only in the PyTorch worker")
        if fused_mlp:
            if batched:
                raise ValueError("fused MLP cannot also be batched")
            rounding = struct.unpack("<I", read_exact(4))[0]
            if rounding not in (0, 1):
                raise ValueError("fused MLP rounding policy is invalid")
            matrices = []
            shapes = []
            for _ in range(3):
                metadata = read_exact(36)
                request_dtype, output_count, input_count, _, byte_length, start_output, shard_length, name_length = struct.unpack("<IIIQIIII", metadata)
                if request_dtype not in (0, 1, 2) or not output_count or not input_count or start_output != 0:
                    raise ValueError("fused MLP matrix descriptor is invalid")
                if shard_length < 1 or shard_length > 4096 or name_length < 1 or name_length > 4096:
                    raise ValueError("fused MLP identity length is invalid")
                shard = bytes(read_exact(shard_length)).decode("utf-8")
                tensor_name = bytes(read_exact(name_length)).decode("utf-8")
                if Path(shard).name != shard or not tensor_name:
                    raise ValueError("fused MLP identity is invalid")
                path = (pool / shard).resolve()
                if path.parent != pool:
                    raise ValueError("fused MLP escapes binary pool")
                if path not in shards:
                    shards[path] = mx.load(str(path))
                weight = shards[path].get(tensor_name)
                expected_dtype = (mx.float32, mx.bfloat16, mx.float16)[request_dtype]
                element_bytes = 4 if request_dtype == 0 else 2
                if weight is None or weight.shape != (output_count, input_count) or weight.dtype != expected_dtype or byte_length != output_count * input_count * element_bytes:
                    raise ValueError(f"{tensor_name}: fused MLP tensor identity diverges")
                matrices.append(weight)
                shapes.append((output_count, input_count))
            if shapes[0] != shapes[1] or shapes[0][1] != features or shapes[2] != (outputs, shapes[0][0]):
                raise ValueError("fused MLP matrix shapes are incompatible")
            inputs = mx.array(np.frombuffer(input_bytes, dtype=np.float32).reshape(rows, features))
            boundary = (lambda value: value.astype(mx.bfloat16).astype(mx.float32)) if rounding == 0 else (lambda value: value)
            gate = boundary(mx.matmul(inputs, matrices[0].T))
            up = boundary(mx.matmul(inputs, matrices[1].T))
            cube = (gate * gate) * gate
            inner = mx.array(math.sqrt(2 / math.pi), dtype=mx.float32) * (gate + mx.array(0.044715, dtype=mx.float32) * cube)
            activated = boundary((mx.array(0.5, dtype=mx.float32) * gate) * (mx.array(1.0, dtype=mx.float32) + mx.tanh(inner)))
            hidden = boundary(activated * up)
            result = boundary(mx.matmul(hidden, matrices[2].T))
            mx.eval(result)
            payload = np.asarray(result, dtype=np.float32).tobytes(order="C")
            sys.stdout.buffer.write(struct.pack("<I", len(payload)))
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
            del inputs, matrices, shapes, gate, up, cube, inner, activated, hidden, result, payload
            continue
        if batched:
            if outputs < 2 or outputs > 256:
                raise ValueError("batched tile requires 2..256 projections")
            inputs = mx.array(np.frombuffer(input_bytes, dtype=np.float32).reshape(rows, features))
            results = []
            for _ in range(outputs):
                metadata = read_exact(32)
                request_dtype, output_count, _, byte_length, start_output, shard_length, name_length = struct.unpack("<IIQIIII", metadata)
                if request_dtype not in (0, 1, 2) or not output_count:
                    raise ValueError("batched tile request is invalid")
                if shard_length < 1 or shard_length > 4096 or name_length < 1 or name_length > 4096:
                    raise ValueError("batched tile identity length is invalid")
                shard = bytes(read_exact(shard_length)).decode("utf-8")
                tensor_name = bytes(read_exact(name_length)).decode("utf-8")
                if Path(shard).name != shard or not tensor_name:
                    raise ValueError("batched tile identity is invalid")
                path = (pool / shard).resolve()
                if path.parent != pool:
                    raise ValueError("batched tile escapes binary pool")
                if path not in shards:
                    shards[path] = mx.load(str(path))
                weight = shards[path].get(tensor_name)
                expected_dtype = (mx.float32, mx.bfloat16, mx.float16)[request_dtype]
                element_bytes = 4 if request_dtype == 0 else 2
                if weight is None or weight.ndim != 2 or weight.shape[1] != features or weight.dtype != expected_dtype:
                    raise ValueError(f"{tensor_name}: MLX tensor identity diverges from batch request")
                if start_output + output_count > weight.shape[0] or byte_length != output_count * features * element_bytes:
                    raise ValueError(f"{tensor_name}: MLX batch range diverges from request")
                results.append(mx.matmul(inputs, weight[start_output:start_output + output_count].T))
            mx.eval(*results)
            for result in results:
                payload = np.asarray(result, dtype=np.float32).tobytes(order="C")
                sys.stdout.buffer.write(struct.pack("<I", len(payload)))
                sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
            del inputs, results, result, payload
            continue
        if dtype_code not in (0, 1, 2):
            raise ValueError(f"unknown storage dtype code {dtype_code}")
        metadata = read_exact(24)
        _, byte_length, start_output, shard_length, name_length = struct.unpack("<QIIII", metadata)
        if shard_length < 1 or shard_length > 4096 or name_length < 1 or name_length > 4096:
            raise ValueError("referenced tile identity length is invalid")
        shard = bytes(read_exact(shard_length)).decode("utf-8")
        tensor_name = bytes(read_exact(name_length)).decode("utf-8")
        if Path(shard).name != shard or not tensor_name:
            raise ValueError("referenced tile identity is invalid")
        path = (pool / shard).resolve()
        if path.parent != pool:
            raise ValueError("referenced tile escapes binary pool")
        if path not in shards:
            shards[path] = mx.load(str(path))
        weight = shards[path].get(tensor_name)
        expected_dtype = (mx.float32, mx.bfloat16, mx.float16)[dtype_code]
        element_bytes = 4 if dtype_code == 0 else 2
        if weight is None or weight.ndim != 2 or weight.shape[1] != features or weight.dtype != expected_dtype:
            raise ValueError(f"{tensor_name}: MLX tensor identity diverges from request")
        if start_output + outputs > weight.shape[0] or byte_length != outputs * features * element_bytes:
            raise ValueError(f"{tensor_name}: MLX tile range diverges from request")
        inputs = mx.array(np.frombuffer(input_bytes, dtype=np.float32).reshape(rows, features))
        result = mx.matmul(inputs, weight[start_output:start_output + outputs].T)
        mx.eval(result)
        payload = np.asarray(result, dtype=np.float32).tobytes(order="C")
        sys.stdout.buffer.write(struct.pack("<I", len(payload)))
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
        del inputs, result, payload


if __name__ == "__main__":
    main()
