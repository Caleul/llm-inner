#!/usr/bin/env python3
"""Persistent MLX Metal worker for F32 inputs and dense F32/F16/BF16 weights."""

import argparse
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
        dtype_code = encoded_dtype & 0x7fffffff
        if not rows or not outputs or not features or not referenced:
            raise ValueError("MLX worker requires a positive referenced tile")
        if dtype_code not in (0, 1, 2):
            raise ValueError(f"unknown storage dtype code {dtype_code}")
        input_bytes = read_exact(rows * features * 4)
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
