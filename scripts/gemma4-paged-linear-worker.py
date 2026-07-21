#!/usr/bin/env python3
"""Persistent binary F32 tile matmul worker for the paged Gemma runtime."""

import argparse
import mmap
from pathlib import Path
import struct
import sys
import torch


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
    parser.add_argument("--binary-pool")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    pool = Path(args.binary_pool).resolve() if args.binary_pool else None
    if pool is not None and not pool.is_dir():
        raise ValueError("--binary-pool must be a directory")
    files, mappings = {}, {}
    try:
        while True:
            header = read_exact(16)
            if header is None:
                return
            rows, outputs, features, encoded_dtype = struct.unpack("<IIII", header)
            if not rows or not outputs or not features:
                raise ValueError("tile dimensions must be positive")
            referenced = bool(encoded_dtype & 0x80000000)
            dtype_code = encoded_dtype & 0x7fffffff
            input_bytes = read_exact(rows * features * 4)
            if dtype_code not in (0, 1, 2):
                raise ValueError(f"unknown storage dtype code {dtype_code}")
            storage_dtype = (torch.float32, torch.bfloat16, torch.float16)[dtype_code]
            element_bytes = 4 if dtype_code == 0 else 2
            expected_weight_bytes = outputs * features * element_bytes
            if referenced:
                if pool is None:
                    raise ValueError("referenced tile requires --binary-pool")
                metadata = read_exact(16)
                byte_offset, byte_length, shard_length = struct.unpack("<QII", metadata)
                if shard_length < 1 or shard_length > 4096:
                    raise ValueError("referenced tile shard length is invalid")
                shard = bytes(read_exact(shard_length)).decode("utf-8")
                if not shard or Path(shard).name != shard:
                    raise ValueError("referenced tile shard must be a basename")
                if byte_length != expected_weight_bytes:
                    raise ValueError(f"referenced tile has {byte_length} bytes; expected {expected_weight_bytes}")
                path = (pool / shard).resolve()
                if path.parent != pool:
                    raise ValueError("referenced tile escapes binary pool")
                if path not in mappings:
                    files[path] = path.open("rb")
                    mappings[path] = mmap.mmap(files[path].fileno(), 0, access=mmap.ACCESS_READ)
                if byte_offset + byte_length > mappings[path].size():
                    raise ValueError("referenced tile range exceeds shard")
                weights = torch.frombuffer(mappings[path], dtype=storage_dtype, count=outputs * features, offset=byte_offset).reshape(outputs, features).float()
            else:
                weight_bytes = read_exact(expected_weight_bytes)
                if weight_bytes is None:
                    raise EOFError("truncated tile payload")
                weights = torch.frombuffer(weight_bytes, dtype=storage_dtype).reshape(outputs, features).float()
            if input_bytes is None:
                raise EOFError("truncated tile input")
            inputs = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(rows, features)
            result = torch.mm(inputs, weights.transpose(0, 1)).contiguous()
            payload = result.numpy().tobytes(order="C")
            sys.stdout.buffer.write(struct.pack("<I", len(payload)))
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
            del inputs, weights, result, payload
    finally:
        for mapping in mappings.values():
            mapping.close()
        for file in files.values():
            file.close()


if __name__ == "__main__":
    main()
