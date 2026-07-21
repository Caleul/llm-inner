#!/usr/bin/env python3
"""Persistent binary F32 tile matmul worker for the paged Gemma runtime."""

import argparse
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
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    while True:
        header = read_exact(16)
        if header is None:
            return
        rows, outputs, features, dtype_code = struct.unpack("<IIII", header)
        if not rows or not outputs or not features:
            raise ValueError("tile dimensions must be positive")
        input_bytes = read_exact(rows * features * 4)
        if dtype_code not in (0, 1, 2):
            raise ValueError(f"unknown storage dtype code {dtype_code}")
        storage_dtype = (torch.float32, torch.bfloat16, torch.float16)[dtype_code]
        element_bytes = 4 if dtype_code == 0 else 2
        weight_bytes = read_exact(outputs * features * element_bytes)
        if input_bytes is None or weight_bytes is None:
            raise EOFError("truncated tile payload")
        inputs = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(rows, features)
        weights = torch.frombuffer(weight_bytes, dtype=storage_dtype).reshape(outputs, features).float()
        result = torch.mm(inputs, weights.transpose(0, 1)).contiguous()
        payload = result.numpy().tobytes(order="C")
        sys.stdout.buffer.write(struct.pack("<I", len(payload)))
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
