#!/usr/bin/env python3
"""Persistent binary F32 tile matmul worker for the paged Gemma runtime."""

import argparse
import math
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
            batched = bool(encoded_dtype & 0x40000000)
            fused_mlp = bool(encoded_dtype & 0x20000000)
            native_bf16 = bool(encoded_dtype & 0x10000000)
            native_attention = bool(encoded_dtype & 0x08000000)
            dtype_code = encoded_dtype & 0x07ffffff
            if native_attention:
                if referenced or batched or fused_mlp or native_bf16 or dtype_code != 0:
                    raise ValueError("native attention flags are invalid")
                batch, query_heads, key_value_heads = rows, outputs, features
                metadata = read_exact(32)
                query_sequence, key_sequence, head_dim, mask_heads, rounding = struct.unpack("<IIIII", metadata[:20])
                scale = struct.unpack("<f", metadata[20:24])[0]
                if not query_sequence or not key_sequence or not head_dim or not mask_heads or query_heads % key_value_heads or mask_heads not in (1, query_heads) or rounding not in (0, 1) or not math.isfinite(scale):
                    raise ValueError("native attention topology is invalid")
                query_bytes = read_exact(batch * query_heads * query_sequence * head_dim * 4)
                key_bytes = read_exact(batch * key_value_heads * key_sequence * head_dim * 4)
                value_bytes = read_exact(batch * key_value_heads * key_sequence * head_dim * 4)
                mask_bytes = read_exact(batch * mask_heads * query_sequence * key_sequence * 4)
                query = torch.frombuffer(query_bytes, dtype=torch.float32).reshape(batch, query_heads, query_sequence, head_dim)
                key = torch.frombuffer(key_bytes, dtype=torch.float32).reshape(batch, key_value_heads, key_sequence, head_dim)
                value = torch.frombuffer(value_bytes, dtype=torch.float32).reshape(batch, key_value_heads, key_sequence, head_dim)
                mask = torch.frombuffer(mask_bytes, dtype=torch.float32).reshape(batch, mask_heads, query_sequence, key_sequence)
                if torch.isnan(mask).any() or torch.isposinf(mask).any():
                    raise ValueError("native attention mask is invalid")
                group = query_heads // key_value_heads
                if group != 1:
                    key = key.repeat_interleave(group, dim=1)
                    value = value.repeat_interleave(group, dim=1)
                if rounding == 0:
                    query, key, value = query.to(torch.bfloat16), key.to(torch.bfloat16), value.to(torch.bfloat16)
                    scores = (torch.matmul(query, key.transpose(-1, -2)).float() * torch.tensor(scale, dtype=torch.float32)).to(torch.bfloat16)
                    scores = (scores + mask.to(torch.bfloat16)).to(torch.bfloat16)
                    probabilities = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
                    context = torch.matmul(probabilities, value).to(torch.bfloat16).float()
                else:
                    scores = torch.matmul(query, key.transpose(-1, -2)) * torch.tensor(scale, dtype=torch.float32) + mask
                    probabilities = torch.softmax(scores, dim=-1)
                    context = torch.matmul(probabilities, value)
                result = context.permute(0, 2, 1, 3).contiguous().reshape(batch * query_sequence, query_heads * head_dim)
                if not torch.isfinite(result).all():
                    raise ValueError("native attention produced non-finite output")
                payload = result.numpy().tobytes(order="C")
                sys.stdout.buffer.write(struct.pack("<I", len(payload)))
                sys.stdout.buffer.write(payload)
                sys.stdout.buffer.flush()
                del query, key, value, mask, scores, probabilities, context, result, payload
                continue
            input_bytes = read_exact(rows * features * 4)
            if fused_mlp:
                if not referenced or batched or native_bf16 or pool is None:
                    raise ValueError("fused MLP requires referenced non-batched storage")
                rounding = struct.unpack("<I", read_exact(4))[0]
                if rounding not in (0, 1):
                    raise ValueError("fused MLP rounding policy is invalid")
                matrices = []
                shapes = []
                for _ in range(3):
                    metadata = read_exact(36)
                    request_dtype, output_count, input_count, byte_offset, byte_length, start_output, shard_length, name_length = struct.unpack("<IIIQIIII", metadata)
                    if request_dtype not in (0, 1, 2) or not output_count or not input_count or start_output != 0:
                        raise ValueError("fused MLP matrix descriptor is invalid")
                    if shard_length < 1 or shard_length > 4096 or name_length < 1 or name_length > 4096:
                        raise ValueError("fused MLP identity length is invalid")
                    shard = bytes(read_exact(shard_length)).decode("utf-8")
                    tensor_name = bytes(read_exact(name_length)).decode("utf-8")
                    if Path(shard).name != shard or not tensor_name:
                        raise ValueError("fused MLP identity is invalid")
                    storage_dtype = (torch.float32, torch.bfloat16, torch.float16)[request_dtype]
                    element_bytes = 4 if request_dtype == 0 else 2
                    if byte_length != output_count * input_count * element_bytes:
                        raise ValueError(f"{tensor_name}: fused MLP byte length is invalid")
                    path = (pool / shard).resolve()
                    if path.parent != pool:
                        raise ValueError("fused MLP escapes binary pool")
                    if path not in mappings:
                        files[path] = path.open("rb")
                        mappings[path] = mmap.mmap(files[path].fileno(), 0, access=mmap.ACCESS_READ)
                    if byte_offset + byte_length > mappings[path].size():
                        raise ValueError("fused MLP range exceeds shard")
                    matrices.append(torch.frombuffer(mappings[path], dtype=storage_dtype, count=output_count * input_count, offset=byte_offset).reshape(output_count, input_count).float())
                    shapes.append((output_count, input_count))
                if shapes[0] != shapes[1] or shapes[0][1] != features or shapes[2] != (outputs, shapes[0][0]):
                    raise ValueError("fused MLP matrix shapes are incompatible")
                inputs = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(rows, features)
                boundary = (lambda value: value.to(torch.bfloat16).float()) if rounding == 0 else (lambda value: value)
                gate = boundary(torch.mm(inputs, matrices[0].transpose(0, 1)))
                up = boundary(torch.mm(inputs, matrices[1].transpose(0, 1)))
                cube = (gate * gate) * gate
                inner = torch.tensor(math.sqrt(2 / math.pi), dtype=torch.float32) * (gate + torch.tensor(0.044715, dtype=torch.float32) * cube)
                activated = boundary((torch.tensor(0.5, dtype=torch.float32) * gate) * (torch.tensor(1.0, dtype=torch.float32) + torch.tanh(inner)))
                hidden = boundary(activated * up)
                result = boundary(torch.mm(hidden, matrices[2].transpose(0, 1))).contiguous()
                payload = result.numpy().tobytes(order="C")
                sys.stdout.buffer.write(struct.pack("<I", len(payload)))
                sys.stdout.buffer.write(payload)
                sys.stdout.buffer.flush()
                del inputs, matrices, shapes, gate, up, cube, inner, activated, hidden, result, payload
                continue
            if batched:
                if not referenced or native_bf16 or pool is None or outputs < 2 or outputs > 256:
                    raise ValueError("batched tile requires 2..256 referenced projections")
                inputs = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(rows, features)
                results = []
                for _ in range(outputs):
                    metadata = read_exact(32)
                    request_dtype, output_count, byte_offset, byte_length, start_output, shard_length, name_length = struct.unpack("<IIQIIII", metadata)
                    if request_dtype not in (0, 1, 2) or not output_count:
                        raise ValueError("batched tile request is invalid")
                    if shard_length < 1 or shard_length > 4096 or name_length < 1 or name_length > 4096:
                        raise ValueError("batched tile identity length is invalid")
                    shard = bytes(read_exact(shard_length)).decode("utf-8")
                    tensor_name = bytes(read_exact(name_length)).decode("utf-8")
                    if Path(shard).name != shard or not tensor_name:
                        raise ValueError("batched tile identity is invalid")
                    storage_dtype = (torch.float32, torch.bfloat16, torch.float16)[request_dtype]
                    element_bytes = 4 if request_dtype == 0 else 2
                    if byte_length != output_count * features * element_bytes:
                        raise ValueError(f"{tensor_name}: batched tile byte length is invalid")
                    path = (pool / shard).resolve()
                    if path.parent != pool:
                        raise ValueError("batched tile escapes binary pool")
                    if path not in mappings:
                        files[path] = path.open("rb")
                        mappings[path] = mmap.mmap(files[path].fileno(), 0, access=mmap.ACCESS_READ)
                    if byte_offset + byte_length > mappings[path].size():
                        raise ValueError("batched tile range exceeds shard")
                    weights = torch.frombuffer(mappings[path], dtype=storage_dtype, count=output_count * features, offset=byte_offset).reshape(output_count, features).float()
                    results.append(torch.mm(inputs, weights.transpose(0, 1)).contiguous())
                    del weights
                for result in results:
                    payload = result.numpy().tobytes(order="C")
                    sys.stdout.buffer.write(struct.pack("<I", len(payload)))
                    sys.stdout.buffer.write(payload)
                sys.stdout.buffer.flush()
                del inputs, results, result, payload
                continue
            if dtype_code not in (0, 1, 2):
                raise ValueError(f"unknown storage dtype code {dtype_code}")
            storage_dtype = (torch.float32, torch.bfloat16, torch.float16)[dtype_code]
            if native_bf16 and (not referenced or dtype_code != 1):
                raise ValueError("native BF16 GEMM requires referenced BF16 storage")
            element_bytes = 4 if dtype_code == 0 else 2
            expected_weight_bytes = outputs * features * element_bytes
            if referenced:
                if pool is None:
                    raise ValueError("referenced tile requires --binary-pool")
                metadata = read_exact(24)
                byte_offset, byte_length, start_output, shard_length, name_length = struct.unpack("<QIIII", metadata)
                if shard_length < 1 or shard_length > 4096 or name_length < 1 or name_length > 4096:
                    raise ValueError("referenced tile identity length is invalid")
                shard = bytes(read_exact(shard_length)).decode("utf-8")
                tensor_name = bytes(read_exact(name_length)).decode("utf-8")
                if not shard or Path(shard).name != shard:
                    raise ValueError("referenced tile shard must be a basename")
                if not tensor_name:
                    raise ValueError("referenced tile tensor name is empty")
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
                weights = torch.frombuffer(mappings[path], dtype=storage_dtype, count=outputs * features, offset=byte_offset).reshape(outputs, features)
                if not native_bf16:
                    weights = weights.float()
            else:
                weight_bytes = read_exact(expected_weight_bytes)
                if weight_bytes is None:
                    raise EOFError("truncated tile payload")
                weights = torch.frombuffer(weight_bytes, dtype=storage_dtype).reshape(outputs, features).float()
            if input_bytes is None:
                raise EOFError("truncated tile input")
            inputs = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(rows, features)
            if native_bf16:
                inputs = inputs.to(torch.bfloat16)
            result = torch.mm(inputs, weights.transpose(0, 1)).contiguous()
            if native_bf16:
                result = result.float().contiguous()
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
