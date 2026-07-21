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


def read_whole_tensor(pool, files, mappings, expected_shape, widen=True):
    metadata = read_exact(36)
    request_dtype, first, second, byte_offset, byte_length, start_output, shard_length, name_length = struct.unpack("<IIIQIIII", metadata)
    if request_dtype not in (0, 1, 2) or (first, second) != expected_shape or start_output != 0:
        raise ValueError("fused attention tensor descriptor is invalid")
    if shard_length < 1 or shard_length > 4096 or name_length < 1 or name_length > 4096:
        raise ValueError("fused attention tensor identity length is invalid")
    shard = bytes(read_exact(shard_length)).decode("utf-8")
    tensor_name = bytes(read_exact(name_length)).decode("utf-8")
    if Path(shard).name != shard or not tensor_name:
        raise ValueError("fused attention tensor identity is invalid")
    storage_dtype = (torch.float32, torch.bfloat16, torch.float16)[request_dtype]
    element_bytes = 4 if request_dtype == 0 else 2
    if byte_length != first * second * element_bytes:
        raise ValueError(f"{tensor_name}: fused attention tensor byte length is invalid")
    path = (pool / shard).resolve()
    if path.parent != pool:
        raise ValueError("fused attention tensor escapes binary pool")
    if path not in mappings:
        files[path] = path.open("rb")
        mappings[path] = mmap.mmap(files[path].fileno(), 0, access=mmap.ACCESS_READ)
    if byte_offset + byte_length > mappings[path].size():
        raise ValueError("fused attention tensor range exceeds shard")
    tensor = torch.frombuffer(mappings[path], dtype=storage_dtype, count=first * second, offset=byte_offset).reshape(first, second)
    return tensor.float() if widen else tensor


def rms_norm_real(tensor, weight, epsilon):
    scale = torch.rsqrt(torch.mean(tensor * tensor, dim=-1, keepdim=True) + torch.tensor(epsilon, dtype=torch.float32))
    normalized = tensor * scale
    return normalized if weight is None else normalized * weight


def rope_real(tensor, positions, rope_kind, theta, rotary_dim, proportional_pairs, proportional_factor, bf16_boundaries=False):
    head_dim = tensor.shape[-1]
    half = rotary_dim // 2
    pairs = torch.arange(half, dtype=torch.float32)
    denominator_width = head_dim if rope_kind == 1 else rotary_dim
    denominator = torch.pow(torch.tensor(theta, dtype=torch.float32), (torch.tensor(2.0, dtype=torch.float32) * pairs) / torch.tensor(denominator_width, dtype=torch.float32))
    if rope_kind == 1:
        denominator = denominator * torch.tensor(proportional_factor, dtype=torch.float32)
    angles = positions[:, None, :, None].float() / denominator[None, None, None, :]
    if rope_kind == 1 and proportional_pairs < half:
        active = (pairs < proportional_pairs)[None, None, None, :]
        angles = torch.where(active, angles, torch.zeros_like(angles))
    cosine, sine = torch.cos(angles), torch.sin(angles)
    if bf16_boundaries:
        cosine, sine = cosine.to(torch.bfloat16).float(), sine.to(torch.bfloat16).float()
    result = tensor.clone()
    first, second = tensor[..., :half], tensor[..., half:rotary_dim]
    direct_first, rotated_first = first * cosine, second * sine
    direct_second, rotated_second = second * cosine, first * sine
    if bf16_boundaries:
        direct_first, rotated_first = direct_first.to(torch.bfloat16).float(), rotated_first.to(torch.bfloat16).float()
        direct_second, rotated_second = direct_second.to(torch.bfloat16).float(), rotated_second.to(torch.bfloat16).float()
        result[..., :half] = (direct_first - rotated_first).to(torch.bfloat16).float()
        result[..., half:rotary_dim] = (direct_second + rotated_second).to(torch.bfloat16).float()
    else:
        result[..., :half] = direct_first - rotated_first
        result[..., half:rotary_dim] = direct_second + rotated_second
    return result


def write_float_tensor(tensor):
    payload = tensor.contiguous().numpy().tobytes(order="C")
    sys.stdout.buffer.write(struct.pack("<I", len(payload)))
    sys.stdout.buffer.write(payload)


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
            fused_attention = bool(encoded_dtype & 0x04000000)
            fused_ple = bool(encoded_dtype & 0x02000000)
            fused_ple_prelude = bool(encoded_dtype & 0x01000000)
            dtype_code = encoded_dtype & 0x00ffffff
            if fused_attention:
                if referenced or batched or fused_mlp or native_bf16 or native_attention or fused_ple or fused_ple_prelude or dtype_code != 0 or pool is None:
                    raise ValueError("fused attention flags are invalid")
                batch, query_heads, key_value_heads = rows, outputs, features
                metadata = read_exact(80)
                query_sequence, source_sequence, hidden_size, head_dim, mask_heads, produces_kv, value_from_key, rope_kind, rotary_dim, proportional_pairs, descriptor_count = struct.unpack("<IIIIIIIIIII", metadata[:44])
                epsilon, scale = struct.unpack("<ff", metadata[44:52])
                theta = struct.unpack("<d", metadata[52:60])[0]
                proportional_factor = struct.unpack("<f", metadata[60:64])[0]
                rounding = struct.unpack("<I", metadata[64:68])[0]
                total_key_sequence = source_sequence + (query_sequence if produces_kv else 0)
                expected_descriptors = 3 + (2 + (0 if value_from_key else 1) if produces_kv else 0)
                if not batch or not query_heads or not key_value_heads or not query_sequence or not hidden_size or not head_dim or not mask_heads or total_key_sequence < 1 or query_heads % key_value_heads or mask_heads not in (1, query_heads) or produces_kv not in (0, 1) or value_from_key not in (0, 1) or (not produces_kv and value_from_key) or rope_kind not in (0, 1) or rounding not in (0, 1) or not rotary_dim or rotary_dim > head_dim or rotary_dim % 2 or proportional_pairs > rotary_dim // 2 or descriptor_count != expected_descriptors or not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(scale) or not math.isfinite(theta) or theta <= 0 or not math.isfinite(proportional_factor) or proportional_factor <= 0:
                    raise ValueError("fused attention topology is invalid")
                if not produces_kv and source_sequence < query_sequence:
                    raise ValueError("shared fused attention cache is shorter than query")
                input_bytes = read_exact(batch * query_sequence * hidden_size * 4)
                position_bytes = read_exact(batch * query_sequence * 4)
                mask_bytes = read_exact(batch * mask_heads * query_sequence * total_key_sequence * 4)
                source_elements = batch * key_value_heads * source_sequence * head_dim
                source_key_bytes = read_exact(source_elements * 4)
                source_value_bytes = read_exact(source_elements * 4)
                inputs = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(batch, query_sequence, hidden_size)
                positions = torch.frombuffer(position_bytes, dtype=torch.int32).reshape(batch, query_sequence)
                mask = torch.frombuffer(mask_bytes, dtype=torch.float32).reshape(batch, mask_heads, query_sequence, total_key_sequence)
                if torch.isnan(mask).any() or torch.isposinf(mask).any():
                    raise ValueError("fused attention mask is invalid")
                if source_sequence:
                    source_key = torch.frombuffer(source_key_bytes, dtype=torch.float32).reshape(batch, key_value_heads, source_sequence, head_dim)
                    source_value = torch.frombuffer(source_value_bytes, dtype=torch.float32).reshape(batch, key_value_heads, source_sequence, head_dim)
                else:
                    source_key = torch.empty((batch, key_value_heads, 0, head_dim), dtype=torch.float32)
                    source_value = torch.empty((batch, key_value_heads, 0, head_dim), dtype=torch.float32)
                query_weight = read_whole_tensor(pool, files, mappings, (query_heads * head_dim, hidden_size))
                query_norm = read_whole_tensor(pool, files, mappings, (head_dim, 1)).reshape(head_dim)
                output_weight = read_whole_tensor(pool, files, mappings, (hidden_size, query_heads * head_dim))
                boundary = (lambda tensor: tensor.to(torch.bfloat16).float()) if rounding == 0 else (lambda tensor: tensor)
                query = boundary(torch.matmul(inputs, query_weight.transpose(0, 1))).reshape(batch, query_sequence, query_heads, head_dim).permute(0, 2, 1, 3)
                query = rope_real(boundary(rms_norm_real(query, query_norm, epsilon)), positions, rope_kind, theta, rotary_dim, proportional_pairs, proportional_factor, rounding == 0)
                key_weight = key_norm = value_weight = current_key_heads = current_key = current_value = None
                if produces_kv:
                    key_weight = read_whole_tensor(pool, files, mappings, (key_value_heads * head_dim, hidden_size))
                    key_norm = read_whole_tensor(pool, files, mappings, (head_dim, 1)).reshape(head_dim)
                    current_key_heads = boundary(torch.matmul(inputs, key_weight.transpose(0, 1))).reshape(batch, query_sequence, key_value_heads, head_dim).permute(0, 2, 1, 3)
                    current_key = rope_real(boundary(rms_norm_real(current_key_heads, key_norm, epsilon)), positions, rope_kind, theta, rotary_dim, proportional_pairs, proportional_factor, rounding == 0)
                    if value_from_key:
                        current_value = boundary(rms_norm_real(current_key_heads, None, epsilon))
                    else:
                        value_weight = read_whole_tensor(pool, files, mappings, (key_value_heads * head_dim, hidden_size))
                        current_value = boundary(torch.matmul(inputs, value_weight.transpose(0, 1))).reshape(batch, query_sequence, key_value_heads, head_dim).permute(0, 2, 1, 3)
                        current_value = boundary(rms_norm_real(current_value, None, epsilon))
                    key = torch.cat((source_key, current_key), dim=2)
                    value = torch.cat((source_value, current_value), dim=2)
                else:
                    key, value = source_key, source_value
                group = query_heads // key_value_heads
                attention_key = key.repeat_interleave(group, dim=1) if group != 1 else key
                attention_value = value.repeat_interleave(group, dim=1) if group != 1 else value
                scores = torch.matmul(query, attention_key.transpose(-1, -2)) * torch.tensor(scale, dtype=torch.float32) + mask
                probabilities = torch.softmax(scores, dim=-1)
                context = torch.matmul(probabilities, attention_value).permute(0, 2, 1, 3).contiguous().reshape(batch, query_sequence, query_heads * head_dim)
                projected = boundary(torch.matmul(context, output_weight.transpose(0, 1)))
                if not torch.isfinite(projected).all() or not torch.isfinite(key).all() or not torch.isfinite(value).all():
                    raise ValueError("fused attention produced non-finite output")
                write_float_tensor(projected)
                if produces_kv:
                    write_float_tensor(key)
                    write_float_tensor(value)
                sys.stdout.buffer.flush()
                del inputs, positions, mask, source_key, source_value, query_weight, query_norm, output_weight, query, key_weight, key_norm, value_weight, current_key_heads, current_key, current_value, key, value, attention_key, attention_value, scores, probabilities, context, projected, boundary
                continue
            if fused_ple:
                if referenced or batched or fused_mlp or native_bf16 or native_attention or fused_attention or fused_ple_prelude or dtype_code != 0 or pool is None or outputs != features:
                    raise ValueError("fused PLE flags are invalid")
                per_layer_width, descriptor_count, rounding = struct.unpack("<III", read_exact(12))
                epsilon = struct.unpack("<f", read_exact(4))[0]
                if not per_layer_width or descriptor_count != 4 or rounding not in (0, 1) or not math.isfinite(epsilon) or epsilon <= 0:
                    raise ValueError("fused PLE metadata is invalid")
                input_bytes = read_exact(rows * features * 4)
                per_layer_bytes = read_exact(rows * per_layer_width * 4)
                inputs = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(rows, features)
                per_layer = torch.frombuffer(per_layer_bytes, dtype=torch.float32).reshape(rows, per_layer_width)
                gate_weight = read_whole_tensor(pool, files, mappings, (per_layer_width, features))
                projection_weight = read_whole_tensor(pool, files, mappings, (features, per_layer_width))
                norm_weight = read_whole_tensor(pool, files, mappings, (features, 1)).reshape(features)
                layer_scalar = read_whole_tensor(pool, files, mappings, (1, 1)).reshape(())
                boundary = (lambda value: value.to(torch.bfloat16).float()) if rounding == 0 else (lambda value: value)
                gate = boundary(torch.mm(inputs, gate_weight.transpose(0, 1)))
                cube = (gate * gate) * gate
                inner = torch.tensor(math.sqrt(2 / math.pi), dtype=torch.float32) * (gate + torch.tensor(0.044715, dtype=torch.float32) * cube)
                activated = boundary((torch.tensor(0.5, dtype=torch.float32) * gate) * (torch.tensor(1.0, dtype=torch.float32) + torch.tanh(inner)))
                gated = boundary(activated * per_layer)
                projected = boundary(torch.mm(gated, projection_weight.transpose(0, 1)))
                normalized = boundary(rms_norm_real(projected, norm_weight, epsilon))
                residual = boundary(inputs + normalized)
                result = boundary(residual * layer_scalar).contiguous()
                if not torch.isfinite(result).all():
                    raise ValueError("fused PLE produced non-finite output")
                write_float_tensor(result)
                sys.stdout.buffer.flush()
                del inputs, per_layer, gate_weight, projection_weight, norm_weight, layer_scalar, boundary, gate, cube, inner, activated, gated, projected, normalized, residual, result
                continue
            if fused_ple_prelude:
                if referenced or batched or fused_mlp or native_bf16 or native_attention or fused_attention or fused_ple or dtype_code != 0 or pool is None:
                    raise ValueError("fused PLE prelude flags are invalid")
                metadata = read_exact(32)
                num_layers, per_layer_width, descriptor_count, rounding = struct.unpack("<IIII", metadata[:16])
                context_scale, combine_scale, epsilon = struct.unpack("<fff", metadata[16:28])
                tile_output_rows = struct.unpack("<I", metadata[28:32])[0]
                if not num_layers or not per_layer_width or outputs != num_layers * per_layer_width or descriptor_count != 2 or rounding not in (0, 1) or not tile_output_rows or not math.isfinite(context_scale) or not math.isfinite(combine_scale) or not math.isfinite(epsilon) or epsilon <= 0:
                    raise ValueError("fused PLE prelude metadata is invalid")
                input_bytes = read_exact(rows * features * 4)
                token_bytes = read_exact(rows * outputs * 4)
                inputs = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(rows, features)
                token_identity = torch.frombuffer(token_bytes, dtype=torch.float32).reshape(rows, num_layers, per_layer_width)
                projection_weight = read_whole_tensor(pool, files, mappings, (outputs, features), widen=False)
                norm_weight = read_whole_tensor(pool, files, mappings, (per_layer_width, 1)).reshape(per_layer_width)
                boundary = (lambda value: value.to(torch.bfloat16).float()) if rounding == 0 else (lambda value: value)
                context = torch.cat([torch.mm(inputs, projection_weight[start:start + tile_output_rows].float().transpose(0, 1)) for start in range(0, outputs, tile_output_rows)], dim=1)
                context = boundary(context)
                context = boundary(context * torch.tensor(context_scale, dtype=torch.float32)).reshape(rows, num_layers, per_layer_width)
                normalized = boundary(rms_norm_real(context, norm_weight, epsilon))
                combined = boundary(normalized + token_identity)
                result = boundary(combined * torch.tensor(combine_scale, dtype=torch.float32)).reshape(rows, outputs).contiguous()
                if not torch.isfinite(result).all():
                    raise ValueError("fused PLE prelude produced non-finite output")
                write_float_tensor(result)
                sys.stdout.buffer.flush()
                del inputs, token_identity, projection_weight, norm_weight, boundary, context, normalized, combined, result
                continue
            if native_attention:
                if referenced or batched or fused_mlp or native_bf16 or fused_attention or fused_ple or fused_ple_prelude or dtype_code != 0:
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
                if rounding not in (0, 1, 2):
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
                    matrix = torch.frombuffer(mappings[path], dtype=storage_dtype, count=output_count * input_count, offset=byte_offset).reshape(output_count, input_count)
                    if rounding == 2 and request_dtype != 1:
                        raise ValueError("native BF16 fused MLP requires BF16 storage")
                    matrices.append(matrix if rounding == 2 else matrix.float())
                    shapes.append((output_count, input_count))
                if shapes[0] != shapes[1] or shapes[0][1] != features or shapes[2] != (outputs, shapes[0][0]):
                    raise ValueError("fused MLP matrix shapes are incompatible")
                inputs = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(rows, features)
                if rounding == 2:
                    native_inputs = inputs.to(torch.bfloat16)
                    gate = torch.mm(native_inputs, matrices[0].transpose(0, 1))
                    up = torch.mm(native_inputs, matrices[1].transpose(0, 1))
                    activated = torch.nn.functional.gelu(gate, approximate="tanh")
                    hidden = activated * up
                    result = torch.mm(hidden, matrices[2].transpose(0, 1)).float().contiguous()
                else:
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
                if rounding == 2:
                    del native_inputs
                else:
                    del boundary, cube, inner
                del inputs, matrices, shapes, gate, up, activated, hidden, result, payload
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
