#!/usr/bin/env python3
"""Persistent binary F32 tile matmul worker for the paged Gemma runtime."""

import argparse
import math
import mmap
from pathlib import Path
import struct
import sys
import time
import torch


_widened_tensor_cache = {}
_widened_tensor_cache_hits = 0
_widened_tensor_cache_bytes = 0
_MAX_WIDENED_CACHE_SOURCE_BYTES = 2 * 1024 * 1024


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


def read_whole_tensor(pool, files, mappings, expected_shape, widen=True, required_dtype=None, return_reference=False):
    global _widened_tensor_cache_hits, _widened_tensor_cache_bytes
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
    if required_dtype is not None and storage_dtype != required_dtype:
        raise ValueError(f"{tensor_name}: fused tensor storage dtype is incompatible")
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
    if widen and storage_dtype != torch.float32 and byte_length <= _MAX_WIDENED_CACHE_SOURCE_BYTES:
        cache_key = (path, byte_offset, byte_length, first, second, storage_dtype)
        result = _widened_tensor_cache.get(cache_key)
        if result is None:
            result = tensor.float().contiguous()
            _widened_tensor_cache[cache_key] = result
            _widened_tensor_cache_bytes += result.numel() * result.element_size()
        else:
            _widened_tensor_cache_hits += 1
    else:
        result = tensor.float() if widen else tensor
    return (result, (path, byte_offset, byte_length, storage_dtype)) if return_reference else result


def rms_norm_real(tensor, weight, epsilon):
    mean_squared = torch.mean(tensor * tensor, dim=-1, keepdim=True) + torch.tensor(epsilon, dtype=torch.float32)
    scale = torch.pow(mean_squared, torch.tensor(-0.5, dtype=torch.float32))
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


def eager_bf16_attention(query, key, value, mask, scale):
    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    value = value.to(torch.bfloat16)
    scores = (torch.matmul(query, key.transpose(-1, -2)).float() * torch.tensor(scale, dtype=torch.float32)).to(torch.bfloat16)
    scores = (scores + mask.to(torch.bfloat16)).to(torch.bfloat16)
    probabilities = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
    return torch.matmul(probabilities, value).to(torch.bfloat16).float()


def write_float_tensor(tensor):
    payload = tensor.contiguous().numpy().tobytes(order="C")
    sys.stdout.buffer.write(struct.pack("<I", len(payload)))
    sys.stdout.buffer.write(payload)


def execute_decoder_layer(pool, files, mappings, inputs, per_layer, positions, mask, source_key, source_value, config):
    attention_started = time.perf_counter()
    batch, query_sequence, hidden_size = inputs.shape
    query_heads = config["query_heads"]
    key_value_heads = config["key_value_heads"]
    head_dim = config["head_dim"]
    produces_kv = config["produces_kv"]
    value_from_key = config["value_from_key"]
    intermediate_size = config["intermediate_size"]
    per_layer_width = config["per_layer_width"]
    boundary = lambda tensor: tensor.to(torch.bfloat16).float()
    project = lambda tensor, weight: torch.matmul(tensor.to(torch.bfloat16), weight.transpose(0, 1)).float()
    input_norm_weight = read_whole_tensor(pool, files, mappings, (hidden_size, 1)).reshape(hidden_size)
    query_weight = read_whole_tensor(pool, files, mappings, (query_heads * head_dim, hidden_size), widen=False, required_dtype=torch.bfloat16)
    query_norm = read_whole_tensor(pool, files, mappings, (head_dim, 1)).reshape(head_dim)
    output_weight = read_whole_tensor(pool, files, mappings, (hidden_size, query_heads * head_dim), widen=False, required_dtype=torch.bfloat16)
    normalized_input = boundary(rms_norm_real(inputs, input_norm_weight, config["input_epsilon"]))
    query = boundary(project(normalized_input, query_weight)).reshape(batch, query_sequence, query_heads, head_dim).permute(0, 2, 1, 3)
    query = rope_real(boundary(rms_norm_real(query, query_norm, config["attention_epsilon"])), positions, config["rope_kind"], config["theta"], config["rotary_dim"], config["proportional_pairs"], config["proportional_factor"], True)
    if produces_kv:
        key_weight = read_whole_tensor(pool, files, mappings, (key_value_heads * head_dim, hidden_size), widen=False, required_dtype=torch.bfloat16)
        key_norm = read_whole_tensor(pool, files, mappings, (head_dim, 1)).reshape(head_dim)
        current_key_heads = boundary(project(normalized_input, key_weight)).reshape(batch, query_sequence, key_value_heads, head_dim).permute(0, 2, 1, 3)
        current_key = rope_real(boundary(rms_norm_real(current_key_heads, key_norm, config["attention_epsilon"])), positions, config["rope_kind"], config["theta"], config["rotary_dim"], config["proportional_pairs"], config["proportional_factor"], True)
        if value_from_key:
            current_value = boundary(rms_norm_real(current_key_heads, None, config["attention_epsilon"]))
        else:
            value_weight = read_whole_tensor(pool, files, mappings, (key_value_heads * head_dim, hidden_size), widen=False, required_dtype=torch.bfloat16)
            current_value = boundary(project(normalized_input, value_weight)).reshape(batch, query_sequence, key_value_heads, head_dim).permute(0, 2, 1, 3)
            current_value = boundary(rms_norm_real(current_value, None, config["attention_epsilon"]))
        key = torch.cat((source_key, current_key), dim=2)
        value = torch.cat((source_value, current_value), dim=2)
    else:
        key, value = source_key, source_value
    group = query_heads // key_value_heads
    attention_key = key if group == 1 else key.repeat_interleave(group, dim=1)
    attention_value = value if group == 1 else value.repeat_interleave(group, dim=1)
    context = eager_bf16_attention(query, attention_key, attention_value, mask, config["scale"]).permute(0, 2, 1, 3).contiguous().reshape(batch, query_sequence, query_heads * head_dim)
    attention_projected = boundary(project(context, output_weight))
    attention_seconds = time.perf_counter() - attention_started
    ffn_started = time.perf_counter()
    post_attention_norm_weight = read_whole_tensor(pool, files, mappings, (hidden_size, 1)).reshape(hidden_size)
    pre_ffn_norm_weight = read_whole_tensor(pool, files, mappings, (hidden_size, 1)).reshape(hidden_size)
    gate_weight, gate_reference = read_whole_tensor(pool, files, mappings, (intermediate_size, hidden_size), widen=False, required_dtype=torch.bfloat16, return_reference=True)
    up_weight, up_reference = read_whole_tensor(pool, files, mappings, (intermediate_size, hidden_size), widen=False, required_dtype=torch.bfloat16, return_reference=True)
    down_weight = read_whole_tensor(pool, files, mappings, (hidden_size, intermediate_size), widen=False, required_dtype=torch.bfloat16)
    post_ffn_norm_weight = read_whole_tensor(pool, files, mappings, (hidden_size, 1)).reshape(hidden_size)
    after_attention = boundary(inputs + boundary(rms_norm_real(attention_projected, post_attention_norm_weight, config["post_attention_epsilon"])))
    ffn_input = boundary(rms_norm_real(after_attention, pre_ffn_norm_weight, config["pre_ffn_epsilon"])).to(torch.bfloat16)
    flattened_ffn = ffn_input.reshape(batch * query_sequence, hidden_size)
    gate_path, gate_offset, gate_length, gate_dtype = gate_reference
    up_path, up_offset, up_length, up_dtype = up_reference
    fused_gate_up = int(batch * query_sequence > 1 and gate_path == up_path and gate_dtype == up_dtype == torch.bfloat16 and gate_length == up_length and up_offset == gate_offset + gate_length)
    if fused_gate_up:
        combined_weight = torch.frombuffer(mappings[gate_path], dtype=torch.bfloat16, count=2 * intermediate_size * hidden_size, offset=gate_offset).reshape(2 * intermediate_size, hidden_size)
        gate, up = torch.mm(flattened_ffn, combined_weight.transpose(0, 1)).split(intermediate_size, dim=1)
    else:
        gate = torch.mm(flattened_ffn, gate_weight.transpose(0, 1))
        up = torch.mm(flattened_ffn, up_weight.transpose(0, 1))
    hidden = torch.nn.functional.gelu(gate, approximate="tanh") * up
    ffn_projected = torch.mm(hidden, down_weight.transpose(0, 1)).float().reshape(batch, query_sequence, hidden_size)
    after_mlp = boundary(after_attention + boundary(rms_norm_real(ffn_projected, post_ffn_norm_weight, config["post_ffn_epsilon"])))
    ffn_seconds = time.perf_counter() - ffn_started
    ple_started = time.perf_counter()
    native_ple = config["native_ple"]
    ple_gate_weight = read_whole_tensor(pool, files, mappings, (per_layer_width, hidden_size), widen=not native_ple, required_dtype=torch.bfloat16 if native_ple else None)
    ple_projection_weight = read_whole_tensor(pool, files, mappings, (hidden_size, per_layer_width), widen=not native_ple, required_dtype=torch.bfloat16 if native_ple else None)
    ple_norm_weight = read_whole_tensor(pool, files, mappings, (hidden_size, 1)).reshape(hidden_size)
    layer_scalar = read_whole_tensor(pool, files, mappings, (1, 1)).reshape(())
    ple_gate = boundary(torch.matmul(after_mlp.to(torch.bfloat16) if native_ple else after_mlp, ple_gate_weight.transpose(0, 1)).float())
    cube = (ple_gate * ple_gate) * ple_gate
    inner = torch.tensor(math.sqrt(2 / math.pi), dtype=torch.float32) * (ple_gate + torch.tensor(0.044715, dtype=torch.float32) * cube)
    ple_activated = boundary((torch.tensor(0.5, dtype=torch.float32) * ple_gate) * (torch.tensor(1.0, dtype=torch.float32) + torch.tanh(inner)))
    ple_gated = boundary(ple_activated * per_layer)
    ple_projected = boundary(torch.matmul(ple_gated.to(torch.bfloat16) if native_ple else ple_gated, ple_projection_weight.transpose(0, 1)).float())
    ple_normalized = boundary(rms_norm_real(ple_projected, ple_norm_weight, config["ple_epsilon"]))
    result = boundary(boundary(after_mlp + ple_normalized) * layer_scalar).contiguous()
    validate_cache = config["validate_outputs"] or produces_kv
    if not torch.isfinite(result).all() or (validate_cache and (not torch.isfinite(key).all() or not torch.isfinite(value).all())):
        raise ValueError("fused decoder layer produced non-finite output")
    return result, key, value, (attention_seconds, ffn_seconds, time.perf_counter() - ple_started, fused_gate_up)


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
            fused_ffn = bool(encoded_dtype & 0x00800000)
            fused_decoder_layer = bool(encoded_dtype & 0x00400000)
            fused_decoder_stack = bool(encoded_dtype & 0x00200000)
            streamed_native_bf16 = bool(encoded_dtype & 0x00100000)
            dtype_code = encoded_dtype & 0x000fffff
            if streamed_native_bf16:
                if not referenced or not native_bf16 or batched or fused_mlp or native_attention or fused_attention or fused_ple or fused_ple_prelude or fused_ffn or fused_decoder_layer or fused_decoder_stack or dtype_code != 1 or pool is None:
                    raise ValueError("streamed native BF16 flags are invalid")
                input_bytes = read_exact(rows * features * 4)
                byte_offset, byte_length, start_output, max_read_bytes, shard_length, name_length = struct.unpack("<QIIIII", read_exact(28))
                if start_output != 0 or byte_length != outputs * features * 2 or max_read_bytes < features * 2 or shard_length < 1 or shard_length > 4096 or name_length < 1 or name_length > 4096:
                    raise ValueError("streamed native BF16 metadata is invalid")
                shard = bytes(read_exact(shard_length)).decode("utf-8")
                tensor_name = bytes(read_exact(name_length)).decode("utf-8")
                if Path(shard).name != shard or not tensor_name:
                    raise ValueError("streamed native BF16 identity is invalid")
                path = (pool / shard).resolve()
                if path.parent != pool:
                    raise ValueError("streamed native BF16 tensor escapes binary pool")
                if path not in mappings:
                    files[path] = path.open("rb")
                    mappings[path] = mmap.mmap(files[path].fileno(), 0, access=mmap.ACCESS_READ)
                if byte_offset + byte_length > mappings[path].size():
                    raise ValueError("streamed native BF16 tensor range exceeds shard")
                inputs = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(rows, features).to(torch.bfloat16)
                chunk_rows = max(1, max_read_bytes // (features * 2))
                for first_output in range(0, outputs, chunk_rows):
                    output_count = min(chunk_rows, outputs - first_output)
                    weights = torch.frombuffer(mappings[path], dtype=torch.bfloat16, count=output_count * features, offset=byte_offset + first_output * features * 2).reshape(output_count, features)
                    tile = torch.mm(inputs, weights.transpose(0, 1)).float().contiguous()
                    write_float_tensor(tile)
                    del weights, tile
                sys.stdout.buffer.flush()
                del inputs
                continue
            if fused_decoder_stack:
                if referenced or batched or fused_mlp or native_attention or fused_attention or fused_ple or fused_ple_prelude or fused_ffn or fused_decoder_layer or dtype_code != 0 or pool is None:
                    raise ValueError("fused decoder stack flags are invalid")
                batch, num_layers, hidden_size = rows, outputs, features
                query_sequence, per_layer_width, rounding = struct.unpack("<III", read_exact(12))
                if not query_sequence or not per_layer_width or rounding not in (0, 2) or (native_bf16 != (rounding == 2)):
                    raise ValueError("fused decoder stack topology is invalid")
                input_bytes = read_exact(batch * query_sequence * hidden_size * 4)
                per_layer_bytes = read_exact(batch * query_sequence * num_layers * per_layer_width * 4)
                position_bytes = read_exact(batch * query_sequence * 4)
                result = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(batch, query_sequence, hidden_size)
                all_per_layer = torch.frombuffer(per_layer_bytes, dtype=torch.float32).reshape(batch, query_sequence, num_layers, per_layer_width)
                positions = torch.frombuffer(position_bytes, dtype=torch.int32).reshape(batch, query_sequence)
                produced_caches, ordered_caches = {}, []
                stack_profile = [0.0] * 4
                cache_hits_before = _widened_tensor_cache_hits
                for expected_layer in range(num_layers):
                    metadata = read_exact(96)
                    layer_index, shared_plus_one, query_heads, key_value_heads, source_sequence, head_dim, mask_heads, value_from_key, rope_kind, rotary_dim, proportional_pairs, intermediate_size, descriptor_count = struct.unpack("<IIIIIIIIIIIII", metadata[:52])
                    attention_epsilon, scale = struct.unpack("<ff", metadata[52:60])
                    theta = struct.unpack("<d", metadata[60:68])[0]
                    proportional_factor, input_epsilon, post_attention_epsilon, pre_ffn_epsilon, post_ffn_epsilon, ple_epsilon = struct.unpack("<ffffff", metadata[68:92])
                    shared = shared_plus_one != 0
                    producer_layer = shared_plus_one - 1
                    produces_kv = not shared
                    total_key_sequence = source_sequence + (query_sequence if produces_kv else 0)
                    expected_descriptors = 14 + (2 + (0 if value_from_key else 1) if produces_kv else 0)
                    epsilons = (attention_epsilon, input_epsilon, post_attention_epsilon, pre_ffn_epsilon, post_ffn_epsilon, ple_epsilon)
                    if layer_index != expected_layer or not query_heads or not key_value_heads or not head_dim or not mask_heads or not intermediate_size or total_key_sequence < 1 or query_heads % key_value_heads or mask_heads not in (1, query_heads) or value_from_key not in (0, 1) or (shared and value_from_key) or (shared and (producer_layer < 0 or producer_layer >= layer_index)) or rope_kind not in (0, 1) or not rotary_dim or rotary_dim > head_dim or rotary_dim % 2 or proportional_pairs > rotary_dim // 2 or descriptor_count != expected_descriptors or any(not math.isfinite(value) or value <= 0 for value in epsilons) or not math.isfinite(scale) or not math.isfinite(theta) or theta <= 0 or not math.isfinite(proportional_factor) or proportional_factor <= 0:
                        raise ValueError(f"fused decoder stack layer {layer_index} is invalid")
                    mask_bytes = read_exact(batch * mask_heads * query_sequence * total_key_sequence * 4)
                    mask = torch.frombuffer(mask_bytes, dtype=torch.float32).reshape(batch, mask_heads, query_sequence, total_key_sequence)
                    if torch.isnan(mask).any() or torch.isposinf(mask).any():
                        raise ValueError("fused decoder stack mask is invalid")
                    if shared:
                        if producer_layer not in produced_caches:
                            raise ValueError("fused decoder stack shared cache is unavailable")
                        source_key, source_value = produced_caches[producer_layer]
                        if source_key.shape != (batch, key_value_heads, source_sequence, head_dim) or source_value.shape != source_key.shape:
                            raise ValueError("fused decoder stack shared cache shape is invalid")
                    else:
                        source_elements = batch * key_value_heads * source_sequence * head_dim
                        source_key_bytes = read_exact(source_elements * 4)
                        source_value_bytes = read_exact(source_elements * 4)
                        if source_sequence:
                            source_key = torch.frombuffer(source_key_bytes, dtype=torch.float32).reshape(batch, key_value_heads, source_sequence, head_dim)
                            source_value = torch.frombuffer(source_value_bytes, dtype=torch.float32).reshape(batch, key_value_heads, source_sequence, head_dim)
                        else:
                            source_key = torch.empty((batch, key_value_heads, 0, head_dim), dtype=torch.float32)
                            source_value = torch.empty((batch, key_value_heads, 0, head_dim), dtype=torch.float32)
                    config = {"query_heads": query_heads, "key_value_heads": key_value_heads, "head_dim": head_dim, "produces_kv": produces_kv, "value_from_key": value_from_key, "intermediate_size": intermediate_size, "per_layer_width": per_layer_width, "input_epsilon": input_epsilon, "attention_epsilon": attention_epsilon, "post_attention_epsilon": post_attention_epsilon, "pre_ffn_epsilon": pre_ffn_epsilon, "post_ffn_epsilon": post_ffn_epsilon, "ple_epsilon": ple_epsilon, "scale": scale, "rope_kind": rope_kind, "theta": theta, "rotary_dim": rotary_dim, "proportional_pairs": proportional_pairs, "proportional_factor": proportional_factor, "native_ple": native_bf16, "validate_outputs": False}
                    result, key, value, layer_profile = execute_decoder_layer(pool, files, mappings, result, all_per_layer[:, :, layer_index, :], positions, mask, source_key, source_value, config)
                    for profile_index, profile_value in enumerate(layer_profile):
                        stack_profile[profile_index] += profile_value
                    if produces_kv:
                        produced_caches[layer_index] = (key, value)
                        ordered_caches.append((key, value))
                write_float_tensor(result)
                for key, value in ordered_caches:
                    write_float_tensor(key)
                    write_float_tensor(value)
                stack_profile.extend((_widened_tensor_cache_hits - cache_hits_before, len(_widened_tensor_cache), _widened_tensor_cache_bytes))
                write_float_tensor(torch.tensor(stack_profile, dtype=torch.float32))
                sys.stdout.buffer.flush()
                del result, all_per_layer, positions, produced_caches, ordered_caches, stack_profile
                continue
            if fused_decoder_layer:
                if referenced or batched or fused_mlp or native_bf16 or native_attention or fused_attention or fused_ple or fused_ple_prelude or fused_ffn or fused_decoder_stack or dtype_code != 0 or pool is None:
                    raise ValueError("fused decoder layer flags are invalid")
                batch, query_heads, key_value_heads = rows, outputs, features
                metadata = read_exact(96)
                query_sequence, source_sequence, hidden_size, head_dim, mask_heads, produces_kv, value_from_key, rope_kind, rotary_dim, proportional_pairs, intermediate_size, per_layer_width, descriptor_count = struct.unpack("<IIIIIIIIIIIII", metadata[:52])
                attention_epsilon, scale = struct.unpack("<ff", metadata[52:60])
                theta = struct.unpack("<d", metadata[60:68])[0]
                proportional_factor, input_epsilon, post_attention_epsilon, pre_ffn_epsilon, post_ffn_epsilon, ple_epsilon = struct.unpack("<ffffff", metadata[68:92])
                total_key_sequence = source_sequence + (query_sequence if produces_kv else 0)
                expected_descriptors = 14 + (2 + (0 if value_from_key else 1) if produces_kv else 0)
                epsilons = (attention_epsilon, input_epsilon, post_attention_epsilon, pre_ffn_epsilon, post_ffn_epsilon, ple_epsilon)
                if not batch or not query_heads or not key_value_heads or not query_sequence or not hidden_size or not head_dim or not mask_heads or not intermediate_size or not per_layer_width or total_key_sequence < 1 or query_heads % key_value_heads or mask_heads not in (1, query_heads) or produces_kv not in (0, 1) or value_from_key not in (0, 1) or (not produces_kv and value_from_key) or rope_kind not in (0, 1) or not rotary_dim or rotary_dim > head_dim or rotary_dim % 2 or proportional_pairs > rotary_dim // 2 or descriptor_count != expected_descriptors or any(not math.isfinite(value) or value <= 0 for value in epsilons) or not math.isfinite(scale) or not math.isfinite(theta) or theta <= 0 or not math.isfinite(proportional_factor) or proportional_factor <= 0:
                    raise ValueError("fused decoder layer topology is invalid")
                if not produces_kv and source_sequence < query_sequence:
                    raise ValueError("shared decoder layer cache is shorter than query")
                input_bytes = read_exact(batch * query_sequence * hidden_size * 4)
                per_layer_bytes = read_exact(batch * query_sequence * per_layer_width * 4)
                position_bytes = read_exact(batch * query_sequence * 4)
                mask_bytes = read_exact(batch * mask_heads * query_sequence * total_key_sequence * 4)
                source_key_bytes = read_exact(batch * key_value_heads * source_sequence * head_dim * 4)
                source_value_bytes = read_exact(batch * key_value_heads * source_sequence * head_dim * 4)
                inputs = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(batch, query_sequence, hidden_size)
                per_layer = torch.frombuffer(per_layer_bytes, dtype=torch.float32).reshape(batch, query_sequence, per_layer_width)
                positions = torch.frombuffer(position_bytes, dtype=torch.int32).reshape(batch, query_sequence)
                mask = torch.frombuffer(mask_bytes, dtype=torch.float32).reshape(batch, mask_heads, query_sequence, total_key_sequence)
                if torch.isnan(mask).any() or torch.isposinf(mask).any():
                    raise ValueError("fused decoder layer mask is invalid")
                if source_sequence:
                    source_key = torch.frombuffer(source_key_bytes, dtype=torch.float32).reshape(batch, key_value_heads, source_sequence, head_dim)
                    source_value = torch.frombuffer(source_value_bytes, dtype=torch.float32).reshape(batch, key_value_heads, source_sequence, head_dim)
                else:
                    source_key = torch.empty((batch, key_value_heads, 0, head_dim), dtype=torch.float32)
                    source_value = torch.empty((batch, key_value_heads, 0, head_dim), dtype=torch.float32)
                config = {"query_heads": query_heads, "key_value_heads": key_value_heads, "head_dim": head_dim, "produces_kv": produces_kv, "value_from_key": value_from_key, "intermediate_size": intermediate_size, "per_layer_width": per_layer_width, "input_epsilon": input_epsilon, "attention_epsilon": attention_epsilon, "post_attention_epsilon": post_attention_epsilon, "pre_ffn_epsilon": pre_ffn_epsilon, "post_ffn_epsilon": post_ffn_epsilon, "ple_epsilon": ple_epsilon, "scale": scale, "rope_kind": rope_kind, "theta": theta, "rotary_dim": rotary_dim, "proportional_pairs": proportional_pairs, "proportional_factor": proportional_factor, "native_ple": False, "validate_outputs": True}
                result, key, value, _ = execute_decoder_layer(pool, files, mappings, inputs, per_layer, positions, mask, source_key, source_value, config)
                write_float_tensor(result)
                if produces_kv:
                    write_float_tensor(key)
                    write_float_tensor(value)
                sys.stdout.buffer.flush()
                del inputs, per_layer, positions, mask, source_key, source_value, config, key, value, result
                continue
            if fused_attention:
                if referenced or batched or fused_mlp or native_bf16 or native_attention or fused_ple or fused_ple_prelude or fused_ffn or fused_decoder_layer or fused_decoder_stack or dtype_code != 0 or pool is None:
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
                if not batch or not query_heads or not key_value_heads or not query_sequence or not hidden_size or not head_dim or not mask_heads or total_key_sequence < 1 or query_heads % key_value_heads or mask_heads not in (1, query_heads) or produces_kv not in (0, 1) or value_from_key not in (0, 1) or (not produces_kv and value_from_key) or rope_kind not in (0, 1) or rounding not in (0, 1, 2) or not rotary_dim or rotary_dim > head_dim or rotary_dim % 2 or proportional_pairs > rotary_dim // 2 or descriptor_count != expected_descriptors or not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(scale) or not math.isfinite(theta) or theta <= 0 or not math.isfinite(proportional_factor) or proportional_factor <= 0:
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
                native_projections = rounding == 2
                required_projection_dtype = torch.bfloat16 if native_projections else None
                query_weight = read_whole_tensor(pool, files, mappings, (query_heads * head_dim, hidden_size), widen=not native_projections, required_dtype=required_projection_dtype)
                query_norm = read_whole_tensor(pool, files, mappings, (head_dim, 1)).reshape(head_dim)
                output_weight = read_whole_tensor(pool, files, mappings, (hidden_size, query_heads * head_dim), widen=not native_projections, required_dtype=required_projection_dtype)
                boundary = (lambda tensor: tensor.to(torch.bfloat16).float()) if rounding in (0, 2) else (lambda tensor: tensor)
                project = (lambda tensor, weight: torch.matmul(tensor.to(torch.bfloat16), weight.transpose(0, 1)).float()) if native_projections else (lambda tensor, weight: torch.matmul(tensor, weight.transpose(0, 1)))
                query = boundary(project(inputs, query_weight)).reshape(batch, query_sequence, query_heads, head_dim).permute(0, 2, 1, 3)
                query = rope_real(boundary(rms_norm_real(query, query_norm, epsilon)), positions, rope_kind, theta, rotary_dim, proportional_pairs, proportional_factor, rounding in (0, 2))
                key_weight = key_norm = value_weight = current_key_heads = current_key = current_value = None
                if produces_kv:
                    key_weight = read_whole_tensor(pool, files, mappings, (key_value_heads * head_dim, hidden_size), widen=not native_projections, required_dtype=required_projection_dtype)
                    key_norm = read_whole_tensor(pool, files, mappings, (head_dim, 1)).reshape(head_dim)
                    current_key_heads = boundary(project(inputs, key_weight)).reshape(batch, query_sequence, key_value_heads, head_dim).permute(0, 2, 1, 3)
                    current_key = rope_real(boundary(rms_norm_real(current_key_heads, key_norm, epsilon)), positions, rope_kind, theta, rotary_dim, proportional_pairs, proportional_factor, rounding in (0, 2))
                    if value_from_key:
                        current_value = boundary(rms_norm_real(current_key_heads, None, epsilon))
                    else:
                        value_weight = read_whole_tensor(pool, files, mappings, (key_value_heads * head_dim, hidden_size), widen=not native_projections, required_dtype=required_projection_dtype)
                        current_value = boundary(project(inputs, value_weight)).reshape(batch, query_sequence, key_value_heads, head_dim).permute(0, 2, 1, 3)
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
                projected = boundary(project(context, output_weight))
                if not torch.isfinite(projected).all() or not torch.isfinite(key).all() or not torch.isfinite(value).all():
                    raise ValueError("fused attention produced non-finite output")
                write_float_tensor(projected)
                if produces_kv:
                    write_float_tensor(key)
                    write_float_tensor(value)
                sys.stdout.buffer.flush()
                del inputs, positions, mask, source_key, source_value, query_weight, query_norm, output_weight, query, key_weight, key_norm, value_weight, current_key_heads, current_key, current_value, key, value, attention_key, attention_value, scores, probabilities, context, projected, boundary, project
                continue
            if fused_ple:
                if referenced or batched or fused_mlp or native_bf16 or native_attention or fused_attention or fused_ple_prelude or fused_ffn or dtype_code != 0 or pool is None or outputs != features:
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
                if referenced or batched or fused_mlp or native_bf16 or native_attention or fused_attention or fused_ple or fused_ffn or dtype_code != 0 or pool is None:
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
                if referenced or batched or fused_mlp or native_bf16 or fused_attention or fused_ple or fused_ple_prelude or fused_ffn or dtype_code != 0:
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
                    context = eager_bf16_attention(query, key, value, mask, scale)
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
                del query, key, value, mask, context, result, payload
                continue
            input_bytes = read_exact(rows * features * 4)
            if fused_ffn:
                if not referenced or batched or fused_mlp or native_bf16 or native_attention or fused_attention or fused_ple or fused_ple_prelude or dtype_code != 0 or pool is None or outputs != features:
                    raise ValueError("fused FFN flags are invalid")
                intermediate_size, pre_epsilon, post_epsilon = struct.unpack("<Iff", read_exact(12))
                if not intermediate_size or not math.isfinite(pre_epsilon) or pre_epsilon <= 0 or not math.isfinite(post_epsilon) or post_epsilon <= 0:
                    raise ValueError("fused FFN metadata is invalid")
                inputs = torch.frombuffer(input_bytes, dtype=torch.float32).reshape(rows, features)
                pre_norm_weight = read_whole_tensor(pool, files, mappings, (features, 1)).reshape(features)
                gate_weight = read_whole_tensor(pool, files, mappings, (intermediate_size, features), widen=False, required_dtype=torch.bfloat16)
                up_weight = read_whole_tensor(pool, files, mappings, (intermediate_size, features), widen=False, required_dtype=torch.bfloat16)
                down_weight = read_whole_tensor(pool, files, mappings, (features, intermediate_size), widen=False, required_dtype=torch.bfloat16)
                post_norm_weight = read_whole_tensor(pool, files, mappings, (features, 1)).reshape(features)
                boundary = lambda value: value.to(torch.bfloat16).float()
                normalized_input = boundary(rms_norm_real(inputs, pre_norm_weight, pre_epsilon))
                native_input = normalized_input.to(torch.bfloat16)
                gate = torch.mm(native_input, gate_weight.transpose(0, 1))
                up = torch.mm(native_input, up_weight.transpose(0, 1))
                activated = torch.nn.functional.gelu(gate, approximate="tanh")
                hidden = activated * up
                projected = torch.mm(hidden, down_weight.transpose(0, 1)).float()
                normalized_output = boundary(rms_norm_real(projected, post_norm_weight, post_epsilon))
                result = boundary(inputs + normalized_output).contiguous()
                if not torch.isfinite(result).all():
                    raise ValueError("fused FFN produced non-finite output")
                write_float_tensor(result)
                sys.stdout.buffer.flush()
                del inputs, pre_norm_weight, gate_weight, up_weight, down_weight, post_norm_weight, boundary, normalized_input, native_input, gate, up, activated, hidden, projected, normalized_output, result
                continue
            if fused_mlp:
                if not referenced or batched or native_bf16 or fused_ffn or pool is None:
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
