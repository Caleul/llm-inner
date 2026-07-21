#!/usr/bin/env python3
"""Persistent MLX Metal worker for F32 inputs and dense F32/F16/BF16 weights."""

import argparse
from collections import OrderedDict
import math
from pathlib import Path
import struct
import sys

import mlx.core as mx
import numpy as np


FUSED_DECODER_STACK_FLAG = 0x00200000
FUSED_DECODER_STACK_EPILOGUE_FLAG = 0x00080000
FUSED_TOKEN_FORWARD_FLAG = 0x00040000
FUSED_PLE_PRELUDE_FLAG = 0x01000000
_widened_tensor_cache = {}
_widened_tensor_cache_hits = 0
_widened_tensor_cache_bytes = 0
_shard_sizes = {}
_binary_files = {}
_embedding_row_cache = OrderedDict()
EMBEDDING_ROW_CACHE_LIMIT = 4096


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


def write_float_tensor(tensor):
    payload = np.asarray(tensor, dtype=np.float32).tobytes(order="C")
    sys.stdout.buffer.write(struct.pack("<I", len(payload)))
    sys.stdout.buffer.write(payload)


def read_whole_tensor(pool, shards, expected_shape, widen=True, required_dtype=None, large_descriptor=False):
    global _widened_tensor_cache_hits, _widened_tensor_cache_bytes
    metadata = read_exact(40 if large_descriptor else 36)
    if large_descriptor:
        request_dtype, first, second, byte_offset, byte_length, start_output, shard_length, name_length = struct.unpack("<IIIQQIII", metadata)
    else:
        request_dtype, first, second, byte_offset, byte_length, start_output, shard_length, name_length = struct.unpack("<IIIQIIII", metadata)
    if request_dtype not in (0, 1, 2) or (first, second) != expected_shape or start_output != 0:
        raise ValueError("MLX decoder stack tensor descriptor is invalid")
    if shard_length < 1 or shard_length > 4096 or name_length < 1 or name_length > 4096:
        raise ValueError("MLX decoder stack tensor identity length is invalid")
    shard = bytes(read_exact(shard_length)).decode("utf-8")
    tensor_name = bytes(read_exact(name_length)).decode("utf-8")
    if Path(shard).name != shard or not tensor_name:
        raise ValueError("MLX decoder stack tensor identity is invalid")
    path = (pool / shard).resolve()
    shard_size = _shard_sizes.get(path)
    if shard_size is None:
        shard_size = path.stat().st_size
        _shard_sizes[path] = shard_size
    if path.parent != pool or byte_offset + byte_length > shard_size:
        raise ValueError("MLX decoder stack tensor escapes its shard")
    if path not in shards:
        shards[path] = mx.load(str(path))
    tensor = shards[path].get(tensor_name)
    expected_dtype = (mx.float32, mx.bfloat16, mx.float16)[request_dtype]
    element_bytes = 4 if request_dtype == 0 else 2
    if tensor is None or tensor.size != first * second or tensor.dtype != expected_dtype or byte_length != first * second * element_bytes:
        raise ValueError(f"{tensor_name}: MLX decoder stack tensor identity diverges")
    if required_dtype is not None and tensor.dtype != required_dtype:
        raise ValueError(f"{tensor_name}: MLX decoder stack tensor dtype is incompatible")
    tensor = tensor.reshape(expected_shape)
    if widen and tensor.dtype != mx.float32:
        cache_key = (path, tensor_name, expected_shape)
        result = _widened_tensor_cache.get(cache_key)
        if result is None:
            result = tensor.astype(mx.float32)
            _widened_tensor_cache[cache_key] = result
            _widened_tensor_cache_bytes += result.size * 4
        else:
            _widened_tensor_cache_hits += 1
        return result
    return tensor


def round_numpy_f32_to_bf16(values):
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    bits = contiguous.view(np.uint32)
    rounded = bits + np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((rounded >> np.uint32(16)) << np.uint32(16)).view(np.float32)


def read_scaled_bf16_embedding_rows(pool, shards, expected_shape, token_ids, scale):
    metadata = read_exact(40)
    request_dtype, first, second, byte_offset, byte_length, start_output, shard_length, name_length = struct.unpack("<IIIQQIII", metadata)
    if request_dtype != 1 or (first, second) != expected_shape or start_output != 0 or byte_length != first * second * 2:
        raise ValueError("MLX token forward embedding descriptor is invalid")
    if shard_length < 1 or shard_length > 4096 or name_length < 1 or name_length > 4096:
        raise ValueError("MLX token forward embedding identity length is invalid")
    shard = bytes(read_exact(shard_length)).decode("utf-8")
    tensor_name = bytes(read_exact(name_length)).decode("utf-8")
    if Path(shard).name != shard or not tensor_name:
        raise ValueError("MLX token forward embedding identity is invalid")
    path = (pool / shard).resolve()
    if path.parent != pool:
        raise ValueError("MLX token forward embedding escapes its shard")
    shard_size = _shard_sizes.get(path)
    if shard_size is None:
        shard_size = path.stat().st_size
        _shard_sizes[path] = shard_size
    if byte_offset + byte_length > shard_size:
        raise ValueError("MLX token forward embedding range exceeds its shard")
    if path not in shards:
        shards[path] = mx.load(str(path))
    tensor = shards[path].get(tensor_name)
    if tensor is None or tensor.shape != expected_shape or tensor.dtype != mx.bfloat16:
        raise ValueError(f"{tensor_name}: MLX token forward embedding identity diverges")
    flat_tokens = token_ids.reshape(-1)
    unique_rows = {}
    row_bytes = second * 2
    source = _binary_files.get(path)
    if source is None:
        source = path.open("rb")
        _binary_files[path] = source
    for token in np.unique(flat_tokens):
        cache_key = (path, tensor_name, int(token), np.float32(scale).tobytes())
        row = _embedding_row_cache.get(cache_key)
        if row is None:
            source.seek(byte_offset + int(token) * row_bytes)
            payload = source.read(row_bytes)
            if len(payload) != row_bytes:
                raise ValueError("MLX token forward embedding row is truncated")
            bits = np.frombuffer(payload, dtype="<u2").astype(np.uint32) << np.uint32(16)
            product = np.multiply(bits.view(np.float32), np.float32(scale), dtype=np.float32)
            row = round_numpy_f32_to_bf16(product)
            _embedding_row_cache[cache_key] = row
            if len(_embedding_row_cache) > EMBEDDING_ROW_CACHE_LIMIT:
                _embedding_row_cache.popitem(last=False)
        else:
            _embedding_row_cache.move_to_end(cache_key)
        unique_rows[int(token)] = row
    values = np.stack([unique_rows[int(token)] for token in flat_tokens], axis=0)
    return mx.array(values.reshape((*token_ids.shape, second)))


def rms_norm_real(tensor, weight, epsilon):
    normalized = tensor * mx.rsqrt(mx.mean(tensor * tensor, axis=-1, keepdims=True) + mx.array(epsilon, dtype=mx.float32))
    return normalized if weight is None else normalized * weight


def rms_norm_cpu_cascade_f32(tensor, weight, epsilon):
    """Replay the artifact-declared PyTorch ARM F32 cascade without reassociation."""
    vectors = np.asarray(tensor, dtype=np.float32).reshape(-1, tensor.shape[-1])
    width = vectors.shape[1]
    lanes, registers, levels = 4, 4, 4
    if width % (lanes * registers):
        raise ValueError("MLX fused PLE prelude cascade width is incompatible")
    unit_count = width // (lanes * registers)
    level_power = max(4, math.ceil(math.log2(unit_count)) >> 2)
    level_step, level_mask = 1 << level_power, (1 << level_power) - 1
    accumulators = np.zeros((levels, vectors.shape[0], registers, lanes), dtype=np.float32)
    units = vectors.reshape(vectors.shape[0], unit_count, registers, lanes)
    unit = 0
    while unit + level_step <= unit_count:
        for _ in range(level_step):
            squared = np.multiply(units[:, unit], units[:, unit], dtype=np.float32)
            accumulators[0] = np.add(accumulators[0], squared, dtype=np.float32)
            unit += 1
        for level in range(1, levels):
            accumulators[level] = np.add(accumulators[level], accumulators[level - 1], dtype=np.float32)
            accumulators[level - 1].fill(0)
            if unit & (level_mask << (level * level_power)):
                break
    while unit < unit_count:
        squared = np.multiply(units[:, unit], units[:, unit], dtype=np.float32)
        accumulators[0] = np.add(accumulators[0], squared, dtype=np.float32)
        unit += 1
    for level in range(1, levels):
        accumulators[0] = np.add(accumulators[0], accumulators[level], dtype=np.float32)
    for register in range(1, registers):
        accumulators[0, :, 0] = np.add(accumulators[0, :, 0], accumulators[0, :, register], dtype=np.float32)
    sums = np.zeros(vectors.shape[0], dtype=np.float32)
    for lane in range(lanes):
        sums = np.add(sums, accumulators[0, :, 0, lane], dtype=np.float32)
    means = np.divide(sums, np.float32(width), dtype=np.float32)
    rooted = np.array([np.float32(math.sqrt(float(np.float32(value + np.float32(epsilon))))) for value in means], dtype=np.float32)
    scales = np.divide(np.float32(1.0), rooted, dtype=np.float32)
    normalized = np.multiply(vectors, scales[:, None], dtype=np.float32)
    normalized = np.multiply(normalized, np.asarray(weight, dtype=np.float32)[None, :], dtype=np.float32)
    return normalized.reshape(tensor.shape)


def execute_ple_prelude_values(inputs, token_identity, projection_weight, norm_weight, rows, num_layers, per_layer_width, context_scale, combine_scale, epsilon, tile_output_rows):
    boundary = lambda value: value.astype(mx.bfloat16).astype(mx.float32)
    outputs = num_layers * per_layer_width
    context_tiles = [
        mx.matmul(inputs, projection_weight[start:start + tile_output_rows].T)
        for start in range(0, outputs, tile_output_rows)
    ]
    mx.eval(*context_tiles)
    context = boundary(mx.concatenate(context_tiles, axis=1))
    context = boundary(context * mx.array(context_scale, dtype=mx.float32)).reshape((rows, num_layers, per_layer_width))
    mx.eval(context, norm_weight)
    normalized_values = rms_norm_cpu_cascade_f32(np.asarray(context, dtype=np.float32), np.asarray(norm_weight, dtype=np.float32), epsilon)
    normalized = boundary(mx.array(normalized_values))
    combined = boundary(normalized + token_identity)
    return boundary(combined * mx.array(combine_scale, dtype=mx.float32)).reshape((rows, outputs))


def rope_real(tensor, positions, rope_kind, theta, rotary_dim, proportional_pairs, proportional_factor):
    boundary = lambda value: value.astype(mx.bfloat16).astype(mx.float32)
    head_dim, half = tensor.shape[-1], rotary_dim // 2
    pairs = mx.arange(half, dtype=mx.float32)
    denominator_width = head_dim if rope_kind == 1 else rotary_dim
    denominator = mx.power(mx.array(theta, dtype=mx.float32), (mx.array(2.0, dtype=mx.float32) * pairs) / mx.array(denominator_width, dtype=mx.float32))
    if rope_kind == 1:
        denominator = denominator * mx.array(proportional_factor, dtype=mx.float32)
    angles = positions[:, None, :, None].astype(mx.float32) / denominator[None, None, None, :]
    if rope_kind == 1 and proportional_pairs < half:
        angles = mx.where((pairs < proportional_pairs)[None, None, None, :], angles, mx.zeros_like(angles))
    cosine, sine = boundary(mx.cos(angles)), boundary(mx.sin(angles))
    first, second = tensor[..., :half], tensor[..., half:rotary_dim]
    rotated_first = boundary(boundary(first * cosine) - boundary(second * sine))
    rotated_second = boundary(boundary(second * cosine) + boundary(first * sine))
    return mx.concatenate((rotated_first, rotated_second, tensor[..., rotary_dim:]), axis=-1)


def execute_decoder_layer_mlx(pool, shards, inputs, per_layer, positions, mask, source_key, source_value, config):
    batch, query_sequence, hidden_size = inputs.shape
    query_heads, key_value_heads, head_dim = config["query_heads"], config["key_value_heads"], config["head_dim"]
    produces_kv, value_from_key = config["produces_kv"], config["value_from_key"]
    intermediate_size, per_layer_width = config["intermediate_size"], config["per_layer_width"]
    boundary = lambda value: value.astype(mx.bfloat16).astype(mx.float32)
    project = lambda value, weight: mx.matmul(value.astype(mx.bfloat16), weight.T).astype(mx.float32)
    input_norm_weight = read_whole_tensor(pool, shards, (hidden_size, 1)).reshape((hidden_size,))
    query_weight = read_whole_tensor(pool, shards, (query_heads * head_dim, hidden_size), widen=False, required_dtype=mx.bfloat16)
    query_norm = read_whole_tensor(pool, shards, (head_dim, 1)).reshape((head_dim,))
    output_weight = read_whole_tensor(pool, shards, (hidden_size, query_heads * head_dim), widen=False, required_dtype=mx.bfloat16)
    normalized_input = boundary(rms_norm_real(inputs, input_norm_weight, config["input_epsilon"]))
    query = mx.transpose(boundary(project(normalized_input, query_weight)).reshape((batch, query_sequence, query_heads, head_dim)), (0, 2, 1, 3))
    query = rope_real(boundary(rms_norm_real(query, query_norm, config["attention_epsilon"])), positions, config["rope_kind"], config["theta"], config["rotary_dim"], config["proportional_pairs"], config["proportional_factor"])
    if produces_kv:
        key_weight = read_whole_tensor(pool, shards, (key_value_heads * head_dim, hidden_size), widen=False, required_dtype=mx.bfloat16)
        key_norm = read_whole_tensor(pool, shards, (head_dim, 1)).reshape((head_dim,))
        current_key_heads = mx.transpose(boundary(project(normalized_input, key_weight)).reshape((batch, query_sequence, key_value_heads, head_dim)), (0, 2, 1, 3))
        current_key = rope_real(boundary(rms_norm_real(current_key_heads, key_norm, config["attention_epsilon"])), positions, config["rope_kind"], config["theta"], config["rotary_dim"], config["proportional_pairs"], config["proportional_factor"])
        if value_from_key:
            current_value = boundary(rms_norm_real(current_key_heads, None, config["attention_epsilon"]))
        else:
            value_weight = read_whole_tensor(pool, shards, (key_value_heads * head_dim, hidden_size), widen=False, required_dtype=mx.bfloat16)
            current_value = mx.transpose(boundary(project(normalized_input, value_weight)).reshape((batch, query_sequence, key_value_heads, head_dim)), (0, 2, 1, 3))
            current_value = boundary(rms_norm_real(current_value, None, config["attention_epsilon"]))
        key = mx.concatenate((source_key, current_key), axis=2)
        value = mx.concatenate((source_value, current_value), axis=2)
    else:
        key, value = source_key, source_value
    group = query_heads // key_value_heads
    attention_key = key if group == 1 else mx.repeat(key, group, axis=1)
    attention_value = value if group == 1 else mx.repeat(value, group, axis=1)
    scores = mx.matmul(query, mx.swapaxes(attention_key, -1, -2)) * mx.array(config["scale"], dtype=mx.float32) + mask
    probabilities = mx.softmax(scores, axis=-1)
    context = mx.transpose(mx.matmul(probabilities, attention_value), (0, 2, 1, 3)).reshape((batch, query_sequence, query_heads * head_dim))
    attention_projected = boundary(project(context, output_weight))
    post_attention_norm_weight = read_whole_tensor(pool, shards, (hidden_size, 1)).reshape((hidden_size,))
    pre_ffn_norm_weight = read_whole_tensor(pool, shards, (hidden_size, 1)).reshape((hidden_size,))
    gate_weight = read_whole_tensor(pool, shards, (intermediate_size, hidden_size), widen=False, required_dtype=mx.bfloat16)
    up_weight = read_whole_tensor(pool, shards, (intermediate_size, hidden_size), widen=False, required_dtype=mx.bfloat16)
    down_weight = read_whole_tensor(pool, shards, (hidden_size, intermediate_size), widen=False, required_dtype=mx.bfloat16)
    post_ffn_norm_weight = read_whole_tensor(pool, shards, (hidden_size, 1)).reshape((hidden_size,))
    after_attention = boundary(inputs + boundary(rms_norm_real(attention_projected, post_attention_norm_weight, config["post_attention_epsilon"])))
    ffn_input = boundary(rms_norm_real(after_attention, pre_ffn_norm_weight, config["pre_ffn_epsilon"])).astype(mx.bfloat16)
    gate, up = mx.matmul(ffn_input, gate_weight.T), mx.matmul(ffn_input, up_weight.T)
    cube = (gate * gate) * gate
    inner = mx.array(math.sqrt(2 / math.pi), dtype=mx.float32) * (gate + mx.array(0.044715, dtype=mx.float32) * cube)
    hidden = (mx.array(0.5, dtype=mx.float32) * gate) * (mx.array(1.0, dtype=mx.float32) + mx.tanh(inner)) * up
    ffn_projected = mx.matmul(hidden.astype(mx.bfloat16), down_weight.T).astype(mx.float32)
    after_mlp = boundary(after_attention + boundary(rms_norm_real(ffn_projected, post_ffn_norm_weight, config["post_ffn_epsilon"])))
    ple_gate_weight = read_whole_tensor(pool, shards, (per_layer_width, hidden_size))
    ple_projection_weight = read_whole_tensor(pool, shards, (hidden_size, per_layer_width))
    ple_norm_weight = read_whole_tensor(pool, shards, (hidden_size, 1)).reshape((hidden_size,))
    layer_scalar = read_whole_tensor(pool, shards, (1, 1)).reshape(())
    ple_gate = boundary(mx.matmul(after_mlp, ple_gate_weight.T))
    cube = (ple_gate * ple_gate) * ple_gate
    inner = mx.array(math.sqrt(2 / math.pi), dtype=mx.float32) * (ple_gate + mx.array(0.044715, dtype=mx.float32) * cube)
    ple_activated = boundary((mx.array(0.5, dtype=mx.float32) * ple_gate) * (mx.array(1.0, dtype=mx.float32) + mx.tanh(inner)))
    ple_projected = boundary(mx.matmul(boundary(ple_activated * per_layer), ple_projection_weight.T))
    ple_normalized = boundary(rms_norm_real(ple_projected, ple_norm_weight, config["ple_epsilon"]))
    result = boundary(boundary(after_mlp + ple_normalized) * layer_scalar)
    valid = mx.all(mx.isfinite(result))
    if produces_kv:
        valid = valid & mx.all(mx.isfinite(key)) & mx.all(mx.isfinite(value))
    return result, key, value, valid


def execute_decoder_stack_request(pool, shards, batch, num_layers, hidden_size, native_bf16_ple, fused_epilogue, fused_token_forward):
    if native_bf16_ple:
        raise ValueError("MLX decoder stack does not support native-bf16-ple")
    cache_hits_before = _widened_tensor_cache_hits
    query_sequence, per_layer_width = struct.unpack("<II", read_exact(8))
    if not query_sequence or not per_layer_width:
        raise ValueError("MLX decoder stack topology is invalid")
    rows = batch * query_sequence
    if fused_token_forward:
        descriptor_count, rounding, tile_output_rows, vocabulary_size, token_scale, per_layer_scale, context_scale, combine_scale = struct.unpack("<IIIIffff", read_exact(32))
        epsilon = struct.unpack("<f", read_exact(4))[0]
        if descriptor_count != 4 or rounding != 0 or not tile_output_rows or not vocabulary_size or not all(math.isfinite(value) for value in (token_scale, per_layer_scale, context_scale, combine_scale, epsilon)) or epsilon <= 0:
            raise ValueError("MLX token forward prelude metadata is invalid")
        token_ids = np.frombuffer(read_exact(rows * 4), dtype=np.int32).reshape(batch, query_sequence)
        if (token_ids < 0).any() or (token_ids >= vocabulary_size).any():
            raise ValueError("MLX token forward token id is outside the vocabulary")
        result = read_scaled_bf16_embedding_rows(pool, shards, (vocabulary_size, hidden_size), token_ids, token_scale)
        token_identity = read_scaled_bf16_embedding_rows(pool, shards, (vocabulary_size, num_layers * per_layer_width), token_ids, per_layer_scale).reshape((rows, num_layers, per_layer_width))
        projection_weight = read_whole_tensor(pool, shards, (num_layers * per_layer_width, hidden_size), widen=False, required_dtype=mx.bfloat16, large_descriptor=True)
        norm_weight = read_whole_tensor(pool, shards, (per_layer_width, 1), large_descriptor=True).reshape((per_layer_width,))
        result = result.reshape((rows, hidden_size))
        all_per_layer = execute_ple_prelude_values(result, token_identity, projection_weight, norm_weight, rows, num_layers, per_layer_width, context_scale, combine_scale, epsilon, tile_output_rows).reshape((batch, query_sequence, num_layers, per_layer_width))
        result = result.reshape((batch, query_sequence, hidden_size))
    else:
        input_bytes = read_exact(rows * hidden_size * 4)
        per_layer_bytes = read_exact(rows * num_layers * per_layer_width * 4)
        result = mx.array(np.frombuffer(input_bytes, dtype=np.float32).reshape(batch, query_sequence, hidden_size))
        all_per_layer = mx.array(np.frombuffer(per_layer_bytes, dtype=np.float32).reshape(batch, query_sequence, num_layers, per_layer_width))
    position_bytes = read_exact(batch * query_sequence * 4)
    positions = mx.array(np.frombuffer(position_bytes, dtype=np.int32).reshape(batch, query_sequence))
    produced_caches, ordered_caches = {}, []
    all_valid = mx.array(True)
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
            raise ValueError(f"MLX decoder stack layer {layer_index} is invalid")
        mask_bytes = read_exact(batch * mask_heads * query_sequence * total_key_sequence * 4)
        mask_values = np.frombuffer(mask_bytes, dtype=np.float32).reshape(batch, mask_heads, query_sequence, total_key_sequence)
        if np.isnan(mask_values).any() or np.isposinf(mask_values).any():
            raise ValueError("MLX decoder stack mask is invalid")
        mask = mx.array(mask_values)
        if shared:
            if producer_layer not in produced_caches:
                raise ValueError("MLX decoder stack shared cache is unavailable")
            source_key, source_value = produced_caches[producer_layer]
            if source_key.shape != (batch, key_value_heads, source_sequence, head_dim) or source_value.shape != source_key.shape:
                raise ValueError("MLX decoder stack shared cache shape is invalid")
        else:
            source_elements = batch * key_value_heads * source_sequence * head_dim
            source_key_bytes = read_exact(source_elements * 4)
            source_value_bytes = read_exact(source_elements * 4)
            if source_sequence:
                source_key = mx.array(np.frombuffer(source_key_bytes, dtype=np.float32).reshape(batch, key_value_heads, source_sequence, head_dim))
                source_value = mx.array(np.frombuffer(source_value_bytes, dtype=np.float32).reshape(batch, key_value_heads, source_sequence, head_dim))
            else:
                source_key = mx.zeros((batch, key_value_heads, 0, head_dim), dtype=mx.float32)
                source_value = mx.zeros((batch, key_value_heads, 0, head_dim), dtype=mx.float32)
        config = {"query_heads": query_heads, "key_value_heads": key_value_heads, "head_dim": head_dim, "produces_kv": produces_kv, "value_from_key": value_from_key, "intermediate_size": intermediate_size, "per_layer_width": per_layer_width, "input_epsilon": input_epsilon, "attention_epsilon": attention_epsilon, "post_attention_epsilon": post_attention_epsilon, "pre_ffn_epsilon": pre_ffn_epsilon, "post_ffn_epsilon": post_ffn_epsilon, "ple_epsilon": ple_epsilon, "scale": scale, "rope_kind": rope_kind, "theta": theta, "rotary_dim": rotary_dim, "proportional_pairs": proportional_pairs, "proportional_factor": proportional_factor}
        result, key, value, valid = execute_decoder_layer_mlx(pool, shards, result, all_per_layer[:, :, layer_index, :], positions, mask, source_key, source_value, config)
        all_valid = all_valid & valid
        if produces_kv:
            produced_caches[layer_index] = (key, value)
            ordered_caches.append((key, value))
    epilogue = None
    if fused_epilogue:
        vocabulary_size, norm_epsilon, softcap = struct.unpack("<Iff", read_exact(12))
        if not vocabulary_size or not math.isfinite(norm_epsilon) or norm_epsilon <= 0 or not math.isfinite(softcap) or softcap <= 0:
            raise ValueError("MLX decoder stack epilogue metadata is invalid")
        norm_weight = read_whole_tensor(pool, shards, (hidden_size, 1)).reshape((hidden_size,))
        head_weight = read_whole_tensor(pool, shards, (vocabulary_size, hidden_size), widen=False, required_dtype=mx.bfloat16)
        epilogue = (norm_weight, head_weight, norm_epsilon, softcap)
    evaluation = [result, all_valid]
    for key, value in ordered_caches:
        evaluation.extend((key, value))
    mx.eval(*evaluation)
    if not bool(np.asarray(all_valid).item()):
        raise ValueError("MLX decoder stack produced non-finite output")
    logits = None
    if epilogue is not None:
        norm_weight, head_weight, norm_epsilon, softcap = epilogue
        boundary = lambda value: value.astype(mx.bfloat16).astype(mx.float32)
        epilogue_input = result[:, -1:, :] if fused_token_forward else result
        final_hidden = boundary(rms_norm_real(epilogue_input, norm_weight, norm_epsilon))
        raw_logits = mx.matmul(final_hidden.astype(mx.bfloat16), head_weight.T).astype(mx.float32)
        logits = boundary(mx.tanh(boundary(raw_logits / mx.array(softcap, dtype=mx.float32))))
        logits = boundary(logits * mx.array(softcap, dtype=mx.float32))
        logits_valid = mx.all(mx.isfinite(logits))
        mx.eval(logits, logits_valid)
        if not bool(np.asarray(logits_valid).item()):
            raise ValueError("MLX decoder stack epilogue produced non-finite output")
    if not fused_token_forward:
        write_float_tensor(result)
    for key, value in ordered_caches:
        write_float_tensor(key)
        write_float_tensor(value)
    if logits is not None:
        write_float_tensor(logits)
    profile = np.array((0, 0, 0, 0, _widened_tensor_cache_hits - cache_hits_before, len(_widened_tensor_cache), _widened_tensor_cache_bytes), dtype=np.float32)
    write_float_tensor(mx.array(profile))
    sys.stdout.buffer.flush()


def execute_ple_prelude_request(pool, shards, rows, outputs, features):
    metadata = read_exact(32)
    num_layers, per_layer_width, descriptor_count, rounding = struct.unpack("<IIII", metadata[:16])
    context_scale, combine_scale, epsilon = struct.unpack("<fff", metadata[16:28])
    tile_output_rows = struct.unpack("<I", metadata[28:32])[0]
    if (not num_layers or not per_layer_width or outputs != num_layers * per_layer_width
            or descriptor_count != 2 or rounding not in (0, 1) or not tile_output_rows
            or not math.isfinite(context_scale) or not math.isfinite(combine_scale)
            or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("MLX fused PLE prelude metadata is invalid")
    input_bytes = read_exact(rows * features * 4)
    token_bytes = read_exact(rows * outputs * 4)
    inputs = mx.array(np.frombuffer(input_bytes, dtype=np.float32).reshape(rows, features))
    token_identity = mx.array(np.frombuffer(token_bytes, dtype=np.float32).reshape(rows, num_layers, per_layer_width))
    projection_weight = read_whole_tensor(pool, shards, (outputs, features), widen=False)
    norm_weight = read_whole_tensor(pool, shards, (per_layer_width, 1)).reshape((per_layer_width,))
    if rounding == 0:
        result = execute_ple_prelude_values(inputs, token_identity, projection_weight, norm_weight, rows, num_layers, per_layer_width, context_scale, combine_scale, epsilon, tile_output_rows)
    else:
        context = mx.matmul(inputs, projection_weight.T) * mx.array(context_scale, dtype=mx.float32)
        context = context.reshape((rows, num_layers, per_layer_width))
        result = ((rms_norm_real(context, norm_weight, epsilon) + token_identity) * mx.array(combine_scale, dtype=mx.float32)).reshape((rows, outputs))
    valid = mx.all(mx.isfinite(result))
    mx.eval(result, valid)
    if not bool(np.asarray(valid).item()):
        raise ValueError("MLX fused PLE prelude produced non-finite output")
    write_float_tensor(result)
    sys.stdout.buffer.flush()


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
        if not rows or not outputs or not features:
            raise ValueError("MLX worker requires positive dimensions")
        batched = bool(encoded_dtype & 0x40000000)
        fused_mlp = bool(encoded_dtype & 0x20000000)
        native_bf16 = bool(encoded_dtype & 0x10000000)
        native_attention = bool(encoded_dtype & 0x08000000)
        fused_attention = bool(encoded_dtype & 0x04000000)
        dtype_code = encoded_dtype & 0x03ffffff
        if fused_attention:
            raise ValueError("fused attention is available only in the PyTorch worker")
        if native_attention:
            raise ValueError("native attention is available only in the PyTorch worker")
        fused_ple_prelude = bool(encoded_dtype & FUSED_PLE_PRELUDE_FLAG)
        if fused_ple_prelude:
            if referenced or batched or fused_mlp or native_bf16 or native_attention or fused_attention or dtype_code != FUSED_PLE_PRELUDE_FLAG:
                raise ValueError("MLX fused PLE prelude flags are invalid")
            execute_ple_prelude_request(pool, shards, rows, outputs, features)
            continue
        fused_decoder_stack = bool(encoded_dtype & FUSED_DECODER_STACK_FLAG)
        if fused_decoder_stack:
            fused_epilogue = bool(encoded_dtype & FUSED_DECODER_STACK_EPILOGUE_FLAG)
            fused_token_forward = bool(encoded_dtype & FUSED_TOKEN_FORWARD_FLAG)
            expected_code = FUSED_DECODER_STACK_FLAG | (FUSED_DECODER_STACK_EPILOGUE_FLAG if fused_epilogue else 0) | (FUSED_TOKEN_FORWARD_FLAG if fused_token_forward else 0)
            if referenced or batched or fused_mlp or native_attention or fused_attention or dtype_code != expected_code:
                raise ValueError("MLX decoder stack flags are invalid")
            if fused_token_forward and not fused_epilogue:
                raise ValueError("MLX token forward requires the fused epilogue")
            execute_decoder_stack_request(pool, shards, rows, outputs, features, native_bf16, fused_epilogue, fused_token_forward)
            continue
        if not referenced:
            raise ValueError("MLX worker requires a referenced tile")
        input_bytes = read_exact(rows * features * 4)
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
        if native_bf16:
            if expected_dtype != mx.bfloat16:
                raise ValueError("native BF16 GEMM requires BF16 storage")
            inputs = inputs.astype(mx.bfloat16)
        result = mx.matmul(inputs, weight[start_output:start_output + outputs].T)
        if native_bf16:
            result = result.astype(mx.float32)
        mx.eval(result)
        payload = np.asarray(result, dtype=np.float32).tobytes(order="C")
        sys.stdout.buffer.write(struct.pack("<I", len(payload)))
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
        del inputs, result, payload


if __name__ == "__main__":
    main()
