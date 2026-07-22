#!/usr/bin/env python3
"""Persistent MLX Metal worker for F32 inputs and dense F32/F16/BF16 weights."""

import argparse
from collections import OrderedDict
import hashlib
import math
from pathlib import Path
import re
import struct
import sys
import time

import mlx.core as mx
import numpy as np


FUSED_DECODER_STACK_FLAG = 0x00200000
FUSED_DECODER_STACK_EPILOGUE_FLAG = 0x00080000
FUSED_TOKEN_FORWARD_FLAG = 0x00040000
FUSED_TOKEN_GENERATION_FLAG = 0x00020000
COMPILED_TOKEN_GENERATION_FLAG = 0x00010000
SESSION_TOKEN_GENERATION_FLAG = 0x00008000
STREAM_TOKEN_GENERATION_FLAG = 0x00004000
CONTROL_TOKEN_GENERATION_FLAG = 0x00002000
STREAM_TOKEN_FRAME = 0x544F4B4E
FUSED_PLE_PRELUDE_FLAG = 0x01000000
_widened_tensor_cache = {}
_widened_tensor_cache_hits = 0
_widened_tensor_cache_bytes = 0
_shard_sizes = {}
_binary_files = {}
_embedding_row_cache = OrderedDict()
_resident_generation_model = None
_resident_generation_control_model = None
_resident_generation_session = None
_head_quantization = "off"
_decoder_quantization = "off"
_decoder_quantization_layers = None
EMBEDDING_ROW_CACHE_LIMIT = 4096
TOKEN_PRELUDE_CACHE_LIMIT = 1024


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


def write_bytes(payload):
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


def read_bf16_embedding_reference(pool, shards, expected_shape):
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
    return path, tensor_name, byte_offset, second


def materialize_scaled_bf16_embedding_rows(reference, token_ids, scale):
    path, tensor_name, byte_offset, second = reference
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


def read_scaled_bf16_embedding_rows(pool, shards, expected_shape, token_ids, scale):
    return materialize_scaled_bf16_embedding_rows(read_bf16_embedding_reference(pool, shards, expected_shape), token_ids, scale)


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


def rope_real(tensor, positions, rope_kind, theta, rotary_dim, proportional_pairs, proportional_factor, round_bf16=True, factor_context=None, compile_safe=False):
    boundary = (lambda value: value.astype(mx.bfloat16).astype(mx.float32)) if round_bf16 else (lambda value: value)
    head_dim, half = tensor.shape[-1], rotary_dim // 2
    factor_key = (head_dim, rope_kind, theta, rotary_dim, proportional_pairs, proportional_factor, round_bf16)
    factors = factor_context["factors"].get(factor_key) if factor_context is not None else None
    if factor_context is not None:
        factor_context["uses"] += 1
    if factors is None:
        pairs = mx.arange(half, dtype=mx.float32)
        denominator_width = head_dim if rope_kind == 1 else rotary_dim
        denominator = mx.power(mx.array(theta, dtype=mx.float32), (mx.array(2.0, dtype=mx.float32) * pairs) / mx.array(denominator_width, dtype=mx.float32))
        if rope_kind == 1:
            denominator = denominator * mx.array(proportional_factor, dtype=mx.float32)
        angles = positions[:, None, :, None].astype(mx.float32) / denominator[None, None, None, :]
        if rope_kind == 1 and proportional_pairs < half:
            angles = mx.where((pairs < proportional_pairs)[None, None, None, :], angles, mx.zeros_like(angles))
        factors = boundary(mx.cos(angles)), boundary(mx.sin(angles))
        if factor_context is not None:
            factor_context["factors"][factor_key] = factors
            factor_context["builds"] += 1
    cosine, sine = factors
    if compile_safe:
        first = mx.take(tensor, mx.array(np.arange(half, dtype=np.int32)), axis=-1)
        second = mx.take(tensor, mx.array(np.arange(half, rotary_dim, dtype=np.int32)), axis=-1)
        remainder = None if rotary_dim == head_dim else mx.take(tensor, mx.array(np.arange(rotary_dim, head_dim, dtype=np.int32)), axis=-1)
    else:
        first, second = tensor[..., :half], tensor[..., half:rotary_dim]
        remainder = None if rotary_dim == head_dim else tensor[..., rotary_dim:]
    rotated_first = boundary(boundary(first * cosine) - boundary(second * sine))
    rotated_second = boundary(boundary(second * cosine) + boundary(first * sine))
    return mx.concatenate((rotated_first, rotated_second), axis=-1) if remainder is None else mx.concatenate((rotated_first, rotated_second, remainder), axis=-1)


def read_decoder_layer_weights(pool, shards, hidden_size, config):
    query_heads, key_value_heads, head_dim = config["query_heads"], config["key_value_heads"], config["head_dim"]
    intermediate_size, per_layer_width = config["intermediate_size"], config["per_layer_width"]
    weights = {
        "input_norm": read_whole_tensor(pool, shards, (hidden_size, 1)).reshape((hidden_size,)),
        "query": read_whole_tensor(pool, shards, (query_heads * head_dim, hidden_size), widen=False, required_dtype=mx.bfloat16),
        "query_norm": read_whole_tensor(pool, shards, (head_dim, 1)).reshape((head_dim,)),
        "output": read_whole_tensor(pool, shards, (hidden_size, query_heads * head_dim), widen=False, required_dtype=mx.bfloat16),
    }
    if config["produces_kv"]:
        weights["key"] = read_whole_tensor(pool, shards, (key_value_heads * head_dim, hidden_size), widen=False, required_dtype=mx.bfloat16)
        weights["key_norm"] = read_whole_tensor(pool, shards, (head_dim, 1)).reshape((head_dim,))
        if not config["value_from_key"]:
            weights["value"] = read_whole_tensor(pool, shards, (key_value_heads * head_dim, hidden_size), widen=False, required_dtype=mx.bfloat16)
    weights.update({
        "post_attention_norm": read_whole_tensor(pool, shards, (hidden_size, 1)).reshape((hidden_size,)),
        "pre_ffn_norm": read_whole_tensor(pool, shards, (hidden_size, 1)).reshape((hidden_size,)),
        "gate": read_whole_tensor(pool, shards, (intermediate_size, hidden_size), widen=False, required_dtype=mx.bfloat16),
        "up": read_whole_tensor(pool, shards, (intermediate_size, hidden_size), widen=False, required_dtype=mx.bfloat16),
        "down": read_whole_tensor(pool, shards, (hidden_size, intermediate_size), widen=False, required_dtype=mx.bfloat16),
        "post_ffn_norm": read_whole_tensor(pool, shards, (hidden_size, 1)).reshape((hidden_size,)),
        "ple_gate": read_whole_tensor(pool, shards, (per_layer_width, hidden_size)),
        "ple_projection": read_whole_tensor(pool, shards, (hidden_size, per_layer_width)),
        "ple_norm": read_whole_tensor(pool, shards, (hidden_size, 1)).reshape((hidden_size,)),
        "layer_scalar": read_whole_tensor(pool, shards, (1, 1)).reshape(()),
    })
    if _decoder_quantization == "q8-ffn-gate-up-down":
        weights["down_control"] = weights["down"]
    selected = set()
    quantize_gate_up = _decoder_quantization in ("q8-ffn", "q8-all", "q8-ffn-gate-up-attention", "q8-ffn-gate-up-down")
    quantize_gate_up = quantize_gate_up or (_decoder_quantization == "q8-ffn-gate-up" and (_decoder_quantization_layers is None or config["layer_index"] in _decoder_quantization_layers))
    quantize_gate_up = quantize_gate_up or _decoder_quantization == "q4-ffn-gate-up"
    quantize_gate_up = quantize_gate_up or (_decoder_quantization == "q8-ffn-gate-up-first-half" and config["layer_index"] < 21)
    quantize_gate_up = quantize_gate_up or (_decoder_quantization == "q8-ffn-gate-up-last-half" and config["layer_index"] >= 21)
    if quantize_gate_up:
        selected.update(("gate", "up"))
    quantize_down = _decoder_quantization in ("q8-ffn", "q8-all", "q8-ffn-down")
    quantize_down = quantize_down or (_decoder_quantization == "q8-ffn-gate-up-down" and (_decoder_quantization_layers is None or config["layer_index"] in _decoder_quantization_layers))
    if quantize_down:
        selected.add("down")
    if _decoder_quantization in ("q8-attention", "q8-all", "q8-ffn-gate-up-attention"):
        selected.update(("query", "key", "value", "output"))
    for name in sorted(selected.intersection(weights)):
        q4_layer = _decoder_quantization == "q4-ffn-gate-up" and (_decoder_quantization_layers is None or config["layer_index"] in _decoder_quantization_layers)
        bits = 4 if q4_layer and name in ("gate", "up") else 8
        quantized, scales, biases = mx.quantize(weights[name], group_size=64, bits=bits)
        mx.eval(quantized, scales, biases)
        weights[name] = (quantized, scales, biases, 64, bits)
    if isinstance(weights["gate"], tuple) and isinstance(weights["up"], tuple):
        gate_up = tuple(mx.concatenate((weights["gate"][index], weights["up"][index]), axis=0) for index in range(3))
        mx.eval(*gate_up)
        weights["gate_up"] = (*gate_up, 64, weights["gate"][4])
        del weights["gate"]
        del weights["up"]
    return weights


def matrix_contract(value):
    if isinstance(value, tuple):
        extra = tuple((tuple(entry.shape), str(entry.dtype)) for entry in value[5:])
        return (tuple((tuple(entry.shape), str(entry.dtype)) for entry in value[:3]), value[3], value[4], extra)
    return (tuple(value.shape), str(value.dtype))


def matrix_project(value, weight):
    if isinstance(weight, tuple):
        quantized, scales, biases, group_size, bits = weight[:5]
        projected = mx.quantized_matmul(value, quantized, scales, biases, transpose=True, group_size=group_size, bits=bits).astype(mx.float32)
        if len(weight) == 11:
            correction_quantized, correction_scales, correction_biases = weight[5:8]
            projected = projected + mx.quantized_matmul(value, correction_quantized, correction_scales, correction_biases, transpose=True, group_size=group_size, bits=bits).astype(mx.float32)
        return projected
    return mx.matmul(value, weight.T).astype(mx.float32)


def execute_decoder_layer_mlx(pool, shards, inputs, per_layer, positions, mask, source_key, source_value, config, weights=None, rope_factor_context=None, source_cache_validated=False, compile_safe_rope=False, incremental_fixed_shape=False, defer_result_validation=False):
    batch, query_sequence, hidden_size = (1, 1, weights["input_norm"].shape[0]) if incremental_fixed_shape else inputs.shape
    query_heads, key_value_heads, head_dim = config["query_heads"], config["key_value_heads"], config["head_dim"]
    produces_kv, value_from_key = config["produces_kv"], config["value_from_key"]
    intermediate_size, per_layer_width = config["intermediate_size"], config["per_layer_width"]
    real = config.get("rounding") == "real"
    boundary = (lambda value: value) if real else (lambda value: value.astype(mx.bfloat16).astype(mx.float32))
    project = (lambda value, weight: matrix_project(value, weight)) if real else (lambda value, weight: matrix_project(value.astype(mx.bfloat16), weight))
    if weights is None:
        weights = read_decoder_layer_weights(pool, shards, hidden_size, config)
    input_norm_weight, query_weight, query_norm, output_weight = weights["input_norm"], weights["query"], weights["query_norm"], weights["output"]
    normalized_input = boundary(rms_norm_real(inputs, input_norm_weight, config["input_epsilon"]))
    query = mx.transpose(boundary(project(normalized_input, query_weight)).reshape((batch, query_sequence, query_heads, head_dim)), (0, 2, 1, 3))
    query = rope_real(boundary(rms_norm_real(query, query_norm, config["attention_epsilon"])), positions, config["rope_kind"], config["theta"], config["rotary_dim"], config["proportional_pairs"], config["proportional_factor"], not real, rope_factor_context, compile_safe_rope)
    if produces_kv:
        key_weight, key_norm = weights["key"], weights["key_norm"]
        current_key_heads = mx.transpose(boundary(project(normalized_input, key_weight)).reshape((batch, query_sequence, key_value_heads, head_dim)), (0, 2, 1, 3))
        current_key = rope_real(boundary(rms_norm_real(current_key_heads, key_norm, config["attention_epsilon"])), positions, config["rope_kind"], config["theta"], config["rotary_dim"], config["proportional_pairs"], config["proportional_factor"], not real, rope_factor_context, compile_safe_rope)
        if value_from_key:
            current_value = boundary(rms_norm_real(current_key_heads, None, config["attention_epsilon"]))
        else:
            value_weight = weights["value"]
            current_value = mx.transpose(boundary(project(normalized_input, value_weight)).reshape((batch, query_sequence, key_value_heads, head_dim)), (0, 2, 1, 3))
            current_value = boundary(rms_norm_real(current_value, None, config["attention_epsilon"]))
        key = mx.concatenate((source_key, current_key), axis=2)
        value = mx.concatenate((source_value, current_value), axis=2)
    else:
        key, value = source_key, source_value
    group = query_heads // key_value_heads
    if incremental_fixed_shape and group != 1:
        grouped_query = query.reshape((1, key_value_heads, group, 1, head_dim))
        grouped_key, grouped_value = mx.expand_dims(key, axis=2), mx.expand_dims(value, axis=2)
        grouped_mask = mask if config["mask_heads"] == 1 else mask.reshape((1, key_value_heads, group, 1, -1))
        scores = mx.matmul(grouped_query, mx.swapaxes(grouped_key, -1, -2)) * mx.array(config["scale"], dtype=mx.float32) + grouped_mask
        probabilities = mx.softmax(scores, axis=-1)
        attention_context = mx.matmul(probabilities, grouped_value).reshape((1, query_heads, 1, head_dim))
    else:
        attention_key = key if group == 1 else mx.repeat(key, group, axis=1)
        attention_value = value if group == 1 else mx.repeat(value, group, axis=1)
        scores = mx.matmul(query, mx.swapaxes(attention_key, -1, -2)) * mx.array(config["scale"], dtype=mx.float32) + mask
        probabilities = mx.softmax(scores, axis=-1)
        attention_context = mx.matmul(probabilities, attention_value)
    context = mx.transpose(attention_context, (0, 2, 1, 3)).reshape((batch, query_sequence, query_heads * head_dim))
    attention_projected = boundary(project(context, output_weight))
    post_attention_norm_weight, pre_ffn_norm_weight = weights["post_attention_norm"], weights["pre_ffn_norm"]
    gate_weight, up_weight, down_weight, post_ffn_norm_weight = weights.get("gate"), weights.get("up"), weights["down"], weights["post_ffn_norm"]
    after_attention = boundary(inputs + boundary(rms_norm_real(attention_projected, post_attention_norm_weight, config["post_attention_epsilon"])))
    ffn_input = boundary(rms_norm_real(after_attention, pre_ffn_norm_weight, config["pre_ffn_epsilon"]))
    if not real:
        ffn_input = ffn_input.astype(mx.bfloat16)
    if "gate_up" in weights:
        gate_up = matrix_project(ffn_input, weights["gate_up"])
        if incremental_fixed_shape:
            paired = gate_up.reshape((1, 1, 2, intermediate_size))
            gate = mx.take(paired, mx.array(0, dtype=mx.int32), axis=2)
            up = mx.take(paired, mx.array(1, dtype=mx.int32), axis=2)
        else:
            gate, up = mx.split(gate_up, 2, axis=-1)
    else:
        if gate_weight is None or up_weight is None:
            raise ValueError("MLX decoder gate/up projection contract is incomplete")
        gate, up = matrix_project(ffn_input, gate_weight), matrix_project(ffn_input, up_weight)
    cube = (gate * gate) * gate
    inner = mx.array(math.sqrt(2 / math.pi), dtype=mx.float32) * (gate + mx.array(0.044715, dtype=mx.float32) * cube)
    hidden = (mx.array(0.5, dtype=mx.float32) * gate) * (mx.array(1.0, dtype=mx.float32) + mx.tanh(inner)) * up
    ffn_projected = matrix_project(hidden if real else hidden.astype(mx.bfloat16), down_weight)
    after_mlp = boundary(after_attention + boundary(rms_norm_real(ffn_projected, post_ffn_norm_weight, config["post_ffn_epsilon"])))
    ple_gate_weight, ple_projection_weight, ple_norm_weight, layer_scalar = weights["ple_gate"], weights["ple_projection"], weights["ple_norm"], weights["layer_scalar"]
    ple_gate = boundary(mx.matmul(after_mlp, ple_gate_weight.T))
    cube = (ple_gate * ple_gate) * ple_gate
    inner = mx.array(math.sqrt(2 / math.pi), dtype=mx.float32) * (ple_gate + mx.array(0.044715, dtype=mx.float32) * cube)
    ple_activated = boundary((mx.array(0.5, dtype=mx.float32) * ple_gate) * (mx.array(1.0, dtype=mx.float32) + mx.tanh(inner)))
    ple_projected = boundary(mx.matmul(boundary(ple_activated * per_layer), ple_projection_weight.T))
    ple_normalized = boundary(rms_norm_real(ple_projected, ple_norm_weight, config["ple_epsilon"]))
    result = boundary(boundary(after_mlp + ple_normalized) * layer_scalar)
    valid = mx.array(True) if defer_result_validation else mx.all(mx.isfinite(result))
    if produces_kv:
        validation_key = current_key if source_cache_validated else key
        validation_value = current_value if source_cache_validated else value
        valid = valid & mx.all(mx.isfinite(validation_key)) & mx.all(mx.isfinite(validation_value))
    return result, key, value, valid, weights


def compile_incremental_decoder_step(pool, shards, layer_plans, epilogue, rounding):
    producer_layers = tuple(config["layer_index"] for config, _ in layer_plans if config["produces_kv"])
    producer_offsets = {layer_index: offset for offset, layer_index in enumerate(producer_layers)}

    def execute(inputs, per_layer_inputs, positions, source_keys, source_values, masks):
        result, next_keys, next_values, produced = inputs, [], [], {}
        all_valid = mx.array(True)
        rope_factor_context = {"factors": {}, "builds": 0, "uses": 0}
        for layer_index, (config, weights) in enumerate(layer_plans):
            source_layer = layer_index if config["produces_kv"] else config["producer_layer"]
            if config["produces_kv"]:
                source_offset = producer_offsets[source_layer]
                source_key, source_value = source_keys[source_offset], source_values[source_offset]
            else:
                source_key, source_value = produced[source_layer]
            result, key, value, valid, _ = execute_decoder_layer_mlx(pool, shards, result, per_layer_inputs[layer_index], positions, masks[layer_index], source_key, source_value, config, weights, rope_factor_context, True, True, True, True)
            all_valid = all_valid & valid
            if config["produces_kv"]:
                produced[layer_index] = (key, value)
                next_keys.append(key)
                next_values.append(value)
        # A non-finite residual propagates through the remaining normalization,
        # projection, and residual chain. Validate the terminal state once and
        # only the newly appended KV rows before they become reusable cache.
        all_valid = all_valid & mx.all(mx.isfinite(result))
        raw_logits = execute_decoder_head(result, epilogue, rounding=rounding)
        logits = finalize_decoder_logits(raw_logits, epilogue, rounding=rounding)
        return result, raw_logits, logits, tuple(next_keys), tuple(next_values), all_valid

    return mx.compile(execute, shapeless=True), producer_layers


def incremental_compile_signature(layer_plans, epilogue, rounding):
    topology = tuple(tuple(sorted(config.items())) for config, _ in layer_plans)
    weight_contracts = tuple(tuple((name, matrix_contract(value)) for name, value in sorted(weights.items())) for _, weights in layer_plans)
    head_contract = matrix_contract(epilogue[1])
    epilogue_contract = (tuple(epilogue[0].shape), str(epilogue[0].dtype), head_contract, epilogue[2], epilogue[3])
    return rounding, topology, weight_contracts, epilogue_contract


def execute_decoder_head(result, epilogue, terminal_only=True, rounding="native-bf16"):
    norm_weight, head_weight, norm_epsilon, softcap = epilogue
    real = rounding == "real"
    boundary = (lambda value: value) if real else (lambda value: value.astype(mx.bfloat16).astype(mx.float32))
    epilogue_input = result[:, -1:, :] if terminal_only else result
    final_hidden = boundary(rms_norm_real(epilogue_input, norm_weight, norm_epsilon))
    head_input = final_hidden if real else final_hidden.astype(mx.bfloat16)
    return matrix_project(head_input, head_weight)


def finalize_decoder_logits(raw_logits, epilogue, rounding="native-bf16"):
    softcap = epilogue[3]
    real = rounding == "real"
    boundary = (lambda value: value) if real else (lambda value: value.astype(mx.bfloat16).astype(mx.float32))
    logits = boundary(mx.tanh(boundary(raw_logits / mx.array(softcap, dtype=mx.float32))))
    result = boundary(logits * mx.array(softcap, dtype=mx.float32))
    return result.astype(mx.bfloat16).astype(mx.float32) if real else result


def execute_decoder_epilogue(result, epilogue, terminal_only=True, rounding="native-bf16"):
    return finalize_decoder_logits(execute_decoder_head(result, epilogue, terminal_only, rounding), epilogue, rounding)


def build_quantized_head(head_weight, group_size, bits, chunk_rows=4096):
    quantized, scales, biases = mx.quantize(head_weight, group_size=group_size, bits=bits)
    mx.eval(quantized, scales, biases)
    correction_chunks, error_chunks = [], []
    for first in range(0, head_weight.shape[0], chunk_rows):
        last = min(head_weight.shape[0], first + chunk_rows)
        dequantized = mx.dequantize(quantized[first:last], scales[first:last], biases[first:last], group_size=group_size, bits=bits, dtype=mx.float32)
        exact = head_weight[first:last].astype(mx.float32)
        residual = exact - dequantized
        correction_quantized, correction_scales, correction_biases = mx.quantize(residual, group_size=group_size, bits=bits)
        correction = mx.dequantize(correction_quantized, correction_scales, correction_biases, group_size=group_size, bits=bits, dtype=mx.float32)
        remaining = (residual - correction).reshape((last - first, head_weight.shape[1] // group_size, group_size))
        group_max_errors = mx.max(mx.abs(remaining), axis=2)
        group_l2_errors = mx.sqrt(mx.sum(remaining * remaining, axis=2))
        mx.eval(correction_quantized, correction_scales, correction_biases, group_max_errors, group_l2_errors)
        correction_chunks.append((correction_quantized, correction_scales, correction_biases))
        error_chunks.append((group_max_errors, group_l2_errors))
    correction_quantized = mx.concatenate(tuple(entry[0] for entry in correction_chunks), axis=0)
    correction_scales = mx.concatenate(tuple(entry[1] for entry in correction_chunks), axis=0)
    correction_biases = mx.concatenate(tuple(entry[2] for entry in correction_chunks), axis=0)
    max_errors = mx.concatenate(tuple(entry[0] for entry in error_chunks), axis=0)
    l2_errors = mx.concatenate(tuple(entry[1] for entry in error_chunks), axis=0)
    mx.eval(correction_quantized, correction_scales, correction_biases, max_errors, l2_errors)
    return quantized, scales, biases, group_size, bits, correction_quantized, correction_scales, correction_biases, head_weight, max_errors, l2_errors


def build_shortlist_quantized_head(head_weight, group_size=64, bits=8):
    quantized, scales, biases = mx.quantize(head_weight, group_size=group_size, bits=bits)
    mx.eval(quantized, scales, biases, head_weight)
    return quantized, scales, biases, group_size, bits, head_weight


def quantized_head_error_bounds(result, epilogue, rounding):
    norm_weight, head_weight, norm_epsilon, softcap = epilogue
    if not isinstance(head_weight, tuple) or len(head_weight) != 11:
        return None
    group_size, exact_weight, max_errors, l2_errors = head_weight[3], head_weight[8], head_weight[9], head_weight[10]
    real = rounding == "real"
    boundary = (lambda value: value) if real else (lambda value: value.astype(mx.bfloat16).astype(mx.float32))
    final_hidden = boundary(rms_norm_real(result[:, -1:, :], norm_weight, norm_epsilon))
    head_input = final_hidden if real else final_hidden.astype(mx.bfloat16)
    grouped = head_input.astype(mx.float32).reshape((1, 1, exact_weight.shape[1] // group_size, group_size))
    grouped_l1 = mx.sum(mx.abs(grouped), axis=3)
    grouped_l2 = mx.sqrt(mx.sum(grouped * grouped, axis=3))
    linf_bound = mx.matmul(grouped_l1, max_errors.T)
    l2_bound = mx.matmul(grouped_l2, l2_errors.T)
    return mx.minimum(linf_bound, l2_bound)


def rank_exact_logits(logits, top_k):
    values = logits.reshape((-1,))
    selected_value = mx.argmax(values)
    candidates_value = mx.argpartition(values, -top_k)[-top_k:]
    candidate_logits_value = values[candidates_value]
    finite_value = mx.all(mx.isfinite(values))
    mx.eval(selected_value, candidates_value, candidate_logits_value, finite_value)
    if not bool(np.asarray(finite_value).item()):
        raise ValueError("MLX resident generation produced non-finite logits")
    selected = int(np.asarray(selected_value).item())
    candidates = np.asarray(candidates_value, dtype=np.int32)
    candidate_logits = np.asarray(candidate_logits_value, dtype=np.float32)
    ordered = np.lexsort((candidates, -candidate_logits))
    return selected, candidates[ordered], candidate_logits[ordered]


def incremental_topology_mask(config, key_sequence, absolute_position):
    values = np.zeros((1, config["mask_heads"], 1, key_sequence), dtype=np.float32)
    first_key = max(0, absolute_position - config["sliding_window"] + 1) if config["sliding_window"] else 0
    last_key = min(absolute_position, key_sequence - 1) if config["causal"] else key_sequence - 1
    if first_key:
        values[..., :first_key] = -np.inf
    if last_key + 1 < key_sequence:
        values[..., last_key + 1:] = -np.inf
    return mx.array(values)


def rank_terminal_logits(logits, raw_logits, top_k, result=None, epilogue=None, rounding="native-bf16"):
    if result is not None and epilogue is not None and isinstance(epilogue[1], tuple) and len(epilogue[1]) == 6:
        values = logits.reshape((-1,))
        shortlist_size = min(values.size, max(top_k, 16))
        candidates_value = mx.argpartition(values, -shortlist_size)[-shortlist_size:]
        exact_weight = epilogue[1][5]
        exact_epilogue = (epilogue[0], mx.take(exact_weight, candidates_value, axis=0), epilogue[2], epilogue[3])
        exact_values = finalize_decoder_logits(execute_decoder_head(result, exact_epilogue, rounding=rounding), exact_epilogue, rounding=rounding).reshape((-1,))
        mx.eval(candidates_value, exact_values)
        candidate_ids = np.asarray(candidates_value, dtype=np.int32)
        candidate_logits = np.asarray(exact_values, dtype=np.float32)
        ordered = np.lexsort((candidate_ids, -candidate_logits))
        return int(candidate_ids[ordered[0]]), candidate_ids[ordered[:top_k]], candidate_logits[ordered[:top_k]], logits, False, False
    error_bounds = None if result is None or epilogue is None else quantized_head_error_bounds(result, epilogue, rounding)
    if error_bounds is None:
        selected, candidates, candidate_logits = rank_exact_logits(logits, top_k)
        return selected, candidates, candidate_logits, logits, False, False
    values = logits.reshape((-1,))
    certificate_k = min(values.size, max(top_k, 16))
    candidates_value = mx.argpartition(values, -certificate_k)[-certificate_k:]
    exact_weight = epilogue[1][8]
    candidate_weight = mx.take(exact_weight, candidates_value, axis=0)
    exact_epilogue = (epilogue[0], candidate_weight, epilogue[2], epilogue[3])
    exact_candidate_raw = execute_decoder_head(result, exact_epilogue, rounding=rounding)
    exact_candidate_values = finalize_decoder_logits(exact_candidate_raw, exact_epilogue, rounding=rounding).reshape((-1,))
    best_offset = mx.argmax(exact_candidate_values)
    best_token = candidates_value[best_offset]
    best_logit = exact_candidate_values[best_offset]
    upper_bounds = finalize_decoder_logits(raw_logits + error_bounds, epilogue, rounding=rounding).reshape((-1,))
    vocabulary = mx.arange(values.size, dtype=mx.int32)
    maximum_other = mx.max(mx.where(vocabulary == best_token, mx.array(-math.inf, dtype=mx.float32), upper_bounds))
    finite_value = mx.all(mx.isfinite(values)) & mx.all(mx.isfinite(error_bounds)) & mx.all(mx.isfinite(exact_candidate_values))
    mx.eval(best_token, best_logit, maximum_other, candidates_value, exact_candidate_values, finite_value)
    if not bool(np.asarray(finite_value).item()):
        raise ValueError("MLX certified quantized head produced non-finite values")
    certified = float(np.asarray(best_logit).item()) > float(np.asarray(maximum_other).item())
    if not certified:
        exact_epilogue = (epilogue[0], exact_weight, epilogue[2], epilogue[3])
        exact_logits = execute_decoder_epilogue(result, exact_epilogue, rounding=rounding)
        selected, candidates, candidate_logits = rank_exact_logits(exact_logits, top_k)
        return selected, candidates, candidate_logits, exact_logits, False, True
    candidate_ids = np.asarray(candidates_value, dtype=np.int32)
    exact_values = np.asarray(exact_candidate_values, dtype=np.float32)
    ordered = np.lexsort((candidate_ids, -exact_values))[:top_k]
    selected = int(np.asarray(best_token).item())
    return selected, candidate_ids[ordered], exact_values[ordered], logits, True, False


def prefill_topology_mask(config, query_sequence):
    values = np.zeros((1, config["mask_heads"], query_sequence, query_sequence), dtype=np.float32)
    for query in range(query_sequence):
        first_key = max(0, query - config["sliding_window"] + 1) if config["sliding_window"] else 0
        last_key = query if config["causal"] else query_sequence - 1
        if first_key:
            values[..., query, :first_key] = -np.inf
        if last_key + 1 < query_sequence:
            values[..., query, last_key + 1:] = -np.inf
    return mx.array(values)


def execute_cached_incremental_token_prelude(model, token_id):
    cache = model["token_prelude_cache"]
    cached = cache.get(token_id)
    if cached is not None:
        cache.move_to_end(token_id)
        return cached[0], cached[1], True
    token_ids = np.array([[token_id]], dtype=np.int32)
    result, all_per_layer = model["execute_token_prelude"](token_ids)
    cache[token_id] = (result, all_per_layer)
    if len(cache) > TOKEN_PRELUDE_CACHE_LIMIT:
        cache.popitem(last=False)
    return result, all_per_layer, False


def cache_prefill_token_prelude_rows(model, token_ids, result, all_per_layer):
    cache = model["token_prelude_cache"]
    for index, token_id in enumerate(np.asarray(token_ids, dtype=np.int32).reshape(-1)):
        token = int(token_id)
        cache[token] = (result[:, index:index + 1, :], all_per_layer[:, index:index + 1, :, :])
        cache.move_to_end(token)
    while len(cache) > TOKEN_PRELUDE_CACHE_LIMIT:
        cache.popitem(last=False)


def emit_resident_generation(model, result, produced_caches, raw_logits, logits, first_forward_seconds, max_new_tokens, eos_token_id, top_k, cache_hits_before, input_token_ids, session_id=None, stream=False, prefix_tokens_reused=0, prefill_tokens_computed=None, initial_rope_factor_context=None, initial_topology_mask_builds=0, initial_topology_mask_uses=0, initial_kv_prefix_validation_scans_avoided=0):
    global _resident_generation_session
    generated_ids, forward_seconds, top_ids, top_values = [], [first_forward_seconds], [], []
    terminal_values = None
    rope_factor_builds = 0 if initial_rope_factor_context is None else initial_rope_factor_context["builds"]
    rope_factor_uses = 0 if initial_rope_factor_context is None else initial_rope_factor_context["uses"]
    topology_mask_builds, topology_mask_uses = initial_topology_mask_builds, initial_topology_mask_uses
    kv_prefix_validation_scans_avoided = initial_kv_prefix_validation_scans_avoided
    quantized_head_certified_steps, quantized_head_exact_fallback_steps = 0, 0
    token_selection_seconds, terminal_logit_transfer_seconds = 0.0, 0.0
    token_prelude_cache_hits, token_prelude_cache_misses = 0, 0
    token_prelude_seconds, compiled_decoder_graph_seconds = 0.0, 0.0
    while len(generated_ids) < max_new_tokens:
        selection_started = time.perf_counter()
        token_id, ranked_ids, ranked_values, selected_logits, certified, exact_fallback = rank_terminal_logits(logits, raw_logits, top_k, result, model["epilogue"], model["rounding"])
        token_selection_seconds += time.perf_counter() - selection_started
        quantized_head_certified_steps += int(certified)
        quantized_head_exact_fallback_steps += int(exact_fallback)
        generated_ids.append(token_id)
        top_ids.append(ranked_ids)
        top_values.append(ranked_values)
        if stream:
            sys.stdout.buffer.write(struct.pack("<IIf", STREAM_TOKEN_FRAME, token_id, forward_seconds[-1]))
            sys.stdout.buffer.write(ranked_ids.astype(np.int32).tobytes(order="C"))
            sys.stdout.buffer.write(ranked_values.astype(np.float32).tobytes(order="C"))
            sys.stdout.buffer.flush()
        if len(generated_ids) == max_new_tokens or token_id == eos_token_id:
            transfer_started = time.perf_counter()
            terminal_values = np.asarray(selected_logits, dtype=np.float32).reshape(-1)
            terminal_logit_transfer_seconds += time.perf_counter() - transfer_started
            break
        incremental_started = time.perf_counter()
        prelude_started = time.perf_counter()
        result, all_per_layer, prelude_cache_hit = execute_cached_incremental_token_prelude(model, token_id)
        token_prelude_seconds += time.perf_counter() - prelude_started
        token_prelude_cache_hits += int(prelude_cache_hit)
        token_prelude_cache_misses += int(not prelude_cache_hit)
        absolute_position = model["prompt_length"] + len(generated_ids) - 1
        positions = mx.array(np.array([[absolute_position]], dtype=np.int32))
        topology_masks, layer_masks = {}, []
        for layer_index, (config, weights) in enumerate(model["layer_plans"]):
            source_layer = layer_index if config["produces_kv"] else config["producer_layer"]
            source_key = produced_caches[source_layer][0]
            key_sequence = source_key.shape[2] + 1
            topology_key = (config["mask_heads"], key_sequence, config["sliding_window"], config["causal"])
            mask = topology_masks.get(topology_key)
            topology_mask_uses += 1
            if mask is None:
                mask = incremental_topology_mask(config, key_sequence, absolute_position)
                topology_masks[topology_key] = mask
                topology_mask_builds += 1
            layer_masks.append(mask)
        source_keys = tuple(produced_caches[layer][0] for layer in model["producer_layers"])
        source_values = tuple(produced_caches[layer][1] for layer in model["producer_layers"])
        per_layer_inputs = tuple(all_per_layer[:, :, layer_index, :] for layer_index in range(len(model["layer_plans"])))
        decoder_graph_started = time.perf_counter()
        result, raw_logits, logits, next_keys, next_values, all_valid = model["execute_incremental_decoder_step"](result, per_layer_inputs, positions, source_keys, source_values, tuple(layer_masks))
        produced_caches = {layer: (next_keys[offset], next_values[offset]) for offset, layer in enumerate(model["producer_layers"])}
        rope_factor_builds += model["rope_factor_builds_per_step"]
        rope_factor_uses += model["rope_factor_uses_per_step"]
        kv_prefix_validation_scans_avoided += sum(1 for key in source_keys if key.shape[2])
        evaluation = [raw_logits, logits, all_valid]
        for key, value in produced_caches.values():
            evaluation.extend((key, value))
        mx.eval(*evaluation)
        if not bool(np.asarray(all_valid).item()):
            raise ValueError("MLX resident generation produced non-finite state")
        compiled_decoder_graph_seconds += time.perf_counter() - decoder_graph_started
        forward_seconds.append(time.perf_counter() - incremental_started)
    resident_kv_bytes = sum((key.size + value.size) * 4 for key, value in produced_caches.values())
    if session_id is not None:
        cached_tokens = np.concatenate((np.asarray(input_token_ids, dtype=np.int32).reshape(-1), np.asarray(generated_ids[:-1], dtype=np.int32)))
        _resident_generation_session = {"id": session_id, "token_ids": cached_tokens, "caches": produced_caches}
    if stream:
        sys.stdout.buffer.write(struct.pack("<I", 0))
    else:
        write_bytes(np.asarray(generated_ids, dtype=np.int32).tobytes(order="C"))
        write_bytes(np.asarray(forward_seconds, dtype=np.float32).tobytes(order="C"))
        write_bytes(np.stack(top_ids).astype(np.int32).tobytes(order="C"))
        write_bytes(np.stack(top_values).astype(np.float32).tobytes(order="C"))
    write_bytes(hashlib.sha256(np.asarray(terminal_values, dtype=np.float32).tobytes(order="C")).digest())
    computed = len(np.asarray(input_token_ids).reshape(-1)) if prefill_tokens_computed is None else prefill_tokens_computed
    cached_context_tokens = len(_resident_generation_session["token_ids"]) if session_id is not None else len(np.asarray(input_token_ids).reshape(-1)) + len(generated_ids) - 1
    compiled_incremental_decoder_steps = max(0, len(generated_ids) - 1)
    decoder_layer_validity_scans_avoided = compiled_incremental_decoder_steps * max(0, len(model["layer_plans"]) - 1)
    profile = np.array((1, len(generated_ids), max(0, len(generated_ids) - 1), terminal_values.nbytes, rope_factor_builds, max(0, rope_factor_uses - rope_factor_builds), topology_mask_builds, max(0, topology_mask_uses - topology_mask_builds), len(generated_ids), kv_prefix_validation_scans_avoided, compiled_incremental_decoder_steps, 1 if model["incremental_compiler_cache_hit"] else 0, _widened_tensor_cache_hits - cache_hits_before, len(_widened_tensor_cache), _widened_tensor_cache_bytes, resident_kv_bytes, prefix_tokens_reused, computed, 1 if prefix_tokens_reused else 0, cached_context_tokens, quantized_head_certified_steps, quantized_head_exact_fallback_steps, token_prelude_cache_hits, token_prelude_cache_misses, first_forward_seconds, sum(forward_seconds[1:]), token_prelude_seconds, compiled_decoder_graph_seconds, token_selection_seconds, terminal_logit_transfer_seconds, decoder_layer_validity_scans_avoided), dtype=np.float32)
    write_float_tensor(mx.array(profile))
    sys.stdout.buffer.flush()


def continuation_topology_mask(config, query_sequence, key_sequence, absolute_start):
    values = np.zeros((1, config["mask_heads"], query_sequence, key_sequence), dtype=np.float32)
    for query in range(query_sequence):
        absolute_position = absolute_start + query
        first_key = max(0, absolute_position - config["sliding_window"] + 1) if config["sliding_window"] else 0
        last_key = min(absolute_position, key_sequence - 1) if config["causal"] else key_sequence - 1
        if first_key:
            values[..., query, :first_key] = -np.inf
        if last_key + 1 < key_sequence:
            values[..., query, last_key + 1:] = -np.inf
    return mx.array(values)


def execute_compiled_token_generation(pool, shards, token_ids, max_new_tokens, eos_token_id, top_k, session_id=None, stream=False, control=False):
    model = _resident_generation_control_model if control else _resident_generation_model
    if model is None:
        raise ValueError("MLX resident control generation model is not compiled" if control else "MLX resident generation model is not compiled")
    if control and session_id is not None:
        raise ValueError("MLX resident control generation does not accept a session id")
    if (token_ids < 0).any() or (token_ids >= model["vocabulary_size"]).any():
        raise ValueError("MLX compiled generation token id is outside the vocabulary")
    if eos_token_id is not None and (eos_token_id < 0 or eos_token_id >= model["vocabulary_size"]):
        raise ValueError("MLX compiled generation EOS is outside the vocabulary")
    model["incremental_compiler_cache_hit"] = True
    cache_hits_before, started = _widened_tensor_cache_hits, time.perf_counter()
    global _resident_generation_session
    model["prompt_length"] = token_ids.shape[1]
    reusable = _resident_generation_session if session_id is not None and _resident_generation_session is not None and _resident_generation_session["id"] == session_id else None
    cached_count = len(reusable["token_ids"]) if reusable is not None else 0
    if reusable is not None and (cached_count >= token_ids.shape[1] or not np.array_equal(token_ids.reshape(-1)[:cached_count], reusable["token_ids"])):
        reusable = None
        cached_count = 0
    current_token_ids = token_ids[:, cached_count:] if reusable is not None else token_ids
    result, all_per_layer = model["execute_token_prelude"](current_token_ids)
    cache_prefill_token_prelude_rows(model, current_token_ids, result, all_per_layer)
    positions = mx.array(np.arange(cached_count, token_ids.shape[1], dtype=np.int32).reshape(1, -1))
    produced_caches = {}
    rope_factor_context = {"factors": {}, "builds": 0, "uses": 0}
    topology_masks, topology_mask_uses = {}, 0
    kv_prefix_validation_scans_avoided = 0
    for layer_index, (config, weights) in enumerate(model["layer_plans"]):
        if config["produces_kv"]:
            if reusable is not None:
                source_key, source_value = reusable["caches"][layer_index]
            else:
                source_key = mx.zeros((1, config["key_value_heads"], 0, config["head_dim"]), dtype=mx.float32)
                source_value = mx.zeros((1, config["key_value_heads"], 0, config["head_dim"]), dtype=mx.float32)
        else:
            source_key, source_value = produced_caches[config["producer_layer"]]
        key_sequence = source_key.shape[2] + (current_token_ids.shape[1] if config["produces_kv"] else 0)
        topology_key = (config["mask_heads"], key_sequence, config["sliding_window"], config["causal"])
        mask = topology_masks.get(topology_key)
        topology_mask_uses += 1
        if mask is None:
            mask = continuation_topology_mask(config, current_token_ids.shape[1], key_sequence, cached_count) if reusable is not None else prefill_topology_mask(config, token_ids.shape[1])
            topology_masks[topology_key] = mask
        result, key, value, _valid, _ = execute_decoder_layer_mlx(pool, shards, result, all_per_layer[:, :, layer_index, :], positions, mask, source_key, source_value, config, weights, rope_factor_context, True)
        if config["produces_kv"]:
            produced_caches[layer_index] = (key, value)
            if source_key.shape[2]:
                kv_prefix_validation_scans_avoided += 1
    # Every non-finite decoder result propagates through the following RMSNorm,
    # residual, and projection chain. Validate the only resident prefill state
    # once here instead of scheduling the same reduction after all 42 layers.
    all_valid = mx.all(mx.isfinite(result))
    appended_tokens = current_token_ids.shape[1]
    for key, value in produced_caches.values():
        all_valid = all_valid & mx.all(mx.isfinite(key[..., -appended_tokens:, :])) & mx.all(mx.isfinite(value[..., -appended_tokens:, :]))
    raw_logits = execute_decoder_head(result, model["epilogue"], rounding=model["rounding"])
    logits = finalize_decoder_logits(raw_logits, model["epilogue"], rounding=model["rounding"])
    evaluation = [raw_logits, logits, all_valid]
    for key, value in produced_caches.values():
        evaluation.extend((key, value))
    mx.eval(*evaluation)
    if not bool(np.asarray(all_valid).item()):
        raise ValueError("MLX compiled generation prefill produced non-finite state")
    emit_resident_generation(model, result, produced_caches, raw_logits, logits, time.perf_counter() - started, max_new_tokens, eos_token_id, top_k, cache_hits_before, token_ids, session_id, stream, cached_count, current_token_ids.shape[1], rope_factor_context, len(topology_masks), topology_mask_uses, kv_prefix_validation_scans_avoided)


def execute_decoder_stack_request(pool, shards, batch, num_layers, hidden_size, native_bf16_ple, fused_epilogue, fused_token_forward, fused_token_generation):
    if native_bf16_ple:
        raise ValueError("MLX decoder stack does not support native-bf16-ple")
    cache_hits_before = _widened_tensor_cache_hits
    request_started = time.perf_counter()
    query_sequence, per_layer_width, rounding_code = struct.unpack("<III", read_exact(12))
    if not query_sequence or not per_layer_width or rounding_code not in (0, 1, 2) or (native_bf16_ple != (rounding_code == 2)):
        raise ValueError("MLX decoder stack topology is invalid")
    rounding = "real" if rounding_code == 1 else "native-bf16-ple" if rounding_code == 2 else "native-bf16"
    rows = batch * query_sequence
    generation = None
    if fused_token_forward:
        descriptor_count, rounding, tile_output_rows, vocabulary_size, token_scale, per_layer_scale, context_scale, combine_scale = struct.unpack("<IIIIffff", read_exact(32))
        epsilon = struct.unpack("<f", read_exact(4))[0]
        if descriptor_count != 4 or rounding != 0 or not tile_output_rows or not vocabulary_size or not all(math.isfinite(value) for value in (token_scale, per_layer_scale, context_scale, combine_scale, epsilon)) or epsilon <= 0:
            raise ValueError("MLX token forward prelude metadata is invalid")
        token_ids = np.frombuffer(read_exact(rows * 4), dtype=np.int32).reshape(batch, query_sequence)
        if (token_ids < 0).any() or (token_ids >= vocabulary_size).any():
            raise ValueError("MLX token forward token id is outside the vocabulary")
        if fused_token_generation:
            max_new_tokens, eos_plus_one, top_k = struct.unpack("<III", read_exact(12))
            if batch != 1 or not max_new_tokens or max_new_tokens > 4096 or not top_k or top_k > 64 or eos_plus_one > vocabulary_size:
                raise ValueError("MLX resident generation metadata is invalid")
            generation = {"max_new_tokens": max_new_tokens, "eos_token_id": None if eos_plus_one == 0 else eos_plus_one - 1, "top_k": top_k}
        token_embedding_reference = read_bf16_embedding_reference(pool, shards, (vocabulary_size, hidden_size))
        per_layer_embedding_reference = read_bf16_embedding_reference(pool, shards, (vocabulary_size, num_layers * per_layer_width))
        projection_weight = read_whole_tensor(pool, shards, (num_layers * per_layer_width, hidden_size), widen=False, required_dtype=mx.bfloat16, large_descriptor=True)
        prelude_norm_weight = read_whole_tensor(pool, shards, (per_layer_width, 1), large_descriptor=True).reshape((per_layer_width,))
        def execute_token_prelude(current_token_ids):
            current_rows = current_token_ids.size
            current_result = materialize_scaled_bf16_embedding_rows(token_embedding_reference, current_token_ids, token_scale).reshape((current_rows, hidden_size))
            current_identity = materialize_scaled_bf16_embedding_rows(per_layer_embedding_reference, current_token_ids, per_layer_scale).reshape((current_rows, num_layers, per_layer_width))
            current_per_layer = execute_ple_prelude_values(current_result, current_identity, projection_weight, prelude_norm_weight, current_rows, num_layers, per_layer_width, context_scale, combine_scale, epsilon, tile_output_rows)
            return current_result.reshape((*current_token_ids.shape, hidden_size)), current_per_layer.reshape((*current_token_ids.shape, num_layers, per_layer_width))
        result, all_per_layer = execute_token_prelude(token_ids)
    else:
        input_bytes = read_exact(rows * hidden_size * 4)
        per_layer_bytes = read_exact(rows * num_layers * per_layer_width * 4)
        result = mx.array(np.frombuffer(input_bytes, dtype=np.float32).reshape(batch, query_sequence, hidden_size))
        all_per_layer = mx.array(np.frombuffer(per_layer_bytes, dtype=np.float32).reshape(batch, query_sequence, num_layers, per_layer_width))
    position_bytes = read_exact(batch * query_sequence * 4)
    positions = mx.array(np.frombuffer(position_bytes, dtype=np.int32).reshape(batch, query_sequence))
    produced_caches, ordered_caches, layer_plans = {}, [], []
    rope_factor_context = {"factors": {}, "builds": 0, "uses": 0}
    for expected_layer in range(num_layers):
        metadata = read_exact(96)
        layer_index, shared_plus_one, query_heads, key_value_heads, source_sequence, head_dim, mask_heads, value_from_key, rope_kind, rotary_dim, proportional_pairs, intermediate_size, descriptor_count = struct.unpack("<IIIIIIIIIIIII", metadata[:52])
        attention_epsilon, scale = struct.unpack("<ff", metadata[52:60])
        theta = struct.unpack("<d", metadata[60:68])[0]
        proportional_factor, input_epsilon, post_attention_epsilon, pre_ffn_epsilon, post_ffn_epsilon, ple_epsilon = struct.unpack("<ffffff", metadata[68:92])
        topology = struct.unpack("<I", metadata[92:96])[0]
        causal, sliding_window = bool(topology & 0x80000000), topology & 0x7fffffff
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
        config = {"layer_index": layer_index, "producer_layer": producer_layer if shared else None, "query_heads": query_heads, "key_value_heads": key_value_heads, "head_dim": head_dim, "mask_heads": mask_heads, "causal": causal, "sliding_window": sliding_window, "produces_kv": produces_kv, "value_from_key": value_from_key, "intermediate_size": intermediate_size, "per_layer_width": per_layer_width, "input_epsilon": input_epsilon, "attention_epsilon": attention_epsilon, "post_attention_epsilon": post_attention_epsilon, "pre_ffn_epsilon": pre_ffn_epsilon, "post_ffn_epsilon": post_ffn_epsilon, "ple_epsilon": ple_epsilon, "scale": scale, "rope_kind": rope_kind, "theta": theta, "rotary_dim": rotary_dim, "proportional_pairs": proportional_pairs, "proportional_factor": proportional_factor, "rounding": rounding}
        result, key, value, _valid, weights = execute_decoder_layer_mlx(pool, shards, result, all_per_layer[:, :, layer_index, :], positions, mask, source_key, source_value, config, rope_factor_context=rope_factor_context)
        layer_plans.append((config, weights))
        if produces_kv:
            produced_caches[layer_index] = (key, value)
            ordered_caches.append((key, value))
    # The exported prefill state is exactly the terminal hidden vector plus the
    # producer-owned caches. One terminal check preserves the fail-closed
    # contract without retaining a reduction node for every intermediate layer.
    all_valid = mx.all(mx.isfinite(result))
    for key, value in ordered_caches:
        all_valid = all_valid & mx.all(mx.isfinite(key)) & mx.all(mx.isfinite(value))
    epilogue = None
    if fused_epilogue:
        vocabulary_size, norm_epsilon, softcap = struct.unpack("<Iff", read_exact(12))
        if not vocabulary_size or not math.isfinite(norm_epsilon) or norm_epsilon <= 0 or not math.isfinite(softcap) or softcap <= 0:
            raise ValueError("MLX decoder stack epilogue metadata is invalid")
        final_norm_weight = read_whole_tensor(pool, shards, (hidden_size, 1)).reshape((hidden_size,))
        head_weight = read_whole_tensor(pool, shards, (vocabulary_size, hidden_size), widen=False, required_dtype=mx.bfloat16)
        if _head_quantization == "q8-shortlist":
            head_weight = build_shortlist_quantized_head(head_weight)
        elif _head_quantization != "off":
            head_bits = 8 if _head_quantization == "q8" else 4
            group_size = 64
            head_weight = build_quantized_head(head_weight, group_size, head_bits)
        epilogue = (final_norm_weight, head_weight, norm_epsilon, softcap)
    evaluation = [result, all_valid]
    for key, value in ordered_caches:
        evaluation.extend((key, value))
    mx.eval(*evaluation)
    if not bool(np.asarray(all_valid).item()):
        raise ValueError("MLX decoder stack produced non-finite output")
    raw_logits, logits = None, None
    if epilogue is not None:
        raw_logits = execute_decoder_head(result, epilogue, fused_token_forward, rounding)
        logits = finalize_decoder_logits(raw_logits, epilogue, rounding)
        if generation is None:
            logits_valid = mx.all(mx.isfinite(logits))
            mx.eval(logits, logits_valid)
            if not bool(np.asarray(logits_valid).item()):
                raise ValueError("MLX decoder stack epilogue produced non-finite output")
        else:
            mx.eval(raw_logits, logits)
    if generation is not None:
        global _resident_generation_model, _resident_generation_control_model
        compile_signature = incremental_compile_signature(layer_plans, epilogue, rounding)
        previous_model = _resident_generation_model
        incremental_compiler_cache_hit = previous_model is not None and previous_model.get("compile_signature") == compile_signature
        if incremental_compiler_cache_hit:
            execute_incremental_decoder_step, producer_layers = previous_model["execute_incremental_decoder_step"], previous_model["producer_layers"]
        else:
            execute_incremental_decoder_step, producer_layers = compile_incremental_decoder_step(pool, shards, layer_plans, epilogue, rounding)
        rope_factor_uses_per_step = sum(1 + int(config["produces_kv"]) for config, _ in layer_plans)
        rope_factor_builds_per_step = len({(config["head_dim"], config["rope_kind"], config["theta"], config["rotary_dim"], config["proportional_pairs"], config["proportional_factor"], rounding != "real") for config, _ in layer_plans})
        _resident_generation_model = {"pool": pool, "shards": shards, "execute_token_prelude": execute_token_prelude, "execute_incremental_decoder_step": execute_incremental_decoder_step, "producer_layers": producer_layers, "compile_signature": compile_signature, "incremental_compiler_cache_hit": incremental_compiler_cache_hit, "rope_factor_uses_per_step": rope_factor_uses_per_step, "rope_factor_builds_per_step": rope_factor_builds_per_step, "layer_plans": layer_plans, "epilogue": epilogue, "vocabulary_size": vocabulary_size, "prompt_length": query_sequence, "rounding": rounding, "token_prelude_cache": OrderedDict()}
        control_layer_plans = []
        for config, weights in layer_plans:
            if "down_control" not in weights:
                control_layer_plans = []
                break
            control_weights = {name: value for name, value in weights.items() if name != "down_control"}
            control_weights["down"] = weights["down_control"]
            del weights["down_control"]
            control_layer_plans.append((config, control_weights))
        if control_layer_plans:
            control_signature = incremental_compile_signature(control_layer_plans, epilogue, rounding)
            previous_control = _resident_generation_control_model
            control_cache_hit = previous_control is not None and previous_control.get("compile_signature") == control_signature
            if control_cache_hit:
                control_incremental_step, control_producers = previous_control["execute_incremental_decoder_step"], previous_control["producer_layers"]
            else:
                control_incremental_step, control_producers = compile_incremental_decoder_step(pool, shards, control_layer_plans, epilogue, rounding)
            _resident_generation_control_model = {**_resident_generation_model, "execute_incremental_decoder_step": control_incremental_step, "producer_layers": control_producers, "compile_signature": control_signature, "incremental_compiler_cache_hit": control_cache_hit, "layer_plans": control_layer_plans, "token_prelude_cache": OrderedDict()}
        else:
            _resident_generation_control_model = None
        emit_resident_generation(_resident_generation_model, result, produced_caches, raw_logits, logits, time.perf_counter() - request_started, generation["max_new_tokens"], generation["eos_token_id"], generation["top_k"], cache_hits_before, token_ids, initial_rope_factor_context=rope_factor_context, initial_topology_mask_builds=num_layers, initial_topology_mask_uses=num_layers)
        return
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


def parse_decoder_quantization_layers(value):
    layers = set()
    for part in value.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if match is None:
            raise ValueError(f"invalid decoder quantization layer range: {part}")
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        if start > end or start < 0 or end > 41:
            raise ValueError(f"decoder quantization layers must be between 0 and 41: {part}")
        layers.update(range(start, end + 1))
    if not layers:
        raise ValueError("decoder quantization layers cannot be empty")
    return frozenset(layers)


def main():
    global _head_quantization, _decoder_quantization, _decoder_quantization_layers
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--binary-pool", required=True)
    parser.add_argument("--head-quantization", choices=("off", "q8", "q8-shortlist", "q4"), default="off")
    parser.add_argument("--decoder-quantization", choices=("off", "q8-ffn", "q8-ffn-gate-up", "q8-ffn-gate-up-attention", "q8-ffn-gate-up-down", "q4-ffn-gate-up", "q8-ffn-gate-up-first-half", "q8-ffn-gate-up-last-half", "q8-ffn-down", "q8-attention", "q8-all"), default="off")
    parser.add_argument("--decoder-quantization-layers")
    args = parser.parse_args()
    _head_quantization = args.head_quantization
    _decoder_quantization = args.decoder_quantization
    if args.decoder_quantization_layers is not None:
        if _decoder_quantization not in ("q8-ffn-gate-up", "q8-ffn-gate-up-down", "q4-ffn-gate-up"):
            raise ValueError("--decoder-quantization-layers requires --decoder-quantization q8-ffn-gate-up, q8-ffn-gate-up-down, or q4-ffn-gate-up")
        _decoder_quantization_layers = parse_decoder_quantization_layers(args.decoder_quantization_layers)
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
        compiled_token_generation = bool(encoded_dtype & COMPILED_TOKEN_GENERATION_FLAG)
        if compiled_token_generation:
            session_generation = bool(encoded_dtype & SESSION_TOKEN_GENERATION_FLAG)
            stream_generation = bool(encoded_dtype & STREAM_TOKEN_GENERATION_FLAG)
            control_generation = bool(encoded_dtype & CONTROL_TOKEN_GENERATION_FLAG)
            expected_code = COMPILED_TOKEN_GENERATION_FLAG | (SESSION_TOKEN_GENERATION_FLAG if session_generation else 0) | (STREAM_TOKEN_GENERATION_FLAG if stream_generation else 0) | (CONTROL_TOKEN_GENERATION_FLAG if control_generation else 0)
            if referenced or batched or fused_mlp or native_bf16 or native_attention or fused_attention or dtype_code != expected_code:
                raise ValueError("MLX compiled generation flags are invalid")
            if not rows or not outputs or not features or outputs > 4096 or features > 64:
                raise ValueError("MLX compiled generation topology is invalid")
            eos_plus_one = struct.unpack("<I", read_exact(4))[0]
            session_id = struct.unpack("<I", read_exact(4))[0] if session_generation else None
            if session_generation and session_id == 0:
                raise ValueError("MLX compiled generation session id is invalid")
            token_ids = np.frombuffer(read_exact(rows * 4), dtype=np.int32).reshape(1, rows)
            execute_compiled_token_generation(pool, shards, token_ids, outputs, None if eos_plus_one == 0 else eos_plus_one - 1, features, session_id, stream_generation, control_generation)
            continue
        fused_decoder_stack = bool(encoded_dtype & FUSED_DECODER_STACK_FLAG)
        if fused_decoder_stack:
            fused_epilogue = bool(encoded_dtype & FUSED_DECODER_STACK_EPILOGUE_FLAG)
            fused_token_forward = bool(encoded_dtype & FUSED_TOKEN_FORWARD_FLAG)
            fused_token_generation = bool(encoded_dtype & FUSED_TOKEN_GENERATION_FLAG)
            expected_code = FUSED_DECODER_STACK_FLAG | (FUSED_DECODER_STACK_EPILOGUE_FLAG if fused_epilogue else 0) | (FUSED_TOKEN_FORWARD_FLAG if fused_token_forward else 0) | (FUSED_TOKEN_GENERATION_FLAG if fused_token_generation else 0)
            if referenced or batched or fused_mlp or native_attention or fused_attention or dtype_code != expected_code:
                raise ValueError("MLX decoder stack flags are invalid")
            if fused_token_forward and not fused_epilogue:
                raise ValueError("MLX token forward requires the fused epilogue")
            if fused_token_generation and not fused_token_forward:
                raise ValueError("MLX resident generation requires token forward")
            execute_decoder_stack_request(pool, shards, rows, outputs, features, native_bf16, fused_epilogue, fused_token_forward, fused_token_generation)
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
