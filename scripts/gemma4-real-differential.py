#!/usr/bin/env python3
"""Real greedy generation: E4B IEEE-BF16 versus no-intermediate-rounding."""

import argparse
import base64
from collections import OrderedDict
import json
import platform
import resource
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as functional
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4 import modeling_gemma4

EXACT_LOGIT_CHUNK = 8192
REAL_DTYPE = torch.float64
ROUNDING_POLICY = "none"
COMPARISON_SESSION_LIMIT = 4
COMPARISON_SESSIONS = OrderedDict()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-ids")
    inputs.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--logit-chunk", type=int, default=8192)
    parser.add_argument("--inspect-logit", type=int, action="append", default=[])
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--serve-jsonl", action="store_true")
    parser.add_argument("--precision", choices=["f32", "f64"], default="f64")
    parser.add_argument("--rounding-policy", choices=["none", "layer-bf16", "operation-bf16"], default="none")
    return parser.parse_args()


def exact_linear(self, value):
    value = value.to(REAL_DTYPE)
    if self.out_features <= 32768:
        bias = self.bias.to(REAL_DTYPE) if self.bias is not None else None
        return rounded_operation(functional.linear(value, self.weight.to(REAL_DTYPE), bias))
    chunks = []
    for start in range(0, self.out_features, EXACT_LOGIT_CHUNK):
        end = min(start + EXACT_LOGIT_CHUNK, self.out_features)
        bias = self.bias[start:end].to(REAL_DTYPE) if self.bias is not None else None
        chunks.append(functional.linear(value, self.weight[start:end].to(REAL_DTYPE), bias))
    return rounded_operation(torch.cat(chunks, dim=-1))


def exact_rms_norm(self, hidden_states):
    hidden_states = hidden_states.to(REAL_DTYPE)
    result = hidden_states * torch.pow(hidden_states.pow(2).mean(-1, keepdim=True) + float(self.eps), -0.5)
    return rounded_operation(result * self.weight.to(REAL_DTYPE) if self.with_scale else result)


def exact_eager_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kwargs):
    if scaling is None:
        scaling = module.head_dim ** -0.5
    key = modeling_gemma4.repeat_kv(key, module.num_key_value_groups)
    value = modeling_gemma4.repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key.transpose(2, 3)) * float(scaling)
    if softcap is not None:
        scores = torch.tanh(scores / float(softcap)) * float(softcap)
    if attention_mask is not None:
        scores = scores + attention_mask
    weights = functional.softmax(scores, dim=-1)
    output = rounded_operation(torch.matmul(weights, value).transpose(1, 2).contiguous())
    return output, weights


def rounded_operation(value):
    return value.to(torch.bfloat16).to(REAL_DTYPE) if ROUNDING_POLICY == "operation-bf16" else value


def boundary_values(text, input_ids):
    embeddings = text.embed_tokens(input_ids)
    raw_per_layer = text.get_per_layer_inputs(input_ids, embeddings)
    return embeddings.detach(), text.project_per_layer_inputs(embeddings, raw_per_layer).detach()


def exact_text_forward(text, hidden, per_layer, position_start=0, cache=None, capture_layer_hidden=False):
    hidden = hidden.to(REAL_DTYPE)
    per_layer = per_layer.to(REAL_DTYPE)
    cache = cache if cache is not None else DynamicCache(config=text.config)
    keys = position_start + hidden.shape[1]
    queries = torch.arange(position_start, keys, dtype=torch.long)[:, None]
    key_positions = torch.arange(keys, dtype=torch.long)[None, :]
    causal = torch.where(key_positions <= queries, torch.tensor(0.0, dtype=REAL_DTYPE), torch.tensor(float("-inf"), dtype=REAL_DTYPE))
    masks = {"full_attention": causal[None, None], "sliding_attention": causal[None, None]}
    position_embeddings = {}
    for layer_type in text.unique_layer_types:
        head_dim = text.config.global_head_dim if layer_type == "full_attention" and text.config.global_head_dim else text.config.head_dim
        exponent = torch.arange(0, head_dim, 2, dtype=REAL_DTYPE) / head_dim
        theta = float(text.config.rope_parameters[layer_type]["rope_theta"])
        inv_frequency = theta ** (-exponent)
        positions = torch.arange(position_start, position_start + hidden.shape[1], dtype=REAL_DTYPE)
        frequencies = positions[:, None] * inv_frequency[None, :]
        embedding = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(0)
        position_embeddings[layer_type] = (embedding.cos(), embedding.sin())
    layer_hidden = []
    for index, layer in enumerate(text.layers[: text.config.num_hidden_layers]):
        hidden = layer(
            hidden,
            per_layer[:, :, index, :],
            position_embeddings=position_embeddings[text.config.layer_types[index]],
            attention_mask=masks[text.config.layer_types[index]],
            position_ids=torch.arange(position_start, position_start + hidden.shape[1], dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
        )
        if ROUNDING_POLICY == "layer-bf16":
            hidden = hidden.to(torch.bfloat16).to(REAL_DTYPE)
        if capture_layer_hidden:
            layer_hidden.append(hidden.detach())
    return text.norm(hidden), cache, layer_hidden


def metrics(baseline, candidate):
    difference = (candidate - baseline).abs()
    denominator = baseline.abs().clamp_min(torch.finfo(torch.float32).tiny)
    baseline_argmax = int(torch.argmax(baseline).item())
    candidate_argmax = int(torch.argmax(candidate).item())
    return {
        "values": baseline.numel(),
        "equalAfterFinalBf16": int(torch.eq(candidate, baseline).sum().item()),
        "divergenceRate": float(torch.ne(candidate, baseline).double().mean().item()),
        "maxAbsError": float(difference.max().item()),
        "meanAbsError": float(difference.double().mean().item()),
        "maxRelativeError": float((difference / denominator).max().item()),
        "baselineArgmax": baseline_argmax,
        "candidateArgmax": candidate_argmax,
        "argmaxEqual": baseline_argmax == candidate_argmax,
        "baselineArgmaxLogit": float(baseline[baseline_argmax].item()),
        "candidateArgmaxLogit": float(candidate[candidate_argmax].item()),
    }


def top_logits(values, count=10):
    selected = torch.topk(values, min(count, values.numel()))
    return [
        {"tokenId": int(token), "logit": float(logit)}
        for logit, token in zip(selected.values.tolist(), selected.indices.tolist())
    ]


def inspected_logits(baseline, candidate, dimensions):
    inspected = []
    for dimension in dimensions:
        if dimension < 0 or dimension >= baseline.numel():
            raise ValueError(f"--inspect-logit {dimension} is outside [0, {baseline.numel()})")
        baseline_value = float(baseline[dimension].item())
        candidate_value = float(candidate[dimension].item())
        inspected.append({
            "dimension": dimension,
            "baseline": baseline_value,
            "candidate": candidate_value,
            "absoluteError": abs(candidate_value - baseline_value),
            "equalAfterFinalBf16": candidate_value == baseline_value,
        })
    return inspected


def baseline_next(model, input_ids, cache=None, capture_layer_hidden=False):
    started = time.perf_counter()
    layer_hidden = []
    hooks = []
    if capture_layer_hidden:
        for layer in model.model.language_model.layers[: model.config.get_text_config().num_hidden_layers]:
            hooks.append(layer.register_forward_hook(lambda _module, _inputs, output: layer_hidden.append((output[0] if isinstance(output, tuple) else output).detach())))
    try:
        output = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()
    logits = output.logits[0, -1].float().cpu()
    return logits, output.past_key_values, time.perf_counter() - started, layer_hidden


def exact_next(model, input_ids, cache=None, position_start=0, capture_layer_hidden=False):
    text = model.model.language_model
    started = time.perf_counter()
    hidden_boundary, per_layer_boundary = boundary_values(text, input_ids)
    original_linear = torch.nn.Linear.forward
    original_norm = modeling_gemma4.Gemma4RMSNorm.forward
    original_attention = modeling_gemma4.eager_attention_forward
    torch.nn.Linear.forward = exact_linear
    modeling_gemma4.Gemma4RMSNorm.forward = exact_rms_norm
    modeling_gemma4.eager_attention_forward = exact_eager_attention
    try:
        final_hidden, cache, layer_hidden = exact_text_forward(text, hidden_boundary, per_layer_boundary, position_start, cache, capture_layer_hidden)
        exact_logits = exact_linear(model.lm_head, final_hidden[:, -1:, :])[0, -1]
        softcap = model.config.get_text_config().final_logit_softcapping
        if softcap is not None:
            exact_logits = torch.tanh(exact_logits / float(softcap)) * float(softcap)
        logits = exact_logits.to(torch.bfloat16).float().cpu()
    finally:
        torch.nn.Linear.forward = original_linear
        modeling_gemma4.Gemma4RMSNorm.forward = original_norm
        modeling_gemma4.eager_attention_forward = original_attention
    return logits, cache, time.perf_counter() - started, layer_hidden


def encode_terminal_hidden(layers):
    encoded = []
    for hidden in layers:
        terminal = hidden[0, -1].float().cpu().contiguous()
        encoded.append(base64.b64encode(terminal.numpy().tobytes()).decode("ascii"))
    return encoded


def load_runtime(source, threads, logit_chunk):
    global EXACT_LOGIT_CHUNK
    if threads < 0:
        raise ValueError("threads must be zero (runtime default) or a positive integer")
    if threads:
        torch.set_num_threads(threads)
    EXACT_LOGIT_CHUNK = logit_chunk
    torch.set_grad_enabled(False)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        source, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager",
    ).eval()
    return model, AutoTokenizer.from_pretrained(source)


def compare_request(model, tokenizer, source, prompt, input_ids, max_new_tokens, inspect_logit, threads=0, precision="f64", rounding_policy="none", session_id=None, eos_token_id=None, capture_layer_hidden=False):
    global REAL_DTYPE, ROUNDING_POLICY
    if max_new_tokens < 1 or max_new_tokens > 256:
        raise ValueError("maxNewTokens must be between 1 and 256")
    if threads < 0:
        raise ValueError("threads must be zero or positive")
    if threads:
        torch.set_num_threads(threads)
    if precision not in ("f32", "f64"):
        raise ValueError("precision must be f32 or f64")
    REAL_DTYPE = torch.float32 if precision == "f32" else torch.float64
    if rounding_policy not in ("none", "layer-bf16", "operation-bf16"):
        raise ValueError("roundingPolicy must be none, layer-bf16 or operation-bf16")
    if not isinstance(capture_layer_hidden, bool):
        raise ValueError("captureLayerHidden must be boolean")
    ROUNDING_POLICY = rounding_policy
    started = time.time()
    if prompt is not None:
        parsed_input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    else:
        token_text = input_ids if input_ids is not None else "2"
        parsed_input_ids = [int(token) for token in token_text.split(",")]
    if not parsed_input_ids or any(token < 0 for token in parsed_input_ids):
        raise ValueError("inputIds requires comma-separated non-negative integers")
    if session_id is not None and (not isinstance(session_id, int) or isinstance(session_id, bool) or session_id < 1 or session_id > 0xffffffff):
        raise ValueError("sessionId must be an integer between 1 and 4294967295")
    if eos_token_id is not None and (not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool) or eos_token_id < 0 or eos_token_id >= tokenizer.vocab_size):
        raise ValueError("eosTokenId must be a valid tokenizer vocabulary id")
    baseline_ids, candidate_ids = list(parsed_input_ids), list(parsed_input_ids)
    baseline_generated, candidate_generated, steps = [], [], []
    session = COMPARISON_SESSIONS.get(session_id) if session_id is not None else None
    baseline_reused = len(session["baseline_token_ids"]) if session is not None and len(session["baseline_token_ids"]) < len(parsed_input_ids) and parsed_input_ids[:len(session["baseline_token_ids"])] == session["baseline_token_ids"] else 0
    candidate_reused = len(session["candidate_token_ids"]) if session is not None and len(session["candidate_token_ids"]) < len(parsed_input_ids) and parsed_input_ids[:len(session["candidate_token_ids"])] == session["candidate_token_ids"] else 0
    baseline_cache = session["baseline_cache"] if baseline_reused else None
    candidate_cache = session["candidate_cache"] if candidate_reused else None
    for step in range(max_new_tokens):
        contexts_equal = baseline_ids == candidate_ids
        baseline_input = baseline_ids[baseline_reused:] if step == 0 and baseline_reused else baseline_ids if step == 0 else [baseline_ids[-1]]
        candidate_input = candidate_ids[candidate_reused:] if step == 0 and candidate_reused else candidate_ids if step == 0 else [candidate_ids[-1]]
        baseline_logits, baseline_cache, baseline_seconds, baseline_layer_hidden = baseline_next(model, torch.tensor([baseline_input], dtype=torch.long), baseline_cache, capture_layer_hidden)
        candidate_logits, candidate_cache, candidate_seconds, candidate_layer_hidden = exact_next(model, torch.tensor([candidate_input], dtype=torch.long), candidate_cache, len(candidate_ids) - len(candidate_input), capture_layer_hidden)
        comparison = metrics(baseline_logits, candidate_logits)
        baseline_token, candidate_token = comparison["baselineArgmax"], comparison["candidateArgmax"]
        baseline_ids.append(baseline_token); candidate_ids.append(candidate_token)
        baseline_generated.append(baseline_token); candidate_generated.append(candidate_token)
        steps.append({
            "step": step, "contextsEqualBeforeStep": contexts_equal,
            "baselineContextLength": len(baseline_ids) - 1, "candidateContextLength": len(candidate_ids) - 1,
            "baselineToken": baseline_token, "candidateToken": candidate_token,
            "baselineTokenText": tokenizer.decode([baseline_token], skip_special_tokens=False),
            "candidateTokenText": tokenizer.decode([candidate_token], skip_special_tokens=False),
            "baselineSeconds": baseline_seconds, "candidateSeconds": candidate_seconds,
            "metrics": comparison, "inspectedLogits": inspected_logits(baseline_logits, candidate_logits, inspect_logit),
            "baselineTopLogits": top_logits(baseline_logits), "candidateTopLogits": top_logits(candidate_logits),
            **({"layerHiddenEncoding": "terminal-token-f32le-base64", "baselineLayerHidden": encode_terminal_hidden(baseline_layer_hidden), "candidateLayerHidden": encode_terminal_hidden(candidate_layer_hidden)} if capture_layer_hidden else {}),
        })
        if baseline_token == eos_token_id:
            break
    baseline_total = sum(entry["baselineSeconds"] for entry in steps)
    candidate_total = sum(entry["candidateSeconds"] for entry in steps)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(peak_rss if platform.system() == "Darwin" else peak_rss * 1024)
    if session_id is not None:
        COMPARISON_SESSIONS.pop(session_id, None)
        COMPARISON_SESSIONS[session_id] = {"baseline_token_ids": baseline_ids[:-1], "candidate_token_ids": candidate_ids[:-1], "baseline_cache": baseline_cache, "candidate_cache": candidate_cache}
        while len(COMPARISON_SESSIONS) > COMPARISON_SESSION_LIMIT:
            COMPARISON_SESSIONS.popitem(last=False)
    return {
        "kind": "gemma4-exact-real-simplified-differential", "schemaVersion": 7,
        "source": str(Path(source).resolve()), "inputIds": [parsed_input_ids], "prompt": prompt,
        "baseline": "Transformers eager BF16 with declared intermediate rounding",
        "candidate": f"{precision.upper()} tensor operations from common BF16 embedding/PLE boundaries; BF16 RNE only at terminal logits",
        "maxNewTokens": max_new_tokens, "executionThreads": torch.get_num_threads(), "candidatePrecision": precision, "roundingPolicy": rounding_policy,
        "baselineGeneratedTokenIds": baseline_generated, "candidateGeneratedTokenIds": candidate_generated,
        "baselineGeneratedText": tokenizer.decode(baseline_generated, skip_special_tokens=True),
        "candidateGeneratedText": tokenizer.decode(candidate_generated, skip_special_tokens=True),
        "baselineFullText": tokenizer.decode(baseline_ids, skip_special_tokens=True),
        "candidateFullText": tokenizer.decode(candidate_ids, skip_special_tokens=True),
        "generatedTokensEqual": baseline_generated == candidate_generated,
        "firstDivergentStep": next((entry["step"] for entry in steps if entry["baselineToken"] != entry["candidateToken"]), None),
        "steps": steps, "elapsedSeconds": time.time() - started,
        "sessionId": session_id, "baselinePrefixTokensReused": baseline_reused, "candidatePrefixTokensReused": candidate_reused,
        "baselinePrefillTokensComputed": len(parsed_input_ids) - baseline_reused, "candidatePrefillTokensComputed": len(parsed_input_ids) - candidate_reused,
        "performance": {
            "baselineSeconds": baseline_total, "candidateSeconds": candidate_total,
            "baselineTokensPerSecond": len(baseline_generated) / baseline_total,
            "candidateTokensPerSecond": len(candidate_generated) / candidate_total,
            "candidateSpeedup": baseline_total / candidate_total,
            "processPeakRssBytes": peak_rss_bytes,
        },
    }


def main():
    args = parse_args()
    initialization_started = time.time()
    model, tokenizer = load_runtime(args.source, args.threads, args.logit_chunk)
    initialization_seconds = time.time() - initialization_started
    if args.serve_jsonl:
        print(json.dumps({"ready": True, "source": str(Path(args.source).resolve()), "initializationSeconds": initialization_seconds}), flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("mode") == "decode":
                    token_ids = request.get("tokenIds")
                    if not isinstance(token_ids, list) or any(not isinstance(token, int) or token < 0 for token in token_ids):
                        raise ValueError("tokenIds must be an array of non-negative integers")
                    report = {"text": tokenizer.decode(token_ids, skip_special_tokens=True)}
                elif request.get("mode") == "encode":
                    text = request.get("text")
                    add_special_tokens = request.get("addSpecialTokens", True)
                    if not isinstance(text, str) or not text or not isinstance(add_special_tokens, bool):
                        raise ValueError("encode requires non-empty text and boolean addSpecialTokens")
                    report = {"tokenIds": tokenizer.encode(text, add_special_tokens=add_special_tokens)}
                else:
                    report = compare_request(model, tokenizer, args.source, request.get("prompt"), request.get("inputIds"), int(request.get("maxNewTokens", 1)), request.get("inspectLogit", []), int(request.get("threads", 0)), request.get("precision", "f64"), request.get("roundingPolicy", "none"), request.get("sessionId"), request.get("eosTokenId"), request.get("captureLayerHidden", False))
                print(json.dumps({"id": request.get("id"), "report": report}), flush=True)
            except Exception as error:
                print(json.dumps({"id": request.get("id") if "request" in locals() else None, "error": str(error)}), flush=True)
        return
    if not args.output:
        raise ValueError("--output is required outside --serve-jsonl")
    report = compare_request(model, tokenizer, args.source, args.prompt, args.input_ids, args.max_new_tokens, args.inspect_logit, args.threads, args.precision, args.rounding_policy)
    report["initializationSeconds"] = initialization_seconds
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
