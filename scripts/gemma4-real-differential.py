#!/usr/bin/env python3
"""One-token E4B IEEE-BF16 versus no-intermediate-rounding differential."""

import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as functional
from transformers import Gemma4ForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4 import modeling_gemma4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--input-ids", default="2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--logit-chunk", type=int, default=8192)
    return parser.parse_args()


def exact_linear(self, value):
    value = value.double()
    if self.out_features <= 32768:
        bias = self.bias.double() if self.bias is not None else None
        return functional.linear(value, self.weight.double(), bias)
    chunks = []
    for start in range(0, self.out_features, EXACT_LOGIT_CHUNK):
        end = min(start + EXACT_LOGIT_CHUNK, self.out_features)
        bias = self.bias[start:end].double() if self.bias is not None else None
        chunks.append(functional.linear(value, self.weight[start:end].double(), bias))
    return torch.cat(chunks, dim=-1)


def exact_rms_norm(self, hidden_states):
    hidden_states = hidden_states.double()
    result = hidden_states * torch.pow(hidden_states.pow(2).mean(-1, keepdim=True) + float(self.eps), -0.5)
    return result * self.weight.double() if self.with_scale else result


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
    output = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return output, weights


def boundary_values(text, input_ids):
    embeddings = text.embed_tokens(input_ids)
    raw_per_layer = text.get_per_layer_inputs(input_ids, embeddings)
    return embeddings.detach(), text.project_per_layer_inputs(embeddings, raw_per_layer).detach()


def exact_text_forward(text, hidden, per_layer, position_start=0):
    hidden = hidden.double()
    per_layer = per_layer.double()
    cache = DynamicCache(config=text.config)
    causal = torch.full((hidden.shape[1], hidden.shape[1]), float("-inf"), dtype=torch.float64).triu(diagonal=1)
    masks = {"full_attention": causal[None, None], "sliding_attention": causal[None, None]}
    position_embeddings = {}
    for layer_type in text.unique_layer_types:
        head_dim = text.config.global_head_dim if layer_type == "full_attention" and text.config.global_head_dim else text.config.head_dim
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        theta = float(text.config.rope_parameters[layer_type]["rope_theta"])
        inv_frequency = theta ** (-exponent)
        positions = torch.arange(position_start, position_start + hidden.shape[1], dtype=torch.float64)
        frequencies = positions[:, None] * inv_frequency[None, :]
        embedding = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(0)
        position_embeddings[layer_type] = (embedding.cos(), embedding.sin())
    for index, layer in enumerate(text.layers[: text.config.num_hidden_layers]):
        hidden = layer(
            hidden,
            per_layer[:, :, index, :],
            position_embeddings=position_embeddings[text.config.layer_types[index]],
            attention_mask=masks[text.config.layer_types[index]],
            position_ids=torch.arange(position_start, position_start + hidden.shape[1], dtype=torch.long).unsqueeze(0),
            past_key_values=cache,
        )
    return text.norm(hidden)


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


args = parse_args()
EXACT_LOGIT_CHUNK = args.logit_chunk
started = time.time()
torch.set_grad_enabled(False)
model = Gemma4ForConditionalGeneration.from_pretrained(
    args.source,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
).eval()
parsed_input_ids = [int(token) for token in args.input_ids.split(",")]
if not parsed_input_ids or any(token < 0 for token in parsed_input_ids):
    raise ValueError("--input-ids requires comma-separated non-negative integers")
input_ids = torch.tensor([parsed_input_ids], dtype=torch.long)
baseline_output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
baseline_logits = baseline_output.logits[0, -1].float().cpu()
text = model.model.language_model
hidden_boundary, per_layer_boundary = boundary_values(text, input_ids)

original_linear = torch.nn.Linear.forward
original_norm = modeling_gemma4.Gemma4RMSNorm.forward
original_attention = modeling_gemma4.eager_attention_forward
torch.nn.Linear.forward = exact_linear
modeling_gemma4.Gemma4RMSNorm.forward = exact_rms_norm
modeling_gemma4.eager_attention_forward = exact_eager_attention
try:
    final_hidden = exact_text_forward(text, hidden_boundary, per_layer_boundary)
    exact_logits = exact_linear(model.lm_head, final_hidden[:, -1:, :])[0, -1]
    softcap = model.config.get_text_config().final_logit_softcapping
    if softcap is not None:
        exact_logits = torch.tanh(exact_logits / float(softcap)) * float(softcap)
    candidate_logits = exact_logits.to(torch.bfloat16).float().cpu()
finally:
    torch.nn.Linear.forward = original_linear
    modeling_gemma4.Gemma4RMSNorm.forward = original_norm
    modeling_gemma4.eager_attention_forward = original_attention

report = {
    "kind": "gemma4-exact-real-simplified-differential",
    "schemaVersion": 1,
    "source": str(Path(args.source).resolve()),
    "inputIds": [parsed_input_ids],
    "baseline": "Transformers eager BF16 with declared intermediate rounding",
    "candidate": "F64 tensor operations from common BF16 embedding/PLE boundaries; BF16 RNE only at terminal logits",
    "metrics": metrics(baseline_logits, candidate_logits),
    "elapsedSeconds": time.time() - started,
}
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
