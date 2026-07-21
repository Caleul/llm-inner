#!/usr/bin/env python3
"""Calibration matrix for the persistent simplified Gemma runtime."""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "Write one short sentence about the Moon:",
    "Translate to Portuguese: Good morning",
    "Complete the Python expression: sum([1, 2, 3]) ==",
    "Water freezes at",
    "Era uma vez um pequeno robô que",
    "Answer yes or no: Is the Earth round?",
]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--precision", choices=["f32", "f64"], default="f32")
    parser.add_argument("--rounding-policy", choices=["none", "layer-bf16", "operation-bf16"], default="none")
    parser.add_argument("--prompts-json")
    return parser.parse_args()


def load_runtime_module():
    path = Path(__file__).with_name("gemma4-real-differential.py")
    spec = importlib.util.spec_from_file_location("gemma4_real_differential", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    args = arguments()
    prompts = json.loads(Path(args.prompts_json).read_text()) if args.prompts_json else DEFAULT_PROMPTS
    if not isinstance(prompts, list) or not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("prompts must be a non-empty JSON string array")
    runtime = load_runtime_module()
    model, tokenizer = runtime.load_runtime(args.source, args.threads, 8192)
    reports = [runtime.compare_request(model, tokenizer, args.source, prompt, None, args.tokens, [], args.threads, args.precision, args.rounding_policy) for prompt in prompts]
    steps = [step for report in reports for step in report["steps"]]
    speedups = [report["performance"]["candidateSpeedup"] for report in reports]
    result = {
        "kind": "gemma4-compiled-calibration-matrix", "schemaVersion": 2,
        "source": str(Path(args.source).resolve()), "configuration": {
            "prompts": len(prompts), "tokensPerPrompt": args.tokens, "threads": args.threads,
            "precision": args.precision, "roundingPolicy": args.rounding_policy,
        },
        "summary": {
            "promptsWithAllTokensEqual": sum(report["generatedTokensEqual"] for report in reports),
            "promptAgreementRate": sum(report["generatedTokensEqual"] for report in reports) / len(reports),
            "argmaxEqualSteps": sum(step["metrics"]["argmaxEqual"] for step in steps),
            "argmaxAgreementRate": sum(step["metrics"]["argmaxEqual"] for step in steps) / len(steps),
            "meanLogitDivergenceRate": statistics.fmean(step["metrics"]["divergenceRate"] for step in steps),
            "maximumAbsoluteError": max(step["metrics"]["maxAbsError"] for step in steps),
            "medianCandidateSpeedup": statistics.median(speedups),
            "minimumCandidateSpeedup": min(speedups), "maximumCandidateSpeedup": max(speedups),
            "processPeakRssBytes": max(report["performance"]["processPeakRssBytes"] for report in reports),
        },
        "cases": [{
            "prompt": report["prompt"], "tokensEqual": report["generatedTokensEqual"],
            "firstDivergentStep": report["firstDivergentStep"],
            "baselineTokenIds": report["baselineGeneratedTokenIds"], "candidateTokenIds": report["candidateGeneratedTokenIds"],
            "baselineText": report["baselineGeneratedText"], "candidateText": report["candidateGeneratedText"],
            "speedup": report["performance"]["candidateSpeedup"],
            "steps": [{
                "step": step["step"], "contextsEqualBeforeStep": step["contextsEqualBeforeStep"],
                "baselineToken": step["baselineToken"], "candidateToken": step["candidateToken"],
                "argmaxEqual": step["metrics"]["argmaxEqual"],
                "divergenceRate": step["metrics"]["divergenceRate"], "maxAbsError": step["metrics"]["maxAbsError"],
            } for step in report["steps"]],
        } for report in reports],
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
