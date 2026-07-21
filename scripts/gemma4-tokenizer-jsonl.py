#!/usr/bin/env python3
"""Small persistent tokenizer transport for compiled Gemma 4 generation."""

import argparse
import json
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    print(json.dumps({
        "ready": True,
        "source": str(Path(args.source).resolve()),
        "role": "tokenizer-only",
        "initializationSeconds": time.perf_counter() - started,
    }), flush=True)
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            mode = request.get("mode")
            if mode == "encode":
                text = request.get("text")
                add_special_tokens = request.get("addSpecialTokens", True)
                if not isinstance(text, str) or not text or not isinstance(add_special_tokens, bool):
                    raise ValueError("encode requires non-empty text and boolean addSpecialTokens")
                report = {"tokenIds": tokenizer.encode(text, add_special_tokens=add_special_tokens)}
            elif mode == "decode":
                token_ids = request.get("tokenIds")
                if not isinstance(token_ids, list) or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in token_ids):
                    raise ValueError("tokenIds must be an array of non-negative integers")
                report = {"text": tokenizer.decode(token_ids, skip_special_tokens=True)}
            else:
                raise ValueError("mode must be encode or decode")
            print(json.dumps({"id": request.get("id"), "report": report}), flush=True)
        except Exception as error:
            print(json.dumps({"id": request.get("id") if isinstance(request, dict) else None, "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
