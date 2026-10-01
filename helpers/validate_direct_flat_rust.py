"""Final chosen-logit differential; no intermediate or token-ID acceptance.

Validation loads PyTorch's checkpoint. The generated executable only receives
embedding rows as arguments and is also run with checkpoint reads denied.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM


def widened_bits(value):
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("executable")
    parser.add_argument("dimension", type=int)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, local_files_only=True, dtype=torch.float16,
        attn_implementation="eager").eval()
    embedding = model.get_input_embeddings()
    width = embedding.embedding_dim
    vocab = model.get_output_embeddings().out_features
    context = model.config.max_position_embeddings
    if not 0 <= args.dimension < vocab:
        raise ValueError("Dimension outside discovered vocabulary")
    lengths = sorted({n for n in [1, 2, 3, 5, 8, 13] if n <= context} | {min(context, 17)})
    generator = torch.Generator().manual_seed(97319)
    cases = []
    for length in lengths:
        ids = torch.randint(0, embedding.num_embeddings, (length,), generator=generator)
        with torch.no_grad():
            cases.append(("checkpoint-embeddings", embedding(ids).detach()))
        for magnitude in [2.0 ** -24, 0.03125, 1.0]:
            values = torch.rand(length, width, generator=generator) * (2 * magnitude) - magnitude
            cases.append((f"matrix-{magnitude}", values.half()))
    comparisons = []
    executable = str(Path(args.executable).resolve())
    checkpoint = str(Path(args.checkpoint).resolve())
    denied = False
    with tempfile.TemporaryDirectory(prefix="llm-inner-runtime-independent-") as cwd:
        for label, matrix in cases:
            with torch.no_grad():
                reference = model(inputs_embeds=matrix.unsqueeze(0), use_cache=False).logits[0, :, args.dimension]
            command = [executable] + [",".join(repr(float(x)) for x in row) for row in matrix]
            # macOS sandbox is a validation harness restriction, not a compiler
            # dependency. Do not modify or move the user's checkpoint files.
            if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists():
                profile = '(version 1)(allow default)(deny file-read* (subpath ' + json.dumps(checkpoint) + '))'
                command = ["/usr/bin/sandbox-exec", "-p", profile] + command
                denied = True
            result = subprocess.run(command, cwd=cwd, env={"PATH": os.environ.get("PATH", "")},
                                    text=True, capture_output=True, check=True)
            actual = [int(line) for line in result.stdout.splitlines()]
            expected = [widened_bits(x) for x in reference]
            if len(actual) != len(expected):
                raise ValueError("Executable did not return every requested position")
            for position, (got, want) in enumerate(zip(actual, expected)):
                comparisons.append({"case": label, "length": len(matrix), "position": position,
                                    "actualF64Bits": got, "pytorchF64Bits": want, "equal": got == want})
    report = {"dimension": args.dimension, "width": width, "vocab": vocab, "context": context,
              "input": "finite-f16-embedding-matrix", "torch": torch.__version__,
              "backend": "cpu-eager", "checkpointReadsDeniedAtRuntime": denied,
              "comparisons": comparisons, "exactFinalParity": all(c["equal"] for c in comparisons)}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    if not report["exactFinalParity"]:
        first = next(c for c in comparisons if not c["equal"])
        raise ValueError(f"Final logit mismatch: {first}")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}))


if __name__ == "__main__":
    main()
