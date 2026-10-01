"""Check residual absorption endpoints; not final-logit acceptance."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    codes = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = codes.view(torch.float16)
    values = values[torch.isfinite(values) & (values != 0)]
    magnitudes = values.abs().double()
    _, exponents = torch.frexp(magnitudes)
    gaps = torch.where(magnitudes < 2 ** -14, 2 ** -25,
                       torch.ldexp(torch.ones_like(magnitudes), exponents - 13))
    margins = gaps * (1 - 2 ** -12)
    bounds = margins.half()
    bounds = torch.where(bounds.double() >= margins,
                         torch.nextafter(bounds, torch.zeros_like(bounds)), bounds)
    comparisons = []
    for sign in [-1, 1]:
        actual = values + sign * bounds
        differences = actual.view(torch.int16) != values.view(torch.int16)
        comparisons.append({"sign": sign, "scalars": actual.numel(),
                            "nonzeroCorrectionScalars": int((bounds != 0).sum()),
                            "bitDifferences": int(differences.sum())})
    # Rounded addition is monotone. Agreement at both correction endpoints
    # proves every finite half delta between them retains this nonzero x.
    report = {"rule": "R16(R32(x+delta)) == x inside a strict midpoint margin",
              "torch": torch.__version__, "backend": "cpu", "dtype": "float16",
              "comparisons": comparisons, "completeModelParity": False}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    assert not any(c["bitDifferences"] for c in comparisons), report
    print(json.dumps(report))


if __name__ == "__main__":
    main()
