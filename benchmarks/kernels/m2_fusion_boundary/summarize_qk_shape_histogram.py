"""Extract scheduled-token frequencies from vLLM iteration logs."""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

PATTERN = re.compile(
    r"Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, "
    r"(\d+) generation requests, (\d+) generation tokens, "
    r"iteration elapsed time: ([\d.]+) ms"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line in args.log.open(encoding="utf-8", errors="replace"):
        if "(dummy)" in line:
            continue
        match = PATTERN.search(line)
        if match is None:
            continue
        iteration, ctx_req, ctx_tok, gen_req, gen_tok, elapsed = match.groups()
        ctx_req, ctx_tok, gen_req, gen_tok = map(
            int, (ctx_req, ctx_tok, gen_req, gen_tok)
        )
        phase = "mixed" if ctx_req and gen_req else (
            "prefill" if ctx_req else "decode"
        )
        rows.append(
            (int(iteration), phase, ctx_req, ctx_tok, gen_req, gen_tok,
             ctx_tok + gen_tok, float(elapsed))
        )
    if not rows:
        raise SystemExit("No iteration details found in server log")
    with (args.output / "iterations.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("iteration", "phase", "context_requests",
                         "context_tokens", "generation_requests",
                         "generation_tokens", "scheduled_tokens",
                         "logged_cpu_ms"))
        writer.writerows(rows)
    counts = Counter((row[1], row[6]) for row in rows)
    with (args.output / "shape_histogram.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("phase", "scheduled_tokens", "iterations"))
        writer.writerows((phase, tokens, count)
                         for (phase, tokens), count in sorted(counts.items()))
    summary = {
        "iterations": len(rows),
        "phase_counts": dict(Counter(row[1] for row in rows)),
        "decode_le_4": sum(count for (phase, tokens), count in counts.items()
                           if phase == "decode" and tokens <= 4),
        "decode_8_to_256": sum(count for (phase, tokens), count in counts.items()
                               if phase == "decode" and 8 <= tokens <= 256),
    }
    (args.output / "shape_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
