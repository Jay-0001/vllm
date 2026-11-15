"""Summarize the Qwen7B intermediate-shape Q/K RoPE crossover."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    summary = {"array_job_id": args.job_id, "cases": []}
    for task, name in enumerate(("mid_low", "mid_high")):
        path = args.root / f"qk_rope_{name}_{args.job_id}_{task}"
        result = json.loads((path / "result.json").read_text())
        assert result["shapes"] == name
        assert len(result["cases"]) == 3
        for case in result["cases"]:
            errors = case["errors"]["oracle_vs_inductor"]
            assert all(error == 0 for error in errors.values()), errors
            baseline = case["samples"]["inductor_us"]
            combined = case["samples"]["combined_us"]
            assert len(baseline) == len(combined) == 2
            summary["cases"].append(
                {
                    "tokens": case["tokens"],
                    "inductor_median_us": statistics.median(baseline),
                    "combined_median_us": statistics.median(combined),
                    "paired_improvement_pct": [
                        100 * (stock - oracle) / stock
                        for stock, oracle in zip(baseline, combined)
                    ],
                }
            )
    summary["cases"].sort(key=lambda case: case["tokens"])
    path = args.root / f"qk_mid_summary_{args.job_id}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
