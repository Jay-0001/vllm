"""Validate the ABBA model run and summarize paired batch latency."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    arms = ("stock", "oracle", "oracle", "stock")
    runs = []
    for index, arm in enumerate(arms):
        path = args.root / f"model_{arm}_{args.job_id}_{index}" / "result.json"
        runs.append(json.loads(path.read_text()))
    assert len({run["prompt_sha256"] for run in runs}) == 1
    summary = {"job_id": args.job_id, "cases": []}
    for case_index in range(2):
        cases = [run["cases"][case_index] for run in runs]
        assert len({case["name"] for case in cases}) == 1
        assert len({case["token_sha256"] for case in cases}) == 1, (
            "Model output tokens differ between arms"
        )
        medians = [statistics.median(case["latencies_seconds"]) for case in cases]
        stock = statistics.mean((medians[0], medians[3]))
        oracle = statistics.mean((medians[1], medians[2]))
        pair_improvements = [
            100 * (medians[0] - medians[1]) / medians[0],
            100 * (medians[3] - medians[2]) / medians[3],
        ]
        summary["cases"].append(
            {
                "name": cases[0]["name"],
                "run_medians_seconds": medians,
                "stock_mean_median_seconds": stock,
                "oracle_mean_median_seconds": oracle,
                "oracle_improvement_pct": 100 * (stock - oracle) / stock,
                "pair_improvements_pct": pair_improvements,
                "token_sha256": cases[0]["token_sha256"],
            }
        )
    path = args.root / f"model_comparison_{args.job_id}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
