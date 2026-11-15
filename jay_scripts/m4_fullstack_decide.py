#!/usr/bin/env python3
"""Summarize paired 7B serving outcomes without hiding failed arms."""

import argparse
import json
import re
from pathlib import Path


def read_case(root, utilization, arm, trace):
    case_dir = root / f"{utilization}_{arm}"
    result_path = case_dir / f"{trace}.json"
    status_path = case_dir / "status.txt"
    status = status_path.read_text() if status_path.exists() else "missing"
    result = None
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            status += "\ninvalid_result_json"
    log_path = case_dir / "server.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    capacities = re.findall(r"GPU KV cache size:\s*([\d,]+) tokens", log)
    memory_path = case_dir / f"{trace}_device_mib.txt"
    samples = []
    if memory_path.exists():
        samples = [
            int(line.strip())
            for line in memory_path.read_text().splitlines()
            if line.strip().isdigit()
        ]
    return {
        "status": status.strip(),
        "result": result,
        "kv_cache_tokens": int(capacities[-1].replace(",", "")) if capacities else None,
        "device_peak_mib": max(samples, default=None),
        "device_sample_count": len(samples),
        "stream_active_count": log.count("M4_STREAM_ACTIVE"),
    }


def metrics(case):
    result = case["result"]
    if result is None:
        return None
    return {
        "completed": result.get("completed"),
        "failed": result.get("failed"),
        "request_throughput": result.get("request_throughput"),
        "output_throughput": result.get("output_throughput"),
        "p99_ttft_ms": result.get("p99_ttft_ms"),
        "p99_tpot_ms": result.get("p99_tpot_ms"),
        "device_peak_mib": case["device_peak_mib"],
        "device_sample_count": case["device_sample_count"],
        "kv_cache_tokens": case["kv_cache_tokens"],
    }


def text_match(a, b):
    if a["result"] is None or b["result"] is None:
        return None
    left = a["result"].get("generated_texts", [])
    right = b["result"].get("generated_texts", [])
    if len(left) != len(right) or not left:
        return None
    return sum(bool(x) and x == y for x, y in zip(left, right)) / len(left)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    all_cases = {}
    lines = [
        "# M4 full-stack decision summary",
        "",
        "One run per case. Inspect server and client logs before concluding.",
        "Device peaks are sampled and can miss short transients.",
        "",
        "| Utilization | Trace | Arm | Completed | Failed | Req/s | "
        "p99 TTFT ms | p99 TPOT ms | Device peak MiB | KV tokens |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for utilization in ("0.96", "0.99"):
        for trace in ("long_prefill", "mixed"):
            for arm in ("stock_default", "stock_eager", "stream_eager"):
                key = f"{utilization}_{arm}_{trace}"
                case = read_case(root, utilization, arm, trace)
                all_cases[key] = case
                values = metrics(case) or {}
                fields = [
                    utilization,
                    trace,
                    arm,
                    values.get("completed"),
                    values.get("failed"),
                    values.get("request_throughput"),
                    values.get("p99_ttft_ms"),
                    values.get("p99_tpot_ms"),
                    values.get("device_peak_mib"),
                    values.get("kv_cache_tokens"),
                ]
                lines.append("| " + " | ".join(map(str, fields)) + " |")

    lines.extend(["", "## Greedy output agreement", ""])
    for utilization in ("0.96", "0.99"):
        for trace in ("long_prefill", "mixed"):
            stock = all_cases[f"{utilization}_stock_eager_{trace}"]
            stream = all_cases[f"{utilization}_stream_eager_{trace}"]
            match = text_match(stock, stream)
            lines.append(f"- {utilization}, {trace}: {match}")
    lines.extend(
        [
            "",
            "## Pre-registered decision gate",
            "",
            "Continue M4 only if streamed execution produces a material",
            "capacity outcome (production stock fails while stream completes",
            "at the same setting, or at least 5% more KV tokens than eager",
            "stock), preserves at least 99% greedy outputs,",
            "and stays within 20% of production stock p99 TTFT and TPOT with",
            "no lower completed-request throughput. Compare eager stock and",
            "stream to isolate the tile change. A failed or missing case is",
            "evidence to inspect, not a zero in a favorable average.",
        ]
    )
    (root / "summary.json").write_text(json.dumps(all_cases, indent=2))
    (root / "DECISION.md").write_text("\n".join(lines) + "\n")
    print(root / "DECISION.md")


if __name__ == "__main__":
    main()
