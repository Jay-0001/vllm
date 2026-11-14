#!/usr/bin/env python3
"""Screen the physical memory cost of a Qwen-style gated MLP."""

import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--arm", choices=("stock", "tiled"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=25)
    return parser.parse_args()


def device_memory_mib():
    try:
        return int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as functional
    import triton

    assert torch.cuda.device_count() == 1, "Expected one visible A100"
    assert "A100" in torch.cuda.get_device_name(0)
    torch.manual_seed(20261114)
    hidden, intermediate = 3584, 18944
    dtype = torch.bfloat16
    x = torch.randn(args.tokens, hidden, device="cuda", dtype=dtype)
    gate_weight = torch.randn(
        intermediate, hidden, device="cuda", dtype=dtype
    ) / hidden**0.5
    up_weight = torch.randn(
        intermediate, hidden, device="cuda", dtype=dtype
    ) / hidden**0.5
    down_weight_t = torch.randn(
        intermediate, hidden, device="cuda", dtype=dtype
    ) / intermediate**0.5

    def stock():
        gate = x @ gate_weight.T
        up = x @ up_weight.T
        return (functional.silu(gate) * up) @ down_weight_t

    def tiled():
        accumulated = torch.zeros(
            args.tokens, hidden, device="cuda", dtype=torch.float32
        )
        for start in range(0, intermediate, args.tile):
            end = min(start + args.tile, intermediate)
            gate = x @ gate_weight[start:end].T
            up = x @ up_weight[start:end].T
            partial = (functional.silu(gate) * up) @ down_weight_t[start:end]
            accumulated = accumulated + partial.float()
        return accumulated.to(dtype)

    with torch.no_grad():
        reference = stock()
        torch.cuda.synchronize()
        run = torch.compile(
            stock if args.arm == "stock" else tiled, fullgraph=True
        )
        compile_start = time.perf_counter()
        candidate = run()
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - compile_start
        print(f"first_call_seconds={compile_seconds:.3f}", flush=True)
        finite = bool(torch.isfinite(candidate).all().item())
        difference = candidate.float() - reference.float()
        max_abs_error = float(difference.abs().max())
        reference_max_abs = float(reference.float().abs().max())
        relative_max_error = max_abs_error / max(reference_max_abs, 1.0)
        relative_l2_error = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float())
        )
        del reference, candidate
        gc.collect()

        for _ in range(3):
            result = run()
        torch.cuda.synchronize()
        del result
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        driver_before = device_memory_mib()

        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--loop-ms=20",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        events = []
        for _ in range(args.repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = run()
            end.record()
            events.append((begin, end))
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        del result
        sampler.terminate()
        try:
            sample_output, _ = sampler.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            sampler.kill()
            sample_output, _ = sampler.communicate()
        driver_samples = [
            int(line.strip())
            for line in sample_output.splitlines()
            if line.strip().isdigit()
        ]
        driver_after = device_memory_mib()
        latencies_ms = [begin.elapsed_time(end) for begin, end in events]

    record = {
        "arm": args.arm,
        "tokens": args.tokens,
        "tile": args.tile,
        "shape_source": "Qwen2.5-7B-Instruct hidden/intermediate dimensions",
        "hidden": hidden,
        "intermediate": intermediate,
        "dtype": str(dtype),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compile_seconds": compile_seconds,
        "finite": finite,
        "max_abs_error": max_abs_error,
        "relative_max_error": relative_max_error,
        "relative_l2_error": relative_l2_error,
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "incremental_allocated_bytes": peak_allocated - baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "incremental_reserved_bytes": peak_reserved - baseline_reserved,
        "driver_before_mib": driver_before,
        "driver_sample_max_mib": max(driver_samples, default=None),
        "driver_sample_count": len(driver_samples),
        "driver_after_mib": driver_after,
        "latencies_ms": latencies_ms,
        "latency_median_ms": statistics.median(latencies_ms),
        "latency_p95_ms": sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))],
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (args.output / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
