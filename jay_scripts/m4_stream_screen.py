#!/usr/bin/env python3
"""Compare a compiled MLP with an explicitly streamed tile reduction."""

import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("stock", "stream"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--tile", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=25)
    return parser.parse_args()


def device_memory_mib():
    value = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(value.strip())


def sample_driver_memory(run, repeats, torch):
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
    try:
        for _ in range(repeats):
            result = run()
            del result
        torch.cuda.synchronize()
    finally:
        sampler.terminate()
        try:
            output, _ = sampler.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            sampler.kill()
            output, _ = sampler.communicate()
    samples = [int(line) for line in output.splitlines() if line.strip().isdigit()]
    return max(samples, default=None), len(samples)


def main():
    args = arguments()
    assert args.tokens > 0 and args.tile > 0 and args.repeats > 0
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as functional
    import triton

    assert torch.cuda.device_count() == 1
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

    def tile_kernel(input_tensor, gate_block, up_block, down_block):
        gate = input_tensor @ gate_block.T
        up = input_tensor @ up_block.T
        return (functional.silu(gate) * up) @ down_block

    compiled_tile = torch.compile(tile_kernel, fullgraph=True)

    def stream():
        accumulator = torch.zeros(
            args.tokens, hidden, device="cuda", dtype=torch.float32
        )
        for start in range(0, intermediate, args.tile):
            end = min(start + args.tile, intermediate)
            partial = compiled_tile(
                x,
                gate_weight[start:end],
                up_weight[start:end],
                down_weight_t[start:end],
            )
            accumulator.add_(partial.float())
            del partial
        return accumulator.to(dtype)

    with torch.no_grad():
        reference = stock()
        torch.cuda.synchronize()
        run = torch.compile(stock, fullgraph=True) if args.arm == "stock" else stream
        start = time.perf_counter()
        candidate = run()
        torch.cuda.synchronize()
        first_call_seconds = time.perf_counter() - start
        print(f"first_call_seconds={first_call_seconds:.3f}", flush=True)
        difference = candidate.float() - reference.float()
        relative_l2_error = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float())
        )
        max_abs_error = float(difference.abs().max())
        finite = bool(torch.isfinite(candidate).all().item())
        del reference, candidate, difference
        gc.collect()

        for _ in range(3):
            result = run()
            del result
        torch.cuda.synchronize()

        gpu_ms = []
        wall_ms = []
        for _ in range(args.repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            begin.record()
            result = run()
            end.record()
            torch.cuda.synchronize()
            gpu_ms.append(begin.elapsed_time(end))
            wall_ms.append((time.perf_counter() - wall_start) * 1000)
            del result
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        driver_before = device_memory_mib()
        driver_sample_max, driver_sample_count = sample_driver_memory(
            run, args.repeats, torch
        )
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        driver_after = device_memory_mib()

    record = {
        "arm": args.arm,
        "tokens": args.tokens,
        "tile": args.tile,
        "repeats": args.repeats,
        "hidden": hidden,
        "intermediate": intermediate,
        "dtype": str(dtype),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "first_call_seconds": first_call_seconds,
        "dynamo_unique_graphs": torch._dynamo.utils.counters["stats"].get(
            "unique_graphs", 0
        ),
        "finite": finite,
        "relative_l2_error": relative_l2_error,
        "max_abs_error": max_abs_error,
        "gpu_ms": gpu_ms,
        "gpu_median_ms": statistics.median(gpu_ms),
        "wall_ms": wall_ms,
        "wall_median_ms": statistics.median(wall_ms),
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "incremental_allocated_bytes": peak_allocated - baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "incremental_reserved_bytes": peak_reserved - baseline_reserved,
        "driver_before_mib": driver_before,
        "driver_sample_max_mib": driver_sample_max,
        "driver_sample_count": driver_sample_count,
        "driver_after_mib": driver_after,
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (args.output / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
