"""Screen Qwen-style decoder partition boundaries on one CUDA GPU."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


SIZES = {"0.5b": (896, 4864), "7b": (3584, 18944)}


def add_norm(x, residual, norm_weight):
    updated = x.float() + residual.float()
    scale = torch.rsqrt(updated.square().mean(dim=-1, keepdim=True) + 1e-6)
    normalized = (updated * scale).to(x.dtype) * norm_weight
    return normalized, updated


def mlp(normalized, gate_up_weight, down_weight):
    gate_up = normalized @ gate_up_weight
    gate, up = gate_up.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    return activated @ down_weight


def full(x, residual, norm_weight, gate_up_weight, down_weight):
    normalized, updated = add_norm(x, residual, norm_weight)
    return mlp(normalized, gate_up_weight, down_weight), updated


def norm_gate(x, residual, norm_weight, gate_up_weight):
    normalized, updated = add_norm(x, residual, norm_weight)
    return normalized @ gate_up_weight, updated


def activation_down(gate_up, down_weight):
    gate, up = gate_up.chunk(2, dim=-1)
    return (F.silu(gate) * up) @ down_weight


def make_runner(arm):
    compile_fn = lambda fn: torch.compile(fn, backend="inductor", fullgraph=True)
    if arm == "max_autotune":
        compiled = torch.compile(
            full, backend="inductor", fullgraph=True, mode="max-autotune"
        )
        return lambda args: compiled(*args)
    if arm in ("stock", "benchmark_fusion"):
        compiled = compile_fn(full)
        return lambda args: compiled(*args)
    if arm == "split_norm":
        first = compile_fn(add_norm)
        second = compile_fn(mlp)

        def run(args):
            x, residual, norm_weight, gate_up_weight, down_weight = args
            normalized, updated = first(x, residual, norm_weight)
            return second(normalized, gate_up_weight, down_weight), updated

        return run
    if arm == "split_activation":
        first = compile_fn(norm_gate)
        second = compile_fn(activation_down)

        def run(args):
            x, residual, norm_weight, gate_up_weight, down_weight = args
            gate_up, updated = first(x, residual, norm_weight, gate_up_weight)
            return second(gate_up, down_weight), updated

        return run
    raise ValueError(arm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=SIZES, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument(
        "--arm",
        choices=(
            "stock",
            "benchmark_fusion",
            "max_autotune",
            "split_norm",
            "split_activation",
        ),
        required=True,
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This screen requires exactly one visible CUDA GPU")
    if args.tokens < 1 or args.warmups < 1 or args.iterations < 1:
        parser.error("tokens, warmups, and iterations must be positive")
    if args.arm == "benchmark_fusion" and os.getenv(
        "TORCHINDUCTOR_BENCHMARK_FUSION"
    ) != "1":
        raise RuntimeError("Set TORCHINDUCTOR_BENCHMARK_FUSION=1 for this arm")
    if args.arm != "benchmark_fusion" and os.getenv(
        "TORCHINDUCTOR_BENCHMARK_FUSION"
    ) == "1":
        raise RuntimeError("Benchmark fusion must be off for other arms")

    torch.manual_seed(0)
    hidden, intermediate = SIZES[args.model_size]
    dtype = torch.bfloat16
    device = "cuda"
    x = torch.randn(args.tokens, hidden, device=device, dtype=dtype)
    residual = torch.randn_like(x)
    norm_weight = torch.ones(hidden, device=device, dtype=dtype)
    gate_up_weight = torch.randn(
        hidden, 2 * intermediate, device=device, dtype=dtype
    ) / hidden**0.5
    down_weight = torch.randn(
        intermediate, hidden, device=device, dtype=dtype
    ) / intermediate**0.5
    inputs = (x, residual, norm_weight, gate_up_weight, down_weight)
    runner = make_runner(args.arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        expected = full(*inputs)
        torch.cuda.synchronize()
        start_compile = time.perf_counter()
        actual = runner(inputs)
        torch.cuda.synchronize()
        first_call_s = time.perf_counter() - start_compile
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0.04, atol=0.04)
        errors = [
            (got.float() - want.float()).abs().max().item()
            for got, want in zip(actual, expected)
        ]

        for _ in range(args.warmups):
            runner(inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        gpu_ms = []
        cpu_ms = []
        for _ in range(args.iterations):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            cpu_start = time.perf_counter()
            begin.record()
            runner(inputs)
            end.record()
            end.synchronize()
            cpu_ms.append((time.perf_counter() - cpu_start) * 1000)
            gpu_ms.append(begin.elapsed_time(end))

        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        profile_error = None

        if args.profile:
            try:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    runner(inputs)
                    torch.cuda.synchronize()
                profiler.export_chrome_trace(
                    str(args.output.with_suffix(".trace.json"))
                )
            except RuntimeError as exc:
                profile_error = str(exc)

    result = {
        "arm": args.arm,
        "model_size": args.model_size,
        "tokens": args.tokens,
        "hidden": hidden,
        "intermediate": intermediate,
        "dtype": str(dtype),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "benchmark_fusion_env": os.getenv("TORCHINDUCTOR_BENCHMARK_FUSION", "0"),
        "first_call_s": first_call_s,
        "gpu_ms": gpu_ms,
        "gpu_median_ms": statistics.median(gpu_ms),
        "cpu_median_ms": statistics.median(cpu_ms),
        "max_abs_error": errors,
        "allocated_before_bytes": allocated_before,
        "reserved_before_bytes": reserved_before,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "profile_error": profile_error,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "gpu_ms"}))


if __name__ == "__main__":
    main()
