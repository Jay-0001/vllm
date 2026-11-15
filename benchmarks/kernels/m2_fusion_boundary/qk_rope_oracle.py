"""Compare Inductor's Qwen2.5-7B Q/K RoPE region with one Triton launch."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.testing import do_bench_cudagraph

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

HEAD_DIM = 128
Q_HEADS = 28
K_HEADS = 4
Q_WIDTH = Q_HEADS * HEAD_DIM
K_WIDTH = K_HEADS * HEAD_DIM
QK_WIDTH = Q_WIDTH + K_WIDTH
QKV_WIDTH = Q_WIDTH + 2 * K_WIDTH
CACHE_ROWS = 32768
BLOCK = 256


def native_region(qkv, positions, cache):
    """Use the model's native, out-of-place Neox RoPE graph."""
    q, k, v = qkv.split((Q_WIDTH, K_WIDTH, K_WIDTH), dim=-1)
    q, k = RotaryEmbedding.forward_static(
        positions, q, k, HEAD_DIM, HEAD_DIM, cache, True
    )
    return q, k, v


@triton.jit
def combined_qk_kernel(
    qkv,
    positions,
    cache,
    q_out,
    k_out,
    tokens: tl.constexpr,
    q_width: tl.constexpr,
    k_width: tl.constexpr,
    qkv_width: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
):
    qk_width = q_width + k_width
    half_dim = head_dim // 2
    index = tl.program_id(0) * block + tl.arange(0, block)
    mask = index < tokens * qk_width
    token = index // qk_width
    column = index % qk_width
    rotary_column = column % head_dim
    paired = tl.where(
        rotary_column < half_dim, column + half_dim, column - half_dim
    )
    position = tl.load(positions + token, mask, other=0)
    x = tl.load(qkv + token * qkv_width + column, mask, other=0).to(tl.float32)
    xp = tl.load(qkv + token * qkv_width + paired, mask, other=0).to(
        tl.float32
    )
    frequency = position * head_dim + rotary_column % half_dim
    cosine = tl.load(cache + frequency, mask, other=0).to(tl.float32)
    sine = tl.load(cache + frequency + half_dim, mask, other=0).to(tl.float32)
    rotated = tl.where(
        rotary_column < half_dim, x * cosine - xp * sine, x * cosine + xp * sine
    )
    tl.store(q_out + token * q_width + column, rotated, mask & (column < q_width))
    tl.store(
        k_out + token * k_width + column - q_width,
        rotated,
        mask & (column >= q_width),
    )


def make_cache():
    frequency = 1_000_000.0 ** (
        -torch.arange(64, device="cuda", dtype=torch.float32) / 64
    )
    angles = torch.arange(CACHE_ROWS, device="cuda")[:, None] * frequency
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)


def check_outputs(reference, candidate):
    errors = {}
    for name, expected, actual in zip(("q", "k", "v"), reference, candidate):
        assert expected.shape == actual.shape, name
        assert expected.dtype == actual.dtype, name
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
        errors[name] = (actual.float() - expected.float()).abs().max().item()
    return errors


def measured(fn, repeats):
    return float(do_bench_cudagraph(fn, rep=repeats)) * 1000


def inspect_baseline(debug_dir):
    paths = sorted(debug_dir.rglob("output_code.py"))
    wrappers = []
    for path in paths:
        code = path.read_text()
        call = code.split("def call(", 1)[-1]
        call = call.split("def benchmark_compiled_module", 1)[0]
        launches = re.findall(r"\b([\w.]+)\.run\(", call)
        wrappers.append({"path": str(path), "launches": launches})
    return wrappers


def run_shape(tokens, cache, output, repeats):
    torch.manual_seed(0)
    qkv = torch.randn(tokens, QKV_WIDTH, device="cuda", dtype=torch.bfloat16)
    positions = torch.randint(
        0, CACHE_ROWS, (tokens,), device="cuda", dtype=torch.int64
    )
    compiled = torch.compile(native_region, backend="inductor", fullgraph=True)
    q_out = torch.empty((tokens, Q_WIDTH), device="cuda", dtype=torch.bfloat16)
    k_out = torch.empty((tokens, K_WIDTH), device="cuda", dtype=torch.bfloat16)

    def oracle():
        kernel = combined_qk_kernel[(triton.cdiv(tokens * QK_WIDTH, BLOCK),)](
            qkv,
            positions,
            cache,
            q_out,
            k_out,
            tokens,
            Q_WIDTH,
            K_WIDTH,
            QKV_WIDTH,
            HEAD_DIM,
            BLOCK,
        )
        return (q_out, k_out, qkv[:, QK_WIDTH:]), kernel

    eager = native_region(qkv, positions, cache)
    baseline = compiled(qkv, positions, cache)
    candidate, kernel = oracle()
    torch.cuda.synchronize()
    errors = {
        "inductor_vs_eager": check_outputs(eager, baseline),
        "oracle_vs_inductor": check_outputs(baseline, candidate),
    }
    assert candidate[2].data_ptr() == baseline[2].data_ptr()
    assert candidate[2].stride() == baseline[2].stride()

    # Warm both paths, then alternate measurement order to reduce drift.
    for _ in range(10):
        compiled(qkv, positions, cache)
        oracle()
    torch.cuda.synchronize()
    samples = {"inductor_us": [], "combined_us": []}
    for order in (("inductor", "combined"), ("combined", "inductor")):
        for arm in order:
            fn = (
                lambda: compiled(qkv, positions, cache)
                if arm == "inductor"
                else oracle()[0]
            )
            samples[f"{arm}_us"].append(measured(fn, repeats))

    asm = getattr(kernel, "asm", {})
    for kind in ("ptx", "ttgir"):
        if kind in asm:
            (output / f"combined_{tokens}.{kind}").write_text(str(asm[kind]))
    return {"tokens": tokens, "errors": errors, "samples": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", choices=("decode", "prefill"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bench-rep-ms", type=int, default=40)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1, "A single A100 is required"
    args.output.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output / "torch_compile_debug"
    assert debug_dir.exists(), "Set TORCH_COMPILE_DEBUG_DIR before Python starts"
    shapes = (1, 4) if args.shapes == "decode" else (512, 2048)
    cache = make_cache()
    result = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "shapes": args.shapes,
        "timing": "Triton CUDA graph replay; microseconds per region call",
        "cases": [],
    }
    for tokens in shapes:
        case = run_shape(tokens, cache, args.output, args.bench_rep_ms)
        result["cases"].append(case)
        print(json.dumps(case), flush=True)
    result["generated_baseline_wrappers"] = inspect_baseline(debug_dir)
    assert result["generated_baseline_wrappers"], "No Inductor code dump was found"
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
