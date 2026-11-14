#!/usr/bin/env python3
"""Capture a real model block and audit its TorchInductor lowering on one GPU."""

import argparse
import importlib.util
import json
import os
import platform
import statistics
import time
import traceback
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("convnext", "diffusion", "mamba"), required=True
    )
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--length", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_case(kind, batch, length, torch):
    torch.manual_seed(20261114)
    if kind == "convnext":
        import torchvision

        model = torchvision.models.convnext_tiny(weights=None)
        model = model.eval().to(device="cuda", dtype=torch.bfloat16)
        model = model.to(memory_format=torch.channels_last)
        sample = torch.randn(
            batch, 3, 224, 224, device="cuda", dtype=torch.bfloat16
        ).contiguous(memory_format=torch.channels_last)
        target = model.features[1][0]
        run_full = lambda: model(sample)
        detail = {"library": "torchvision", "version": torchvision.__version__}
    elif kind == "diffusion":
        import diffusers

        model = diffusers.UNet2DModel(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            norm_num_groups=32,
            norm_eps=1e-6,
            downsample_padding=0,
            freq_shift=1,
            flip_sin_to_cos=False,
        )
        model = model.eval().to(device="cuda", dtype=torch.bfloat16)
        model = model.to(memory_format=torch.channels_last)
        sample = torch.randn(
            batch, 3, 32, 32, device="cuda", dtype=torch.bfloat16
        ).contiguous(memory_format=torch.channels_last)
        target = model.up_blocks[2].resnets[0]
        run_full = lambda: model(sample, 10, return_dict=False)
        detail = {
            "library": "diffusers",
            "version": diffusers.__version__,
            "architecture": "google/ddpm-cifar10-32 config, random weights",
        }
    else:
        import transformers

        config = transformers.MambaConfig(
            hidden_size=768,
            state_size=16,
            expand=2,
            conv_kernel=4,
            time_step_rank=48,
            num_hidden_layers=24,
            vocab_size=50280,
            use_cache=False,
        )
        model = transformers.MambaModel(config).eval().to("cuda")
        ids = torch.randint(0, 50280, (batch, length), device="cuda")
        target = model.layers[0].mixer
        run_full = lambda: model(input_ids=ids, use_cache=False, return_dict=False)
        detail = {
            "library": "transformers",
            "version": transformers.__version__,
            "architecture": "state-spaces/mamba-130m-hf config, random weights",
            "mamba_ssm_installed": importlib.util.find_spec("mamba_ssm") is not None,
            "causal_conv1d_installed": (
                importlib.util.find_spec("causal_conv1d") is not None
            ),
            "mixer_use_mambapy": getattr(target, "use_mambapy", None),
            "mixer_use_associative_scan": getattr(
                target, "use_associative_scan", None
            ),
        }
    return model, target, run_full, detail


def describe(value, torch):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (tuple, list)):
        return [describe(item, torch) for item in value]
    if isinstance(value, dict):
        return {key: describe(item, torch) for key, item in value.items()}
    return repr(value)


def measure(call, torch, repetitions=30):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def profile_call(label, call, output, torch):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        call()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(output / f"{label}.trace.json"))
    return [
        event.name
        for event in profiler.events()
        if "CUDA" in str(event.device_type)
    ]


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "batch": args.batch,
        "length": args.length,
        "layout_optimization_env": os.environ.get(
            "TORCHINDUCTOR_LAYOUT_OPTIMIZATION", "unset"
        ),
        "benchmark_fusion_env": os.environ.get(
            "TORCHINDUCTOR_BENCHMARK_FUSION", "unset"
        ),
        "platform": platform.platform(),
        "status": "failed",
    }
    try:
        import torch

        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible CUDA GPU")
        if "A100" not in torch.cuda.get_device_name():
            raise RuntimeError("This scout requires an A100")
        import torch._inductor.config as inductor_config

        setting = {
            "convnext": ("layout_optimization", "TORCHINDUCTOR_LAYOUT_OPTIMIZATION"),
            "diffusion": ("benchmark_fusion", "TORCHINDUCTOR_BENCHMARK_FUSION"),
        }.get(args.model)
        if setting is not None:
            key, environment = setting
            if not hasattr(inductor_config, key):
                raise RuntimeError(f"Installed Inductor lacks {key}")
            if int(bool(getattr(inductor_config, key))) != int(
                os.environ[environment]
            ):
                raise RuntimeError(f"Installed Inductor ignored {environment}")
        result.update(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            layout_optimization_config=getattr(
                inductor_config, "layout_optimization", None
            ),
            benchmark_fusion_config=getattr(
                inductor_config, "benchmark_fusion", None
            ),
        )
        model, target, run_full, detail = make_case(
            args.model, args.batch, args.length, torch
        )
        result.update(detail)
        captured = {}

        def hook(_module, inputs, kwargs):
            if not captured:
                captured["args"] = inputs
                captured["kwargs"] = kwargs

        handle = target.register_forward_pre_hook(hook, with_kwargs=True)
        with torch.inference_mode():
            run_full()
        handle.remove()
        if not captured:
            raise RuntimeError("Full model forward did not reach target block")
        inputs, kwargs = captured["args"], captured["kwargs"]
        result.update(
            target=next(
                name for name, layer in model.named_modules() if layer is target
            ),
            input_spec=describe(inputs, torch),
            kwargs_spec=describe(kwargs, torch),
        )
        with torch.inference_mode():
            reference = target(*inputs, **kwargs)
        try:
            with torch.inference_mode():
                explanation = torch._dynamo.explain(target)(*inputs, **kwargs)
            result["dynamo_graph_count"] = explanation.graph_count
            result["dynamo_graph_break_count"] = explanation.graph_break_count
            result["dynamo_break_reasons"] = [
                str(reason) for reason in explanation.break_reasons
            ]
            (args.output / "dynamo_graphs.txt").write_text(
                "\n\n".join(str(graph.graph) for graph in explanation.graphs)
            )
        except Exception as error:
            result["dynamo_explain_error"] = repr(error)
        torch._dynamo.reset()
        start = time.perf_counter()
        compiled = torch.compile(
            target,
            backend="inductor",
            fullgraph=args.model != "mamba",
        )
        with torch.inference_mode():
            actual = compiled(*inputs, **kwargs)
        torch.cuda.synchronize()
        result["first_compiled_call_s"] = time.perf_counter() - start
        expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(reference)
        actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
        if expected_spec != actual_spec:
            raise AssertionError("Compiled output tree differs from eager")
        errors = []
        for expected, observed in zip(expected_leaves, actual_leaves):
            if not isinstance(expected, torch.Tensor):
                if expected != observed:
                    raise AssertionError("Compiled non-tensor output differs")
                continue
            if not torch.isfinite(observed).all():
                raise AssertionError("Compiled output is non-finite")
            tolerance = 2e-2 if expected.dtype == torch.bfloat16 else 1e-4
            errors.append((observed.float() - expected.float()).abs().max().item())
            result["max_abs_error"] = max(errors)
            torch.testing.assert_close(
                observed, expected, rtol=tolerance, atol=tolerance
            )
        result["max_abs_error"] = max(errors, default=0.0)
        with torch.inference_mode():
            eager_call = lambda: target(*inputs, **kwargs)
            compiled_call = lambda: compiled(*inputs, **kwargs)
            torch.cuda.synchronize()
            base_allocated = torch.cuda.memory_allocated()
            base_reserved = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            compiled_call()
            torch.cuda.synchronize()
            result["compiled_peak_allocated_bytes"] = (
                torch.cuda.max_memory_allocated()
            )
            result["compiled_peak_reserved_bytes"] = (
                torch.cuda.max_memory_reserved()
            )
            result["compiled_peak_allocated_over_baseline_bytes"] = (
                result["compiled_peak_allocated_bytes"] - base_allocated
            )
            result["compiled_peak_reserved_over_baseline_bytes"] = (
                result["compiled_peak_reserved_bytes"] - base_reserved
            )
            result["eager_timing"] = measure(eager_call, torch)
            result["compiled_timing"] = measure(compiled_call, torch)
            result["eager_cuda_events"] = profile_call(
                "eager", eager_call, args.output, torch
            )
            result["compiled_cuda_events"] = profile_call(
                "compiled", compiled_call, args.output, torch
            )
            result["eager_cuda_event_count"] = len(result["eager_cuda_events"])
            result["compiled_cuda_event_count"] = len(
                result["compiled_cuda_events"]
            )
        cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
        result["generated_python"] = [
            str(path.relative_to(cache)) for path in cache.rglob("*.py")
        ]
        result["status"] = "passed"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n"
        )
        print(json.dumps({key: value for key, value in result.items() if key not in (
            "eager_cuda_events", "compiled_cuda_events", "error"
        )}, indent=2, default=str), flush=True)
        if "error" in result:
            print(result["error"], flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
