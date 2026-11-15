#!/usr/bin/env python3
"""Measure fixed tensor-parallel inference shapes in one vLLM engine."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from vllm import LLM, SamplingParams


SHAPES = (("decode_1", 1, 128, 32), ("decode_16", 16, 128, 32),
          ("mixed_4", 4, 512, 16))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=6)
    args = parser.parse_args()

    rng = np.random.default_rng(1)
    cases = []
    for name, batch, input_len, output_len in SHAPES:
        prompts = [{"prompt_token_ids": rng.integers(100, 10000, input_len).tolist()}
                   for _ in range(batch)]
        cases.append((name, batch, input_len, output_len, prompts))

    engine_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=2,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        seed=1,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=16,
        gpu_memory_utilization=0.80,
        enable_prefix_caching=False,
    )
    startup_seconds = time.perf_counter() - engine_start
    records = []
    for name, batch, input_len, output_len, prompts in cases:
        params = SamplingParams(temperature=0, ignore_eos=True,
                                max_tokens=output_len, detokenize=False)
        latencies = []
        digests = []
        for iteration in range(args.warmups + args.repetitions):
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            elapsed = time.perf_counter() - start
            tokens = [result.outputs[0].token_ids for result in outputs]
            current_digest = hashlib.sha256(
                json.dumps(tokens).encode("utf-8")
            ).hexdigest()
            if iteration >= args.warmups:
                latencies.append(elapsed)
                digests.append(current_digest)
        records.append({
            "shape": name,
            "batch": batch,
            "input_tokens": input_len,
            "output_tokens": output_len,
            "latencies_seconds": latencies,
            "median_seconds": statistics.median(latencies),
            "output_sha256": digests,
            "stable_output": len(set(digests)) == 1,
        })
        print(json.dumps(records[-1]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": args.model,
        "route": "nccl" if args.disable_custom_all_reduce else "default",
        "startup_seconds": startup_seconds,
        "shapes": records,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
