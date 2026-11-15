"""Measure deterministic Qwen7B batch latency and record generated tokens."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    prompts = [
        {"prompt_token_ids": [(i * 17 + j * 13) % 10000 for j in range(512)]}
        for i in range(4)
    ]
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="bfloat16",
        seed=0,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=8,
        gpu_memory_utilization=0.80,
        compilation_config={"cudagraph_mode": "NONE", "compile_sizes": [4]},
    )
    cases = (
        ("decode", 8 if args.smoke else 128, 2 if args.smoke else 12),
        ("prefill", 1, 2 if args.smoke else 12),
    )
    result = {"prompt_sha256": digest(prompts), "cases": []}
    for label, output_len, iterations in cases:
        sampling = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=output_len,
            detokenize=False,
        )
        llm.generate(prompts, sampling, use_tqdm=False)
        latencies = []
        token_hashes = []
        for _ in range(iterations):
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            latencies.append(time.perf_counter() - start)
            token_hashes.append(
                digest([list(item.outputs[0].token_ids) for item in outputs])
            )
        assert len(set(token_hashes)) == 1, f"Nondeterministic {label} outputs"
        case = {
            "name": label,
            "input_len": 512,
            "batch_size": 4,
            "output_len": output_len,
            "latencies_seconds": latencies,
            "token_sha256": token_hashes[0],
        }
        result["cases"].append(case)
        print(json.dumps(case), flush=True)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
