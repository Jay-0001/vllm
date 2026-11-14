# M2 last-chance gate: deployed Qwen2.5-7B on A100

## Decision

Use one final **mechanism-first** Qwen-7B screen. Natural-language prompt
content alone does not alter the dense decoder graph at a fixed shape; request
length, concurrency, prefix reuse, and scheduling do change the shapes and
frequency of graph execution. A mixed serving trace can validate the value of
a proven graph-level improvement, but it cannot by itself establish a weak
fusion boundary.

The previous Qwen-0.5B M2 split screen produced the same five physical GPU
kernels for stock and both manual split arms. ConvNeXt's layout toggle also
produced identical physical code. Mamba's speedup came from an existing
compile-only associative-scan algorithm. The diffusion job failed before
model import and is not negative compiler evidence.

## Gate 1: inspect the existing 7B compiled graph without a GPU job

The earlier Qwen-7B latency run logged an AOT cache under
`specialization_experiments/cache/qwen7b-512x4-23650318`. On the Delta login
node, from `/u/jsaravananindira/vllm`, run:

```bash
bash jay_scripts/m2_qwen7b_cache_inventory.sh
```

The script copies only `.py`, `.json`, and `.txt` cache artifacts into tarballs
in `jay_outputs/m2_qwen7b_last_chance/inventory`. It does not start Slurm or
copy weights/binaries. Transfer those tarballs to this workspace for review.
The November 8 shape run used repository head
`a7cb57a3eae26a220fdc78527b285ad659afeb43`, while the November 14 scouts
used `14bdc66af753c800fd652639e185dc5cfd4e99cc`. The 512×4 cache's
exact head should be checked from its source metadata on Delta; do not assume
it matches either run. Use that artifact to choose a site, then reproduce any
candidate on the current deployed commit before claiming a result. If the cache
is missing, run the fallback current-commit graph dump:

```bash
sbatch jay_scripts/m2_qwen7b_compile_dump.sbatch
```

This is a single 3-minute A100 allocation based on approximately 2 minutes
estimated for 7B startup, compilation and two measured iterations. Its input
text is irrelevant to graph capture at the fixed batch/length; it is not the
real-request validation step. The job saves all artifacts under this
experiment directory and does not enable CUDA Graph capture.

Inventory only three regions with plausible physical alternatives:

1. Residual plus RMSNorm feeding the Q/K/V or MLP projections: look for a
   materialized conversion or an avoidable read/write around the GEMM.
2. MLP gate/up GEMM, SiLU-multiply, and down GEMM: look for a distinct legal
   epilogue or materialization choice, not a second Python call.
3. A reduction or fan-out where generated code shows separate reads of the
   same tensor or a layout conflict across consumers.

For each, record FX inputs/strides, actual generated launch sequence, bytes
written between operations, and whether the vLLM custom-op boundary prevents
Inductor from seeing a proposed fusion. Reject a site immediately if the
alternative lowers to the same physical code.

## Gate 2: one small oracle, conditional on Gate 1

For the highest-cost site with a real alternative, compare default Inductor,
its built-in fusion benchmarking or autotuning where applicable, and at most
two **legal, semantics-equivalent** partitions/layouts. Match dtype, shape,
weights, attention backend and compiler commit. Start with the frequent
decode token counts 1–4 from the prior serving trace, then one held-out
prefill or mixed batch size that changes the generated lowering. Check output
error, generated code, CUDA-event region latency, compile cost and peak memory.
Run paired arms in alternating order. For a service-relevant candidate, require
at least a 5% repeated GPU-region improvement on two shapes before serving.
A site that produces no distinct code, or an oracle matched by built-in tuning,
is closed without a serving sweep.

## Gate 3: real-request validation only for a surviving site

Replay a fixed text-request dataset with recorded prompt lengths, output caps,
arrival times and concurrency, using the *same request sequence* for stock and
oracle. Include both the previous low-rate regime and a burstier regime with
long prefill and simultaneous decode. Save the dataset hash and per-request
TTFT/TPOT, throughput, failure count, scheduler shapes, cache/capture mode and
GPU memory. Use a short paired A/B/A/B run; derive its Slurm limit from the
measured startup plus replay duration, capped at 1.5 times that estimate.
Do not call a collection of hand-written prompts a production trace. Require
at least a 2% repeated improvement in a relevant service metric (TTFT, TPOT or
goodput), with paired-run uncertainty excluding zero; otherwise classify the
kernel result as an isolated microbenchmark gain.

The prior 7B shape run used random prompt tokens, 128 requests at 1 request/s,
and recorded 5,744 decode iterations at scheduled token counts 1–4 versus
70 mixed and 58 prefill iterations. It gives a shape-frequency prior, not a
replayable natural-language request corpus.

## Stop rule

Close **M2 as a thesis prospect for this vLLM/A100 deployment** if Gate 1
finds no real candidate after the three regions, or Gate 2 finds that all
correct alternatives lower identically, lose, or are matched by built-in
tuning. If a distinct oracle wins, require a repeatable advantage on at least
two held-out shapes and a measured cause (conversion, materialization,
occupancy or spills) before Gate 3. A service gain must exceed paired-run
noise and a predeclared practical threshold. A negative result here does not
prove that every Inductor workload lacks a gap; it ends this bounded thesis
search and frees effort for the next ranked hypothesis.

## Final Q/K RoPE oracle (prepared, not run)

This last test uses the current checked-out vLLM/PyTorch commit. First submit
the existing [full-model compile dump](../../jay_scripts/m2_qwen7b_compile_dump.sbatch)
to confirm that deployed Qwen2.5-7B still emits separate Q and K RoPE
launches. Then submit the [isolated oracle array](../../jay_scripts/m2_qwen7b_qk_rope_oracle.sbatch):

```bash
cd /u/jsaravananindira/vllm
git status --short
git rev-parse HEAD
sbatch jay_scripts/m2_qwen7b_compile_dump.sbatch
sbatch jay_scripts/m2_qwen7b_qk_rope_oracle.sbatch
```

The array runs decode shapes 1 and 4 in task 0 and prefill shapes 512 and
2048 in task 1. Each task requests one A100 for 4 minutes 30 seconds. The
expected runtime is about 3 minutes per task including isolated compilation;
the full-model dump has its own 3-minute allocation. The array runs no model
weights: it compiles the actual `RotaryEmbedding.forward_static` Q/K region
with Qwen2.5-7B dimensions, then compares one Triton Q/K kernel. Inputs,
positions, BF16 cache, QKV and V view are shared between arms. Correctness is
checked against eager vLLM and the current Inductor result before timing.

Each `qk_rope_*_<job-id>/` result directory contains `environment.txt`,
`run.log`, `result.json`, Inductor's generated `output_code.py` under
`torch_compile_debug/`, and the combined kernel's PTX and Triton IR. Times
are CUDA graph replay microseconds per region call, with alternating arm
order. This isolates GPU region cost; it does not establish serving benefit.

Inspect `generated_baseline_wrappers[].launches` in `result.json` and the
full-model dump. If current Inductor already emits one launch, there is no
missing Q/K fusion to claim. If it emits two, require a correct combined
kernel to win by at least 5% repeatedly on two shapes, including a held-out
shape, before considering a serving replay. Otherwise close this bounded M2
prospect. The old-cache analysis is in
[`inventory/PHYSICAL_REGION_REVIEW.md`](inventory/PHYSICAL_REGION_REVIEW.md).
