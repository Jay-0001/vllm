# M4 initial physical peak screen — array 23969387

## Setup and scope

Six fresh processes on one visible A100-SXM4-40GB, PyTorch 2.14.0+cu132,
Triton 3.8.0, BF16, random Qwen2.5-7B MLP-shaped weights, 25 steady CUDA
event samples per case. Each case was run once. The six task job IDs differ
from the common array ID used in folder names. This is a PyTorch subgraph
screen, not a vLLM serving or near-OOM experiment.

## Paired results

| Tokens | Stock / tiled median (ms) | Tiled latency penalty | Stock / tiled incremental allocated peak (MiB) | Saving | Stock / tiled incremental reserved peak (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.3645 / 0.6963 | +91.0% | 5.06 / 4.88 | 0.19 MiB | 0 / 4 |
| 512 | 0.9062 / 1.9735 | +117.8% | 40.50 / 40.38 | 0.12 MiB | 0 / 2 |
| 2048 | 4.0966 / 5.8540 | +42.9% | 162 / 156 | 6 MiB | 74 / 0 |

At 2048 tokens, both outputs were finite; relative L2 error against eager
stock was 0.00420 for compiled stock and 0.00438 for tiled. This meets the
provisional 1% error bound. Compile first-call times were 4.90 and 6.39 s.
The sampled device memory rose from 1125 to 1199 MiB for stock and remained
at 1125 MiB for tiled, but the sampler captured only four and six points.
The allocator's exact reserved high-water marks corroborate a 74 MiB
difference in this run. It may depend on cache history and needs paired
replication before a physical-VRAM claim.

## Mechanism visible in generated code

The 2048-token [stock wrapper](23969387_4_2048_stock/debug/torch_compile_debug/run_2026_11_14_12_30_50_795119-pid_2082425/torchinductor/model__0_inference_0.0/output_code.py)
allocates two `(2048, 18944)` BF16 tensors, each about 74 MiB. It reuses the
first for fused SiLU/product, then runs a separate down-projection GEMM. Thus
the first materialization premise holds for this extracted motif.

The [tiled wrapper](23969387_5_2048_tiled/debug/torch_compile_debug/run_2026_11_14_12_31_18_854120-pid_903673/torchinductor/model__0_inference_0.0/output_code.py)
uses ten tiles, but makes ten `(2048, 3584)` partial down-projection outputs
of about 14 MiB each. Its final Triton add reads all ten partials at once.
Those live outputs alone account for roughly 140 MiB, explaining why the
allocated peak remains near stock despite smaller gate/up tiles. The graph
has 30 GEMMs rather than stock's three, explaining the latency cost.

## Decision

**Reject this tiled PyTorch formulation.** At 2048 tokens it misses both
provisional thresholds: at least 64 MiB lower incremental allocated peak
and at most 20% median latency penalty. Smaller shapes also lose on latency.
The observation does **not** reject M4 generally: stock materializes full
buffers, while this oracle fails to stream its partial reductions.

The next minimal screen should enforce a single accumulator across tiles,
for example a compiled per-tile function called in a Python loop with an
in-place accumulation between calls. Inspect generated code to verify that
only one tile's gate/up buffers and one partial output remain live. Measure
allocated, reserved, and device peak plus latency on the 2048-token shape
first, in paired fresh processes on the same GPU. If that achieves substantial
peak reduction but remains far beyond the latency target, a fused kernel is
only justified if a realistic near-memory-limit workload needs the saved
capacity. Otherwise close this motif and move to the next candidate.
