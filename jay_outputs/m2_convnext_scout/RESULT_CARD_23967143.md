# ConvNeXt-Tiny M2 result card: Delta array 23967143

**Decision:** Close this batch-1, channels-last layout-toggle candidate. Do not
spend A100 time on the planned batch-4 pair from this same candidate unless a
separate graph inspection identifies a real layout alternative.

---

Both arms captured `features.1.0` from the full ConvNeXt-Tiny forward at
`[1, 96, 56, 56]`, BF16 channels-last. Both passed correctness with zero
observed maximum absolute difference from eager. PyTorch was `2.14.0+cu132`,
TorchVision `0.29.0+cu132`, on one A100-SXM4-40GB. Each had one Dynamo graph
and no breaks.

| Inductor layout optimization | Eager median | Compiled median | GPU launches |
| --- | ---: | ---: | ---: |
| On, stock | 209.73 µs | 220.43 µs | 6 |
| Off | 211.02 µs | 221.52 µs | 6 |

Each median is from 30 CUDA-event samples. Compiled times range 214–332 µs
with layout optimization on and 217–334 µs with it off, so the 1.09 µs median
difference is not persuasive. The eager path launches eight kernels.

The two generated `output_code.py` files differ only in the absolute cache
paths embedded as comments. Both call the same depthwise convolution, fused
LayerNorm/permute Triton kernel, two GEMMs, GELU Triton kernel and residual
Triton kernel, with the same strides and six physical launches. This toggle
therefore did not expose a competing layout or fusion boundary at this site.
The slightly slower compiled median than eager is not a validated backend
regression: only one process per arm was sampled, in fixed order, and the
CUDA-event measurement can include host launch gaps.

Next inspection should seek a *different* connected graph region with a
materializing conversion or competing consumer layout, rather than sweeping
more batches of this unchanged block. The raw JSON, cache and traces live in
the two `23967143_*` directories beside this card.
