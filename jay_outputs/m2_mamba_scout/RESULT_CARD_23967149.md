# Mamba-130M M2 result card: Delta array 23967149

**Decision:** The mixer is visible to Inductor and compiles correctly, but the
large eager-to-compiled gain is an algorithm/path change, not evidence that
Inductor chose a weak fusion boundary. Do not promote M2 from this result.

---

The target was `layers.0.mixer` captured from a full config-only Mamba-130M
forward at batch 1. It used FP32, no recurrent cache, and no installed
`mamba_ssm` or `causal_conv1d` package. Both lengths had one Dynamo graph and
zero graph breaks; compiled output was finite and passed the eager comparison.

| Prefill tokens | Eager median | Compiled median | Ratio | Eager / compiled GPU launches | Max absolute error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 3,077.47 µs | 304.61 µs | 10.10× | 121 / 13 | 0.0000687 |
| 128 | 10,851.49 µs | 317.90 µs | 34.13× | 409 / 13 | 0.0001221 |

These are 30 CUDA-event samples for one mixer, not full-model serving latency.
The captured FX graph contains `torch.ops.higher_order.associative_scan` at
both lengths. The eager profiler instead shows token-count-dependent repeated
elementwise and GEMV launches. Transformers' Mamba implementation selects the
associative scan during Dynamo compilation and otherwise uses a sequential
fallback; the measured comparison crosses that source-level branch. The
compiled 13-launch path is already much better than this fallback.

The next fair baseline is the optimized Mamba kernel route used in real
deployment, with the same weights, dtype, sequence phase and output tolerance.
Only if a distinct legal Inductor partition beats that route would M2 have a
Mamba-specific candidate. The current prefill-only result says nothing about
stateful decode or Mamba-2.

Source: [Transformers Mamba scan selection](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba/modeling_mamba.py).
