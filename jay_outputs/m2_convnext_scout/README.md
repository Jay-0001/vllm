# M2 ConvNeXt layout scout

## Why this block

ConvNeXt-Tiny supplies a repeated, real inference graph whose CNBlock contains
depthwise convolution, NCHW-to-NHWC permutation, LayerNorm, two linear layers,
GELU, layer scale, and a residual. This creates a plausible conflict between
convolution and MLP layout preferences. The chosen block is
`features[1][0]`, captured during a complete model forward at 224×224.
TorchInductor's layout optimization is an existing alternative, so first ask
whether its two settings produce different generated code and whether its
default is already best. We are not claiming a weakness in advance.

The architecture comes from `torchvision.models.convnext_tiny(weights=None)`.
Random weights preserve the graph and timing path; they do not support model
quality claims. The input and weights use BF16 and channels-last memory format.

## Cases and submission

| Array task | Batch | Layout optimization | A100 limit |
| ---: | ---: | ---: | ---: |
| 0 | 1 | on, stock | 20 min |
| 1 | 1 | off | 20 min |
| 2 | 4 | on, stock | 20 min |
| 3 | 4 | off | 20 min |

From `/u/jsaravananindira/vllm` on Delta's login node:

```bash
test -x .venv/bin/python
.venv/bin/python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__)'
sbatch jay_scripts/m2_convnext.sbatch
```

The default submission runs the batch-1 pair with one GPU at a time. If the
pair compiles correctly and lowers differently, submit the held-out batch-4
pair with `sbatch --array=2-3%2 jay_scripts/m2_convnext.sbatch`. Each task has
its own 20-minute limit, cache, debug files, traces, and `result.json`
under this directory. The Slurm log is `slurm-<array-job>_<task>.out` here.
The script refuses a local non-Slurm launch.

## Decision gate

Compare tasks 0/1 and 2/3. Check `dynamo_graph_count`,
`dynamo_graph_break_count`, `generated_python`, compiled CUDA events, and the
actual wrapper/Triton code in each cache. A latency difference with identical
physical launches is not fusion-boundary regret. If code differs, repeat both
arms in alternating order and on a held-out shape before attributing a win to
layout choice. Profile conversion bytes, materialization, occupancy and spills
for a surviving loss; then test a legal local partition or layout oracle.

Source: [TorchVision ConvNeXt implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py).
