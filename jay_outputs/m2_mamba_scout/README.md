# M2 Mamba graph-visibility scout

## Why this block

Mamba has a state-space mixer with input projection, depthwise causal
convolution, activation, selective scan, and output projection. In vLLM,
`vllm::mamba_mixer` is an explicit compilation split operation. Before testing
any partition, establish whether Inductor sees the mixer internals in an
independent PyTorch implementation, whether it produces finite/correct output,
and which operations remain opaque. The target is the first mixer's input from
a full `transformers.MambaModel` forward. This is Mamba-1 prefill only; it does
not test recurrent decode or establish a Mamba-2 issue.

The dimensions match `state-spaces/mamba-130m-hf` (hidden 768, expansion 2,
state 16, convolution width 4, 24 layers). We instantiate from config with
random weights. The job records optional kernel packages and mixer scan
settings, saved FX graphs, compiled wrappers, and CUDA traces. If custom kernels
dominate, label the limitation as visibility/dispatch, not Inductor fusion.

## Preparation and cases

From Delta's login node:

```bash
cd /u/jsaravananindira/vllm
.venv/bin/python -c 'import torch, transformers; from transformers import MambaModel; print(torch.__version__, transformers.__version__)'
sbatch jay_scripts/m2_mamba.sbatch
```

The installed vLLM dependency range for Transformers should supply Mamba. If
the import fails, resolve that in the login-node venv with `uv` before
submission. The slow or fast Mamba path may also require `mamba-ssm` and
`causal-conv1d`; do not install them blindly because changing the path changes
the experimental question. Record the actual path first.

| Array task | Prefill length | Batch | A100 limit |
| ---: | ---: | ---: | ---: |
| 0 | 32 | 1 | 20 min |
| 1 | 128 | 1 | 20 min |

Each task has a separate cache, debug files, profiler traces and JSON result
here; Slurm output is here too. Default array concurrency is one; override
with `sbatch --array=0-1%2 jay_scripts/m2_mamba.sbatch` if desired.

## Decision gate

First check finite output, numerical agreement, `dynamo_graph_count`, graph
breaks, visible FX operations and the generated wrappers. If the custom scan
or mixer is opaque, inspect vLLM's custom-op seam and compare equivalent
correct boundaries only after proving distinct physical code. A successful
Mamba prefill result alone cannot justify a cross-model fusion rule.

Sources: [Mamba-130M config](https://huggingface.co/state-spaces/mamba-130m-hf/blob/main/config.json),
[vLLM compilation config](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py),
[Transformers Mamba implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba/modeling_mamba.py).
