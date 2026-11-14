# M2 diffusion U-Net scout

## Why this block

A diffusion U-Net introduces GroupNorm, SiLU, convolutions, time-embedding
addition, skip connections, and attention outside the block. The target
`up_blocks[2].resnets[0]` is captured from a full 32×32 UNet2DModel forward,
so its feature shapes and skip input come from the actual model. This probes a
different reduction/layout/fan-out mechanism from decoder RMSNorm and ConvNeXt
LayerNorm. The configuration matches the public
`google/ddpm-cifar10-32` UNet topology; weights are random.

Channels-last is recommended in diffusion optimization guidance and is used
for both arms. The intervention toggles only Inductor's built-in fusion
benchmarking, prompted by an open PyTorch issue about missed
pointwise/reduction/transpose fusion. This may expose a ranking error; if the
same generated code appears, the default may already be best or the relevant
alternative may never have been generated.
Diffusion compilation is already studied; novelty would require
a measured, current, specific lowering miss.

## Preparation and cases

On Delta's login node, install `diffusers` in the existing venv if missing:

```bash
cd /u/jsaravananindira/vllm
uv pip install diffusers
.venv/bin/python -c 'import torch, diffusers; print(torch.__version__, diffusers.__version__)'
sbatch jay_scripts/m2_diffusion.sbatch
```

Do not install packages inside the GPU job. Record the resolved version from
`result.json`; pin that version for repeat/held-out experiments. No model
weight download or Hugging Face login is needed.

| Array task | Batch | Fusion benchmarking | A100 limit |
| ---: | ---: | ---: | ---: |
| 0 | 1 | off, stock | 20 min |
| 1 | 1 | on | 20 min |
| 2 | 2 | off, stock | 20 min |
| 3 | 2 | on | 20 min |

The default submission runs the batch-1 pair one GPU at a time. If it reveals
a correct and distinct lowering, submit the held-out pair with
`sbatch --array=2-3%2 jay_scripts/m2_diffusion.sbatch`. All Slurm logs, traces,
caches, debug files, metadata, and `result.json` files stay in this directory.

## Decision gate

Require correct compiled outputs, a captured Inductor graph, and a distinct
lowering before treating a timing difference as candidate M2 evidence. Inspect
GroupNorm reduction kernels, layout conversions, and intermediate writes in
the generated wrappers and traces. Repeat promising cases in alternating
order with held-out batch/size and a second U-Net before claiming a transferable
rule. If the compile fails or graph breaks, classify that as coverage rather
than fusion-boundary regret.

Sources: [DDPM CIFAR-10 config](https://huggingface.co/google/ddpm-cifar10-32/blob/2a68f6bc203d54b1578d0d52bcb5681fb066e95c/config.json),
[Diffusers compile guidance](https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0),
[PyTorch fusion issue](https://github.com/pytorch/pytorch/issues/128063).
