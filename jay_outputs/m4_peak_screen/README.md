# M4 physical peak rejection screen

This is a proposed A100 experiment. No result is recorded yet.

The motif is the Qwen2.5-7B-Instruct gated MLP with hidden size 3584 and
intermediate size 18944, expressed in PyTorch with random BF16 weights. The
stock arm compiles gate projection, up projection, SiLU/product, and down
projection as one graph. The tiled arm computes 2048 intermediate channels at
a time and accumulates partial down projections. Both arms use the same inputs
and operations, apart from reduction order. This screen measures a decoder
subgraph, not vLLM request latency or a production integration.

The first question is whether generated stock code writes full intermediate
tensors. Inspect each stock `debug` directory and compiler cache before
interpreting peak differences. A tiled implementation is interesting only if
it saves actual device memory with acceptable graph latency and numerical
error. If stock already avoids the full write, or if chunking is slower for
no material peak gain, record the negative result and stop this motif.

## Delta submission

Fetch or copy this branch into a separate Delta worktree, such as
`/u/jsaravananindira/vllm-m4`, on the login node. Use the existing tested
CUDA Python environment through `M4_PYTHON`, or create `.venv` in that
worktree. This screen only imports PyTorch and Triton for computation; the
vLLM import records its version and path. From the M4 repository root:

```bash
git branch --show-current
git rev-parse --short HEAD
export M4_PYTHON=/u/jsaravananindira/vllm/.venv/bin/python
test -x "$M4_PYTHON"
"$M4_PYTHON" -c 'import torch,triton,vllm; print(torch.__version__,triton.__version__,vllm.__version__)'
sbatch --export=ALL jay_scripts/m4_peak_screen.sbatch
```

The default array runs one case at a time. Each case gets one A100 and a
20-minute limit. Cases 0–5 pair stock/tiled at 64, 512, and 2048 tokens.
For two available GPUs, `sbatch --array=0-5%2
jay_scripts/m4_peak_screen.sbatch` allows two cases concurrently. Do not run
the sbatch file as a local shell script.

Each task writes `result.json`, `environment.txt`, generated-code debug files,
compiler cache, and a Slurm log under this directory. The two arms are fresh
processes with separate compiler caches. The script records first-call compile
time, CUDA event latency, numerical error, allocated/reserved peaks, and
20 ms sampled device memory. Timing and memory use separate steady passes.
The device samples can miss a short spike; use
PyTorch's exact allocator peaks as the primary transient measure. Persistent
weights and input remain live in both arms. No CUDA Graph or KV cache is used.

## Decision rule and next gate

First reject the motif if the stock generated code lacks a full avoidable
intermediate. Otherwise compare paired shapes and inspect the difference
between peak and baseline allocation. Check output error and compile time.
One favorable shape is a lead, not a thesis result. Only a repeatable,
material peak reduction within a stated latency target warrants alternating
repetitions, a memory-pressure/OOM test, and an actual vLLM integration check.

No extra Python package is required beyond the CUDA PyTorch, Triton, and vLLM
already used in the Delta environment. `nvidia-smi` is required for sampled
device VRAM. Nsight tools are optional for a later surviving candidate.
