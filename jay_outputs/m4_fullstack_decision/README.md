# M4 full-stack decision experiment

This is the planned hard decision after the [streamed tile screen](../m4_stream_screen/RESULT_CARD_23999692.md).
It runs the actual Qwen2.5-7B-Instruct model through vLLM on one A100.
The experimental Qwen MLP route is enabled only for inputs of at least 512
tokens; short decode batches retain the ordinary path. It supports only
unquantized, single-GPU Qwen2. This branch-local intervention is an oracle
for capacity feasibility, not a candidate production implementation.

The job runs three arms at GPU-memory-utilization 0.96 and 0.99:

1. `stock_default`: current vLLM compilation and CUDA Graph defaults.
2. `stock_eager`: stock operations with `--enforce-eager`.
3. `stream_eager`: the opt-in MLP route with `--enforce-eager`.

The eager pair isolates the MLP change. Production stock is the deployment
baseline. Each arm serves two seeded request regimes: 96 long-prefill
requests of about 2048 input/64 output tokens at high concurrency, then 128
mixed-length requests of about 1536 input/128 output tokens at 4 req/s.
All use greedy decoding and fixed seed. Detailed per-request outputs,
TTFT/TPOT, errors, throughput, server logs, and sampled device VRAM are
saved under this directory. All arms use the same model, dtype, scheduler
limits, and GPU allocation. A failed server or client remains a failed case.
The streamed server log must contain `M4_STREAM_ACTIVE` to verify that its
long-prefill path actually ran.

## Delta submission

From `/u/jsaravananindira/vllm-m4` on the Delta login node, after updating
this branch:

```bash
export M4_PYTHON=/u/jsaravananindira/vllm/.venv/bin/python
test -x "$M4_PYTHON"
bash -n jay_scripts/m4_fullstack_decision.sbatch
"$M4_PYTHON" -c 'import ast,pathlib; ast.parse(pathlib.Path("jay_scripts/m4_fullstack_decide.py").read_text())'
sbatch jay_scripts/m4_fullstack_decision.sbatch
```

One job reserves one A100 for up to four hours. The script runs cases
sequentially, each server in a fresh process. Inspect
`jay_outputs/m4_fullstack_decision/<jobid>/DECISION.md`, `summary.json`,
each case's `server.log` and client JSON, and the Slurm log. The memory
utilization sweep may include failures; that is part of the result.

## Pre-registered decision

Continue M4 only if the streamed route produces a material capacity outcome
(production stock fails while stream completes at the same setting, or the
streamed route adds at least 5% KV tokens relative to eager stock),
preserves at least 99% greedy outputs, and stays within 20% of production
stock p99 TTFT and TPOT without lowering completed-request throughput.
Compare the eager pair to attribute differences to the MLP route. A failure
to meet these gates closes this MLP motif; it does not prove tile-level
memory optimization is impossible for all inference graphs.
