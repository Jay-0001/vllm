# M4 streamed reduction follow-up

This is the next rejection experiment after [array 23969387](../m4_peak_screen/RESULT_CARD_23969387.md).
No result is recorded here yet. The prior whole-graph tiled arm retained ten
partial down-projection outputs, leaving allocated peak almost unchanged.

This job compares compiled stock with an explicitly streamed arm at 2048
tokens. The streamed arm compiles only one tile's gate/up/down computation.
Python calls that compiled tile repeatedly and accumulates each partial into
one FP32 output buffer in place. The outer loop is deliberately outside the
compiled graph, so inspect the generated code and peak measurements to check
that partial outputs are reused. This is a memory-feasibility oracle, not an
optimized serving path.

Both arms run as **separate Python processes within one 20-minute, one-A100
allocation**. Each has its own compiler cache, debug code, environment record,
and result file under this directory. Timing and memory use separate steady
passes. The script reports CUDA event and synchronized CPU wall latency,
allocated/reserved peaks, sampled device VRAM, first-call time, and error.

From the M4 worktree root on Delta's login node:

```bash
git branch --show-current
git rev-parse --short HEAD
export M4_PYTHON=/u/jsaravananindira/vllm/.venv/bin/python
test -x "$M4_PYTHON"
"$M4_PYTHON" -c 'import torch,triton,vllm; print(torch.__version__,triton.__version__,vllm.__version__)'
sbatch jay_scripts/m4_stream_screen.sbatch
```

Inspect `jay_outputs/m4_stream_screen/<jobid>/stock/result.json` and
`.../stream/result.json`, the Slurm log, and compiler debug wrappers. Use the
same provisional screen gate as the first run: at 2048 tokens, at least 64 MiB
less incremental allocated peak, no more than 20% higher median graph latency,
and relative L2 error below 1%. Also examine synchronized wall latency and
reserved/device peaks. A memory win with a large latency loss motivates a
fused-kernel feasibility check only if a realistic near-capacity workload
benefits from that saved headroom.
