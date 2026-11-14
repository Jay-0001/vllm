# Diffusion U-Net M2 result card: Delta array 23967145

**Decision:** Inconclusive. Neither arm reached model construction, Dynamo,
Inductor, correctness checking, or timing. This is an environment/setup
failure, not evidence for or against the M2 hypothesis.

---

Both batch-1 cases (stock and `TORCHINDUCTOR_BENCHMARK_FUSION=1`) failed at
`import diffusers` with `ModuleNotFoundError: No module named 'diffusers'`.
Their `result.json` files report `status=failed` and contain no graph count or
latency. The A100, PyTorch `2.14.0+cu132`, and configuration flags were
recorded before the import.

On Delta's **login node**, use the existing venv and check/install the package:

```bash
cd /u/jsaravananindira/vllm
uv pip install diffusers
.venv/bin/python -c 'import diffusers; print(diffusers.__version__)'
sbatch jay_scripts/m2_diffusion.sbatch
```

Record the resolved version. The revised Slurm limit is 90 seconds per task,
using a roughly 60-second per-case estimate from the first-wave jobs. If the first
successful compiled block reveals distinct physical kernels and correct
output, then run the held-out batch-2 pair; otherwise stop this candidate.
