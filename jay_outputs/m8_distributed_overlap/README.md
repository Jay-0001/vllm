# M8 tensor-parallel communication screen

**Status:** Prepared locally; no Delta job has been submitted. Hypothesis: for Qwen2.5-7B-Instruct at tensor parallel size 2, at least two shapes leave a measured communication stall after the faster tested vLLM all-reduce route. A faster isolated collective alone is insufficient.

## First gate

The screen holds model, dtype, tensor parallel size, CUDA graph defaults, memory limits, prompts, and generation parameters fixed. It compares vLLM's default all-reduce dispatch against `--disable-custom-all-reduce` (NCCL). The ABBA array order reduces time drift. The three shapes are batch 1 / 128 input / 32 output, batch 16 / 128 / 32, and batch 4 / 512 / 16. Each array element has a 20-minute cap and loads the model once for all three shapes. Six timed repeats follow two warmups per shape. The script saves individual latencies, startup time, output hashes, backend logs, GPU inventory and topology. Compare only runs with matching interconnect class; rerun if Slurm assigns different GPU pairs. It does not claim request-level serving performance.

The profile job records torch profiler traces for batch 1 and 16 under the default route. Inspect GPU kernel timelines for all-reduce duration, dependency waits, compute overlap, and whether the default route actually selected its custom kernel. The trace is diagnostic; compare unprofiled latency results for performance. A useful M8 finding requires a visible communication tail on both shapes, a legal independent compute window, and a route or overlap oracle that improves end-to-end latency beyond run-to-run variation. If the default path is already near the best route, communication is minor, or the two shapes lack independent work to hide it, stop this serving-based M8 screen.

`--enable-dbo` is not an applicable Qwen tensor-parallel baseline here: the fork documents it for DP+expert-parallel MoE with DeepEP. An FSDP/headroom hypothesis would need its own branch, workload and existing-control comparison.

## Delta preparation and submission

The local branch is `thesis/m8-distributed-overlap-screen` at `W:\Building\Thesis\vllm_m8`. After committing and pushing that branch, create the analogous Delta worktree from the existing clone:

```powershell
git -C W:\Building\Thesis\vllm_m8 push -u origin thesis/m8-distributed-overlap-screen
```

```bash
cd /u/jsaravananindira/vllm
git fetch origin thesis/m8-distributed-overlap-screen
git worktree add -b thesis/m8-distributed-overlap-screen /u/jsaravananindira/vllm_m8 origin/thesis/m8-distributed-overlap-screen
cd /u/jsaravananindira/vllm_m8
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python -e . --torch-backend=auto
.venv/bin/vllm download-kernels
mkdir -p jay_outputs/m8_distributed_overlap
sbatch jay_scripts/m8_tp_screen.sbatch
```

Review `screen_*/environment.txt`, `run.log`, and `results.json` before submitting profiles. If both A100s are visible and the result is promising:

```bash
cd /u/jsaravananindira/vllm_m8
sbatch jay_scripts/m8_tp_profile.sbatch
```

Run the one-time environment setup on a Delta compute allocation according to site policy. The experiment jobs require only the already used vLLM/PyTorch/Triton stack, model cache access, two A100s allocated on one node, and enough disk for traces. Nsight Systems is optional for a later finer-grained pass. Save a result card here after the jobs; keep raw outputs here as well.

Keep the original Delta checkout and its `.venv` free for the other Codex instance. This worktree has its own branch, virtual environment, cache directories and output tree. Do not run competing GPU measurements on the same allocated devices; queue the array or use separate allocations. A later M8 hypothesis gets another branch and worktree.

## Decision record

Record Delta job IDs, exact git SHA, topology (NVLink/PCIe path), selected all-reduce backend, paired latency medians and dispersion, output-hash agreement, trace attribution, and a **continue/stop** judgment. Do not infer a scheduling gap from route timing alone. If a tail survives, the next experiment should compare a legal fixed overlap oracle against the fastest current route on held-out shapes, with setup and memory cost counted.
