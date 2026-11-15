# Q/K RoPE crossover and burst-shape screens

The first Qwen7B region screen tested scheduled token counts 1, 4, 512 and
2048. The combined kernel won only on the two small shapes; no intermediate
sizes were measured. This screen fills that gap with 8, 16, 32, 64, 128 and
256, while a separate bursty serving run records how often those shapes
occur. These are scheduled tokens in one forward pass, not request lengths.

## Delta commands

```bash
cd /u/jsaravananindira/vllm
git pull --ff-only origin thesis/m2-fusion-boundary-screen

# Two independent A100 tasks, three shapes each.
sbatch jay_scripts/m2_qwen7b_qk_mid_sweep.sbatch

# Substitute the returned array job ID after both tasks finish.
.venv/bin/python benchmarks/kernels/m2_fusion_boundary/summarize_qk_mid_sweep.py \
  --root jay_outputs/m2_qwen7b_last_chance --job-id ARRAY_JOB_ID

# One separate A100 job to measure a busier scheduler shape histogram.
sbatch jay_scripts/m2_qwen7b_burst_shape_hist.sbatch
```

Each intermediate-shape task requests 90 seconds. The prior two-shape tasks
took 47–50 seconds; one extra compilation makes about 60 seconds plausible,
keeping the request near 1.5 times estimated runtime. Each task saves a
partial `result.json` after every shape, generated Inductor wrappers,
combined-kernel PTX/IR, correctness errors, and paired CUDA graph replay
times. The summary script requires exactly three complete cases per task and
exact Q/K/V equality with the tested input. Do not launch the scripts locally.

The burst job requests five minutes. It runs 128 deterministic random-length
requests at 16 requests/s, with up to 32 concurrent requests, and saves
`serve.json`, server/client logs, `iterations.csv`, `shape_histogram.csv`, and
`shape_summary.json` under `burst_shapes_<job-id>/`. This is a synthetic
higher-load counterpoint to the earlier 1 request/s trace, not a production
trace. Check actual `client.time` and job wall time before allocating follow-up
jobs; the five-minute limit assumes roughly 3–4 minutes for startup, compile
and replay, and should be revised if the measurement differs.

## Decision

If the one-launch oracle loses by 8–32 scheduled tokens, the win is likely
limited to lightly occupied decode. Run only a short full-model pair if that
latency regime is a real target. If the win extends into more frequent burst
shapes, the shape-dependent selector has a stronger workload basis. In either
case, isolated region timing does not establish a model-level gain, and the
existing model comparison should follow only for shapes with a repeatable
region win. A positive serving trace frequency is opportunity, not speedup.
