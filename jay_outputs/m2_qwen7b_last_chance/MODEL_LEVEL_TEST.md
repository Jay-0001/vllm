# M2 model-level gate: Qwen7B Q/K RoPE

## Question

The one-launch Q/K oracle saved about 1.6–2.0 microseconds per isolated
decode region call at one and four tokens, while losing at prefill sizes.
Does selecting it only for small decode shapes improve the actual Qwen2.5-7B
request path after compilation, scheduling, attention, and sampling?

The intervention is opt-in through `M2_QK_MODEL_ORACLE=1`. A startup hook
registers an experimental PyTorch custom op and patches the native RoPE path
only for BF16 Qwen7B's 28 Q heads, 4 K heads, head size 128, Neox layout,
and at most four scheduled tokens. It produces contiguous Q/K outputs and
leaves the QKV backing store, including V, intact. The stock arm loads the
same hook directory but leaves the model unchanged. Both arms compile size 4
and disable CUDA graphs. A generated-code check must show the experimental
op in the oracle graph. Any unexpected graph breaks or extra materialization
must be evaluated before interpreting timing.

## Delta commands

```bash
cd /u/jsaravananindira/vllm
git pull --ff-only origin thesis/m2-fusion-boundary-screen

# Compile and execute one short oracle case first. Inspect its log and code.
sbatch --array=1 --export=ALL,M2_SMOKE=1 jay_scripts/m2_qwen7b_model_oracle.sbatch

# After the smoke passes, run serial ABBA jobs on A100 nodes.
sbatch jay_scripts/m2_qwen7b_model_oracle.sbatch

# Substitute the returned array job ID after all four tasks complete.
.venv/bin/python benchmarks/kernels/m2_fusion_boundary/compare_model_latency.py \
  --root jay_outputs/m2_qwen7b_last_chance --job-id ARRAY_JOB_ID
```

The smoke runs two short decode iterations and two prefill controls after
warmup. The full array uses order stock/oracle/oracle/stock, 12 decode batch
iterations with 512 input and 128 output tokens, and 12 prefill controls with
512 input and one output token. All arms use four deterministic token prompts,
greedy decoding, identical compilation config and isolated caches. Each job
saves its generated code, environment, process wall time, per-iteration batch
latencies, and generated-token hash in `model_<arm>_<job>_<task>/`. The
comparison script fails if prompt or output token hashes differ.

Each task requests one A100 for 3 minutes 30 seconds. The previous full-model
compile dump took 139 seconds; generation work makes about 140–160 seconds
plausible, putting the request at roughly 1.3–1.5 times that estimate. The
smoke should take less. If runtime is materially different, revise the next
allocation from its measured `process.time` instead of increasing limits
without evidence.

## Decision rule

Advance only if the oracle op is present in generated model code, token
outputs match, decode improvement repeats across the two ABBA pairs, and the
prefill control does not suffer a material regression. Report full batch
latency separately from the isolated GPU-region gain; do not multiply the
microsecond savings by layer count as a measured service result. Even a
model-level win is one RoPE instance, so a broader compiler thesis would need
a transferable criterion for when nonlocal fan-out fusion wins versus loses.
