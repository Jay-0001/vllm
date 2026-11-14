# M2 A100 fusion-boundary screen

This is a **rejection experiment**, not an optimized vLLM model path. It tests
whether Inductor's post-lowering choice for a Qwen-style decoder motif is
repeatedly worse than a small legal partition oracle. The motif follows
`Qwen2DecoderLayer.forward`'s post-attention residual/RMSNorm and
`Qwen2MLP.forward`'s gate-up projection, SiLU-and-multiply, and down projection.
The script expresses these operations with ordinary PyTorch so Inductor can see
the whole graph. vLLM may instead use its custom operations or different
linear backends, so any win here requires an actual vLLM graph check before a
serving claim.

The dimensions come from the [0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)
and [7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)
model configurations. The existing 0.5B A100 `spec64` artifact under
`jay_outputs/a100_shape64_sources` contains Inductor-generated
`fused_add_rms_norm` and adjacent GEMM kernels. It establishes that this kind
of boundary occurs in the local stack; it does not establish a performance
miss. That artifact used FP16, while this screen uses BF16 to match the
existing 7B bench configuration. Compare only paired arms within this screen.

---

## Arms and measured outputs

- `stock`: one `torch.compile(fullgraph=True)` call.
- `benchmark_fusion`: the same call with the installed PyTorch's
  `TORCHINDUCTOR_BENCHMARK_FUSION=1` control. Confirm in generated logs that the
  installed build recognizes and exercises it.
- `max_autotune`: the stock graph with PyTorch's built-in `max-autotune` mode.
  Run it on promising cases before claiming a gap beyond existing tuning.
- `split_norm`: compiled residual/RMSNorm followed by a compiled MLP.
- `split_activation`: compiled residual/RMSNorm/gate-up followed by a compiled
  activation/down projection.

All arms use the same deterministic BF16 tensors, reference expression, shape,
and correctness check. Each fresh process reports first-call wall time, GPU
event latency samples, CPU wall median, allocated/reserved peaks, and maximum
absolute output error. `TORCH_COMPILE_DEBUG=1` saves generated code and fusion
IR in the per-case compiler cache; `TORCH_LOGS` saves fusion and recompile
decisions. The optional PyTorch profiler trace records kernel launches.
CUDA Graphs are absent from this screen so launch and partition costs remain
visible. Interpret the result as a graph benchmark, not request latency.

---

## Delta launch

Copy these files to the same paths in the Delta clone, or commit and push this
branch locally and then fetch it on Delta. From `/u/jsaravananindira/vllm` on
the **Delta login node**, with the files present:

```bash
git branch --show-current
test -x .venv/bin/python
MODEL_SIZE=0.5b TOKENS='1 4 64' REPEATS=1 PROFILE=1 \
  sbatch --export=ALL benchmarks/kernels/m2_fusion_boundary/delta_a100.sbatch
```

This requests one A100. The script rejects a different GPU and rejects more
than one visible CUDA device. The Slurm log is `jay_outputs/m2-screen-<job>.out`.
Results, detailed logs, compiler caches, environment details, and profiler
traces are under `m2_experiments/{results,logs,cache,metadata}/<job>/`.

If the first pass has a credible gap, repeat in alternating order and include
held-out shapes, then confirm with the 7B dimensions. If the 0.5B pass is
negative, still check 7B at a few sizes before rejecting this **7B motif**:

```bash
MODEL_SIZE=0.5b TOKENS='2 8 128 256' REPEATS=3 PROFILE=0 \
  sbatch --export=ALL benchmarks/kernels/m2_fusion_boundary/delta_a100.sbatch
MODEL_SIZE=7b TOKENS='1 4 64' REPEATS=1 PROFILE=1 \
  sbatch --export=ALL benchmarks/kernels/m2_fusion_boundary/delta_a100.sbatch
MODEL_SIZE=7b TOKENS='4 64' REPEATS=1 PROFILE=0 \
  ARMS='stock split_norm split_activation benchmark_fusion max_autotune' \
  sbatch --export=ALL benchmarks/kernels/m2_fusion_boundary/delta_a100.sbatch
```

Those are follow-up commands, not part of the initial allocation. Choose a
smaller shape set if the first job shows high compile cost; extend the Slurm
time limit only for a surviving case. Repeat any promising 7B case three
times in alternating order with held-out sizes. If
both model sizes show identical generated code, no stable advantage beyond
run variation, or a built-in tuning arm matches the oracle, record a
negative M2 result for this motif. A positive screen requires a consistent
advantage across held-out shapes, matching outputs, a generated-code/profile
mechanism (conversion, materialization, occupancy, or spills), and a cheap
feature that predicts the winning boundary. Confirm the mechanism on the
actual vLLM compiled graph and then check request-level impact.

---

## Dependencies and limits

The existing Delta `.venv` needs CUDA-enabled PyTorch with `torch.compile`,
Triton, and vLLM only for the environment version record. No extra library is
needed for the first screen. `torch.profiler` normally uses CUDA/CUPTI from
the installed stack. Nsight Compute or Systems is useful only after a gap
survives, to establish register spills, occupancy, memory traffic, and launch
behavior. If `benchmark_fusion` is unavailable in the installed PyTorch,
record that and compare the remaining arms; do not silently replace it with
a different tuning mode. Pin the exact PyTorch/vLLM commits and A100 hardware
before interpreting a result.
