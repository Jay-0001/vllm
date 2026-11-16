# M2: shape-aware Q/K RoPE fusion — conclusion so far

## Research gap, hypothesis, and surveyed work

**Gap.** A legal, nonlocal inference-graph region may be missing from the
physical alternatives that current TorchInductor emits. Q and K are sibling
consumers of one QKV projection; combining their RoPE transforms can save a
launch, but doing so may hurt larger shapes. The unresolved compiler question
is when to generate and select such a region after actual lowering and
workload frequency are counted. We have not determined whether Inductor
never generated the alternative or generated and rejected it.

**Hypothesis.** A shape-aware rule that combines Q/K RoPE at small and medium
scheduled-token counts, and retains separate kernels for larger counts, can
beat the deployed default on a relevant inference trace. The broad M2 thesis
would require that this rule transfers beyond one Qwen7B instance and improves
full-model latency or another relevant service metric after compilation and
dispatch costs.

**Research connection.** The surveyed H2 work on fusion and partitioning
motivates bounded region selection. SpaceFusion++ treats over-fusion as a real
cost and leaves graph-level region selection outside its scheduler; XNNC's
adjacent greedy merging motivates a possible nonlocal candidate gap. Inductor
already has greedy fusion scoring and candidate benchmarking, so these are
motivations, not proof of novelty. Constable and Tensor Seeks Layout warn that
downstream costs can defeat a locally attractive graph choice. H4 work on
connected-operator lowering and H5 work on search budget imply that a
candidate must be judged by generated code, resources, compile cost, and
reuse. Wider search systems such as Mirage, Neptune, Trinity and Nautilus
already occupy a generic graph-superoptimization claim. The relevant sources
and chronology are in the project's gap register and H2 research map.

## Experimental progression

1. The initial Qwen-0.5B manual split arms lowered to the same physical
   kernels as stock. ConvNeXt's layout toggle likewise produced identical
   physical code. The Mamba gain came from an existing compile-only scan
   algorithm, not weak fusion. The diffusion run failed at dependency import.
   None supported M2.
2. Inspection of an older Qwen2.5-7B AOT cache found the concrete fan-out:
   a QKV GEMM followed by separate Q and K RoPE launches. The current-commit
   full-model compile dump confirmed the same physical boundary.
3. A one-launch Triton oracle matched Inductor's Q, K and V exactly on the
   seeded BF16 inputs tested. Isolated A100 CUDA graph replay showed about
   51–55% region savings at 1–32 tokens, 38% at 64, 25% at 128, a near tie at
   256, and losses of about 30% and 22% at 512 and 2048. These are region
   comparisons, with two timing samples per shape in one job.
4. A separate synthetic 16-request/s serving run recorded 463 pure-decode
   scheduler iterations. Of these, 392 (84.7%) scheduled 8–29 tokens, inside
   the observed winning range. Its 106 mixed iterations scheduled 139–994
   total tokens, so an unconditional fused route would be unsafe. The burst
   completed 128 requests, with mean TPOT 16.51 ms. It was a stock-only
   shape-frequency run, not an A/B service measurement.

## What the evidence establishes

The current Qwen7B/Inductor path has a demonstrable, shape-dependent
**physical-region opportunity**. The alternative is semantically matched for
the tested inputs, visibly changes two launches to one, and wins at shapes
frequent in the synthetic burst. It is not merely a renamed graph partition.

It does **not** establish worse downstream model performance, a compiler
cost-model failure, or a general thesis contribution. The existing vLLM
rotary CUDA/FlashInfer routes, where semantically and operationally comparable,
remain important controls. An isolated saving of roughly 2–3 microseconds per
layer may become small beside a 16.51 ms TPOT, and custom-op dispatch or
compilation may erase it. The current model-level script selects only through
four tokens and therefore does not test the burst's dominant 8–29 range.

## How this could become a thesis

A defensible title is **Shape-aware nonlocal fusion selection for inference
graphs**. The contribution would have three parts: (1) identify fan-out
regions that current lowering leaves as separate launches; (2) cheaply
predict or measure a physical crossover using scheduled shape, consumer
layout, traffic and launch/resource cost; (3) dispatch a correct fused or
separate route only when the choice repays compilation and integration costs.
The target is lower downstream regret at a bounded search budget, not a
bespoke Qwen kernel or a new generic compiler framework.

To support that claim, first compare the existing vLLM fused rotary routes
and inspect whether Inductor considered this candidate. Then integrate a
selector covering the observed 8–32-token decode regime while keeping mixed
and prefill on stock, verify generated execution, and run paired identical
request traces with output, TPOT/TTFT/goodput, memory and compile-cost checks.
Repeat on another RoPE configuration or a second fan-out motif with held-out
shapes. If service improvement is absent, below the predeclared practical
threshold, or already achieved by an existing route, close M2 as a compiler
thesis despite the valid local win and move to the next ranked hypothesis.
