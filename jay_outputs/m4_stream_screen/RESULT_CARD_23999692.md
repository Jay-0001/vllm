# M4 streamed tile screen — job 23999692

## Paired setup

Stock and streamed arms ran as fresh processes in one 20-minute allocation
with one visible A100-SXM4-40GB. Both used 2048 tokens, Qwen2.5-7B MLP
dimensions, BF16, PyTorch 2.14.0+cu132 and Triton 3.8.0. Each arm recorded
25 synchronized steady timing samples and a separate memory pass. The same
job saw both arms; this is one independent run, not a repeated study.

| Measure | Stock | Streamed |
| --- | ---: | ---: |
| Median GPU event latency | 4.189 ms | 7.443 ms (+77.7%) |
| Median synchronized wall latency | 4.217 ms | 7.472 ms (+77.2%) |
| Baseline allocated | 412.125 MiB | 412.125 MiB |
| Incremental allocated peak | 148 MiB | 70 MiB (78 MiB saved) |
| Incremental reserved peak | 74 MiB | 0 MiB |
| Sampled device memory | 1051→1125 MiB | 1051→1051 MiB |
| Relative L2 error against eager stock | 0.00420 | 0.00438 |
| First-call wall time | 10.41 s | 5.77 s |
| Dynamo unique graphs | 1 | 2 |

Both outputs were finite. The 20 ms device sampler yielded only four stock
and seven streamed points. The allocator's reserved peak supports the 74 MiB
physical difference in this run, but repeated runs and a relevant workload
are needed before treating that as deployable headroom. First-call times are
affected by compiler caches and are not a controlled compile-time comparison.

## Mechanism

The streamed arm compiles a per-tile computation and calls it from a Python
loop that accumulates each partial in place. The two generated wrappers under
`23999692/stream/debug` cover tile shapes 2048 and the final 512 channels;
they allocate tile-sized gate/up outputs and one `(2048, 3584)` partial. The
ten partial outputs no longer coexist, unlike in the earlier whole-graph
tiled arm. The substantial latency cost is consistent with repeated GEMMs
and per-tile calls; this screen does not isolate those components.

The 148 MiB stock incremental allocated peak differs from the earlier
screen's 162 MiB because this runner releases each output during its memory
pass. Compare paired arms within this job, not peaks across the two jobs.

## Decision

The streamed oracle **passes the provisional 64 MiB allocated-peak saving**
and 1% relative L2 error gates but **fails the 20% latency gate**. This
validates a memory/latency tradeoff for the isolated motif, not a useful
serving route or a thesis gap.

Before building a fused kernel, profile a realistic 7B inference workload
near its memory-capacity boundary. Determine whether roughly 74–78 MiB of
reclaimable physical headroom changes KV capacity, OOM frequency, or goodput
under a fixed latency target. If the headroom is irrelevant, close this
motif. If it matters, prototype one fused tile kernel and compare its
physical peak and latency against stock and existing library routes on
multiple shapes. Only then consider a graph-level selector or integration.
