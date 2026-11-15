# M4 full-stack decision summary

One run per case. Inspect server and client logs before concluding.
Device peaks are sampled and can miss short transients.

| Utilization | Trace | Arm | Completed | Failed | Req/s | p99 TTFT ms | p99 TPOT ms | Device peak MiB | KV tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.96 | long_prefill | stock_default | None | None | None | None | None | None | None |
| 0.96 | long_prefill | stock_eager | None | None | None | None | None | None | None |
| 0.96 | long_prefill | stream_eager | None | None | None | None | None | None | None |
| 0.96 | mixed | stock_default | None | None | None | None | None | None | None |
| 0.96 | mixed | stock_eager | None | None | None | None | None | None | None |
| 0.96 | mixed | stream_eager | None | None | None | None | None | None | None |
| 0.99 | long_prefill | stock_default | None | None | None | None | None | None | None |
| 0.99 | long_prefill | stock_eager | None | None | None | None | None | None | None |
| 0.99 | long_prefill | stream_eager | None | None | None | None | None | None | None |
| 0.99 | mixed | stock_default | None | None | None | None | None | None | None |
| 0.99 | mixed | stock_eager | None | None | None | None | None | None | None |
| 0.99 | mixed | stream_eager | None | None | None | None | None | None | None |

## Greedy output agreement

- 0.96, long_prefill: None
- 0.96, mixed: None
- 0.99, long_prefill: None
- 0.99, mixed: None

## Pre-registered decision gate

Continue M4 only if streamed execution produces a material
capacity outcome (production stock fails while stream completes
at the same setting, or at least 5% more KV tokens than eager
stock), preserves at least 99% greedy outputs,
and stays within 20% of production stock p99 TTFT and TPOT with
no lower completed-request throughput. Compare eager stock and
stream to isolate the tile change. A failed or missing case is
evidence to inspect, not a zero in a favorable average.
