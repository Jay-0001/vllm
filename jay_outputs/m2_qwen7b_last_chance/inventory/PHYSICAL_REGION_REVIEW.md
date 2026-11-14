# Qwen2.5-7B cache: physical-region review

## Provenance

The source cache is `qwen7b-512x4-23650318`, built on commit
`8530a778f78dc7cd9a31e108ebc035c4b27ceb3c` with an A100-SXM4-40GB.
The inventory was copied on commit `66072a4657ffca59244e3aeffa7fd63e83443951`.
The copied `dynamic` and `spec2048` trees contain generated Inductor wrappers
and Triton sources, not an execution trace from the current branch.

## Matched decoder motif

The main dynamic wrapper is `chfyxingvu4rsdcq3rbicwuosex77xepc7i3ljvfc4voghokjj77.py`.
Its physical sequence (lines 1058–1119) is:

1. External output GEMM → residual/RMSNorm Triton reduction.
2. External gate/up GEMM → Triton SiLU/gate → external down GEMM.
3. Residual/RMSNorm Triton reduction → external QKV GEMM.
4. Two separate Triton RoPE kernels for Q and K.

The matched 2,048-token static wrapper is
`ccscvhpxhqitqugvj3vyfnfpzynt46dlm2pzjzt2nimtjoxczfod.py`.
Its sequence (lines 1808–1871) has the same nine launches and the same
intermediate boundaries. Output, gate/up and QKV GEMMs use Triton templates;
down GEMM remains external. Names such as
`triton_tem_fused_fused_add_rms_norm_mm_t_2` do **not** mean the norm and
GEMM share a launch: the norm writes `buf2` on line 1820 and the GEMM reads
it on line 1827. QKV likewise reads the separate norm output `buf8`.

The spec cache also includes dynamic-style wrappers. More cache files and
different wrapper names alone are not evidence of a changed physical region.

## Candidate assessment

| Site | Observed physical change | M2 fusion candidate? |
| --- | --- | --- |
| Output, gate/up, QKV GEMMs at static 2,048 | External GEMM becomes Triton GEMM; buffer and launch boundaries are unchanged. | Different implementation route, not a different fusion partition. Could isolate as a GEMM autotuning/control experiment, but it does not establish M2 regret. |
| Residual/RMSNorm → GEMM | Separate norm launch and materialized `buf2`/`buf8` in both arms. At 2,048 tokens each 2,048×3,584 BF16 norm buffer is about 14 MiB. | Plausible materialization cost, but no generated fused alternative in these artifacts. |
| Gate/up → SiLU/gate → down | Separate GEMM, activation and GEMM in both arms. The 2,048×37,888 BF16 gate/up buffer is about 148 MiB; the 2,048×18,944 activation buffer is about 74 MiB. | Large traffic, but no legal alternative or measured win is present. A GEMM epilogue experiment would need to handle the gate/up pairing correctly. |
| QKV → Q/K RoPE | Both arms materialize QKV and launch separate Q and K RoPE kernels. The two kernels both read `buf9` (dynamic lines 1114, 1119; static lines 1867, 1871). | The clearest untested fan-out candidate: one Q/K transform launch might remove one launch. The Q and K slices may still require the same bytes read, and the cache contains no alternative, so this remains a hypothesis. |

The two RoPE kernels produce different sized Q and K outputs, and V is returned
as a view of QKV. Thus an oracle must preserve output layout, aliasing,
semantics, and the downstream attention interface. It should first compare
one fused Q/K kernel against the current two at matched prefill and decode
shapes. Kernel timing and generated code are required before any serving test.

## Decision

These artifacts show **no existing alternate fusion partition** for the
matched 7B decoder motif. The static path does show a different GEMM lowering,
which is useful as a control but does not satisfy the M2 hypothesis. The Q/K
RoPE fan-out is the only concrete follow-up site visible here; it is not yet
a demonstrated Inductor weakness. Reproduce its baseline on the current
compiler commit before testing an oracle. Do not infer a full-request win
from these wrappers alone.
