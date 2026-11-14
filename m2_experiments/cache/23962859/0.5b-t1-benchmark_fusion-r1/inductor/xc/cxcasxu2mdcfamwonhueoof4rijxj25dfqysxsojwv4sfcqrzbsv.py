
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'r0_': 16128}, 'kernel_num_gb': 1.0752e-05, 'kernel_flop': 0, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp10 = tl.full([1, 1], 896.0, tl.float32)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tl.full([1, 1], 1e-06, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp4, r0_mask)
    tl.store(out_ptr2 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp18, r0_mask)


def get_args():
    arg_0 = rand_strided((1, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((1, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((896,), (1,), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((1, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 1, 896,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(lambda: call(args), device='cuda', rep=40)
    num_gb = 1.0752e-05
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
