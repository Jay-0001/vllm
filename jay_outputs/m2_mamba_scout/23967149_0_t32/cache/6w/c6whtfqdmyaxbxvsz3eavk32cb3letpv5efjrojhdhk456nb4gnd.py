
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_mul_mul_add0(arg0_0, arg0_1, arg1_0, arg1_1):
    tmp0 = arg0_0 * arg1_0
    tmp1 = arg1_0 * arg0_1
    tmp2 = tmp1 + arg1_1
    return tmp0, tmp2

@triton_heuristics.persistent_reduction(
    size_hints={'y': 2048, 'x': 16, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_convolution_exp_mul_neg_permute_silu_slice_softplus_split_split_with_sizes_squeeze_transpose_unsqueeze_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 202752, 'x': 6391808, 'r0_': 196608}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__unsafe_view_add_clone_convolution_exp_mul_neg_permute_silu_slice_softplus_split_split_with_sizes_squeeze_transpose_unsqueeze_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 16
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, None, :]
    roffset = r0_offset
    rindex = r0_index
    x1 = xindex
    y0 = yindex
    r0_2 = r0_index
    tmp0 = tl.load(in_ptr0 + (x1 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + 1536*r0_2), ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (48 + x1 + 80*r0_2), xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr4 + (r0_2 + 32*y0), ymask, eviction_policy='evict_last', other=0.0)
    tmp1 = libdevice.exp(tmp0)
    tmp2 = -tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full([1, 1, 1], 20.0, tl.float32)
    tmp7 = tmp5 > tmp6
    tmp8 = libdevice.exp(tmp5)
    tmp9 = libdevice.log1p(tmp8)
    tmp10 = tl.where(tmp7, tmp5, tmp9)
    tmp11 = tmp2 * tmp10
    tmp12 = libdevice.exp(tmp11)
    tmp14 = tmp10 * tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = tmp12.to(tl.float32)
    tmp18 = tl.broadcast_to(tmp17, [YBLOCK, XBLOCK, R0_BLOCK])
    tmp19 = tmp16.to(tl.float32)
    tmp20 = tl.broadcast_to(tmp19, [YBLOCK, XBLOCK, R0_BLOCK])
    tmp21, tmp22, = tl.associative_scan((tmp18, tmp20,), 2, _triton_helper_fn_mul_mul_add0)
    tl.store(out_ptr0 + (x1 + 16*y0 + 24576*r0_2), tmp22, xmask & ymask)
