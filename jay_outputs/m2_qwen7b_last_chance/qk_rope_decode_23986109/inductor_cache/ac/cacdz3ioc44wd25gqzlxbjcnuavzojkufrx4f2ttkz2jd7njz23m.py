
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 35840}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = (x0).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x0).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 64, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (128*x1 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.where(tmp6, tmp9, 0)
    tmp11 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tl.device_assert(((0 <= tl.broadcast_to(tmp14, [XBLOCK])) & (tl.broadcast_to(tmp14, [XBLOCK]) < 32768)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp14, [XBLOCK]) < 32768")
    tmp16 = tl.load(in_ptr2 + (128*tmp14 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp7 * tmp16
    tmp18 = tl.load(in_ptr0 + (64 + 128*x1 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr2 + (64 + 128*tmp14 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp17 - tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp6, tmp21, tmp22)
    tmp24 = tmp0 >= tmp5
    tmp25 = tl.full([1], 128, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr0 + (64 + 128*x1 + ((-64) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (0))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp30 = tl.where(tmp24, tmp29, 0)
    tmp31 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp30 < 0
    tmp34 = tl.where(tmp33, tmp32, tmp30)
    tl.device_assert(((0 <= tl.broadcast_to(tmp34, [XBLOCK])) & (tl.broadcast_to(tmp34, [XBLOCK]) < 32768)) | ~(tmp24 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp34, [XBLOCK]) < 32768")
    tmp36 = tl.load(in_ptr2 + (128*tmp34 + ((-64) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tmp27 * tmp36
    tmp38 = tl.load(in_ptr0 + (128*x1 + ((-64) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr2 + (64 + 128*tmp34 + ((-64) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tmp38 * tmp39
    tmp41 = tmp37 + tmp40
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp24, tmp41, tmp42)
    tmp44 = tl.where(tmp6, tmp23, tmp43)
    tl.store(out_ptr0 + (x2), tmp44, xmask)
