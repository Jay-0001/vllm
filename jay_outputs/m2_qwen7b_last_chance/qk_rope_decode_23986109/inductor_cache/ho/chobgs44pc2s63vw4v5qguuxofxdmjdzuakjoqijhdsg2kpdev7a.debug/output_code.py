# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_qwen7b_last_chance/qk_rope_decode_23986109/inductor_cache/mk/cmkxfic3kouhcp5zikjiekc4mvv6f64ag7kzb32jmlzq3dqkijjp.py
# Topologically Sorted Source Nodes: [split, cos_sin, chunk, query, chunk_1, unsqueeze, mul, unsqueeze_1, mul_1, o1, mul_2, mul_3, o2, output], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
# Source node to ATen node mapping:
#   chunk => split
#   chunk_1 => split_1
#   cos_sin => index
#   mul => mul_25
#   mul_1 => mul_28
#   mul_2 => mul_33
#   mul_3 => mul_36
#   o1 => sub_15
#   o2 => add_64
#   output => cat
#   query => view
#   split => split_with_sizes
#   unsqueeze => unsqueeze
#   unsqueeze_1 => unsqueeze_1
# Graph fragment:
#   %arg1_1 : Tensor "bf16[s30, 4608][4608, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg3_1 : Tensor "i64[s30][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "bf16[32768, 128][128, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%arg1_1, [3584, 512, 512], -1), kwargs = {})
#   %index : Tensor "bf16[s30, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg4_1, [%arg3_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view : Tensor "bf16[s30, 28, 128][4608, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [%arg0_1, -1, 128]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view, 64, -1), kwargs = {})
#   %unsqueeze : Tensor "bf16[s30, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_25 : Tensor "bf16[s30, 28, 64][1792, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_5, %unsqueeze), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[s30, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_28 : Tensor "bf16[s30, 28, 64][1792, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_6, %unsqueeze_1), kwargs = {})
#   %sub_15 : Tensor "bf16[s30, 28, 64][1792, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_25, %mul_28), kwargs = {})
#   %mul_33 : Tensor "bf16[s30, 28, 64][1792, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_6, %unsqueeze), kwargs = {})
#   %mul_36 : Tensor "bf16[s30, 28, 64][1792, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_5, %unsqueeze_1), kwargs = {})
#   %add_64 : Tensor "bf16[s30, 28, 64][1792, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %mul_36), kwargs = {})
#   %cat : Tensor "bf16[s30, 28, 128][3584, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_15, %add_64], -1), kwargs = {})
#   return %cat
triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_0 = async_compile.triton('triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 143392}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 28)
    x2 = xindex // 3584
    x4 = xindex
    tmp0 = (x0).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x0).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 64, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (128*x1 + 4608*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp8 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp8)
    tl.device_assert(((0 <= tl.broadcast_to(tmp12, [XBLOCK])) & (tl.broadcast_to(tmp12, [XBLOCK]) < 32768)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp12, [XBLOCK]) < 32768")
    tmp14 = tl.load(in_ptr2 + (128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr0 + (64 + 128*x1 + 4608*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (64 + 128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 - tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tmp0 >= tmp5
    tmp23 = tl.full([1], 128, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (64 + 128*x1 + 4608*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp26 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp26)
    tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 32768)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 32768")
    tmp32 = tl.load(in_ptr2 + (128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.load(in_ptr0 + (128*x1 + 4608*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr2 + (64 + 128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp22, tmp37, tmp38)
    tmp40 = tl.where(tmp6, tmp21, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_qwen7b_last_chance/qk_rope_decode_23986109/inductor_cache/xl/cxlguuyealuk64xgbui3t545wzyj5tlnnqh2hhamv55c43hvhks3.py
# Topologically Sorted Source Nodes: [split, cos_sin, chunk, key, chunk_2, unsqueeze_2, mul_4, unsqueeze_3, mul_5, o1_1, mul_6, mul_7, o2_1, output_1], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
# Source node to ATen node mapping:
#   chunk => split
#   chunk_2 => split_2
#   cos_sin => index
#   key => view_2
#   mul_4 => mul_64
#   mul_5 => mul_67
#   mul_6 => mul_72
#   mul_7 => mul_75
#   o1_1 => sub_32
#   o2_1 => add_126
#   output_1 => cat_1
#   split => split_with_sizes
#   unsqueeze_2 => unsqueeze_2
#   unsqueeze_3 => unsqueeze_3
# Graph fragment:
#   %arg1_1 : Tensor "bf16[s30, 4608][4608, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg3_1 : Tensor "i64[s30][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "bf16[32768, 128][128, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%arg1_1, [3584, 512, 512], -1), kwargs = {})
#   %index : Tensor "bf16[s30, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg4_1, [%arg3_1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%index, 64, -1), kwargs = {})
#   %view_2 : Tensor "bf16[s30, 4, 128][4608, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [%arg0_1, -1, 128]), kwargs = {})
#   %split_2 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_2, 64, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[s30, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, -2), kwargs = {})
#   %mul_64 : Tensor "bf16[s30, 4, 64][256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[s30, 1, 64][128, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_4, -2), kwargs = {})
#   %mul_67 : Tensor "bf16[s30, 4, 64][256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_3), kwargs = {})
#   %sub_32 : Tensor "bf16[s30, 4, 64][256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_64, %mul_67), kwargs = {})
#   %mul_72 : Tensor "bf16[s30, 4, 64][256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %unsqueeze_2), kwargs = {})
#   %mul_75 : Tensor "bf16[s30, 4, 64][256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_7, %unsqueeze_3), kwargs = {})
#   %add_126 : Tensor "bf16[s30, 4, 64][256, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_72, %mul_75), kwargs = {})
#   %cat_1 : Tensor "bf16[s30, 4, 128][512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_32, %add_126], -1), kwargs = {})
#   return %cat_1
triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1 = async_compile.triton('triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 20512}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 4)
    x2 = xindex // 512
    x4 = xindex
    tmp0 = (x0).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x0).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 64, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (3584 + 128*x1 + 4608*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp8 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp8)
    tl.device_assert(((0 <= tl.broadcast_to(tmp12, [XBLOCK])) & (tl.broadcast_to(tmp12, [XBLOCK]) < 32768)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp12, [XBLOCK]) < 32768")
    tmp14 = tl.load(in_ptr2 + (128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr0 + (3648 + 128*x1 + 4608*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (64 + 128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 - tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tmp0 >= tmp5
    tmp23 = tl.full([1], 128, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr0 + (3648 + 128*x1 + 4608*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp26 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp26)
    tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 32768)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 32768")
    tmp32 = tl.load(in_ptr2 + (128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.load(in_ptr0 + (3584 + 128*x1 + 4608*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr2 + (64 + 128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp22, tmp37, tmp38)
    tmp40 = tl.where(tmp6, tmp21, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
        args.clear()
        s30 = arg0_1
        s80 = s30
        assert_size_stride_grouped((arg1_1, arg3_1, arg4_1), ((s30, 4608), (s30, ), (32768, 128)), ((4608, 1), (1, ), (128, 1)), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            arg4_1 = copy_if_misaligned(arg4_1)
            buf0 = empty_strided_cuda((s30, 28, 128), (3584, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, cos_sin, chunk, query, chunk_1, unsqueeze, mul, unsqueeze_1, mul_1, o1, mul_2, mul_3, o2, output], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            # [Provenance debug handles] triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_0:1
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_0_xnumel = 3584*s30
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_0.run(arg1_1, arg3_1, arg4_1, buf0, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_0_xnumel, stream=raw_stream0)
            buf1 = empty_strided_cuda((s30, 4, 128), (512, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [split, cos_sin, chunk, key, chunk_2, unsqueeze_2, mul_4, unsqueeze_3, mul_5, o1_1, mul_6, mul_7, o2_1, output_1], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            # [Provenance debug handles] triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1:2
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1_xnumel = 512*s30
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1.run(arg1_1, arg3_1, arg4_1, buf1, triton_poi_fused_add_cat_index_select_mul_split_split_with_sizes_sub_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg3_1
            del arg4_1
        return (reinterpret_tensor(buf0, (s30, 3584), (3584, 1), 0), reinterpret_tensor(buf1, (s30, 512), (512, 1), 0), reinterpret_tensor(arg1_1, (s30, 512), (4608, 1), 4096), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 4
    arg1_1 = rand_strided((4, 4608), (4608, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = 4
    arg3_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg4_1 = rand_strided((32768, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
