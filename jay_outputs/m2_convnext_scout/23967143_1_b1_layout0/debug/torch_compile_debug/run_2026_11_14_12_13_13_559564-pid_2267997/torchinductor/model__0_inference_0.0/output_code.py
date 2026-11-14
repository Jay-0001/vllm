# AOT ID: ['0_inference']
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
from torch._inductor.runtime.runtime_utils import assert_tensor_metadata

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


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_convnext_scout/23967143_1_b1_layout0/cache/5o/c5o5r4pxxp34masvlq35ye5yicl4cwfnkdd5qs4buirinku77jpf.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.permute, aten.native_layer_norm]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => permute
#   input_3 => add, add_1, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %buf0 : Tensor "bf16[1, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = PlaceHolder[target=buf0]
#   %arg2_1 : Tensor "bf16[96][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %getitem_1 : Tensor "f32[1, 56, 56, 1][3136, 56, 1, 3136]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf2 : Tensor "f32[1, 56, 56, 1][3136, 56, 1, 3136]cuda:0" = PlaceHolder[target=buf2]
#   %arg4_1 : Tensor "bf16[96][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "bf16[96][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convolution : Tensor "bf16[1, 96, 56, 56][301056, 1, 5376, 96]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg3_1, %arg1_1, %arg2_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 96), kwargs = {})
#   %permute : Tensor "bf16[1, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convolution, [0, 2, 3, 1]), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[1, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_1), kwargs = {})
#   %add : Tensor "f32[1, 56, 56, 1][3136, 56, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[1, 56, 56, 1][3136, 56, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[1, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[1, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg4_1), kwargs = {})
#   %add_1 : Tensor "f32[1, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   return %getitem_1,%buf2,%convert_element_type_1
triton_per_fused_convolution_native_layer_norm_permute_0 = async_compile.triton('triton_per_fused_convolution_native_layer_norm_permute_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_convolution_native_layer_norm_permute_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 1806912}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_convolution_native_layer_norm_permute_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 3136
    r0_numel = 96
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 96*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = (tl.full([1, 1], 96, tl.int32)).to(tl.float32)
    tmp12 = (tmp10 / tmp11)
    tmp13 = tmp4 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = tmp3 - tmp12
    tmp20 = tl.full([1, 1], 96.0, tl.float32)
    tmp21 = (tmp18 / tmp20)
    tmp22 = tl.full([1, 1], 1e-06, tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 * tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 96*x0), tmp32, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_convnext_scout/23967143_1_b1_layout0/cache/57/c57eodlnrhqsak4fqefekxeispdv7l7ocly6hhjfuiz6pi7aaocr.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.view, aten.gelu]
# Source node to ATen node mapping:
#   input_4 => view_1
#   input_5 => add_2, convert_element_type_5, convert_element_type_6, erf, mul_2, mul_3, mul_4
# Graph fragment:
#   %addmm : Tensor "bf16[3136, 384][384, 1]cuda:0" = PlaceHolder[target=addmm]
#   %view_1 : Tensor "bf16[1, 56, 56, 384][1204224, 21504, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 56, 56, 384]), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[1, 56, 56, 384][1204224, 21504, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %mul_2 : Tensor "f32[1, 56, 56, 384][1204224, 21504, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, 0.5), kwargs = {})
#   %mul_3 : Tensor "f32[1, 56, 56, 384][1204224, 21504, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[1, 56, 56, 384][1204224, 21504, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_3,), kwargs = {})
#   %add_2 : Tensor "f32[1, 56, 56, 384][1204224, 21504, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_4 : Tensor "f32[1, 56, 56, 384][1204224, 21504, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %add_2), kwargs = {})
#   %convert_element_type_6 : Tensor "bf16[1, 56, 56, 384][1204224, 21504, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   return %convert_element_type_6
triton_poi_fused_gelu_view_1 = async_compile.triton('triton_poi_fused_gelu_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_gelu_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 7225344}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_view_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.5, tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp5 = tmp1 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_convnext_scout/23967143_1_b1_layout0/cache/cn/ccnmtev4vubqpwkqgpbjyhspc4toutwnwwkkyoyffo2i6qum7fac.py
# Topologically Sorted Source Nodes: [input_6, input_7, result, result_1], Original ATen: [aten.view, aten.permute, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_6 => view_3
#   input_7 => permute_3
#   result => mul_5
#   result_1 => add_3
# Graph fragment:
#   %arg0_1 : Tensor "bf16[96, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %addmm_1 : Tensor "bf16[3136, 96][96, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %arg3_1 : Tensor "bf16[1, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = PlaceHolder[target=arg3_1]
#   %view_3 : Tensor "bf16[1, 56, 56, 96][301056, 5376, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [1, 56, 56, 96]), kwargs = {})
#   %permute_3 : Tensor "bf16[1, 96, 56, 56][301056, 1, 5376, 96]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 3, 1, 2]), kwargs = {})
#   %mul_5 : Tensor "bf16[1, 96, 56, 56][301056, 1, 5376, 96]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %permute_3), kwargs = {})
#   %add_3 : Tensor "bf16[1, 96, 56, 56][301056, 1, 5376, 96]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg3_1), kwargs = {})
#   return %add_3
triton_poi_fused_add_mul_permute_view_2 = async_compile.triton('triton_poi_fused_add_mul_permute_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_mul_permute_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 2408640}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_permute_view_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
        args.clear()
        assert_size_stride_grouped((arg3_1, arg1_1), ((1, 96, 56, 56), (96, 1, 7, 7)), ((301056, 1, 5376, 96), (49, 1, 7, 1)), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg3_1 = copy_if_misaligned(arg3_1)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:1
            buf0 = extern_kernels.convolution(arg3_1, arg1_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
            assert_tensor_metadata(buf0, (1, 96, 56, 56), (301056, 1, 5376, 96), torch.bfloat16, 'torch.ops.aten.convolution.default')
            del arg1_1
            assert_size_stride_grouped((arg2_1, arg4_1, arg5_1), ((96, ), (96, ), (96, )), ((1, ), (1, ), (1, )), 'input')
            buf4 = reinterpret_tensor(buf0, (1, 56, 56, 96), (301056, 5376, 96, 1), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.permute, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_layer_norm_permute_0:2
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_layer_norm_permute_0.run(buf4, arg2_1, arg4_1, arg5_1, 3136, 96, stream=raw_stream0)
            del arg2_1
            del arg4_1
            del arg5_1
            assert_size_stride_grouped((arg7_1, arg6_1), ((384, ), (384, 96)), ((1, ), (96, 1)), 'input')
            buf5 = empty_strided_cuda((3136, 384), (384, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4], Original ATen: [aten.convolution, aten.permute, aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:3
            extern_kernels.addmm(arg7_1, reinterpret_tensor(buf4, (3136, 96), (96, 1), 0), reinterpret_tensor(arg6_1, (96, 384), (1, 96), 0), alpha=1, beta=1, out=buf5)
            del arg6_1
            del arg7_1
            del buf4
            buf6 = reinterpret_tensor(buf5, (1, 56, 56, 384), (1204224, 21504, 384, 1), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.view, aten.gelu]
            # [Provenance debug handles] triton_poi_fused_gelu_view_1:4
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_gelu_view_1.run(buf6, 1204224, stream=raw_stream0)
            assert_size_stride_grouped((arg9_1, arg8_1), ((96, ), (96, 384)), ((1, ), (384, 1)), 'input')
            buf7 = empty_strided_cuda((3136, 96), (96, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:5
            extern_kernels.addmm(arg9_1, reinterpret_tensor(buf6, (3136, 384), (384, 1), 0), reinterpret_tensor(arg8_1, (384, 96), (1, 384), 0), alpha=1, beta=1, out=buf7)
            del arg8_1
            del arg9_1
            del buf6
            assert_size_stride(arg0_1, (96, 1, 1), (1, 1, 1), 'input')
            buf8 = reinterpret_tensor(buf7, (1, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [input_6, input_7, result, result_1], Original ATen: [aten.view, aten.permute, aten.mul, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_mul_permute_view_2:6
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_permute_view_2.run(buf8, arg0_1, arg3_1, 301056, stream=raw_stream0)
            del arg0_1
            del arg3_1
        return (buf8, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((96, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((96, 1, 7, 7), (49, 1, 7, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((384, 96), (96, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
