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


# kernel path: /u/jsaravananindira/vllm-m4/jay_outputs/m4_peak_screen/23969387_5_2048_tiled/cache/5d/c5dlpf2qwinidn4hwf24r2kp4ut4oa7q77elfixdqtigtnr6bvep.py
# Topologically Sorted Source Nodes: [silu, mul], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   mul => mul
#   silu => add, convert_element_type_4, convert_element_type_5, div, exp, neg
# Graph fragment:
#   %mm : Tensor "bf16[2048, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm]
#   %mm_1 : Tensor "bf16[2048, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %convert_element_type_4 : Tensor "f32[2048, 2048][2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %neg : Tensor "f32[2048, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_4,), kwargs = {})
#   %exp : Tensor "f32[2048, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add : Tensor "f32[2048, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[2048, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_4, %add), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[2048, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %mul : Tensor "bf16[2048, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, %mm_1), kwargs = {})
#   return %mul
triton_poi_fused_mul_silu_0 = async_compile.triton('triton_poi_fused_mul_silu_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_silu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 33554432}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(in_out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm-m4/jay_outputs/m4_peak_screen/23969387_5_2048_tiled/cache/jk/cjki2carul7m5yuf4xvydvom2sgqyqv6weg2ccys2ugogv2pchib.py
# Topologically Sorted Source Nodes: [silu_9, mul_9], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   mul_9 => mul_9
#   silu_9 => add_18, convert_element_type_85, convert_element_type_86, div_9, exp_9, neg_9
# Graph fragment:
#   %mm_27 : Tensor "bf16[2048, 512][512, 1]cuda:0" = PlaceHolder[target=mm_27]
#   %mm_28 : Tensor "bf16[2048, 512][512, 1]cuda:0" = PlaceHolder[target=mm_28]
#   %convert_element_type_85 : Tensor "f32[2048, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_27, torch.float32), kwargs = {})
#   %neg_9 : Tensor "f32[2048, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_85,), kwargs = {})
#   %exp_9 : Tensor "f32[2048, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_9,), kwargs = {})
#   %add_18 : Tensor "f32[2048, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_9, 1), kwargs = {})
#   %div_9 : Tensor "f32[2048, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_85, %add_18), kwargs = {})
#   %convert_element_type_86 : Tensor "bf16[2048, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_9, torch.bfloat16), kwargs = {})
#   %mul_9 : Tensor "bf16[2048, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_86, %mm_28), kwargs = {})
#   return %mul_9
triton_poi_fused_mul_silu_1 = async_compile.triton('triton_poi_fused_mul_silu_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_silu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 8388608}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(in_out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm-m4/jay_outputs/m4_peak_screen/23969387_5_2048_tiled/cache/sz/cszvkocayuyjs4jm5mjqwkjte66iw2bnisf7xf4sdked7i4n5est.py
# Topologically Sorted Source Nodes: [accumulated_1, float_2, accumulated_2, float_3, accumulated_3, float_4, accumulated_4, float_5, accumulated_5, float_6, accumulated_6, float_7, accumulated_7, float_8, accumulated_8, float_9, accumulated_9, float_10, accumulated_10, to], Original ATen: [aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   accumulated_1 => convert_element_type_8
#   accumulated_10 => add_19
#   accumulated_2 => add_3
#   accumulated_3 => add_5
#   accumulated_4 => add_7
#   accumulated_5 => add_9
#   accumulated_6 => add_11
#   accumulated_7 => add_13
#   accumulated_8 => add_15
#   accumulated_9 => add_17
#   float_10 => convert_element_type_89
#   float_2 => convert_element_type_17
#   float_3 => convert_element_type_26
#   float_4 => convert_element_type_35
#   float_5 => convert_element_type_44
#   float_6 => convert_element_type_53
#   float_7 => convert_element_type_62
#   float_8 => convert_element_type_71
#   float_9 => convert_element_type_80
#   to => convert_element_type_90
# Graph fragment:
#   %mm_2 : Tensor "bf16[2048, 3584][3584, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %mm_5 : Tensor "bf16[2048, 3584][3584, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %mm_8 : Tensor "bf16[2048, 3584][3584, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %mm_11 : Tensor "bf16[2048, 3584][3584, 1]cuda:0" = PlaceHolder[target=mm_11]
#   %mm_14 : Tensor "bf16[2048, 3584][3584, 1]cuda:0" = PlaceHolder[target=mm_14]
#   %mm_17 : Tensor "bf16[2048, 3584][3584, 1]cuda:0" = PlaceHolder[target=mm_17]
#   %mm_20 : Tensor "bf16[2048, 3584][3584, 1]cuda:0" = PlaceHolder[target=mm_20]
#   %mm_23 : Tensor "bf16[2048, 3584][3584, 1]cuda:0" = PlaceHolder[target=mm_23]
#   %mm_26 : Tensor "bf16[2048, 3584][3584, 1]cuda:0" = PlaceHolder[target=mm_26]
#   %add_17 : Tensor "f32[2048, 3584][3584, 1]cuda:0" = PlaceHolder[target=add_17]
#   %mm_29 : Tensor "bf16[2048, 3584][3584, 1]cuda:0" = PlaceHolder[target=mm_29]
#   %convert_element_type_8 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_17 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_5, torch.float32), kwargs = {})
#   %add_3 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_8, %convert_element_type_17), kwargs = {})
#   %convert_element_type_26 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_8, torch.float32), kwargs = {})
#   %add_5 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %convert_element_type_26), kwargs = {})
#   %convert_element_type_35 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_11, torch.float32), kwargs = {})
#   %add_7 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %convert_element_type_35), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_14, torch.float32), kwargs = {})
#   %add_9 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %convert_element_type_44), kwargs = {})
#   %convert_element_type_53 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_17, torch.float32), kwargs = {})
#   %add_11 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %convert_element_type_53), kwargs = {})
#   %convert_element_type_62 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_20, torch.float32), kwargs = {})
#   %add_13 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %convert_element_type_62), kwargs = {})
#   %convert_element_type_71 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_23, torch.float32), kwargs = {})
#   %add_15 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %convert_element_type_71), kwargs = {})
#   %convert_element_type_80 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_26, torch.float32), kwargs = {})
#   %add_17 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %convert_element_type_80), kwargs = {})
#   %convert_element_type_89 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_29, torch.float32), kwargs = {})
#   %add_19 : Tensor "f32[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %convert_element_type_89), kwargs = {})
#   %convert_element_type_90 : Tensor "bf16[2048, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_19, torch.bfloat16), kwargs = {})
#   return %add_17,%convert_element_type_90
triton_poi_fused__to_copy_add_2 = async_compile.triton('triton_poi_fused__to_copy_add_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 176160768}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7340032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp14 = tl.load(in_ptr5 + (x0), None).to(tl.float32)
    tmp17 = tl.load(in_ptr6 + (x0), None).to(tl.float32)
    tmp20 = tl.load(in_ptr7 + (x0), None).to(tl.float32)
    tmp23 = tl.load(in_ptr8 + (x0), None).to(tl.float32)
    tmp26 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 + tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp29, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride_grouped((arg1_1, arg0_1), ((2048, 3584), (18944, 3584)), ((3584, 1), (3584, 1)), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((2048, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, numpy_t, gate], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (3584, 2048), (1, 3584), 0), out=buf0)
            assert_size_stride(arg2_1, (18944, 3584), (3584, 1), 'input')
            arg2_1 = copy_if_misaligned(arg2_1)
            buf1 = empty_strided_cuda((2048, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_1, numpy_t_1, up], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, (3584, 2048), (1, 3584), 0), out=buf1)
            buf2 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [silu, mul], Original ATen: [aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused_mul_silu_0:3
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_0.run(buf2, buf1, 4194304, stream=raw_stream0)
            assert_size_stride(arg3_1, (18944, 3584), (3584, 1), 'input')
            arg3_1 = copy_if_misaligned(arg3_1)
            buf3 = empty_strided_cuda((2048, 3584), (3584, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu, mul, getitem_2, partial], Original ATen: [aten.silu, aten.mul, aten.slice, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(buf2, reinterpret_tensor(arg3_1, (2048, 3584), (3584, 1), 0), out=buf3)
            buf4 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [getitem_3, numpy_t_2, gate_1], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (3584, 2048), (1, 3584), 7340032), out=buf4)
            buf5 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [getitem_4, numpy_t_3, up_1], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:6
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, (3584, 2048), (1, 3584), 7340032), out=buf5)
            buf6 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [silu_1, mul_1], Original ATen: [aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused_mul_silu_0:7
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_0.run(buf6, buf5, 4194304, stream=raw_stream0)
            buf7 = empty_strided_cuda((2048, 3584), (3584, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_1, mul_1, getitem_5, partial_1], Original ATen: [aten.silu, aten.mul, aten.slice, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:8
            extern_kernels.mm(buf6, reinterpret_tensor(arg3_1, (2048, 3584), (3584, 1), 7340032), out=buf7)
            buf8 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [getitem_6, numpy_t_4, gate_2], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:9
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (3584, 2048), (1, 3584), 14680064), out=buf8)
            buf9 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [getitem_7, numpy_t_5, up_2], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:10
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, (3584, 2048), (1, 3584), 14680064), out=buf9)
            buf10 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [silu_2, mul_2], Original ATen: [aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused_mul_silu_0:11
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_0.run(buf10, buf9, 4194304, stream=raw_stream0)
            buf11 = empty_strided_cuda((2048, 3584), (3584, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_2, mul_2, getitem_8, partial_2], Original ATen: [aten.silu, aten.mul, aten.slice, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:12
            extern_kernels.mm(buf10, reinterpret_tensor(arg3_1, (2048, 3584), (3584, 1), 14680064), out=buf11)
            buf12 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [getitem_9, numpy_t_6, gate_3], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:13
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (3584, 2048), (1, 3584), 22020096), out=buf12)
            buf13 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [getitem_10, numpy_t_7, up_3], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:14
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, (3584, 2048), (1, 3584), 22020096), out=buf13)
            buf14 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [silu_3, mul_3], Original ATen: [aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused_mul_silu_0:15
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_0.run(buf14, buf13, 4194304, stream=raw_stream0)
            buf15 = empty_strided_cuda((2048, 3584), (3584, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_3, mul_3, getitem_11, partial_3], Original ATen: [aten.silu, aten.mul, aten.slice, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:16
            extern_kernels.mm(buf14, reinterpret_tensor(arg3_1, (2048, 3584), (3584, 1), 22020096), out=buf15)
            buf16 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [getitem_12, numpy_t_8, gate_4], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:17
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (3584, 2048), (1, 3584), 29360128), out=buf16)
            buf17 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, numpy_t_9, up_4], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:18
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, (3584, 2048), (1, 3584), 29360128), out=buf17)
            buf18 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [silu_4, mul_4], Original ATen: [aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused_mul_silu_0:19
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_0.run(buf18, buf17, 4194304, stream=raw_stream0)
            buf19 = empty_strided_cuda((2048, 3584), (3584, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_4, mul_4, getitem_14, partial_4], Original ATen: [aten.silu, aten.mul, aten.slice, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:20
            extern_kernels.mm(buf18, reinterpret_tensor(arg3_1, (2048, 3584), (3584, 1), 29360128), out=buf19)
            buf20 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [getitem_15, numpy_t_10, gate_5], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:21
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (3584, 2048), (1, 3584), 36700160), out=buf20)
            buf21 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, numpy_t_11, up_5], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:22
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, (3584, 2048), (1, 3584), 36700160), out=buf21)
            buf22 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [silu_5, mul_5], Original ATen: [aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused_mul_silu_0:23
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_0.run(buf22, buf21, 4194304, stream=raw_stream0)
            buf23 = empty_strided_cuda((2048, 3584), (3584, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_5, mul_5, getitem_17, partial_5], Original ATen: [aten.silu, aten.mul, aten.slice, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:24
            extern_kernels.mm(buf22, reinterpret_tensor(arg3_1, (2048, 3584), (3584, 1), 36700160), out=buf23)
            buf24 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [getitem_18, numpy_t_12, gate_6], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:25
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (3584, 2048), (1, 3584), 44040192), out=buf24)
            buf25 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [getitem_19, numpy_t_13, up_6], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:26
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, (3584, 2048), (1, 3584), 44040192), out=buf25)
            buf26 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [silu_6, mul_6], Original ATen: [aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused_mul_silu_0:27
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_0.run(buf26, buf25, 4194304, stream=raw_stream0)
            buf27 = empty_strided_cuda((2048, 3584), (3584, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_6, mul_6, getitem_20, partial_6], Original ATen: [aten.silu, aten.mul, aten.slice, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:28
            extern_kernels.mm(buf26, reinterpret_tensor(arg3_1, (2048, 3584), (3584, 1), 44040192), out=buf27)
            buf28 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [getitem_21, numpy_t_14, gate_7], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:29
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (3584, 2048), (1, 3584), 51380224), out=buf28)
            buf29 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [getitem_22, numpy_t_15, up_7], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:30
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, (3584, 2048), (1, 3584), 51380224), out=buf29)
            buf30 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [silu_7, mul_7], Original ATen: [aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused_mul_silu_0:31
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_0.run(buf30, buf29, 4194304, stream=raw_stream0)
            buf31 = empty_strided_cuda((2048, 3584), (3584, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_7, mul_7, getitem_23, partial_7], Original ATen: [aten.silu, aten.mul, aten.slice, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:32
            extern_kernels.mm(buf30, reinterpret_tensor(arg3_1, (2048, 3584), (3584, 1), 51380224), out=buf31)
            buf32 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [getitem_24, numpy_t_16, gate_8], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:33
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (3584, 2048), (1, 3584), 58720256), out=buf32)
            buf33 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [getitem_25, numpy_t_17, up_8], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:34
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, (3584, 2048), (1, 3584), 58720256), out=buf33)
            buf34 = buf32; del buf32  # reuse
            # Topologically Sorted Source Nodes: [silu_8, mul_8], Original ATen: [aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused_mul_silu_0:35
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_0.run(buf34, buf33, 4194304, stream=raw_stream0)
            del buf33
            buf35 = empty_strided_cuda((2048, 3584), (3584, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_8, mul_8, getitem_26, partial_8], Original ATen: [aten.silu, aten.mul, aten.slice, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:36
            extern_kernels.mm(buf34, reinterpret_tensor(arg3_1, (2048, 3584), (3584, 1), 58720256), out=buf35)
            del buf34
            buf37 = empty_strided_cuda((2048, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_27, numpy_t_18, gate_9], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:37
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (3584, 512), (1, 3584), 66060288), out=buf37)
            del arg0_1
            buf38 = empty_strided_cuda((2048, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_28, numpy_t_19, up_9], Original ATen: [aten.slice, aten.permute, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:38
            extern_kernels.mm(arg1_1, reinterpret_tensor(arg2_1, (3584, 512), (1, 3584), 66060288), out=buf38)
            del arg1_1
            del arg2_1
            buf39 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [silu_9, mul_9], Original ATen: [aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused_mul_silu_1:39
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_1.run(buf39, buf38, 1048576, stream=raw_stream0)
            del buf38
            buf40 = empty_strided_cuda((2048, 3584), (3584, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [silu_9, mul_9, getitem_29, partial_9], Original ATen: [aten.silu, aten.mul, aten.slice, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:40
            extern_kernels.mm(buf39, reinterpret_tensor(arg3_1, (512, 3584), (3584, 1), 66060288), out=buf40)
            del arg3_1
            del buf39
            buf41 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [accumulated_1, float_2, accumulated_2, float_3, accumulated_3, float_4, accumulated_4, float_5, accumulated_5, float_6, accumulated_6, float_7, accumulated_7, float_8, accumulated_8, float_9, accumulated_9, float_10, accumulated_10, to], Original ATen: [aten.add, aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_2:41
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_2.run(buf41, buf3, buf7, buf11, buf15, buf19, buf23, buf27, buf31, buf35, 7340032, stream=raw_stream0)
            del buf11
            del buf15
            del buf19
            del buf23
            del buf27
            del buf3
            del buf31
            del buf35
            del buf7
        return (buf41, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((18944, 3584), (3584, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((2048, 3584), (3584, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((18944, 3584), (3584, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((18944, 3584), (3584, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
