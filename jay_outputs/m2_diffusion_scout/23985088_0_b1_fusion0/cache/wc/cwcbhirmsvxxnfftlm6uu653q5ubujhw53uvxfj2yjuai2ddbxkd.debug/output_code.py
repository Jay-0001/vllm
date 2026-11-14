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


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_diffusion_scout/23985088_0_b1_fusion0/cache/nm/cnm5d7miastkl27tbra4nl36teab2ozupzeel2fwkcrohi7dbeo4.py
# Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => clone, convert_element_type, var_mean, view
# Graph fragment:
#   %arg2_1 : Tensor "bf16[1, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = PlaceHolder[target=arg2_1]
#   %clone : Tensor "bf16[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%arg2_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone, torch.float32), kwargs = {})
#   %view : Tensor "f32[1, 32, 16, 256][131072, 4096, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [1, 32, 16, 256]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf1
triton_red_fused_native_group_norm_0 = async_compile.triton('triton_red_fused_native_group_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_native_group_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 512, 'r0_': 262144}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_native_group_norm_0(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 16)
        r0_2 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0 + 512*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_diffusion_scout/23985088_0_b1_fusion0/cache/vj/cvjrn24mlfwwm7rtuv5ttrtw6ype5g6ifxnaq6fhok7pjzkj3bsb.py
# Topologically Sorted Source Nodes: [hidden_states, hidden_states_1], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states => add, add_1, clone, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, var_mean, view, view_1
#   hidden_states_1 => add_2, convert_element_type_4, convert_element_type_5, div, exp, neg
# Graph fragment:
#   %arg2_1 : Tensor "bf16[1, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = PlaceHolder[target=arg2_1]
#   %getitem_1 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=buf1]
#   %arg0_1 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[512][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convert_element_type_4 : Tensor "f32[1, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = PlaceHolder[target=convert_element_type_4]
#   %clone : Tensor "bf16[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%arg2_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone, torch.float32), kwargs = {})
#   %view : Tensor "f32[1, 32, 16, 256][131072, 4096, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [1, 32, 16, 256]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[1, 32, 16, 256][131072, 4096, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[1, 32, 16, 256][131072, 4096, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul, [1, 512, 16, 16]), kwargs = {})
#   %unsqueeze : Tensor "bf16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg0_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "bf16[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %unsqueeze_2 : Tensor "bf16[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 3), kwargs = {})
#   %mul_1 : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "bf16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg1_1, 0), kwargs = {})
#   %unsqueeze_4 : Tensor "bf16[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_3, 2), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 3), kwargs = {})
#   %add_1 : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_4,), kwargs = {})
#   %exp : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_2 : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_4, %add_2), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   return %convert_element_type_4,%convert_element_type_5
triton_poi_fused_native_group_norm_silu_1 = async_compile.triton('triton_poi_fused_native_group_norm_silu_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_native_group_norm_silu_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 788736}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 // 16), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0 // 16), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1], 4096.0, tl.float32)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tl.full([1], 1e-06, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = -tmp18
    tmp20 = libdevice.exp(tmp19)
    tmp21 = tl.full([1], 1.0, tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = (tmp18 / tmp22)
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_diffusion_scout/23985088_0_b1_fusion0/cache/tg/ctg4paj4iz6ppfw6az74ejls3ksbsyg7w6b4ogp3nk7qjmxiikau.py
# Topologically Sorted Source Nodes: [temb], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   temb => add_3, convert_element_type_6, convert_element_type_7, div_1, exp_1, neg_1
# Graph fragment:
#   %arg5_1 : Tensor "bf16[1, 512][512, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convert_element_type_6 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %neg_1 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_6,), kwargs = {})
#   %exp_1 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_1,), kwargs = {})
#   %add_3 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1), kwargs = {})
#   %div_1 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_6, %add_3), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_7
triton_poi_fused_silu_2 = async_compile.triton('triton_poi_fused_silu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_silu_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 3072}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_diffusion_scout/23985088_0_b1_fusion0/cache/iz/cizrvt5xp53b5givn6wpbzbufynrufbgo2m2upklxjwfk6co3bj6.py
# Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, temb_1, hidden_states_3, hidden_states_4], Original ATen: [aten.silu, aten.convolution, aten.unsqueeze, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_1 => add_2, convert_element_type_5, div, exp, neg
#   hidden_states_2 => convolution
#   hidden_states_3 => add_4
#   hidden_states_4 => clone_1, convert_element_type_11, var_mean_1, view_2
#   temb_1 => unsqueeze_6, unsqueeze_7
# Graph fragment:
#   %buf5 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf5]
#   %arg4_1 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %addmm : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=addmm]
#   %neg : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_4,), kwargs = {})
#   %exp : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_2 : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_4, %add_2), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_5, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%addmm, 2), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, 3), kwargs = {})
#   %add_4 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %unsqueeze_7), kwargs = {})
#   %clone_1 : Tensor "bf16[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_4,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_11 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_1, torch.float32), kwargs = {})
#   %view_2 : Tensor "f32[1, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_11, [1, 32, 8, 256]), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_2, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_3,%buf9
triton_red_fused_add_convolution_native_group_norm_silu_unsqueeze_3 = async_compile.triton('triton_red_fused_add_convolution_native_group_norm_silu_unsqueeze_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_silu_unsqueeze_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 512, 'r0_': 132096}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_silu_unsqueeze_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp7_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = (r0_index % 8)
        r0_2 = r0_index // 8
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8*x0 + 256*r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 8*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 8*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight, roffset == 0
        )
        tmp7_mean = tl.where(r0_mask & xmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(r0_mask & xmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(r0_mask & xmask, tmp7_weight_next, tmp7_weight)
    tmp8, tmp9, tmp10 = triton_helpers.welford(tmp7_mean, tmp7_m2, tmp7_weight, 1)
    tmp7 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tl.store(out_ptr1 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_diffusion_scout/23985088_0_b1_fusion0/cache/2c/c2ccb3azzvuj67mc2wqtgo4wpn4srqkyxjcrxn3j2nb5jnuo5uxf.py
# Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, temb_1, hidden_states_3, hidden_states_4, hidden_states_5], Original ATen: [aten.silu, aten.convolution, aten.unsqueeze, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_1 => add_2, convert_element_type_5, div, exp, neg
#   hidden_states_2 => convolution
#   hidden_states_3 => add_4
#   hidden_states_4 => add_5, add_6, clone_1, convert_element_type_11, convert_element_type_12, mul_2, mul_3, rsqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_8, unsqueeze_9, var_mean_1, view_2, view_3
#   hidden_states_5 => add_7, convert_element_type_15, convert_element_type_16, div_2, exp_2, neg_2
#   temb_1 => unsqueeze_6, unsqueeze_7
# Graph fragment:
#   %buf5 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf5]
#   %arg4_1 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %addmm : Tensor "bf16[1, 256][256, 1]cuda:0" = PlaceHolder[target=addmm]
#   %getitem_3 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=getitem_3]
#   %buf9 : Tensor "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0" = PlaceHolder[target=buf9]
#   %arg8_1 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %convert_element_type_15 : Tensor "f32[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convert_element_type_15]
#   %neg : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_4,), kwargs = {})
#   %exp : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_2 : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_4, %add_2), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[1, 512, 16, 16][131072, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %convolution : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_5, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%addmm, 2), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, 3), kwargs = {})
#   %add_4 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %unsqueeze_7), kwargs = {})
#   %clone_1 : Tensor "bf16[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_4,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_11 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_1, torch.float32), kwargs = {})
#   %view_2 : Tensor "f32[1, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_11, [1, 32, 8, 256]), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_2, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[1, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %getitem_3), kwargs = {})
#   %add_5 : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 32, 8, 256][65536, 2048, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %view_3 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_2, [1, 256, 16, 16]), kwargs = {})
#   %unsqueeze_8 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg8_1, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "bf16[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, 2), kwargs = {})
#   %unsqueeze_10 : Tensor "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 3), kwargs = {})
#   %mul_3 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %unsqueeze_10), kwargs = {})
#   %unsqueeze_11 : Tensor "bf16[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "bf16[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_11, 2), kwargs = {})
#   %unsqueeze_13 : Tensor "bf16[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, 3), kwargs = {})
#   %add_6 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %unsqueeze_13), kwargs = {})
#   %convert_element_type_12 : Tensor "bf16[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.bfloat16), kwargs = {})
#   %convert_element_type_15 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_12, torch.float32), kwargs = {})
#   %neg_2 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_15,), kwargs = {})
#   %exp_2 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %add_7 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_2, 1), kwargs = {})
#   %div_2 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_15, %add_7), kwargs = {})
#   %convert_element_type_16 : Tensor "bf16[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.bfloat16), kwargs = {})
#   return %convert_element_type_15,%convert_element_type_16
triton_poi_fused_add_convolution_native_group_norm_silu_unsqueeze_4 = async_compile.triton('triton_poi_fused_add_convolution_native_group_norm_silu_unsqueeze_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_convolution_native_group_norm_silu_unsqueeze_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 395520}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_native_group_norm_silu_unsqueeze_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0 // 8), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0 // 8), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 - tmp6
    tmp9 = tl.full([1], 2048.0, tl.float32)
    tmp10 = (tmp8 / tmp9)
    tmp11 = tl.full([1], 1e-06, tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp7 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = -tmp22
    tmp24 = libdevice.exp(tmp23)
    tmp25 = tl.full([1], 1.0, tl.float32)
    tmp26 = tmp24 + tmp25
    tmp27 = (tmp22 / tmp26)
    tmp28 = tmp27.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp28, None)
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_diffusion_scout/23985088_0_b1_fusion0/cache/wp/cwplvs3fbfhmjeor7t3o4wowtf75a5qxobxerbwqyz6wnogej2ut.py
# Topologically Sorted Source Nodes: [input_tensor, hidden_states_5, hidden_states_7, output_tensor], Original ATen: [aten.convolution, aten.silu, aten.div]
# Source node to ATen node mapping:
#   hidden_states_5 => add_7, convert_element_type_16, div_2, exp_2, neg_2
#   hidden_states_7 => convolution_1
#   input_tensor => convolution_2
#   output_tensor => add_8
# Graph fragment:
#   %buf11 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf11]
#   %arg13_1 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %buf14 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=buf14]
#   %arg11_1 : Tensor "bf16[256][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %convolution_2 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg12_1, %arg13_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %neg_2 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_15,), kwargs = {})
#   %exp_2 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %add_7 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_2, 1), kwargs = {})
#   %div_2 : Tensor "f32[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_15, %add_7), kwargs = {})
#   %convert_element_type_16 : Tensor "bf16[1, 256, 16, 16][65536, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.bfloat16), kwargs = {})
#   %convolution_1 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_16, %arg10_1, %arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_8 : Tensor "bf16[1, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %convolution_1), kwargs = {})
#   return %add_8
triton_poi_fused_convolution_div_silu_5 = async_compile.triton('triton_poi_fused_convolution_div_silu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convolution_div_silu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 525312}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_div_silu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1 = args
        args.clear()
        assert_size_stride(arg2_1, (1, 512, 16, 16), (131072, 1, 8192, 512), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf0 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
            buf1 = empty_strided_cuda((1, 32, 1, 1), (32, 1, 32, 32), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states], Original ATen: [aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_native_group_norm_0:1
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_native_group_norm_0.run(arg2_1, buf0, buf1, 32, 4096, stream=raw_stream0)
            assert_size_stride_grouped((arg0_1, arg1_1), ((512, ), (512, )), ((1, ), (1, )), 'input')
            buf4 = empty_strided_cuda((1, 512, 16, 16), (131072, 1, 8192, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states, hidden_states_1], Original ATen: [aten.native_group_norm, aten.silu]
            # [Provenance debug handles] triton_poi_fused_native_group_norm_silu_1:2
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_native_group_norm_silu_1.run(arg2_1, buf0, buf1, arg0_1, arg1_1, buf4, 131072, stream=raw_stream0)
            del arg0_1
            del arg1_1
            assert_size_stride(arg3_1, (256, 512, 3, 3), (4608, 1, 1536, 512), 'input')
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:3
            buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_tensor_metadata(buf5, (1, 256, 16, 16), (65536, 1, 4096, 256), torch.bfloat16, 'torch.ops.aten.convolution.default')
            del arg3_1
            del buf4
            assert_size_stride(arg5_1, (1, 512), (512, 1), 'input')
            arg5_1 = copy_if_misaligned(arg5_1)
            buf6 = empty_strided_cuda((1, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [temb], Original ATen: [aten.silu]
            # [Provenance debug handles] triton_poi_fused_silu_2:4
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_silu_2.run(arg5_1, buf6, 512, stream=raw_stream0)
            del arg5_1
            assert_size_stride_grouped((arg7_1, arg6_1), ((256, ), (256, 512)), ((1, ), (512, 1)), 'input')
            buf7 = empty_strided_cuda((1, 256), (256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [temb, linear], Original ATen: [aten.silu, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:5
            extern_kernels.addmm(arg7_1, buf6, reinterpret_tensor(arg6_1, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf7)
            del arg6_1
            del arg7_1
            del buf6
            assert_size_stride(arg4_1, (256, ), (1, ), 'input')
            buf8 = buf1; del buf1  # reuse
            buf9 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, temb_1, hidden_states_3, hidden_states_4], Original ATen: [aten.silu, aten.convolution, aten.unsqueeze, aten.add, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_add_convolution_native_group_norm_silu_unsqueeze_3:6
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_add_convolution_native_group_norm_silu_unsqueeze_3.run(buf5, arg4_1, buf7, buf8, buf9, 32, 2048, stream=raw_stream0)
            assert_size_stride(arg12_1, (256, 512, 1, 1), (512, 1, 512, 512), 'input')
            # Topologically Sorted Source Nodes: [input_tensor], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:7
            buf11 = extern_kernels.convolution(arg2_1, arg12_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_tensor_metadata(buf11, (1, 256, 16, 16), (65536, 1, 4096, 256), torch.bfloat16, 'torch.ops.aten.convolution.default')
            del arg12_1
            del arg2_1
            assert_size_stride_grouped((arg8_1, arg9_1), ((256, ), (256, )), ((1, ), (1, )), 'input')
            buf13 = empty_strided_cuda((1, 256, 16, 16), (65536, 1, 4096, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, temb_1, hidden_states_3, hidden_states_4, hidden_states_5], Original ATen: [aten.silu, aten.convolution, aten.unsqueeze, aten.add, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_add_convolution_native_group_norm_silu_unsqueeze_4:8
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_convolution_native_group_norm_silu_unsqueeze_4.run(buf5, arg4_1, buf7, buf8, buf9, arg8_1, arg9_1, buf13, 65536, stream=raw_stream0)
            del arg4_1
            del arg8_1
            del arg9_1
            del buf5
            del buf7
            del buf8
            del buf9
            assert_size_stride(arg10_1, (256, 256, 3, 3), (2304, 1, 768, 256), 'input')
            # Topologically Sorted Source Nodes: [hidden_states_5, hidden_states_7], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:9
            buf14 = extern_kernels.convolution(buf13, arg10_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_tensor_metadata(buf14, (1, 256, 16, 16), (65536, 1, 4096, 256), torch.bfloat16, 'torch.ops.aten.convolution.default')
            del arg10_1
            del buf13
            assert_size_stride_grouped((arg13_1, arg11_1), ((256, ), (256, )), ((1, ), (1, )), 'input')
            buf15 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [input_tensor, hidden_states_5, hidden_states_7, output_tensor], Original ATen: [aten.convolution, aten.silu, aten.div]
            # [Provenance debug handles] triton_poi_fused_convolution_div_silu_5:10
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_div_silu_5.run(buf15, arg13_1, buf14, arg11_1, 65536, stream=raw_stream0)
            del arg11_1
            del arg13_1
            del buf14
        return (buf15, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((256, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
