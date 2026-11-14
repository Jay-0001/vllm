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


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_mamba_scout/23967149_0_t32/cache/m2/cm2gsdpxncitsgsyp7zpp6orsx673dbmlzahlvidwjr52eaivlta.py
# Topologically Sorted Source Nodes: [linear, projected_states, chunk, squeeze, unsqueeze, conv1d], Original ATen: [aten._unsafe_view, aten.transpose, aten.split, aten.squeeze, aten.unsqueeze, aten.convolution]
# Source node to ATen node mapping:
#   chunk => split
#   conv1d => convolution
#   linear => view_1
#   projected_states => permute_1
#   squeeze => squeeze
#   unsqueeze => unsqueeze
# Graph fragment:
#   %mm : Tensor "f32[32, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_1 : Tensor "f32[1, 32, 3072][98304, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 32, 3072]), kwargs = {})
#   %permute_1 : Tensor "f32[1, 3072, 32][98304, 1, 3072]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 2, 1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%permute_1, 1536, 1), kwargs = {})
#   %squeeze : Tensor "f32[1536, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%arg3_1, 1), kwargs = {})
#   %unsqueeze : Tensor "f32[1536, 1, 4][4, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze, 1), kwargs = {})
#   %convolution : Tensor "f32[1, 1536, 35][53760, 35, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %unsqueeze, %arg4_1, [1], [3], [1], False, [0], 1536), kwargs = {})
#   return %buf1
triton_poi_fused__unsafe_view_convolution_split_squeeze_transpose_unsqueeze_0 = async_compile.triton('triton_poi_fused__unsafe_view_convolution_split_squeeze_transpose_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused__unsafe_view_convolution_split_squeeze_transpose_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 196608, 'x': 393216}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_convolution_split_squeeze_transpose_unsqueeze_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 3072*x1), xmask & ymask)
    tl.store(out_ptr0 + (x1 + 32*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_mamba_scout/23967149_0_t32/cache/qc/cqco4ozdou7ttsmma6siilvr65l3kkledgvngu4amsg63obnvqdn.py
# Topologically Sorted Source Nodes: [linear, projected_states, chunk, squeeze, unsqueeze, conv1d, out, out_1], Original ATen: [aten._unsafe_view, aten.transpose, aten.split, aten.squeeze, aten.unsqueeze, aten.convolution, aten.slice, aten.silu]
# Source node to ATen node mapping:
#   chunk => split
#   conv1d => convolution
#   linear => view_1
#   out => slice_1
#   out_1 => add, div, exp_1, neg_1
#   projected_states => permute_1
#   squeeze => squeeze
#   unsqueeze => unsqueeze
# Graph fragment:
#   %buf2 : Tensor "f32[1, 1536, 35][53760, 35, 1]cuda:0" = PlaceHolder[target=buf2]
#   %arg4_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %view_1 : Tensor "f32[1, 32, 3072][98304, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 32, 3072]), kwargs = {})
#   %permute_1 : Tensor "f32[1, 3072, 32][98304, 1, 3072]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 2, 1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%permute_1, 1536, 1), kwargs = {})
#   %squeeze : Tensor "f32[1536, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%arg3_1, 1), kwargs = {})
#   %unsqueeze : Tensor "f32[1536, 1, 4][4, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze, 1), kwargs = {})
#   %convolution : Tensor "f32[1, 1536, 35][53760, 35, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %unsqueeze, %arg4_1, [1], [3], [1], False, [0], 1536), kwargs = {})
#   %slice_1 : Tensor "f32[1, 1536, 32][53760, 35, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convolution, 2, 0, 32), kwargs = {})
#   %neg_1 : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_1,), kwargs = {})
#   %exp_1 : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_1,), kwargs = {})
#   %add : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1), kwargs = {})
#   %div : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_1, %add), kwargs = {})
#   return %div
triton_poi_fused__unsafe_view_convolution_silu_slice_split_squeeze_transpose_unsqueeze_1 = async_compile.triton('triton_poi_fused__unsafe_view_convolution_silu_slice_split_squeeze_transpose_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_view_convolution_silu_slice_split_squeeze_transpose_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 595968}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_convolution_silu_slice_split_squeeze_transpose_unsqueeze_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 35*x1), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = -tmp2
    tmp4 = libdevice.exp(tmp3)
    tmp5 = tl.full([1], 1.0, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = (tmp2 / tmp6)
    tl.store(out_ptr0 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_mamba_scout/23967149_0_t32/cache/6w/c6whtfqdmyaxbxvsz3eavk32cb3letpv5efjrojhdhk456nb4gnd.py
# Topologically Sorted Source Nodes: [linear, projected_states, chunk, squeeze, unsqueeze, conv1d, out, out_1, hidden_states_B_C_2, linear_1, split, exp, A, getitem_7, time_step_1, getitem_6, dt, dt_1, getitem_8, mul, discrete_A, movedim, getitem_9, getitem_10, discrete_B, transpose_3, getitem_11, deltaB_u, movedim_1, associative_scan], Original ATen: [aten._unsafe_view, aten.transpose, aten.split, aten.squeeze, aten.unsqueeze, aten.convolution, aten.slice, aten.silu, aten.split_with_sizes, aten.exp, aten.neg, aten.clone, aten.add, aten.softplus, aten.mul, aten.permute]
# Source node to ATen node mapping:
#   A => neg
#   associative_scan => associative_scan
#   chunk => split
#   conv1d => convolution
#   deltaB_u => mul_2
#   discrete_A => exp_3
#   discrete_B => mul_1
#   dt => add_1
#   dt_1 => exp_2, gt, log1p, where
#   exp => exp
#   getitem_10 => unsqueeze_6
#   getitem_11 => unsqueeze_7
#   getitem_6 => unsqueeze_1
#   getitem_7 => unsqueeze_2, unsqueeze_3
#   getitem_8 => unsqueeze_4
#   getitem_9 => unsqueeze_5
#   hidden_states_B_C_2 => permute_2
#   linear => view_1
#   linear_1 => view_3
#   movedim => permute_13
#   movedim_1 => permute_14
#   mul => mul
#   out => slice_1
#   out_1 => add, div, exp_1, neg_1
#   projected_states => permute_1
#   split => split_with_sizes
#   squeeze => squeeze
#   time_step_1 => clone, permute_7, view_5
#   transpose_3 => permute_8
#   unsqueeze => unsqueeze
# Graph fragment:
#   %arg2_1 : Tensor "f32[1536, 16][16, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %mm_2 : Tensor "f32[32, 1536][1536, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %arg7_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %mm_1 : Tensor "f32[32, 80][80, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %div : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0" = PlaceHolder[target=div]
#   %view_1 : Tensor "f32[1, 32, 3072][98304, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 32, 3072]), kwargs = {})
#   %permute_1 : Tensor "f32[1, 3072, 32][98304, 1, 3072]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 2, 1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%permute_1, 1536, 1), kwargs = {})
#   %squeeze : Tensor "f32[1536, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%arg3_1, 1), kwargs = {})
#   %unsqueeze : Tensor "f32[1536, 1, 4][4, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze, 1), kwargs = {})
#   %convolution : Tensor "f32[1, 1536, 35][53760, 35, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %unsqueeze, %arg4_1, [1], [3], [1], False, [0], 1536), kwargs = {})
#   %slice_1 : Tensor "f32[1, 1536, 32][53760, 35, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convolution, 2, 0, 32), kwargs = {})
#   %neg_1 : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_1,), kwargs = {})
#   %exp_1 : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_1,), kwargs = {})
#   %add : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1), kwargs = {})
#   %div : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_1, %add), kwargs = {})
#   %permute_2 : Tensor "f32[1, 32, 1536][49152, 1, 32]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%div, [0, 2, 1]), kwargs = {})
#   %view_3 : Tensor "f32[1, 32, 80][2560, 80, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [1, 32, 80]), kwargs = {})
#   %split_with_sizes : [num_users=3] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%view_3, [48, 16, 16], -1), kwargs = {})
#   %exp : Tensor "f32[1536, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%arg2_1,), kwargs = {})
#   %neg : Tensor "f32[1536, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%exp,), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 1536, 16][24576, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%neg, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 1536, 1, 16][24576, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 2), kwargs = {})
#   %view_5 : Tensor "f32[1, 32, 1536][49152, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [1, 32, 1536]), kwargs = {})
#   %permute_7 : Tensor "f32[1, 1536, 32][49152, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1]), kwargs = {})
#   %clone : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
#   %unsqueeze_1 : Tensor "f32[1536, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, 1), kwargs = {})
#   %add_1 : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clone, %unsqueeze_1), kwargs = {})
#   %gt : Tensor "b8[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 20), kwargs = {})
#   %exp_2 : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%add_1,), kwargs = {})
#   %log1p : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp_2,), kwargs = {})
#   %where : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %log1p), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 1536, 32, 1][49152, 32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 3), kwargs = {})
#   %mul : Tensor "f32[1, 1536, 32, 16][786432, 512, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %unsqueeze_4), kwargs = {})
#   %exp_3 : Tensor "f32[1, 1536, 32, 16][786432, 512, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul,), kwargs = {})
#   %permute_13 : Tensor "f32[32, 1, 1536, 16][16, 786432, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%exp_3, [2, 0, 1, 3]), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 1536, 32, 1][49152, 32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 3), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[1, 1, 32, 16][2560, 2560, 80, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_3, 1), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1536, 32, 16][786432, 512, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %unsqueeze_6), kwargs = {})
#   %permute_8 : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_2, [0, 2, 1]), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 1536, 32, 1][49152, 32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_8, 3), kwargs = {})
#   %mul_2 : Tensor "f32[1, 1536, 32, 16][786432, 512, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_7), kwargs = {})
#   %permute_14 : Tensor "f32[32, 1, 1536, 16][16, 786432, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_2, [2, 0, 1, 3]), kwargs = {})
#   %associative_scan : [num_users=1] = call_function[target=torch.ops.higher_order.associative_scan](args = (%associative_scan_combine_graph_0, [%permute_13, %permute_14], ()), kwargs = {})
#   return %getitem_6
triton_per_fused__unsafe_view_add_clone_convolution_exp_mul_neg_permute_silu_slice_softplus_split_split_with_sizes_squeeze_transpose_unsqueeze_2 = async_compile.triton('triton_per_fused__unsafe_view_add_clone_convolution_exp_mul_neg_permute_silu_slice_softplus_split_split_with_sizes_squeeze_transpose_unsqueeze_2', '''
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
''', device_str='cuda')


# kernel path: /u/jsaravananindira/vllm/jay_outputs/m2_mamba_scout/23967149_0_t32/cache/2z/c2z5u4mlddknsguh3y3byumy5nzn2oujlppvji3o7d6fvzxa4fwk.py
# Topologically Sorted Source Nodes: [linear, projected_states, chunk, squeeze, unsqueeze, conv1d, out, out_1, hidden_states_B_C_2, transpose_3, scan_output, squeeze_1, scan_output_1, getitem_15, mul_3, scan_output_2, silu_1, scan_output_3], Original ATen: [aten._unsafe_view, aten.transpose, aten.split, aten.squeeze, aten.unsqueeze, aten.convolution, aten.slice, aten.silu, aten.view, aten.mul, aten.add]
# Source node to ATen node mapping:
#   chunk => split
#   conv1d => convolution
#   getitem_15 => unsqueeze_10, unsqueeze_9
#   hidden_states_B_C_2 => permute_2
#   linear => view_1
#   mul_3 => mul_3
#   out => slice_1
#   out_1 => add, div, exp_1, neg_1
#   projected_states => permute_1
#   scan_output => view_8
#   scan_output_1 => permute_18
#   scan_output_2 => add_2
#   scan_output_3 => mul_4
#   silu_1 => add_3, div_1, exp_4, neg_2
#   squeeze => squeeze
#   squeeze_1 => squeeze_1
#   transpose_3 => permute_8
#   unsqueeze => unsqueeze
# Graph fragment:
#   %bmm : Tensor "f32[32, 1536, 1][1536, 1, 1]cuda:0" = PlaceHolder[target=bmm]
#   %div : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0" = PlaceHolder[target=div]
#   %arg8_1 : Tensor "f32[1536][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %mm : Tensor "f32[32, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_1 : Tensor "f32[1, 32, 3072][98304, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 32, 3072]), kwargs = {})
#   %permute_1 : Tensor "f32[1, 3072, 32][98304, 1, 3072]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 2, 1]), kwargs = {})
#   %split : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%permute_1, 1536, 1), kwargs = {})
#   %squeeze : Tensor "f32[1536, 4][4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%arg3_1, 1), kwargs = {})
#   %unsqueeze : Tensor "f32[1536, 1, 4][4, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze, 1), kwargs = {})
#   %convolution : Tensor "f32[1, 1536, 35][53760, 35, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %unsqueeze, %arg4_1, [1], [3], [1], False, [0], 1536), kwargs = {})
#   %slice_1 : Tensor "f32[1, 1536, 32][53760, 35, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convolution, 2, 0, 32), kwargs = {})
#   %neg_1 : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_1,), kwargs = {})
#   %exp_1 : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_1,), kwargs = {})
#   %add : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1), kwargs = {})
#   %div : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_1, %add), kwargs = {})
#   %permute_2 : Tensor "f32[1, 32, 1536][49152, 1, 32]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%div, [0, 2, 1]), kwargs = {})
#   %permute_8 : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_2, [0, 2, 1]), kwargs = {})
#   %view_8 : Tensor "f32[1, 32, 1536, 1][49152, 1536, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [1, 32, 1536, 1]), kwargs = {})
#   %squeeze_1 : Tensor "f32[1, 32, 1536][49152, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%view_8, -1), kwargs = {})
#   %permute_18 : Tensor "f32[1, 1536, 32][49152, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%squeeze_1, [0, 2, 1]), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[1, 1536][1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg8_1, 0), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1536, 1][1536, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, 2), kwargs = {})
#   %mul_3 : Tensor "f32[1, 1536, 32][49152, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_8, %unsqueeze_10), kwargs = {})
#   %add_2 : Tensor "f32[1, 1536, 32][49152, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_18, %mul_3), kwargs = {})
#   %neg_2 : Tensor "f32[1, 1536, 32][49152, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%getitem_1,), kwargs = {})
#   %exp_4 : Tensor "f32[1, 1536, 32][49152, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %add_3 : Tensor "f32[1, 1536, 32][49152, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_4, 1), kwargs = {})
#   %div_1 : Tensor "f32[1, 1536, 32][49152, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%getitem_1, %add_3), kwargs = {})
#   %mul_4 : Tensor "f32[1, 1536, 32][49152, 1, 1536]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %div_1), kwargs = {})
#   return %mul_4
triton_poi_fused__unsafe_view_add_convolution_mul_silu_slice_split_squeeze_transpose_unsqueeze_view_3 = async_compile.triton('triton_poi_fused__unsafe_view_add_convolution_mul_silu_slice_split_squeeze_transpose_unsqueeze_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused__unsafe_view_add_convolution_mul_silu_slice_split_squeeze_transpose_unsqueeze_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 589824, 'x': 399360}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_convolution_mul_silu_slice_split_squeeze_transpose_unsqueeze_view_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 1536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 1536*y0), xmask & ymask)
    tmp1 = tl.load(in_out_ptr0 + (y0 + 32*x1), xmask & ymask)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (1536 + x1 + 3072*y0), xmask & ymask)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = -tmp5
    tmp7 = libdevice.exp(tmp6)
    tmp8 = tl.full([1, 1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = (tmp5 / tmp9)
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (y0 + 32*x1), tmp11, xmask & ymask)
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
        assert_size_stride_grouped((arg0_1, arg1_1), ((1, 32, 768), (3072, 768)), ((24576, 768, 1), (768, 1)), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((32, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(reinterpret_tensor(arg0_1, (32, 768), (768, 1), 0), reinterpret_tensor(arg1_1, (768, 3072), (1, 768), 0), out=buf0)
            del arg0_1
            del arg1_1
            buf1 = empty_strided_cuda((1, 1536, 32), (49152, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, projected_states, chunk, squeeze, unsqueeze, conv1d], Original ATen: [aten._unsafe_view, aten.transpose, aten.split, aten.squeeze, aten.unsqueeze, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_convolution_split_squeeze_transpose_unsqueeze_0:2
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_convolution_split_squeeze_transpose_unsqueeze_0.run(buf0, buf1, 1536, 32, stream=raw_stream0)
            assert_size_stride(arg3_1, (1536, 1, 4), (4, 4, 1), 'input')
            # Topologically Sorted Source Nodes: [linear, projected_states, chunk, squeeze, unsqueeze, conv1d], Original ATen: [aten._unsafe_view, aten.transpose, aten.split, aten.squeeze, aten.unsqueeze, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:3
            buf2 = extern_kernels.convolution(buf1, arg3_1, stride=(1,), padding=(3,), dilation=(1,), transposed=False, output_padding=(0,), groups=1536, bias=None)
            assert_tensor_metadata(buf2, (1, 1536, 35), (53760, 35, 1), torch.float32, 'torch.ops.aten.convolution.default')
            del arg3_1
            assert_size_stride(arg4_1, (1536, ), (1, ), 'input')
            buf3 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [linear, projected_states, chunk, squeeze, unsqueeze, conv1d, out, out_1], Original ATen: [aten._unsafe_view, aten.transpose, aten.split, aten.squeeze, aten.unsqueeze, aten.convolution, aten.slice, aten.silu]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_convolution_silu_slice_split_squeeze_transpose_unsqueeze_1:4
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_convolution_silu_slice_split_squeeze_transpose_unsqueeze_1.run(buf2, arg4_1, buf3, 49152, stream=raw_stream0)
            del arg4_1
            del buf2
            assert_size_stride(arg5_1, (80, 1536), (1536, 1), 'input')
            buf4 = empty_strided_cuda((32, 80), (80, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, projected_states, chunk, squeeze, unsqueeze, conv1d, out, out_1, hidden_states_B_C_2, linear_1], Original ATen: [aten._unsafe_view, aten.transpose, aten.split, aten.squeeze, aten.unsqueeze, aten.convolution, aten.slice, aten.silu, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(reinterpret_tensor(buf3, (32, 1536), (1, 32), 0), reinterpret_tensor(arg5_1, (1536, 80), (1, 1536), 0), out=buf4)
            del arg5_1
            assert_size_stride(arg6_1, (1536, 48), (48, 1), 'input')
            buf5 = empty_strided_cuda((32, 1536), (1536, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, split, time_step_1], Original ATen: [aten._unsafe_view, aten.split_with_sizes, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:6
            extern_kernels.mm(reinterpret_tensor(buf4, (32, 48), (80, 1), 0), reinterpret_tensor(arg6_1, (48, 1536), (1, 48), 0), out=buf5)
            del arg6_1
            assert_size_stride_grouped((arg2_1, arg7_1), ((1536, 16), (1536, )), ((16, 1), (1, )), 'input')
            buf7 = empty_strided_cuda((32, 1, 1536, 16), (24576, 786432, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, projected_states, chunk, squeeze, unsqueeze, conv1d, out, out_1, hidden_states_B_C_2, linear_1, split, exp, A, getitem_7, time_step_1, getitem_6, dt, dt_1, getitem_8, mul, discrete_A, movedim, getitem_9, getitem_10, discrete_B, transpose_3, getitem_11, deltaB_u, movedim_1, associative_scan], Original ATen: [aten._unsafe_view, aten.transpose, aten.split, aten.squeeze, aten.unsqueeze, aten.convolution, aten.slice, aten.silu, aten.split_with_sizes, aten.exp, aten.neg, aten.clone, aten.add, aten.softplus, aten.mul, aten.permute]
            # [Provenance debug handles] triton_per_fused__unsafe_view_add_clone_convolution_exp_mul_neg_permute_silu_slice_softplus_split_split_with_sizes_squeeze_transpose_unsqueeze_2:7
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_clone_convolution_exp_mul_neg_permute_silu_slice_softplus_split_split_with_sizes_squeeze_transpose_unsqueeze_2.run(arg2_1, buf5, arg7_1, buf4, buf3, buf7, 1536, 16, 32, stream=raw_stream0)
            del arg2_1
            del arg7_1
            buf8 = reinterpret_tensor(buf5, (32, 1536, 1), (1536, 1, 1), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [linear_1, split, all_states, transpose_8, scan_output, unsqueeze_1], Original ATen: [aten._unsafe_view, aten.split_with_sizes, aten.permute, aten.transpose, aten.view, aten.unsqueeze, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:8
            extern_kernels.bmm(reinterpret_tensor(buf7, (32, 1536, 16), (24576, 16, 1), 0), reinterpret_tensor(buf4, (32, 16, 1), (80, 1, 1), 64), out=buf8)
            del buf4
            del buf7
            assert_size_stride(arg8_1, (1536, ), (1, ), 'input')
            buf9 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [linear, projected_states, chunk, squeeze, unsqueeze, conv1d, out, out_1, hidden_states_B_C_2, transpose_3, scan_output, squeeze_1, scan_output_1, getitem_15, mul_3, scan_output_2, silu_1, scan_output_3], Original ATen: [aten._unsafe_view, aten.transpose, aten.split, aten.squeeze, aten.unsqueeze, aten.convolution, aten.slice, aten.silu, aten.view, aten.mul, aten.add]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_convolution_mul_silu_slice_split_squeeze_transpose_unsqueeze_view_3:9
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_convolution_mul_silu_slice_split_squeeze_transpose_unsqueeze_view_3.run(buf9, buf8, arg8_1, buf0, 32, 1536, stream=raw_stream0)
            del arg8_1
            del buf0
            del buf8
            assert_size_stride(arg9_1, (768, 1536), (1536, 1), 'input')
            buf10 = empty_strided_cuda((32, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, projected_states, chunk, squeeze, unsqueeze, conv1d, out, out_1, hidden_states_B_C_2, transpose_3, scan_output, squeeze_1, scan_output_1, getitem_15, mul_3, scan_output_2, silu_1, scan_output_3, transpose_10, contextualized_states], Original ATen: [aten._unsafe_view, aten.transpose, aten.split, aten.squeeze, aten.unsqueeze, aten.convolution, aten.slice, aten.silu, aten.view, aten.mul, aten.add, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:10
            extern_kernels.mm(reinterpret_tensor(buf9, (32, 1536), (1, 32), 0), reinterpret_tensor(arg9_1, (1536, 768), (1, 1536), 0), out=buf10)
            del arg9_1
            del buf9
        return (reinterpret_tensor(buf10, (1, 32, 768), (24576, 768, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 32, 768), (24576, 768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1536, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1536, 1, 4), (4, 4, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((80, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1536, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
