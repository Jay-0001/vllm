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


# kernel path: /u/jsaravananindira/vllm/m2_experiments/cache/23962859/0.5b-t1-split_activation-r1/inductor/is/cisgplk7ach4lzeu7l6g5t4swoyyapa2wrdiixuumbkklfx3xopi.py
# Topologically Sorted Source Nodes: [float_1, float_2, updated, square, mean, add_1, scale, mul, to, normalized], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_1 => add_1
#   float_1 => convert_element_type
#   float_2 => convert_element_type_1
#   mean => mean
#   mul => mul
#   normalized => mul_1
#   scale => rsqrt
#   square => pow_1
#   to => convert_element_type_2
#   updated => add
# Graph fragment:
#   %arg0_1 : Tensor "bf16[1, 896][896, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[1, 896][896, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %add : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=add]
#   %buf1 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=buf1]
#   %arg2_1 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convert_element_type : Tensor "f32[1, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[1, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %add : Tensor "f32[1, 896][896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, %convert_element_type_1), kwargs = {})
#   %pow_1 : Tensor "f32[1, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : Tensor "f32[1, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_1 : Tensor "bf16[1, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %arg2_1), kwargs = {})
#   return %add,%buf1,%mul_1
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'r0_': 16128}, 'backend_hash': '6AD75AB4E17FA69A29C4138A72623B20ADA7CF527C65714535CC47B5D5DCFE91', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
        assert_size_stride_grouped((arg0_1, arg1_1, arg2_1), ((1, 896), (1, 896), (896, )), ((896, 1), (896, 1), (1, )), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf0 = empty_strided_cuda((1, 896), (896, 1), torch.float32)
            buf2 = empty_strided_cuda((1, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_1, float_2, updated, square, mean, add_1, scale, mul, to, normalized], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0:1
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0.run(arg0_1, arg1_1, arg2_1, buf0, buf2, 1, 896, stream=raw_stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            assert_size_stride(arg3_1, (896, 9728), (9728, 1), 'input')
            arg3_1 = copy_if_misaligned(arg3_1)
            buf3 = empty_strided_cuda((1, 9728), (9728, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [square, mean, add_1, scale, mul, to, normalized, matmul], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(buf2, arg3_1, out=buf3)
            del arg3_1
            del buf2
        return (buf3, buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((1, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((896, 9728), (9728, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
