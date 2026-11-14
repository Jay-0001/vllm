s72 = 4096


import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'graph_breaks,recompiles'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/u/jsaravananindira/vllm/jay_outputs/m2_qwen7b_last_chance/compile_23985448/cache/torch_compile_cache/torch_aot_compile/332a226df3c0977300a9fe8d75a14235ddb168817982a6b35b1f59211c84bcb7/inductor_cache'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/u/jsaravananindira/vllm/jay_outputs/m2_qwen7b_last_chance/compile_23985448/debug'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TRITON_CACHE_AUTOTUNING'] = '1'
os.environ['TRITON_PTXAS_BLACKWELL_PATH'] = '/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2/bin/ptxas'
os.environ['TRITON_CACHE_DIR'] = '/u/jsaravananindira/vllm/jay_outputs/m2_qwen7b_last_chance/compile_23985448/cache/torch_compile_cache/torch_aot_compile/332a226df3c0977300a9fe8d75a14235ddb168817982a6b35b1f59211c84bcb7/inductor_cache/triton/0'
os.environ.pop('TORCHDYNAMO_REPRO_AFTER', None)
os.environ.pop('TORCHDYNAMO_REPRO_LEVEL', None)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
import math
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.recompile_limit = 16
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.enable_cpp_symbolic_shape_guards = False
torch._dynamo.config.nested_graph_breaks = False
torch._dynamo.config.enable_aot_compile = True
torch._inductor.config.enable_auto_functionalized_v2 = False
torch._inductor.config.graph_partition = True
torch._inductor.config.deterministic = False
torch._inductor.config.combo_kernels = True
torch._inductor.config.benchmark_combo_kernel = True
torch._inductor.config.assume_32bit_indexing = False
torch._inductor.config.triton.autotune_at_compile_time = True
torch._inductor.config._cache_config_ignore_prefix = ['trace', 'cuda.cutlass_dir', 'cutlass.cutlass_dir', 'xpu.cutlass_dir', 'worker_start_method', 'compile_threads', 'post_grad_custom_post_pass', 'post_grad_custom_pre_pass', 'joint_custom_pre_pass', 'joint_custom_post_pass', 'pre_grad_custom_pass', '_fuse_ddp_communication_passes', '_pre_fusion_custom_pass', 'cudagraph_policy', 'force_disable_cudagraph_TESTING_ONLY', 'pre_grad_pass_timing', 'fx_graph_cache', 'fx_graph_remote_cache', 'autotune_local_cache', 'autotune_remote_cache', 'pre_grad_custom_pass']
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.triton.store_cubin = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.bundled_autograd_cache = True
torch._functorch.config.bypass_autograd_cache_key = True
torch._functorch.config.autograd_cache_normalize_inputs = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.force_non_lazy_backward_lowering = True
torch._functorch.config.strict_autograd_cache = True
torch._functorch.config.force_autograd_cache = True
torch._functorch.config.selective_decompose = False




isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.14.0+cu132
# torch cuda version: 13.2
# torch git version: 08187d9e0fba026dc8217405802ab5381dc88d90


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2026 NVIDIA Corporation 
# Built on Thu_Mar_19_11:12:51_PM_PDT_2026 
# Cuda compilation tools, release 13.2, V13.2.78 
# Build cuda_13.2.r13.2/compiler.37668154_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-40GB : 1 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1):
        view = torch.ops.aten.view.default(arg0_1, [-1, 3584]);  arg0_1 = None
        permute = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm = torch.ops.aten.mm.default(view, permute);  view = permute = None
        fused_add_rms_norm = torch.ops.vllm_ir.fused_add_rms_norm.default(mm, arg4_1, arg3_1, 1e-06);  mm = arg4_1 = arg3_1 = None
        getitem = fused_add_rms_norm[0]
        getitem_1 = fused_add_rms_norm[1];  fused_add_rms_norm = None
        permute_1 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_1 = torch.ops.aten.mm.default(getitem, permute_1);  getitem = permute_1 = None
        slice_1 = torch.ops.aten.slice.Tensor(mm_1, 1, 0, 18944)
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        neg = torch.ops.aten.neg.default(convert_element_type_4)
        exp = torch.ops.aten.exp.default(neg);  neg = None
        add_18 = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div = torch.ops.aten.div.Tensor(convert_element_type_4, add_18);  convert_element_type_4 = add_18 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        slice_2 = torch.ops.aten.slice.Tensor(mm_1, 1, 18944, 9223372036854775807);  mm_1 = None
        mul_16 = torch.ops.aten.mul.Tensor(convert_element_type_5, slice_2);  convert_element_type_5 = slice_2 = None
        permute_2 = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_2 = torch.ops.aten.mm.default(mul_16, permute_2);  mul_16 = permute_2 = None
        fused_add_rms_norm_1 = torch.ops.vllm_ir.fused_add_rms_norm.default(mm_2, getitem_1, arg7_1, 1e-06);  mm_2 = getitem_1 = arg7_1 = None
        getitem_2 = fused_add_rms_norm_1[0]
        getitem_3 = fused_add_rms_norm_1[1];  fused_add_rms_norm_1 = None
        permute_3 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm = torch.ops.aten.addmm.default(arg9_1, getitem_2, permute_3);  arg9_1 = getitem_2 = permute_3 = None
        split_with_sizes = torch.ops.aten.split_with_sizes.default(addmm, [3584, 512, 512], -1);  addmm = None
        getitem_4 = split_with_sizes[0]
        getitem_5 = split_with_sizes[1]
        getitem_6 = split_with_sizes[2];  split_with_sizes = None
        index = torch.ops.aten.index.Tensor(arg11_1, [arg10_1]);  arg11_1 = arg10_1 = None
        split = torch.ops.aten.split.Tensor(index, 64, -1);  index = None
        getitem_7 = split[0]
        getitem_8 = split[1];  split = None
        view_1 = torch.ops.aten.view.default(getitem_4, [arg1_1, -1, 128]);  getitem_4 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(getitem_7, -2)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(getitem_8, -2)
        split_1 = torch.ops.aten.split.Tensor(view_1, 64, -1);  view_1 = None
        getitem_9 = split_1[0]
        getitem_10 = split_1[1];  split_1 = None
        mul_54 = torch.ops.aten.mul.Tensor(getitem_9, unsqueeze)
        mul_57 = torch.ops.aten.mul.Tensor(getitem_10, unsqueeze_1)
        sub_28 = torch.ops.aten.sub.Tensor(mul_54, mul_57);  mul_54 = mul_57 = None
        mul_62 = torch.ops.aten.mul.Tensor(getitem_10, unsqueeze);  getitem_10 = unsqueeze = None
        mul_65 = torch.ops.aten.mul.Tensor(getitem_9, unsqueeze_1);  getitem_9 = unsqueeze_1 = None
        add_104 = torch.ops.aten.add.Tensor(mul_62, mul_65);  mul_62 = mul_65 = None
        cat = torch.ops.aten.cat.default([sub_28, add_104], -1);  sub_28 = add_104 = None
        view_3 = torch.ops.aten.view.default(getitem_5, [arg1_1, -1, 128]);  getitem_5 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(getitem_7, -2);  getitem_7 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(getitem_8, -2);  getitem_8 = None
        split_2 = torch.ops.aten.split.Tensor(view_3, 64, -1);  view_3 = None
        getitem_11 = split_2[0]
        getitem_12 = split_2[1];  split_2 = None
        mul_93 = torch.ops.aten.mul.Tensor(getitem_11, unsqueeze_2)
        mul_96 = torch.ops.aten.mul.Tensor(getitem_12, unsqueeze_3)
        sub_45 = torch.ops.aten.sub.Tensor(mul_93, mul_96);  mul_93 = mul_96 = None
        mul_101 = torch.ops.aten.mul.Tensor(getitem_12, unsqueeze_2);  getitem_12 = unsqueeze_2 = None
        mul_104 = torch.ops.aten.mul.Tensor(getitem_11, unsqueeze_3);  getitem_11 = unsqueeze_3 = None
        add_166 = torch.ops.aten.add.Tensor(mul_101, mul_104);  mul_101 = mul_104 = None
        cat_1 = torch.ops.aten.cat.default([sub_45, add_166], -1);  sub_45 = add_166 = None
        empty = torch.ops.aten.empty.memory_format([arg1_1, 3584], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False);  arg1_1 = None
        view_6 = torch.ops.aten.view.default(empty, [-1, 28, 128]);  empty = None
        view_8 = torch.ops.aten.view.default(getitem_6, [-1, 4, 128]);  getitem_6 = None
        return (cat_1, view_8, cat, view_6, getitem_3)

def load_args(reader):
    buf0 = reader.storage(None, 7168*s72, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (s72, 28, 128), dtype=torch.bfloat16, is_leaf=True)  # arg0_1
    reader.symint(4096, expr='s72')  # arg1_1
    buf1 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (3584, 3584), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
    buf2 = reader.storage(None, 7168, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (3584,), dtype=torch.bfloat16, is_leaf=True)  # arg3_1
    buf3 = reader.storage(None, 7168*s72, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (s72, 3584), dtype=torch.bfloat16, is_leaf=True)  # arg4_1
    buf4 = reader.storage(None, 271581184, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (37888, 3584), dtype=torch.bfloat16, is_leaf=True)  # arg5_1
    buf5 = reader.storage(None, 135790592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (3584, 18944), dtype=torch.bfloat16, is_leaf=True)  # arg6_1
    buf6 = reader.storage(None, 7168, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (3584,), dtype=torch.bfloat16, is_leaf=True)  # arg7_1
    buf7 = reader.storage(None, 33030144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (4608, 3584), dtype=torch.bfloat16, is_leaf=True)  # arg8_1
    buf8 = reader.storage(None, 9216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (4608,), dtype=torch.bfloat16, is_leaf=True)  # arg9_1
    buf9 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf9, (s72,), dtype=torch.int64, is_leaf=True)  # arg10_1
    buf10 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf10, (32768, 128), dtype=torch.bfloat16, is_leaf=True)  # arg11_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)