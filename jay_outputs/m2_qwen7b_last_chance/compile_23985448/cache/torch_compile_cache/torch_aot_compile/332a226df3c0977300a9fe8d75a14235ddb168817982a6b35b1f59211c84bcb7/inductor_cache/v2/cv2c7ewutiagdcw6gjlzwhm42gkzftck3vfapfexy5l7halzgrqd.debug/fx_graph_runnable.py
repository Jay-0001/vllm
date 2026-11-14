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



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg0_1, torch.int64);  arg0_1 = None
        embedding = torch.ops.aten.embedding.default(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        rms_norm = torch.ops.vllm_ir.rms_norm.default(embedding, arg3_1, 1e-06);  arg3_1 = None
        permute = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm = torch.ops.aten.addmm.default(arg5_1, rms_norm, permute);  arg5_1 = rms_norm = permute = None
        split_with_sizes = torch.ops.aten.split_with_sizes.default(addmm, [3584, 512, 512], -1);  addmm = None
        getitem = split_with_sizes[0]
        getitem_1 = split_with_sizes[1]
        getitem_2 = split_with_sizes[2];  split_with_sizes = None
        index = torch.ops.aten.index.Tensor(arg7_1, [arg6_1]);  arg7_1 = arg6_1 = None
        split = torch.ops.aten.split.Tensor(index, 64, -1);  index = None
        getitem_3 = split[0]
        getitem_4 = split[1];  split = None
        view = torch.ops.aten.view.default(getitem, [arg1_1, -1, 128]);  getitem = None
        unsqueeze = torch.ops.aten.unsqueeze.default(getitem_3, -2)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(getitem_4, -2)
        split_1 = torch.ops.aten.split.Tensor(view, 64, -1);  view = None
        getitem_5 = split_1[0]
        getitem_6 = split_1[1];  split_1 = None
        mul_32 = torch.ops.aten.mul.Tensor(getitem_5, unsqueeze)
        mul_35 = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze_1)
        sub_19 = torch.ops.aten.sub.Tensor(mul_32, mul_35);  mul_32 = mul_35 = None
        mul_40 = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze);  getitem_6 = unsqueeze = None
        mul_43 = torch.ops.aten.mul.Tensor(getitem_5, unsqueeze_1);  getitem_5 = unsqueeze_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_40, mul_43);  mul_40 = mul_43 = None
        cat = torch.ops.aten.cat.default([sub_19, add_75], -1);  sub_19 = add_75 = None
        view_2 = torch.ops.aten.view.default(getitem_1, [arg1_1, -1, 128]);  getitem_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(getitem_3, -2);  getitem_3 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(getitem_4, -2);  getitem_4 = None
        split_2 = torch.ops.aten.split.Tensor(view_2, 64, -1);  view_2 = None
        getitem_7 = split_2[0]
        getitem_8 = split_2[1];  split_2 = None
        mul_71 = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze_2)
        mul_74 = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_3)
        sub_36 = torch.ops.aten.sub.Tensor(mul_71, mul_74);  mul_71 = mul_74 = None
        mul_79 = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_2);  getitem_8 = unsqueeze_2 = None
        mul_82 = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze_3);  getitem_7 = unsqueeze_3 = None
        add_137 = torch.ops.aten.add.Tensor(mul_79, mul_82);  mul_79 = mul_82 = None
        cat_1 = torch.ops.aten.cat.default([sub_36, add_137], -1);  sub_36 = add_137 = None
        empty = torch.ops.aten.empty.memory_format([arg1_1, 3584], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False);  arg1_1 = None
        view_5 = torch.ops.aten.view.default(empty, [-1, 28, 128]);  empty = None
        view_7 = torch.ops.aten.view.default(getitem_2, [-1, 4, 128]);  getitem_2 = None
        return (cat_1, view_7, cat, view_5, embedding)

def load_args(reader):
    buf0 = reader.storage(None, 16384, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf0, (s72,), dtype=torch.int32, is_leaf=True)  # arg0_1
    reader.symint(4096, expr='s72')  # arg1_1
    buf1 = reader.storage(None, 1089994752, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (152064, 3584), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
    buf2 = reader.storage(None, 7168, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (3584,), dtype=torch.bfloat16, is_leaf=True)  # arg3_1
    buf3 = reader.storage(None, 33030144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (4608, 3584), dtype=torch.bfloat16, is_leaf=True)  # arg4_1
    buf4 = reader.storage(None, 9216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (4608,), dtype=torch.bfloat16, is_leaf=True)  # arg5_1
    buf5 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf5, (s72,), dtype=torch.int64, is_leaf=True)  # arg6_1
    buf6 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (32768, 128), dtype=torch.bfloat16, is_leaf=True)  # arg7_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)