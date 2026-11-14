
import os
os.environ['TORCHINDUCTOR_BENCHMARK_FUSION'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'graph_breaks,recompiles'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/u/jsaravananindira/vllm/jay_outputs/m2_diffusion_scout/23985088_1_b1_fusion1/cache'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/u/jsaravananindira/vllm/jay_outputs/m2_diffusion_scout/23985088_1_b1_fusion1/debug'
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

torch._inductor.config.deterministic = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.enabled = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
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



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1):
        clone = torch.ops.aten.clone.default(arg2_1, memory_format = torch.contiguous_format)
        convert_element_type = torch.ops.prims.convert_element_type.default(clone, torch.float32);  clone = None
        view = torch.ops.aten.view.default(convert_element_type, [1, 32, 16, 256]);  convert_element_type = None
        var_mean = torch.ops.aten.var_mean.correction(view, [2, 3], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub = torch.ops.aten.sub.Tensor(view, getitem_1);  view = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        view_1 = torch.ops.aten.view.default(mul, [1, 512, 16, 16]);  mul = None
        unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(view_1, unsqueeze_2);  view_1 = unsqueeze_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        neg = torch.ops.aten.neg.default(convert_element_type_4)
        exp = torch.ops.aten.exp.default(neg);  neg = None
        add_2 = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div = torch.ops.aten.div.Tensor(convert_element_type_4, add_2);  convert_element_type_4 = add_2 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        convolution = torch.ops.aten.convolution.default(convert_element_type_5, arg3_1, arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_5 = arg3_1 = arg4_1 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        neg_1 = torch.ops.aten.neg.default(convert_element_type_6)
        exp_1 = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_3 = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1 = torch.ops.aten.div.Tensor(convert_element_type_6, add_3);  convert_element_type_6 = add_3 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        permute = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm = torch.ops.aten.addmm.default(arg7_1, convert_element_type_7, permute);  arg7_1 = convert_element_type_7 = permute = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(addmm, 2);  addmm = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, 3);  unsqueeze_6 = None
        add_4 = torch.ops.aten.add.Tensor(convolution, unsqueeze_7);  convolution = unsqueeze_7 = None
        clone_1 = torch.ops.aten.clone.default(add_4, memory_format = torch.contiguous_format);  add_4 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(clone_1, torch.float32);  clone_1 = None
        view_2 = torch.ops.aten.view.default(convert_element_type_11, [1, 32, 8, 256]);  convert_element_type_11 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(view_2, [2, 3], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1 = torch.ops.aten.sub.Tensor(view_2, getitem_3);  view_2 = getitem_3 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        view_3 = torch.ops.aten.view.default(mul_2, [1, 256, 16, 16]);  mul_2 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(arg8_1, 0);  arg8_1 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(unsqueeze_9, 3);  unsqueeze_9 = None
        mul_3 = torch.ops.aten.mul.Tensor(view_3, unsqueeze_10);  view_3 = unsqueeze_10 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(arg9_1, 0);  arg9_1 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(unsqueeze_11, 2);  unsqueeze_11 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, 3);  unsqueeze_12 = None
        add_6 = torch.ops.aten.add.Tensor(mul_3, unsqueeze_13);  mul_3 = unsqueeze_13 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(convert_element_type_12, torch.float32);  convert_element_type_12 = None
        neg_2 = torch.ops.aten.neg.default(convert_element_type_15)
        exp_2 = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_7 = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2 = torch.ops.aten.div.Tensor(convert_element_type_15, add_7);  convert_element_type_15 = add_7 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        convolution_1 = torch.ops.aten.convolution.default(convert_element_type_16, arg10_1, arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_16 = arg10_1 = arg11_1 = None
        convolution_2 = torch.ops.aten.convolution.default(arg2_1, arg12_1, arg13_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg12_1 = arg13_1 = None
        add_8 = torch.ops.aten.add.Tensor(convolution_2, convolution_1);  convolution_2 = convolution_1 = None
        return (add_8,)

def load_args(reader):
    buf0 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (512,), dtype=torch.bfloat16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (512,), dtype=torch.bfloat16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (1, 512, 16, 16), (131072, 1, 8192, 512), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 2359296, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (256, 512, 3, 3), (4608, 1, 1536, 512), dtype=torch.bfloat16, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (256,), dtype=torch.bfloat16, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (1, 512), dtype=torch.bfloat16, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (256, 512), dtype=torch.bfloat16, is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (256,), dtype=torch.bfloat16, is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (256,), dtype=torch.bfloat16, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (256,), dtype=torch.bfloat16, is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf10, (256, 256, 3, 3), (2304, 1, 768, 256), dtype=torch.bfloat16, is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf11, (256,), dtype=torch.bfloat16, is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf12, (256, 512, 1, 1), (512, 1, 512, 512), dtype=torch.bfloat16, is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf13, (256,), dtype=torch.bfloat16, is_leaf=True)  # arg13_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)