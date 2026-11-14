
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'graph_breaks,recompiles'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/u/jsaravananindira/vllm/jay_outputs/m2_convnext_scout/23967143_0_b1_layout1/cache'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/u/jsaravananindira/vllm/jay_outputs/m2_convnext_scout/23967143_0_b1_layout1/debug'
os.environ['TORCHINDUCTOR_LAYOUT_OPTIMIZATION'] = '1'
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
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
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



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1):
        convolution = torch.ops.aten.convolution.default(arg3_1, arg1_1, arg2_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 96);  arg1_1 = arg2_1 = None
        permute = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None
        convert_element_type = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view = torch.ops.aten.view.default(convert_element_type_1, [3136, 96]);  convert_element_type_1 = None
        permute_1 = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm = torch.ops.aten.addmm.default(arg7_1, view, permute_1);  arg7_1 = view = permute_1 = None
        view_1 = torch.ops.aten.view.default(addmm, [1, 56, 56, 384]);  addmm = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.5)
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.7071067811865476);  convert_element_type_5 = None
        erf = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_2 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_2, add_2);  mul_2 = add_2 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        view_2 = torch.ops.aten.view.default(convert_element_type_6, [3136, 384]);  convert_element_type_6 = None
        permute_2 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg9_1, view_2, permute_2);  arg9_1 = view_2 = permute_2 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [1, 56, 56, 96]);  addmm_1 = None
        permute_3 = torch.ops.aten.permute.default(view_3, [0, 3, 1, 2]);  view_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(arg0_1, permute_3);  arg0_1 = permute_3 = None
        add_3 = torch.ops.aten.add.Tensor(mul_5, arg3_1);  mul_5 = arg3_1 = None
        return (add_3,)

def load_args(reader):
    buf0 = reader.storage(None, 192, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (96, 1, 1), dtype=torch.bfloat16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 9408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (96, 1, 7, 7), (49, 1, 7, 1), dtype=torch.bfloat16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 192, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (96,), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 602112, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (1, 96, 56, 56), (301056, 1, 5376, 96), dtype=torch.bfloat16, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 192, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (96,), dtype=torch.bfloat16, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 192, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (96,), dtype=torch.bfloat16, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 73728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (384, 96), dtype=torch.bfloat16, is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 768, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (384,), dtype=torch.bfloat16, is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 73728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (96, 384), dtype=torch.bfloat16, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 192, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (96,), dtype=torch.bfloat16, is_leaf=True)  # arg9_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)