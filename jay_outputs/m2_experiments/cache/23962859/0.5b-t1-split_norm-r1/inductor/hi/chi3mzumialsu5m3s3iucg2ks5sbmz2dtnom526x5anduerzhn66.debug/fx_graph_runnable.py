
import os
os.environ['TRITON_CACHE_DIR'] = '/u/jsaravananindira/vllm/m2_experiments/cache/23962859/0.5b-t1-split_norm-r1/triton'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/u/jsaravananindira/vllm/m2_experiments/cache/23962859/0.5b-t1-split_norm-r1/inductor'
os.environ['TORCH_LOGS'] = 'fusion,recompiles'
os.environ['TORCHINDUCTOR_BENCHMARK_FUSION'] = '0'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/u/jsaravananindira/vllm/m2_experiments/cache/23962859/0.5b-t1-split_norm-r1/debug'
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



    def forward(self, arg0_1, arg1_1, arg2_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        add = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(add, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1 = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul = torch.ops.aten.mul.Tensor(add, rsqrt);  rsqrt = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_2, arg2_1);  convert_element_type_2 = arg2_1 = None
        return (mul_1, add)

def load_args(reader):
    buf0 = reader.storage(None, 1792, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (1, 896), dtype=torch.bfloat16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1792, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (1, 896), dtype=torch.bfloat16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1792, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (896,), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)