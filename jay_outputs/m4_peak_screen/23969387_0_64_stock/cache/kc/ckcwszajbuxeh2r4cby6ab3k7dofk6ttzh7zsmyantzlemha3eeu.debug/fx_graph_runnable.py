
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'graph_breaks,recompiles'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/u/jsaravananindira/vllm-m4/jay_outputs/m4_peak_screen/23969387_0_64_stock/cache'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/u/jsaravananindira/vllm-m4/jay_outputs/m4_peak_screen/23969387_0_64_stock/debug'
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



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        permute = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm = torch.ops.aten.mm.default(arg1_1, permute);  permute = None
        permute_1 = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm_1 = torch.ops.aten.mm.default(arg1_1, permute_1);  arg1_1 = permute_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        neg = torch.ops.aten.neg.default(convert_element_type_4)
        exp = torch.ops.aten.exp.default(neg);  neg = None
        add = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div = torch.ops.aten.div.Tensor(convert_element_type_4, add);  convert_element_type_4 = add = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type_5, mm_1);  convert_element_type_5 = mm_1 = None
        mm_2 = torch.ops.aten.mm.default(mul, arg3_1);  mul = arg3_1 = None
        return (mm_2,)

def load_args(reader):
    buf0 = reader.storage(None, 135790592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (18944, 3584), dtype=torch.bfloat16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 458752, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (64, 3584), dtype=torch.bfloat16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 135790592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (18944, 3584), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 135790592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (18944, 3584), dtype=torch.bfloat16, is_leaf=True)  # arg3_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)