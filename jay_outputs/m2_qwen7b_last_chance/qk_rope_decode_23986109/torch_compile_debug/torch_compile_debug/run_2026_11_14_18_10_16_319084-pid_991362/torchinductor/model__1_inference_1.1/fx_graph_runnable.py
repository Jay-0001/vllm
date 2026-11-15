s30 = 4


import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/u/jsaravananindira/vllm/jay_outputs/m2_qwen7b_last_chance/qk_rope_decode_23986109/inductor_cache'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/u/jsaravananindira/vllm/jay_outputs/m2_qwen7b_last_chance/qk_rope_decode_23986109/torch_compile_debug'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TRITON_CACHE_AUTOTUNING'] = '1'
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
torch._inductor.config.deterministic = False
torch._inductor.config.triton.store_cubin = False
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



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
        split_with_sizes = torch.ops.aten.split_with_sizes.default(arg1_1, [3584, 512, 512], -1);  arg1_1 = None
        getitem = split_with_sizes[0]
        getitem_1 = split_with_sizes[1]
        getitem_2 = split_with_sizes[2];  split_with_sizes = None
        index = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
        split = torch.ops.aten.split.Tensor(index, 64, -1);  index = None
        getitem_3 = split[0]
        getitem_4 = split[1];  split = None
        view = torch.ops.aten.view.default(getitem, [arg0_1, -1, 128]);  getitem = None
        unsqueeze = torch.ops.aten.unsqueeze.default(getitem_3, -2)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(getitem_4, -2)
        split_1 = torch.ops.aten.split.Tensor(view, 64, -1);  view = None
        getitem_5 = split_1[0]
        getitem_6 = split_1[1];  split_1 = None
        mul_25 = torch.ops.aten.mul.Tensor(getitem_5, unsqueeze)
        mul_28 = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze_1)
        sub_15 = torch.ops.aten.sub.Tensor(mul_25, mul_28);  mul_25 = mul_28 = None
        mul_33 = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze);  getitem_6 = unsqueeze = None
        mul_36 = torch.ops.aten.mul.Tensor(getitem_5, unsqueeze_1);  getitem_5 = unsqueeze_1 = None
        add_64 = torch.ops.aten.add.Tensor(mul_33, mul_36);  mul_33 = mul_36 = None
        cat = torch.ops.aten.cat.default([sub_15, add_64], -1);  sub_15 = add_64 = None
        view_1 = torch.ops.aten.view.default(cat, [arg2_1, 3584]);  cat = None
        view_2 = torch.ops.aten.view.default(getitem_1, [arg0_1, -1, 128]);  getitem_1 = arg0_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(getitem_3, -2);  getitem_3 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(getitem_4, -2);  getitem_4 = None
        split_2 = torch.ops.aten.split.Tensor(view_2, 64, -1);  view_2 = None
        getitem_7 = split_2[0]
        getitem_8 = split_2[1];  split_2 = None
        mul_64 = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze_2)
        mul_67 = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_3)
        sub_32 = torch.ops.aten.sub.Tensor(mul_64, mul_67);  mul_64 = mul_67 = None
        mul_72 = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_2);  getitem_8 = unsqueeze_2 = None
        mul_75 = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze_3);  getitem_7 = unsqueeze_3 = None
        add_126 = torch.ops.aten.add.Tensor(mul_72, mul_75);  mul_72 = mul_75 = None
        cat_1 = torch.ops.aten.cat.default([sub_32, add_126], -1);  sub_32 = add_126 = None
        view_3 = torch.ops.aten.view.default(cat_1, [arg2_1, 512]);  cat_1 = arg2_1 = None
        return (view_1, view_3, getitem_2)

def load_args(reader):
    reader.symint(4, expr='s30')  # arg0_1
    buf0 = reader.storage(None, 9216*s30, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (s30, 4608), dtype=torch.bfloat16, is_leaf=True)  # arg1_1
    reader.symint(4, expr='s30')  # arg2_1
    buf1 = reader.storage(None, 8*s30, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (s30,), dtype=torch.int64, is_leaf=True)  # arg3_1
    buf2 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (32768, 128), dtype=torch.bfloat16, is_leaf=True)  # arg4_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)