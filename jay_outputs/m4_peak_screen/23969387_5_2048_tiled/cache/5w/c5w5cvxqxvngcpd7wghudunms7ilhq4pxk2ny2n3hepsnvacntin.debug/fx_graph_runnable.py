
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'graph_breaks,recompiles'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/u/jsaravananindira/vllm-m4/jay_outputs/m4_peak_screen/23969387_5_2048_tiled/cache'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/u/jsaravananindira/vllm-m4/jay_outputs/m4_peak_screen/23969387_5_2048_tiled/debug'
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



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        full_default = torch.ops.aten.full.default([2048, 3584], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None
        slice_1 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 2048)
        permute = torch.ops.aten.permute.default(slice_1, [1, 0]);  slice_1 = None
        mm = torch.ops.aten.mm.default(arg1_1, permute);  permute = None
        slice_2 = torch.ops.aten.slice.Tensor(arg2_1, 0, 0, 2048)
        permute_1 = torch.ops.aten.permute.default(slice_2, [1, 0]);  slice_2 = None
        mm_1 = torch.ops.aten.mm.default(arg1_1, permute_1);  permute_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        neg = torch.ops.aten.neg.default(convert_element_type_4)
        exp = torch.ops.aten.exp.default(neg);  neg = None
        add = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div = torch.ops.aten.div.Tensor(convert_element_type_4, add);  convert_element_type_4 = add = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type_5, mm_1);  convert_element_type_5 = mm_1 = None
        slice_3 = torch.ops.aten.slice.Tensor(arg3_1, 0, 0, 2048)
        mm_2 = torch.ops.aten.mm.default(mul, slice_3);  mul = slice_3 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        slice_4 = torch.ops.aten.slice.Tensor(arg0_1, 0, 2048, 4096)
        permute_2 = torch.ops.aten.permute.default(slice_4, [1, 0]);  slice_4 = None
        mm_3 = torch.ops.aten.mm.default(arg1_1, permute_2);  permute_2 = None
        slice_5 = torch.ops.aten.slice.Tensor(arg2_1, 0, 2048, 4096)
        permute_3 = torch.ops.aten.permute.default(slice_5, [1, 0]);  slice_5 = None
        mm_4 = torch.ops.aten.mm.default(arg1_1, permute_3);  permute_3 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        neg_1 = torch.ops.aten.neg.default(convert_element_type_13)
        exp_1 = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_2 = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1 = torch.ops.aten.div.Tensor(convert_element_type_13, add_2);  convert_element_type_13 = add_2 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_14, mm_4);  convert_element_type_14 = mm_4 = None
        slice_6 = torch.ops.aten.slice.Tensor(arg3_1, 0, 2048, 4096)
        mm_5 = torch.ops.aten.mm.default(mul_1, slice_6);  mul_1 = slice_6 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        add_3 = torch.ops.aten.add.Tensor(convert_element_type_8, convert_element_type_17);  convert_element_type_8 = convert_element_type_17 = None
        slice_7 = torch.ops.aten.slice.Tensor(arg0_1, 0, 4096, 6144)
        permute_4 = torch.ops.aten.permute.default(slice_7, [1, 0]);  slice_7 = None
        mm_6 = torch.ops.aten.mm.default(arg1_1, permute_4);  permute_4 = None
        slice_8 = torch.ops.aten.slice.Tensor(arg2_1, 0, 4096, 6144)
        permute_5 = torch.ops.aten.permute.default(slice_8, [1, 0]);  slice_8 = None
        mm_7 = torch.ops.aten.mm.default(arg1_1, permute_5);  permute_5 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        neg_2 = torch.ops.aten.neg.default(convert_element_type_22)
        exp_2 = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_4 = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2 = torch.ops.aten.div.Tensor(convert_element_type_22, add_4);  convert_element_type_22 = add_4 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_23, mm_7);  convert_element_type_23 = mm_7 = None
        slice_9 = torch.ops.aten.slice.Tensor(arg3_1, 0, 4096, 6144)
        mm_8 = torch.ops.aten.mm.default(mul_2, slice_9);  mul_2 = slice_9 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        add_5 = torch.ops.aten.add.Tensor(add_3, convert_element_type_26);  add_3 = convert_element_type_26 = None
        slice_10 = torch.ops.aten.slice.Tensor(arg0_1, 0, 6144, 8192)
        permute_6 = torch.ops.aten.permute.default(slice_10, [1, 0]);  slice_10 = None
        mm_9 = torch.ops.aten.mm.default(arg1_1, permute_6);  permute_6 = None
        slice_11 = torch.ops.aten.slice.Tensor(arg2_1, 0, 6144, 8192)
        permute_7 = torch.ops.aten.permute.default(slice_11, [1, 0]);  slice_11 = None
        mm_10 = torch.ops.aten.mm.default(arg1_1, permute_7);  permute_7 = None
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        neg_3 = torch.ops.aten.neg.default(convert_element_type_31)
        exp_3 = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_6 = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3 = torch.ops.aten.div.Tensor(convert_element_type_31, add_6);  convert_element_type_31 = add_6 = None
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_32, mm_10);  convert_element_type_32 = mm_10 = None
        slice_12 = torch.ops.aten.slice.Tensor(arg3_1, 0, 6144, 8192)
        mm_11 = torch.ops.aten.mm.default(mul_3, slice_12);  mul_3 = slice_12 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        add_7 = torch.ops.aten.add.Tensor(add_5, convert_element_type_35);  add_5 = convert_element_type_35 = None
        slice_13 = torch.ops.aten.slice.Tensor(arg0_1, 0, 8192, 10240)
        permute_8 = torch.ops.aten.permute.default(slice_13, [1, 0]);  slice_13 = None
        mm_12 = torch.ops.aten.mm.default(arg1_1, permute_8);  permute_8 = None
        slice_14 = torch.ops.aten.slice.Tensor(arg2_1, 0, 8192, 10240)
        permute_9 = torch.ops.aten.permute.default(slice_14, [1, 0]);  slice_14 = None
        mm_13 = torch.ops.aten.mm.default(arg1_1, permute_9);  permute_9 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        neg_4 = torch.ops.aten.neg.default(convert_element_type_40)
        exp_4 = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_8 = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4 = torch.ops.aten.div.Tensor(convert_element_type_40, add_8);  convert_element_type_40 = add_8 = None
        convert_element_type_41 = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_41, mm_13);  convert_element_type_41 = mm_13 = None
        slice_15 = torch.ops.aten.slice.Tensor(arg3_1, 0, 8192, 10240)
        mm_14 = torch.ops.aten.mm.default(mul_4, slice_15);  mul_4 = slice_15 = None
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        add_9 = torch.ops.aten.add.Tensor(add_7, convert_element_type_44);  add_7 = convert_element_type_44 = None
        slice_16 = torch.ops.aten.slice.Tensor(arg0_1, 0, 10240, 12288)
        permute_10 = torch.ops.aten.permute.default(slice_16, [1, 0]);  slice_16 = None
        mm_15 = torch.ops.aten.mm.default(arg1_1, permute_10);  permute_10 = None
        slice_17 = torch.ops.aten.slice.Tensor(arg2_1, 0, 10240, 12288)
        permute_11 = torch.ops.aten.permute.default(slice_17, [1, 0]);  slice_17 = None
        mm_16 = torch.ops.aten.mm.default(arg1_1, permute_11);  permute_11 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        neg_5 = torch.ops.aten.neg.default(convert_element_type_49)
        exp_5 = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_10 = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5 = torch.ops.aten.div.Tensor(convert_element_type_49, add_10);  convert_element_type_49 = add_10 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_50, mm_16);  convert_element_type_50 = mm_16 = None
        slice_18 = torch.ops.aten.slice.Tensor(arg3_1, 0, 10240, 12288)
        mm_17 = torch.ops.aten.mm.default(mul_5, slice_18);  mul_5 = slice_18 = None
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        add_11 = torch.ops.aten.add.Tensor(add_9, convert_element_type_53);  add_9 = convert_element_type_53 = None
        slice_19 = torch.ops.aten.slice.Tensor(arg0_1, 0, 12288, 14336)
        permute_12 = torch.ops.aten.permute.default(slice_19, [1, 0]);  slice_19 = None
        mm_18 = torch.ops.aten.mm.default(arg1_1, permute_12);  permute_12 = None
        slice_20 = torch.ops.aten.slice.Tensor(arg2_1, 0, 12288, 14336)
        permute_13 = torch.ops.aten.permute.default(slice_20, [1, 0]);  slice_20 = None
        mm_19 = torch.ops.aten.mm.default(arg1_1, permute_13);  permute_13 = None
        convert_element_type_58 = torch.ops.prims.convert_element_type.default(mm_18, torch.float32);  mm_18 = None
        neg_6 = torch.ops.aten.neg.default(convert_element_type_58)
        exp_6 = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_12 = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6 = torch.ops.aten.div.Tensor(convert_element_type_58, add_12);  convert_element_type_58 = add_12 = None
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_59, mm_19);  convert_element_type_59 = mm_19 = None
        slice_21 = torch.ops.aten.slice.Tensor(arg3_1, 0, 12288, 14336)
        mm_20 = torch.ops.aten.mm.default(mul_6, slice_21);  mul_6 = slice_21 = None
        convert_element_type_62 = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        add_13 = torch.ops.aten.add.Tensor(add_11, convert_element_type_62);  add_11 = convert_element_type_62 = None
        slice_22 = torch.ops.aten.slice.Tensor(arg0_1, 0, 14336, 16384)
        permute_14 = torch.ops.aten.permute.default(slice_22, [1, 0]);  slice_22 = None
        mm_21 = torch.ops.aten.mm.default(arg1_1, permute_14);  permute_14 = None
        slice_23 = torch.ops.aten.slice.Tensor(arg2_1, 0, 14336, 16384)
        permute_15 = torch.ops.aten.permute.default(slice_23, [1, 0]);  slice_23 = None
        mm_22 = torch.ops.aten.mm.default(arg1_1, permute_15);  permute_15 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        neg_7 = torch.ops.aten.neg.default(convert_element_type_67)
        exp_7 = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_14 = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7 = torch.ops.aten.div.Tensor(convert_element_type_67, add_14);  convert_element_type_67 = add_14 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_68, mm_22);  convert_element_type_68 = mm_22 = None
        slice_24 = torch.ops.aten.slice.Tensor(arg3_1, 0, 14336, 16384)
        mm_23 = torch.ops.aten.mm.default(mul_7, slice_24);  mul_7 = slice_24 = None
        convert_element_type_71 = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        add_15 = torch.ops.aten.add.Tensor(add_13, convert_element_type_71);  add_13 = convert_element_type_71 = None
        slice_25 = torch.ops.aten.slice.Tensor(arg0_1, 0, 16384, 18432)
        permute_16 = torch.ops.aten.permute.default(slice_25, [1, 0]);  slice_25 = None
        mm_24 = torch.ops.aten.mm.default(arg1_1, permute_16);  permute_16 = None
        slice_26 = torch.ops.aten.slice.Tensor(arg2_1, 0, 16384, 18432)
        permute_17 = torch.ops.aten.permute.default(slice_26, [1, 0]);  slice_26 = None
        mm_25 = torch.ops.aten.mm.default(arg1_1, permute_17);  permute_17 = None
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        neg_8 = torch.ops.aten.neg.default(convert_element_type_76)
        exp_8 = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_16 = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8 = torch.ops.aten.div.Tensor(convert_element_type_76, add_16);  convert_element_type_76 = add_16 = None
        convert_element_type_77 = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_77, mm_25);  convert_element_type_77 = mm_25 = None
        slice_27 = torch.ops.aten.slice.Tensor(arg3_1, 0, 16384, 18432)
        mm_26 = torch.ops.aten.mm.default(mul_8, slice_27);  mul_8 = slice_27 = None
        convert_element_type_80 = torch.ops.prims.convert_element_type.default(mm_26, torch.float32);  mm_26 = None
        add_17 = torch.ops.aten.add.Tensor(add_15, convert_element_type_80);  add_15 = convert_element_type_80 = None
        slice_28 = torch.ops.aten.slice.Tensor(arg0_1, 0, 18432, 18944);  arg0_1 = None
        permute_18 = torch.ops.aten.permute.default(slice_28, [1, 0]);  slice_28 = None
        mm_27 = torch.ops.aten.mm.default(arg1_1, permute_18);  permute_18 = None
        slice_29 = torch.ops.aten.slice.Tensor(arg2_1, 0, 18432, 18944);  arg2_1 = None
        permute_19 = torch.ops.aten.permute.default(slice_29, [1, 0]);  slice_29 = None
        mm_28 = torch.ops.aten.mm.default(arg1_1, permute_19);  arg1_1 = permute_19 = None
        convert_element_type_85 = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        neg_9 = torch.ops.aten.neg.default(convert_element_type_85)
        exp_9 = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_18 = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9 = torch.ops.aten.div.Tensor(convert_element_type_85, add_18);  convert_element_type_85 = add_18 = None
        convert_element_type_86 = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        mul_9 = torch.ops.aten.mul.Tensor(convert_element_type_86, mm_28);  convert_element_type_86 = mm_28 = None
        slice_30 = torch.ops.aten.slice.Tensor(arg3_1, 0, 18432, 18944);  arg3_1 = None
        mm_29 = torch.ops.aten.mm.default(mul_9, slice_30);  mul_9 = slice_30 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        add_19 = torch.ops.aten.add.Tensor(add_17, convert_element_type_89);  add_17 = convert_element_type_89 = None
        convert_element_type_90 = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        return (convert_element_type_90,)

def load_args(reader):
    buf0 = reader.storage(None, 135790592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (18944, 3584), dtype=torch.bfloat16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 14680064, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (2048, 3584), dtype=torch.bfloat16, is_leaf=True)  # arg1_1
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