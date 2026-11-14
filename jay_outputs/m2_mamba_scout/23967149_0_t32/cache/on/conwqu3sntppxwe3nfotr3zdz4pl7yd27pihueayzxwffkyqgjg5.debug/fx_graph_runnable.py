
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'graph_breaks,recompiles'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/u/jsaravananindira/vllm/jay_outputs/m2_mamba_scout/23967149_0_t32/cache'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/u/jsaravananindira/vllm/jay_outputs/m2_mamba_scout/23967149_0_t32/debug'
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
torch._dynamo.config.nested_graph_breaks = False
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
class associative_scan_combine_graph_0_0(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        mul = torch.ops.aten.mul.Tensor(arg0_1, arg2_1);  arg0_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(arg2_1, arg1_1);  arg2_1 = arg1_1 = None
        add = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        return [mul, add]


class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.associative_scan_combine_graph_0 = associative_scan_combine_graph_0_0()



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1):
        permute = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view = torch.ops.aten.view.default(arg0_1, [32, 768]);  arg0_1 = None
        mm = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1 = torch.ops.aten.view.default(mm, [1, 32, 3072]);  mm = None
        permute_1 = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        exp = torch.ops.aten.exp.default(arg2_1);  arg2_1 = None
        neg = torch.ops.aten.neg.default(exp);  exp = None
        split = torch.ops.aten.split.Tensor(permute_1, 1536, 1);  permute_1 = None
        getitem = split[0]
        getitem_1 = split[1];  split = None
        squeeze = torch.ops.aten.squeeze.dim(arg3_1, 1);  arg3_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(squeeze, 1);  squeeze = None
        convolution = torch.ops.aten.convolution.default(getitem, unsqueeze, arg4_1, [1], [3], [1], False, [0], 1536);  getitem = unsqueeze = arg4_1 = None
        slice_1 = torch.ops.aten.slice.Tensor(convolution, 2, 0, 32);  convolution = None
        neg_1 = torch.ops.aten.neg.default(slice_1)
        exp_1 = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div = torch.ops.aten.div.Tensor(slice_1, add);  slice_1 = add = None
        permute_2 = torch.ops.aten.permute.default(div, [0, 2, 1]);  div = None
        permute_3 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        view_2 = torch.ops.aten.view.default(permute_2, [32, 1536])
        mm_1 = torch.ops.aten.mm.default(view_2, permute_3);  view_2 = permute_3 = None
        view_3 = torch.ops.aten.view.default(mm_1, [1, 32, 80]);  mm_1 = None
        split_with_sizes = torch.ops.aten.split_with_sizes.default(view_3, [48, 16, 16], -1);  view_3 = None
        getitem_2 = split_with_sizes[0]
        getitem_3 = split_with_sizes[1]
        getitem_4 = split_with_sizes[2];  split_with_sizes = None
        permute_6 = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        view_4 = torch.ops.aten.view.default(getitem_2, [32, 48]);  getitem_2 = None
        mm_2 = torch.ops.aten.mm.default(view_4, permute_6);  view_4 = permute_6 = None
        view_5 = torch.ops.aten.view.default(mm_2, [1, 32, 1536]);  mm_2 = None
        permute_7 = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
        clone = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        permute_8 = torch.ops.aten.permute.default(permute_2, [0, 2, 1]);  permute_2 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(arg7_1, 1);  arg7_1 = None
        add_1 = torch.ops.aten.add.Tensor(clone, unsqueeze_1);  clone = unsqueeze_1 = None
        exp_2 = torch.ops.aten.exp.default(add_1)
        log1p = torch.ops.aten.log1p.default(exp_2);  exp_2 = None
        gt = torch.ops.aten.gt.Scalar(add_1, 20)
        where = torch.ops.aten.where.self(gt, add_1, log1p);  gt = add_1 = log1p = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(neg, 0);  neg = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where, 3)
        mul = torch.ops.aten.mul.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
        exp_3 = torch.ops.aten.exp.default(mul);  mul = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(where, 3);  where = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(getitem_3, 1);  getitem_3 = None
        mul_1 = torch.ops.aten.mul.Tensor(unsqueeze_5, unsqueeze_6);  unsqueeze_5 = unsqueeze_6 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(permute_8, 3)
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_7);  mul_1 = unsqueeze_7 = None
        permute_13 = torch.ops.aten.permute.default(exp_3, [2, 0, 1, 3]);  exp_3 = None
        permute_14 = torch.ops.aten.permute.default(mul_2, [2, 0, 1, 3]);  mul_2 = None
        associative_scan_combine_graph_0 = self.associative_scan_combine_graph_0
        associative_scan = torch.ops.higher_order.associative_scan(associative_scan_combine_graph_0, [permute_13, permute_14], ());  associative_scan_combine_graph_0 = permute_13 = permute_14 = None
        getitem_6 = associative_scan[1];  associative_scan = None
        permute_16 = torch.ops.aten.permute.default(getitem_6, [1, 2, 0, 3]);  getitem_6 = None
        permute_17 = torch.ops.aten.permute.default(permute_16, [0, 2, 1, 3]);  permute_16 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(getitem_4, -1);  getitem_4 = None
        expand = torch.ops.aten.expand.default(permute_17, [1, 32, 1536, 16]);  permute_17 = None
        view_6 = torch.ops.aten.view.default(expand, [32, 1536, 16]);  expand = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_8, [1, 32, 16, 1]);  unsqueeze_8 = None
        view_7 = torch.ops.aten.view.default(expand_1, [32, 16, 1]);  expand_1 = None
        bmm = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = view_7 = None
        view_8 = torch.ops.aten.view.default(bmm, [1, 32, 1536, 1]);  bmm = None
        squeeze_1 = torch.ops.aten.squeeze.dim(view_8, -1);  view_8 = None
        permute_18 = torch.ops.aten.permute.default(squeeze_1, [0, 2, 1]);  squeeze_1 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(arg8_1, 0);  arg8_1 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        mul_3 = torch.ops.aten.mul.Tensor(permute_8, unsqueeze_10);  permute_8 = unsqueeze_10 = None
        add_2 = torch.ops.aten.add.Tensor(permute_18, mul_3);  permute_18 = mul_3 = None
        neg_2 = torch.ops.aten.neg.default(getitem_1)
        exp_4 = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_3 = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_1 = torch.ops.aten.div.Tensor(getitem_1, add_3);  getitem_1 = add_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(add_2, div_1);  add_2 = div_1 = None
        permute_19 = torch.ops.aten.permute.default(mul_4, [0, 2, 1]);  mul_4 = None
        permute_20 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        view_9 = torch.ops.aten.view.default(permute_19, [32, 1536]);  permute_19 = None
        mm_3 = torch.ops.aten.mm.default(view_9, permute_20);  view_9 = permute_20 = None
        view_10 = torch.ops.aten.view.default(mm_3, [1, 32, 768]);  mm_3 = None
        return (view_10,)

def load_args(reader):
    buf0 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 32, 768), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf1, (3072, 768), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1536, 16), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1536, 1, 4), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1536,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 491520, device=device(type='cuda', index=0))
    reader.tensor(buf5, (80, 1536), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf6, (1536, 48), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf7, (1536,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf8, (1536,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768, 1536), is_leaf=True)  # arg9_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)