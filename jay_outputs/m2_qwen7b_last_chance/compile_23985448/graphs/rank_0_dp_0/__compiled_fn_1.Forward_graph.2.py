from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[s72, 28, 128]", arg1_1: "Sym(s72)", arg2_1: "bf16[3584, 3584]", arg3_1: "bf16[3584]", arg4_1: "bf16[s72, 3584]", arg5_1: "bf16[37888, 3584]", arg6_1: "bf16[3584, 18944]", arg7_1: "bf16[3584]"):
        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:594 in forward, code: return output.view(-1, hidden_size)
        view: "bf16[s72, 3584]" = torch.ops.aten.view.default(arg0_1, [-1, 3584]);  arg0_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        permute: "bf16[3584, 3584]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm: "bf16[s72, 3584]" = torch.ops.aten.mm.default(view, permute);  view = permute = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/layernorm.py:96 in forward_native, code: self.weight.data if self.pass_weight_add else None,
        alias: "bf16[3584]" = torch.ops.aten.alias.default(arg3_1);  arg3_1 = None

        # File: /u/jsaravananindira/vllm/vllm/ir/op.py:524 in __call__, code: return self.torch_op(*args, **kwargs)
        fused_add_rms_norm = torch.ops.vllm_ir.fused_add_rms_norm.default(mm, arg4_1, alias, 1e-06);  mm = arg4_1 = alias = None
        getitem: "bf16[s72, 3584]" = fused_add_rms_norm[0]
        getitem_1: "bf16[s72, 3584]" = fused_add_rms_norm[1];  fused_add_rms_norm = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        permute_1: "bf16[3584, 37888]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_1: "bf16[s72, 37888]" = torch.ops.aten.mm.default(getitem, permute_1);  getitem = permute_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/activation.py:144 in forward_native, code: return F.silu(x[..., :d]) * x[..., d:]
        slice_1: "bf16[s72, 18944]" = torch.ops.aten.slice.Tensor(mm_1, 1, 0, 18944)
        convert_element_type_4: "f32[s72, 18944]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        neg: "f32[s72, 18944]" = torch.ops.aten.neg.default(convert_element_type_4)
        exp: "f32[s72, 18944]" = torch.ops.aten.exp.default(neg);  neg = None
        add_18: "f32[s72, 18944]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[s72, 18944]" = torch.ops.aten.div.Tensor(convert_element_type_4, add_18);  convert_element_type_4 = add_18 = None
        convert_element_type_5: "bf16[s72, 18944]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        slice_2: "bf16[s72, 18944]" = torch.ops.aten.slice.Tensor(mm_1, 1, 18944, 9223372036854775807);  mm_1 = None
        mul_16: "bf16[s72, 18944]" = torch.ops.aten.mul.Tensor(convert_element_type_5, slice_2);  convert_element_type_5 = slice_2 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        permute_2: "bf16[18944, 3584]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm_2: "bf16[s72, 3584]" = torch.ops.aten.mm.default(mul_16, permute_2);  mul_16 = permute_2 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/layernorm.py:96 in forward_native, code: self.weight.data if self.pass_weight_add else None,
        alias_1: "bf16[3584]" = torch.ops.aten.alias.default(arg7_1);  arg7_1 = None

        # File: /u/jsaravananindira/vllm/vllm/ir/op.py:524 in __call__, code: return self.torch_op(*args, **kwargs)
        fused_add_rms_norm_1 = torch.ops.vllm_ir.fused_add_rms_norm.default(mm_2, getitem_1, alias_1, 1e-06);  mm_2 = getitem_1 = alias_1 = None
        getitem_2: "bf16[s72, 3584]" = fused_add_rms_norm_1[0];  fused_add_rms_norm_1 = None
        return (getitem_2,)
