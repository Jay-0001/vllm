from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, output_111: "bf16[s72, 28, 128]", s72: "Sym(s72)", l_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_: "bf16[3584, 3584]", l_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_: "bf16[3584]", residual_53: "bf16[s72, 3584]", l_self_modules_layers_modules_27_modules_mlp_modules_gate_up_proj_parameters_weight_: "bf16[37888, 3584]", l_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_: "bf16[3584, 18944]", l_self_modules_norm_parameters_weight_: "bf16[3584]"):
        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:594 in forward, code: return output.view(-1, hidden_size)
        view: "bf16[s72, 3584]" = output_111.view(-1, 3584);  output_111 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        linear: "bf16[s72, 3584]" = torch._C._nn.linear(view, l_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_, None);  view = l_self_modules_layers_modules_27_modules_self_attn_modules_o_proj_parameters_weight_ = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/layernorm.py:96 in forward_native, code: self.weight.data if self.pass_weight_add else None,
        _get_data_attr: "bf16[3584]" = torch._C._autograd._get_data_attr(l_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_);  l_self_modules_layers_modules_27_modules_post_attention_layernorm_parameters_weight_ = None

        # File: /u/jsaravananindira/vllm/vllm/ir/op.py:524 in __call__, code: return self.torch_op(*args, **kwargs)
        fused_add_rms_norm_maybe_inplace = torch.ops.vllm_ir.fused_add_rms_norm.default(linear, residual_53, _get_data_attr, 1e-06, None);  linear = residual_53 = _get_data_attr = None
        getitem: "bf16[s72, 3584]" = fused_add_rms_norm_maybe_inplace[0]
        getitem_1: "bf16[s72, 3584]" = fused_add_rms_norm_maybe_inplace[1];  fused_add_rms_norm_maybe_inplace = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        linear_1: "bf16[s72, 37888]" = torch._C._nn.linear(getitem, l_self_modules_layers_modules_27_modules_mlp_modules_gate_up_proj_parameters_weight_, None);  getitem = l_self_modules_layers_modules_27_modules_mlp_modules_gate_up_proj_parameters_weight_ = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/activation.py:144 in forward_native, code: return F.silu(x[..., :d]) * x[..., d:]
        getitem_2: "bf16[s72, 18944]" = linear_1[(Ellipsis, slice(None, 18944, None))]
        silu: "bf16[s72, 18944]" = torch.nn.functional.silu(getitem_2);  getitem_2 = None
        getitem_3: "bf16[s72, 18944]" = linear_1[(Ellipsis, slice(18944, None, None))];  linear_1 = None
        mul: "bf16[s72, 18944]" = silu * getitem_3;  silu = getitem_3 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        linear_2: "bf16[s72, 3584]" = torch._C._nn.linear(mul, l_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_, None);  mul = l_self_modules_layers_modules_27_modules_mlp_modules_down_proj_parameters_weight_ = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/layernorm.py:96 in forward_native, code: self.weight.data if self.pass_weight_add else None,
        _get_data_attr_1: "bf16[3584]" = torch._C._autograd._get_data_attr(l_self_modules_norm_parameters_weight_);  l_self_modules_norm_parameters_weight_ = None

        # File: /u/jsaravananindira/vllm/vllm/ir/op.py:524 in __call__, code: return self.torch_op(*args, **kwargs)
        fused_add_rms_norm_maybe_inplace_1 = torch.ops.vllm_ir.fused_add_rms_norm.default(linear_2, getitem_1, _get_data_attr_1, 1e-06, None);  linear_2 = getitem_1 = _get_data_attr_1 = None
        getitem_4: "bf16[s72, 3584]" = fused_add_rms_norm_maybe_inplace_1[0];  fused_add_rms_norm_maybe_inplace_1 = None
        return (getitem_4,)
