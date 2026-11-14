from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, output_3: "bf16[s72, 28, 128]", s72: "Sym(s72)", l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_: "bf16[3584, 3584]", l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_: "bf16[3584]", hidden_states: "bf16[s72, 3584]", l_self_modules_layers_modules_0_modules_mlp_modules_gate_up_proj_parameters_weight_: "bf16[37888, 3584]", l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_: "bf16[3584, 18944]", l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_: "bf16[3584]", l_self_modules_layers_modules_1_modules_self_attn_modules_qkv_proj_parameters_weight_: "bf16[4608, 3584]", l_self_modules_layers_modules_1_modules_self_attn_modules_qkv_proj_parameters_bias_: "bf16[4608]", l_positions_: "i64[s72]", l_self_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_: "bf16[32768, 128]"):
        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:594 in forward, code: return output.view(-1, hidden_size)
        view: "bf16[s72, 3584]" = output_3.view(-1, 3584);  output_3 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        linear: "bf16[s72, 3584]" = torch._C._nn.linear(view, l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_, None);  view = l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_ = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/layernorm.py:96 in forward_native, code: self.weight.data if self.pass_weight_add else None,
        _get_data_attr: "bf16[3584]" = torch._C._autograd._get_data_attr(l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_);  l_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_ = None

        # File: /u/jsaravananindira/vllm/vllm/ir/op.py:524 in __call__, code: return self.torch_op(*args, **kwargs)
        fused_add_rms_norm_maybe_inplace = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(linear, hidden_states, _get_data_attr, 1e-06, None);  linear = hidden_states = _get_data_attr = None
        getitem: "bf16[s72, 3584]" = fused_add_rms_norm_maybe_inplace[0]
        getitem_1: "bf16[s72, 3584]" = fused_add_rms_norm_maybe_inplace[1];  fused_add_rms_norm_maybe_inplace = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        linear_1: "bf16[s72, 37888]" = torch._C._nn.linear(getitem, l_self_modules_layers_modules_0_modules_mlp_modules_gate_up_proj_parameters_weight_, None);  getitem = l_self_modules_layers_modules_0_modules_mlp_modules_gate_up_proj_parameters_weight_ = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/activation.py:144 in forward_native, code: return F.silu(x[..., :d]) * x[..., d:]
        getitem_2: "bf16[s72, 18944]" = linear_1[(Ellipsis, slice(None, 18944, None))]
        silu: "bf16[s72, 18944]" = torch.nn.functional.silu(getitem_2);  getitem_2 = None
        getitem_3: "bf16[s72, 18944]" = linear_1[(Ellipsis, slice(18944, None, None))];  linear_1 = None
        mul: "bf16[s72, 18944]" = silu * getitem_3;  silu = getitem_3 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        linear_2: "bf16[s72, 3584]" = torch._C._nn.linear(mul, l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_, None);  mul = l_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_ = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/layernorm.py:96 in forward_native, code: self.weight.data if self.pass_weight_add else None,
        _get_data_attr_1: "bf16[3584]" = torch._C._autograd._get_data_attr(l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_);  l_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_ = None

        # File: /u/jsaravananindira/vllm/vllm/ir/op.py:524 in __call__, code: return self.torch_op(*args, **kwargs)
        fused_add_rms_norm_maybe_inplace_1 = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(linear_2, getitem_1, _get_data_attr_1, 1e-06, None);  linear_2 = getitem_1 = _get_data_attr_1 = None
        getitem_4: "bf16[s72, 3584]" = fused_add_rms_norm_maybe_inplace_1[0]
        getitem_5: "bf16[s72, 3584]" = fused_add_rms_norm_maybe_inplace_1[1];  fused_add_rms_norm_maybe_inplace_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        linear_3: "bf16[s72, 4608]" = torch._C._nn.linear(getitem_4, l_self_modules_layers_modules_1_modules_self_attn_modules_qkv_proj_parameters_weight_, l_self_modules_layers_modules_1_modules_self_attn_modules_qkv_proj_parameters_bias_);  getitem_4 = l_self_modules_layers_modules_1_modules_self_attn_modules_qkv_proj_parameters_weight_ = l_self_modules_layers_modules_1_modules_self_attn_modules_qkv_proj_parameters_bias_ = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/models/qwen2.py:214 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        split = linear_3.split([3584, 512, 512], dim = -1);  linear_3 = None
        getitem_6: "bf16[s72, 3584]" = split[0]

        # No stacktrace found for following nodes
        sym_size_int: "Sym(s72)" = torch.ops.aten.sym_size.int(getitem_6, 0)

        # File: /u/jsaravananindira/vllm/vllm/model_executor/models/qwen2.py:214 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        getitem_7: "bf16[s72, 512]" = split[1]

        # No stacktrace found for following nodes
        sym_size_int_1: "Sym(s72)" = torch.ops.aten.sym_size.int(getitem_7, 0)

        # File: /u/jsaravananindira/vllm/vllm/model_executor/models/qwen2.py:214 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        getitem_8: "bf16[s72, 512]" = split[2];  split = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:171 in forward_static, code: positions = positions.flatten()
        flatten: "i64[s72]" = l_positions_.flatten();  l_positions_ = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:173 in forward_static, code: cos_sin = cos_sin_cache.index_select(0, positions)
        index_select: "bf16[s72, 128]" = l_self_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_.index_select(0, flatten);  l_self_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_ = flatten = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:174 in forward_static, code: cos, sin = cos_sin.chunk(2, dim=-1)
        chunk = index_select.chunk(2, dim = -1);  index_select = None
        getitem_9: "bf16[s72, 64]" = chunk[0]
        getitem_10: "bf16[s72, 64]" = chunk[1];  chunk = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:177 in forward_static, code: query = query.view(num_tokens, -1, head_size)
        view_1: "bf16[s72, 28, 128]" = getitem_6.view(s72, -1, 128);  getitem_6 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:178 in forward_static, code: query_rot = query[..., :rotary_dim]
        getitem_11: "bf16[s72, 28, 128]" = view_1[(Ellipsis, slice(None, 128, None))]

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:179 in forward_static, code: query_pass = query[..., rotary_dim:]
        getitem_12: "bf16[s72, 28, 0]" = view_1[(Ellipsis, slice(128, None, None))];  view_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:154 in forward_static, code: cos = cos.unsqueeze(-2).to(x.dtype)
        unsqueeze: "bf16[s72, 1, 64]" = getitem_9.unsqueeze(-2)
        to: "bf16[s72, 1, 64]" = unsqueeze.to(torch.bfloat16);  unsqueeze = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:155 in forward_static, code: sin = sin.unsqueeze(-2).to(x.dtype)
        unsqueeze_1: "bf16[s72, 1, 64]" = getitem_10.unsqueeze(-2)
        to_1: "bf16[s72, 1, 64]" = unsqueeze_1.to(torch.bfloat16);  unsqueeze_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:158 in forward_static, code: x1, x2 = torch.chunk(x, 2, dim=-1)
        chunk_1 = torch.chunk(getitem_11, 2, dim = -1);  getitem_11 = None
        getitem_13: "bf16[s72, 28, 64]" = chunk_1[0]
        getitem_14: "bf16[s72, 28, 64]" = chunk_1[1];  chunk_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: o1 = x1 * cos - x2 * sin
        mul_1: "bf16[s72, 28, 64]" = getitem_13 * to
        mul_2: "bf16[s72, 28, 64]" = getitem_14 * to_1
        sub: "bf16[s72, 28, 64]" = mul_1 - mul_2;  mul_1 = mul_2 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:164 in forward_static, code: o2 = x2 * cos + x1 * sin
        mul_3: "bf16[s72, 28, 64]" = getitem_14 * to;  getitem_14 = to = None
        mul_4: "bf16[s72, 28, 64]" = getitem_13 * to_1;  getitem_13 = to_1 = None
        add: "bf16[s72, 28, 64]" = mul_3 + mul_4;  mul_3 = mul_4 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:167 in forward_static, code: output = torch.cat((o1, o2), dim=-1)
        cat: "bf16[s72, 28, 128]" = torch.cat((sub, add), dim = -1);  sub = add = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:186 in forward_static, code: query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        cat_1: "bf16[s72, 28, 128]" = torch.cat((cat, getitem_12), dim = -1);  cat = getitem_12 = None
        reshape: "bf16[s72, 3584]" = cat_1.reshape(sym_size_int, 3584);  cat_1 = sym_size_int = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:191 in forward_static, code: key = key.view(num_tokens, -1, head_size)
        view_2: "bf16[s72, 4, 128]" = getitem_7.view(s72, -1, 128);  getitem_7 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:192 in forward_static, code: key_rot = key[..., :rotary_dim]
        getitem_15: "bf16[s72, 4, 128]" = view_2[(Ellipsis, slice(None, 128, None))]

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:193 in forward_static, code: key_pass = key[..., rotary_dim:]
        getitem_16: "bf16[s72, 4, 0]" = view_2[(Ellipsis, slice(128, None, None))];  view_2 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:154 in forward_static, code: cos = cos.unsqueeze(-2).to(x.dtype)
        unsqueeze_2: "bf16[s72, 1, 64]" = getitem_9.unsqueeze(-2);  getitem_9 = None
        to_2: "bf16[s72, 1, 64]" = unsqueeze_2.to(torch.bfloat16);  unsqueeze_2 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:155 in forward_static, code: sin = sin.unsqueeze(-2).to(x.dtype)
        unsqueeze_3: "bf16[s72, 1, 64]" = getitem_10.unsqueeze(-2);  getitem_10 = None
        to_3: "bf16[s72, 1, 64]" = unsqueeze_3.to(torch.bfloat16);  unsqueeze_3 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:158 in forward_static, code: x1, x2 = torch.chunk(x, 2, dim=-1)
        chunk_2 = torch.chunk(getitem_15, 2, dim = -1);  getitem_15 = None
        getitem_17: "bf16[s72, 4, 64]" = chunk_2[0]
        getitem_18: "bf16[s72, 4, 64]" = chunk_2[1];  chunk_2 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: o1 = x1 * cos - x2 * sin
        mul_5: "bf16[s72, 4, 64]" = getitem_17 * to_2
        mul_6: "bf16[s72, 4, 64]" = getitem_18 * to_3
        sub_1: "bf16[s72, 4, 64]" = mul_5 - mul_6;  mul_5 = mul_6 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:164 in forward_static, code: o2 = x2 * cos + x1 * sin
        mul_7: "bf16[s72, 4, 64]" = getitem_18 * to_2;  getitem_18 = to_2 = None
        mul_8: "bf16[s72, 4, 64]" = getitem_17 * to_3;  getitem_17 = to_3 = None
        add_1: "bf16[s72, 4, 64]" = mul_7 + mul_8;  mul_7 = mul_8 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:167 in forward_static, code: output = torch.cat((o1, o2), dim=-1)
        cat_2: "bf16[s72, 4, 128]" = torch.cat((sub_1, add_1), dim = -1);  sub_1 = add_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:200 in forward_static, code: key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        cat_3: "bf16[s72, 4, 128]" = torch.cat((cat_2, getitem_16), dim = -1);  cat_2 = getitem_16 = None
        reshape_1: "bf16[s72, 512]" = cat_3.reshape(sym_size_int_1, 512);  cat_3 = sym_size_int_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:543 in forward, code: output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        size = torch.Size([s72, 3584]);  s72 = None
        empty: "bf16[s72, 3584]" = torch.empty(size, dtype = torch.bfloat16, device = device(type='cuda', index=0));  size = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:548 in forward, code: query = query.view(-1, self.num_heads, self.head_size)
        view_3: "bf16[s72, 28, 128]" = reshape.view(-1, 28, 128);  reshape = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:549 in forward, code: output = output.view(-1, self.num_heads, self.head_size_v)
        view_4: "bf16[s72, 28, 128]" = empty.view(-1, 28, 128);  empty = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:551 in forward, code: key = key.view(-1, self.num_kv_heads, self.head_size)
        view_5: "bf16[s72, 4, 128]" = reshape_1.view(-1, 4, 128);  reshape_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:553 in forward, code: value = value.view(-1, self.num_kv_heads, self.head_size_v)
        view_6: "bf16[s72, 4, 128]" = getitem_8.view(-1, 4, 128);  getitem_8 = None
        return (view_5, view_6, view_3, view_4, getitem_5)
