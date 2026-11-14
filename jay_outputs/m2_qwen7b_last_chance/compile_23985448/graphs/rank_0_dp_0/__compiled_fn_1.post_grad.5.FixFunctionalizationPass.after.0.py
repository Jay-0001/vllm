from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i32[s72]", arg1_1: "Sym(s72)", arg2_1: "bf16[152064, 3584]", arg3_1: "bf16[3584]", arg4_1: "bf16[4608, 3584]", arg5_1: "bf16[4608]", arg6_1: "i64[s72]", arg7_1: "bf16[32768, 128]"):
        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:530 in forward, code: return self.quant_method.embedding(self, input_.long())
        convert_element_type: "i64[s72]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.int64);  arg0_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:91 in embedding, code: return F.embedding(input_, layer.weight)
        embedding: "bf16[s72, 3584]" = torch.ops.aten.embedding.default(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None

        # File: /u/jsaravananindira/vllm/vllm/ir/op.py:365 in __call__, code: return self.torch_op(*args, **kwargs)
        convert_element_type_default: "f32[s72, 3584]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)
        pow_tensor_scalar: "f32[s72, 3584]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[s72, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor: "f32[s72, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[s72, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[s72, 3584]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1: "bf16[s72, 3584]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[s72, 3584]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, arg3_1);  convert_element_type_default_1 = arg3_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        permute: "bf16[3584, 4608]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm: "bf16[s72, 4608]" = torch.ops.aten.addmm.default(arg5_1, mul_tensor_1, permute);  arg5_1 = mul_tensor_1 = permute = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/models/qwen2.py:214 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        split_with_sizes = torch.ops.aten.split_with_sizes.default(addmm, [3584, 512, 512], -1);  addmm = None
        getitem: "bf16[s72, 3584]" = split_with_sizes[0]
        getitem_1: "bf16[s72, 512]" = split_with_sizes[1]
        getitem_2: "bf16[s72, 512]" = split_with_sizes[2];  split_with_sizes = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:173 in forward_static, code: cos_sin = cos_sin_cache.index_select(0, positions)
        index: "bf16[s72, 128]" = torch.ops.aten.index.Tensor(arg7_1, [arg6_1]);  arg7_1 = arg6_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:174 in forward_static, code: cos, sin = cos_sin.chunk(2, dim=-1)
        split = torch.ops.aten.split.Tensor(index, 64, -1);  index = None
        getitem_3: "bf16[s72, 64]" = split[0]
        getitem_4: "bf16[s72, 64]" = split[1];  split = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:177 in forward_static, code: query = query.view(num_tokens, -1, head_size)
        view: "bf16[s72, 28, 128]" = torch.ops.aten.reshape.default(getitem, [arg1_1, -1, 128]);  getitem = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:158 in forward_static, code: x1, x2 = torch.chunk(x, 2, dim=-1)
        split_1 = torch.ops.aten.split.Tensor(view, 64, -1);  view = None
        getitem_5: "bf16[s72, 28, 64]" = split_1[0]
        getitem_6: "bf16[s72, 28, 64]" = split_1[1];  split_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:191 in forward_static, code: key = key.view(num_tokens, -1, head_size)
        view_2: "bf16[s72, 4, 128]" = torch.ops.aten.reshape.default(getitem_1, [arg1_1, -1, 128]);  getitem_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:158 in forward_static, code: x1, x2 = torch.chunk(x, 2, dim=-1)
        split_2 = torch.ops.aten.split.Tensor(view_2, 64, -1);  view_2 = None
        getitem_7: "bf16[s72, 4, 64]" = split_2[0]
        getitem_8: "bf16[s72, 4, 64]" = split_2[1];  split_2 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:154 in forward_static, code: cos = cos.unsqueeze(-2).to(x.dtype)
        unsqueeze_2: "bf16[s72, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem_3, -2)

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: o1 = x1 * cos - x2 * sin
        mul_71: "bf16[s72, 4, 64]" = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze_2)

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:155 in forward_static, code: sin = sin.unsqueeze(-2).to(x.dtype)
        unsqueeze_3: "bf16[s72, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem_4, -2)

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: o1 = x1 * cos - x2 * sin
        mul_74: "bf16[s72, 4, 64]" = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_3)
        sub_36: "bf16[s72, 4, 64]" = torch.ops.aten.sub.Tensor(mul_71, mul_74);  mul_71 = mul_74 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:164 in forward_static, code: o2 = x2 * cos + x1 * sin
        mul_79: "bf16[s72, 4, 64]" = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_2);  getitem_8 = unsqueeze_2 = None
        mul_82: "bf16[s72, 4, 64]" = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze_3);  getitem_7 = unsqueeze_3 = None
        add_137: "bf16[s72, 4, 64]" = torch.ops.aten.add.Tensor(mul_79, mul_82);  mul_79 = mul_82 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:167 in forward_static, code: output = torch.cat((o1, o2), dim=-1)
        cat_1: "bf16[s72, 4, 128]" = torch.ops.aten.cat.default([sub_36, add_137], -1);  sub_36 = add_137 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:553 in forward, code: value = value.view(-1, self.num_kv_heads, self.head_size_v)
        view_7: "bf16[s72, 4, 128]" = torch.ops.aten.reshape.default(getitem_2, [-1, 4, 128]);  getitem_2 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:154 in forward_static, code: cos = cos.unsqueeze(-2).to(x.dtype)
        unsqueeze: "bf16[s72, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem_3, -2);  getitem_3 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: o1 = x1 * cos - x2 * sin
        mul_32: "bf16[s72, 28, 64]" = torch.ops.aten.mul.Tensor(getitem_5, unsqueeze)

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:155 in forward_static, code: sin = sin.unsqueeze(-2).to(x.dtype)
        unsqueeze_1: "bf16[s72, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem_4, -2);  getitem_4 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: o1 = x1 * cos - x2 * sin
        mul_35: "bf16[s72, 28, 64]" = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze_1)
        sub_19: "bf16[s72, 28, 64]" = torch.ops.aten.sub.Tensor(mul_32, mul_35);  mul_32 = mul_35 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:164 in forward_static, code: o2 = x2 * cos + x1 * sin
        mul_40: "bf16[s72, 28, 64]" = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze);  getitem_6 = unsqueeze = None
        mul_43: "bf16[s72, 28, 64]" = torch.ops.aten.mul.Tensor(getitem_5, unsqueeze_1);  getitem_5 = unsqueeze_1 = None
        add_75: "bf16[s72, 28, 64]" = torch.ops.aten.add.Tensor(mul_40, mul_43);  mul_40 = mul_43 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:167 in forward_static, code: output = torch.cat((o1, o2), dim=-1)
        cat: "bf16[s72, 28, 128]" = torch.ops.aten.cat.default([sub_19, add_75], -1);  sub_19 = add_75 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:543 in forward, code: output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        empty: "bf16[s72, 3584]" = torch.ops.aten.empty.memory_format([arg1_1, 3584], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False);  arg1_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:549 in forward, code: output = output.view(-1, self.num_heads, self.head_size_v)
        view_5: "bf16[s72, 28, 128]" = torch.ops.aten.reshape.default(empty, [-1, 28, 128]);  empty = None
        return (cat_1, view_7, cat, view_5, embedding)
