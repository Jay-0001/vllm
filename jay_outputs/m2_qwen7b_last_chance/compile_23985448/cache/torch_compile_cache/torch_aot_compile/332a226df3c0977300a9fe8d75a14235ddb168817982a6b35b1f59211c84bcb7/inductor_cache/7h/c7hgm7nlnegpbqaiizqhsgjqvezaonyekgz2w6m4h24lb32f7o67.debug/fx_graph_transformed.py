class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[s72, 28, 128]", arg1_1: "Sym(s72)", arg2_1: "bf16[3584, 3584]", arg3_1: "bf16[3584]", arg4_1: "bf16[s72, 3584]", arg5_1: "bf16[37888, 3584]", arg6_1: "bf16[3584, 18944]", arg7_1: "bf16[3584]", arg8_1: "bf16[4608, 3584]", arg9_1: "bf16[4608]", arg10_1: "i64[s72]", arg11_1: "bf16[32768, 128]"):
        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:594 in forward, code: return output.view(-1, hidden_size)
        view: "bf16[s72, 3584]" = torch.ops.aten.reshape.default(arg0_1, [-1, 3584]);  arg0_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        permute: "bf16[3584, 3584]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm: "bf16[s72, 3584]" = torch.ops.aten.mm.default(view, permute);  view = permute = None

        # File: /u/jsaravananindira/vllm/vllm/ir/op.py:524 in __call__, code: return self.torch_op(*args, **kwargs)
        convert_element_type_default_4: "f32[s72, 3584]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_default_5: "f32[s72, 3584]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        add_tensor_2: "f32[s72, 3584]" = torch.ops.aten.add.Tensor(convert_element_type_default_4, convert_element_type_default_5);  convert_element_type_default_4 = convert_element_type_default_5 = None
        convert_element_type_default_6: "bf16[s72, 3584]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16)
        pow_tensor_scalar_1: "f32[s72, 3584]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor_2, 2)
        mean_dim_1: "f32[s72, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True);  pow_tensor_scalar_1 = None
        add_tensor_3: "f32[s72, 1]" = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06);  mean_dim_1 = None
        rsqrt_default_1: "f32[s72, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_2: "f32[s72, 3584]" = torch.ops.aten.mul.Tensor(add_tensor_2, rsqrt_default_1);  add_tensor_2 = rsqrt_default_1 = None
        convert_element_type_default_7: "bf16[s72, 3584]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        mul_tensor_3: "bf16[s72, 3584]" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, arg3_1);  convert_element_type_default_7 = arg3_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        permute_1: "bf16[3584, 37888]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_1: "bf16[s72, 37888]" = torch.ops.aten.mm.default(mul_tensor_3, permute_1);  mul_tensor_3 = permute_1 = None

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

        # File: /u/jsaravananindira/vllm/vllm/ir/op.py:524 in __call__, code: return self.torch_op(*args, **kwargs)
        convert_element_type_default: "f32[s72, 3584]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_default_1: "f32[s72, 3584]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_6, torch.float32);  convert_element_type_default_6 = None
        add_tensor: "f32[s72, 3584]" = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_1);  convert_element_type_default = convert_element_type_default_1 = None
        convert_element_type_default_2: "bf16[s72, 3584]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16)
        pow_tensor_scalar: "f32[s72, 3584]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[s72, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[s72, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[s72, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[s72, 3584]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None
        convert_element_type_default_3: "bf16[s72, 3584]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[s72, 3584]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, arg7_1);  convert_element_type_default_3 = arg7_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/utils.py:633 in cuda_unquantized_gemm, code: return torch.nn.functional.linear(x, weight, bias)
        permute_3: "bf16[3584, 4608]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "bf16[s72, 4608]" = torch.ops.aten.addmm.default(arg9_1, mul_tensor_1, permute_3);  arg9_1 = mul_tensor_1 = permute_3 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/models/qwen2.py:214 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        split_with_sizes = torch.ops.aten.split_with_sizes.default(addmm, [3584, 512, 512], -1);  addmm = None
        getitem_4: "bf16[s72, 3584]" = split_with_sizes[0]
        getitem_5: "bf16[s72, 512]" = split_with_sizes[1]
        getitem_6: "bf16[s72, 512]" = split_with_sizes[2];  split_with_sizes = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:173 in forward_static, code: cos_sin = cos_sin_cache.index_select(0, positions)
        index: "bf16[s72, 128]" = torch.ops.aten.index.Tensor(arg11_1, [arg10_1]);  arg11_1 = arg10_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:174 in forward_static, code: cos, sin = cos_sin.chunk(2, dim=-1)
        split = torch.ops.aten.split.Tensor(index, 64, -1);  index = None
        getitem_7: "bf16[s72, 64]" = split[0]
        getitem_8: "bf16[s72, 64]" = split[1];  split = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:177 in forward_static, code: query = query.view(num_tokens, -1, head_size)
        view_1: "bf16[s72, 28, 128]" = torch.ops.aten.reshape.default(getitem_4, [arg1_1, -1, 128]);  getitem_4 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:158 in forward_static, code: x1, x2 = torch.chunk(x, 2, dim=-1)
        split_1 = torch.ops.aten.split.Tensor(view_1, 64, -1);  view_1 = None
        getitem_9: "bf16[s72, 28, 64]" = split_1[0]
        getitem_10: "bf16[s72, 28, 64]" = split_1[1];  split_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:191 in forward_static, code: key = key.view(num_tokens, -1, head_size)
        view_3: "bf16[s72, 4, 128]" = torch.ops.aten.reshape.default(getitem_5, [arg1_1, -1, 128]);  getitem_5 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:158 in forward_static, code: x1, x2 = torch.chunk(x, 2, dim=-1)
        split_2 = torch.ops.aten.split.Tensor(view_3, 64, -1);  view_3 = None
        getitem_11: "bf16[s72, 4, 64]" = split_2[0]
        getitem_12: "bf16[s72, 4, 64]" = split_2[1];  split_2 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:154 in forward_static, code: cos = cos.unsqueeze(-2).to(x.dtype)
        unsqueeze_2: "bf16[s72, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem_7, -2)

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: o1 = x1 * cos - x2 * sin
        mul_93: "bf16[s72, 4, 64]" = torch.ops.aten.mul.Tensor(getitem_11, unsqueeze_2)

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:155 in forward_static, code: sin = sin.unsqueeze(-2).to(x.dtype)
        unsqueeze_3: "bf16[s72, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem_8, -2)

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: o1 = x1 * cos - x2 * sin
        mul_96: "bf16[s72, 4, 64]" = torch.ops.aten.mul.Tensor(getitem_12, unsqueeze_3)
        sub_45: "bf16[s72, 4, 64]" = torch.ops.aten.sub.Tensor(mul_93, mul_96);  mul_93 = mul_96 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:164 in forward_static, code: o2 = x2 * cos + x1 * sin
        mul_101: "bf16[s72, 4, 64]" = torch.ops.aten.mul.Tensor(getitem_12, unsqueeze_2);  getitem_12 = unsqueeze_2 = None
        mul_104: "bf16[s72, 4, 64]" = torch.ops.aten.mul.Tensor(getitem_11, unsqueeze_3);  getitem_11 = unsqueeze_3 = None
        add_166: "bf16[s72, 4, 64]" = torch.ops.aten.add.Tensor(mul_101, mul_104);  mul_101 = mul_104 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:167 in forward_static, code: output = torch.cat((o1, o2), dim=-1)
        cat_1: "bf16[s72, 4, 128]" = torch.ops.aten.cat.default([sub_45, add_166], -1);  sub_45 = add_166 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:553 in forward, code: value = value.view(-1, self.num_kv_heads, self.head_size_v)
        view_8: "bf16[s72, 4, 128]" = torch.ops.aten.reshape.default(getitem_6, [-1, 4, 128]);  getitem_6 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:154 in forward_static, code: cos = cos.unsqueeze(-2).to(x.dtype)
        unsqueeze: "bf16[s72, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem_7, -2);  getitem_7 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: o1 = x1 * cos - x2 * sin
        mul_54: "bf16[s72, 28, 64]" = torch.ops.aten.mul.Tensor(getitem_9, unsqueeze)

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:155 in forward_static, code: sin = sin.unsqueeze(-2).to(x.dtype)
        unsqueeze_1: "bf16[s72, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem_8, -2);  getitem_8 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: o1 = x1 * cos - x2 * sin
        mul_57: "bf16[s72, 28, 64]" = torch.ops.aten.mul.Tensor(getitem_10, unsqueeze_1)
        sub_28: "bf16[s72, 28, 64]" = torch.ops.aten.sub.Tensor(mul_54, mul_57);  mul_54 = mul_57 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:164 in forward_static, code: o2 = x2 * cos + x1 * sin
        mul_62: "bf16[s72, 28, 64]" = torch.ops.aten.mul.Tensor(getitem_10, unsqueeze);  getitem_10 = unsqueeze = None
        mul_65: "bf16[s72, 28, 64]" = torch.ops.aten.mul.Tensor(getitem_9, unsqueeze_1);  getitem_9 = unsqueeze_1 = None
        add_104: "bf16[s72, 28, 64]" = torch.ops.aten.add.Tensor(mul_62, mul_65);  mul_62 = mul_65 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:167 in forward_static, code: output = torch.cat((o1, o2), dim=-1)
        cat: "bf16[s72, 28, 128]" = torch.ops.aten.cat.default([sub_28, add_104], -1);  sub_28 = add_104 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:543 in forward, code: output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        empty: "bf16[s72, 3584]" = torch.ops.aten.empty.memory_format([arg1_1, 3584], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False);  arg1_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/attention/attention.py:549 in forward, code: output = output.view(-1, self.num_heads, self.head_size_v)
        view_6: "bf16[s72, 28, 128]" = torch.ops.aten.reshape.default(empty, [-1, 28, 128]);  empty = None
        return (cat_1, view_8, cat, view_6, convert_element_type_default_2)
