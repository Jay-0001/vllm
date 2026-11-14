class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[s72, 28, 128]", arg1_1: "Sym(s72)", arg2_1: "bf16[3584, 3584]", arg3_1: "bf16[3584]", arg4_1: "bf16[s72, 3584]", arg5_1: "bf16[37888, 3584]", arg6_1: "bf16[3584, 18944]", arg7_1: "bf16[3584]"):
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
        pow_tensor_scalar: "f32[s72, 3584]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[s72, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[s72, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[s72, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[s72, 3584]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None
        convert_element_type_default_3: "bf16[s72, 3584]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[s72, 3584]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, arg7_1);  convert_element_type_default_3 = arg7_1 = None
        return (mul_tensor_1,)
