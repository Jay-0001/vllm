class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 896]", arg1_1: "bf16[1, 896]", arg2_1: "bf16[896]"):
        # File: /u/jsaravananindira/vllm/benchmarks/kernels/m2_fusion_boundary/screen.py:18 in add_norm, code: updated = x.float() + residual.float()
        convert_element_type: "f32[1, 896]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[1, 896]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        add: "f32[1, 896]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None

        # File: /u/jsaravananindira/vllm/benchmarks/kernels/m2_fusion_boundary/screen.py:19 in add_norm, code: scale = torch.rsqrt(updated.square().mean(dim=-1, keepdim=True) + 1e-6)
        pow_1: "f32[1, 896]" = torch.ops.aten.pow.Tensor_Scalar(add, 2)
        mean: "f32[1, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1: "f32[1, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None

        # File: /u/jsaravananindira/vllm/benchmarks/kernels/m2_fusion_boundary/screen.py:20 in add_norm, code: normalized = (updated * scale).to(x.dtype) * norm_weight
        mul: "f32[1, 896]" = torch.ops.aten.mul.Tensor(add, rsqrt);  rsqrt = None
        convert_element_type_2: "bf16[1, 896]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[1, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_2, arg2_1);  convert_element_type_2 = arg2_1 = None
        return (mul_1, add)
