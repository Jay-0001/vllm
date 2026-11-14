class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[96, 1, 1]", arg1_1: "bf16[96, 1, 7, 7]", arg2_1: "bf16[96]", arg3_1: "bf16[1, 96, 56, 56]", arg4_1: "bf16[96]", arg5_1: "bf16[96]", arg6_1: "bf16[384, 96]", arg7_1: "bf16[384]", arg8_1: "bf16[96, 384]", arg9_1: "bf16[96]"):
        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/torchvision/models/convnext.py:64 in forward, code: result = self.layer_scale * self.block(input)
        convolution: "bf16[1, 96, 56, 56]" = torch.ops.aten.convolution.default(arg3_1, arg1_1, arg2_1, [1, 1], [3, 3], [1, 1], False, [0, 0], 96);  arg1_1 = arg2_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/torchvision/ops/misc.py:321 in forward, code: return torch.permute(x, self.dims)
        permute: "bf16[1, 56, 56, 96]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/torchvision/models/convnext.py:64 in forward, code: result = self.layer_scale * self.block(input)
        convert_element_type: "f32[1, 56, 56, 96]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem: "f32[1, 56, 56, 1]" = var_mean[0]
        getitem_1: "f32[1, 56, 56, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[1, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[1, 56, 56, 96]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        mul: "f32[1, 56, 56, 96]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 56, 56, 96]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_1: "f32[1, 56, 56, 96]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[1, 56, 56, 96]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[3136, 96]" = torch.ops.aten.view.default(convert_element_type_1, [3136, 96]);  convert_element_type_1 = None
        permute_1: "bf16[96, 384]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "bf16[3136, 384]" = torch.ops.aten.addmm.default(arg7_1, view, permute_1);  arg7_1 = view = permute_1 = None
        view_1: "bf16[1, 56, 56, 384]" = torch.ops.aten.view.default(addmm, [1, 56, 56, 384]);  addmm = None
        convert_element_type_5: "f32[1, 56, 56, 384]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_2: "f32[1, 56, 56, 384]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.5)
        mul_3: "f32[1, 56, 56, 384]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.7071067811865476);  convert_element_type_5 = None
        erf: "f32[1, 56, 56, 384]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_2: "f32[1, 56, 56, 384]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[1, 56, 56, 384]" = torch.ops.aten.mul.Tensor(mul_2, add_2);  mul_2 = add_2 = None
        convert_element_type_6: "bf16[1, 56, 56, 384]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        view_2: "bf16[3136, 384]" = torch.ops.aten.view.default(convert_element_type_6, [3136, 384]);  convert_element_type_6 = None
        permute_2: "bf16[384, 96]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1: "bf16[3136, 96]" = torch.ops.aten.addmm.default(arg9_1, view_2, permute_2);  arg9_1 = view_2 = permute_2 = None
        view_3: "bf16[1, 56, 56, 96]" = torch.ops.aten.view.default(addmm_1, [1, 56, 56, 96]);  addmm_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/torchvision/ops/misc.py:321 in forward, code: return torch.permute(x, self.dims)
        permute_3: "bf16[1, 96, 56, 56]" = torch.ops.aten.permute.default(view_3, [0, 3, 1, 2]);  view_3 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/torchvision/models/convnext.py:64 in forward, code: result = self.layer_scale * self.block(input)
        mul_5: "bf16[1, 96, 56, 56]" = torch.ops.aten.mul.Tensor(arg0_1, permute_3);  arg0_1 = permute_3 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/torchvision/models/convnext.py:66 in forward, code: result += input
        add_3: "bf16[1, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_5, arg3_1);  mul_5 = arg3_1 = None
        return (add_3,)
