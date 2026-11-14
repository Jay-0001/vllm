class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[512]", arg1_1: "bf16[512]", arg2_1: "bf16[1, 512, 16, 16]", arg3_1: "bf16[256, 512, 3, 3]", arg4_1: "bf16[256]", arg5_1: "bf16[1, 512]", arg6_1: "bf16[256, 512]", arg7_1: "bf16[256]", arg8_1: "bf16[256]", arg9_1: "bf16[256]", arg10_1: "bf16[256, 256, 3, 3]", arg11_1: "bf16[256]", arg12_1: "bf16[256, 512, 1, 1]", arg13_1: "bf16[256]"):
        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/diffusers/models/resnet.py:326 in forward, code: hidden_states = self.norm1(hidden_states)
        clone: "bf16[1, 512, 16, 16]" = torch.ops.aten.clone.default(arg2_1, memory_format = torch.contiguous_format)
        convert_element_type: "f32[1, 512, 16, 16]" = torch.ops.prims.convert_element_type.default(clone, torch.float32);  clone = None
        view: "f32[1, 32, 16, 256]" = torch.ops.aten.reshape.default(convert_element_type, [1, 32, 16, 256]);  convert_element_type = None
        var_mean = torch.ops.aten.var_mean.correction(view, [2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 32, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 32, 16, 256]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = getitem_1 = None
        add: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 32, 16, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        view_1: "f32[1, 512, 16, 16]" = torch.ops.aten.reshape.default(mul, [1, 512, 16, 16]);  mul = None
        unsqueeze: "bf16[1, 512]" = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        unsqueeze_1: "bf16[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "bf16[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_1: "f32[1, 512, 16, 16]" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_2);  view_1 = unsqueeze_2 = None
        unsqueeze_3: "bf16[1, 512]" = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
        unsqueeze_4: "bf16[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "bf16[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        add_1: "f32[1, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        convert_element_type_1: "bf16[1, 512, 16, 16]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        convert_element_type_4: "f32[1, 512, 16, 16]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        neg: "f32[1, 512, 16, 16]" = torch.ops.aten.neg.default(convert_element_type_4)
        exp: "f32[1, 512, 16, 16]" = torch.ops.aten.exp.default(neg);  neg = None
        add_2: "f32[1, 512, 16, 16]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[1, 512, 16, 16]" = torch.ops.aten.div.Tensor(convert_element_type_4, add_2);  convert_element_type_4 = add_2 = None
        convert_element_type_5: "bf16[1, 512, 16, 16]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/diffusers/models/resnet.py:340 in forward, code: hidden_states = self.conv1(hidden_states)
        convolution: "bf16[1, 256, 16, 16]" = torch.ops.aten.convolution.default(convert_element_type_5, arg3_1, arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_5 = arg3_1 = arg4_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/diffusers/models/resnet.py:344 in forward, code: temb = self.nonlinearity(temb)
        convert_element_type_6: "f32[1, 512]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        neg_1: "f32[1, 512]" = torch.ops.aten.neg.default(convert_element_type_6)
        exp_1: "f32[1, 512]" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_3: "f32[1, 512]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[1, 512]" = torch.ops.aten.div.Tensor(convert_element_type_6, add_3);  convert_element_type_6 = add_3 = None
        convert_element_type_7: "bf16[1, 512]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/diffusers/models/resnet.py:345 in forward, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        permute: "bf16[512, 256]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "bf16[1, 256]" = torch.ops.aten.addmm.default(arg7_1, convert_element_type_7, permute);  arg7_1 = convert_element_type_7 = permute = None
        unsqueeze_6: "bf16[1, 256, 1]" = torch.ops.aten.unsqueeze.default(addmm, 2);  addmm = None
        unsqueeze_7: "bf16[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 3);  unsqueeze_6 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/diffusers/models/resnet.py:349 in forward, code: hidden_states = hidden_states + temb
        add_4: "bf16[1, 256, 16, 16]" = torch.ops.aten.add.Tensor(convolution, unsqueeze_7);  convolution = unsqueeze_7 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = self.norm2(hidden_states)
        clone_1: "bf16[1, 256, 16, 16]" = torch.ops.aten.clone.default(add_4, memory_format = torch.contiguous_format);  add_4 = None
        convert_element_type_11: "f32[1, 256, 16, 16]" = torch.ops.prims.convert_element_type.default(clone_1, torch.float32);  clone_1 = None
        view_2: "f32[1, 32, 8, 256]" = torch.ops.aten.reshape.default(convert_element_type_11, [1, 32, 8, 256]);  convert_element_type_11 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(view_2, [2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/diffusers/models/resnet.py:373 in forward, code: input_tensor = self.conv_shortcut(input_tensor)
        convolution_2: "bf16[1, 256, 16, 16]" = torch.ops.aten.convolution.default(arg2_1, arg12_1, arg13_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg12_1 = arg13_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/diffusers/models/resnet.py:350 in forward, code: hidden_states = self.norm2(hidden_states)
        sub_1: "f32[1, 32, 8, 256]" = torch.ops.aten.sub.Tensor(view_2, getitem_3);  view_2 = getitem_3 = None
        add_5: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_2: "f32[1, 32, 8, 256]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        view_3: "f32[1, 256, 16, 16]" = torch.ops.aten.reshape.default(mul_2, [1, 256, 16, 16]);  mul_2 = None
        unsqueeze_8: "bf16[1, 256]" = torch.ops.aten.unsqueeze.default(arg8_1, 0);  arg8_1 = None
        unsqueeze_9: "bf16[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None
        unsqueeze_10: "bf16[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 3);  unsqueeze_9 = None
        mul_3: "f32[1, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_3, unsqueeze_10);  view_3 = unsqueeze_10 = None
        unsqueeze_11: "bf16[1, 256]" = torch.ops.aten.unsqueeze.default(arg9_1, 0);  arg9_1 = None
        unsqueeze_12: "bf16[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 2);  unsqueeze_11 = None
        unsqueeze_13: "bf16[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 3);  unsqueeze_12 = None
        add_6: "f32[1, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_3, unsqueeze_13);  mul_3 = unsqueeze_13 = None
        convert_element_type_12: "bf16[1, 256, 16, 16]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/diffusers/models/resnet.py:362 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        convert_element_type_15: "f32[1, 256, 16, 16]" = torch.ops.prims.convert_element_type.default(convert_element_type_12, torch.float32);  convert_element_type_12 = None
        neg_2: "f32[1, 256, 16, 16]" = torch.ops.aten.neg.default(convert_element_type_15)
        exp_2: "f32[1, 256, 16, 16]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_7: "f32[1, 256, 16, 16]" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[1, 256, 16, 16]" = torch.ops.aten.div.Tensor(convert_element_type_15, add_7);  convert_element_type_15 = add_7 = None
        convert_element_type_16: "bf16[1, 256, 16, 16]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/diffusers/models/resnet.py:365 in forward, code: hidden_states = self.conv2(hidden_states)
        convolution_1: "bf16[1, 256, 16, 16]" = torch.ops.aten.convolution.default(convert_element_type_16, arg10_1, arg11_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_16 = arg10_1 = arg11_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/diffusers/models/resnet.py:375 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_8: "bf16[1, 256, 16, 16]" = torch.ops.aten.add.Tensor(convolution_2, convolution_1);  convolution_2 = convolution_1 = None
        return (add_8,)
