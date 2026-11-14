class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[18944, 3584]", arg1_1: "bf16[2048, 3584]", arg2_1: "bf16[18944, 3584]", arg3_1: "bf16[18944, 3584]"):
        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:70 in tiled, code: accumulated = torch.zeros(
        full_default: "f32[2048, 3584]" = torch.ops.aten.full.default([2048, 3584], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:75 in tiled, code: gate = x @ gate_weight[start:end].T
        slice_1: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 2048)
        permute: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_1, [1, 0]);  slice_1 = None
        mm: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute);  permute = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:76 in tiled, code: up = x @ up_weight[start:end].T
        slice_2: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 0, 2048)
        permute_1: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_2, [1, 0]);  slice_2 = None
        mm_1: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_1);  permute_1 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:77 in tiled, code: partial = (functional.silu(gate) * up) @ down_weight_t[start:end]
        convert_element_type_4: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        neg: "f32[2048, 2048]" = torch.ops.aten.neg.default(convert_element_type_4)
        exp: "f32[2048, 2048]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[2048, 2048]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_4, add);  convert_element_type_4 = add = None
        convert_element_type_5: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mul: "bf16[2048, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_5, mm_1);  convert_element_type_5 = mm_1 = None
        slice_3: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 0, 2048)
        mm_2: "bf16[2048, 3584]" = torch.ops.aten.mm.default(mul, slice_3);  mul = slice_3 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:78 in tiled, code: accumulated = accumulated + partial.float()
        convert_element_type_8: "f32[2048, 3584]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:75 in tiled, code: gate = x @ gate_weight[start:end].T
        slice_4: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2048, 4096)
        permute_2: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_4, [1, 0]);  slice_4 = None
        mm_3: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_2);  permute_2 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:76 in tiled, code: up = x @ up_weight[start:end].T
        slice_5: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2048, 4096)
        permute_3: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_5, [1, 0]);  slice_5 = None
        mm_4: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_3);  permute_3 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:77 in tiled, code: partial = (functional.silu(gate) * up) @ down_weight_t[start:end]
        convert_element_type_13: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        neg_1: "f32[2048, 2048]" = torch.ops.aten.neg.default(convert_element_type_13)
        exp_1: "f32[2048, 2048]" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_2: "f32[2048, 2048]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_13, add_2);  convert_element_type_13 = add_2 = None
        convert_element_type_14: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        mul_1: "bf16[2048, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_14, mm_4);  convert_element_type_14 = mm_4 = None
        slice_6: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 2048, 4096)
        mm_5: "bf16[2048, 3584]" = torch.ops.aten.mm.default(mul_1, slice_6);  mul_1 = slice_6 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:78 in tiled, code: accumulated = accumulated + partial.float()
        convert_element_type_17: "f32[2048, 3584]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        add_3: "f32[2048, 3584]" = torch.ops.aten.add.Tensor(convert_element_type_8, convert_element_type_17);  convert_element_type_8 = convert_element_type_17 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:75 in tiled, code: gate = x @ gate_weight[start:end].T
        slice_7: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 4096, 6144)
        permute_4: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_7, [1, 0]);  slice_7 = None
        mm_6: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_4);  permute_4 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:76 in tiled, code: up = x @ up_weight[start:end].T
        slice_8: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 4096, 6144)
        permute_5: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_8, [1, 0]);  slice_8 = None
        mm_7: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_5);  permute_5 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:77 in tiled, code: partial = (functional.silu(gate) * up) @ down_weight_t[start:end]
        convert_element_type_22: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        neg_2: "f32[2048, 2048]" = torch.ops.aten.neg.default(convert_element_type_22)
        exp_2: "f32[2048, 2048]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_4: "f32[2048, 2048]" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_22, add_4);  convert_element_type_22 = add_4 = None
        convert_element_type_23: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        mul_2: "bf16[2048, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_23, mm_7);  convert_element_type_23 = mm_7 = None
        slice_9: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 4096, 6144)
        mm_8: "bf16[2048, 3584]" = torch.ops.aten.mm.default(mul_2, slice_9);  mul_2 = slice_9 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:78 in tiled, code: accumulated = accumulated + partial.float()
        convert_element_type_26: "f32[2048, 3584]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        add_5: "f32[2048, 3584]" = torch.ops.aten.add.Tensor(add_3, convert_element_type_26);  add_3 = convert_element_type_26 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:75 in tiled, code: gate = x @ gate_weight[start:end].T
        slice_10: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 6144, 8192)
        permute_6: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_10, [1, 0]);  slice_10 = None
        mm_9: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_6);  permute_6 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:76 in tiled, code: up = x @ up_weight[start:end].T
        slice_11: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 6144, 8192)
        permute_7: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_11, [1, 0]);  slice_11 = None
        mm_10: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_7);  permute_7 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:77 in tiled, code: partial = (functional.silu(gate) * up) @ down_weight_t[start:end]
        convert_element_type_31: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        neg_3: "f32[2048, 2048]" = torch.ops.aten.neg.default(convert_element_type_31)
        exp_3: "f32[2048, 2048]" = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_6: "f32[2048, 2048]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_31, add_6);  convert_element_type_31 = add_6 = None
        convert_element_type_32: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        mul_3: "bf16[2048, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_32, mm_10);  convert_element_type_32 = mm_10 = None
        slice_12: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 6144, 8192)
        mm_11: "bf16[2048, 3584]" = torch.ops.aten.mm.default(mul_3, slice_12);  mul_3 = slice_12 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:78 in tiled, code: accumulated = accumulated + partial.float()
        convert_element_type_35: "f32[2048, 3584]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        add_7: "f32[2048, 3584]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_35);  add_5 = convert_element_type_35 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:75 in tiled, code: gate = x @ gate_weight[start:end].T
        slice_13: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 8192, 10240)
        permute_8: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_13, [1, 0]);  slice_13 = None
        mm_12: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_8);  permute_8 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:76 in tiled, code: up = x @ up_weight[start:end].T
        slice_14: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 8192, 10240)
        permute_9: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_14, [1, 0]);  slice_14 = None
        mm_13: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_9);  permute_9 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:77 in tiled, code: partial = (functional.silu(gate) * up) @ down_weight_t[start:end]
        convert_element_type_40: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        neg_4: "f32[2048, 2048]" = torch.ops.aten.neg.default(convert_element_type_40)
        exp_4: "f32[2048, 2048]" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_8: "f32[2048, 2048]" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_40, add_8);  convert_element_type_40 = add_8 = None
        convert_element_type_41: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        mul_4: "bf16[2048, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_41, mm_13);  convert_element_type_41 = mm_13 = None
        slice_15: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 8192, 10240)
        mm_14: "bf16[2048, 3584]" = torch.ops.aten.mm.default(mul_4, slice_15);  mul_4 = slice_15 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:78 in tiled, code: accumulated = accumulated + partial.float()
        convert_element_type_44: "f32[2048, 3584]" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        add_9: "f32[2048, 3584]" = torch.ops.aten.add.Tensor(add_7, convert_element_type_44);  add_7 = convert_element_type_44 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:75 in tiled, code: gate = x @ gate_weight[start:end].T
        slice_16: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 10240, 12288)
        permute_10: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_16, [1, 0]);  slice_16 = None
        mm_15: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_10);  permute_10 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:76 in tiled, code: up = x @ up_weight[start:end].T
        slice_17: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 10240, 12288)
        permute_11: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_17, [1, 0]);  slice_17 = None
        mm_16: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_11);  permute_11 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:77 in tiled, code: partial = (functional.silu(gate) * up) @ down_weight_t[start:end]
        convert_element_type_49: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        neg_5: "f32[2048, 2048]" = torch.ops.aten.neg.default(convert_element_type_49)
        exp_5: "f32[2048, 2048]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_10: "f32[2048, 2048]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_49, add_10);  convert_element_type_49 = add_10 = None
        convert_element_type_50: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        mul_5: "bf16[2048, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_50, mm_16);  convert_element_type_50 = mm_16 = None
        slice_18: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 10240, 12288)
        mm_17: "bf16[2048, 3584]" = torch.ops.aten.mm.default(mul_5, slice_18);  mul_5 = slice_18 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:78 in tiled, code: accumulated = accumulated + partial.float()
        convert_element_type_53: "f32[2048, 3584]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        add_11: "f32[2048, 3584]" = torch.ops.aten.add.Tensor(add_9, convert_element_type_53);  add_9 = convert_element_type_53 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:75 in tiled, code: gate = x @ gate_weight[start:end].T
        slice_19: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 12288, 14336)
        permute_12: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_19, [1, 0]);  slice_19 = None
        mm_18: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_12);  permute_12 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:76 in tiled, code: up = x @ up_weight[start:end].T
        slice_20: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 12288, 14336)
        permute_13: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_20, [1, 0]);  slice_20 = None
        mm_19: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_13);  permute_13 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:77 in tiled, code: partial = (functional.silu(gate) * up) @ down_weight_t[start:end]
        convert_element_type_58: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_18, torch.float32);  mm_18 = None
        neg_6: "f32[2048, 2048]" = torch.ops.aten.neg.default(convert_element_type_58)
        exp_6: "f32[2048, 2048]" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_12: "f32[2048, 2048]" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_58, add_12);  convert_element_type_58 = add_12 = None
        convert_element_type_59: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        mul_6: "bf16[2048, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_59, mm_19);  convert_element_type_59 = mm_19 = None
        slice_21: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 12288, 14336)
        mm_20: "bf16[2048, 3584]" = torch.ops.aten.mm.default(mul_6, slice_21);  mul_6 = slice_21 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:78 in tiled, code: accumulated = accumulated + partial.float()
        convert_element_type_62: "f32[2048, 3584]" = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        add_13: "f32[2048, 3584]" = torch.ops.aten.add.Tensor(add_11, convert_element_type_62);  add_11 = convert_element_type_62 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:75 in tiled, code: gate = x @ gate_weight[start:end].T
        slice_22: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 14336, 16384)
        permute_14: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_22, [1, 0]);  slice_22 = None
        mm_21: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_14);  permute_14 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:76 in tiled, code: up = x @ up_weight[start:end].T
        slice_23: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 14336, 16384)
        permute_15: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_23, [1, 0]);  slice_23 = None
        mm_22: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_15);  permute_15 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:77 in tiled, code: partial = (functional.silu(gate) * up) @ down_weight_t[start:end]
        convert_element_type_67: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        neg_7: "f32[2048, 2048]" = torch.ops.aten.neg.default(convert_element_type_67)
        exp_7: "f32[2048, 2048]" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_14: "f32[2048, 2048]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_67, add_14);  convert_element_type_67 = add_14 = None
        convert_element_type_68: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        mul_7: "bf16[2048, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_68, mm_22);  convert_element_type_68 = mm_22 = None
        slice_24: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 14336, 16384)
        mm_23: "bf16[2048, 3584]" = torch.ops.aten.mm.default(mul_7, slice_24);  mul_7 = slice_24 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:78 in tiled, code: accumulated = accumulated + partial.float()
        convert_element_type_71: "f32[2048, 3584]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        add_15: "f32[2048, 3584]" = torch.ops.aten.add.Tensor(add_13, convert_element_type_71);  add_13 = convert_element_type_71 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:75 in tiled, code: gate = x @ gate_weight[start:end].T
        slice_25: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 16384, 18432)
        permute_16: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_25, [1, 0]);  slice_25 = None
        mm_24: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_16);  permute_16 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:76 in tiled, code: up = x @ up_weight[start:end].T
        slice_26: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 16384, 18432)
        permute_17: "bf16[3584, 2048]" = torch.ops.aten.permute.default(slice_26, [1, 0]);  slice_26 = None
        mm_25: "bf16[2048, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_17);  permute_17 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:77 in tiled, code: partial = (functional.silu(gate) * up) @ down_weight_t[start:end]
        convert_element_type_76: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        neg_8: "f32[2048, 2048]" = torch.ops.aten.neg.default(convert_element_type_76)
        exp_8: "f32[2048, 2048]" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_16: "f32[2048, 2048]" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        div_8: "f32[2048, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_76, add_16);  convert_element_type_76 = add_16 = None
        convert_element_type_77: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        mul_8: "bf16[2048, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_77, mm_25);  convert_element_type_77 = mm_25 = None
        slice_27: "bf16[2048, 3584]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 16384, 18432)
        mm_26: "bf16[2048, 3584]" = torch.ops.aten.mm.default(mul_8, slice_27);  mul_8 = slice_27 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:78 in tiled, code: accumulated = accumulated + partial.float()
        convert_element_type_80: "f32[2048, 3584]" = torch.ops.prims.convert_element_type.default(mm_26, torch.float32);  mm_26 = None
        add_17: "f32[2048, 3584]" = torch.ops.aten.add.Tensor(add_15, convert_element_type_80);  add_15 = convert_element_type_80 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:75 in tiled, code: gate = x @ gate_weight[start:end].T
        slice_28: "bf16[512, 3584]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 18432, 18944);  arg0_1 = None
        permute_18: "bf16[3584, 512]" = torch.ops.aten.permute.default(slice_28, [1, 0]);  slice_28 = None
        mm_27: "bf16[2048, 512]" = torch.ops.aten.mm.default(arg1_1, permute_18);  permute_18 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:76 in tiled, code: up = x @ up_weight[start:end].T
        slice_29: "bf16[512, 3584]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 18432, 18944);  arg2_1 = None
        permute_19: "bf16[3584, 512]" = torch.ops.aten.permute.default(slice_29, [1, 0]);  slice_29 = None
        mm_28: "bf16[2048, 512]" = torch.ops.aten.mm.default(arg1_1, permute_19);  arg1_1 = permute_19 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:77 in tiled, code: partial = (functional.silu(gate) * up) @ down_weight_t[start:end]
        convert_element_type_85: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        neg_9: "f32[2048, 512]" = torch.ops.aten.neg.default(convert_element_type_85)
        exp_9: "f32[2048, 512]" = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_18: "f32[2048, 512]" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        div_9: "f32[2048, 512]" = torch.ops.aten.div.Tensor(convert_element_type_85, add_18);  convert_element_type_85 = add_18 = None
        convert_element_type_86: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        mul_9: "bf16[2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_86, mm_28);  convert_element_type_86 = mm_28 = None
        slice_30: "bf16[512, 3584]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 18432, 18944);  arg3_1 = None
        mm_29: "bf16[2048, 3584]" = torch.ops.aten.mm.default(mul_9, slice_30);  mul_9 = slice_30 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:78 in tiled, code: accumulated = accumulated + partial.float()
        convert_element_type_89: "f32[2048, 3584]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        add_19: "f32[2048, 3584]" = torch.ops.aten.add.Tensor(add_17, convert_element_type_89);  add_17 = convert_element_type_89 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:79 in tiled, code: return accumulated.to(dtype)
        convert_element_type_90: "bf16[2048, 3584]" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        return (convert_element_type_90,)
