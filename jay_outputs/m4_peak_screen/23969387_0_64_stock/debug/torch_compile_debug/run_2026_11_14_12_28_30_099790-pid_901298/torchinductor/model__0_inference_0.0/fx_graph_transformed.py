class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[18944, 3584]", arg1_1: "bf16[64, 3584]", arg2_1: "bf16[18944, 3584]", arg3_1: "bf16[18944, 3584]"):
        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:65 in stock, code: gate = x @ gate_weight.T
        permute: "bf16[3584, 18944]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "bf16[64, 18944]" = torch.ops.aten.mm.default(arg1_1, permute);  permute = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:67 in stock, code: return (functional.silu(gate) * up) @ down_weight_t
        convert_element_type_4: "f32[64, 18944]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        neg: "f32[64, 18944]" = torch.ops.aten.neg.default(convert_element_type_4)
        exp: "f32[64, 18944]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[64, 18944]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[64, 18944]" = torch.ops.aten.div.Tensor(convert_element_type_4, add);  convert_element_type_4 = add = None
        convert_element_type_5: "bf16[64, 18944]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:66 in stock, code: up = x @ up_weight.T
        permute_1: "bf16[3584, 18944]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm_1: "bf16[64, 18944]" = torch.ops.aten.mm.default(arg1_1, permute_1);  arg1_1 = permute_1 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_peak_screen.py:67 in stock, code: return (functional.silu(gate) * up) @ down_weight_t
        mul: "bf16[64, 18944]" = torch.ops.aten.mul.Tensor(convert_element_type_5, mm_1);  convert_element_type_5 = mm_1 = None
        mm_2: "bf16[64, 3584]" = torch.ops.aten.mm.default(mul, arg3_1);  mul = arg3_1 = None
        return (mm_2,)
