class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s42)", arg1_1: "Sym(s26)", arg2_1: "bf16[s42, 3584]", arg3_1: "bf16[2048, 3584]", arg4_1: "bf16[s42, 3584]", arg5_1: "bf16[s42, 3584]"):
        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_stream_screen.py:91 in tile_kernel, code: gate = input_tensor @ gate_block.T
        permute: "bf16[3584, s42]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm: "bf16[2048, s42]" = torch.ops.aten.mm.default(arg3_1, permute);  permute = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_stream_screen.py:93 in tile_kernel, code: return (functional.silu(gate) * up) @ down_block
        convert_element_type_4: "f32[2048, s42]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        neg: "f32[2048, s42]" = torch.ops.aten.neg.default(convert_element_type_4)
        exp: "f32[2048, s42]" = torch.ops.aten.exp.default(neg);  neg = None
        add_10: "f32[2048, s42]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[2048, s42]" = torch.ops.aten.div.Tensor(convert_element_type_4, add_10);  convert_element_type_4 = add_10 = None
        convert_element_type_5: "bf16[2048, s42]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_stream_screen.py:92 in tile_kernel, code: up = input_tensor @ up_block.T
        permute_1: "bf16[3584, s42]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_1: "bf16[2048, s42]" = torch.ops.aten.mm.default(arg3_1, permute_1);  arg3_1 = permute_1 = None

        # File: /u/jsaravananindira/vllm-m4/jay_scripts/m4_stream_screen.py:93 in tile_kernel, code: return (functional.silu(gate) * up) @ down_block
        mul_10: "bf16[2048, s42]" = torch.ops.aten.mul.Tensor(convert_element_type_5, mm_1);  convert_element_type_5 = mm_1 = None
        mm_2: "bf16[2048, 3584]" = torch.ops.aten.mm.default(mul_10, arg5_1);  mul_10 = arg5_1 = None
        return (mm_2,)
