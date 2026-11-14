class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 896]", arg1_1: "bf16[896, 9728]", arg2_1: "bf16[4864, 896]"):
        # File: /u/jsaravananindira/vllm/benchmarks/kernels/m2_fusion_boundary/screen.py:25 in mlp, code: gate_up = normalized @ gate_up_weight
        mm: "bf16[64, 9728]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # File: /u/jsaravananindira/vllm/benchmarks/kernels/m2_fusion_boundary/screen.py:26 in mlp, code: gate, up = gate_up.chunk(2, dim=-1)
        split = torch.ops.aten.split.Tensor(mm, 4864, -1);  mm = None
        getitem: "bf16[64, 4864]" = split[0]
        getitem_1: "bf16[64, 4864]" = split[1];  split = None

        # File: /u/jsaravananindira/vllm/benchmarks/kernels/m2_fusion_boundary/screen.py:27 in mlp, code: activated = F.silu(gate) * up
        convert_element_type_2: "f32[64, 4864]" = torch.ops.prims.convert_element_type.default(getitem, torch.float32);  getitem = None
        neg: "f32[64, 4864]" = torch.ops.aten.neg.default(convert_element_type_2)
        exp: "f32[64, 4864]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[64, 4864]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[64, 4864]" = torch.ops.aten.div.Tensor(convert_element_type_2, add);  convert_element_type_2 = add = None
        convert_element_type_3: "bf16[64, 4864]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mul: "bf16[64, 4864]" = torch.ops.aten.mul.Tensor(convert_element_type_3, getitem_1);  convert_element_type_3 = getitem_1 = None

        # File: /u/jsaravananindira/vllm/benchmarks/kernels/m2_fusion_boundary/screen.py:28 in mlp, code: return activated @ down_weight
        mm_1: "bf16[64, 896]" = torch.ops.aten.mm.default(mul, arg2_1);  mul = arg2_1 = None
        return (mm_1,)
