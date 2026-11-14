class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4, 9728]", arg1_1: "bf16[4864, 896]"):
        # File: /u/jsaravananindira/vllm/benchmarks/kernels/m2_fusion_boundary/screen.py:42 in activation_down, code: gate, up = gate_up.chunk(2, dim=-1)
        split = torch.ops.aten.split.Tensor(arg0_1, 4864, -1);  arg0_1 = None
        getitem: "bf16[4, 4864]" = split[0]
        getitem_1: "bf16[4, 4864]" = split[1];  split = None

        # File: /u/jsaravananindira/vllm/benchmarks/kernels/m2_fusion_boundary/screen.py:43 in activation_down, code: return (F.silu(gate) * up) @ down_weight
        convert_element_type: "f32[4, 4864]" = torch.ops.prims.convert_element_type.default(getitem, torch.float32);  getitem = None
        neg: "f32[4, 4864]" = torch.ops.aten.neg.default(convert_element_type)
        exp: "f32[4, 4864]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[4, 4864]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[4, 4864]" = torch.ops.aten.div.Tensor(convert_element_type, add);  convert_element_type = add = None
        convert_element_type_1: "bf16[4, 4864]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mul: "bf16[4, 4864]" = torch.ops.aten.mul.Tensor(convert_element_type_1, getitem_1);  convert_element_type_1 = getitem_1 = None
        mm: "bf16[4, 896]" = torch.ops.aten.mm.default(mul, arg1_1);  mul = arg1_1 = None
        return (mm,)
