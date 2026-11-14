class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 128, 768]", arg1_1: "f32[3072, 768]", arg2_1: "f32[1536, 16]", arg3_1: "f32[1536, 1, 4]", arg4_1: "f32[1536]", arg5_1: "f32[80, 1536]", arg6_1: "f32[1536, 48]", arg7_1: "f32[1536]", arg8_1: "f32[1536]", arg9_1: "f32[768, 1536]"):
        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:376 in forward, code: projected_states = self.in_proj(hidden_states).transpose(1, 2)
        permute: "f32[768, 3072]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view: "f32[128, 768]" = torch.ops.aten.view.default(arg0_1, [128, 768]);  arg0_1 = None
        mm: "f32[128, 3072]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "f32[1, 128, 3072]" = torch.ops.aten.view.default(mm, [1, 128, 3072]);  mm = None
        permute_1: "f32[1, 3072, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:378 in forward, code: A = -torch.exp(self.A_log.float())
        exp: "f32[1536, 16]" = torch.ops.aten.exp.default(arg2_1);  arg2_1 = None
        neg: "f32[1536, 16]" = torch.ops.aten.neg.default(exp);  exp = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:400 in forward, code: hidden_states_B_C, gate = projected_states.chunk(2, dim=1)
        split = torch.ops.aten.split.Tensor(permute_1, 1536, 1);  permute_1 = None
        getitem: "f32[1, 1536, 128]" = split[0]
        getitem_1: "f32[1, 1536, 128]" = split[1];  split = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:421 in forward, code: self.conv1d.weight.squeeze(1),
        squeeze: "f32[1536, 4]" = torch.ops.aten.squeeze.dim(arg3_1, 1);  arg3_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:93 in causal_conv1d_fn, code: weight=weight.unsqueeze(1),
        unsqueeze: "f32[1536, 1, 4]" = torch.ops.aten.unsqueeze.default(squeeze, 1);  squeeze = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:91 in causal_conv1d_fn, code: out = F.conv1d(
        convolution: "f32[1, 1536, 131]" = torch.ops.aten.convolution.default(getitem, unsqueeze, arg4_1, [1], [3], [1], False, [0], 1536);  getitem = unsqueeze = arg4_1 = None
        slice_1: "f32[1, 1536, 128]" = torch.ops.aten.slice.Tensor(convolution, 2, 0, 128);  convolution = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        neg_1: "f32[1, 1536, 128]" = torch.ops.aten.neg.default(slice_1)
        exp_1: "f32[1, 1536, 128]" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add: "f32[1, 1536, 128]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div: "f32[1, 1536, 128]" = torch.ops.aten.div.Tensor(slice_1, add);  slice_1 = add = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:431 in forward, code: hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C.transpose(1, 2), attention_mask)
        permute_2: "f32[1, 128, 1536]" = torch.ops.aten.permute.default(div, [0, 2, 1]);  div = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:433 in forward, code: self.x_proj(hidden_states_B_C),
        permute_3: "f32[1536, 80]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        view_2: "f32[128, 1536]" = torch.ops.aten.view.default(permute_2, [128, 1536])
        mm_1: "f32[128, 80]" = torch.ops.aten.mm.default(view_2, permute_3);  view_2 = permute_3 = None
        view_3: "f32[1, 128, 80]" = torch.ops.aten.view.default(mm_1, [1, 128, 80]);  mm_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:432 in forward, code: time_step, B, C = torch.split(
        split_with_sizes = torch.ops.aten.split_with_sizes.default(view_3, [48, 16, 16], -1);  view_3 = None
        getitem_2: "f32[1, 128, 48]" = split_with_sizes[0]
        getitem_3: "f32[1, 128, 16]" = split_with_sizes[1]
        getitem_4: "f32[1, 128, 16]" = split_with_sizes[2];  split_with_sizes = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:438 in forward, code: time_step = self.dt_proj.weight @ time_step.transpose(1, 2)
        permute_6: "f32[48, 1536]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        view_4: "f32[128, 48]" = torch.ops.aten.view.default(getitem_2, [128, 48]);  getitem_2 = None
        mm_2: "f32[128, 1536]" = torch.ops.aten.mm.default(view_4, permute_6);  view_4 = permute_6 = None
        view_5: "f32[1, 128, 1536]" = torch.ops.aten.view.default(mm_2, [1, 128, 1536]);  mm_2 = None
        permute_7: "f32[1, 1536, 128]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
        clone: "f32[1, 1536, 128]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:460 in forward, code: hidden_states_B_C.transpose(1, 2),
        permute_8: "f32[1, 1536, 128]" = torch.ops.aten.permute.default(permute_2, [0, 2, 1]);  permute_2 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:204 in mamba_selective_scan, code: dt = dt + delta_bias.to(dt.dtype)[..., None]
        unsqueeze_1: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, 1);  arg7_1 = None
        add_1: "f32[1, 1536, 128]" = torch.ops.aten.add.Tensor(clone, unsqueeze_1);  clone = unsqueeze_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:206 in mamba_selective_scan, code: dt = F.softplus(dt)
        exp_2: "f32[1, 1536, 128]" = torch.ops.aten.exp.default(add_1)
        log1p: "f32[1, 1536, 128]" = torch.ops.aten.log1p.default(exp_2);  exp_2 = None
        gt: "b8[1, 1536, 128]" = torch.ops.aten.gt.Scalar(add_1, 20)
        where: "f32[1, 1536, 128]" = torch.ops.aten.where.self(gt, add_1, log1p);  gt = add_1 = log1p = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:213 in mamba_selective_scan, code: discrete_A = torch.exp(A[None, :, None, :] * dt[:, :, :, None])
        unsqueeze_2: "f32[1, 1536, 16]" = torch.ops.aten.unsqueeze.default(neg, 0);  neg = None
        unsqueeze_3: "f32[1, 1536, 1, 16]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
        unsqueeze_4: "f32[1, 1536, 128, 1]" = torch.ops.aten.unsqueeze.default(where, 3)
        mul: "f32[1, 1536, 128, 16]" = torch.ops.aten.mul.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
        exp_3: "f32[1, 1536, 128, 16]" = torch.ops.aten.exp.default(mul);  mul = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:214 in mamba_selective_scan, code: discrete_B = dt[:, :, :, None] * B[:, None, :, :].float()
        unsqueeze_5: "f32[1, 1536, 128, 1]" = torch.ops.aten.unsqueeze.default(where, 3);  where = None
        unsqueeze_6: "f32[1, 1, 128, 16]" = torch.ops.aten.unsqueeze.default(getitem_3, 1);  getitem_3 = None
        mul_1: "f32[1, 1536, 128, 16]" = torch.ops.aten.mul.Tensor(unsqueeze_5, unsqueeze_6);  unsqueeze_5 = unsqueeze_6 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:215 in mamba_selective_scan, code: deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
        unsqueeze_7: "f32[1, 1536, 128, 1]" = torch.ops.aten.unsqueeze.default(permute_8, 3)
        mul_2: "f32[1, 1536, 128, 16]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_7);  mul_1 = unsqueeze_7 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:236 in mamba_selective_scan, code: _, all_states = associative_scan(
        permute_13: "f32[128, 1, 1536, 16]" = torch.ops.aten.permute.default(exp_3, [2, 0, 1, 3]);  exp_3 = None
        permute_14: "f32[128, 1, 1536, 16]" = torch.ops.aten.permute.default(mul_2, [2, 0, 1, 3]);  mul_2 = None
        associative_scan_combine_graph_0 = self.associative_scan_combine_graph_0
        associative_scan = torch.ops.higher_order.associative_scan(associative_scan_combine_graph_0, [permute_13, permute_14], ());  associative_scan_combine_graph_0 = permute_13 = permute_14 = None
        getitem_6: "f32[128, 1, 1536, 16]" = associative_scan[1];  associative_scan = None
        permute_16: "f32[1, 1536, 128, 16]" = torch.ops.aten.permute.default(getitem_6, [1, 2, 0, 3]);  getitem_6 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:243 in mamba_selective_scan, code: scan_output = torch.matmul(all_states.transpose(1, 2).to(input_dtype), C.unsqueeze(-1))
        permute_17: "f32[1, 128, 1536, 16]" = torch.ops.aten.permute.default(permute_16, [0, 2, 1, 3]);  permute_16 = None
        unsqueeze_8: "f32[1, 128, 16, 1]" = torch.ops.aten.unsqueeze.default(getitem_4, -1);  getitem_4 = None
        expand: "f32[1, 128, 1536, 16]" = torch.ops.aten.expand.default(permute_17, [1, 128, 1536, 16]);  permute_17 = None
        view_6: "f32[128, 1536, 16]" = torch.ops.aten.view.default(expand, [128, 1536, 16]);  expand = None
        expand_1: "f32[1, 128, 16, 1]" = torch.ops.aten.expand.default(unsqueeze_8, [1, 128, 16, 1]);  unsqueeze_8 = None
        view_7: "f32[128, 16, 1]" = torch.ops.aten.view.default(expand_1, [128, 16, 1]);  expand_1 = None
        bmm: "f32[128, 1536, 1]" = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = view_7 = None
        view_8: "f32[1, 128, 1536, 1]" = torch.ops.aten.view.default(bmm, [1, 128, 1536, 1]);  bmm = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:244 in mamba_selective_scan, code: scan_output = scan_output.squeeze(-1).transpose(1, 2)
        squeeze_1: "f32[1, 128, 1536]" = torch.ops.aten.squeeze.dim(view_8, -1);  view_8 = None
        permute_18: "f32[1, 1536, 128]" = torch.ops.aten.permute.default(squeeze_1, [0, 2, 1]);  squeeze_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:270 in mamba_selective_scan, code: scan_output = scan_output + hidden_states * D[None, :, None]
        unsqueeze_9: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(arg8_1, 0);  arg8_1 = None
        unsqueeze_10: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        mul_3: "f32[1, 1536, 128]" = torch.ops.aten.mul.Tensor(permute_8, unsqueeze_10);  permute_8 = unsqueeze_10 = None
        add_2: "f32[1, 1536, 128]" = torch.ops.aten.add.Tensor(permute_18, mul_3);  permute_18 = mul_3 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:273 in mamba_selective_scan, code: scan_output = scan_output * F.silu(z)
        neg_2: "f32[1, 1536, 128]" = torch.ops.aten.neg.default(getitem_1)
        exp_4: "f32[1, 1536, 128]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_3: "f32[1, 1536, 128]" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_1: "f32[1, 1536, 128]" = torch.ops.aten.div.Tensor(getitem_1, add_3);  getitem_1 = add_3 = None
        mul_4: "f32[1, 1536, 128]" = torch.ops.aten.mul.Tensor(add_2, div_1);  add_2 = div_1 = None

        # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:481 in forward, code: contextualized_states = self.out_proj(scan_output.transpose(1, 2).to(dtype))
        permute_19: "f32[1, 128, 1536]" = torch.ops.aten.permute.default(mul_4, [0, 2, 1]);  mul_4 = None
        permute_20: "f32[1536, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        view_9: "f32[128, 1536]" = torch.ops.aten.view.default(permute_19, [128, 1536]);  permute_19 = None
        mm_3: "f32[128, 768]" = torch.ops.aten.mm.default(view_9, permute_20);  view_9 = permute_20 = None
        view_10: "f32[1, 128, 768]" = torch.ops.aten.view.default(mm_3, [1, 128, 768]);  mm_3 = None
        return (view_10,)

    class associative_scan_combine_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[1, 1536, 16]", arg1_1: "f32[1, 1536, 16]", arg2_1: "f32[1, 1536, 16]", arg3_1: "f32[1, 1536, 16]"):
            # File: /u/jsaravananindira/vllm/.venv/lib/python3.12/site-packages/transformers/models/mamba/modeling_mamba.py:233 in combine_fn, code: return a_left * a_right, a_right * b_left + b_right
            mul: "f32[1, 1536, 16]" = torch.ops.aten.mul.Tensor(arg0_1, arg2_1);  arg0_1 = None
            mul_1: "f32[1, 1536, 16]" = torch.ops.aten.mul.Tensor(arg2_1, arg1_1);  arg2_1 = arg1_1 = None
            add: "f32[1, 1536, 16]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
            return [mul, add]
