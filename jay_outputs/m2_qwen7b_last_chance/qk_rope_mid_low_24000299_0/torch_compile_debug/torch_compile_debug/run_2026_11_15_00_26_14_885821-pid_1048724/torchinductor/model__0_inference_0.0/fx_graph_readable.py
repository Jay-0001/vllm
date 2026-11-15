class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8, 4608]", arg1_1: "i64[8]", arg2_1: "bf16[32768, 128]"):
        # File: /u/jsaravananindira/vllm/benchmarks/kernels/m2_fusion_boundary/qk_rope_oracle.py:29 in native_region, code: q, k, v = qkv.split((Q_WIDTH, K_WIDTH, K_WIDTH), dim=-1)
        split_with_sizes = torch.ops.aten.split_with_sizes.default(arg0_1, [3584, 512, 512], -1);  arg0_1 = None
        getitem: "bf16[8, 3584]" = split_with_sizes[0]
        getitem_1: "bf16[8, 512]" = split_with_sizes[1]
        getitem_2: "bf16[8, 512]" = split_with_sizes[2];  split_with_sizes = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:173 in forward_static, code: cos_sin = cos_sin_cache.index_select(0, positions)
        index: "bf16[8, 128]" = torch.ops.aten.index.Tensor(arg2_1, [arg1_1]);  arg2_1 = arg1_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:174 in forward_static, code: cos, sin = cos_sin.chunk(2, dim=-1)
        split = torch.ops.aten.split.Tensor(index, 64, -1);  index = None
        getitem_3: "bf16[8, 64]" = split[0]
        getitem_4: "bf16[8, 64]" = split[1];  split = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:177 in forward_static, code: query = query.view(num_tokens, -1, head_size)
        view: "bf16[8, 28, 128]" = torch.ops.aten.view.default(getitem, [8, -1, 128]);  getitem = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:154 in forward_static, code: cos = cos.unsqueeze(-2).to(x.dtype)
        unsqueeze: "bf16[8, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem_3, -2)

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:155 in forward_static, code: sin = sin.unsqueeze(-2).to(x.dtype)
        unsqueeze_1: "bf16[8, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem_4, -2)

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:158 in forward_static, code: x1, x2 = torch.chunk(x, 2, dim=-1)
        split_1 = torch.ops.aten.split.Tensor(view, 64, -1);  view = None
        getitem_5: "bf16[8, 28, 64]" = split_1[0]
        getitem_6: "bf16[8, 28, 64]" = split_1[1];  split_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: o1 = x1 * cos - x2 * sin
        mul: "bf16[8, 28, 64]" = torch.ops.aten.mul.Tensor(getitem_5, unsqueeze)
        mul_1: "bf16[8, 28, 64]" = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze_1)
        sub: "bf16[8, 28, 64]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:164 in forward_static, code: o2 = x2 * cos + x1 * sin
        mul_2: "bf16[8, 28, 64]" = torch.ops.aten.mul.Tensor(getitem_6, unsqueeze);  getitem_6 = unsqueeze = None
        mul_3: "bf16[8, 28, 64]" = torch.ops.aten.mul.Tensor(getitem_5, unsqueeze_1);  getitem_5 = unsqueeze_1 = None
        add: "bf16[8, 28, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:167 in forward_static, code: output = torch.cat((o1, o2), dim=-1)
        cat: "bf16[8, 28, 128]" = torch.ops.aten.cat.default([sub, add], -1);  sub = add = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:186 in forward_static, code: query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        view_1: "bf16[8, 3584]" = torch.ops.aten.view.default(cat, [8, 3584]);  cat = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:191 in forward_static, code: key = key.view(num_tokens, -1, head_size)
        view_2: "bf16[8, 4, 128]" = torch.ops.aten.view.default(getitem_1, [8, -1, 128]);  getitem_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:154 in forward_static, code: cos = cos.unsqueeze(-2).to(x.dtype)
        unsqueeze_2: "bf16[8, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem_3, -2);  getitem_3 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:155 in forward_static, code: sin = sin.unsqueeze(-2).to(x.dtype)
        unsqueeze_3: "bf16[8, 1, 64]" = torch.ops.aten.unsqueeze.default(getitem_4, -2);  getitem_4 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:158 in forward_static, code: x1, x2 = torch.chunk(x, 2, dim=-1)
        split_2 = torch.ops.aten.split.Tensor(view_2, 64, -1);  view_2 = None
        getitem_7: "bf16[8, 4, 64]" = split_2[0]
        getitem_8: "bf16[8, 4, 64]" = split_2[1];  split_2 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:163 in forward_static, code: o1 = x1 * cos - x2 * sin
        mul_4: "bf16[8, 4, 64]" = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze_2)
        mul_5: "bf16[8, 4, 64]" = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_3)
        sub_1: "bf16[8, 4, 64]" = torch.ops.aten.sub.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:164 in forward_static, code: o2 = x2 * cos + x1 * sin
        mul_6: "bf16[8, 4, 64]" = torch.ops.aten.mul.Tensor(getitem_8, unsqueeze_2);  getitem_8 = unsqueeze_2 = None
        mul_7: "bf16[8, 4, 64]" = torch.ops.aten.mul.Tensor(getitem_7, unsqueeze_3);  getitem_7 = unsqueeze_3 = None
        add_1: "bf16[8, 4, 64]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/common.py:167 in forward_static, code: output = torch.cat((o1, o2), dim=-1)
        cat_1: "bf16[8, 4, 128]" = torch.ops.aten.cat.default([sub_1, add_1], -1);  sub_1 = add_1 = None

        # File: /u/jsaravananindira/vllm/vllm/model_executor/layers/rotary_embedding/base.py:200 in forward_static, code: key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        view_3: "bf16[8, 512]" = torch.ops.aten.view.default(cat_1, [8, 512]);  cat_1 = None
        return (view_1, view_3, getitem_2)
