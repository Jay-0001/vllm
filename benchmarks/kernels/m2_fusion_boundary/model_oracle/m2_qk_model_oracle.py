"""Experimental single-launch Q/K RoPE replacement for Qwen2.5-7B decode."""

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


@triton.jit
def _qk_kernel(
    q,
    k,
    positions,
    cache,
    q_out,
    k_out,
    tokens: tl.constexpr,
    q_width: tl.constexpr,
    k_width: tl.constexpr,
    q_stride: tl.constexpr,
    k_stride: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
):
    width = q_width + k_width
    half = head_dim // 2
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < tokens * width
    row = index // width
    col = index % width
    is_q = col < q_width
    source_col = tl.where(is_q, col, col - q_width)
    rotary_col = source_col % head_dim
    pair_col = tl.where(rotary_col < half, source_col + half, source_col - half)
    source = tl.where(is_q, q + row * q_stride + source_col,
                      k + row * k_stride + source_col)
    paired = tl.where(is_q, q + row * q_stride + pair_col,
                      k + row * k_stride + pair_col)
    pos = tl.load(positions + row, valid, other=0)
    x = tl.load(source, valid, other=0).to(tl.float32)
    xp = tl.load(paired, valid, other=0).to(tl.float32)
    phase = pos * head_dim + rotary_col % half
    cosine = tl.load(cache + phase, valid, other=0).to(tl.float32)
    sine = tl.load(cache + phase + half, valid, other=0).to(tl.float32)
    value = tl.where(rotary_col < half, x * cosine - xp * sine,
                     x * cosine + xp * sine)
    tl.store(q_out + row * q_width + col, value, valid & is_q)
    tl.store(k_out + row * k_width + col - q_width, value, valid & ~is_q)


@torch.library.custom_op("m2::qk_rope", mutates_args=())
def qk_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous rotated Q/K while leaving the QKV backing store intact."""
    tokens, q_width = query.shape
    k_width = key.shape[1]
    q_out = torch.empty((tokens, q_width), device=query.device, dtype=query.dtype)
    k_out = torch.empty((tokens, k_width), device=key.device, dtype=key.dtype)
    _qk_kernel[(triton.cdiv(tokens * (q_width + k_width), 256),)](
        query,
        key,
        positions,
        cache,
        q_out,
        k_out,
        tokens,
        q_width,
        k_width,
        query.stride(0),
        key.stride(0),
        128,
        256,
    )
    return q_out, k_out


@qk_rope.register_fake
def _qk_rope_fake(query, key, positions, cache):
    tokens, q_width = query.shape
    k_width = key.shape[1]
    return (
        torch.empty((tokens, q_width), device=query.device, dtype=query.dtype),
        torch.empty((tokens, k_width), device=key.device, dtype=key.dtype),
    )


def install():
    """Replace only the tested Qwen7B BF16 Neox full-RoPE decode case."""
    original = RotaryEmbedding.forward_static

    def selected(
        positions, query, key, head_size, rotary_dim, cos_sin_cache,
        is_neox_style,
    ):
        if (
            key is not None
            and head_size == rotary_dim == 128
            and is_neox_style
            and query.ndim == key.ndim == 2
            and query.shape[1] == 3584
            and key.shape[1] == 512
            and query.shape[0] <= 4
            and query.dtype == key.dtype == torch.bfloat16
            and query.is_cuda
            and cos_sin_cache.shape[1] == 128
        ):
            return qk_rope(query, key, positions.flatten(), cos_sin_cache)
        return original(
            positions, query, key, head_size, rotary_dim, cos_sin_cache,
            is_neox_style,
        )

    RotaryEmbedding.forward_static = staticmethod(selected)
