# Copyright (c) 2026, FlashAttention contributors.

import os

import torch


def _load_vllm_unified_attention():
    from flash_attn.ops.triton.kernel_unified_attention import unified_attention

    return unified_attention


def _next_power_of_2(x):
    return 1 << (x - 1).bit_length()


def can_use_triton_unified_attention(
    q,
    k_cache,
    v_cache,
    k,
    v,
    rotary_cos,
    rotary_sin,
    cache_seqlens,
    cache_batch_idx,
    cache_leftpad,
    block_table,
    alibi_slopes,
    causal,
    window_size,
    softcap,
    s_aux,
):
    if os.environ.get("FLASH_ATTN_DISABLE_TRITON_UNIFIED_ATTENTION"):
        return False
    if os.environ.get("FLASH_ATTN_FORCE_FLASH_KVCACHE"):
        return False
    if block_table is None or cache_seqlens is None:
        return False
    if q.shape[-1] != 512:
        return False
    if not causal:
        return False
    if k is not None or v is not None:
        return False
    if rotary_cos is not None or rotary_sin is not None:
        return False
    if cache_batch_idx is not None or cache_leftpad is not None:
        return False
    if alibi_slopes is not None or softcap != 0.0 or s_aux is not None:
        return False
    if window_size != (-1, -1):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        return False
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        return False
    if block_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        return False
    if block_table.stride(-1) != 1 or cache_seqlens.stride(-1) != 1:
        return False
    if q.shape[0] != block_table.shape[0] or q.shape[0] != cache_seqlens.shape[0]:
        return False
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        return False
    if k_cache.shape != v_cache.shape:
        return False
    if k_cache.shape[-1] != 512:
        return False
    if q.shape[2] % k_cache.shape[2] != 0:
        return False
    return True


def triton_unified_attention_with_kvcache(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_table,
    softmax_scale,
):
    unified_attention = _load_vllm_unified_attention()

    batch_size, seqlen_q, nheads_q, head_dim = q.shape
    assert head_dim == 512

    q_ua = q.reshape(batch_size * seqlen_q, nheads_q, head_dim)
    out_ua = torch.empty_like(q_ua)
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device
    )

    unified_attention(
        q=q_ua,
        k=k_cache,
        v=v_cache,
        out=out_ua,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=seqlen_q,
        seqused_k=cache_seqlens,
        max_seqlen_k=int(cache_seqlens.max().item()),
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    return out_ua.reshape(batch_size, seqlen_q, nheads_q, head_dim)


def triton_unified_attention_varlen_paged(
    q,
    k_cache,
    v_cache,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    block_table,
    softmax_scale,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
):
    unified_attention = _load_vllm_unified_attention()
    out = torch.empty_like(q)
    unified_attention(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
        softcap=softcap,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    return out
