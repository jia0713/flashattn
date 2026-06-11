# Copyright (c) 2026, FlashAttention contributors.

import os
import sys
import types
from pathlib import Path

import torch


def _load_vllm_unified_attention():
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "3rdparty" / "vllm",
        Path.home() / "workspace" / "vllm",
    ]
    for vllm_root in candidates:
        if vllm_root.exists():
            vllm_path = str(vllm_root)
            if vllm_path not in sys.path:
                sys.path.insert(0, vllm_path)
            break

    import triton
    import triton.language as tl

    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.HAS_TRITON = True
    triton_utils.triton = triton
    triton_utils.tl = tl
    triton_utils.LOG2E = 1.4426950408889634
    triton_utils.LOGE2 = 0.6931471805599453
    sys.modules["vllm.triton_utils"] = triton_utils

    from vllm.v1.attention.ops.triton_unified_attention import kernel_unified_attention

    return kernel_unified_attention, triton


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
    if q.shape[-1] != 512 or q.shape[1] != 1:
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
    kernel_unified_attention, triton = _load_vllm_unified_attention()

    batch_size, seqlen_q, nheads_q, head_dim = q.shape
    assert seqlen_q == 1
    assert head_dim == 512

    q_ua = q.reshape(batch_size, nheads_q, head_dim)
    out_ua = torch.empty_like(q_ua)
    cu_seqlens_q = torch.arange(
        batch_size + 1, dtype=torch.int32, device=q.device
    )
    nheads_k = k_cache.shape[2]
    num_queries_per_kv = nheads_q // nheads_k
    block_m = max(16, _next_power_of_2(num_queries_per_kv))
    block_q = 1
    total_num_q_blocks = q_ua.shape[0] // block_q + batch_size

    kernel_unified_attention[(total_num_q_blocks, nheads_k)](
        output_ptr=out_ua,
        segm_output_ptr=out_ua,
        segm_max_ptr=out_ua,
        segm_expsum_ptr=out_ua,
        query_ptr=q_ua,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        sink_ptr=None,
        block_tables_ptr=block_table,
        seq_lens_ptr=cache_seqlens,
        alibi_slopes_ptr=None,
        qq_bias_ptr=None,
        k_scale_cache_ptr=k_cache,
        v_scale_cache_ptr=v_cache,
        scale=softmax_scale,
        k_scale=None,
        v_scale=None,
        out_scale=1.0,
        softcap=0.0,
        num_query_heads=nheads_q,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q_ua.stride(0),
        query_stride_1=q_ua.stride(1),
        output_stride_0=out_ua.stride(0),
        output_stride_1=out_ua.stride(1),
        qq_bias_stride_0=0,
        BLOCK_SIZE=k_cache.shape[1],
        TILE_SIZE=16,
        HEAD_SIZE=head_dim,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_dim),
        USE_ALIBI_SLOPES=False,
        USE_ALIBI_SQRT=False,
        USE_QQ_BIAS=False,
        USE_SOFTCAP=False,
        USE_SINKS=False,
        SLIDING_WINDOW=0,
        USE_MM_PREFIX=False,
        MAX_MM_RANGES=0,
        mm_prefix_range_ptr=None,
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
        stride_ks_blk=0,
        stride_ks_slot=0,
        stride_ks_head=0,
        stride_vs_blk=0,
        stride_vs_slot=0,
        stride_vs_head=0,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        num_seqs=batch_size,
        BLOCK_M=block_m,
        NUM_SEGMENTS_PER_SEQ=1,
        USE_FP8=False,
        IS_3D=False,
        KV_QUANT_MODE=0,
        CHUNK_LOOKBACK=-1,
        CHUNK_SIZE=-1,
        num_stages=1,
    )
    return out_ua.reshape(batch_size, seqlen_q, nheads_q, head_dim)
