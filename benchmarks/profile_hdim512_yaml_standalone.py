#!/usr/bin/env python3
# Standalone headdim=512 paged varlen unified-attention profiler.
# Runtime dependencies: torch, triton. PyYAML is optional.

import argparse
import ast
import csv
import math
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


class KVQuantMode(IntEnum):
    NONE = 0
    FP8 = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3


is_batch_invariant = False
float8_info = torch.finfo(torch.float8_e4m3fn)


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared ``@triton.jit`` helpers used by the unified attention kernel
and ``reduce_segments``.

These are plain attention-loop helpers — mask building, ALiBi / QQ-bias
score post-processing, online-softmax bookkeeping, tile-loop bounds,
sequence lookup — extracted so the 2D and 3D paths of the unified
kernel (and any future consumer) share a single implementation.
"""


# ===========================================================================
# Scalar helpers (reused by every kernel + reduce_segments)
# ===========================================================================


@triton.jit
def cdiv_fn(x, y):
    """Ceiling division.  Kept as a helper to keep kernel bodies terse."""
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    """Softcap (aka tanh-style clamp) used to bound attention scores.

    ``x * tanh(S / x)`` rewritten to avoid a direct ``tanh`` call.
    """
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


# ===========================================================================
# Attention loop
# ===========================================================================


@triton.jit
def resolve_seq_and_query_len(
    query_start_len_ptr,
    seq_lens_ptr,
    q_block_global_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    """Resolve the (sequence, q-block-within-sequence) pair and load the
    per-sequence lengths.

    Shared across every attention kernel — the ``q_block_global_idx``
    program id indexes into the flattened ``(seq, q_block_in_seq)``
    space, and a binary search over ``query_start_len_ptr`` recovers
    the (seq, local-q-block) pair.

    Returns ``(seq_idx, q_block_local_idx, cur_batch_in_all_start_index,
    cur_batch_query_len, seq_len)``.  Callers must still early-return
    when ``q_block_local_idx * BLOCK_Q >= cur_batch_query_len`` (Triton
    helpers cannot return from the caller).
    """
    # find_seq_idx is defined below; forward use is fine inside @triton.jit.
    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_start = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    return seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    """Binary search over the cumulative query-length prefix.

    When ``use_q_block_mode`` is True, the prefix values are reshaped
    into units of ``BLOCK_Q`` plus one entry per boundary — matching
    the q-block grid laid out by the attention kernels.  When False
    we search the plain cumulative-length prefix (used by
    ``reduce_segments`` which iterates over raw query tokens).
    """
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def init_softmax_M(
    sink_ptr,
    query_offset_1,
    query_mask_1,
    segm_idx_or_0,
    BLOCK_M: tl.constexpr,
    USE_SINKS: tl.constexpr,
    IS_3D: tl.constexpr,
):
    """Initial row-max ``M`` for the online softmax.

    Without sinks: ``-inf``.  With sinks: load the per-head sink bias
    once.  In 3D mode only segment 0 loads — ``reduce_segments`` adds
    the sink contribution exactly once across segments, so other
    segments must start from ``-inf``.

    ``segm_idx_or_0`` is the 3D segment index or 0 for 2D (caller
    passes ``0`` when ``IS_3D`` is False).
    """
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    if USE_SINKS:
        load_sinks = (not IS_3D) or (segm_idx_or_0 == 0)
        if load_sinks:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(tl.float32)
    return M


@triton.jit
def compute_tile_loop_bounds(
    context_len,
    seq_len,
    cur_batch_query_len,
    q_block_local_idx,
    segm_idx_or_0,
    tiles_per_segment_or_0,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    IS_3D: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    """Compute the tile-loop bounds ``(loop_lo, loop_hi)`` and the
    derived ``max_seq_prefix_len`` used for per-tile masking.

    Combines three concerns into one helper:

    1. Longest prefix spanned by any query token in this q-block.
       Clamped to ``seq_len`` (causal) or extended to it when
       mm_prefix is active (bidirectional ranges can reach past the
       causal prefix).
    2. Sliding-window pruning: narrows ``[tile_start, tile_end)`` to
       only tiles that can contain an allowed key under SWA.
    3. 3D scoping: when ``IS_3D`` is True, further narrows to the
       segment's slice via ``(segm_idx * tiles_per_segment,
       (segm_idx + 1) * tiles_per_segment)``.
    """
    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    if USE_MM_PREFIX:
        # image bidirectional attention ranges require a full range
        # including q_block padding to make sure doc mask is correct
        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
    else:
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        q_abs = context_len + qpos_lo
        if CHUNK_LOOKBACK > -1:
            # Chunked attention: align lower bound to the start of the
            # lookback'th previous chunk.
            first_allowed_key = ((q_abs // CHUNK_SIZE) - CHUNK_LOOKBACK) * CHUNK_SIZE
        else:
            first_allowed_key = q_abs - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    if IS_3D:
        loop_lo = max(segm_idx_or_0 * tiles_per_segment_or_0, tile_start)
        loop_hi = min((segm_idx_or_0 + 1) * tiles_per_segment_or_0, tile_end)
    else:
        loop_lo = tile_start
        loop_hi = tile_end

    return loop_lo, loop_hi, max_seq_prefix_len


@triton.jit
def store_segm_reduce_scalars(
    segm_max_ptr,
    segm_expsum_ptr,
    query_offset_0,
    query_offset_1,
    segm_idx,
    M,
    L,
    query_mask_0,
    query_mask_1,
    num_query_heads: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """Store per-segment ``M`` and ``L`` for ``reduce_segments`` to
    combine into the final softmax.

    Shared across every 3D attention epilogue; the per-token output
    stripes are mode-specific (flat / 2-stream split / 4-stream split)
    and stay inlined.
    """
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def compute_kv_seq_mask(
    query_abs_pos,
    seq_offset,
    seq_idx,
    mm_prefix_range_ptr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    """Build the KV mask for one tile.

    Causal (key <= query) by default; AND-ed with either chunked
    attention (``CHUNK_LOOKBACK >= 0``) or sliding window
    (``SLIDING_WINDOW > 0``); OR-ed with the bidirectional ranges from
    ``mm_prefix_range`` when PrefixLM / multimodal attention is active.
    Order matches FlexAttention: ``(causal AND window) OR mm_prefix``.
    Chunked attention takes precedence over sliding window when both
    are non-default — the launcher zeros ``CHUNK_LOOKBACK`` whenever
    sliding window is disabled.
    """
    # Compute attention mask: causal by default (key <= query)
    seq_mask = seq_offset[None, :] <= query_abs_pos

    # Apply sliding window / chunked attention to base mask
    # BEFORE mm_prefix OR.
    # Order must match FlexAttention:
    #   (causal AND sliding_window) OR mm_prefix
    if CHUNK_LOOKBACK > -1:
        seq_mask = seq_mask & (
            (query_abs_pos // CHUNK_SIZE - seq_offset[None, :] // CHUNK_SIZE)
            <= CHUNK_LOOKBACK
        )
    elif SLIDING_WINDOW > 0:
        seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

    # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
    # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
    if USE_MM_PREFIX:
        for i in range(MAX_MM_RANGES):
            range_start = tl.load(
                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
            )
            range_end = tl.load(
                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
            )
            is_valid = range_start < range_end
            q_in_range = (
                (query_abs_pos >= range_start) & (query_abs_pos <= range_end) & is_valid
            )
            k_in_range = (
                (seq_offset[None, :] >= range_start)
                & (seq_offset[None, :] <= range_end)
                & is_valid
            )
            seq_mask |= q_in_range & k_in_range
    return seq_mask


@triton.jit
def apply_alibi_to_score(
    S,
    alibi_slope,
    seq_offset,
    context_len,
    query_pos,
    USE_ALIBI_SQRT: tl.constexpr,
):
    """Add the ALiBi positional bias (linear or sqrt variant) to S in-place."""
    if USE_ALIBI_SQRT:
        relative_pos = seq_offset - (context_len + query_pos[:, None])
        alibi_offset = tl.where(
            relative_pos <= 0,
            -tl.sqrt((-relative_pos).to(tl.float32)),
            0.0,
        )
    else:
        alibi_offset = seq_offset - context_len
    return S + alibi_slope[:, None] * alibi_offset


@triton.jit
def load_qq_bias_tile(
    qq_bias_row_ptrs,
    seq_offset,
    context_len,
    qq_bias_stride_0,
):
    """Load the qq-bias slice for keys that correspond to query rows."""
    key_rel_pos = seq_offset - context_len
    is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
    return tl.load(
        qq_bias_row_ptrs + key_rel_pos[None, :],
        mask=is_query_key[None, :],
        other=0.0,
    )


@triton.jit
def softmax_step(S, M, L):
    """Online softmax update for one tile.

    Returns ``(M_new, L_new, P, alpha)``.  Caller is responsible for
    rescaling its accumulator(s) by ``alpha[:, None]`` — done outside so
    kernels with a different number / shape of accumulators can reuse
    the same step.
    """
    # compute running maximum
    # m_j : (BLOCK_M,)
    m_j = tl.maximum(M, tl.max(S, axis=1))
    # For sliding window there's a chance the max is -inf due to masking of
    # the entire row. In this case we need to set m_j 0 to avoid NaN
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    # P : (BLOCK_M, TILE_SIZE)
    P = tl.exp(S - m_j[:, None])
    # l_j : (BLOCK_M,)
    l_j = tl.sum(P, axis=1)
    # alpha : (BLOCK_M, )
    alpha = tl.exp(M - m_j)
    # update constants
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha


@triton.jit
def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    """Cast a loaded KV tile to Q's dtype, dequantizing if needed.

    Modes handled inside the core kernel:

    - ``KV_QUANT_MODE == 0`` (NONE) and ``2`` (INT8 per-token-head) and
      ``3`` (FP8 per-token-head): plain cast.  Per-token-head modes apply
      their scales separately on S/P inside the loop.
    - ``KV_QUANT_MODE == 1`` (FP8 per-tensor): dequantize using the
      tensor-wide scale.
    """
    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)


@triton.jit
def kernel_unified_attention(
    # Output destinations.  In 2D mode we write the final result into
    # ``output_ptr``; in 3D mode we write per-segment partials into the
    # three ``segm_*`` tensors and ``output_ptr`` is unused (callers may
    # pass any non-null pointer).
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    # Inputs
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    # Per-(token, head) scale caches (used iff KV_QUANT_MODE in {2, 3}).
    # For other modes callers may pass any non-null pointer.
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    # Scalars
    scale,
    k_scale,
    v_scale,
    out_scale,
    softcap,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    # Toggles 2D vs 3D layout.  The 2D path runs the full sequence in one
    # tile loop and writes to ``output_ptr``.  The 3D path scopes the loop
    # to ``[segm_idx, segm_idx+1) × tiles_per_segment`` and writes
    # per-segment partials, finalized by ``reduce_segments``.
    IS_3D: tl.constexpr,
    # KV cache quantization mode handled inside this kernel via constexpr
    # branches: NONE (0), FP8_PER_TENSOR (1), INT8_PER_TOKEN_HEAD (2),
    # FP8_PER_TOKEN_HEAD (3).
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    # Chunked / block-local attention.  ``CHUNK_LOOKBACK >= 0`` enables
    # chunked masking (used by Gemma3 block-local layers); takes precedence
    # over ``SLIDING_WINDOW`` inside the helpers.  ``-1`` disables.
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
    )

    # iterate through tiles (now limited to the sliding window range)
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )
        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)

        # Per-(token, head) scales for INT8 / FP8 per-token-head modes.
        if USE_PER_TOKEN_HEAD_SCALES:
            scale_idx = (
                physical_block_idx * stride_ks_blk
                + (seq_offset % BLOCK_SIZE) * stride_ks_slot
                + kv_head_idx * stride_ks_head
            )
            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )
            v_scale_idx = (
                physical_block_idx * stride_vs_blk
                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
                + kv_head_idx * stride_vs_head
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: fuse softmax_scale with per-head k_scale
            # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
            S += tl.dot(Q, K) * (scale * k_token_head_scales[None, :])
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue ---------------------------------------------------------
    if IS_3D:
        # Store per-segment partials; finalized by ``reduce_segments``.
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
            + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
        )
        tl.store(
            segm_output_ptr + segm_output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        if USE_FP8:
            acc = acc * tl.load(out_scale)
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
        output_offset = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_d[None, :]
        )
        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization."""
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill and head_size >= 512:
        return 16
    if is_prefill:
        return 32
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    return 16 if element_size >= 2 else 32


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    use_alibi_sqrt=False,
    # KV cache quantization mode and per-token-head scale caches.
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE,
    k_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    # Chunked attention: restrict attention to aligned blocks with lookback.
    chunk_lookback=-1,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_per_token_head_scales = kv_quant_mode in (
        KVQuantMode.INT8_PER_TOKEN_HEAD,
        KVQuantMode.FP8_PER_TOKEN_HEAD,
    )
    if use_per_token_head_scales:
        assert k_scale_cache is not None and v_scale_cache is not None, (
            f"{kv_quant_mode.name} requires k_scale_cache / v_scale_cache"
        )

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Compute chunked block size from sliding window if needed.
    chunk_size = -1
    if sliding_window_val > 0 and chunk_lookback > -1:
        chunk_size = sliding_window_val // (chunk_lookback + 1)
        assert chunk_size > 0, "sliding_window must be > chunk_lookback+1"
    elif sliding_window_val <= 0:
        chunk_lookback = -1

    TILE_SIZE_PREFILL = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=True
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes at least one prefill request, or
    # 3. The number of sequences exceeds the configured threshold, or
    # 4. Batch invariance is enabled
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    # The kernel signature is the same for 2D and 3D — only the launch
    # grid + a handful of constexpr toggles differ.  Per-token-head scale
    # caches and their strides are required arguments; non-per-token-head
    # modes pass dummy zeros (the code path is dead-code eliminated by
    # the ``USE_PER_TOKEN_HEAD_SCALES`` constexpr branch in the kernel).
    if use_per_token_head_scales:
        ks_strides = k_scale_cache.stride()
        vs_strides = v_scale_cache.stride()
        ks_blk, ks_slot, ks_head = ks_strides[0], ks_strides[1], ks_strides[2]
        vs_blk, vs_slot, vs_head = vs_strides[0], vs_strides[1], vs_strides[2]
        k_scale_ptr = k_scale_cache
        v_scale_ptr = v_scale_cache
    else:
        ks_blk = ks_slot = ks_head = 0
        vs_blk = vs_slot = vs_head = 0
        # Pass the K cache as a stand-in pointer; never dereferenced.
        k_scale_ptr = k
        v_scale_ptr = v

    # 3D needs real segm tensors; 2D never touches them but Triton wants
    # a non-null pointer.  Reuse ``out`` as the placeholder.
    segm_output_ptr = softmax_segm_output if use_3d else out
    segm_max_ptr = softmax_segm_max if use_3d else out
    segm_expsum_ptr = softmax_segm_expsum if use_3d else out
    num_segments = num_par_softmax_segments if use_3d else 1

    grid: tuple[Any, ...]
    if not use_3d:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL
    else:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE

    kernel_unified_attention[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        k_scale_cache_ptr=k_scale_ptr,
        v_scale_cache_ptr=v_scale_ptr,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
        SLIDING_WINDOW=(1 + window_size[0]),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        stride_ks_blk=ks_blk,
        stride_ks_slot=ks_slot,
        stride_ks_head=ks_head,
        stride_vs_blk=vs_blk,
        stride_vs_slot=vs_slot,
        stride_vs_head=vs_head,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=output_scale is not None,
        IS_3D=use_3d,
        KV_QUANT_MODE=kv_quant_mode,
        CHUNK_LOOKBACK=chunk_lookback,
        CHUNK_SIZE=chunk_size,
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )


def _parse_scalar(value):
    value = value.strip()
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value in ("null", "None"):
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def load_yaml_cases(path):
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return data or {}
    except ModuleNotFoundError:
        pass

    cases = {}
    current_name = None
    current_key = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" ") and line.endswith(":"):
                current_name = line[:-1]
                cases[current_name] = {}
                current_key = None
                continue
            if current_name is None:
                continue
            stripped = line.strip()
            if stripped.startswith("-"):
                if current_key is None:
                    raise ValueError(f"list item without key: {raw!r}")
                cases[current_name][current_key].append(_parse_scalar(stripped[1:]))
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                cases[current_name][key] = _parse_scalar(value)
                current_key = None
            else:
                cases[current_name][key] = []
                current_key = key
    return cases


def dtype_from_case(case, override):
    if override == "bf16":
        return torch.bfloat16, "bf16"
    if override == "fp16":
        return torch.float16, "fp16"
    if case.get("is_fp16", False):
        return torch.float16, "fp16"
    return torch.bfloat16, "bf16"


def _int_list(values):
    return [int(x) for x in values]


def _prefix_sum(lengths):
    out = [0]
    for length in lengths:
        out.append(out[-1] + int(length))
    return out


def normalize_case(case):
    case = dict(case)
    api = str(case.get("api", ""))
    is_kvcache = api == "flash_attn_with_kvcache"
    bsz = int(case["batch_size"]) if "batch_size" in case else None

    if is_kvcache and bsz is not None:
        if "max_seqlen_q" in case:
            case["seqlens_q"] = [int(case["max_seqlen_q"])] * bsz
        if "max_seqlen_kv" in case:
            case["seqlens_kv"] = [int(case["max_seqlen_kv"])] * bsz

    if "seqlens_q" not in case and "max_seqlen_q" in case and "batch_size" in case:
        case["seqlens_q"] = [int(case["max_seqlen_q"])] * int(case["batch_size"])
    if "seqlens_kv" not in case:
        if "seqlens_k" in case:
            case["seqlens_kv"] = case["seqlens_k"]
        elif "max_seqlen_kv" in case and "batch_size" in case:
            case["seqlens_kv"] = [int(case["max_seqlen_kv"])] * int(case["batch_size"])
    if "batch_size" in case and "seqlens_q" in case and len(case["seqlens_q"]) == 1:
        case["seqlens_q"] = [int(case["seqlens_q"][0])] * int(case["batch_size"])
    if "batch_size" in case and "seqlens_kv" in case and len(case["seqlens_kv"]) == 1:
        if int(case["seqlens_kv"][0]) == 0 and "max_seqlen_kv" in case:
            case["seqlens_kv"] = [int(case["max_seqlen_kv"])] * int(case["batch_size"])
        else:
            case["seqlens_kv"] = [int(case["seqlens_kv"][0])] * int(case["batch_size"])
    if "cu_seqlens_q" not in case and "seqlens_q" in case:
        case["cu_seqlens_q"] = _prefix_sum(case["seqlens_q"])
    if "cu_seqlens_kv" not in case and "seqlens_kv" in case:
        case["cu_seqlens_kv"] = _prefix_sum(case["seqlens_kv"])
    if "max_seqlen_q" not in case and "seqlens_q" in case:
        case["max_seqlen_q"] = max(_int_list(case["seqlens_q"]))
    if "max_seqlen_kv" not in case and "seqlens_kv" in case:
        case["max_seqlen_kv"] = max(_int_list(case["seqlens_kv"]))
    if "batch_size" not in case and "cu_seqlens_q" in case:
        case["batch_size"] = len(case["cu_seqlens_q"]) - 1
    if "seqlens_q" in case and int(case.get("total_seqlens_q", 0)) <= 0:
        case["total_seqlens_q"] = sum(_int_list(case["seqlens_q"]))
    if "seqlens_kv" in case and int(case.get("total_seqlens_kv", 0)) <= 0:
        case["total_seqlens_kv"] = sum(_int_list(case["seqlens_kv"]))
    return case


def paged_case_status(case):
    required = [
        "batch_size",
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "seqlens_q",
        "seqlens_kv",
        "num_heads_q",
        "num_heads_kv",
        "head_dim",
        "paged_block_size",
    ]
    missing = [key for key in required if key not in case]
    if missing:
        return False, f"missing required paged attention fields: {','.join(missing)}"
    paged = bool(case.get("paged_kv", True)) or "paged_block_size" in case
    if not paged:
        return False, "non-paged attention is not supported by this profiler"
    if not bool(case.get("causal", True)):
        return False, "Triton unified profiler only supports causal attention"
    bsz = int(case["batch_size"])
    if len(case["cu_seqlens_q"]) != bsz + 1:
        return False, "len(cu_seqlens_q) must be batch_size + 1"
    if len(case["cu_seqlens_kv"]) != bsz + 1:
        return False, "len(cu_seqlens_kv) must be batch_size + 1"
    if int(case["num_heads_q"]) % int(case["num_heads_kv"]) != 0:
        return False, "num_heads_q must be divisible by num_heads_kv"
    return True, "ok"


def make_varlen_paged_case(case, dtype, device):
    torch.manual_seed(int(case.get("hash_code", 0)) & 0x7FFFFFFF)
    hq = int(case["num_heads_q"])
    hkv = int(case["num_heads_kv"])
    d = int(case["head_dim"])
    dv = int(case.get("head_dim_v", d))
    pbs = int(case["paged_block_size"])
    bsz = int(case["batch_size"])
    total_q = int(case.get("total_seqlens_q", -1))
    if total_q <= 0:
        total_q = sum(int(x) for x in case["seqlens_q"])
    seqlens_k = [int(x) for x in case["seqlens_kv"]]
    num_blocks_per_seq = [math.ceil(x / pbs) for x in seqlens_k]
    max_blocks_per_seq = max(num_blocks_per_seq)
    num_blocks = max(int(case.get("paged_num_blocks", 0)), sum(num_blocks_per_seq), 1)

    q = torch.randn(total_q, hq, d, device=device, dtype=dtype)
    k = torch.randn(num_blocks, pbs, hkv, d, device=device, dtype=dtype)
    v = torch.randn(num_blocks, pbs, hkv, dv, device=device, dtype=dtype)
    block_table = torch.empty((bsz, max_blocks_per_seq), device=device, dtype=torch.int32)
    offset = 0
    for b, nblocks in enumerate(num_blocks_per_seq):
        blocks = torch.arange(offset, offset + nblocks, device=device, dtype=torch.int32)
        if nblocks < max_blocks_per_seq:
            blocks = torch.nn.functional.pad(blocks, (0, max_blocks_per_seq - nblocks))
        block_table[b] = blocks
        offset += nblocks
    cu_q = torch.tensor(case["cu_seqlens_q"], device=device, dtype=torch.int32)
    cu_k = torch.tensor(case["cu_seqlens_kv"], device=device, dtype=torch.int32)
    return q, k, v, cu_q, cu_k, block_table


def varlen_paged_reference(q, k, v, cu_q, cu_k, block_table, causal, scale):
    bsz = cu_q.numel() - 1
    hq = q.shape[1]
    hkv = k.shape[2]
    groups = hq // hkv
    pbs = k.shape[1]
    outs = []
    for b in range(bsz):
        q0, q1 = int(cu_q[b].item()), int(cu_q[b + 1].item())
        k0, k1 = int(cu_k[b].item()), int(cu_k[b + 1].item())
        sq, sk = q1 - q0, k1 - k0
        nblocks = math.ceil(sk / pbs)
        blocks = block_table[b, :nblocks]
        kb = k[blocks].reshape(-1, hkv, k.shape[-1])[:sk]
        vb = v[blocks].reshape(-1, hkv, v.shape[-1])[:sk]
        qb = q[q0:q1]
        per_head = []
        for h in range(hq):
            hk = h // groups
            scores = torch.matmul(qb[:, h, :].float(), kb[:, hk, :].float().transpose(0, 1))
            scores = scores * scale
            if causal:
                q_pos = torch.arange(sq, device=q.device) + (sk - sq)
                k_pos = torch.arange(sk, device=q.device)
                scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            per_head.append(torch.matmul(probs, vb[:, hk, :].float()).to(q.dtype))
        outs.append(torch.stack(per_head, dim=1))
    return torch.cat(outs, dim=0)


def run_triton_unified_varlen(
    q,
    k,
    v,
    cu_q,
    cu_k,
    max_q,
    max_k,
    block_table,
    case,
):
    seqused_k = cu_k[1:] - cu_k[:-1]
    out = torch.empty_like(q)
    unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_q,
        max_seqlen_q=max_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_k,
        softmax_scale=float(case.get("softmax_scale") or (int(case["head_dim"]) ** -0.5)),
        causal=bool(case.get("causal", False)),
        window_size=(int(case.get("window_left", -1)), int(case.get("window_right", -1))),
        block_table=block_table,
        softcap=float(case.get("softcap", 0.0)),
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    return out


def run_flash_varlen(q, k, v, cu_q, cu_k, max_q, max_k, block_table, case):
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=bool(case.get("causal", False)),
        window_size=(int(case.get("window_left", -1)), int(case.get("window_right", -1))),
        alibi_slopes=None,
        deterministic=bool(case.get("deterministic", False)),
        return_attn_probs=False,
        softcap=float(case.get("softcap", 0.0)),
        block_table=block_table,
    )


def flash_backend_available(require=False):
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func  # noqa: F401

        return True, ""
    except Exception as exc:
        if require:
            raise
        return False, repr(exc)


def time_ms(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def attention_tflops(case, ms):
    hq = int(case["num_heads_q"])
    d = int(case["head_dim"])
    seqlens_q = [int(x) for x in case["seqlens_q"]]
    seqlens_k = [int(x) for x in case["seqlens_kv"]]
    flops = 4 * hq * d * sum(sq * sk for sq, sk in zip(seqlens_q, seqlens_k))
    return flops / (ms * 1e-3) / 1e12


def shape_info(case):
    return {
        "count": int(case.get("count", 1)),
        "hash_code": case.get("hash_code", ""),
        "batch_size": int(case["batch_size"]),
        "batch_size_c": int(case.get("batch_size_c", case["batch_size"])),
        "total_seqlens_q": int(case.get("total_seqlens_q", sum(case["seqlens_q"]))),
        "total_seqlens_kv": int(case.get("total_seqlens_kv", sum(case["seqlens_kv"]))),
        "max_seqlen_q": int(case["max_seqlen_q"]),
        "max_seqlen_kv": int(case["max_seqlen_kv"]),
        "seqlens_q": ";".join(str(int(x)) for x in case["seqlens_q"]),
        "seqlens_kv": ";".join(str(int(x)) for x in case["seqlens_kv"]),
        "num_heads_q": int(case["num_heads_q"]),
        "num_heads_kv": int(case["num_heads_kv"]),
        "head_dim": int(case["head_dim"]),
        "head_dim_v": int(case.get("head_dim_v", case["head_dim"])),
        "paged_block_size": int(case["paged_block_size"]),
        "paged_num_blocks": int(case.get("paged_num_blocks", 0)),
        "causal": bool(case.get("causal", False)),
        "window_left": int(case.get("window_left", -1)),
        "window_right": int(case.get("window_right", -1)),
        "softcap": float(case.get("softcap", 0.0)),
    }


def validate(out, ref, dtype, atol=None, rtol=None):
    default_atol = 5e-2 if dtype is torch.bfloat16 else 2e-2
    default_rtol = 5e-2 if dtype is torch.bfloat16 else 2e-2
    atol = default_atol if atol is None else atol
    rtol = default_rtol if rtol is None else rtol
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    return {
        "max_abs": diff.max().item(),
        "max_rel": (diff / denom).max().item(),
        "allclose": torch.allclose(out, ref, atol=atol, rtol=rtol),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default="benchmarks/headdim512.yaml")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--skip-reference-profile", action="store_true")
    parser.add_argument(
        "--backends",
        choices=["auto", "flash", "triton", "both"],
        default="auto",
        help="auto runs installed flash_attn on MetaX/MACA and Triton unified on CUDA/NVIDIA.",
    )
    parser.add_argument(
        "--flash-backend",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use installed flash_attn package if available. The script never imports flash-attn source code.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")

    device_name = torch.cuda.get_device_name()
    device_capability = ".".join(str(x) for x in torch.cuda.get_device_capability())
    is_maca_device = "MetaX" in device_name or "MACA" in device_name or "MUSA" in device_name
    cases = load_yaml_cases(args.yaml)
    flash_available, flash_error = flash_backend_available(
        require=args.flash_backend == "on"
    )
    csv_path = args.csv or f"hdim512_yaml_profile_{datetime.now():%Y%m%d_%H%M%S}.csv"
    fields = [
        "case",
        "backend",
        "dtype",
        "device_name",
        "device_capability",
        "api",
        "count",
        "hash_code",
        "batch_size",
        "batch_size_c",
        "total_seqlens_q",
        "total_seqlens_kv",
        "max_seqlen_q",
        "max_seqlen_kv",
        "seqlens_q",
        "seqlens_kv",
        "num_heads_q",
        "num_heads_kv",
        "head_dim",
        "head_dim_v",
        "paged_block_size",
        "paged_num_blocks",
        "causal",
        "window_left",
        "window_right",
        "softcap",
        "time_ms",
        "tflops",
        "max_abs",
        "max_rel",
        "allclose",
        "status",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, case in cases.items():
            case = normalize_case(case)
            dtype, dtype_name = dtype_from_case(case, args.dtype)
            supported, reason = paged_case_status(case)
            if not supported:
                row = {
                    "case": name,
                    "backend": "unsupported",
                    "dtype": dtype_name,
                    "api": case.get("api"),
                    "status": reason,
                }
                writer.writerow(row)
                print(row)
                continue

            q, k, v, cu_q, cu_k, block_table = make_varlen_paged_case(case, dtype, "cuda")
            scale = float(case.get("softmax_scale") or (int(case["head_dim"]) ** -0.5))
            ref = varlen_paged_reference(
                q,
                k,
                v,
                cu_q,
                cu_k,
                block_table,
                bool(case.get("causal", False)),
                scale,
            )
            torch.cuda.synchronize()
            common = {
                "case": name,
                "dtype": dtype_name,
                "device_name": device_name,
                "device_capability": device_capability,
                "api": case.get("api"),
                **shape_info(case),
            }

            if not args.skip_reference_profile:
                ms = time_ms(
                    lambda: varlen_paged_reference(
                        q,
                        k,
                        v,
                        cu_q,
                        cu_k,
                        block_table,
                        bool(case.get("causal", False)),
                        scale,
                    ),
                    args.warmup,
                    args.repeat,
                )
                row = {
                    **common,
                    "backend": "torch_reference",
                    "time_ms": f"{ms:.6f}",
                    "tflops": f"{attention_tflops(case, ms):.6f}",
                    "max_abs": "0",
                    "max_rel": "0",
                    "allclose": True,
                    "status": "ok",
                }
                writer.writerow(row)
                print(row)

            if args.backends == "auto":
                requested_backends = ["flash"] if is_maca_device else ["triton"]
            elif args.backends == "both":
                requested_backends = ["flash", "triton"]
            else:
                requested_backends = [args.backends]

            backends = []
            if "flash" in requested_backends and args.flash_backend != "off" and flash_available:
                backends.append(("flash_attn_varlen", run_flash_varlen))
            elif "flash" in requested_backends and args.flash_backend != "off":
                row = {
                    **common,
                    "backend": "flash_attn_varlen",
                    "time_ms": "",
                    "tflops": "",
                    "max_abs": "",
                    "max_rel": "",
                    "allclose": "",
                    "status": f"SKIPPED: installed flash_attn is unavailable: {flash_error}",
                }
                writer.writerow(row)
                print(row)
            if "triton" in requested_backends:
                backends.append(("triton_unified_attention", run_triton_unified_varlen))
            for backend_name, backend_fn in backends:
                try:
                    out = backend_fn(
                        q,
                        k,
                        v,
                        cu_q,
                        cu_k,
                        int(case["max_seqlen_q"]),
                        int(case["max_seqlen_kv"]),
                        block_table,
                        case,
                    )
                    torch.cuda.synchronize()
                    metrics = validate(out, ref, dtype, args.atol, args.rtol)
                    ms = time_ms(
                        lambda: backend_fn(
                            q,
                            k,
                            v,
                            cu_q,
                            cu_k,
                            int(case["max_seqlen_q"]),
                            int(case["max_seqlen_kv"]),
                            block_table,
                            case,
                        ),
                        args.warmup,
                        args.repeat,
                    )
                    row = {
                        **common,
                        "backend": backend_name,
                        "time_ms": f"{ms:.6f}",
                        "tflops": f"{attention_tflops(case, ms):.6f}",
                        "max_abs": f"{metrics['max_abs']:.9g}",
                        "max_rel": f"{metrics['max_rel']:.9g}",
                        "allclose": metrics["allclose"],
                        "status": "ok",
                    }
                except Exception as exc:
                    row = {
                        **common,
                        "backend": backend_name,
                        "time_ms": "",
                        "tflops": "",
                        "max_abs": "",
                        "max_rel": "",
                        "allclose": "",
                        "status": f"ERROR: {exc!r}",
                    }
                writer.writerow(row)
                print(row)
            del q, k, v, cu_q, cu_k, block_table, ref
            torch.cuda.empty_cache()
    print(f"saved={csv_path}")


if __name__ == "__main__":
    main()
