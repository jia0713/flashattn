# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union

import os
import torch
import torch.nn as nn

from flash_attn.ops.triton.unified_attention import (
    can_use_triton_unified_attention,
    triton_unified_attention_with_kvcache,
)

# isort: off
# We need to import the CUDA kernels after importing torch
import flash_attn_2_cuda as flash_attn_cuda

# isort: on
if 'MHA_INPUT_COLLECTION' in os.environ:
    from flash_attn.tuning.modules.yaml_writer import YamlWriter

def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    # This should match the fwd block sizes in the CUDA kernel
    # If the block sizes not match, the test will failed when dropout=True
    is_xcore1000_series = torch.cuda.get_device_capability("cuda") == (8, 0)
    is_xcore1500_series = torch.cuda.get_device_capability("cuda") == (8, 6)
    assert (is_xcore1000_series or is_xcore1500_series), "Arch xcore must in 1000 series or 1500 series"
    assert head_dim <= 256

    if is_xcore1000_series:
        if head_dim <= 32:
            return 128
        if head_dim <= 64:
            return 64
        elif head_dim <= 96:
            return 64
        elif head_dim <= 128:
            return 64
        elif head_dim <= 160:
            return 32
        elif head_dim <= 192:
            return 64
        elif head_dim <= 224:
            return 64
        elif head_dim <= 256:
            return 64
    elif is_xcore1500_series:
        if head_dim <= 32:
            return 64
        if head_dim <= 64:
            return 64
        elif head_dim <= 96:
            return 64
        elif head_dim <= 128:
            return 64
        elif head_dim <= 160:
            return 64
        elif head_dim <= 192:
            return 64
        elif head_dim <= 224:
            return 64
        elif head_dim <= 256:
            return 64


def _get_block_size(device, head_dim, is_dropout, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32:
        return 128, 128
    if head_dim <= 64:
        return (128, 128) if not is_dropout else (128, 64)
    elif head_dim <= 96:
        return (64, 64) if (is_sm8x and is_causal) else (128, 64)
    elif head_dim <= 128:
        if is_sm8x:
            return (64, 64) if (not is_dropout and is_causal) else (128, 32)
        else:
            return 128, (64 if not is_dropout else 32)
    elif head_dim <= 160:
        if is_sm8x:
            return (128, 64) if not is_causal else (64, 64)
        else:
            return 128, 32
    elif head_dim <= 192:
        return (128, 64) if not is_dropout else (64, 64)
    elif head_dim <= 224:
        return (128, 64) if (is_sm80 or is_sm90) else (64, 64)
    elif head_dim <= 256:
        return (128, 64) if is_sm80 else (64, 64)

def bert_attn_mask_convert(q, k, attn_mask):
    if attn_mask is not None:
        bs, sq, nh, _ = q.shape
        _, sk, _, _ = k.shape
        if attn_mask.shape != (sq, sk) and attn_mask.shape != (1, sk) and \
            attn_mask.shape != (sq, 1) and attn_mask.shape == (bs, sk) :
            print(f"got an *deprecated* attn_mask shape {bs, sk}, It looks like you're using a *deprecated* bert attn_mask api ? please use correct shape {bs, 1, 1, sk}")
            attn_mask = attn_mask.reshape(bs, 1, 1, sk)

    return attn_mask

def _flash_attn_forward(
    q, k, v, dropout_p, softmax_scale, causal, window_size, softcap=0.0, alibi_slopes=None, return_softmax=False, s_aux=None, attn_mask=None, return_max_logit=False,
):
    attn_mask = bert_attn_mask_convert(q,k,attn_mask)
    maybe_contiguous = lambda x: x.contiguous() if (x is not None) and (x.stride(-1) != 1) else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if (attn_mask is not None):
        attn_mask = attn_mask.contiguous()

    if 'MHA_INPUT_COLLECTION' in os.environ:
        writer = YamlWriter()
        if writer:
            qkvpacked,kvpacked = writer.is_packed(q,k)
            problem = dict(
                q = q,
                v = v,
                k = k,
                qkvpacked = qkvpacked,
                kvpacked = kvpacked,
                alibi_slopes = alibi_slopes,
                attn_mask = attn_mask,
                dropout_p = dropout_p,
                softmax_scale = softmax_scale,
                causal = causal,
                window_size_left = window_size[0],
                window_size_right = window_size[1],
                softcap = softcap,
                return_softmax = return_softmax
            )
            writer.write("_flash_attn_forward", q.dtype == torch.bfloat16, problem)
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, attn_mask, max_logit = flash_attn_cuda.fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        attn_mask,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        return_softmax,
        None,
        s_aux,
        return_max_logit,
    )
    if return_max_logit:
        return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, attn_mask, max_logit
    else:
        return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, attn_mask


def _flash_attn_varlen_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    return_softmax=False,
    block_table=None,
    leftpad_k=None,
    seqused_k=None,
    s_aux=None,
    return_max_logit=False,
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    if 'MHA_INPUT_COLLECTION' in os.environ:
        writer = YamlWriter()
        if writer:
            qkvpacked,kvpacked = writer.is_packed(q,k,is_varlen = True)
            problem = dict(
                q = q,
                k = k,
                v = v,
                qkvpacked = qkvpacked,
                kvpacked = kvpacked,
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_k = cu_seqlens_k,
                seqused_k=seqused_k,
                leftpad_k=leftpad_k,
                block_table=block_table,
                alibi_slopes = alibi_slopes,
                max_seqlen_q = max_seqlen_q,
                max_seqlen_k = max_seqlen_k,
                dropout_p = dropout_p,
                softmax_scale = softmax_scale,
                zero_tensors = False,
                causal = causal,
                window_size_left = window_size[0],
                window_size_right = window_size[1],
                softcap = softcap,
                return_softmax = return_softmax
            )
            writer.write("_flash_attn_varlen_forward", q.dtype == torch.bfloat16, problem)
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, max_logit = flash_attn_cuda.varlen_fwd(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        return_softmax,
        None,
        s_aux,
        return_max_logit,
    )
    if return_max_logit:
        return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, max_logit
    else:
        return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


def _flash_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    rng_state=None,
    attn_mask=None,
):
    attn_mask = bert_attn_mask_convert(q,k,attn_mask)
    maybe_contiguous = lambda x: x.contiguous() if (x is not None) and (x.stride(-1) != 1) else x
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    if (attn_mask is not None):
        attn_mask = attn_mask.contiguous()

    if 'MHA_INPUT_COLLECTION' in os.environ:
        writer = YamlWriter()
        if writer:
            qkvpacked,kvpacked = writer.is_packed(q,k)
            problem = dict(
                dout = dout,
                q = q,
                k = k,
                v = v,
                qkvpacked = qkvpacked,
                kvpacked = kvpacked,
                out = out,
                softmax_lse = softmax_lse,
                dq = dq,
                dk = dk,
                dv = dv,
                alibi_slopes = alibi_slopes,
                attn_mask = attn_mask,
                dropout_p = dropout_p,
                softmax_scale = softmax_scale,
                causal = causal,
                window_size_left = window_size[0],
                window_size_right = window_size[1],
                softcap = softcap,
                deterministic = deterministic,
                gen = None,
                rng_state = rng_state
            )
            writer.write("_flash_attn_backward", q.dtype == torch.bfloat16, problem)
    dq, dk, dv, softmax_d, = flash_attn_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        alibi_slopes,
        attn_mask,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        deterministic,
        None,
        rng_state,
    )
    return dq, dk, dv, softmax_d


def _flash_attn_varlen_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    rng_state=None,
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]

    if 'MHA_INPUT_COLLECTION' in os.environ:
        writer = YamlWriter()
        if writer:
            qkvpacked,kvpacked = writer.is_packed(q, k, is_varlen = True)
            problem = dict(
                dout = dout,
                q = q,
                k = k,
                v = v,
                out = out,
                qkvpacked = qkvpacked,
                kvpacked = kvpacked,
                softmax_lse = softmax_lse,
                dq = dq,
                dk = dk,
                dv = dv,
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_k = cu_seqlens_k,
                seqused_k = None,
                alibi_slopes = alibi_slopes,
                max_seqlen_q = max_seqlen_q,
                max_seqlen_k = max_seqlen_k,
                dropout_p = dropout_p,
                softmax_scale = softmax_scale,
                zero_tensors = False,
                causal = causal,
                window_size_left = window_size[0],
                window_size_right = window_size[1],
                softcap = softcap,
                deterministic = deterministic,
                gen = None,
                rng_state = rng_state
            )
            writer.write("_flash_attn_varlen_backward", q.dtype == torch.bfloat16, problem)
    dq, dk, dv, softmax_d, = flash_attn_cuda.varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        deterministic,
        None,
        rng_state,
    )
    # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return dq, dk, dv, softmax_d


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        attn_mask,
        deterministic,
        return_softmax,
        s_aux,
        return_max_logit,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        results = _flash_attn_forward(
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            attn_mask=attn_mask,
            return_softmax=return_softmax and dropout_p > 0,
            s_aux=s_aux,
            return_max_logit=return_max_logit,
        )
        if return_max_logit:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, attn_mask, max_logit = results
        else:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, attn_mask = results
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap=softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.attn_mask = attn_mask
        ctx.deterministic = deterministic
        if not return_softmax:
            if return_max_logit:
                return (out, max_logit)
            else:
                return out
        else:
            return (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        IS_MHA_INIT_DEVICE_MALLOC = os.getenv("MHA_INIT_DEVICE_MALLOC", "False") == "True"
        dqkv = {}
        if IS_MHA_INIT_DEVICE_MALLOC:
            dqkv = torch.zeros(qkv_shape, dtype=q.dtype, device=q.device)
        else:
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dqkv[:, :, 0],
            dqkv[:, :, 1],
            dqkv[:, :, 2],
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes,
            attn_mask=ctx.attn_mask,
            deterministic=ctx.deterministic,
            rng_state=rng_state,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        s_aux,
        return_max_logit,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        results = _flash_attn_varlen_forward(
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            s_aux=s_aux,
            return_max_logit=return_max_logit,
        )
        if return_max_logit:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, max_logit = results
        else:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = results
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap=softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        if not return_softmax:
            if return_max_logit:
                return (out, max_logit)
            else:
                return out
        else:
            return (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        IS_MHA_INIT_DEVICE_MALLOC = os.getenv("MHA_INIT_DEVICE_MALLOC", "False") == "True"
        dqkv = {}
        if IS_MHA_INIT_DEVICE_MALLOC:
            dqkv = torch.zeros(qkv_shape, dtype=q.dtype, device=q.device)
        else:
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            rng_state=rng_state,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        s_aux,
        return_max_logit,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        results = _flash_attn_forward(
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            attn_mask=None,
            return_softmax=return_softmax and dropout_p > 0,
            s_aux=s_aux,
            return_max_logit=return_max_logit,
        )
        if return_max_logit:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, attn_mask, max_logit = results
        else:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, attn_mask = results
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap=softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.attn_mask = None
        ctx.deterministic = deterministic
        if not return_softmax:
            if return_max_logit:
                return (out, max_logit)
            else:
                return out
        else:
            return (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        IS_MHA_INIT_DEVICE_MALLOC = os.getenv("MHA_INIT_DEVICE_MALLOC", "False") == "True"
        dq = {}
        dkv = {}
        if IS_MHA_INIT_DEVICE_MALLOC:
            dq = torch.zeros_like(q)
            dkv = torch.zeros(kv_shape, dtype=k.dtype, device=k.device)
        else:
            dq = torch.empty_like(q)
            dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[:, :, 0],
            dkv[:, :, 1],
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes,
            attn_mask=ctx.attn_mask,
            deterministic=ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        s_aux,
        return_max_logit,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        results = _flash_attn_varlen_forward(
            q,
            kv[:, 0],
            kv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            s_aux=s_aux,
            return_max_logit=return_max_logit,
        )
        if return_max_logit:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, max_logit = results
        else:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = results
        ctx.save_for_backward(
            q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        if not return_softmax:
            if return_max_logit:
                return (out, max_logit)
            else:
                return out
        else:
            return (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        IS_MHA_INIT_DEVICE_MALLOC = os.getenv("MHA_INIT_DEVICE_MALLOC", "False") == "True"
        dq = {}
        dkv = {}
        if IS_MHA_INIT_DEVICE_MALLOC:
            dq = torch.zeros_like(q)
            dkv = torch.zeros(kv_shape, dtype=k.dtype, device=k.device)
        else:
            dq = torch.empty_like(q)
            dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[:, 0],
            dkv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        attn_mask,
        deterministic,
        return_softmax,
        s_aux,
        return_max_logit,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        ctx.qk_headdim = q.shape[-1]
        results = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            attn_mask=attn_mask,
            return_softmax=return_softmax and dropout_p > 0,
            s_aux=s_aux,
            return_max_logit=return_max_logit,
        )
        if return_max_logit:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, attn_mask, max_logit = results
        else:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, attn_mask = results
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.attn_mask = attn_mask
        ctx.deterministic = deterministic
        if not return_softmax:
            if return_max_logit:
                return (out, max_logit)
            else:
                return out
        else:
            return (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        IS_MHA_INIT_DEVICE_MALLOC = os.getenv("MHA_INIT_DEVICE_MALLOC", "False") == "True"
        dq = {}
        dk = {}
        dv = {}
        if IS_MHA_INIT_DEVICE_MALLOC:
            dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        else:
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes,
            attn_mask=ctx.attn_mask,
            deterministic=ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : ctx.qk_headdim]  # We could have padded the head dimension
        dk = dk[..., : ctx.qk_headdim]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        block_table,
        s_aux,
        return_max_logit,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        ctx.qk_headdim = q.shape[-1]
        results = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
            s_aux=s_aux,
            return_max_logit=return_max_logit,
        )
        if return_max_logit:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, max_logit = results
        else:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = results
        ctx.save_for_backward(
            q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        if not return_softmax:
            if return_max_logit:
                return (out, max_logit)
            else:
                return out
        else:
            return (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        IS_MHA_INIT_DEVICE_MALLOC = os.getenv("MHA_INIT_DEVICE_MALLOC", "False") == "True"
        dq = {}
        dk = {}
        dv = {}
        if IS_MHA_INIT_DEVICE_MALLOC:
            dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        else:
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : ctx.qk_headdim]  # We could have padded the head dimension
        dk = dk[..., : ctx.qk_headdim]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    attn_mask=None,
    deterministic=False,
    return_attn_probs=False,
    softcap=0.0,  # <=0.0 means deactivate
    s_aux=None,
    return_max_logit=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
            the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        softcap: float. Anything > 0 activates softcapping attention.
        s_aux [optional]: (nheads) bf16 tensor. A attention sink value per head that is appended as an
           additional attention logit to each query's attention scores before softmax.
        return_max_logit: bool.Whether to return the attention max logit.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        max_logit: [optional, if return_max_logit=True]: (nheads).fp32 tensor The max of
            per-head the matrix QK^T * scaling.
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        attn_mask,
        deterministic,
        return_attn_probs,
        s_aux,
        return_max_logit,
    )


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    softcap=0.0,  # <=0.0 means deactivate
    s_aux=None,
    return_max_logit=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        kv: (batch_size, seqlen, 2, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        softcap: float. Anything > 0 activates softcapping attention.
        s_aux [optional]: (nheads) bf16 tensor. A attention sink value per head that is appended as an
           additional attention logit to each query's attention scores before softmax.
        return_max_logit: bool.Whether to return the attention max logit.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        max_logit: [optional, if return_max_logit=True]: (nheads).fp32 tensor The max of
            per-head the matrix QK^T * scaling.
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        s_aux,
        return_max_logit,
    )


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    attn_mask=None,
    softcap=0.0,  # 0.0 means deactivated
    s_aux=None,
    return_max_logit=False
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        softcap: float. Anything > 0 activates softcapping attention.
        s_aux [optional]: (nheads) bf16 tensor. A attention sink value per head that is appended as an
           additional attention logit to each query's attention scores before softmax.
        return_max_logit: bool.Whether to return the attention max logit.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        max_logit: [optional, if return_max_logit=True]: (nheads).fp32 tensor The max of
            per-head the matrix QK^T * scaling.
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        attn_mask,
        deterministic,
        return_attn_probs,
        s_aux,
        return_max_logit,
    )


def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    softcap=0.0,  # 0.0 means deactivated
    s_aux=None,
    return_max_logit=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_varlen_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_varlen_kvpacked_func and flash_attn_varlen_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        softcap: float. Anything > 0 activates softcapping attention.
        s_aux [optional]: (nheads) bf16 tensor. A attention sink value per head that is appended as an
           additional attention logit to each query's attention scores before softmax.
        return_max_logit: bool.Whether to return the attention max logit.
    Return:
        out: (total, nheads, headdim).
        max_logit: [optional, if return_max_logit=True]: (nheads).fp32 tensor The max of
            per-head the matrix QK^T * scaling.
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenQKVPackedFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        s_aux,
        return_max_logit,
    )


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    softcap=0.0,  # 0.0 means deactivated
    s_aux=None,
    return_max_logit=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        softcap: float. Anything > 0 activates softcapping attention.
        s_aux [optional]: (nheads) bf16 tensor. A attention sink value per head that is appended as an
           additional attention logit to each query's attention scores before softmax.
        return_max_logit: bool.Whether to return the attention max logit.
    Return:
        out: (total, nheads, headdim).
        max_logit: [optional, if return_max_logit=True]: (nheads).fp32 tensor The max of
            per-head the matrix QK^T * scaling.
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        s_aux,
        return_max_logit,
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    softcap=0.0,  # 0.0 means deactivated
    block_table=None,
    s_aux=None,
    return_max_logit=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        If block_table is not none, the shape changes to (num_blocks, page_block_size, nheads_k, headdim)
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        If block_table is not none, the shape changes to (num_blocks, page_block_size, nheads_k, headdim)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        softcap: float. Anything > 0 activates softcapping attention.
        block_table: Optional block_table of dtype int32 and shape [batch_size, num_blocks_per_seq] used for paged attention.
        s_aux [optional]: (nheads) bf16 tensor. A attention sink value per head that is appended as an
           additional attention logit to each query's attention scores before softmax.
        return_max_logit: bool.Whether to return the attention max logit.
    Return:
        out: (total, nheads, headdim).
        max_logit: [optional, if return_max_logit=True]: (nheads).fp32 tensor The max of
            per-head the matrix QK^T * scaling.
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        s_aux,
        return_max_logit,
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    softcap=0.0,  # 0.0 means deactivated
    s_aux=None,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        softcap: float. Anything > 0 activates softcapping attention.
        s_aux [optional]: (nheads) bf16 tensor. A attention sink value per head that is appended as an
           additional attention logit to each query's attention scores before softmax.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)

    if 'MHA_INPUT_COLLECTION' in os.environ:
        writer = YamlWriter()
        if writer:
            problem = dict(
                q = q,
                k_cache = k_cache,
                v_cache = v_cache,
                k = k,
                v = v,
                seqlens_k = cache_seqlens,
                rotary_cose = rotary_cos,
                rotary_sin = rotary_sin,
                cache_batch_idx = cache_batch_idx,
                block_table = block_table,
                alibi_slopes = alibi_slopes,
                softmax_scale = softmax_scale,
                causal = causal,
                window_size_left = window_size[0],
                window_size_right = window_size[1],
                rotary_interleaved = rotary_interleaved,
                num_splits = num_splits,
                softcap = softcap
            )
            writer.write("flash_attn_with_kvcache", q.dtype == torch.bfloat16, problem)
    if can_use_triton_unified_attention(
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
        return triton_unified_attention_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            block_table,
            softmax_scale,
        )
    out, softmax_lse = flash_attn_cuda.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
        s_aux,
    )
    return out

def flash_attn_with_kvcache_dequant(
    q,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    dequant_group=-1,
    softcap=0.0,  # 0.0 means deactivated
):

    """
    Performs attention with INT8 key/value cache, dequanting kv cache with fp16/bf16 key/value scale.

    the dequantization is applied as:
    k_cache = k_cache * kscale, v_cache = v_cache * vscale.
    Note:
        Currently only supports cached key/value (k and v must be None)
        Dequantization group size must be 8 elements for hardware acceleration compatibility

    Arguments:
        q: (batch_size, seqlen, nheads, headdim), torch.float16/bfloat16
        k_cache: (num_blocks, page_block_size, nheads_k, headdim) with block_table,
                torch.int8. Page block size must be multiple of 256.
        v_cache: (num_blocks, page_block_size, nheads_k, headdim) with block_table,
                torch.int8
        kscale: (num_blocks, page_block_size, nheads_k, headdim/dequant_group), torch.float16/bfloat16. Dequantization scale for k_cache
        vscale: (num_blocks, page_block_size, nheads_k, headdim/dequant_group), torch.float16/bfloat16. Dequantization scale for v_cache
        cache_seqlens: int or (batch_size,), torch.int32. Current sequence lengths in KV cache
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32
        softmax_scale: float. QK^T scaling before softmax. Default: 1/sqrt(headdim)
        causal: bool. Applies causal mask if True
        window_size: (left, right). Sliding window attention when != (-1, -1)
        num_splits: int. Key/value chunking control. 0 for auto-determination.

    Return:
        out: (batch_size, seqlen, nheads, headdim)
        softmax_lse [optional]: (batch_size, nheads, seqlen). Logsumexp values
    """


    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)

    out, softmax_lse = flash_attn_cuda.fwd_kvcache_dequant(
        q,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
        dequant_group
    )
    return out
