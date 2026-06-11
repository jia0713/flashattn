import argparse
import os
import sys
from contextlib import contextmanager

import torch


@contextmanager
def disable_triton_unified_attention():
    old = os.environ.get("FLASH_ATTN_DISABLE_TRITON_UNIFIED_ATTENTION")
    os.environ["FLASH_ATTN_DISABLE_TRITON_UNIFIED_ATTENTION"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("FLASH_ATTN_DISABLE_TRITON_UNIFIED_ATTENTION", None)
        else:
            os.environ["FLASH_ATTN_DISABLE_TRITON_UNIFIED_ATTENTION"] = old


def make_case(
    batch_size,
    seqlen_k,
    nheads_q,
    nheads_k,
    headdim,
    page_block_size,
    dtype,
    device,
):
    torch.manual_seed(0)
    num_logical_blocks = (seqlen_k + page_block_size - 1) // page_block_size
    num_blocks = max(1024, batch_size * num_logical_blocks)
    num_blocks = ((num_blocks + batch_size - 1) // batch_size) * batch_size

    q = torch.randn(batch_size, 1, nheads_q, headdim, device=device, dtype=dtype)
    k_cache = torch.randn(
        num_blocks, page_block_size, nheads_k, headdim, device=device, dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.randperm(
        num_blocks, dtype=torch.int32, device=device
    ).reshape(batch_size, -1)[:, :num_logical_blocks].contiguous()
    cache_seqlens = torch.full(
        (batch_size,), seqlen_k, dtype=torch.int32, device=device
    )
    return q, k_cache, v_cache, block_table, cache_seqlens


def run_one_case(
    batch_size,
    seqlen_k,
    nheads_q,
    nheads_k,
    dtype,
    page_block_size,
    atol,
    rtol,
):
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache

    device = "cuda"
    q, k_cache, v_cache, block_table, cache_seqlens = make_case(
        batch_size=batch_size,
        seqlen_k=seqlen_k,
        nheads_q=nheads_q,
        nheads_k=nheads_k,
        headdim=512,
        page_block_size=page_block_size,
        dtype=dtype,
        device=device,
    )

    with disable_triton_unified_attention():
        out_flash = flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True,
            num_splits=1,
        )
    out_triton = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
        num_splits=1,
    )
    torch.cuda.synchronize()

    diff = (out_triton.float() - out_flash.float()).abs()
    denom = out_flash.float().abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    ok = torch.allclose(out_triton, out_flash, atol=atol, rtol=rtol)
    print(
        f"dtype={dtype} B={batch_size} Sk={seqlen_k} "
        f"Hq={nheads_q} Hkv={nheads_k} max_abs={max_abs:.6g} "
        f"max_rel={max_rel:.6g} allclose={ok}"
    )
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-block-size", type=int, default=16)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "both"], default="both")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")

    dtypes = []
    if args.dtype in ("bf16", "both"):
        dtypes.append(torch.bfloat16)
    if args.dtype in ("fp16", "both"):
        dtypes.append(torch.float16)

    batch_sizes = [1, 2, 4] if args.quick else [1, 2, 4, 8, 16, 32, 64]
    seqlens_k = [512, 2048] if args.quick else [512, 1024, 2048, 4096, 8192, 16384]
    head_cases = [(8, 8), (16, 4)]

    failures = 0
    for dtype in dtypes:
        default_atol = 5e-2 if dtype is torch.bfloat16 else 2e-2
        default_rtol = 5e-2 if dtype is torch.bfloat16 else 2e-2
        atol = args.atol if args.atol is not None else default_atol
        rtol = args.rtol if args.rtol is not None else default_rtol
        for nheads_q, nheads_k in head_cases:
            for seqlen_k in seqlens_k:
                for batch_size in batch_sizes:
                    ok = run_one_case(
                        batch_size=batch_size,
                        seqlen_k=seqlen_k,
                        nheads_q=nheads_q,
                        nheads_k=nheads_k,
                        dtype=dtype,
                        page_block_size=args.page_block_size,
                        atol=atol,
                        rtol=rtol,
                    )
                    failures += int(not ok)

    if failures:
        print(f"FAILED: {failures} cases did not match", file=sys.stderr)
        raise SystemExit(1)
    print("PASSED")


if __name__ == "__main__":
    main()
