import argparse
import csv
import os
from contextlib import contextmanager
from datetime import datetime

import torch


@contextmanager
def route(mode):
    old_disable = os.environ.get("FLASH_ATTN_DISABLE_TRITON_UNIFIED_ATTENTION")
    if mode == "flash":
        os.environ["FLASH_ATTN_DISABLE_TRITON_UNIFIED_ATTENTION"] = "1"
    else:
        os.environ.pop("FLASH_ATTN_DISABLE_TRITON_UNIFIED_ATTENTION", None)
    try:
        yield
    finally:
        if old_disable is None:
            os.environ.pop("FLASH_ATTN_DISABLE_TRITON_UNIFIED_ATTENTION", None)
        else:
            os.environ["FLASH_ATTN_DISABLE_TRITON_UNIFIED_ATTENTION"] = old_disable


def make_case(batch_size, seqlen_k, nheads_q, nheads_k, page_block_size, dtype):
    num_logical_blocks = (seqlen_k + page_block_size - 1) // page_block_size
    num_blocks = max(1024, batch_size * num_logical_blocks)
    num_blocks = ((num_blocks + batch_size - 1) // batch_size) * batch_size

    q = torch.randn(batch_size, 1, nheads_q, 512, device="cuda", dtype=dtype)
    k_cache = torch.randn(
        num_blocks, page_block_size, nheads_k, 512, device="cuda", dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.randperm(
        num_blocks, dtype=torch.int32, device="cuda"
    ).reshape(batch_size, -1)[:, :num_logical_blocks].contiguous()
    cache_seqlens = torch.full(
        (batch_size,), seqlen_k, dtype=torch.int32, device="cuda"
    )
    return q, k_cache, v_cache, block_table, cache_seqlens


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


def bandwidth_gbs(batch_size, seqlen_k, nheads_q, nheads_k, dtype, ms):
    elem_size = torch.empty((), dtype=dtype).element_size()
    q_bytes = batch_size * nheads_q * 512 * elem_size
    kv_bytes = batch_size * seqlen_k * nheads_k * 512 * elem_size * 2
    return (q_bytes + kv_bytes) / 1e9 / (ms / 1e3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--page-block-size", type=int, default=16)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--batch-sizes", default=None)
    parser.add_argument("--seq-lens", default=None)
    args = parser.parse_args()

    from flash_attn.flash_attn_interface import flash_attn_with_kvcache

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.batch_sizes is not None:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    else:
        batch_sizes = [1, 2, 4] if args.quick else [1, 2, 4, 8, 16, 32, 64, 128]
    if args.seq_lens is not None:
        seqlens_k = [int(x) for x in args.seq_lens.split(",")]
    else:
        seqlens_k = [512, 2048] if args.quick else [512, 1024, 2048, 4096, 8192, 16384]
    head_cases = [(8, 8), (16, 4)]
    modes = ["flash", "triton_unified"]

    csv_path = args.csv or f"benchmark_hdim512_kvcache_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "dtype",
                "batch_size",
                "seq_len_kv",
                "num_heads_q",
                "num_heads_kv",
                "headdim",
                "time_ms",
                "bandwidth_GB_s",
            ]
        )
        print(
            f"{'mode':>15} {'dtype':>6} {'B':>5} {'Sk':>8} "
            f"{'Hq':>4} {'Hkv':>4} {'ms':>10} {'GB/s':>12}"
        )
        for nheads_q, nheads_k in head_cases:
            for seqlen_k in seqlens_k:
                for batch_size in batch_sizes:
                    torch.manual_seed(0)
                    try:
                        q, k_cache, v_cache, block_table, cache_seqlens = make_case(
                            batch_size,
                            seqlen_k,
                            nheads_q,
                            nheads_k,
                            args.page_block_size,
                            dtype,
                        )
                    except Exception as exc:
                        for mode in modes:
                            writer.writerow(
                                [
                                    mode,
                                    args.dtype,
                                    batch_size,
                                    seqlen_k,
                                    nheads_q,
                                    nheads_k,
                                    512,
                                    "ERROR",
                                    repr(exc),
                                ]
                            )
                        print(
                            f"{'alloc':>15} {args.dtype:>6} {batch_size:>5} "
                            f"{seqlen_k:>8} {nheads_q:>4} {nheads_k:>4} "
                            f"{'ERROR':>10} {repr(exc)}"
                        )
                        torch.cuda.empty_cache()
                        continue
                    for mode in modes:
                        def fn():
                            flash_attn_with_kvcache(
                                q,
                                k_cache,
                                v_cache,
                                cache_seqlens=cache_seqlens,
                                block_table=block_table,
                                causal=True,
                                num_splits=1,
                            )

                        try:
                            with route(mode):
                                ms = time_ms(fn, args.warmup, args.repeat)
                            bw = bandwidth_gbs(
                                batch_size, seqlen_k, nheads_q, nheads_k, dtype, ms
                            )
                            writer.writerow(
                                [
                                    mode,
                                    args.dtype,
                                    batch_size,
                                    seqlen_k,
                                    nheads_q,
                                    nheads_k,
                                    512,
                                    f"{ms:.6f}",
                                    f"{bw:.2f}",
                                ]
                            )
                            print(
                                f"{mode:>15} {args.dtype:>6} {batch_size:>5} "
                                f"{seqlen_k:>8} {nheads_q:>4} {nheads_k:>4} "
                                f"{ms:>10.6f} {bw:>12.2f}"
                            )
                        except Exception as exc:
                            writer.writerow(
                                [
                                    mode,
                                    args.dtype,
                                    batch_size,
                                    seqlen_k,
                                    nheads_q,
                                    nheads_k,
                                    512,
                                    "ERROR",
                                    repr(exc),
                                ]
                            )
                            print(
                                f"{mode:>15} {args.dtype:>6} {batch_size:>5} "
                                f"{seqlen_k:>8} {nheads_q:>4} {nheads_k:>4} "
                                f"{'ERROR':>10} {repr(exc)}"
                            )
                    del q, k_cache, v_cache, block_table, cache_seqlens
                    torch.cuda.empty_cache()
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
