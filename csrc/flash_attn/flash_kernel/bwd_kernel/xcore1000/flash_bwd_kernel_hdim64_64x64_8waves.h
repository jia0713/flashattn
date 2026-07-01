/***************************************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cutlass/cutlass.h>
#include <mctlass/array.h>
#include <mctlass/numeric_types.h>
#include <mctlass/numeric_conversion.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "philox.cuh"

#include "alibi.h"
#include "bwd_hdim64_utils.h"

#include "attn_mask.h"


namespace flash {

using namespace cute;


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_first, bool Is_last, bool Seq_parallel=false, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock_hdim64_64x64_8waves(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(typename Kernel_traits::TiledMmaSdP::TiledShape_MNK{}))::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr int AtomLayoutNS = Kernel_traits::kNWarps / AtomLayoutMS;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);
    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    const index_t row_offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + ((m_block_max - 1) * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128ll * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded
        // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
        + (!params.deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);
    const index_t row_offset_lse =
        (params.unpadded_lse ? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
                             : (bidb * params.h + bidh) * params.seqlen_q) +
        (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum =
        (params.unpadded_lse
             ? bidh * (params.total_q + maxValidBlockSizeM * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb
             : (bidb * params.h + bidh) * params.seqlen_q_rounded) +
        (m_block_max - 1) * kBlockM;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.dq_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(params.h * params.d_rounded, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO_b128{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed_b128{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    // Double buffer for sQ
    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO_b128{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed_b128{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(),
                                                typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sK = make_tensor(Kernel_traits::Is_K_in_regs ? sQ.data():sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(Kernel_traits::Is_K_in_regs ? sdO.data() + size(sdO) : sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sdS = make_tensor(!Kernel_traits::Is_V_in_regs ? sV.data() + size(sV) : (Kernel_traits::Is_K_in_regs ? sdO.data() + size(sdO) : sK.data() + size(sK)),
                             typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    // sP and sdQ share the same memory so be careful
    Tensor sdQ = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutdQ{});
    Tensor sKt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO = std::conditional_t<
        Is_first,
        typename Kernel_traits::GmemTiledCopydO,
        typename Kernel_traits::GmemTiledCopyQKV
    >;

    typename Kernel_traits::GmemTiledCopyKV_4x2 gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    using GmemLayoutAtomdQaccum = std::conditional_t<
        !Seq_parallel,
        typename Kernel_traits::GmemTiledCopydQaccum,
        typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd
    >;
    GmemLayoutAtomdQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_KV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);
    // if (cute::thread0()) { print(tdQgdQaccum.layout()); printf("\n"); }
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx < 64) {
    //     printf("tidx = %d, tdQgdQaccum = 0x%p\n", tidx, tdQgdQaccum.data());
    // }

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);         // (MMA,MMA_N,MMA_K)
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);         // (MMA,MMA_N,MMA_K)
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);      // (MMA,MMA_N,MMA_K)
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);        // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);             // (MMA, MMA_M, MMA_K)
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);    // (MMA, MMA_N, MMA_K)

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_N
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_N

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    // if (cute::thread(0, 0) && n_block == 0) { printf("sK layout: "); print(sK.layout()); printf("\n"); }
    // if (cute::thread(0, 0) && n_block == 0) { print(tSsK.layout()); printf("\n"); }
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    // Partition sP and sdS to match the accumulator partitioning
    // This has to be tiled_mma_sdp, not tiled_mma_dkv
    auto smem_tiled_copy_PdS = make_tiled_copy_C(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)
    // if (cute::thread(0, 0) && n_block == 0) { printf("sP layout: "); print(sP.layout()); printf("\n"); }
    // if (cute::thread(0, 0) && n_block == 0) { print(tPsP.layout()); printf("\n"); }
    // if (n_block == 0 && blockIdx.x == 0 && blockIdx.y == 0 && tidx < 64) {
    //     printf("tidx=%d, tPsP = 0x%p\n", tidx, tPsP.data());
    // }
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);

    Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdSt);        // (MMA, MMA_M, MMA_K)
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle); // (MMA, MMA_N, MMA_K)

    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);            // (MMA, MMA_M, MMA_K)
    Tensor tdVrdOt = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle); // (MMA, MMA_N, MMA_K)

    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
    Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);

    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_KV.partition_D(cKV);

    // Prologue

    // We'll advance gdQ and gdQaccum before the 1st read/write.
    tdQgdQ.data() = tdQgdQ.data() + kBlockM * params.dq_row_stride;
    const index_t dq_accum_block_stride = index_t(kBlockM) * params.h * params.d_rounded;
    tdQgdQaccum.data() = tdQgdQaccum.data() + dq_accum_block_stride;

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);
    // If not local, we're guaranteed that m_block_min <= m_block:
    // We checked earlier that n_block * kBlockN < actual_seqlen_k, so in the causal case,
    // n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k < actual_seqlen_q.
    // So m_block_min <= (actual_seqlen_q - 1) / kBlockM.
    // Recall that m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM) = (actual_seqlen_q + kBlockM - 1) / kBlockM.
    // So m_block_m - 1 = (actual_seqlen_q - 1) / kBlockM.
    // We conclude that m_block_min <= m_block, so we will always have at least 1 iteration of the for loop.
    // However, if local, then this possible to have some blocks of K & V not attending to any query.
    // We might need to exit early and write 0 to dK and dV for those blocks.
    // Otherwise we get wrong result for the case where we don't enter the for loop.
    // And we might read OOB elements from gQ and gdO.
    // This also covers the case where actual_seqlen_q == 0
    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
          + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
          + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                 make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                 make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }

    if ((!Is_first && !Seq_parallel) || params.deterministic) { __syncthreads(); }

    // Clear the smem tiles to account for predicated off loads
    Tensor tVrV = make_fragment_like(tVsV);
    Tensor tKrK = make_fragment_like(tKsK);
    flash::copy_multirow_b64<Is_even_MN, Is_even_K>(
        tKgK,tKrK,tKVcKV,params.d,binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::copy_multirow_b64<Is_even_MN, Is_even_K>(
        tVgV,tVrV,tKVcKV,params.d,binfo.actual_seqlen_k - n_block * kBlockN
    );

    auto warp_id = threadIdx.x / 64;
    auto lane_id = threadIdx.x % 64;

    //////////////////////////////////////////////////////////////////////
    // Tensor P and dS when sts and lds
    // sts pointer
    auto row = lane_id / 16;
    auto col = lane_id % 16;
    col ^= (row * 4);

    Element *dStsmem_ptr_sts = reinterpret_cast<Element *>(sdS.data().get()) + warp_id * 4 * 64 + row * 64 + 4 * col;
    Tensor tdStsdSt = make_tensor(make_smem_ptr(dStsmem_ptr_sts), make_layout(Shape<_4, _1, _2>{},
                                                                          Stride<_1, _0, Int<32*64>>{}));
    Element *Ptsmem_ptr_sts = reinterpret_cast<Element *>(sP.data().get()) + warp_id * 4 * 64 + row * 64 + 4 * col;
    Tensor tPtsPt = make_tensor(make_smem_ptr(Ptsmem_ptr_sts), make_layout(Shape<_4, _1, _2>{},
                                                                          Stride<_1, _0, Int<32*64>>{}));

     // lds pointer
    auto warp_layout = tile_to_shape(Layout<Shape<_4,_4>,Stride<_4,_1>>{},Shape<_4,_16>{});
    auto coord = warp_layout.get_hier_coord(lane_id);
    auto lds_row = cute::get<0>(coord);
    auto col_coord = cute::get<1>(coord);
    auto lds_col = cute::get<0>(col_coord) + cute::get<1>(col_coord) * 4;
    lds_col ^= (lds_row * 4);

    Element *dStsmem_ptr_lds = reinterpret_cast<Element *>(sdS.data().get()) + warp_id % 2 * 32 * 64 + lds_row * 64 + lds_col * 4;
    Tensor tdKsdSt = make_tensor(make_smem_ptr(dStsmem_ptr_lds), make_layout(Shape<_4, _2, _4>{},
                                                                          Stride<_1, Int<16*64>, Int<4*64>>{}));
    Element *Ptsmem_ptr_lds = reinterpret_cast<Element *>(sP.data().get()) + warp_id % 2 * 32 * 64 + lds_row * 64 + lds_col * 4;
    Tensor tdVsPt = make_tensor(make_smem_ptr(Ptsmem_ptr_lds), make_layout(Shape<_4, _2, _4>{},
                                                                          Stride<_1, Int<16*64>, Int<4*64>>{}));

    /////////////////////////////////////////////////////////////////////
    // load Qt and dOt, use lds_b64 and perm shfl
    using ThreadLayoutAtomQ = Layout<Shape<_4,_4>,Stride<_1,_4>>;

    auto warp_layout_Q = tile_to_shape(ThreadLayoutAtomQ{},Shape<_16,_4>{});
    auto thread_layout_Q = tile_to_shape(warp_layout_Q,Shape<_16,_16>{});

    auto ldsQ_coord = thread_layout_Q.get_hier_coord(warp_id / 2 * 64 + lane_id);
    auto ldsQ_row_coord = cute::get<0>(ldsQ_coord);
    auto ldsQ_col_coord = cute::get<1>(ldsQ_coord);

    auto ldsQ_row = warp_layout_Q(ldsQ_row_coord);
    auto ldsQ_col = warp_layout_Q(ldsQ_col_coord);
    ldsQ_col ^= (ldsQ_row % 8) * 2;

    Element *Qtsmem_ptr_lds = reinterpret_cast<Element *>(sQ.data().get()) + ldsQ_row * 64 + ldsQ_col * 4;
    Tensor tdKsQt = make_tensor(make_smem_ptr(Qtsmem_ptr_lds), make_layout(Shape<_4, _1, _4>{},
                                                                          Stride<_1, _0, Int<16*64>>{}));
    Element *dOtsmem_ptr_lds = reinterpret_cast<Element *>(sdO.data().get()) + ldsQ_row * 64 + ldsQ_col * 4;
    Tensor tdVsdOt = make_tensor(make_smem_ptr(dOtsmem_ptr_lds), make_layout(Shape<_4, _1, _4>{},
                                                                          Stride<_1, _0, Int<16*64>>{}));

    cute::copy(gmem_tiled_copy_KV, tKrK, tKsK);
    cute::copy(gmem_tiled_copy_KV, tVrV, tVsV);


    // ((2, 4), 1, 1) => (2, 4, 1)
    Tensor tKrK_reshape = make_tensor(tKrK.data(), make_layout(make_shape(size<0, 0>(tKrK), size<0, 1>(tKrK), size<2>(tKrK))));
    Tensor tKtrKt = permute_b16<size<0>(tKrK_reshape), size<1>(tKrK_reshape)>(tKrK_reshape);


    auto skt_sts_row = threadIdx.x / 8 ;
    auto skt_sts_col = threadIdx.x % 8;
    auto skt_sts_ptr = sKt.data().get() + skt_sts_row * 64 + skt_sts_col * 8;
    *(reinterpret_cast<cute::uint128_t*>(skt_sts_ptr)) = *(reinterpret_cast<cute::uint128_t*>(tKtrKt.data()));

    sync_threads();

    Tensor tSrK_copy_view = smem_thr_copy_KV.retile_D(tSrK);
    cute::copy(smem_tiled_copy_KV, tSsK, tSrK_copy_view);
    Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
    CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
    cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);

    auto skt_lds_row = __lane_id() / 16;
    auto skt_lds_col = __lane_id() % 16;
    auto skt_lds_ptr = sKt.data().get() + warp_id / 4 * 64 + skt_lds_row * 256 + skt_lds_col * 4;
    auto tdQsKt  = make_tensor(make_smem_ptr(skt_lds_ptr),make_layout(tdQrKt.shape(), Stride<_1,_128,_1024>{}));

    cute::copy(tdQsKt,tdQrKt);

    auto tdOrdO = make_fragment_like(tdOsdO);
    flash::copy_b128<Is_even_MN, Is_even_K>(
        tdOgdO, tdOrdO, tQcQ, params.d, binfo.actual_seqlen_q - m_block * kBlockM
    );

    sync_threads();

    auto tQrQ = make_fragment_like(tQsQ);
    flash::copy_b128<Is_even_MN, Is_even_K>(
        tQgQ, tQrQ, tQcQ, params.d, binfo.actual_seqlen_q - m_block * kBlockM
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);
    // Convert to ((1, 4), MMA_N, MMA_N) then take only the row indices.
    Tensor taccScS_row = logical_divide(taccScS, Shape<_4>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
    // and scores would be zero. With LSE = 0, probs will be all 1's, and when we multiply
    // with V (which would be zero), we're fine. However, with ALiBi, we might modify these
    // scores, and probs can become NaN. Instead if we set LSE = inf for OOB rows, probs are always 0.

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    clear(acc_dv);
    clear(acc_dk);

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
    Tensor dP_sum = make_fragment_like(lse);
    Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
    // Softcapping - calculating dTanh and scaling dS later with it
    Tensor dtanh = make_tensor_like(scores);
    Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
    Tensor dS = make_tensor(acc_dp.data(), scores.layout());
    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());

    for (; m_block >= m_block_min; --m_block) {
        clear(acc_s);
        flash::barrier();
        cute::copy(tQrQ,tQsQ);
        cute::copy(tdOrdO,tdOsdO);
        sync_threads();

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi)));
        }

        flash::gemm</*A_in_regs=*/false, /*B_in_regs=*/Kernel_traits::Is_K_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
                    smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }

        if constexpr (Has_attn_mask) {
            Element *bias_ptr = reinterpret_cast<Element *>(params.attn_mask_ptr)
                                    + bidb % params.attn_mask_batch_shape * params.attn_mask_batch_stride
                                    + bidh % params.attn_mask_nheads_shape * params.attn_mask_nheads_stride;
            if(flash::use_attn_mask_merge_ldg(params)) {
                flash::apply_attn_mask</*mergeLdg=*/true, Is_even_MN>(
                    scores,
                    n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    m_block * kBlockM + get<0>(taccScS_row(0)), /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    AtomLayoutMS * 16,
                    AtomLayoutNS * 16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            } else {
                flash::apply_attn_mask(
                    scores,
                    n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16, /*col_idx_offset_*/
                    binfo.actual_seqlen_k,
                    m_block * kBlockM + get<0>(taccScS_row(0)), /*row_idx_offset_*/
                    binfo.actual_seqlen_q,
                    AtomLayoutMS * 16,
                    AtomLayoutNS * 16,
                    params.scale_softmax,
                    bias_ptr,
                    params.attn_mask_row_stride,
                    params.attn_mask_col_stride);
            }
        }

        if constexpr (Has_alibi) {
            alibi.apply_alibi(scores, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16,
                              m_block * kBlockM + get<0>(taccScS_row(0)), AtomLayoutMS * 16, AtomLayoutNS * 16);
        }

        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                flash::apply_mask(scores, binfo.actual_seqlen_k,
                                  n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16, AtomLayoutNS * 16);
            }
        } else if constexpr (Is_causal) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                flash::apply_mask_causal(scores, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16,
                                         binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                         binfo.actual_seqlen_q,
                                         // binfo.actual_seqlen_k, m_block * kBlockM + (tidx / 32) % AtomLayoutMS * 16 + (tidx % 32) / 4,
                                         AtomLayoutMS * 16, AtomLayoutNS * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                flash::apply_mask_local(scores, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right,
                                        AtomLayoutNS * 16);
            }
        }

        clear(acc_dp);

        flash::gemm</*A_in_regs=*/false, /*B_in_regs=*/Kernel_traits::Is_V_in_regs>(
            acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV
        );

        // Compute the exponential value.
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        if constexpr (Is_dropout) {
            int warp_id = tidx / 64;
            int block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
            int block_col_idx = n_block * (kBlockN / 64);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, AtomLayoutMS, AtomLayoutNS, kBlockN>(
                acc_s, block_row_idx, block_col_idx, n_block
            );
        }

        if(m_block > m_block_min) {
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccScS_row(mi));
                lse(mi) = gLSE(row);
            }
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
        }
        // Convert scores from fp32 to fp16/bf16
        CONVERT_TENSOR_TYPE(ElementAccum,Element,acc_s,rP)
        if constexpr (Is_dropout) {
            flash::relu_(rP);
        }
        // Reshape rP from (nrow=(2, MMA_N), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_N, MMA_N / 2)
        // if using m16n8k16 or ((2, 2, 1), MMA_N, MMA_N) if using m16n8k8.
        Tensor tPrP = make_tensor(rP.data(), acc_s.layout());
        // Shuffle and pemute, then sts64
        flash::sts_transpose(tPrP,tPtsPt);


        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult<Is_dropout>(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        // Convert dS from fp32 to fp16
        CONVERT_TENSOR_TYPE(ElementAccum, Element, dS_reshaped, tdSrdS)
        Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);                                        // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
        sync_threads();

        tdQgdQaccum.data() = tdQgdQaccum.data() + (-dq_accum_block_stride);
        clear(acc_dq);
        flash::gemm</*A_in_regs=*/false, /*B_in_regs=*/Kernel_traits::Is_K_in_regs>(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq,
                    smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt);
        #pragma unroll
        for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }

        sync_threads();
        // Shuffle and pemute, then sts64
        flash::sts_transpose(tdSrdS,tdStsdSt);
        sync_threads();

        if(m_block > m_block_min) {

            tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            // Advance gdO,gQ
            flash::copy_b128<true, Is_even_K>(tdOgdO, tdOrdO, tQcQ, params.d);
            flash::copy_b128<true, Is_even_K>(tQgQ, tQrQ, tQcQ, params.d);
        }

        // lds with customize method
        flash::lds_transpose(tdKsQt, tdKrQt);
        gemm_A_in_reg(acc_dk,tdKrdSt,tdKrQt,tdKsdSt,tiled_mma_dkv);

        // lds with customize method
        flash::lds_transpose(tdVsdOt, tdVrdOt);
        gemm_A_in_reg(acc_dv,tdVrPt,tdVrdOt,tdVsPt,tiled_mma_dkv);
    }

    // Epilogue

    if constexpr (Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dv from fp32 to fp16
    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_dk, rdK)
    CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_dv, rdV)

    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    if constexpr (!Is_last) { __syncthreads(); }

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
       + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
       + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    auto reorder_warp = [](int warp_id) {
        // 0,1,2,3,4,5,6,7 -> 0,1,4,5,2,3,6,7
        if (2 <= warp_id && warp_id <= 3) {
            warp_id += 2;
        }
        else if (4 <= warp_id && warp_id <= 5) {
            warp_id -= 2;
        }
        return warp_id;
    };
    int warp_id_new = reorder_warp(warp_id);
    // Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Element *dKsmem_ptr = reinterpret_cast<Element *>(sdK.data().get()) + (warp_id_new * 64 + lane_id) * 8;
    Tensor tdKsdK = make_tensor(make_smem_ptr(dKsmem_ptr), make_layout(Shape<Shape<_1, _8>, _1, _1>{},
                                                                          Stride<Stride<_0, _1>, _0, _0>{}));
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    // Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Element *dVsmem_ptr = reinterpret_cast<Element *>(sdV.data().get()) + (warp_id_new * 64 + lane_id) * 8;
    Tensor tdVsdV = make_tensor(make_smem_ptr(dVsmem_ptr), make_layout(Shape<Shape<_1, _8>, _1, _1>{},
                                                                          Stride<Stride<_0, _1>, _0, _0>{}));
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    sync_threads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy_without_clear<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::copy_without_clear<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
}

} // namespace flash
