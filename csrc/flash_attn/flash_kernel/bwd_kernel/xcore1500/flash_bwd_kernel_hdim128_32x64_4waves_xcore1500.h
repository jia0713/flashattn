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

#include <type_traits>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "bwd_hdim64_utils.h"
#include "softmax.h"
#include "philox.cuh"

#include "alibi.h"
#include "attn_mask.h"


namespace flash {

namespace xcore1500 {

using namespace cute;

// Double buffer
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Has_attn_mask, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_first, bool Is_last, bool Seq_parallel=false, typename Params>
__forceinline__ __device__ void compute_dq_dk_dv_1colblock_hdim128_32x64_xcore1500(const Params &params, const int bidb, const int bidh, const int n_block) {

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
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr int AtomLayoutNS = kNWarps / AtomLayoutMS;
    constexpr int MMA_N_SdP = kBlockN / AtomLayoutNS / 16;
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


    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV_swz242{});
    Tensor sK_NoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKV_NoSwizzle{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed_swz242{});
    Tensor sKt_NoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});

    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV_swz333{});
    Tensor sV_NoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutKV_NoSwizzle{});

    //reuse
    Tensor sQ = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutQdO_swz242{});
    Tensor sQ_NoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdO_NoSwizzle{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed_swz242{});
    Tensor sQt_NoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});

    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1)*size(sQ), typename Kernel_traits::SmemLayoutQdO_swz242{});
    Tensor sdO_NoSwizzle = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdO_NoSwizzle{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed_swz242{});
    Tensor sdOt_NoSwizzle = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});

    Tensor sP = make_tensor(sdO.data() + (Double_buffer ? 2 : 1)*size(sdO), typename Kernel_traits::SmemLayoutPdS_swz242{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed_swz242{});
    Tensor sPt_NoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});

    Tensor sdS = make_tensor(sP.data() + size(sP), typename Kernel_traits::SmemLayoutPdS_swz242{});
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed_swz242{});
    Tensor sdSt_NoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});


    typename Kernel_traits::GmemTiledCopyBsm1x8 gmem_tiled_copy_ldgbsm;
    auto gmem_thr_copy_ldgbsm = gmem_tiled_copy_ldgbsm.get_thread_slice(tidx);

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    const int swz242_offset = cute::get_swizzle_offset<8,2,4,2>(tidx);
    const int swz333_offset = cute::get_swizzle_offset<8,3,3,3>(tidx);
    const int swz242_diff_global = (tidx & 63) >= 32 ? ((tidx & 1) == 0 ? 8 : -8): 0;

    Tensor tQgQ = gmem_thr_copy_ldgbsm.partition_S(gQ);
    tQgQ = make_tensor(tQgQ.data() + swz242_offset + swz242_diff_global, layout(tQgQ));
    Tensor tQsQ = gmem_thr_copy_ldgbsm.partition_D(sQ_NoSwizzle);

    Tensor tdOgdO = gmem_thr_copy_ldgbsm.partition_S(gdO);
    tdOgdO = make_tensor(tdOgdO.data() + swz242_offset + swz242_diff_global, layout(tdOgdO));
    Tensor tdOsdO = gmem_thr_copy_ldgbsm.partition_D(sdO_NoSwizzle);

    Tensor tdOgO = gmem_thr_copy_ldgbsm.partition_S(gO);

    Tensor tKgK = gmem_thr_copy_ldgbsm.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    tKgK = make_tensor(tKgK.data() + swz242_offset, layout(tKgK));
    Tensor tKsK = gmem_thr_copy_ldgbsm.partition_D(sK_NoSwizzle);

    Tensor tVgV = gmem_thr_copy_ldgbsm.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    tVgV = make_tensor(tVgV.data() + swz333_offset, layout(tVgV));
    Tensor tVsV = gmem_thr_copy_ldgbsm.partition_D(sV_NoSwizzle);

    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);


    typename Kernel_traits::TiledMmaSdP_b128 tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ_NoSwizzle);         // (MMA,MMA_N,MMA_K)
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK_NoSwizzle);         // (MMA,MMA_N,MMA_K)
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO_NoSwizzle);      // (MMA,MMA_N,MMA_K)
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV_NoSwizzle);        // (MMA,MMA_N,MMA_K)

    Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
    Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
    // Reshape acc_s from (MMA=4, MMA_N, MMA_N) to (col=(2, MMA_N), row=(2, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
    Tensor dtanh = make_tensor_like(scores);
    // Reshape acc_dp from (MMA=4, MMA_N, MMA_N) to (col=(2, MMA_N), row=(2, MMA_N))
    Tensor dS = make_tensor(acc_dp.data(), scores.layout());
    Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdSt_NoSwizzle); // (MMA, MMA_N, MMA_N)
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQt_NoSwizzle);   // (MMA, MMA_K, MMA_N)
    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPt_NoSwizzle);   // (MMA, MMA_N, MMA_N)
    Tensor tdVrdOt = thr_mma_dkv.partition_fragment_B(sdOt_NoSwizzle); // (MMA, MMA_K, MMA_N)

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);                      // (MMA, MMA_N, MMA_N)
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKt_NoSwizzle);    // (MMA, MMA_K, MMA_N)

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K


    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::UniversalCopyAtomB128{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    const int swz242_diff_lds_b128 = (tidx & 7) >= 4 ? ((tidx & 31) < 16 ? 8 : -8) : 0;
    tSsQ = make_tensor(tSsQ.data() + swz242_diff_lds_b128, layout(tSsQ));
    tdPsdO = make_tensor(tdPsdO.data() + swz242_diff_lds_b128, layout(tdPsdO));

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB128{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    // Partition sP and sdS to match the accumulator partitioning
    auto smem_tiled_copy_PdS = make_tiled_copy_C(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::LDSB64Trans4x16Atom{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::LDSB64Trans4x16Atom{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);

    const int swz242_diff_lds_trans_b64 = (tidx & 31) >= 16 ? ((tidx & 3) < 2 ? 8 : -8) : 0;
    tdKsQt = make_tensor(tdKsQt.data() + swz242_diff_lds_trans_b64, layout(tdKsQt));
    tdVsdOt = make_tensor(tdVsdOt.data() + swz242_diff_lds_trans_b64, layout(tdVsdOt));

    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_dq);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
    Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::LDSB64Trans4x16Atom{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    //////////////////////////////////////////////////////////////////////

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));     // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tQcQ_noSwizzle = gmem_thr_copy_ldgbsm.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKcK_noSwizzle = gmem_thr_copy_ldgbsm.partition_S(cKV);    // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tQcQ = make_tensor(tQcQ_noSwizzle.data() + make_coord(0, swz242_offset + swz242_diff_global), layout(tQcQ_noSwizzle));
    Tensor tKcK = make_tensor(tKcK_noSwizzle.data() + make_coord(0, swz242_offset), layout(tKcK_noSwizzle));
    Tensor tVcV = make_tensor(tKcK_noSwizzle.data() + make_coord(0, swz333_offset), layout(tKcK_noSwizzle));

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Prologue

    // We'll advance gdQ and gdQaccum before the 1st read/write.
    tdQgdQ.data() = tdQgdQ.data() + kBlockM * params.dq_row_stride;
    tdQgdQaccum.data() = tdQgdQaccum.data() + kBlockM * params.h * params.d_rounded;

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

    if (params.deterministic) { flash::sync_threads<4>(); }


    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);
    // Convert to ((1, 4), MMA_N, MMA_N) then take only the row indices.
    Tensor taccScS_row = logical_divide(taccScS, Shape<_4>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    flash::copy_b128_bsm_async<Is_even_MN, Is_even_K>(
        tKgK, tKsK, tKcK, params.d, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::barrier_gvm();
    cute::copy(smem_tiled_copy_KV, tSsK, tSrK);
    cute::copy(smem_tiled_copy_Kt, tdQsKt, tdQrKt);   // lds trans

    flash::copy_b128_bsm_async<Is_even_MN, Is_even_K>(
        tVgV, tVsV, tVcV, params.d, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::barrier_gvm();
    cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV);

    // smem Q reuse smem K
    flash::sync_threads<4>();
    flash::copy_b128_bsm_async<Is_even_MN, Is_even_K>(
        tQgQ, tQsQ, tQcQ, params.d, binfo.actual_seqlen_q - m_block * kBlockM
    );

    flash::copy_b128_bsm_async<Is_even_MN, Is_even_K>(
        tdOgdO, tdOsdO, tQcQ, params.d, binfo.actual_seqlen_q - m_block * kBlockM
    );

    flash::Dropout dropout(params.rng_state_seed, params.rng_state_offset, params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);


    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ?
            0.0f :
            reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    clear(acc_dv);
    clear(acc_dk);


    constexpr int atomic_add_cnt = size(tdQgdQaccum);
    uint32_t ping_pong = 0;
    flash::cp_async_wait<0>();

    auto process_m_block = [&](auto Need_mask) {
        clear(acc_s);
        const int sQdO_offset = ping_pong == 0 ? size(sQ) : -size(sQ);
        // arrive all ldg in pre loop and not arrive atomic add here
        flash::barrier_gvm<atomic_add_cnt, 4>();
        if (m_block > m_block_min) {
            // Double buffer
            tQsQ.data() = tQsQ.data() + sQdO_offset;
            // Prefetch global -> smem : Q
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            flash::copy_b128_bsm_async</*Is_even_MN=*/true, Is_even_K>(tQgQ, tQsQ, tQcQ, params.d);

            // Double buffer
            tdOsdO.data() = tdOsdO.data() + sQdO_offset;
            // Prefetch global -> smem : dO
            tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
            flash::copy_b128_bsm_async</*Is_even_MN=*/true, Is_even_K>(tdOgdO, tdOsdO, tQcQ, params.d);
        }

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi)));
        }

        // lds, reg
        flash::gemm_prefetch_lds</*A_in_regs=*/false, /*B_in_regs=*/true, 2>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
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

        if constexpr (decltype(Need_mask)::value) {
            if constexpr (!Is_causal && !Is_local) {
                flash::apply_mask(scores, binfo.actual_seqlen_k,
                                  n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16, AtomLayoutNS * 16);
            } else if constexpr (Is_causal) {
                flash::apply_mask_causal(scores, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16,
                                         binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                         binfo.actual_seqlen_q,
                                         // binfo.actual_seqlen_k, m_block * kBlockM + (tidx / 32) % AtomLayoutMS * 16 + (tidx % 32) / 4,
                                         AtomLayoutMS * 16,
                                         AtomLayoutNS * 16);
            } else if constexpr (Is_local) {
                flash::apply_mask_local(scores, n_block * kBlockN + (tidx / 64 / AtomLayoutMS) * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right,
                                        AtomLayoutNS * 16);
            }
        }

        // Compute the exponential value.
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);

        if constexpr (Is_dropout) {
            int warp_id = tidx / 64;
            int block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            // Need col to be multiples of 64, since we're doing dropout with block of 16 x 64
            int block_col_idx = n_block * (kBlockN / 64);
            dropout.template mc_apply_dropout</*encode_dropout_in_sign_bit=*/true, AtomLayoutNS>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS, kBlockN , n_block
            );
        }

        // Convert scores from fp32 to fp16/bf16
        CONVERT_TENSOR_TYPE(ElementAccum,Element,acc_s,rP)
        if constexpr (Is_dropout) {
            flash::relu_(rP);
        }
        cute::copy(smem_tiled_copy_PdS, rP, tPsP);

        clear(acc_dp);
        //lds, reg
        flash::gemm</*A_in_regs=*/false, /*B_in_regs=*/true>(
            acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV
        );

        cute::copy(smem_tiled_copy_QdOt, tdVsdOt, tdVrdOt);

        typedef __NATIVE_VECTOR__(2, float) Float2;
        auto get_scale = [](Float2 p, Float2 dp, float d) {
            Float2 scale_vec = {1.0f, 1.0f};
            Float2 beta_vec = {-d, -d};
            if constexpr (!Is_dropout) {
                Float2 res_vec = __builtin_mxc_pk_fma_f32(dp, scale_vec, beta_vec);
                return res_vec;
            } else {
                Float2 res_vec{d, d};
                if (p[0] >= 0) res_vec[0] = dp[0] - d;
                if (p[1] >= 0) res_vec[1] = dp[1] - d;
                return res_vec;
            }
        };
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            Float2 beta_vec = {0.0f, 0.0f};
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ni += 2) {
                Float2 score_vec = {scores(mi, ni), scores(mi, ni + 1)};
                Float2 dS_vec = {dS(mi, ni), dS(mi, ni + 1)};
                Float2 scale_vec = get_scale(score_vec, dS_vec, dP_sum(mi));
                score_vec = __builtin_mxc_pk_fma_f32(score_vec, scale_vec, beta_vec);
                if constexpr (Is_softcap) {
                    scale_vec = {dtanh(mi, ni), dtanh(mi, ni + 1)};
                    score_vec = __builtin_mxc_pk_fma_f32(score_vec, scale_vec, beta_vec);
                }
                dS(mi, ni) = score_vec[0];
                dS(mi, ni + 1) = score_vec[1];
            }
        }

        // Convert dS from fp32 to fp16/bf16
        CONVERT_TENSOR_TYPE(ElementAccum, Element, dS_reshaped, rdS)
        // only if warp Nx1, we can directly cute::copy(rdS, tdQrdS)
        cute::copy(smem_tiled_copy_PdS, rdS, tdSsdS);

        if (m_block > m_block_min) {
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccScS_row(mi));
                lse(mi) = gLSE(row);
            }
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
        }

        flash::sync_threads<4>();

        // lds trans, reg
        flash::gemm_prefetch_lds</*A_in_regs=*/false, /*B_in_regs=*/false, 2>(
            acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv,
            smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);

        // Double buffer
        tSsQ.data() = tSsQ.data() + sQdO_offset;
        tdKsQt.data() = tdKsQt.data() + sQdO_offset;

        // lds trans, reg
        flash::gemm_prefetch_lds</*A_in_regs=*/false, /*B_in_regs=*/true, 2>(
            acc_dv, tdVrPt, tdVrdOt, tdVsPt, tdVsdOt, tiled_mma_dkv,
            smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);

        // Double buffer
        tdPsdO.data() = tdPsdO.data() + sQdO_offset;
        tdVsdOt.data() = tdVsdOt.data() + sQdO_offset;
        ping_pong ^= 1;

        clear(acc_dq);
        // lds, reg
        flash::gemm_prefetch_lds</*A_in_regs=*/false, /*B_in_regs=*/true, 3>(
            acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq,
            smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt);

        tdQgdQaccum.data() = tdQgdQaccum.data() + (-int(kBlockM * params.h * params.d_rounded));
        #pragma unroll
        for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }

    };

    const bool masking_col_block = !Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k;
    if (masking_col_block) {
        for (; m_block >= m_block_min; --m_block) {
            process_m_block(std::true_type{});
        }
    } else if constexpr (!Is_causal && !Is_local) {
        for (; m_block >= m_block_min; --m_block) {
            process_m_block(std::false_type{});
        }
    } else if constexpr (Is_causal) {
        const int causal_mask_boundary = (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k;
        const int masking_m_block_max = causal_mask_boundary <= 0 ? -1 : cute::ceil_div(causal_mask_boundary, kBlockM) - 1;
        for (; m_block > masking_m_block_max && m_block >= m_block_min; --m_block) {
            process_m_block(std::false_type{});
        }
        for (; m_block >= m_block_min; --m_block) {
            process_m_block(std::true_type{});
        }
    } else if constexpr (Is_local) {
        const int local_mask_low_boundary = (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right;
        const int local_mask_high_boundary = n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left;
        const int no_mask_m_block_min = local_mask_low_boundary <= 0 ? 0 : cute::ceil_div(local_mask_low_boundary, kBlockM);
        const int no_mask_m_block_max = local_mask_high_boundary <= 0 ? -1 : cute::ceil_div(local_mask_high_boundary, kBlockM) - 2;

        for (; m_block > no_mask_m_block_max && m_block >= m_block_min; --m_block) {
            process_m_block(std::true_type{});
        }
        for (; m_block >= no_mask_m_block_min && m_block >= m_block_min; --m_block) {
            process_m_block(std::false_type{});
        }
        for (; m_block >= m_block_min; --m_block) {
            process_m_block(std::true_type{});
        }
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

    Tensor sdK = make_tensor(sKt.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
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
    if constexpr (!Is_last) { flash::sync_threads<4>(); }

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
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    flash::sync_threads<4>();
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

} // namespace xcore1500

} // namespace flash
