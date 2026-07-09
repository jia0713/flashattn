#pragma once

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif
#include <ATen/cuda/CUDAGraphsUtils.cuh> // For at::cuda::philox::unpack

#include "flash_parameter.h"
#include "flash_splitkv.h"
#include "host_utils.h"
#include "../flash_dispatch/fwd_split_meta.h"

using namespace mcFlashAttn;

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    // if (max_splits < 64 || batch_nheads_mblocks / 64 > 10) {
    //     return 1;
    // }
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

SplitKVAccumTensors malloc_accum_by_numsplits(Flash_fwd_params &params) {

    auto num_heads = params.h;
    auto batch_size = params.b;
    auto max_seqlen_q = params.seqlen_q;
    auto head_size_rounded = params.d_rounded;
    auto p_dropout = 1.f - params.p_dropout;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    SplitKVAccumTensors accum;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (params.fwd_kernel_path == FwdKernelPathSplitKV && params.num_splits > 1) {
            accum.softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(torch::kFloat32));
            accum.out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(torch::kFloat32));
            params.softmax_lseaccum_ptr = accum.softmax_lse_accum.data_ptr();
            params.oaccum_ptr = accum.out_accum.data_ptr();
        }
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }
    return accum;
}

void compute_params_numsplits(mcFlashAttn::Flash_fwd_params &params, const int num_splits, bool force_split_kernel){
    auto num_heads = params.h;
    auto batch_size = params.b;
    auto max_seqlen_k = params.seqlen_k;
    auto max_seqlen_q = params.seqlen_q;
    auto p_dropout = 1.f - params.p_dropout;
    auto dprops = flash::mcGetCurrentDeviceProperties();

    params.num_splits = num_splits;
    params.fwd_kernel_path = FwdKernelPathNormal;
    params.fwd_meta_valid = false;
    params.split_meta_valid = false;

    if (!force_split_kernel && num_splits == 1) {
        auto fwd_meta = select_fwd_meta(params);
        check_fwd_meta_supported(fwd_meta);
        set_fwd_meta(params, fwd_meta);
        return;
    }

    auto split_meta = select_fwd_split_meta(params);
    check_fwd_split_meta_supported(split_meta);

    const int num_n_blocks = (max_seqlen_k + split_meta.block_n - 1) / split_meta.block_n;
    const int num_m_blocks = (max_seqlen_q + split_meta.block_m - 1) / split_meta.block_m;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            const int AP_nums = dprops.multiProcessorCount;
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks,  AP_nums * split_meta.block_num_per_ap,
                                                     num_n_blocks, 128);
        }
    } else if (num_splits < 1) {
        params.num_splits = 1;
    }

    if (force_split_kernel || params.num_splits > 1) {
        params.fwd_kernel_path = FwdKernelPathSplitKV;
        set_fwd_split_meta(params, split_meta);
    } else {
        auto fwd_meta = select_fwd_meta(params);
        check_fwd_meta_supported(fwd_meta);
        set_fwd_meta(params, fwd_meta);
        params.num_splits = 1;
    }

}

void update_params_numsplits(mcFlashAttn::Flash_fwd_params &params, const int block_nums_per_AP, const int block_n, const int block_m) {
    auto num_heads = params.h;
    auto batch_size = params.b;
    auto max_seqlen_k = params.seqlen_k;
    auto max_seqlen_q = params.seqlen_q;
    auto p_dropout = 1.f - params.p_dropout;
    auto dprops = flash::mcGetCurrentDeviceProperties();
    const int AP_nums = dprops.multiProcessorCount;

    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    const int num_m_blocks = (max_seqlen_q + block_m - 1) / block_m;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks,  AP_nums * block_nums_per_AP,
                                                    num_n_blocks, 128);
    }
}
