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

using namespace mcFlashAttn;

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
int num_splits_heuristic(int64_t batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    // 改动3: 保证每个 split 主循环至少 8 次, num_n_blocks < 8 时根本不 split
    if (num_n_blocks < 8) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks / 8});
    // if (max_splits < 64 || batch_nheads_mblocks / 64 > 10) {
    //     return 1;
    // }
    // 改动1: 预计算 no-split 效率作为 max_efficiency 初始值, 避免选到比不 split 更差的方案
    // 改动2: xcore1000 上 max_splits <= 13 (由调用点传入)
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    float n_waves_nosplit = float(batch_nheads_mblocks) / num_SMs;
    float max_efficiency = n_waves_nosplit / ceil(n_waves_nosplit);
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    efficiency.push_back(max_efficiency); // index 0 = num_splits=1 基线
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    for (int num_splits = 2; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks) * num_splits / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 2; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85f * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

// Tile size should match the advance dispatch and default dispatch
std::pair<int, int> get_tile_size(int head_size_rounded, int seqlen_k, int seqlen_q) {
    int block_m = 64, block_n = 64;
    if (head_size_rounded == 256 || head_size_rounded == 512) {
        block_m = 64;
        block_n = 32;
    } else if (head_size_rounded == 128) {
        if (seqlen_q <= 16) {
            block_m = 16;
            block_n = 16;
        } else if (seqlen_q <= 32) {
            block_m = 32;
            block_n = 32;
        } else {
            block_m = 64;
            block_n = 64;
        }
    } else if (head_size_rounded == 64) {
        if (seqlen_q <= 16) {
            block_m = 16;
            block_n = 16;
        } else {
            block_m = 64;
            block_n = 64;
        }
    } else {
        block_m = 64;
        block_n = 64;
    }
    return std::make_pair(block_m, block_n);
}

void compute_params_numsplits(mcFlashAttn::Flash_fwd_params &params, const int num_splits){
    // This needs to match with run_mha_fwd_splitkv_dispatch
    auto num_heads = params.h;
    auto head_size = params.d;
    auto batch_size = params.b;
    auto max_seqlen_k = params.seqlen_k;
    auto max_seqlen_q = params.seqlen_q;
    auto head_size_rounded = params.d_rounded;
    auto p_dropout = 1.f - params.p_dropout;
    auto dprops = flash::mcGetCurrentDeviceProperties();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.:kFloat
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const auto [block_m, block_n] = get_tile_size(head_size_rounded, max_seqlen_k, max_seqlen_q);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    const int num_m_blocks = (max_seqlen_q + block_m - 1) / block_m;
    params.num_splits = num_splits;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            const int AP_nums = dprops.multiProcessorCount;
            int block_nums_per_AP = 2;
            // Note: adjust block_nums_per_AP decided by smem size usage to get better perfermance
            if (head_size_rounded == 128) {
                if (params.seqlen_q <= 16) {
                    block_nums_per_AP = 8;
                } else if (params.seqlen_q <= 32) {
                    block_nums_per_AP = 4;
                }
            } else if (head_size_rounded == 64) {
                if (params.seqlen_q <= 16) {
                    block_nums_per_AP = 16;
                } else {
                    block_nums_per_AP = 4;
                }
            }
            // 改动2: xcore1000 (dprops.major == 10) max_splits 限制为 13, 其它架构保持 128
            const int max_splits_kv = (dprops.major == 10) ? 13 : 128;
            params.num_splits = num_splits_heuristic(int64_t(batch_size) * num_heads * num_m_blocks,  AP_nums * block_nums_per_AP,
                                                     num_n_blocks, max_splits_kv);
        }
    }

}
