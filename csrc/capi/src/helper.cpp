#include "flash_attn.h"


// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int64_t batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    //if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    //Technically kBlockM = 64 only for the splitKV kernels
    if (max_splits < 64 || batch_nheads_mblocks / 64 > 10) {
        return 1;
    }
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
            float n_waves = float(batch_nheads_mblocks) * num_splits / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 2; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.96 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

int compute_num_splits(int batch_size,int num_heads,int head_size,int max_seqlen_k,int max_seqlen_q){

    // This needs to match with run_mha_fwd_splitkv_dispatch
    //const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int block_n = 32;
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 32 - 1) / 32;

    int deviceId{};
    mcGetDevice(&deviceId);
    mcDeviceProp_t dprops;
    mcGetDeviceProperties(&dprops, deviceId);

    return num_splits_heuristic(int64_t(batch_size) * num_heads * num_m_blocks, dprops.multiProcessorCount, num_n_blocks, 128);
}

int check_seqlenq_ngroups_swapped(int seqlen_q,int num_heads,int num_heads_k,int window_size_left,
                                int window_size_right, int head_size_og ,Tensor_t alibi_slopes_,float p_droput){

    bool seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && alibi_slopes_ == nullptr;

    return seqlenq_ngroups_swapped;
}

void release_extend_param(mcflashattnExtendParameter_t extend_param){
    if(extend_param){
        if(extend_param->leftpad_k_){
            release_tensor(extend_param->leftpad_k_);
        }
        if(extend_param->block_table_){
            release_tensor(extend_param->block_table_);
        }
        free(extend_param);
    }
}

int head_size_pad(int head_size_og){
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    if (head_size_og % 8 != 0) return round_multiple(head_size_og, 8);
    return head_size_og;
}


#ifdef __cplusplus
}
#endif /* __cplusplus */
