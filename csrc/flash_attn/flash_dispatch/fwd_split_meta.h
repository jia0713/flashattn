#pragma once

#include <stdexcept>
#include <string>

#include "../flash_api/flash_parameter.h"
#include "../utils/arch.h"

namespace mcFlashAttn {

struct FwdKernelMeta {
    bool supported = false;
    int arch = 0;
    int headdim = 0;
    int headdim_v = 0;
    int block_m = 0;
    int block_n = 0;
    int nwarps = 0;
    bool is_q_in_regs = false;
    bool share_q_k_smem = false;
    int block_type = 5;
    int rowblock_parallel = 0;
    int block_num_per_ap = 1;
};

struct FwdSplitKernelMeta {
    bool supported = false;
    int arch = 0;
    int headdim = 0;
    int block_m = 0;
    int block_n = 0;
    int nwarps = 0;
    bool is_q_in_regs = false;
    bool share_q_k_smem = false;
    int block_num_per_ap = 1;
};

inline FwdKernelMeta make_fwd_meta(
        const int arch, const int headdim, const int headdim_v,
        const int block_m, const int block_n, const int nwarps,
        const bool is_q_in_regs, const bool share_q_k_smem,
        const int block_type, const int rowblock_parallel,
        const int block_num_per_ap) {
    return {true, arch, headdim, headdim_v, block_m, block_n, nwarps,
            is_q_in_regs, share_q_k_smem, block_type, rowblock_parallel,
            block_num_per_ap};
}

inline FwdSplitKernelMeta make_fwd_split_meta(
        const int arch, const int headdim,
        const int block_m, const int block_n, const int nwarps,
        const bool is_q_in_regs, const bool share_q_k_smem,
        const int block_num_per_ap) {
    return {true, arch, headdim, block_m, block_n, nwarps,
            is_q_in_regs, share_q_k_smem, block_num_per_ap};
}

inline int fwd_effective_headdim(const int headdim) {
    if (headdim <= 32) { return 32; }
    if (headdim <= 64) { return 64; }
    if (headdim <= 96) { return 96; }
    if (headdim <= 128) { return 128; }
    if (headdim <= 160) { return 160; }
    if (headdim <= 192) { return 192; }
    if (headdim <= 256) { return 256; }
    if (headdim <= 512) { return 512; }
    return 0;
}

inline int fwd_split_effective_headdim(const int headdim) {
    return fwd_effective_headdim(headdim);
}

// Generated selectors and template dispatch cases use the database below.
#include "generated/fwd_kernel_traits_registry.h"

inline void set_fwd_meta(Flash_fwd_params &params, const FwdKernelMeta &meta) {
    params.fwd_meta_valid = meta.supported;
    params.fwd_block_m = meta.block_m;
    params.fwd_block_n = meta.block_n;
    params.fwd_nwarps = meta.nwarps;
    params.fwd_is_q_in_regs = meta.is_q_in_regs;
    params.fwd_share_q_k_smem = meta.share_q_k_smem;
    params.fwd_block_type = meta.block_type;
    params.fwd_rowblock_parallel = meta.rowblock_parallel;
    params.fwd_headdim_v = meta.headdim_v;
    params.fwd_block_num_per_ap = meta.block_num_per_ap;
}

inline void set_fwd_split_meta(Flash_fwd_params &params, const FwdSplitKernelMeta &meta) {
    params.split_meta_valid = meta.supported;
    params.split_block_m = meta.block_m;
    params.split_block_n = meta.block_n;
    params.split_nwarps = meta.nwarps;
    params.split_is_q_in_regs = meta.is_q_in_regs;
    params.split_share_q_k_smem = meta.share_q_k_smem;
    params.split_block_num_per_ap = meta.block_num_per_ap;
}

inline FwdKernelMeta get_fwd_meta_from_params(const Flash_fwd_params &params) {
    return {params.fwd_meta_valid, params.arch, fwd_effective_headdim(params.d),
            params.fwd_headdim_v, params.fwd_block_m, params.fwd_block_n,
            params.fwd_nwarps, params.fwd_is_q_in_regs,
            params.fwd_share_q_k_smem, params.fwd_block_type,
            params.fwd_rowblock_parallel, params.fwd_block_num_per_ap};
}

inline FwdSplitKernelMeta get_fwd_split_meta_from_params(const Flash_fwd_params &params) {
    return {params.split_meta_valid, params.arch, fwd_split_effective_headdim(params.d),
            params.split_block_m, params.split_block_n, params.split_nwarps,
            params.split_is_q_in_regs, params.split_share_q_k_smem,
            params.split_block_num_per_ap};
}

inline void check_fwd_meta_supported(const FwdKernelMeta &meta) {
    if (!meta.supported) {
        throw std::invalid_argument("Unsupported fwd kernel meta: arch=" + std::to_string(meta.arch)
                                    + ", headdim=" + std::to_string(meta.headdim)
                                    + ", headdim_v=" + std::to_string(meta.headdim_v));
    }
}

inline void check_fwd_split_meta_supported(const FwdSplitKernelMeta &meta) {
    if (!meta.supported) {
        throw std::invalid_argument("Unsupported fwd_split kernel meta: arch=" + std::to_string(meta.arch)
                                    + ", headdim=" + std::to_string(meta.headdim));
    }
}

} // namespace mcFlashAttn
