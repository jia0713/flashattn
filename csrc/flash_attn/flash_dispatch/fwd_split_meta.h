#pragma once

#include <stdexcept>
#include <string>

#include "../flash_api/flash_parameter.h"
#include "../utils/arch.h"

namespace mcFlashAttn {

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

inline int default_fwd_split_block_num_per_ap(const int headdim, const int block_m, const int block_n) {
    // Conservative first-pass occupancy model. This should be generated from tuned
    // kernel metadata or benchmark results once the generator path is updated.
    if (headdim == 64) {
        return block_m <= 16 && block_n <= 16 ? 16 : 4;
    }
    if (headdim == 128) {
        if (block_m <= 16) { return 8; }
        if (block_m <= 32) { return 4; }
    }
    return 2;
}

inline FwdSplitKernelMeta make_fwd_split_meta(
        const int arch,
        const int headdim,
        const int block_m,
        const int block_n,
        const int nwarps,
        const bool is_q_in_regs,
        const bool share_q_k_smem) {
    return {
        true,
        arch,
        headdim,
        block_m,
        block_n,
        nwarps,
        is_q_in_regs,
        share_q_k_smem,
        default_fwd_split_block_num_per_ap(headdim, block_m, block_n)
    };
}

inline int fwd_split_effective_headdim(const int headdim) {
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

inline FwdSplitKernelMeta select_fwd_split_meta(const Flash_fwd_params &params) {
    const int arch = params.arch;
    const int headdim = fwd_split_effective_headdim(params.d);
    const int seqlen_q = params.seqlen_q;

    if (arch == Arch::xcore1000) {
        switch (headdim) {
            case 32:  return make_fwd_split_meta(arch, headdim, 64, 64, 4, true, true);
            case 64:
                return seqlen_q <= 16
                    ? make_fwd_split_meta(arch, headdim, 16, 16, 1, true, true)
                    : make_fwd_split_meta(arch, headdim, 64, 64, 4, true, true);
            case 96:  return make_fwd_split_meta(arch, headdim, 64, 64, 4, true, true);
            case 128:
                if (seqlen_q <= 16) { return make_fwd_split_meta(arch, headdim, 16, 16, 1, true, true); }
                if (seqlen_q <= 32) { return make_fwd_split_meta(arch, headdim, 32, 32, 2, true, true); }
                return make_fwd_split_meta(arch, headdim, 64, 64, 4, true, true);
            case 160: return make_fwd_split_meta(arch, headdim, 64, 64, 4, true, true);
            case 192: return make_fwd_split_meta(arch, headdim, 64, 64, 4, true, true);
            case 256: return make_fwd_split_meta(arch, headdim, 64, 32, 4, true, true);
            case 512: return make_fwd_split_meta(arch, headdim, 32, 32, 2, true, true);
            default: break;
        }
    } else if (arch == Arch::xcore1500) {
        switch (headdim) {
            case 32:  return make_fwd_split_meta(arch, headdim, 64, 64, 4, true, true);
            case 64:  return make_fwd_split_meta(arch, headdim, 128, 64, 4, false, false);
            case 96:  return make_fwd_split_meta(arch, headdim, 64, 64, 4, true, true);
            case 128:
                return seqlen_q <= 32
                    ? make_fwd_split_meta(arch, headdim, 16, 32, 1, true, true)
                    : make_fwd_split_meta(arch, headdim, 128, 64, 4, true, true);
            case 160: return make_fwd_split_meta(arch, headdim, 64, 64, 4, true, true);
            case 192: return make_fwd_split_meta(arch, headdim, 128, 64, 4, false, false);
            case 256: return make_fwd_split_meta(arch, headdim, 128, 64, 4, false, false);
            default: break;
        }
    }
    return {};
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

inline FwdSplitKernelMeta get_fwd_split_meta_from_params(const Flash_fwd_params &params) {
    return {
        params.split_meta_valid,
        params.arch,
        fwd_split_effective_headdim(params.d),
        params.split_block_m,
        params.split_block_n,
        params.split_nwarps,
        params.split_is_q_in_regs,
        params.split_share_q_k_smem,
        params.split_block_num_per_ap
    };
}

inline void check_fwd_split_meta_supported(const FwdSplitKernelMeta &meta) {
    if (!meta.supported) {
        throw std::invalid_argument("Unsupported fwd_split kernel meta: arch=" + std::to_string(meta.arch)
                                    + ", headdim=" + std::to_string(meta.headdim));
    }
}

} // namespace mcFlashAttn

#define FWD_SPLIT_META_SWITCH(META, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, ...) \
    [&] {                                                                                         \
        const auto &meta__ = (META);                                                              \
        mcFlashAttn::check_fwd_split_meta_supported(meta__);                                      \
        if (meta__.block_m == 16 && meta__.block_n == 16 && meta__.nwarps == 1 &&                 \
            meta__.is_q_in_regs && meta__.share_q_k_smem) {                                      \
            constexpr static int kBlockM = 16;                                                    \
            constexpr static int kBlockN = 16;                                                    \
            constexpr static int kNWarps = 1;                                                     \
            constexpr static bool Is_Q_in_regs = true;                                           \
            constexpr static bool Share_Q_K_smem = true;                                         \
            return __VA_ARGS__();                                                                 \
        }                                                                                         \
        if (meta__.block_m == 16 && meta__.block_n == 32 && meta__.nwarps == 1 &&                 \
            meta__.is_q_in_regs && meta__.share_q_k_smem) {                                      \
            constexpr static int kBlockM = 16;                                                    \
            constexpr static int kBlockN = 32;                                                    \
            constexpr static int kNWarps = 1;                                                     \
            constexpr static bool Is_Q_in_regs = true;                                           \
            constexpr static bool Share_Q_K_smem = true;                                         \
            return __VA_ARGS__();                                                                 \
        }                                                                                         \
        if (meta__.block_m == 32 && meta__.block_n == 32 && meta__.nwarps == 2 &&                 \
            meta__.is_q_in_regs && meta__.share_q_k_smem) {                                      \
            constexpr static int kBlockM = 32;                                                    \
            constexpr static int kBlockN = 32;                                                    \
            constexpr static int kNWarps = 2;                                                     \
            constexpr static bool Is_Q_in_regs = true;                                           \
            constexpr static bool Share_Q_K_smem = true;                                         \
            return __VA_ARGS__();                                                                 \
        }                                                                                         \
        if (meta__.block_m == 64 && meta__.block_n == 32 && meta__.nwarps == 4 &&                 \
            meta__.is_q_in_regs && meta__.share_q_k_smem) {                                      \
            constexpr static int kBlockM = 64;                                                    \
            constexpr static int kBlockN = 32;                                                    \
            constexpr static int kNWarps = 4;                                                     \
            constexpr static bool Is_Q_in_regs = true;                                           \
            constexpr static bool Share_Q_K_smem = true;                                         \
            return __VA_ARGS__();                                                                 \
        }                                                                                         \
        if (meta__.block_m == 64 && meta__.block_n == 64 && meta__.nwarps == 4 &&                 \
            meta__.is_q_in_regs && meta__.share_q_k_smem) {                                      \
            constexpr static int kBlockM = 64;                                                    \
            constexpr static int kBlockN = 64;                                                    \
            constexpr static int kNWarps = 4;                                                     \
            constexpr static bool Is_Q_in_regs = true;                                           \
            constexpr static bool Share_Q_K_smem = true;                                         \
            return __VA_ARGS__();                                                                 \
        }                                                                                         \
        if (meta__.block_m == 128 && meta__.block_n == 64 && meta__.nwarps == 4 &&                \
            meta__.is_q_in_regs && meta__.share_q_k_smem) {                                      \
            constexpr static int kBlockM = 128;                                                   \
            constexpr static int kBlockN = 64;                                                    \
            constexpr static int kNWarps = 4;                                                     \
            constexpr static bool Is_Q_in_regs = true;                                           \
            constexpr static bool Share_Q_K_smem = true;                                         \
            return __VA_ARGS__();                                                                 \
        }                                                                                         \
        if (meta__.block_m == 128 && meta__.block_n == 64 && meta__.nwarps == 4 &&                \
            !meta__.is_q_in_regs && !meta__.share_q_k_smem) {                                    \
            constexpr static int kBlockM = 128;                                                   \
            constexpr static int kBlockN = 64;                                                    \
            constexpr static int kNWarps = 4;                                                     \
            constexpr static bool Is_Q_in_regs = false;                                          \
            constexpr static bool Share_Q_K_smem = false;                                        \
            return __VA_ARGS__();                                                                 \
        }                                                                                         \
        throw std::invalid_argument("Unsupported fwd_split kernel tuple: block_m="                 \
                                    + std::to_string(meta__.block_m)                              \
                                    + ", block_n=" + std::to_string(meta__.block_n)               \
                                    + ", nwarps=" + std::to_string(meta__.nwarps));               \
    }()
