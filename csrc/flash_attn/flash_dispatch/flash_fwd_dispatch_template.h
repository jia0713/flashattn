/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <iostream>
#include "flash_parameter.h"
#include "../utils/static_switch.h"
#include "fwd_split_meta.h"
#define HDIM_ALL

using namespace mcFlashAttn;

namespace Xcore1000 {

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    int kHeadDimV = kHeadDim
>
void run_flash_fwd_template(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    int kHeadDimV = kHeadDim
>
void run_flash_splitkv_fwd_template(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);

}

namespace Xcore1500 {

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    int kHeadDimV = kHeadDim
>
void run_flash_fwd_template(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    int kHeadDimV = kHeadDim
>
void run_flash_splitkv_fwd_template(Flash_fwd_params &params, mcFlashAttn::Flash_launch_params& launch_params, cudaStream_t stream);

}

namespace mcFlashAttn {

    template<int Headdim, Arch arch, typename elem_type>
    inline void run_mha_fwd_splitkv_dispatch_meta(Flash_fwd_params &params, Flash_launch_params &launch_params, const cudaStream_t stream) {
        if (!params.split_meta_valid) {
            set_fwd_split_meta(params, select_fwd_split_meta(params));
        }
        auto meta = get_fwd_split_meta_from_params(params);
        check_fwd_split_meta_supported(meta);
        CHECK_MSG(meta.arch == arch && meta.headdim == Headdim, "fwd_split kernel meta does not match dispatch arch/head dimension");
        launch_params.block_type = 2;
        FWD_SPLIT_META_SWITCH(meta, arch, Headdim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, [&] {
            if constexpr (arch == Arch::xcore1000) {
                Xcore1000::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type>(params, launch_params, stream);
            } else if constexpr (arch == Arch::xcore1500) {
                Xcore1500::run_flash_splitkv_fwd_template<Headdim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type>(params, launch_params, stream);
            }
        });
    }

    template<int Headdim, Arch arch>
    inline void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, const cudaStream_t stream) {
        if constexpr (Headdim == 0) {
            std::cerr << "Error: HDIM of fwd is set to 0. Ensure that this configuration is only used in a development environment." << std::endl;
            return;
        }
        Flash_launch_params launch_params;
        FP16_SWITCH(!params.is_bf16, [&] {
            run_mha_fwd_splitkv_dispatch_meta<Headdim, arch, elem_type>(params, launch_params, stream);
        });
    }
} // namespace mcFlashAttn end



template<int Headdim, Arch arch>
inline void run_mha_fwd_dispatch(Flash_fwd_params &params, cudaStream_t stream);

template<>
inline void run_mha_fwd_dispatch<0, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream){
    std::cerr << "Error: HDIM of fwd is set to 0. Ensure that this configuration is only used in a development environment." << std::endl;
    return;
}

template<>
inline void run_mha_fwd_dispatch<0, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream){
    std::cerr << "Error: HDIM of fwd is set to 0. Ensure that this configuration is only used in a development environment." << std::endl;
    return;
}

#if CHECK_HDIM(32)
template<>
inline void run_mha_fwd_dispatch<32, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 32;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_fwd_template<Headdim, 128, 128, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_fwd_dispatch<32, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 32;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    if (params.p_dropout < 1.0f) {
        FP16_SWITCH(!params.is_bf16, [&] {
            Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, true, true, elem_type>(params, launch_params, stream);
        });
    } else {
        FP16_SWITCH(!params.is_bf16, [&] {
            Xcore1500::run_flash_fwd_template<Headdim, 128, 128, 4, true, true, elem_type>(params, launch_params, stream);
        });
    }

}

#endif

#if CHECK_HDIM(64)
template<>
inline void run_mha_fwd_dispatch<64, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 64;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_fwd_template<Headdim, 64, 64, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_fwd_dispatch<64, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 64;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, false, false, elem_type>(params, launch_params, stream);
    });
}

#endif

#if CHECK_HDIM(96)
template<>
inline void run_mha_fwd_dispatch<96, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 96;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_fwd_template<Headdim, 128, 64, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_fwd_dispatch<96, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 96;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

#endif

#if CHECK_HDIM(128)
template<>
inline void run_mha_fwd_dispatch<128, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_fwd_template<Headdim, 64, 64, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_fwd_dispatch<128, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, true, true, elem_type>(params, launch_params, stream);
    });
}
#endif

#if CHECK_HDIM(160)
template<>
inline void run_mha_fwd_dispatch<160, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 160;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_fwd_template<Headdim, 64, 32, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_fwd_dispatch<160, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream){
    constexpr static int Headdim = 160;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, true, true, elem_type>(params, launch_params, stream);
    });
}

#endif

#if CHECK_HDIM(192)
template<>
inline void run_mha_fwd_dispatch<192, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        if(params.d_value_rounded == 128) { // only for hdim_q = 192 and hdim_v = 128
            // available traits: [64,64,4], [64,32,4], [128,64,4], [128,64,8]
            Xcore1000::run_flash_fwd_template<Headdim, 128, 64, 8, true, true, elem_type, 128>(params, launch_params, stream);
        } else {
            Xcore1000::run_flash_fwd_template<Headdim, 64, 64, 4, true, true, elem_type>(params, launch_params, stream);
        }
    });
}

template<>
inline void run_mha_fwd_dispatch<192, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        if(params.d_value_rounded == 128) { // only for hdim_q = 192 and hdim_v = 128
            Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 8, true, true, elem_type, 128>(params, launch_params, stream);
        } else {
            Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, false, false, elem_type>(params, launch_params, stream);
        }
    });
}

#endif

#if CHECK_HDIM(256)
template<>
inline void run_mha_fwd_dispatch<256, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        bool is_dropout = params.p_dropout < 1.f;
        if (!is_dropout) {
            Xcore1000::run_flash_fwd_template<Headdim, 64, 32, 4, true, true, elem_type>(params, launch_params, stream);
        }
        else {
            Xcore1000::run_flash_fwd_template<Headdim, 64, 64, 4, true, true, elem_type>(params, launch_params, stream);
        }
    });
}

template<>
inline void run_mha_fwd_dispatch<256, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1500::run_flash_fwd_template<Headdim, 128, 64, 4, false, false, elem_type>(params, launch_params, stream);
    });
}

#endif

#if CHECK_HDIM(512)
template<>
inline void run_mha_fwd_dispatch<512, Arch::xcore1000>(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 512;
    Flash_launch_params launch_params;
    launch_params.rowblock_parallel = 0;
    launch_params.block_type = 5;
    FP16_SWITCH(!params.is_bf16, [&] {
        Xcore1000::run_flash_fwd_template<Headdim, 64, 32, 8, true, true, elem_type>(params, launch_params, stream);
    });
}

template<>
inline void run_mha_fwd_dispatch<512, Arch::xcore1500>(Flash_fwd_params &params, cudaStream_t stream) {
    std::cerr << "Xcore1500 currently does not support headdim 512." << std::endl;
    return;
}

#endif
