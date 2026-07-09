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
    inline void run_mha_fwd_dispatch_meta(Flash_fwd_params &params, Flash_launch_params &launch_params, const cudaStream_t stream) {
        if (!params.fwd_meta_valid) {
            set_fwd_meta(params, select_fwd_meta(params));
        }
        auto meta = get_fwd_meta_from_params(params);
        check_fwd_meta_supported(meta);
        CHECK_MSG(meta.arch == arch && meta.headdim == Headdim, "fwd kernel meta does not match dispatch arch/head dimension");
        launch_params.block_type = meta.block_type;
        launch_params.rowblock_parallel = meta.rowblock_parallel;
        if constexpr (Headdim == 192) {
            if (meta.headdim_v == 128) {
                FWD_META_SWITCH(meta, arch, Headdim, 128, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, [&] {
                    if constexpr (arch == Arch::xcore1000) {
                        Xcore1000::run_flash_fwd_template<Headdim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type, 128>(params, launch_params, stream);
                    } else if constexpr (arch == Arch::xcore1500) {
                        Xcore1500::run_flash_fwd_template<Headdim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type, 128>(params, launch_params, stream);
                    }
                });
            } else {
                FWD_META_SWITCH(meta, arch, Headdim, Headdim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, [&] {
                    if constexpr (arch == Arch::xcore1000) {
                        Xcore1000::run_flash_fwd_template<Headdim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type>(params, launch_params, stream);
                    } else if constexpr (arch == Arch::xcore1500) {
                        Xcore1500::run_flash_fwd_template<Headdim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type>(params, launch_params, stream);
                    }
                });
            }
        } else {
            FWD_META_SWITCH(meta, arch, Headdim, Headdim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, [&] {
                if constexpr (arch == Arch::xcore1000) {
                    Xcore1000::run_flash_fwd_template<Headdim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type>(params, launch_params, stream);
                } else if constexpr (arch == Arch::xcore1500) {
                    Xcore1500::run_flash_fwd_template<Headdim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type>(params, launch_params, stream);
                }
            });
        }
    }

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
inline void run_mha_fwd_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    if constexpr (Headdim == 0) {
        std::cerr << "Error: HDIM of fwd is set to 0. Ensure that this configuration is only used in a development environment." << std::endl;
        return;
    }
    Flash_launch_params launch_params;
    FP16_SWITCH(!params.is_bf16, [&] {
        mcFlashAttn::run_mha_fwd_dispatch_meta<Headdim, arch, elem_type>(params, launch_params, stream);
    });
}
