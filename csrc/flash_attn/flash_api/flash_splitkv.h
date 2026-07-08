#pragma once

#include <ATen/ATen.h>

#include "flash_parameter.h"

struct SplitKVAccumTensors {
    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;
};

SplitKVAccumTensors malloc_accum_by_numsplits(mcFlashAttn::Flash_fwd_params &params);
void compute_params_numsplits(mcFlashAttn::Flash_fwd_params &params, const int num_splits, bool force_split_kernel = false);
void update_params_numsplits(mcFlashAttn::Flash_fwd_params &params, const int block_nums_per_AP, const int block_n, const int block_m);
