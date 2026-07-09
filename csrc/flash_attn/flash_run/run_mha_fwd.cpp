#include <mctlass/numeric_types.h>
#include "run_mha.h"
#include "flash_fwd_dispatch_template.h"
#include "hdim_switch.h"
#include "static_switch.h"

void run_mha_fwd(mcFlashAttn::Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel) {
    (void)force_split_kernel;
    HEADDIM_SWITCH(params.d, {
        ARCH_SWITCH(params.arch, kArch, [&] {
            if (params.fwd_kernel_path == mcFlashAttn::FwdKernelPathNormal) {
                    run_mha_fwd_dispatch<kHeadDimension, kArch>(params,stream);
            }else {
                mcFlashAttn::run_mha_fwd_splitkv_dispatch<kHeadDimension, kArch>(params, stream);
            }
        });
    });
}
