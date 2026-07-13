import argparse
import yaml
import os
import shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Tuple
import numpy as np
import itertools
from bool_switch_process import construct_bool_config

DTYPE_MAP = {
    "bf16": "mctlass::bfloat16_t",
    "fp16": "mctlass::half_t",
}


SM = [80]
BWD_HEADER="""// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_parameter.h"
#include "flash_bwd_global.h"
#include "kernel_traits.h"
#include <mctlass/numeric_types.h>
"""

KERNEL_IMPL_TEMPLATE_BWD = """

template __global__ void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<
                Flash_bwd_kernel_traits<
                    {HEAD_DIM_QK},
                    {BLOCK_M},
                    {BLOCK_N},
                    {k_nwarps},
                    {ATOM_LAYOUT_MSDP},
                    {ATOM_LAYOUT_NDKV},
                    {ATOM_LAYOUT_MDQ},
                    {IS_V_IN_REGS},
                    {IS_K_IN_REGS},
                    {NO_DOUBLE_BUFFER},
                    {DTYPE},
                    {HEAD_DIM_V}>,
                /*Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, Is_deterministic, IsBalance, Arch*/
                {Is_dropout}, {Is_causal}, {Is_local}, {Has_alibi}, {Has_attn_mask}, {Is_even_MN}, {Is_even_K}, {Is_softcap}, {Is_deterministic}, {Is_balance}, {Arch}
                >(mcFlashAttn::Flash_bwd_params params, int gridtype);

"""

FWD_HEADER = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_parameter.h"
#include "flash_fwd_global.h"
#include "kernel_traits.h"
#include <mctlass/numeric_types.h>
"""

KERNEL_IMPL_TEMPLATE_FWD = """

template __global__ void flash_fwd_kernel<
                Flash_fwd_kernel_traits<
                    {HEAD_DIM_QK},
                    {BLOCK_M},
                    {BLOCK_N},
                    {k_nwarps},
                    {Is_Q_in_regs_},
                    {Share_Q_K_smem_},
                    {DTYPE},
                    {HEAD_DIM_V}>,
                /* Is_dropout, Is_causal, Is_local, Has_alibi, Has_attn_mask, IsEvenMNConst, IsEvenKConst, Is_softcap, ReturnSoftmaxConst, Rowblock_Parallel_Num, Merge_attn_mask_ldg, Arch*/
                {Is_dropout}, {Is_causal}, {Is_local}, {Has_alibi}, {Has_attn_mask}, {Is_even_MN}, {Is_even_K}, {Is_softcap}, {Return_softmax}, {Rowblock_Parallel_Num}, {Merge_attn_mask_ldg}, {Arch}
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);

"""

FWD_SPLIT_HEADER = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_parameter.h"
#include "flash_fwd_split_global{SUFFIX}.h"
#include "kernel_traits.h"
#include <mctlass/numeric_types.h>
"""

KERNEL_IMPL_TEMPLATE_FWD_SPLITKV = """

template __global__ void flash_fwd_splitkv_kernel<
                Flash_fwd_kernel_traits<
                    {HEAD_DIM_QK},
                    {BLOCK_M},
                    {BLOCK_N},
                    {k_nwarps},
                    {Is_Q_in_regs_},
                    {Share_Q_K_smem_},
                    {DTYPE},
                    {HEAD_DIM_V}>,
                /* Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst, Is_softcap, Split, Append_KV, Is_page_attn, Arch*/
                {Is_causal}, {Is_local}, {Has_alibi}, {Is_even_MN}, {Is_even_K}, {Is_softcap}, {Split}, {AppendKV}, {Is_page_attn}, {Arch}
                >(mcFlashAttn::Flash_fwd_params params, const int num_m_block, const int block_type);

"""
class Kernel:
    _id_counter = 0

    def __init__(self, config, dtype, api, bool_list):
        # kernel traits params
        self.hdim_qk = config["hdim_qk"]
        self.hdim_v = config["hdim_v"]
        self.block_m = config["block_m"]
        self.block_n = config["block_n"]
        self.k_nwarps = config["k_nwarps"]
        self.dtype = dtype
        self.dtype_short = next(key for key, value in DTYPE_MAP.items() if value == dtype)
        self.api = api
        # bool switch lists
        # self.dropout = bool_list["Is_dropout"]
        # self.causal = bool_list["Is_causal"]
        # self.Is_local = bool_list["Is_local"]
        # self.Has_alibi = bool_list["Has_alibi"]
        # self.Has_attn_mask = bool_list["Has_attn_mask"]
        # self.Is_even_MN = bool_list["Is_even_MN"]
        # self.Is_even_K = bool_list["Is_even_K"]
        self.bool_list = bool_list

        self.id = Kernel._id_counter
        Kernel._id_counter += 1

    @property
    def template(self) -> str:
        raise NotImplementedError("Subclasses should implement this method")

class BwdKernel(Kernel):
    def __init__(self, config, dtype, dropout, causal, bool_list):
        super().__init__(config, dtype, "bwd", bool_list)
        self.atom_layout_msdp = config["atom_layout_msdp"]
        self.atom_layout_ndkv = config["atom_layout_ndkv"]
        self.atom_layout_mdq = config["atom_layout_mdq"]
        self.is_v_in_regs = config["is_v_in_regs"]
        self.is_k_in_regs = config["is_k_in_regs"]
        self.no_double_buffer = config["no_double_buffer"]
        self.name=self.kernel_name
        # self.Is_deterministic = bool_list['Is_deterministic']
        # self.Is_balance = Is_balance
        self.arch = 'Arch::xcore1000' if self.bool_list["Arch"]=='xcore1000' else 'Arch::xcore1500'

    @property
    def template(self) -> str:
        return KERNEL_IMPL_TEMPLATE_BWD.format(
            HEAD_DIM_QK=self.hdim_qk,
            HEAD_DIM_V=self.hdim_v,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            k_nwarps=self.k_nwarps,
            ATOM_LAYOUT_MSDP=self.atom_layout_msdp,
            ATOM_LAYOUT_NDKV=self.atom_layout_ndkv,
            ATOM_LAYOUT_MDQ=self.atom_layout_mdq,
            IS_V_IN_REGS=str(self.is_v_in_regs).lower(),
            IS_K_IN_REGS=str(self.is_k_in_regs).lower(),
            NO_DOUBLE_BUFFER=str(self.no_double_buffer).lower(),
            DTYPE=self.dtype,
            Is_dropout=self.bool_list["Is_dropout"].lower(),
            Is_causal=self.bool_list["Is_causal"].lower(),
            Is_local=self.bool_list["Is_local"].lower(),
            Has_alibi = self.bool_list["Has_alibi"].lower(),
            Has_attn_mask = self.bool_list["Has_attn_mask"].lower(),
            Is_even_MN = self.bool_list["Is_even_MN"].lower(),
            Is_even_K = self.bool_list["Is_even_K"].lower(),
            Is_softcap = self.bool_list["Is_softcap"].lower(),
            Is_deterministic = self.bool_list["Is_deterministic"].lower(),
            Is_balance = self.bool_list["Is_balance"].lower(),
            Arch = self.arch,
        )

    @property
    def filename(self) -> str:
        return f"flash_bwd_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_\
        {self.k_nwarps}wave_{self.dtype_short}_{'dropout_' if self.bool_list['Is_dropout'] == 'true' else ''}\
        {'causal_' if self.bool_list['Is_causal'] == 'true' else ''}{'Is_local_' if self.bool_list['Is_local'] == 'true' else ''}\
        {'alibi_' if self.bool_list['Has_alibi'] == 'true' else ''}{'attn_mask_' if self.bool_list['Has_attn_mask'] == 'true' else ''}\
        {'evemn_' if self.bool_list['Is_even_MN'] == 'true' else ''}{'evenk_' if self.bool_list['Is_even_K'] == 'true' else ''}\
        {'scap_' if self.bool_list['Is_softcap'] == 'true' else ''}{'det_' if self.bool_list['Is_deterministic'] == 'true' else ''}\
        {'bala_' if self.bool_list['Is_balance'] == 'true' else ''}\
        sm80.cpp".replace(" ", "")

    @property
    def file_name(self) -> str:
        return f"flash_bwd_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_\
        {self.k_nwarps}wave_{self.dtype_short}_{'dropout_' if self.bool_list['Is_dropout'] == 'true' else ''}\
        {'causal_' if self.bool_list['Is_causal'] == 'true' else ''}\
        sm80.cpp".replace(" ", "")

    @property
    def kernel_name(self) -> str:
        return
        # return f"flash_bwd_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_\
        # {self.k_nwarps}wave_{self.atom_layout_msdp}x{self.atom_layout_ndkv}x{self.atom_layout_mdq}_\
        # {self.is_v_in_regs}_{self.is_k_in_regs}_{self.dtype_short}_{'dropout_' if self.causal == 'true' else ''}{'causal_' if self.causal == 'true' else ''}"

class FwdKernel(Kernel):
    def __init__(self, config, dtype, dropout, causal, bool_list):
        super().__init__(config, dtype, "fwd", bool_list)
        self.Is_Q_in_regs_ = config["Is_Q_in_regs"]
        self.Share_Q_K_smem_ = config["Share_Q_K_smem"]
        self.Rowblock_Parallel_Num_ = 2 if self.bool_list["Rowblock_Parallel_Num"]=='true' else 1
        self.name=self.kernel_name
        self.dropout = dropout
        self.causal = causal
        self.arch = 'Arch::xcore1000' if self.bool_list["Arch"]=='xcore1000' else 'Arch::xcore1500'

    @property
    def template(self) -> str:
        return KERNEL_IMPL_TEMPLATE_FWD.format(
            HEAD_DIM_QK=self.hdim_qk,
            HEAD_DIM_V=self.hdim_v,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            k_nwarps=self.k_nwarps,
            Is_Q_in_regs_=str(self.Is_Q_in_regs_).lower(),
            Share_Q_K_smem_=str(self.Share_Q_K_smem_).lower(),
            DTYPE=self.dtype,
            Is_dropout=self.bool_list["Is_dropout"].lower(),
            Is_causal=self.bool_list["Is_causal"].lower(),
            Is_local=self.bool_list["Is_local"].lower(),
            Has_alibi = self.bool_list["Has_alibi"].lower(),
            Has_attn_mask = self.bool_list["Has_attn_mask"].lower(),
            Is_even_MN = self.bool_list["Is_even_MN"].lower(),
            Is_even_K = self.bool_list["Is_even_K"].lower(),
            Is_softcap = self.bool_list["Is_softcap"].lower(),
            Return_softmax= self.bool_list["Return_softmax"].lower(),
            Rowblock_Parallel_Num= self.Rowblock_Parallel_Num_ ,
            Merge_attn_mask_ldg= self.bool_list["Merge_attn_mask_ldg"].lower(),
            Arch = self.arch,
        )

    @property
    def filename(self) -> str:
        return f"flash_fwd_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_\
        {self.k_nwarps}wave_{self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_{self.dtype_short}_\
        {'dropout_' if self.bool_list['Is_dropout'] == 'true' else ''}\
        {'causal_' if self.bool_list['Is_causal'] == 'true' else ''}{'Is_local_' if self.bool_list['Is_local'] == 'true' else ''}\
        {'alibi_' if self.bool_list['Has_alibi'] == 'true' else ''}{'attn_mask_' if self.bool_list['Has_attn_mask'] == 'true' else ''}\
        {'evemn_' if self.bool_list['Is_even_MN'] == 'true' else ''}{'evenk_' if self.bool_list['Is_even_K'] == 'true' else ''}\
        {'scap_' if self.bool_list['Is_softcap'] == 'true' else ''}{'rsx_' if self.bool_list['Return_softmax'] == 'true' else ''}\
        {'rpn_' if self.bool_list['Rowblock_Parallel_Num'] == 'true' else ''}\{'maml_' if self.bool_list['Merge_attn_mask_ldg'] == 'true' else ''}\
        sm80.cpp".replace(" ", "")

    @property
    def file_name(self) -> str:
        return f"flash_fwd_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_\
        {self.k_nwarps}wave_{self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_{self.dtype_short}_\
        {'dropout_' if self.bool_list['Is_dropout'] == 'true' else ''}\
        sm80.cpp".replace(" ", "")

    @property
    def kernel_name(self) -> str:
        return
        # return f"flash_fwd_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_\
        # {self.k_nwarps}wave_{self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_{self.dtype_short}_\
        # {'dropout_' if self.bool_list['Is_dropout'] == 'true' else ''}\
        # {'causal_' if self.bool_list['Is_causal'] == 'true' else ''}sm80.cpp"

class FwdSplitKernel(Kernel):
    def __init__(self, config, dtype, dropout, causal, bool_list):
        super().__init__(config, dtype, "fwd_splitkv", bool_list)
        self.Is_Q_in_regs_ = config["Is_Q_in_regs"]
        self.Share_Q_K_smem_ = config["Share_Q_K_smem"]
        self.name=self.kernel_name
        self.dropout = dropout
        self.causal = causal
        self.arch = 'Arch::xcore1000' if self.bool_list["Arch"]=='xcore1000' else 'Arch::xcore1500'

    @property
    def template(self) -> str:
        return KERNEL_IMPL_TEMPLATE_FWD_SPLITKV.format(
            HEAD_DIM_QK=self.hdim_qk,
            HEAD_DIM_V=self.hdim_v,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            k_nwarps=self.k_nwarps,
            Is_Q_in_regs_=str(self.Is_Q_in_regs_).lower(),
            Share_Q_K_smem_=str(self.Share_Q_K_smem_).lower(),
            DTYPE=self.dtype,
            Is_causal=self.bool_list["Is_causal"].lower(),
            Is_local=self.bool_list["Is_local"].lower(),
            Has_alibi = self.bool_list["Has_alibi"].lower(),
            Is_even_MN = self.bool_list["Is_even_MN"].lower(),
            Is_even_K = self.bool_list["Is_even_K"].lower(),
            Is_softcap = self.bool_list["Is_softcap"].lower(),
            Split = self.bool_list["Split"].lower(),
            AppendKV = self.bool_list["AppendKV"].lower(),
            Is_page_attn = self.bool_list["Is_page_attn"].lower(),
            Arch = self.arch,
        )

    @property
    def filename(self) -> str:
        return f"flash_fwd_splitkv_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_\
        {self.k_nwarps}wave_{self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_{self.dtype_short}_\
        {'causal_' if self.bool_list['Is_causal'] == 'true' else ''}{'Is_local_' if self.bool_list['Is_local'] == 'true' else ''}\
        {'alibi_' if self.bool_list['Has_alibi'] == 'true' else ''}{'evemn_' if self.bool_list['Is_even_MN'] == 'true' else ''}\
        {'evenk_' if self.bool_list['Is_even_K'] == 'true' else ''}{'scap_' if self.bool_list['Is_softcap'] == 'true' else ''}\
        {'split_' if self.bool_list['Split'] == 'true' else ''}{'appkv_' if self.bool_list['AppendKV'] == 'true' else ''}\
        {'pattn_' if self.bool_list['Is_page_attn'] == 'true' else ''}\
        sm80.cpp".replace(" ", "")

    @property
    def kernel_id(self) -> str:
        return f"flash_fwd_splitkv_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_\
        {self.k_nwarps}wave_{self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_{self.dtype_short}_\
        {self.bool_list['Is_local']}_{self.bool_list['Has_alibi']}_\
        {self.bool_list['Is_even_MN']}_{self.bool_list['Is_even_K']}_{self.bool_list['Is_softcap']}_\
        {self.bool_list['Split']}_{self.bool_list['AppendKV']}_{self.bool_list['Is_page_attn']}"

    @property
    def file_name(self) -> str:
        if self.hdim_qk == 256:
            return f"flash_fwd_splitkv_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_\
            {self.k_nwarps}wave_{self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_{self.dtype_short}_\
            {'causal_' if self.bool_list['Is_causal'] == 'true' else ''}{'Is_local_' if self.bool_list['Is_local'] == 'true' else ''}\
            {'alibi_' if self.bool_list['Has_alibi'] == 'true' else ''}\
            sm80.cpp".replace(" ", "")
        else:
            return f"flash_fwd_splitkv_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_\
            {self.k_nwarps}wave_{self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_{self.dtype_short}_\
            {'causal_' if self.bool_list['Is_causal'] == 'true' else ''}{'Is_local_' if self.bool_list['Is_local'] == 'true' else ''}\
            sm80.cpp".replace(" ", "")

    @property
    def kernel_name(self) -> str:
        return
        # return f"flash_fwd_splitkv_hdimqk{self.hdim_qk}_hdimv{self.hdim_v}_m{self.block_m}n{self.block_n}_{self.k_nwarps}wave_\
        # {self.Is_Q_in_regs_}_{self.Share_Q_K_smem_}_{self.dtype_short}__\
        # {'dropout_' if self.dropout == 'true' else ''}{'causal_' if self.causal == 'true' else ''}"

def match_kernel(kernel, option):
    head_dims, dtypes, apis = option
    return ((head_dims is None or kernel.hdim_qk in head_dims) and
            (dtypes is None or kernel.dtype_short in dtypes) and
            (apis is None or kernel.api in apis))

def get_all_kernels(include=None, exclude=None, arch='xcore1000', bool_list=None) -> List[Kernel]:
    #kernels = []
    candidates_path = f'out/kernel_traits_candidates_{arch}.yaml'
    kernels = {"fwd": [], "bwd": [], "fwd_split": []}
    unique_configs = set()
    with open(candidates_path, 'r') as file:
        candidates = yaml.safe_load(file)
    for operation, configs in candidates.items():
        for dtype_short, dtype in DTYPE_MAP.items():
            for is_causal in [False, True]:
                for dropout in [False, True]:
                    for config in configs:
                        head_dim = config["hdim_qk"]
                        bool_configs = construct_bool_config(operation, head_dim, arch)
                        for bool_config in bool_configs:
                            kernel = None
                            if operation == "fwd":
                                kernel = FwdKernel(config, dtype, dropout, is_causal, bool_config)
                            elif operation == "fwd_split":
                                kernel = FwdSplitKernel(config, dtype, dropout, is_causal, bool_config)
                            elif operation == "bwd":
                                kernel = BwdKernel(config, dtype, dropout, is_causal, bool_config)

                            config_id = kernel.filename
                            if config_id not in unique_configs:
                                unique_configs.add(config_id)
                                if include and not match_kernel(kernel, include):
                                    continue
                                if exclude and match_kernel(kernel, exclude):
                                    continue
                                if bool_list and not match_bools(kernel, bool_list):
                                    continue
                                kernels[operation].append(kernel)

    for key, kernel_list in kernels.items():
        print(f"{key}:{len(kernel_list)}")
    return kernels


def write_kernel(kernels: Kernel, autogen_dir: Path, api: str) -> None:
    """
    Generate one function and one file, not batch generation.
    """
    for key, kernel_list in kernels.items():
        op_dir = autogen_dir / key
        op_dir.mkdir(parents=True, exist_ok=True)
        hdim256 = 0
        hdim128 = 0
        for kernel in kernel_list:
            filename = kernel.file_name.replace(" ", "")
            kernel_path = op_dir / filename

            if kernel.api == "fwd":
                header = FWD_HEADER
            elif kernel.api == "bwd":
                header = BWD_HEADER
            else:
                if kernel.hdim_qk == 256:
                    hdim256 = hdim256 + 1
                if kernel.hdim_qk == 128:
                    hdim128 = hdim128 + 1
                else:
                    header = FWD_SPLIT_HEADER.format(
                        SUFFIX=""
                    )
            if kernel_path.exists():
                with open(kernel_path,"a") as file:
                    file.write(kernel.template)
            else:
                with open(kernel_path,"a") as file:
                    file.write(header + kernel.template)

    print(f"split_hdim256:{hdim256},split_hdim128:{hdim128}")

def parse_args() -> Tuple[Optional[Tuple[List[int], List[str], List[str]]], Optional[Tuple[List[int], List[str], List[str]]]]:
    parser = argparse.ArgumentParser(description="Generate kernel files and candidate pool.")
    parser.add_argument("-i", "--include", nargs=3, metavar=("HEAD_DIM", "DTYPE", "api"),
                        help="Include only the specified head_dim, dtype, and api. Use '_' to ignore an option.")
    parser.add_argument("-e", "--exclude", nargs=3, metavar=("HEAD_DIM", "DTYPE", "api"),
                        help="Exclude the specified head_dim, dtype, and api. Use '_' to ignore an option.")
    parser.add_argument("-a", "--api", type=str, help="Specify the api type (pyapi/capi) for generated kernels.")
    parser.add_argument("--arch", type=str, help="device arch",  default="xcore1000")
    args = parser.parse_args()

    def parse_option(option):
        if option is None:
            return None
        hdim_qk, dtype, api = option
        head_dims = [int(hd) for hd in hdim_qk.split(',')] if hdim_qk != "_" else None
        dtypes = dtype.split(',') if dtype != "_" else None
        apis = api.split(',') if api != "_" else None
        return head_dims, dtypes, apis

    include = parse_option(args.include)
    exclude = parse_option(args.exclude)

    return args.api, include, exclude, args.arch

def main() -> None:
    api, include, exclude, arch = parse_args()

    output_dir = "../../build_kernel/full_kernels/" + arch
    output_dir = Path(output_dir)
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels = get_all_kernels(include,exclude,arch)
    write_kernel(kernels, output_dir, api)

if __name__ == "__main__":
    main()
