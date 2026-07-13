import argparse
import yaml
import re
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Tuple
import glob
import shutil

from kernel_class import Kernel, FwdKernel, BwdKernel, FwdSplitKernel, DTYPE_MAP
import kernel_impl_template as temps

def _norm_hdim(s):
    if not s: return None
    s = s.strip().lower()
    if s in ("0", "all", "*", "any"): return None
    return s if s.isdigit() else None

def _norm_dtype(s):
    if not s: return None
    s = s.strip().lower()
    if s in ("all", "*", "any"): return None
    if s in ("fp16","half","float16"):   return "float16"
    if s in ("bf16","bfloat16"):         return "bfloat16"
    return None

_HDIM  = _norm_hdim(os.getenv("HDIM"))
_DTYPE = _norm_dtype(os.getenv("DTYPE"))

def _filter(src, hdim: str|None, dtype: str|None):
    if not src or (not hdim and not dtype):
        return src
    pat_hdq = re.compile(rf"hdimqk_{hdim}_") if hdim else None
    pat_dt  = re.compile(rf"_{dtype}_")      if dtype else None
    out=[]
    for ln in src.splitlines():
        s = ln.strip()
        if not s or s in ("{","}","};","},") or s.startswith("#"):
            out.append(ln); continue
        keep = True
        if pat_hdq and not pat_hdq.search(ln): keep=False
        if keep and pat_dt  and not pat_dt.search(ln):  keep=False
        if keep: out.append(ln)
    return "\n".join(out)

def match_kernel(kernel, option):
    head_dims, dtypes, apis = option
    return ((head_dims is None or kernel.hdim_qk in head_dims) and
            (dtypes is None or kernel.dtype_short in dtypes) and
            (apis is None or kernel.api in apis))

def get_all_kernels(data_file, update, arch, include=None, exclude=None):
    kernels = {"fwd": [], "bwd": [], "fwd_split": []}
    hdim_templates=set()
    content=""
    with open(data_file, 'r') as file:
        candidates = yaml.safe_load(file)
        for api, configs in candidates.items():
            content = " {\n"
            if api == "fwd":
                content += _filter(temps.OLD_FWD_MAP,        _HDIM, _DTYPE)
            elif api == "fwd_split":
                content += _filter(temps.OLD_FWD_SPLIT_MAP,  _HDIM, _DTYPE)
            content += "\n"
            for config in configs:
                kernel = None
                if api == "fwd":
                    kernel = FwdKernel(config, arch)
                elif api == "bwd":
                    kernel = BwdKernel(config, arch)
                else:
                    kernel = FwdSplitKernel(config, arch)

                if include and not match_kernel(kernel, include):
                    continue
                if exclude and exclude != (None, None, None):
                    if match_kernel(kernel, exclude):
                        continue

                kernels[api].append(kernel)
                hdim_templates.add(config['hdim_qk'])
                content += write_cpp_map(kernel, config["kernel_id"], api)
            content += "\n};"
            cpp_map_path = f'out/{api}_map.cpp'
            with open(cpp_map_path, 'a') as cpp_file:
                cpp_file.write(temps.MAP_TEMPLATES[api].format(LIST=content))
    if (update):
        write_hdim_template(hdim_templates)
    return kernels


def write_cpp_map(kernel, key, api):
    content = ""

    value = kernel.template.replace("\n", "").replace(" ", "")
    match = re.search(r"run_flash.*?>", value)

    map_item_template = """{{"{KEY}",{VALUE}}},
    """

    if match:
        current_line = match.group()
        current_line = current_line.replace("cutlass", "mctlass")
        current_line = "Xcore1000::" + current_line
        content += f'{{"{key}", {current_line}}},\n'
    return content

def write_kernel(kernels: Kernel, autogen_dir: Path, kernel_paths, api):
    for key, kernel_list in kernels.items():
        op_dir = autogen_dir / key
        op_dir.mkdir(parents=True, exist_ok=True)
        for kernel in kernel_list:
            filename = kernel.filename.replace(" ", "") + ".cpp"
            (op_dir / filename).write_text(kernel.template)
            kernel_path = op_dir / filename
            kernel_path.write_text(kernel.template)
            kernel_paths.append(str(kernel_path).replace('../../', ''))

def write_hdim_template(hdim_list):
    with open('../../csrc/flash_attn/flash_run/flash_headdim.h', 'w') as hdim_temps:
        hdim_temps.write(temps.HDIM_HEADER_TEMPLATE)
        for hdim in hdim_list:
            hdim_temps.write(temps.HDIM_TEMPLATE.format(HDIM=hdim))

def parse_args() -> Tuple[Optional[Tuple[List[int], List[str], List[str]]], Optional[Tuple[List[int], List[str], List[str]]], str]:
    parser = argparse.ArgumentParser(description="Generate kernel files and candidate pool.")
    parser.add_argument("-u", "--update", action='store_true', help="Update mode, do not enable while compiling.")
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
    update = args.update
    api = args.api if args.api else "pyapi"
    arch = args.arch

    return include, exclude, api, update, arch

def prepare_dir(update, arch):
    files_to_delete = glob.glob("out/*_map.cpp")
    for file_path in files_to_delete:
        os.remove(file_path)

    output_dir = "./out"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if update:
        output_dir = "../../build_kernel/run_flash_template/" + arch
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir

def main() -> None:
    include, exclude, api, update, arch = parse_args()
    # CPP_MAP_PATH = "out/kernel_traits_map.cpp"
    output_dir = prepare_dir(update, arch)

    data_file = Path(f"out/kernel_traits_candidates_{arch}.yaml")
    kernels = get_all_kernels(data_file, update, arch, include, exclude)
    if (update):
        kernel_paths = []
        write_kernel(kernels, output_dir, kernel_paths, api)


if __name__ == "__main__":
    main()
