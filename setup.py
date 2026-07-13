# Copyright (c) 2023, Tri Dao.

import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

from typing import Optional, Union

# from build_kernel import build_kernel_static

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "flash_attn"

BASE_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"
)

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    # raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    # output = raw_output.split()
    # release_idx = output.index("release") + 1
    # bare_metal_version = parse(output[release_idx].split(",")[0])
    raw_output = "nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2023 NVIDIA Corporation Built on Mon_Apr__3_17:16:06_PDT_2023 Cuda compilation tools, release 12.1, V12.1.105 Build cuda_12.1.r12.1/compiler.32688072_0"
    bare_metal_version = Version("12.1")
    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]

cmdclass = {}
ext_modules = []

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]

    check_if_cuda_home_none("flash_attn")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.6"):
            raise RuntimeError(
                "FlashAttention is only supported on CUDA 11.6 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )
    # cc_flag.append("-gencode")
    # cc_flag.append("arch=compute_75,code=sm_75")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if CUDA_HOME is not None:
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

    lib_dir = Path(CUDA_HOME).parent.parent / "lib"
    libraries=["mcblas"]
    extra_objects = ['{}/lib{}.so'.format(lib_dir, l) for l in libraries]
    prebuilt_include_dirs = []

    env_var="FLASHATTN_BUILD_PROJECTS"
    build_projects = os.getenv(env_var, "").split(":") if os.getenv(env_var) else [""]
    for project in map(str.strip, build_projects):
        if not project:
            extra_objects.append("./build_kernel/libmcFlashAttnKernelXcore1000Static.a")
            extra_objects.append("./build_kernel/libmcFlashAttnKernelXcore1500Static.a")
        elif project == "C500":
            extra_objects.append("./build_kernel/libmcFlashAttnKernelXcore1000Static.a")
        elif project == "C600":
            extra_objects.append("./build_kernel/libmcFlashAttnKernelXcore1500Static.a")
        else:
            print(f"Unknown build project:{project}")


    extra_objects.append("./build_kernel/libmcFlashAttnHostStatic.a")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    ext_modules.append(
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=[
                "csrc/flash_attn/flash_api/flash_api.cpp",
                "csrc/flash_attn/flash_api/flash_api_bwd.cpp",
                "csrc/flash_attn/flash_api/flash_api_fwd_kvcache.cpp",
                "csrc/flash_attn/flash_api/flash_api_fwd.cpp",
                "csrc/flash_attn/flash_api/flash_parameter_utils.cpp",
                "csrc/flash_attn/flash_api/flash_splitkv.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-w"] + generator_flag,
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-w",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        "-fno-strict-aliasing",
                        "-mllvm",
                        "-metaxgpu-inlinescope=50",
                        "-mllvm",
                        "-metaxgpu-disable-bsm-offset=0",#try to merge ldg/lds
                        "-D__FAST_HALF_CVT__",
                        "-D__MERGE_LDS_B64",
                        "-Xclang",
                        "-menable-no-nans",
                        "-mllvm",
                        "-metaxgpu-enable-promote-kernel-arguments",
                        "-mllvm",
                        "-metaxgpu-const-KArgs-LDU=1",
                        "-mllvm",
                        "-metaxgpu-enable-unorder-dispatch",
                        "-mllvm",
                        "-metaxgpu-LdgBSM-store-MD=0",
                        "-DHDIM_ALL",
                        # "--ptxas-options=-v",
                        # "--ptxas-options=-O2",
                        # "-lineinfo",
                        # "-D__USE_128_32x32", # use 32x32tile in hdim128, otherwise use 64x64
                    ]
                    + generator_flag
                    + cc_flag
                ),
            },
            extra_link_args=["-T", "a.lds"],
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attn" / "flash_api",
                Path(this_dir) / "csrc" / "flash_attn" / "flash_dispatch",
                Path(this_dir) / "csrc" / "flash_attn" / "flash_kernel" ,
                Path(this_dir) / "csrc" / "flash_attn" / "flash_kernel" / "flash_global",
                Path(this_dir) / "csrc" / "flash_attn" / "flash_kernel" / "feature",
                Path(this_dir) / "csrc" / "flash_attn" / "flash_kernel" / "bwd_kernel",
                Path(this_dir) / "csrc" / "flash_attn" / "flash_kernel" / "fwd_kernel",
                Path(this_dir) / "csrc" / "flash_attn" / "flash_kernel" / "kernel_traits",
                Path(this_dir) / "csrc" / "flash_attn" / "flash_run",
                Path(this_dir) / "csrc" / "flash_attn" / "utils",
                Path(this_dir) / "csrc" / "common",
                Path(this_dir) / "csrc" / "mctlass" / "include",
            ] + prebuilt_include_dirs,
            extra_objects = extra_objects,
        )
    )

def get_sha(flash_attn_root: Union[str, Path]) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=flash_attn_root)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "Unknown"

def get_package_version():
    flash_attn_root = Path(__file__).parent.parent
    sha = get_sha(flash_attn_root)
    with open(Path(this_dir) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = str(ast.literal_eval(version_match.group(1)))
    maca_version = os.environ.get("CORE_MODULE_MACA_VERSION")
    if maca_version:
        return public_version + "+metax" + maca_version
    elif sha != "Unknown":
        return public_version + "+git" + sha[:7]
    else:
        return public_version

def get_torch_version():
    torch_version_raw = parse(torch.__version__)
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    return torch_version

def get_wheel_url():
    # Determine the version numbers that will be used to determine the correct wheel
    # We're using the CUDA version used to build torch, not the one currently installed
    # _, cuda_version_raw = get_cuda_bare_metal_version(CUDA_HOME)
    torch_cuda_version = parse(torch.version.cuda)
    # For CUDA 11, we only compile for CUDA 11.8, and for CUDA 12 we only compile for CUDA 12.2
    # to save CI time. Minor versions should be compatible.
    torch_cuda_version = parse("11.8") if torch_cuda_version.major == 11 else parse("12.2")
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    flash_version=(get_package_version().split("+")[0] + "+torch" + get_torch_version()) if get_torch_version() else get_package_version().split("+")[0],
    # cuda_version = f"{cuda_version_raw.major}{cuda_version_raw.minor}"
    cuda_version = f"{torch_cuda_version.major}{torch_cuda_version.minor}"
    torch_version = get_torch_version()
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    # Determine wheel URL based on CUDA version, torch version, python version and OS
    wheel_filename = f"{PACKAGE_NAME}-{flash_version}+cu{cuda_version}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{flash_version}", wheel_name=wheel_filename)
    return wheel_url, wheel_filename


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all flash attention installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print("Guessing wheel URL: ", wheel_url)
        #try:
        #    urllib.request.urlretrieve(wheel_url, wheel_filename)

        #     # Make the archive
        #     # Lifted from the root wheel processing command
        #     # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
        #     if not os.path.exists(self.dist_dir):
        #         os.makedirs(self.dist_dir)

        #     impl_tag, abi_tag, plat_tag = self.get_tag()
        #     archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

        #     wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
        #     print("Raw wheel path", wheel_path)
        #     os.rename(wheel_filename, wheel_path)
        # except urllib.error.HTTPError:
        #     print("Precompiled wheel not found. Building from source...")
        #     # If the wheel could not be downloaded, build from source
        #     super().run()
        print("Building wheel from source...")
        super().run()


# result = subprocess.run(['python', 'build_kernel.py'], text=True)
# if result.returncode != 0:
#     print(f"build kernel static librayr failed: {result.returncode}")
#     print("error:", result.stderr)
#     sys.exit(-1)

setup(
    name=PACKAGE_NAME,
    version=(get_package_version().split("+")[0] + "+torch" + get_torch_version()) if get_torch_version() else get_package_version().split("+")[0],
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "tools",
            "dist",
            "docs",
            "benchmarks",
            "flash_attn.egg-info",
        )
    ),
    package_data = {
        "flash_attn" : [
            "tuning/tuner_config.yaml"
        ]
    },
    author="Tri Dao",
    author_email="trid@cs.stanford.edu",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/flash-attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
        "PyYAML",
    ],
)
