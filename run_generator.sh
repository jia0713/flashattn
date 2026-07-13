#!/bin/bash
set +ex
if [[ "$1" == "capi" ]]; then
    if [ -d "./build_cpp" ]; then
        if [ -d "./build_cpp/kernels" ]; then
            rm -rf "./build_cpp/kernels"
        fi
    else
        mkdir -p "./build_cpp/kernels"
    fi
    api="capi"
else
    if [ -d "./build" ]; then
        if [ -d "./build/kernels" ]; then
            rm -rf "./build/kernels"
        fi
    else
        mkdir -p "./build"
    fi
    api="pyapi"
fi

if [ -d "./build_kernel/full_kernels"]; then
    rm -rf "./build_kernel/full_kernels"
fi

if [ -d "./build_kernel/run_flash_template"]; then
    rm -rf "./build_kernel/run_flash_template"
fi

arch_list=()
if [ "$4" = "xcore1000" ]; then
    arch_list+=("xcore1000")
elif [ "$4" = "xcore1500" ]; then
    arch_list+=("xcore1500")
else
    arch_list+=("xcore1500" "xcore1000")
fi

cd tools/generator
rm -rf out
echo $4
for arch in "${arch_list[@]}"; do
    echo $arch
    python generate_kernel_traits.py --arch $arch
    # Enable this command when push your changes

    if [[ "$2" == "0" ]];then
        # Generate all kernel
        python generate_kernels.py -a $api --arch $arch
    else
        # Generate the kernel specified by parameter 2 for hdim
        if [[ "$3" == "ALL" || "$3" == "all" ]];then
            python generate_kernels.py -a $api -i $2 _ _ -u --arch $arch
        else
            python generate_kernels.py -a $api -i $2 "$3" _ -u --arch $arch
        fi
    fi

    # Generate full kernels with bool switch
    python generate_fullkernels.py -a $api --arch $arch
done

# Generate tuning table for C500 and C550
python generate_tuning_table.py --mxc500 ../../solution_bin/flash_attn_tuning_mxc500.bin --mxc550 ../../solution_bin/flash_attn_tuning_mxc550.bin

# Enable following section when you need to filter the compiling process
# For further detail of parameter setup, check tools/generator/readme.md
# hdim_filter="32"
# dtype_filter="fp16"
# op_filter="_"
# python generate_kernels.py -a $api -i $hdim_filter $dtype_filter $op_filter

cd ../../
