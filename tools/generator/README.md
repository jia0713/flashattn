# Kernel Generator

This project automates the generation and mapping of kernel configurations.

## Project Structure

- **Python Scripts**:
  - `generate_fullkernels.py`: Generates fullly expanded instantiation of kernel templates including kernel_traits and launch params.
  - `generate_kernel_traits.py`:
  This script validates the single source `kernel_traits.yaml`, generates an
  architecture-specific candidate input (`out/kernel_traits_candidates_<arch>.yaml`),
  and generates the host fwd/split meta registry and dispatch switch.

  - `generate_kernels.py`:
  A general script for generating explicit instantitation of global template functions based on the templates defined in (`kernel_impl_template.py`). It integrates the architecture-specific generated candidate input and other properties to define the kernels. The generated templates will be classified by api types by calling (`kernel_class.py`). It creates a mapping of kernel traits (`kernel_traits_map.cpp`) to configurations for efficient retrieval.

  - `kernel_class.py`: Defines kernel configuration classes used by `generate_kernels.py`.
  - `kernel_impl_template.py`: Defines string templates used by `generate_kernels.py`.

- **Configuration Files**:
  - `kernel_traits.yaml`: The only hand-maintained database of kernel candidates
    and fwd/fwd_split dispatch policy for every architecture.
  - `out/kernel_traits_candidates_<arch>.yaml`: Generated generator input.

- **Generated Files**:
  - `kernel_traits_map.cpp`: C++ `unordered_map` that maps `kernel_id` to kernel configurations.

## Usage

0. **Update Rule**:
  - To modify kernel traits, change only `kernel_traits.yaml`, then regenerate.
    The generator validates every dispatch rule references an existing candidate.

1. **Prepare Kernel Traits**:
   - Define traits in `kernel_traits.yaml`. Every fwd/fwd_split candidate has a
     stable `id`; dispatch rules select by that `id` and may constrain dropout,
     MLA, or maximum query sequence length.

  ```
  fwd: hdim_qk, hdim_v, [(block_m, block_n, k_nwarps, Is_Q_in_regs, Share_Q_K_smem)]
  bwd: hdim_qk, hdim_v, [(block_m, block_n, k_nwarps, atom_layout_msdp, atom_layout_ndkv, atom_layout_mdq, is_v_in_regs, is_k_in_regs, no_double_buffer)]
  fwd_split: hdim_qk, hdim_v, [(block_m, block_n, k_nwarps, Is_Q_in_regs, Share_Q_K_smem)]
  ```

  Example:
  ```
  128, 128, [(64,32,4,True,True)]
  128, 128, [(128,64,4,True,True)]
  ```

2. **Generate Kernel Code**:

   Support arch xcore1000 and xcore1500, the default arch is xcore1000

   - Generate architecture-specific kernel candidates and the host registry

   ```bash
   python generate_kernel_traits.py --arch arch
   ```

   - Generate kernels are only expanded by kernel traits

   ```bash
   python generate_kernels.py -u --arch arch
   ```

   - Generate full kernels are fully expanded by kernel traits and bool switch parameters

   ```bash
   python generate_fullkernels.py --arch arch
   ```


## Key Concepts

- **Kernel ID**:
  Each kernel configuration has a unique ID that describes its properties (e.g., `hdimqk_32_hdimv_32_blockm_32_blockn_32_2_2_2_2_True_True_True`). This ID serves as a key to retrieve specific kernel configurations in the generated C++ map.

- **Kernel Traits**:
  Traits are specific parameters or attributes of a kernel, such as block sizes, memory layout (`atom_layout_mdq`, `atom_layout_msdp`, etc.), register usage (`is_k_in_regs`, `is_v_in_regs`), and the number of warps (`k_nwarps`).

## Example

### YAML Example

Here is an example of how a kernel configuration is defined in the YAML file:

```yaml
bwd:
  - block_m: 32
    block_n: 32
    hdim_qk: 32
    hdim_v: 32
    ...
    kernel_id: hdimqk_32_hdimv_32_blockm_32_blockn_32_2_2_2_2_True_True_True
    no_double_buffer: true
  ...
fwd:
  - block_m: 128
    block_n: 128
    hdim_qk: 32
    hdim_v: 32
    ...
    kernel_id: hdimqk_32_hdimv_32_blockm_128_blockn_128_bfloat16_4_True_True_False
  ...
```

### Generated C++ Example

The Python scripts convert the YAML configurations into C++ code like this:

```cpp
std::unordered_map<std::string, KernelConfig> kernel_configs = {
    {"hdimqk_32_hdimv_32_blockm_32_blockn_32_2_2_2_2_True_True_True", KernelConfig(2, 2, 2, 32, 32, 32, 32, true, true, 2, true)},
    {"hdimqk_64_hdimv_64_blockm_64_blockn_64_4_4_1_4_True_True_True", KernelConfig(4, 4, 1, 64, 64, 64, 64, true, true, 4, true)},
};
```
