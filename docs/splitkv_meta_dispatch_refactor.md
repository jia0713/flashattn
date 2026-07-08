# SplitKV Meta Dispatch 重构方案

## 背景

`num_splits` 不只是由 batch/head/sequence length 决定，它还依赖实际 SplitKV kernel 使用的 tile 配置：

- `block_m` 决定 `num_m_blocks = ceil(seqlen_q / block_m)`。
- `block_n` 决定 `num_n_blocks = ceil(seqlen_k / block_n)`。
- 单个 AP 上能并发跑几个 block 会影响 heuristic 里使用的有效 AP 数量。

之前 API 层用一套近似规则估算 `block_m/block_n`，而真正的 kernel tile 是在更晚的 `run_mha_fwd_splitkv_dispatch` 里选择的。这样 workspace 分配、`num_splits` 计算和最终 kernel dispatch 容易出现漂移。

## 目标

让 API 层的 SplitKV 决策和 kernel dispatch 使用同一个完整 kernel trait tuple：

```cpp
struct FwdSplitKernelMeta {
    int arch;
    int headdim;
    int block_m;
    int block_n;
    int nwarps;
    bool is_q_in_regs;
    bool share_q_k_smem;
    int block_num_per_ap;
};
```

API 层在 workspace 分配前计算并保存这份 meta。dispatch 层读取这份 runtime meta，做 assert，然后映射回 compile-time template constants 来实例化 kernel。

## 当前第一版实现

第一版刻意控制范围：不改 C API，不改 generator。

涉及文件：

- `csrc/flash_attn/flash_dispatch/fwd_split_meta.h`
- `csrc/flash_attn/flash_api/flash_parameter.h`
- `csrc/flash_attn/flash_api/flash_splitkv.{h,cpp}`
- `csrc/flash_attn/flash_api/flash_api_fwd.cpp`
- `csrc/flash_attn/flash_api/flash_api_fwd_kvcache.cpp`
- `csrc/flash_attn/flash_dispatch/flash_fwd_dispatch_template.h`

运行流程：

1. API 层先设置基础 `Flash_fwd_params`。
2. API 调用 `compute_params_numsplits(params, user_num_splits, force_split_kernel)`。
3. `compute_params_numsplits` 选择 `FwdSplitKernelMeta`。
4. 选中的 meta 写入 `params.split_*` 字段。
5. 当 `num_splits == 0` 时，用 meta 里的 `block_m/block_n/block_num_per_ap` 计算 heuristic。
6. `malloc_accum_by_numsplits` 在 `params.num_splits` 最终确定后申请 SplitKV accum buffer。
7. `run_mha_fwd_splitkv_dispatch` 读取 `params.split_*`。
8. `FWD_SPLIT_META_SWITCH` 把 runtime tuple 映射成 compile-time template constants，然后 launch kernel。

## 为什么使用 tuple-based dispatch

不建议分别对 `block_m` 和 `block_n` 做独立 switch。

`block_m`、`block_n`、`nwarps`、`is_q_in_regs`、`share_q_k_smem` 是一个完整 kernel trait tuple。某些值单独看是合法的，但组合起来未必有对应 kernel。tuple-based dispatch 可以避免误选一个未配置或未编译的 kernel trait。

## compile-time constants 怎么处理

API 层保存的是 runtime 值，但 kernel template 参数必须是 compile-time constants。

dispatch 层通过检查完整 runtime meta tuple，再暴露 constexpr 值：

```cpp
FWD_SPLIT_META_SWITCH(meta, kBlockM, kBlockN, kNWarps,
                      Is_Q_in_regs, Share_Q_K_smem, [&] {
    Xcore1000::run_flash_splitkv_fwd_template<
        Headdim, kBlockM, kBlockN, kNWarps,
        Is_Q_in_regs, Share_Q_K_smem, elem_type
    >(params, launch_params, stream);
});
```

这样 API 层选出的 `block_m/block_n` 会真正决定 kernel 路径，同时 kernel 仍然拿到 compile-time template constants。

## 当前范围

已包含：

- torch/Python extension forward 路径。
- 使用 SplitKV 的 varlen forward 路径。
- kvcache / dequant kvcache 路径。
- dispatch 前的 SplitKV workspace 分配。
- xcore1000/xcore1500 的 host-side split meta。

暂不包含：

- C API。
- C API 文档。
- generator 改造。
- benchmark 校准后的 `block_num_per_ap`。

## block_num_per_ap

当前 `block_num_per_ap` 是第一版保守 lookup：

- 默认：`2`
- `headdim == 64 && block_m/block_n <= 16`：`16`
- 其他 `headdim == 64`：`4`
- `headdim == 128 && block_m <= 16`：`8`
- `headdim == 128 && block_m <= 32`：`4`

这只是临时策略，后续应该由生成的 kernel metadata 或 benchmark 结果替代。

## 后续 generator 方向

现有两份 kernel traits config 应该成为唯一事实来源：

- `tools/generator/xcore1000_kernel_traits_config.txt`
- `tools/generator/xcore1500_kernel_traits_config.txt`

建议下一步：

1. 扩展或新增 generator，解析 `fwd_split` 段。
2. 生成支持的 `FwdSplitKernelMeta` 表。
3. 生成 dispatch 使用的 tuple switch。
4. 可选：把 `block_num_per_ap` 扩展进 config tuple。

未来 config 形态可以类似：

```text
fwd_split: hdim_qk, hdim_v,
[(block_m, block_n, k_nwarps, Is_Q_in_regs, Share_Q_K_smem, block_num_per_ap)]
```

## 风险和待讨论点

- 手写 meta 表可能和 kernel traits config 漂移，后续应优先 generator 化。
- `block_num_per_ap` 需要 benchmark 验证。
- C API 仍然保留旧逻辑，后续要单独设计。
- 非标准 head dim 依赖和 `HEADDIM_SWITCH` 一致的 effective head dim 规则，需要保持同步。
- unsupported meta 应该直接失败，不应 fallback 到另一套 kernel trait。

## 讨论问题

- `block_num_per_ap` 应该放在 traits config、单独 tuning table，还是 benchmark 输出里？
- generator 是否应该同时生成 runtime meta selection 和 dispatch switch？
- C API 后续是提供 workspace-size query，还是继续让调用方管理 accum tensor？
- meta 长期应该放在 `Flash_fwd_params`，还是放在 host-only launch params 里？
