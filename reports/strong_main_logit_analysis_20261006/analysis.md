# 已完成强模型主输出实验的 logprob 分析

来源：`strong8_weak17_reprompt_greedy_t2_k1_20261005`、`strong4_weak17_reprompt_greedy_t2_k1_20261005`；每组 4631 个 completed rollout。只统计包含 `strong_sampled_token` 和 `replacement_sampled_token` 的实际介入事件。每个介入事件等权，长 rollout 会贡献更多事件。

| 统计 | 8B 主输出，1.7B 检查 | 4B 主输出，1.7B 检查 |
|---|---:|---:|
| 实际介入次数 | 84,526 | 65,866 |
| 介入前 strong−weak logprob gap 均值 | 6.0281 | 5.9807 |
| gap 中位数 | 4.2552 | 4.1944 |
| gap P10 / P90 | 2.2883 / 12.0943 | 2.2820 / 12.0941 |
| 原 token 在强模型中的概率中位数 | 66.0% | 75.3% |
| 原 token 在弱模型中的概率中位数 | 0.855% | 1.017% |
| 原 token 在弱模型中的 rank 中位数 | 3 | 3 |
| 原 token 在弱模型 top100 的覆盖率 | 98.85%（83,552 / 84,526） | 98.82%（65,088 / 65,866） |
| 提示后替换 token 的 logprob 中位数 | −1.3108 | −1.3016 |
| 提示后原 token 的 logprob 和 rank | 未保存 | 未保存 |

这些 gap 描述同一原始上下文下的强弱模型分歧。gap 大于 2 是介入事件的选择条件，因此不能把这些条件统计解释为所有生成 token 的整体差异。两组分歧幅度接近；原 token 大部分仍在弱模型前几个候选位置。

旧 worker 对 repair 后的生成使用 `logprobs=1`，仅保存替换 token 的 logprob。没有保存 repair 后原 token 的 score、top100 或 rank，队列也没有留下可恢复的请求/结果。不同 prompt 下两个不同 token 的概率不能用于判断原 token 的概率升降。

原实验强制禁用原 token；施加禁用后的概率为零是约束造成的结果。要测量 repair 提示的影响，应重放 repair prompt，读取禁用前的分布，对照原 token 的概率、精确 rank、top100 和完整分布距离。重放需要补一次模型前向，不需要重新生成 rollout。

补评分尚未提交：2026-10-06 检查时，集群 SSH 10025 和备用 10024 均连接超时。本地准备 replay 脚本，网络恢复后可提交。
