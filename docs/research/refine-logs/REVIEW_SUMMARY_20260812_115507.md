# Review Summary

**Problem**：OPD 的 token-local supervision 可能不足以纠正失败推理轨迹。  
**Initial Approach**：在高差异位置生成并验证 teacher bridge，然后对有效 chunk 训练。  
**Date**：2026-08-12  
**Rounds**：2  
**Final Score**：7.25 / 10  
**Final Verdict**：REVISE

## Problem Anchor

识别一段连续 teacher correction 是否比孤立 token 更能将 student 从失败 prefix 带回可完成轨迹，并检验这种 external intervention value 是否能真正写入参数。

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What Changed | Status |
|---|---|---|---|
| 1 | on-policy 口径过宽；bridge intervention 不等于 parameter utility；贡献与 TOPD/Relay/FutureBridge 重叠 | 收缩为 recovery horizon；诚实定义 hybrid roll-in；提出两个 kill experiments | 部分解决 |
| 2 | oracle 不等于 scalable selector；writability 不等于 generalization | 冻结三类 horizon 和两类训练后评价；定义停止规则 | 设计解决，待实验 |

## Final Status

- Anchor：preserved
- Focus：tight，唯一中心变量为 recovery span
- Modernity：appropriately frontier-aware，使用 counterfactual intervention、nested horizons、cross-fitting 与 cluster-aware uncertainty
- Remaining blocker：尚无结果证明 horizon heterogeneity、cheap estimation 和 intervention-to-update transfer
