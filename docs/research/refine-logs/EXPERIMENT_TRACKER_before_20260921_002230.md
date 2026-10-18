# 两项长度实验执行清单

日期：2026-09-14。所有生成任务尚未启动；此表不修改旧实验运行状态。

| Run ID | 阶段 | 目的 | 配置/数据 | 新生成量 | 优先级 | 状态 |
|---|---|---|---|---:|---|---|
| L001 | M0/M1 | 旧结果配对与hash核验 | 2254题，主τ2.5，k1/4/8/16；其他同规则阈值附录 | 0 | MUST | TODO |
| L002 | M1 | 四格表和BBH子任务 | 主1vs8，次1vs4/16；27×40 BBH | 0 | MUST | TODO |
| L003 | M0 | E2 prefix/ban/EOS sanity | 9anchor，J1/R2 | 108续写+9teacher片段 | MUST | TODO |
| L004 | M2 | 逐题策略独立种子pilot | 600随机题，τ2.5，k1/4/8，3新seed | 5400整轨迹 | MUST | TODO |
| L005 | M3 | 同状态长度pilot | 90anchor，J2/R4，T1/4/8/16+两student控制 | 3600续写+180teacher片段 | MUST | TODO |
| L006 | M3 | 方差/功效与独立确认冻结 | pilot数据，锁定采样、ε和分析 | 0 | MUST | TODO |
| L007 | M4 | 同状态独立确认 | 120新anchor，J3/R8 | 13440续写+360teacher片段 | MUST | TODO |
| L008 | M4 | 聚类统计与机制抽样 | V曲线、Δ、案例盲审、答案直接代写检查 | 0 | MUST | TODO |
| L009 | M5 | 完整题集3seed确认 | 2254题，k1/4/8，含pilot | 增14886整轨迹 | OPTIONAL | TODO |
| L010 | M6 | 独有成功题独立复测 | 最多100题，k1/k8各16续写 | 最多3200整轨迹 | OPTIONAL | TODO |
| L011 | M6 | k16、第二阈值或语义边界 | 仅根据预定判断门槛追加 | 单独预算 | OPTIONAL | TODO |

完整协议见 INTERVENTION_LENGTH_EXPERIMENT_PLAN.md。先L001/L002和L003，再独立安排L004/L005；当前没有授权或启动执行这些GPU实验。
