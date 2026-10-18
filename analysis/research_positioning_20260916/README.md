# 2026-09-16 研究定位数值核对

原始输入：用户粘贴表格；每个配置单次 rollout。`k1_gap_recovery.csv` 在所有任务使用同一个配置 k=1, tau=2，未挑选每任务最优。

差距弥合率=(A_intervention-A_student)/(A_teacher-A_student)。这只是观察到的准确率差的标准化，不是同一批错误被救回的比例、统计等价性或能力恢复证明。

每次一个 token 不等于全程只介入一次。Teacher占比仅为保留输出token占比，不能用于推断总算力或延迟节省。口径及停止规则见 ../../outputs/benchmark-viz/source.md。

最有价值的下一步：先做逐题配对与独立种子复核；再固定prefix和首个teacher token，比较嵌套teacher片段的长度，交还后关闭后续帮助。加入禁止原token后的student重采样控制，区分一次修正、连续支持、重复救援、直接答案注入和格式/EOS改变。

最近邻：Shen et al. ICML 2026 (2605.16874); Huang et al. RelayLLM 2026 preprint (2601.05167); Li et al. FutureBridge 2026 preprint (2608.06819); Fu et al. R2R NeurIPS 2025; Xu et al. Relay-OPD 2026 preprint (2607.26057)。不能把稀疏单token介入、动态长度或teacher偏好与student效用不同本身声称为新发现。
