# Anchor 逻辑分岔验证实验 Tracker

| Run ID | Milestone | Purpose | 数据/设置 | Metrics | Priority | Status |
|---|---|---|---|---|---|---|
| AB001 | M0 | 统计各 Δ 档可用的不同题目数 | existing Student 8 + bridge 8 | 每档 bridge/question 数、重复率、协变量分布 | MUST | TODO |
| AB002 | M0 | 冻结分层去重抽样 | 每档100题、每题至多1 bridge | balance、sampling weights、seed | MUST | TODO |
| AB003 | M1 | Judge pilot | 30 bridges、真实/placebo/S-S/B-B、A/B swap | JSON有效率、kappa、位置偏差 | MUST | TODO |
| AB004 | M2 | 独立复测提升 | 抽中 bridges 的 fresh Student 8 + Bridge 8 | confirmatory Δ、复现率、beta-binomial posterior | MUST | TODO |
| AB005 | M3 | 双 judge 正式判定 | 每 bridge 6–10 pairs，blind rubric | strict/loose、onset、direction、confidence | MUST | TODO |
| AB006 | M3 | 人工盲审 | 每档≥20 items，覆盖 controls | alpha/kappa、precision/recall | MUST | TODO |
| AB007 | M4 | 聚类感知统计 | GEE/mixed model + cluster bootstrap | Δ slope、F→S conditional slope、placebo gap | MUST | TODO |
| AB008 | M4 | 稳健性分析 | full-text 10%、prompt paraphrase、A/B swap | effect consistency | NICE | TODO |

