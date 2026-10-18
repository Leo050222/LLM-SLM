# 实验总表独立数值复核

复核日期：2026-10-19。对象：[ALL_EXPERIMENTS.md](../experiments/ALL_EXPERIMENTS.md) 的四组关键结果及生成脚本的证据状态逻辑。

**结论：93 项数值与一致性检查全部通过，未发现所核关键表与引用源 JSON 不一致。** 本次读取保存的结果并独立复算，没有重新运行 GPU 实验、生成 rollout 或重新评分。

| 范围 | 检查内容与源文件 |
| --- | --- |
| DeepMath thinking held-out 1000 | baseline、step27/53/80/106/133 的正确数、题数、准确率、相对 baseline 的百分点变化和平均输出 token；同时核对 `thinking=true` 与相同 `data_sha256`。源为 `results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_*/merged/summary.json`，并查看历史 `tmp/deepmath-heldout-1000/` 摘要。625/1000 → 706/1000 对应 62.5% → 70.6%，增加 8.10 个百分点。 |
| Nonthinking OPD 四模型 | 逐项核对 checkpoint、8000 次 rollout、正确数、平均准确率、pass@8、all@8、百分点和相对提升，并用正确数 / rollout 数复算准确率。源：[testset_model_comparison_20260908.json](../../figures/testset_model_comparison_20260908.json)。保留 2026-09-08 历史评分版本，与 thinking 或后来布尔归一化修订的分数分开解释。 |
| 主方法 greedy 八 benchmark | 核对 Student / Hybrid / Teacher、样本数、百分点与相对提升；独立将八个数据集的样本、正确数、介入、teacher token、输出 token 和 timeout 加总，与各方法 Overall 对照。源：[four_metrics.json](../../reports/main_greedy_comparison_20260921/four_metrics.json)。Overall 为按样本合并，4571 题；源报告规定 timeout 计错。 |
| 介入机制描述统计 | 核对两协议的事件数、涉及题数、student top5、高/低支持和平均熵差，使用 `pooled_event`，未混入 dataset macro。源：[temp1 headline](../../reports/intervention_analysis_phase1_20260921/headline.json)、[greedy headline](../../reports/intervention_analysis_greedy_20260921/headline.json)。 |

## 完成状态与解释边界

审阅 [build_experiment_catalog.py](../../tools/organization/build_experiment_catalog.py) 的状态生成逻辑与机器清单：未发现计划文件进入 `result_sources`。B3/B4 同前缀反事实与早期方法计划保持 `planned_only`；两项机制分析保持 `analysis_complete_semantic_and_counterfactual_partial`。阶段完成标记没有被解释为整个系列的全部配置完成。

两份机制 `ANALYSIS_COMPLETE.json` 均声明 `semantic_annotation_completed=false`；headline 中的 120 例仅已准备，尚未人工标注。支持度与熵差仍是描述统计，不构成逻辑分支正确性或因果机制识别的证据。八次采样也不等同于多个独立训练种子。

复核发现实验目录索引将下划线替换为连字符，而章节标题保留下划线，存在 GitHub 内部锚点不匹配；已通知目录生成代理修复。本报告记录发现及通知，不代替发布树的最终链接检查。

本结论限于上述关键表、引用摘要与状态逻辑；未逐条重验所有远端原始回答、checkpoint、logits、评分实现或 Git 历史。
