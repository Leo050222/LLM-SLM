# 全项目实验与结果总目录

证据快照：2026-10-19（Asia/Shanghai）。本文件由 `tools/organization/build_experiment_catalog.py` 从本地保存的 JSON、CSV 和阶段文档生成。

覆盖 `experiments/` 下 **99/99 个顶层目录**，另有 **13 个早期/跨目录系列**与 **106 个西湖运行目录**；扫描 59,154 个文件，索引 814 个汇总或文档产物，保留 17,368 条聚合记录与 311,013 个原始数值。

**存储范围：** 2026-10-19 已连接西湖旧共享存储 `gpu-wangxin:/storage/yukaichengLab/wangxin/test_eiloeo`。本次同步的小结果汇总和 manifest 位于 `results/westlake/`；大型 checkpoints、全量 rollouts 与 BF16 logits 继续保留在西湖。新 H800 PVC 的迁移状态与西湖原盘是两件事，不据此认定原始数据缺失。目录列出本地文件与同步摘要能够核实的实验，不能仅凭摘要声明每个远端原始文件都完成复验。

**口径：** `accuracy` 通常为 0–1，`accuracy_pct` 为 0–100；保留源字段名与单位。八轮采样的 seed/replicate 标准差不等于多次独立训练标准差。`pass@8`、`all@8` 与 rollout 平均准确率分别列示。共享报告、迁移/扩容目录和重评分目录可能指向同一批样本，不能相加为独立实验数。超时的最终答案正确与严格完成率也不能混用。

## 文件入口

- [experiment_series.csv](experiment_series.csv)：逐系列状态、配置入口、结果源和完成标记。
- [experiment_catalog.json](experiment_catalog.json)：机器清单、源 SHA256 与每系列关键数字；[key_results.csv](key_results.csv) 可直接打开。
- [result_records.csv.gz](result_records.csv.gz)：完整聚合结果表（gzip 压缩），保留原字段；`pointer` 定位 JSON 对象或 CSV 数据行（从 0 起）。
- [numeric_metrics.csv.gz](numeric_metrics.csv.gz)：汇总产物内数值长表（gzip 压缩），不复制原始回答或凭据。
- [result_artifacts.csv](result_artifacts.csv)、[coverage.json](coverage.json)：产物摘要与覆盖统计。

## 可以直接复核的主要结果

下面只对明确匹配的配置比较数值，不根据计划推断结果。

### 早期 DeepMath thinking held-out 1000 与 checkpoint

| 条件 | 正确/题数 | accuracy | 相对 baseline 变化（百分点） | 平均输出 tokens | 证据 |
|---|---:|---:|---:|---:|---|
| baseline | 625/1000 | 62.5000% | +0.00 | 10118.18 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811/merged/summary.json) |
| step27 | 650/1000 | 65.0000% | +2.50 | 13218.56 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step27_seed20260811/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step27_seed20260811/merged/summary.json) |
| step53 | 695/1000 | 69.5000% | +7.00 | 13261.73 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step53_seed20260811/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step53_seed20260811/merged/summary.json) |
| step80 | 701/1000 | 70.1000% | +7.60 | 12765.22 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step80_seed20260811/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step80_seed20260811/merged/summary.json) |
| step106 | 706/1000 | 70.6000% | +8.10 | 12687.35 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step106_seed20260811/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step106_seed20260811/merged/summary.json) |
| step133 | 674/1000 | 67.4000% | +4.90 | 12595.20 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step133_seed20260811/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step133_seed20260811/merged/summary.json) |

这是 thinking 协议；下表是 nonthinking 协议，两个表的准确率不可直接视为同协议训练收益。DeepMath 的历史评分后来进行了布尔答案归一化，下面 OPD 四模型表保留 2026-09-08 保存版本；不要与 2026-09-11 的修订分数交叉比较。

### Nonthinking OPD 四种模型（相同测试 1000 题，每题 8 次）

| 模型 | rollout 数 | 正确数 | 平均准确率 | pass@8 | all@8 | 相对 Student（百分点 / 相对比例） |
|---|---:|---:|---:|---:|---:|---:|
| Student-only（step 0） | 8000 | 2247 | 28.0875% | 50.80% | 7.30% | +0.000 / +0.00% |
| Traditional OPD（step 133） | 8000 | 2301 | 28.7625% | 53.70% | 8.60% | +0.675 / +2.40% |
| Bridge OPD（step 1656） | 8000 | 2382 | 29.7750% | 53.30% | 8.00% | +1.688 / +6.01% |
| Good-Bridge OPD（step 590） | 8000 | 2418 | 30.2250% | 54.60% | 7.60% | +2.138 / +7.61% |

数据源：[figures/testset_model_comparison_20260908.json](../../figures/testset_model_comparison_20260908.json)。这里报告保存的单模型 checkpoint 评测；未把评测重复当作独立训练种子。

### 主方法 dual-greedy k=1、threshold=2（八个 benchmark）

| 数据集 | n | Student | Hybrid | Teacher | Hybrid − Student（百分点） | 相对提升 |
|---|---:|---:|---:|---:|---:|---:|
| math500 | 500 | 72.000% | 82.400% | 83.600% | +10.400 | +14.44% |
| olympiad_bench | 674 | 41.988% | 52.967% | 59.496% | +10.979 | +26.15% |
| bbh | 1080 | 43.889% | 78.611% | 80.185% | +34.722 | +79.11% |
| cruxeval_i | 800 | 19.875% | 49.250% | 49.125% | +29.375 | +147.80% |
| cruxeval_o | 800 | 23.250% | 46.250% | 46.250% | +23.000 | +98.92% |
| humaneval_plus | 164 | 64.634% | 82.317% | 84.756% | +17.683 | +27.36% |
| mbpp_plus | 378 | 52.381% | 73.016% | 72.751% | +20.635 | +39.39% |
| livecodebench | 175 | 19.429% | 28.571% | 28.571% | +9.143 | +47.06% |
| Overall | 4571 | 39.379% | 62.196% | 63.706% | +22.818 | +57.94% |

数据源：[reports/main_greedy_comparison_20260921/four_metrics.json](../../reports/main_greedy_comparison_20260921/four_metrics.json)；严格 scoring 与 timeout 规则沿用源汇总。Teacher 强于 Student 的配对不代表每题均强于 Student。

### 八轮 grid 与消融结果的入口

| 系列 | 已保存结果与范围 | 源文件 |
|---|---|---|
| 数学/BBH 25 组 grid × 8 | MATH500、OlympiadBench、BBH；含 seed/replicate SD、pass@8、all@8、超时和 teacher token 占比 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) |
| 代码五 benchmark grid × 8 | CRUXEval-I/O、HumanEval+、MBPP+、LiveCodeBench；汇总评分 463,400 行（25 组 × 五数据集 × 八轮；baseline 另表） | [reports/code_grid_x8_results_20260918/x8_score_summary.json](../../reports/code_grid_x8_results_20260918/x8_score_summary.json) |
| prefix 接管预算 0–20 / 25–40 | 每数据集固定 sample100 × 8；与 teacher-only 配对比较 | [reports/prefix_caps0_20_analysis_20260918/summary.json](../../reports/prefix_caps0_20_analysis_20260918/summary.json)；[reports/prefix_caps0_40_analysis_20260918/metrics.json](../../reports/prefix_caps0_40_analysis_20260918/metrics.json) |
| 随机时机、shift5、reverse、weak reprompt/resample | 分开保留 greedy 与 temp1；详细准确率见逐系列结果 | [reports/random_quartile_comparison_20260918/summary.json](../../reports/random_quartile_comparison_20260918/summary.json) |
| Qwen3 1.7B / 4B / 8B / 14B | 八 benchmark matched greedy；AIME 中 1.7B/14B 使用历史 T=1，需单列协议 | [reports/qwen3_1p7b_4b_8b_14b_all_results_20261005.json](../../reports/qwen3_1p7b_4b_8b_14b_all_results_20261005.json) |

### 介入机制分析的已完成部分与边界

| 协议 | 事件数 | 涉及题数 | teacher token 落在 student top5 | 高支持 / 低支持 | 平均弱熵 − 强熵 |
|---|---:|---:|---:|---:|---:|
| temp1 × 8 | 399,601 | 4,549 | 76.923% | 57.265% / 3.906% | 0.084379 nats |
| greedy | 43,879 | 4,454 | 80.061% | 58.732% / 3.614% | 0.070485 nats |

源文件：[reports/intervention_analysis_phase1_20260921/headline.json](../../reports/intervention_analysis_phase1_20260921/headline.json)、[reports/intervention_analysis_greedy_20260921/headline.json](../../reports/intervention_analysis_greedy_20260921/headline.json)。这些是描述统计；120 例语义盲标复核尚未完成，B3/B4 同前缀反事实生成仍是计划。不能据此声称已证明“选中了正确逻辑分支”或完成因果机制识别。

## 全部顶层实验目录

每个目录列一个索引，包含部署、评分、扩容和复用阶段。状态“已有结果汇总”指至少找到数值证据；不保证该系列全部配置都完成。具体完成范围应以对应 `ALL_*_COMPLETE` / audit 标记和结果行数复核。没有找到汇总时，保持未确认，不用 README 的运行计划补齐结果。

| 编号 | 实验目录 | 证据状态 | 本地文件数 | 聚合记录数 |
|---:|---|---|---:|---:|
| 1 | [anchor_comparison_20260910](#anchor_comparison_20260910) | 已有结果汇总 | 12 | 14 |
| 2 | [baseline_math_gpt4o_mini_rescore_20260916](#baseline_math_gpt4o_mini_rescore_20260916) | 已有结果汇总 | 7 | 18 |
| 3 | [code_grid_x8_20260916](#code_grid_x8_20260916) | 已有结果汇总 | 13 | 2,900 |
| 4 | [code_grid_x8_score_20260918](#code_grid_x8_score_20260918) | 已有结果汇总 | 6 | 1,650 |
| 5 | [code_grid_x8_score_parallel_20260918](#code_grid_x8_score_parallel_20260918) | 已有结果汇总 | 4 | 1,645 |
| 6 | [code_reasoning_grid_20260913](#code_reasoning_grid_20260913) | 已有结果汇总 | 24 | 270 |
| 7 | [colocation_bench_20260918](#colocation_bench_20260918) | 已有结果汇总 | 11 | 4 |
| 8 | [cruxeval_i_fresh100_cap1_20260917](#cruxeval_i_fresh100_cap1_20260917) | 仅定位部署记录 | 20 | 0 |
| 9 | [dynamic_k_entropy_20260914](#dynamic_k_entropy_20260914) | 已有结果汇总 | 4,619 | 95 |
| 10 | [full_teacher_student_greedy_20260920](#full_teacher_student_greedy_20260920) | 已有结果汇总 | 24 | 126 |
| 11 | [full_teacher_student_greedy_8gpu_20260921](#full_teacher_student_greedy_8gpu_20260921) | 已有结果汇总 | 26 | 48 |
| 12 | [full_teacher_student_greedy_score_20260921](#full_teacher_student_greedy_score_20260921) | 已有结果汇总 | 11 | 48 |
| 13 | [full_teacher_student_x8_20260918](#full_teacher_student_x8_20260918) | 已有结果汇总 | 23 | 496 |
| 14 | [full_teacher_student_x8_8gpu_20260918](#full_teacher_student_x8_8gpu_20260918) | 已有结果汇总 | 23 | 336 |
| 15 | [grid_async_20260916](#grid_async_20260916) | 仅代码/状态文档 | 16 | 0 |
| 16 | [grid_async_retry_20260916](#grid_async_retry_20260916) | 仅代码/状态文档 | 17 | 0 |
| 17 | [grid_fifo_20260916](#grid_fifo_20260916) | 仅代码/状态文档 | 12 | 0 |
| 18 | [grid_math_4gpu_20260917](#grid_math_4gpu_20260917) | 仅代码/状态文档 | 11 | 0 |
| 19 | [grid_math_8gpu_20260917](#grid_math_8gpu_20260917) | 仅代码/状态文档 | 11 | 0 |
| 20 | [grid_math_colocated16_20260918](#grid_math_colocated16_20260918) | 已有结果汇总 | 14 | 810 |
| 21 | [grid_math_colocated22_20260918](#grid_math_colocated22_20260918) | 已有结果汇总 | 14 | 810 |
| 22 | [grid_math_colocated8_20260918](#grid_math_colocated8_20260918) | 已有结果汇总 | 12 | 810 |
| 23 | [grid_math_gpt4o_mini_rescore_20260916](#grid_math_gpt4o_mini_rescore_20260916) | 已有结果汇总 | 8 | 90 |
| 24 | [hard_k32_20260911](#hard_k32_20260911) | 已有结果汇总 | 3,648 | 8 |
| 25 | [hard_three_20260911](#hard_three_20260911) | 已有结果汇总 | 10,908 | 53 |
| 26 | [intervention_analysis_greedy_20260921](#intervention_analysis_greedy_20260921) | 数值分析完成；语义/反事实未完 | 11 | 609 |
| 27 | [intervention_analysis_phase1_20260921](#intervention_analysis_phase1_20260921) | 数值分析完成；语义/反事实未完 | 10 | 722 |
| 28 | [intervention_entropy_20260914](#intervention_entropy_20260914) | 已有结果汇总 | 15 | 1,480 |
| 29 | [intervention_logits_bf16_greedy_20260921](#intervention_logits_bf16_greedy_20260921) | 归档校验完成；大载荷保留西湖 | 11 | 3 |
| 30 | [intervention_logits_bf16_x8_20260920](#intervention_logits_bf16_x8_20260920) | 归档校验完成；大载荷保留西湖 | 11 | 3 |
| 31 | [k1_top1_agreement_20260914](#k1_top1_agreement_20260914) | 仅计划/代码证据 | 13 | 0 |
| 32 | [logits_replay_20260918](#logits_replay_20260918) | 仅代码/状态文档 | 3 | 0 |
| 33 | [logits_threshold2_replay_20260918](#logits_threshold2_replay_20260918) | 仅代码/状态文档 | 4 | 0 |
| 34 | [main_greedy_k1_t2_20260921](#main_greedy_k1_t2_20260921) | 已有结果汇总 | 15 | 95 |
| 35 | [main_greedy_score_20260921](#main_greedy_score_20260921) | 已有结果汇总 | 9 | 71 |
| 36 | [math500_three_20260911](#math500_three_20260911) | 已有结果汇总 | 3,515 | 9 |
| 37 | [math_grid_score_029041_20260919](#math_grid_score_029041_20260919) | 已有结果汇总 | 6 | 855 |
| 38 | [math_grid_score_all_2000_20260919](#math_grid_score_all_2000_20260919) | 已有结果汇总 | 10 | 1,335 |
| 39 | [math_reasoning_grid_x8_20260916](#math_reasoning_grid_x8_20260916) | 已有结果汇总 | 14 | 810 |
| 40 | [nojudge_grid_20260912](#nojudge_grid_20260912) | 已有结果汇总 | 2,868 | 576 |
| 41 | [online_comparison_20260911](#online_comparison_20260911) | 已有结果汇总 | 3 | 4 |
| 42 | [prefix_takeover_caps11_20_20260917](#prefix_takeover_caps11_20_20260917) | 已有结果汇总 | 25 | 240 |
| 43 | [prefix_takeover_caps25_40_20260918](#prefix_takeover_caps25_40_20260918) | 已有结果汇总 | 24 | 84 |
| 44 | [prefix_takeover_caps25_40_split_20260918](#prefix_takeover_caps25_40_split_20260918) | 仅定位部署记录 | 24 | 0 |
| 45 | [prefix_takeover_caps5_10_20260917](#prefix_takeover_caps5_10_20260917) | 已有结果汇总 | 22 | 192 |
| 46 | [prefix_takeover_caps_20260917](#prefix_takeover_caps_20260917) | 已有结果汇总 | 22 | 358 |
| 47 | [quartile_greedy_baselines_20260918](#quartile_greedy_baselines_20260918) | 已有结果汇总 | 20 | 1,040 |
| 48 | [quartile_support_k_20260918](#quartile_support_k_20260918) | 已有结果汇总 | 21 | 2,432 |
| 49 | [qwen3_greedy_hard_three_20261005](#qwen3_greedy_hard_three_20261005) | 已有结果汇总 | 14 | 30 |
| 50 | [qwen3_greedy_remaining_20261005](#qwen3_greedy_remaining_20261005) | 已有结果汇总 | 16 | 45 |
| 51 | [qwen3vl_greedy_bench_20261016](#qwen3vl_greedy_bench_20261016) | 仅代码/状态文档 | 7 | 0 |
| 52 | [r1_math_score_20260918](#r1_math_score_20260918) | 已有结果汇总 | 6 | 3 |
| 53 | [random_k8_max8_20260911](#random_k8_max8_20260911) | 已有结果汇总 | 4,649 | 15 |
| 54 | [random_quartile_greedy_k1_20260921](#random_quartile_greedy_k1_20260921) | 已有结果汇总 | 29 | 64 |
| 55 | [random_quartile_k1_20260918](#random_quartile_k1_20260918) | 已有结果汇总 | 28 | 64 |
| 56 | [reverse_greedy_score_20260921](#reverse_greedy_score_20260921) | 已有结果汇总 | 12 | 8 |
| 57 | [reverse_greedy_t2_k1_8gpu_20260921](#reverse_greedy_t2_k1_8gpu_20260921) | 已有结果汇总 | 15 | 24 |
| 58 | [reverse_t2_k1_20260918](#reverse_t2_k1_20260918) | 已有结果汇总 | 14 | 45 |
| 59 | [reverse_t2_k1_score_20260918](#reverse_t2_k1_score_20260918) | 已有结果汇总 | 11 | 24 |
| 60 | [shift5_greedy_k1_t2_20260921](#shift5_greedy_k1_t2_20260921) | 已有结果汇总 | 20 | 41 |
| 61 | [shift5_greedy_score_20260921](#shift5_greedy_score_20260921) | 已有结果汇总 | 10 | 25 |
| 62 | [shift5_k1_t2_20260918](#shift5_k1_t2_20260918) | 已有结果汇总 | 27 | 33 |
| 63 | [signed_gap_audit_20260911](#signed_gap_audit_20260911) | 仅代码/状态文档 | 1 | 0 |
| 64 | [strong4_weak17_reprompt_greedy_t2_k1_20261005](#strong4_weak17_reprompt_greedy_t2_k1_20261005) | 已有结果汇总 | 19 | 68 |
| 65 | [strong8_weak17_reprompt_greedy_t2_k1_20261005](#strong8_weak17_reprompt_greedy_t2_k1_20261005) | 已有结果汇总 | 19 | 68 |
| 66 | [strong_reprompt_score_20261005](#strong_reprompt_score_20261005) | 已有结果汇总 | 12 | 80 |
| 67 | [student_only64_20260910](#student_only64_20260910) | 已有结果汇总 | 9 | 2 |
| 68 | [takeover_ablation_20260911](#takeover_ablation_20260911) | 仅代码/状态文档 | 14 | 0 |
| 69 | [teacher8_student8_20260909](#teacher8_student8_20260909) | 已有结果汇总 | 17 | 28 |
| 70 | [teacher_only_sample100_x8_20260917](#teacher_only_sample100_x8_20260917) | 已有结果汇总 | 24 | 16 |
| 71 | [teacher_prefix_handoff_20260909](#teacher_prefix_handoff_20260909) | 已有结果汇总 | 10 | 8 |
| 72 | [threshold2_luna_k1_20260911](#threshold2_luna_k1_20260911) | 仅代码/状态文档 | 8 | 0 |
| 73 | [threshold3_mathteacher_20260911](#threshold3_mathteacher_20260911) | 已有结果汇总 | 9,161 | 12 |
| 74 | [threshold_k8_nojudge_20260911](#threshold_k8_nojudge_20260911) | 已有结果汇总 | 4,650 | 15 |
| 75 | [token_entropy_distribution_20260914](#token_entropy_distribution_20260914) | 已有结果汇总 | 19 | 55 |
| 76 | [weak1p7_strong4_reprompt_greedy_t2_k1_20261005](#weak1p7_strong4_reprompt_greedy_t2_k1_20261005) | 仅定位部署记录 | 20 | 0 |
| 77 | [weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005](#weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005) | 仅定位部署记录 | 20 | 0 |
| 78 | [weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005](#weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005) | 仅定位部署记录 | 20 | 0 |
| 79 | [weak1p7_strong8_reprompt_greedy_t2_k1_20261005](#weak1p7_strong8_reprompt_greedy_t2_k1_20261005) | 仅定位部署记录 | 20 | 0 |
| 80 | [weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005](#weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005) | 仅定位部署记录 | 20 | 0 |
| 81 | [weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005](#weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005) | 仅定位部署记录 | 20 | 0 |
| 82 | [weak_reprompt_16gpu_2000s_20260920](#weak_reprompt_16gpu_2000s_20260920) | 仅定位部署记录 | 19 | 0 |
| 83 | [weak_reprompt_8gpu_2000s_20260920](#weak_reprompt_8gpu_2000s_20260920) | 仅定位部署记录 | 19 | 0 |
| 84 | [weak_reprompt_clean_greedy_score_20260921](#weak_reprompt_clean_greedy_score_20260921) | 已有结果汇总 | 9 | 8 |
| 85 | [weak_reprompt_clean_greedy_t2_k1_20260921](#weak_reprompt_clean_greedy_t2_k1_20260921) | 已有结果汇总 | 20 | 24 |
| 86 | [weak_reprompt_clean_score_20260921](#weak_reprompt_clean_score_20260921) | 已有结果汇总 | 10 | 8 |
| 87 | [weak_reprompt_clean_t2_k1_20260921](#weak_reprompt_clean_t2_k1_20260921) | 已有结果汇总 | 23 | 24 |
| 88 | [weak_reprompt_nodraft_greedy_score_20260921](#weak_reprompt_nodraft_greedy_score_20260921) | 已有结果汇总 | 9 | 8 |
| 89 | [weak_reprompt_nodraft_greedy_t2_k1_20260921](#weak_reprompt_nodraft_greedy_t2_k1_20260921) | 已有结果汇总 | 19 | 24 |
| 90 | [weak_reprompt_nodraft_score_20260920](#weak_reprompt_nodraft_score_20260920) | 已有结果汇总 | 10 | 34 |
| 91 | [weak_reprompt_nodraft_t2_k1_20260920](#weak_reprompt_nodraft_t2_k1_20260920) | 已有结果汇总 | 19 | 50 |
| 92 | [weak_reprompt_score_20260921](#weak_reprompt_score_20260921) | 仅定位部署记录 | 10 | 0 |
| 93 | [weak_reprompt_t2_k1_20260920](#weak_reprompt_t2_k1_20260920) | 已有结果汇总 | 20 | 16 |
| 94 | [weak_resample_greedy_8gpu_20260921](#weak_resample_greedy_8gpu_20260921) | 已有结果汇总 | 17 | 8 |
| 95 | [weak_resample_greedy_score_20260921](#weak_resample_greedy_score_20260921) | 已有结果汇总 | 9 | 8 |
| 96 | [weak_resample_greedy_t2_k1_20260921](#weak_resample_greedy_t2_k1_20260921) | 已有结果汇总 | 19 | 24 |
| 97 | [weak_resample_score_20260921](#weak_resample_score_20260921) | 已有结果汇总 | 10 | 8 |
| 98 | [weak_resample_t2_k1_20260920](#weak_resample_t2_k1_20260920) | 已有结果汇总 | 19 | 24 |
| 99 | [weak_to_8b_prefix_handoff_20261005](#weak_to_8b_prefix_handoff_20261005) | 仅代码/状态文档 | 11 | 0 |

### anchor_comparison_20260910

- 证据状态：**已有结果汇总**；代码目录：[experiments/anchor_comparison_20260910](../../experiments/anchor_comparison_20260910)。
- 结果入口：[experiments/anchor_comparison_20260910/comparison.json](../../experiments/anchor_comparison_20260910/comparison.json)、[experiments/anchor_comparison_20260910/anchor/summary.json](../../experiments/anchor_comparison_20260910/anchor/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| method=student64；questions=958，correct=24192，accuracy=0.394572，pass_any=0.873695 | [experiments/anchor_comparison_20260910/comparison.json](../../experiments/anchor_comparison_20260910/comparison.json) `/same_subset/student64` |
| method=teacher；questions=958，correct=4526，accuracy=0.590553，pass_any=0.86952 | [experiments/anchor_comparison_20260910/comparison.json](../../experiments/anchor_comparison_20260910/comparison.json) `/same_subset/teacher` |
| method=h32；questions=958，correct=24421，accuracy=0.398307，pass_any=0.879958 | [experiments/anchor_comparison_20260910/comparison.json](../../experiments/anchor_comparison_20260910/comparison.json) `/same_subset/h32` |
| method=h64；questions=958，correct=24396，accuracy=0.397899，pass_any=0.888309 | [experiments/anchor_comparison_20260910/comparison.json](../../experiments/anchor_comparison_20260910/comparison.json) `/same_subset/h64` |
| method=h128；questions=958，correct=24649，accuracy=0.402026，pass_any=0.884134 | [experiments/anchor_comparison_20260910/comparison.json](../../experiments/anchor_comparison_20260910/comparison.json) `/same_subset/h128` |
| method=h256；questions=958，correct=25614，accuracy=0.417765，pass_any=0.890397 | [experiments/anchor_comparison_20260910/comparison.json](../../experiments/anchor_comparison_20260910/comparison.json) `/same_subset/h256` |
| questions=958，correct=26569，accuracy=0.433341，pass_any=0.874739 | [experiments/anchor_comparison_20260910/comparison.json](../../experiments/anchor_comparison_20260910/comparison.json) `/same_subset/anchor` |
| method=student64；questions=1000，correct=25961，accuracy=0.405641，pass_any=0.876 | [experiments/anchor_comparison_20260910/comparison.json](../../experiments/anchor_comparison_20260910/comparison.json) `/full1000/student64` |

本源另有 5 条聚合记录；完整数值见 `result_records.csv.gz`。

### baseline_math_gpt4o_mini_rescore_20260916

- 证据状态：**已有结果汇总**；代码目录：[experiments/baseline_math_gpt4o_mini_rescore_20260916](../../experiments/baseline_math_gpt4o_mini_rescore_20260916)。
- 配置/范围入口：[experiments/baseline_math_gpt4o_mini_rescore_20260916/submission.json](../../experiments/baseline_math_gpt4o_mini_rescore_20260916/submission.json)。
- 已保存阶段完成标记：[reports/baseline_math_rescore_20260916/COMPLETE.json](../../reports/baseline_math_rescore_20260916/COMPLETE.json)、[reports/baseline_math_rescore_20260916/INPUTS_COMPLETE.json](../../reports/baseline_math_rescore_20260916/INPUTS_COMPLETE.json)。
- 结果入口：[reports/baseline_math_rescore_20260916/summary.json](../../reports/baseline_math_rescore_20260916/summary.json)、[reports/baseline_math_rescore_20260916/baseline_comparison.json](../../reports/baseline_math_rescore_20260916/baseline_comparison.json)、[reports/baseline_math_rescore_20260916/baseline_comparison.csv](../../reports/baseline_math_rescore_20260916/baseline_comparison.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| config=student_only / dataset=bbh；correct=465，accuracy=0.430556 | [reports/baseline_math_rescore_20260916/summary.json](../../reports/baseline_math_rescore_20260916/summary.json) `/groups/0` |
| config=teacher_only / dataset=bbh；correct=862，accuracy=0.798148 | [reports/baseline_math_rescore_20260916/summary.json](../../reports/baseline_math_rescore_20260916/summary.json) `/groups/1` |
| config=student_only / dataset=math500；correct=358，accuracy=0.716 | [reports/baseline_math_rescore_20260916/summary.json](../../reports/baseline_math_rescore_20260916/summary.json) `/groups/2` |
| config=teacher_only / dataset=math500；correct=423，accuracy=0.846 | [reports/baseline_math_rescore_20260916/summary.json](../../reports/baseline_math_rescore_20260916/summary.json) `/groups/3` |
| config=student_only / dataset=olympiad_bench；correct=281，accuracy=0.416914 | [reports/baseline_math_rescore_20260916/summary.json](../../reports/baseline_math_rescore_20260916/summary.json) `/groups/4` |
| config=teacher_only / dataset=olympiad_bench；correct=371，accuracy=0.550445 | [reports/baseline_math_rescore_20260916/summary.json](../../reports/baseline_math_rescore_20260916/summary.json) `/groups/5` |

### code_grid_x8_20260916

- 证据状态：**已有结果汇总**；代码目录：[experiments/code_grid_x8_20260916](../../experiments/code_grid_x8_20260916)。
- 配置/范围入口：[experiments/code_grid_x8_20260916/LAUNCH.md](../../experiments/code_grid_x8_20260916/LAUNCH.md)、[experiments/code_grid_x8_20260916/PLAN.json](../../experiments/code_grid_x8_20260916/PLAN.json)、[experiments/code_grid_x8_20260916/configs.json](../../experiments/code_grid_x8_20260916/configs.json)、[experiments/code_grid_x8_20260916/submission.json](../../experiments/code_grid_x8_20260916/submission.json)。
- 已保存阶段完成标记：[reports/code_grid_x8_results_20260918/ALL_X8_SCORING_COMPLETE.json](../../reports/code_grid_x8_results_20260918/ALL_X8_SCORING_COMPLETE.json)。
- 结果入口：[reports/code_grid_x8_scoring_audit_20260918/audit.json](../../reports/code_grid_x8_scoring_audit_20260918/audit.json)、[reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json)、[reports/code_grid_x8_results_20260918/x8_score_summary.json](../../reports/code_grid_x8_results_20260918/x8_score_summary.json)、[results/westlake/opd_eval_code/code_grid_x8_20260916/x8_score_summary.json](../../results/westlake/opd_eval_code/code_grid_x8_20260916/x8_score_summary.json)、[reports/code_grid_x8_results_20260918/pass8_table.csv](../../reports/code_grid_x8_results_20260918/pass8_table.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| config=k16_t2 / dataset=cruxeval_o；questions=800，pass_at_8=0.5225 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/0` |
| config=k16_t2 / dataset=cruxeval_i；questions=800，pass_at_8=0.6025 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/1` |
| config=k16_t2 / dataset=humaneval_plus；questions=164，pass_at_8=0.908537 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/2` |
| config=k16_t2 / dataset=mbpp_plus；questions=378，pass_at_8=0.796296 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/3` |
| config=k16_t2 / dataset=livecodebench；questions=175，pass_at_8=0.382857 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/4` |
| config=k16_t2.5 / dataset=cruxeval_o；questions=800，pass_at_8=0.52125 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/5` |
| config=k16_t2.5 / dataset=cruxeval_i；questions=800，pass_at_8=0.59875 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/6` |
| config=k16_t2.5 / dataset=humaneval_plus；questions=164，pass_at_8=0.914634 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/7` |

本源另有 117 条聚合记录；完整数值见 `result_records.csv.gz`。

### code_grid_x8_score_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/code_grid_x8_score_20260918](../../experiments/code_grid_x8_score_20260918)。
- 配置/范围入口：[experiments/code_grid_x8_score_20260918/LAUNCH.json](../../experiments/code_grid_x8_score_20260918/LAUNCH.json)、[experiments/code_grid_x8_score_20260918/README.md](../../experiments/code_grid_x8_score_20260918/README.md)。
- 已保存阶段完成标记：[reports/code_grid_x8_results_20260918/ALL_X8_SCORING_COMPLETE.json](../../reports/code_grid_x8_results_20260918/ALL_X8_SCORING_COMPLETE.json)。
- 结果入口：[reports/code_grid_x8_scoring_audit_20260918/audit.json](../../reports/code_grid_x8_scoring_audit_20260918/audit.json)、[reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json)、[reports/code_grid_x8_results_20260918/x8_score_summary.json](../../reports/code_grid_x8_results_20260918/x8_score_summary.json)、[reports/code_grid_x8_results_20260918/pass8_table.csv](../../reports/code_grid_x8_results_20260918/pass8_table.csv)、[reports/code_grid_x8_results_20260918/x8_score_summary.csv](../../reports/code_grid_x8_results_20260918/x8_score_summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| config=k16_t2 / dataset=cruxeval_o；questions=800，pass_at_8=0.5225 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/0` |
| config=k16_t2 / dataset=cruxeval_i；questions=800，pass_at_8=0.6025 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/1` |
| config=k16_t2 / dataset=humaneval_plus；questions=164，pass_at_8=0.908537 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/2` |
| config=k16_t2 / dataset=mbpp_plus；questions=378，pass_at_8=0.796296 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/3` |
| config=k16_t2 / dataset=livecodebench；questions=175，pass_at_8=0.382857 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/4` |
| config=k16_t2.5 / dataset=cruxeval_o；questions=800，pass_at_8=0.52125 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/5` |
| config=k16_t2.5 / dataset=cruxeval_i；questions=800，pass_at_8=0.59875 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/6` |
| config=k16_t2.5 / dataset=humaneval_plus；questions=164，pass_at_8=0.914634 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/7` |

本源另有 117 条聚合记录；完整数值见 `result_records.csv.gz`。

### code_grid_x8_score_parallel_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/code_grid_x8_score_parallel_20260918](../../experiments/code_grid_x8_score_parallel_20260918)。
- 配置/范围入口：[experiments/code_grid_x8_score_parallel_20260918/LAUNCH.json](../../experiments/code_grid_x8_score_parallel_20260918/LAUNCH.json)、[experiments/code_grid_x8_score_parallel_20260918/README.md](../../experiments/code_grid_x8_score_parallel_20260918/README.md)。
- 已保存阶段完成标记：[reports/code_grid_x8_results_20260918/ALL_X8_SCORING_COMPLETE.json](../../reports/code_grid_x8_results_20260918/ALL_X8_SCORING_COMPLETE.json)。
- 结果入口：[reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json)、[reports/code_grid_x8_results_20260918/x8_score_summary.json](../../reports/code_grid_x8_results_20260918/x8_score_summary.json)、[reports/code_grid_x8_results_20260918/pass8_table.csv](../../reports/code_grid_x8_results_20260918/pass8_table.csv)、[reports/code_grid_x8_results_20260918/x8_score_summary.csv](../../reports/code_grid_x8_results_20260918/x8_score_summary.csv)、[reports/code_grid_x8_results_20260918/four_metrics_table.csv](../../reports/code_grid_x8_results_20260918/four_metrics_table.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| config=k16_t2 / dataset=cruxeval_o；questions=800，pass_at_8=0.5225 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/0` |
| config=k16_t2 / dataset=cruxeval_i；questions=800，pass_at_8=0.6025 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/1` |
| config=k16_t2 / dataset=humaneval_plus；questions=164，pass_at_8=0.908537 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/2` |
| config=k16_t2 / dataset=mbpp_plus；questions=378，pass_at_8=0.796296 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/3` |
| config=k16_t2 / dataset=livecodebench；questions=175，pass_at_8=0.382857 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/4` |
| config=k16_t2.5 / dataset=cruxeval_o；questions=800，pass_at_8=0.52125 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/5` |
| config=k16_t2.5 / dataset=cruxeval_i；questions=800，pass_at_8=0.59875 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/6` |
| config=k16_t2.5 / dataset=humaneval_plus；questions=164，pass_at_8=0.914634 | [reports/code_grid_x8_results_20260918/pass8_results.json](../../reports/code_grid_x8_results_20260918/pass8_results.json) `/results/7` |

本源另有 117 条聚合记录；完整数值见 `result_records.csv.gz`。

### code_reasoning_grid_20260913

- 证据状态：**已有结果汇总**；代码目录：[experiments/code_reasoning_grid_20260913](../../experiments/code_reasoning_grid_20260913)。
- 配置/范围入口：[experiments/code_reasoning_grid_20260913/LAUNCH.md](../../experiments/code_reasoning_grid_20260913/LAUNCH.md)。
- 结果入口：[experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv](../../experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv)、[results/westlake/opd_eval_code/code_reasoning_grid_20260913/FINAL_COMPARISON.csv](../../results/westlake/opd_eval_code/code_reasoning_grid_20260913/FINAL_COMPARISON.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| config=student_only / dataset=cruxeval_o；n=800，correct=175，acc=0.21875，mean_output_tokens=402.012，mean_interventions=0，teacher_token_fraction=0 | [experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv](../../experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv) `/rows/0` |
| config=student_only / dataset=cruxeval_i；n=800，correct=158，acc=0.1975，mean_output_tokens=349.055，mean_interventions=0，teacher_token_fraction=0 | [experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv](../../experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv) `/rows/1` |
| config=student_only / dataset=humaneval_plus；n=164，correct=103，acc=0.628049，mean_output_tokens=199.183，mean_interventions=0，teacher_token_fraction=0 | [experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv](../../experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv) `/rows/2` |
| config=student_only / dataset=mbpp_plus；n=378，correct=196，acc=0.518519，mean_output_tokens=139.053，mean_interventions=0，teacher_token_fraction=0 | [experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv](../../experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv) `/rows/3` |
| config=student_only / dataset=livecodebench；n=175，correct=31，acc=0.177143，mean_output_tokens=618.223，mean_interventions=0，teacher_token_fraction=0 | [experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv](../../experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv) `/rows/4` |
| config=teacher_only / dataset=cruxeval_o；n=800，correct=374，acc=0.4675，mean_output_tokens=115.116，mean_interventions=0，teacher_token_fraction=1 | [experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv](../../experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv) `/rows/5` |
| config=teacher_only / dataset=cruxeval_i；n=800，correct=406，acc=0.5075，mean_output_tokens=18.235，mean_interventions=0，teacher_token_fraction=1 | [experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv](../../experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv) `/rows/6` |
| config=teacher_only / dataset=humaneval_plus；n=164，correct=143，acc=0.871951，mean_output_tokens=177.634，mean_interventions=0，teacher_token_fraction=1 | [experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv](../../experiments/code_reasoning_grid_20260913/reports/FINAL_COMPARISON.csv) `/rows/7` |

本源另有 127 条聚合记录；完整数值见 `result_records.csv.gz`。

### colocation_bench_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/colocation_bench_20260918](../../experiments/colocation_bench_20260918)。
- 配置/范围入口：[experiments/colocation_bench_20260918/LAUNCH.json](../../experiments/colocation_bench_20260918/LAUNCH.json)、[experiments/colocation_bench_20260918/README.md](../../experiments/colocation_bench_20260918/README.md)。
- 结果入口：[reports/colocation_bench_20260918/results.json](../../reports/colocation_bench_20260918/results.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| n=54 | [reports/colocation_bench_20260918/results.json](../../reports/colocation_bench_20260918/results.json) `/metrics/split8` |
| n=54 | [reports/colocation_bench_20260918/results.json](../../reports/colocation_bench_20260918/results.json) `/metrics/split4` |
| n=54 | [reports/colocation_bench_20260918/results.json](../../reports/colocation_bench_20260918/results.json) `/metrics/colocated4` |
| n=54 | [reports/colocation_bench_20260918/results.json](../../reports/colocation_bench_20260918/results.json) `/metrics/split8_repeat` |

### cruxeval_i_fresh100_cap1_20260917

- 证据状态：**仅定位部署记录**；代码目录：[experiments/cruxeval_i_fresh100_cap1_20260917](../../experiments/cruxeval_i_fresh100_cap1_20260917)。
- 配置/范围入口：[experiments/cruxeval_i_fresh100_cap1_20260917/LAUNCH.json](../../experiments/cruxeval_i_fresh100_cap1_20260917/LAUNCH.json)、[experiments/cruxeval_i_fresh100_cap1_20260917/README.md](../../experiments/cruxeval_i_fresh100_cap1_20260917/README.md)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### dynamic_k_entropy_20260914

- 证据状态：**已有结果汇总**；代码目录：[experiments/dynamic_k_entropy_20260914](../../experiments/dynamic_k_entropy_20260914)。
- 配置/范围入口：[experiments/dynamic_k_entropy_20260914/PLAN.md](../../experiments/dynamic_k_entropy_20260914/PLAN.md)。
- 已保存阶段完成标记：[experiments/dynamic_k_entropy_20260914/results/ALL_GENERATION_COMPLETE.json](../../experiments/dynamic_k_entropy_20260914/results/ALL_GENERATION_COMPLETE.json)、[experiments/dynamic_k_entropy_20260914/results/SCORING_COMPLETE.json](../../experiments/dynamic_k_entropy_20260914/results/SCORING_COMPLETE.json)、[experiments/dynamic_k_entropy_20260914/results/code/VERIFIED.json](../../experiments/dynamic_k_entropy_20260914/results/code/VERIFIED.json)、[experiments/dynamic_k_entropy_20260914/results/math/VERIFIED.json](../../experiments/dynamic_k_entropy_20260914/results/math/VERIFIED.json)。
- 结果入口：[experiments/dynamic_k_entropy_20260914/results/summary.json](../../experiments/dynamic_k_entropy_20260914/results/summary.json)、[results/westlake/opd_eval_dynamic/dynamic_k_entropy_20260914/summary.json](../../results/westlake/opd_eval_dynamic/dynamic_k_entropy_20260914/summary.json)、[experiments/dynamic_k_entropy_20260914/results/extension_vs_k1/summary.json](../../experiments/dynamic_k_entropy_20260914/results/extension_vs_k1/summary.json)、[experiments/dynamic_k_entropy_20260914/results/FINAL_AUDIT.json](../../experiments/dynamic_k_entropy_20260914/results/FINAL_AUDIT.json)、[experiments/dynamic_k_entropy_20260914/results/summary.csv](../../experiments/dynamic_k_entropy_20260914/results/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=834，acc=0.772222，mean_output_tokens=263.095，mean_interventions=17.4907，teacher_token_fraction=0.0714711 | [experiments/dynamic_k_entropy_20260914/results/summary.json](../../experiments/dynamic_k_entropy_20260914/results/summary.json) `/0` |
| dataset=cruxeval_i；n=800，correct=392，acc=0.49，mean_output_tokens=18.345，mean_interventions=3.785，teacher_token_fraction=0.227719 | [experiments/dynamic_k_entropy_20260914/results/summary.json](../../experiments/dynamic_k_entropy_20260914/results/summary.json) `/1` |
| dataset=cruxeval_o；n=800，correct=362，acc=0.4525，mean_output_tokens=91.3438，mean_interventions=2.3125，teacher_token_fraction=0.0292576 | [experiments/dynamic_k_entropy_20260914/results/summary.json](../../experiments/dynamic_k_entropy_20260914/results/summary.json) `/2` |
| dataset=humaneval_plus；n=164，correct=135，acc=0.823171，mean_output_tokens=181.482，mean_interventions=6.21341，teacher_token_fraction=0.0366563 | [experiments/dynamic_k_entropy_20260914/results/summary.json](../../experiments/dynamic_k_entropy_20260914/results/summary.json) `/3` |
| dataset=livecodebench；n=175，correct=48，acc=0.274286，mean_output_tokens=590.64，mean_interventions=29.0971，teacher_token_fraction=0.0509085 | [experiments/dynamic_k_entropy_20260914/results/summary.json](../../experiments/dynamic_k_entropy_20260914/results/summary.json) `/4` |
| dataset=math500；n=500，correct=431，acc=0.862，mean_output_tokens=817.814，mean_interventions=9.438，teacher_token_fraction=0.011768 | [experiments/dynamic_k_entropy_20260914/results/summary.json](../../experiments/dynamic_k_entropy_20260914/results/summary.json) `/5` |
| dataset=mbpp_plus；n=378，correct=270，acc=0.714286，mean_output_tokens=91.8836，mean_interventions=5.90741，teacher_token_fraction=0.0697628 | [experiments/dynamic_k_entropy_20260914/results/summary.json](../../experiments/dynamic_k_entropy_20260914/results/summary.json) `/6` |
| dataset=olympiad_bench；n=674，correct=296，acc=0.439169，mean_output_tokens=1343.2，mean_interventions=15.908，teacher_token_fraction=0.0120621 | [experiments/dynamic_k_entropy_20260914/results/summary.json](../../experiments/dynamic_k_entropy_20260914/results/summary.json) `/7` |

### full_teacher_student_greedy_20260920

- 证据状态：**已有结果汇总**；代码目录：[experiments/full_teacher_student_greedy_20260920](../../experiments/full_teacher_student_greedy_20260920)。
- 配置/范围入口：[experiments/full_teacher_student_greedy_20260920/LAUNCH.json](../../experiments/full_teacher_student_greedy_20260920/LAUNCH.json)、[experiments/full_teacher_student_greedy_20260920/README.md](../../experiments/full_teacher_student_greedy_20260920/README.md)。
- 已保存阶段完成标记：[reports/greedy_vs_temp1_20260921/greedy/ALL_SCORING_COMPLETE.json](../../reports/greedy_vs_temp1_20260921/greedy/ALL_SCORING_COMPLETE.json)、[reports/greedy_vs_temp1_20260921/greedy/INPUTS_COMPLETE.json](../../reports/greedy_vs_temp1_20260921/greedy/INPUTS_COMPLETE.json)。
- 结果入口：[reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json)、[results/westlake/opd_eval_baselines/full_teacher_student_greedy_20260920/summary.json](../../results/westlake/opd_eval_baselines/full_teacher_student_greedy_20260920/summary.json)、[results/westlake/opd_eval_baselines/full_teacher_student_greedy_20260920/code_summary.json](../../results/westlake/opd_eval_baselines/full_teacher_student_greedy_20260920/code_summary.json)、[reports/greedy_vs_temp1_20260921/greedy/summary.csv](../../reports/greedy_vs_temp1_20260921/greedy/summary.csv)、[results/westlake/opd_eval_baselines/full_teacher_student_greedy_20260920/summary.csv](../../results/westlake/opd_eval_baselines/full_teacher_student_greedy_20260920/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh / replicate=0 / role=student；n=1080，correct=474，accuracy=0.438889 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/0` |
| dataset=bbh / replicate=0 / role=teacher；n=1080，correct=866，accuracy=0.801852 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/1` |
| dataset=cruxeval_i / replicate=0 / role=student；n=800，correct=159，accuracy=0.19875 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/2` |
| dataset=cruxeval_i / replicate=0 / role=teacher；n=800，correct=393，accuracy=0.49125 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/3` |
| dataset=cruxeval_o / replicate=0 / role=student；n=800，correct=186，accuracy=0.2325 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/4` |
| dataset=cruxeval_o / replicate=0 / role=teacher；n=800，correct=370，accuracy=0.4625 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/5` |
| dataset=humaneval_plus / replicate=0 / role=student；n=164，correct=106，accuracy=0.646341 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/6` |
| dataset=humaneval_plus / replicate=0 / role=teacher；n=164，correct=139，accuracy=0.847561 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/7` |

本源另有 24 条聚合记录；完整数值见 `result_records.csv.gz`。

### full_teacher_student_greedy_8gpu_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/full_teacher_student_greedy_8gpu_20260921](../../experiments/full_teacher_student_greedy_8gpu_20260921)。
- 配置/范围入口：[experiments/full_teacher_student_greedy_8gpu_20260921/LAUNCH.json](../../experiments/full_teacher_student_greedy_8gpu_20260921/LAUNCH.json)、[experiments/full_teacher_student_greedy_8gpu_20260921/README.md](../../experiments/full_teacher_student_greedy_8gpu_20260921/README.md)。
- 已保存阶段完成标记：[reports/greedy_vs_temp1_20260921/greedy/ALL_SCORING_COMPLETE.json](../../reports/greedy_vs_temp1_20260921/greedy/ALL_SCORING_COMPLETE.json)、[reports/greedy_vs_temp1_20260921/greedy/INPUTS_COMPLETE.json](../../reports/greedy_vs_temp1_20260921/greedy/INPUTS_COMPLETE.json)。
- 结果入口：[reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json)、[reports/greedy_vs_temp1_20260921/greedy/summary.csv](../../reports/greedy_vs_temp1_20260921/greedy/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh / replicate=0 / role=student；n=1080，correct=474，accuracy=0.438889 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/0` |
| dataset=bbh / replicate=0 / role=teacher；n=1080，correct=866，accuracy=0.801852 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/1` |
| dataset=cruxeval_i / replicate=0 / role=student；n=800，correct=159，accuracy=0.19875 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/2` |
| dataset=cruxeval_i / replicate=0 / role=teacher；n=800，correct=393，accuracy=0.49125 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/3` |
| dataset=cruxeval_o / replicate=0 / role=student；n=800，correct=186，accuracy=0.2325 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/4` |
| dataset=cruxeval_o / replicate=0 / role=teacher；n=800，correct=370，accuracy=0.4625 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/5` |
| dataset=humaneval_plus / replicate=0 / role=student；n=164，correct=106，accuracy=0.646341 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/6` |
| dataset=humaneval_plus / replicate=0 / role=teacher；n=164，correct=139，accuracy=0.847561 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/7` |

本源另有 24 条聚合记录；完整数值见 `result_records.csv.gz`。

### full_teacher_student_greedy_score_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/full_teacher_student_greedy_score_20260921](../../experiments/full_teacher_student_greedy_score_20260921)。
- 配置/范围入口：[experiments/full_teacher_student_greedy_score_20260921/LAUNCH.json](../../experiments/full_teacher_student_greedy_score_20260921/LAUNCH.json)、[experiments/full_teacher_student_greedy_score_20260921/README.md](../../experiments/full_teacher_student_greedy_score_20260921/README.md)。
- 已保存阶段完成标记：[reports/greedy_vs_temp1_20260921/greedy/ALL_SCORING_COMPLETE.json](../../reports/greedy_vs_temp1_20260921/greedy/ALL_SCORING_COMPLETE.json)、[reports/greedy_vs_temp1_20260921/greedy/INPUTS_COMPLETE.json](../../reports/greedy_vs_temp1_20260921/greedy/INPUTS_COMPLETE.json)。
- 结果入口：[reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json)、[reports/greedy_vs_temp1_20260921/greedy/summary.csv](../../reports/greedy_vs_temp1_20260921/greedy/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh / replicate=0 / role=student；n=1080，correct=474，accuracy=0.438889 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/0` |
| dataset=bbh / replicate=0 / role=teacher；n=1080，correct=866，accuracy=0.801852 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/1` |
| dataset=cruxeval_i / replicate=0 / role=student；n=800，correct=159，accuracy=0.19875 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/2` |
| dataset=cruxeval_i / replicate=0 / role=teacher；n=800，correct=393，accuracy=0.49125 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/3` |
| dataset=cruxeval_o / replicate=0 / role=student；n=800，correct=186，accuracy=0.2325 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/4` |
| dataset=cruxeval_o / replicate=0 / role=teacher；n=800，correct=370，accuracy=0.4625 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/5` |
| dataset=humaneval_plus / replicate=0 / role=student；n=164，correct=106，accuracy=0.646341 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/6` |
| dataset=humaneval_plus / replicate=0 / role=teacher；n=164，correct=139，accuracy=0.847561 | [reports/greedy_vs_temp1_20260921/greedy/summary.json](../../reports/greedy_vs_temp1_20260921/greedy/summary.json) `/groups/7` |

本源另有 24 条聚合记录；完整数值见 `result_records.csv.gz`。

### full_teacher_student_x8_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/full_teacher_student_x8_20260918](../../experiments/full_teacher_student_x8_20260918)。
- 配置/范围入口：[experiments/full_teacher_student_x8_20260918/LAUNCH.json](../../experiments/full_teacher_student_x8_20260918/LAUNCH.json)、[experiments/full_teacher_student_x8_20260918/README.md](../../experiments/full_teacher_student_x8_20260918/README.md)。
- 已保存阶段完成标记：[reports/greedy_vs_temp1_20260921/temp1/ALL_SCORING_COMPLETE.json](../../reports/greedy_vs_temp1_20260921/temp1/ALL_SCORING_COMPLETE.json)、[reports/greedy_vs_temp1_20260921/temp1/INPUTS_COMPLETE.json](../../reports/greedy_vs_temp1_20260921/temp1/INPUTS_COMPLETE.json)。
- 结果入口：[reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json)、[reports/full_baseline_x8_status_20260918/summary.json](../../reports/full_baseline_x8_status_20260918/summary.json)、[results/westlake/opd_eval_baselines/full_teacher_student_x8_20260918/summary.json](../../results/westlake/opd_eval_baselines/full_teacher_student_x8_20260918/summary.json)、[reports/full_baseline_x8_status_20260918/pass8_results.json](../../reports/full_baseline_x8_status_20260918/pass8_results.json)、[reports/greedy_vs_temp1_20260921/temp1/summary.csv](../../reports/greedy_vs_temp1_20260921/temp1/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh / replicate=0 / role=student；n=1080，correct=456，accuracy=0.422222 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/0` |
| dataset=bbh / replicate=1 / role=student；n=1080，correct=465，accuracy=0.430556 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/1` |
| dataset=bbh / replicate=2 / role=student；n=1080，correct=469，accuracy=0.434259 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/2` |
| dataset=bbh / replicate=3 / role=student；n=1080，correct=458，accuracy=0.424074 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/3` |
| dataset=bbh / replicate=4 / role=student；n=1080，correct=475，accuracy=0.439815 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/4` |
| dataset=bbh / replicate=5 / role=student；n=1080，correct=474，accuracy=0.438889 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/5` |
| dataset=bbh / replicate=6 / role=student；n=1080，correct=459，accuracy=0.425 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/6` |
| dataset=bbh / replicate=7 / role=student；n=1080，correct=462，accuracy=0.427778 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/7` |

本源另有 136 条聚合记录；完整数值见 `result_records.csv.gz`。

### full_teacher_student_x8_8gpu_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/full_teacher_student_x8_8gpu_20260918](../../experiments/full_teacher_student_x8_8gpu_20260918)。
- 配置/范围入口：[experiments/full_teacher_student_x8_8gpu_20260918/LAUNCH.json](../../experiments/full_teacher_student_x8_8gpu_20260918/LAUNCH.json)、[experiments/full_teacher_student_x8_8gpu_20260918/README.md](../../experiments/full_teacher_student_x8_8gpu_20260918/README.md)。
- 已保存阶段完成标记：[reports/greedy_vs_temp1_20260921/temp1/ALL_SCORING_COMPLETE.json](../../reports/greedy_vs_temp1_20260921/temp1/ALL_SCORING_COMPLETE.json)、[reports/greedy_vs_temp1_20260921/temp1/INPUTS_COMPLETE.json](../../reports/greedy_vs_temp1_20260921/temp1/INPUTS_COMPLETE.json)。
- 结果入口：[reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json)、[reports/full_baseline_x8_status_20260918/summary.json](../../reports/full_baseline_x8_status_20260918/summary.json)、[reports/full_baseline_x8_status_20260918/pass8_results.json](../../reports/full_baseline_x8_status_20260918/pass8_results.json)、[reports/greedy_vs_temp1_20260921/temp1/summary.csv](../../reports/greedy_vs_temp1_20260921/temp1/summary.csv)、[reports/full_baseline_x8_status_20260918/pass8_results.csv](../../reports/full_baseline_x8_status_20260918/pass8_results.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh / replicate=0 / role=student；n=1080，correct=456，accuracy=0.422222 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/0` |
| dataset=bbh / replicate=1 / role=student；n=1080，correct=465，accuracy=0.430556 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/1` |
| dataset=bbh / replicate=2 / role=student；n=1080，correct=469，accuracy=0.434259 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/2` |
| dataset=bbh / replicate=3 / role=student；n=1080，correct=458，accuracy=0.424074 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/3` |
| dataset=bbh / replicate=4 / role=student；n=1080，correct=475，accuracy=0.439815 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/4` |
| dataset=bbh / replicate=5 / role=student；n=1080，correct=474，accuracy=0.438889 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/5` |
| dataset=bbh / replicate=6 / role=student；n=1080，correct=459，accuracy=0.425 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/6` |
| dataset=bbh / replicate=7 / role=student；n=1080，correct=462，accuracy=0.427778 | [reports/greedy_vs_temp1_20260921/temp1/summary.json](../../reports/greedy_vs_temp1_20260921/temp1/summary.json) `/groups/7` |

本源另有 136 条聚合记录；完整数值见 `result_records.csv.gz`。

### grid_async_20260916

- 证据状态：**仅代码/状态文档**；代码目录：[experiments/grid_async_20260916](../../experiments/grid_async_20260916)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### grid_async_retry_20260916

- 证据状态：**仅代码/状态文档**；代码目录：[experiments/grid_async_retry_20260916](../../experiments/grid_async_retry_20260916)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### grid_fifo_20260916

- 证据状态：**仅代码/状态文档**；代码目录：[experiments/grid_fifo_20260916](../../experiments/grid_fifo_20260916)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### grid_math_4gpu_20260917

- 证据状态：**仅代码/状态文档**；代码目录：[experiments/grid_math_4gpu_20260917](../../experiments/grid_math_4gpu_20260917)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### grid_math_8gpu_20260917

- 证据状态：**仅代码/状态文档**；代码目录：[experiments/grid_math_8gpu_20260917](../../experiments/grid_math_8gpu_20260917)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### grid_math_colocated16_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/grid_math_colocated16_20260918](../../experiments/grid_math_colocated16_20260918)。
- 配置/范围入口：[experiments/grid_math_colocated16_20260918/LAUNCH.json](../../experiments/grid_math_colocated16_20260918/LAUNCH.json)、[experiments/grid_math_colocated16_20260918/README.md](../../experiments/grid_math_colocated16_20260918/README.md)。
- 结果入口：[reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json)、[reports/math_grid_x8_results_20260919/all_metrics.csv](../../reports/math_grid_x8_results_20260919/all_metrics.csv)、[reports/math_grid_x8_results_20260919/replicate_metrics.csv](../../reports/math_grid_x8_results_20260919/replicate_metrics.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| config=student_only / dataset=math500；questions=500，rollouts=4000，correct=2822，accuracy_pct=70.55，pass_at_8_pct=88，mean_output_tokens=880.94，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/0` |
| config=student_only / dataset=olympiad_bench；questions=674，rollouts=5392，correct=2197，accuracy_pct=40.7455，pass_at_8_pct=64.3917，mean_output_tokens=1467.85，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/1` |
| config=student_only / dataset=bbh；questions=1080，rollouts=8640，correct=3718，accuracy_pct=43.0324，pass_at_8_pct=74.8148，mean_output_tokens=481.056，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/2` |
| config=teacher_only / dataset=math500；questions=500，rollouts=4000，correct=3352，accuracy_pct=83.8，pass_at_8_pct=93，mean_output_tokens=932.448，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/3` |
| config=teacher_only / dataset=olympiad_bench；questions=674，rollouts=5392，correct=3166，accuracy_pct=58.7166，pass_at_8_pct=77.7448，mean_output_tokens=1528.68，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/4` |
| config=teacher_only / dataset=bbh；questions=1080，rollouts=8640，correct=6902，accuracy_pct=79.8843，pass_at_8_pct=90.463，mean_output_tokens=287.422，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/5` |
| config=k1_t2 / dataset=math500；questions=500，rollouts=4000，correct=3265，accuracy_pct=81.625，pass_at_8_pct=92.4，mean_output_tokens=832.829，mean_interventions=9.8125 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/6` |
| config=k1_t2 / dataset=olympiad_bench；questions=674，rollouts=5392，correct=2946，accuracy_pct=54.6365，pass_at_8_pct=73.5905，mean_output_tokens=1422.76，mean_interventions=17.1567 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/7` |

本源另有 73 条聚合记录；完整数值见 `result_records.csv.gz`。

### grid_math_colocated22_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/grid_math_colocated22_20260918](../../experiments/grid_math_colocated22_20260918)。
- 配置/范围入口：[experiments/grid_math_colocated22_20260918/LAUNCH.json](../../experiments/grid_math_colocated22_20260918/LAUNCH.json)、[experiments/grid_math_colocated22_20260918/README.md](../../experiments/grid_math_colocated22_20260918/README.md)。
- 结果入口：[reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json)、[reports/math_grid_x8_results_20260919/all_metrics.csv](../../reports/math_grid_x8_results_20260919/all_metrics.csv)、[reports/math_grid_x8_results_20260919/replicate_metrics.csv](../../reports/math_grid_x8_results_20260919/replicate_metrics.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| config=student_only / dataset=math500；questions=500，rollouts=4000，correct=2822，accuracy_pct=70.55，pass_at_8_pct=88，mean_output_tokens=880.94，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/0` |
| config=student_only / dataset=olympiad_bench；questions=674，rollouts=5392，correct=2197，accuracy_pct=40.7455，pass_at_8_pct=64.3917，mean_output_tokens=1467.85，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/1` |
| config=student_only / dataset=bbh；questions=1080，rollouts=8640，correct=3718，accuracy_pct=43.0324，pass_at_8_pct=74.8148，mean_output_tokens=481.056，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/2` |
| config=teacher_only / dataset=math500；questions=500，rollouts=4000，correct=3352，accuracy_pct=83.8，pass_at_8_pct=93，mean_output_tokens=932.448，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/3` |
| config=teacher_only / dataset=olympiad_bench；questions=674，rollouts=5392，correct=3166，accuracy_pct=58.7166，pass_at_8_pct=77.7448，mean_output_tokens=1528.68，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/4` |
| config=teacher_only / dataset=bbh；questions=1080，rollouts=8640，correct=6902，accuracy_pct=79.8843，pass_at_8_pct=90.463，mean_output_tokens=287.422，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/5` |
| config=k1_t2 / dataset=math500；questions=500，rollouts=4000，correct=3265，accuracy_pct=81.625，pass_at_8_pct=92.4，mean_output_tokens=832.829，mean_interventions=9.8125 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/6` |
| config=k1_t2 / dataset=olympiad_bench；questions=674，rollouts=5392，correct=2946，accuracy_pct=54.6365，pass_at_8_pct=73.5905，mean_output_tokens=1422.76，mean_interventions=17.1567 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/7` |

本源另有 73 条聚合记录；完整数值见 `result_records.csv.gz`。

### grid_math_colocated8_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/grid_math_colocated8_20260918](../../experiments/grid_math_colocated8_20260918)。
- 配置/范围入口：[experiments/grid_math_colocated8_20260918/LAUNCH.json](../../experiments/grid_math_colocated8_20260918/LAUNCH.json)、[experiments/grid_math_colocated8_20260918/README.md](../../experiments/grid_math_colocated8_20260918/README.md)。
- 结果入口：[reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json)、[reports/math_grid_x8_results_20260919/all_metrics.csv](../../reports/math_grid_x8_results_20260919/all_metrics.csv)、[reports/math_grid_x8_results_20260919/replicate_metrics.csv](../../reports/math_grid_x8_results_20260919/replicate_metrics.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| config=student_only / dataset=math500；questions=500，rollouts=4000，correct=2822，accuracy_pct=70.55，pass_at_8_pct=88，mean_output_tokens=880.94，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/0` |
| config=student_only / dataset=olympiad_bench；questions=674，rollouts=5392，correct=2197，accuracy_pct=40.7455，pass_at_8_pct=64.3917，mean_output_tokens=1467.85，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/1` |
| config=student_only / dataset=bbh；questions=1080，rollouts=8640，correct=3718，accuracy_pct=43.0324，pass_at_8_pct=74.8148，mean_output_tokens=481.056，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/2` |
| config=teacher_only / dataset=math500；questions=500，rollouts=4000，correct=3352，accuracy_pct=83.8，pass_at_8_pct=93，mean_output_tokens=932.448，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/3` |
| config=teacher_only / dataset=olympiad_bench；questions=674，rollouts=5392，correct=3166，accuracy_pct=58.7166，pass_at_8_pct=77.7448，mean_output_tokens=1528.68，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/4` |
| config=teacher_only / dataset=bbh；questions=1080，rollouts=8640，correct=6902，accuracy_pct=79.8843，pass_at_8_pct=90.463，mean_output_tokens=287.422，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/5` |
| config=k1_t2 / dataset=math500；questions=500，rollouts=4000，correct=3265，accuracy_pct=81.625，pass_at_8_pct=92.4，mean_output_tokens=832.829，mean_interventions=9.8125 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/6` |
| config=k1_t2 / dataset=olympiad_bench；questions=674，rollouts=5392，correct=2946，accuracy_pct=54.6365，pass_at_8_pct=73.5905，mean_output_tokens=1422.76，mean_interventions=17.1567 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/7` |

本源另有 73 条聚合记录；完整数值见 `result_records.csv.gz`。

### grid_math_gpt4o_mini_rescore_20260916

- 证据状态：**已有结果汇总**；代码目录：[experiments/grid_math_gpt4o_mini_rescore_20260916](../../experiments/grid_math_gpt4o_mini_rescore_20260916)。
- 配置/范围入口：[experiments/grid_math_gpt4o_mini_rescore_20260916/README.md](../../experiments/grid_math_gpt4o_mini_rescore_20260916/README.md)、[experiments/grid_math_gpt4o_mini_rescore_20260916/submission.json](../../experiments/grid_math_gpt4o_mini_rescore_20260916/submission.json)。
- 结果入口：[reports/grid_math_rescore_comparison_20260916/summary.json](../../reports/grid_math_rescore_comparison_20260916/summary.json)、`reports/grid_math_rescore_comparison_20260916/paired_results.json`（大文件保留在工作区；版本指纹见机器清单）、[reports/grid_math_rescore_comparison_20260916/per_dataset.csv](../../reports/grid_math_rescore_comparison_20260916/per_dataset.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh / config=k4_t2；accuracy=77.7778 | [reports/grid_math_rescore_comparison_20260916/summary.json](../../reports/grid_math_rescore_comparison_20260916/summary.json) `/rankings/bbh/old_top/0` |
| dataset=bbh / config=k16_t2；accuracy=80 | [reports/grid_math_rescore_comparison_20260916/summary.json](../../reports/grid_math_rescore_comparison_20260916/summary.json) `/rankings/bbh/new_top/0` |
| dataset=math500 / config=k8_t2；accuracy=86 | [reports/grid_math_rescore_comparison_20260916/summary.json](../../reports/grid_math_rescore_comparison_20260916/summary.json) `/rankings/math500/old_top/0` |
| dataset=math500 / config=k8_t2；accuracy=83.8 | [reports/grid_math_rescore_comparison_20260916/summary.json](../../reports/grid_math_rescore_comparison_20260916/summary.json) `/rankings/math500/new_top/0` |
| dataset=olympiad_bench / config=k8_t2；accuracy=45.6973 | [reports/grid_math_rescore_comparison_20260916/summary.json](../../reports/grid_math_rescore_comparison_20260916/summary.json) `/rankings/olympiad_bench/old_top/0` |
| dataset=olympiad_bench / config=k16_t2；accuracy=55.4896 | [reports/grid_math_rescore_comparison_20260916/summary.json](../../reports/grid_math_rescore_comparison_20260916/summary.json) `/rankings/olympiad_bench/new_top/0` |

### hard_k32_20260911

- 证据状态：**已有结果汇总**；代码目录：[experiments/hard_k32_20260911](../../experiments/hard_k32_20260911)。
- 配置/范围入口：[experiments/hard_k32_20260911/STATUS.md](../../experiments/hard_k32_20260911/STATUS.md)。
- 已保存阶段完成标记：[experiments/hard_k32_20260911/artifacts/ALL_GENERATION_COMPLETE.json](../../experiments/hard_k32_20260911/artifacts/ALL_GENERATION_COMPLETE.json)。
- 结果入口：[experiments/hard_k32_20260911/artifacts/summary.json](../../experiments/hard_k32_20260911/artifacts/summary.json)、[results/westlake/opd_eval_hard/hard_k32_20260911/summary.json](../../results/westlake/opd_eval_hard/hard_k32_20260911/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| method=online / dataset=aime25；questions=30，correct=3，accuracy=0.1 | [experiments/hard_k32_20260911/artifacts/summary.json](../../experiments/hard_k32_20260911/artifacts/summary.json) `/online/aime25` |
| method=online / dataset=aime24；questions=30，correct=4，accuracy=0.133333 | [experiments/hard_k32_20260911/artifacts/summary.json](../../experiments/hard_k32_20260911/artifacts/summary.json) `/online/aime24` |
| method=online / dataset=olympiad_bench；questions=674，correct=242，accuracy=0.35905 | [experiments/hard_k32_20260911/artifacts/summary.json](../../experiments/hard_k32_20260911/artifacts/summary.json) `/online/olympiad_bench` |
| method=online / dataset=bbh；questions=1080，correct=568，accuracy=0.525926 | [experiments/hard_k32_20260911/artifacts/summary.json](../../experiments/hard_k32_20260911/artifacts/summary.json) `/online/bbh` |

### hard_three_20260911

- 证据状态：**已有结果汇总**；代码目录：[experiments/hard_three_20260911](../../experiments/hard_three_20260911)。
- 配置/范围入口：[experiments/hard_three_20260911/STATUS.md](../../experiments/hard_three_20260911/STATUS.md)。
- 已保存阶段完成标记：[experiments/hard_three_20260911/artifacts/ALL_GENERATION_COMPLETE.json](../../experiments/hard_three_20260911/artifacts/ALL_GENERATION_COMPLETE.json)。
- 结果入口：[experiments/hard_three_20260911/artifacts/summary.json](../../experiments/hard_three_20260911/artifacts/summary.json)、[results/westlake/opd_eval_hard/hard_three_20260911/summary.json](../../results/westlake/opd_eval_hard/hard_three_20260911/summary.json)、[experiments/hard_three_20260911/intervention_analysis.json](../../experiments/hard_three_20260911/intervention_analysis.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| method=online / dataset=aime25；questions=30，correct=7，accuracy=0.233333 | [experiments/hard_three_20260911/artifacts/summary.json](../../experiments/hard_three_20260911/artifacts/summary.json) `/online/aime25` |
| method=online / dataset=aime24；questions=30，correct=3，accuracy=0.1 | [experiments/hard_three_20260911/artifacts/summary.json](../../experiments/hard_three_20260911/artifacts/summary.json) `/online/aime24` |
| method=online / dataset=olympiad_bench；questions=674，correct=232，accuracy=0.344214 | [experiments/hard_three_20260911/artifacts/summary.json](../../experiments/hard_three_20260911/artifacts/summary.json) `/online/olympiad_bench` |
| method=online / dataset=bbh；questions=1080，correct=484，accuracy=0.448148 | [experiments/hard_three_20260911/artifacts/summary.json](../../experiments/hard_three_20260911/artifacts/summary.json) `/online/bbh` |
| method=student / dataset=aime24；questions=30，correct=5，accuracy=0.166667 | [experiments/hard_three_20260911/artifacts/summary.json](../../experiments/hard_three_20260911/artifacts/summary.json) `/student/aime24` |
| method=student / dataset=aime25；questions=30，correct=4，accuracy=0.133333 | [experiments/hard_three_20260911/artifacts/summary.json](../../experiments/hard_three_20260911/artifacts/summary.json) `/student/aime25` |
| method=student / dataset=olympiad_bench；questions=674，correct=202，accuracy=0.299703 | [experiments/hard_three_20260911/artifacts/summary.json](../../experiments/hard_three_20260911/artifacts/summary.json) `/student/olympiad_bench` |
| method=student / dataset=bbh；questions=1080，correct=378，accuracy=0.35 | [experiments/hard_three_20260911/artifacts/summary.json](../../experiments/hard_three_20260911/artifacts/summary.json) `/student/bbh` |

本源另有 4 条聚合记录；完整数值见 `result_records.csv.gz`。

### intervention_analysis_greedy_20260921

- 证据状态：**数值分析完成；语义/反事实未完**；代码目录：[experiments/intervention_analysis_greedy_20260921](../../experiments/intervention_analysis_greedy_20260921)。
- 配置/范围入口：[experiments/intervention_analysis_greedy_20260921/README.md](../../experiments/intervention_analysis_greedy_20260921/README.md)。
- 已保存阶段完成标记：[reports/intervention_analysis_greedy_20260921/ANALYSIS_COMPLETE.json](../../reports/intervention_analysis_greedy_20260921/ANALYSIS_COMPLETE.json)、[reports/intervention_analysis_greedy_20260921/EXTRACTION_COMPLETE.json](../../reports/intervention_analysis_greedy_20260921/EXTRACTION_COMPLETE.json)、[reports/intervention_analysis_greedy_20260921/continuous_support/COMPLETE.json](../../reports/intervention_analysis_greedy_20260921/continuous_support/COMPLETE.json)、[reports/intervention_analysis_greedy_20260921/probability_pair/COMPLETE.json](../../reports/intervention_analysis_greedy_20260921/probability_pair/COMPLETE.json)、[reports/intervention_analysis_greedy_20260921/regime_comparison/COMPLETE.json](../../reports/intervention_analysis_greedy_20260921/regime_comparison/COMPLETE.json)。
- 结果入口：[reports/intervention_analysis_greedy_20260921/headline.json](../../reports/intervention_analysis_greedy_20260921/headline.json)、[reports/intervention_analysis_greedy_20260921/ANALYSIS_COMPLETE.json](../../reports/intervention_analysis_greedy_20260921/ANALYSIS_COMPLETE.json)、[reports/intervention_analysis_greedy_20260921/measurement_audit.json](../../reports/intervention_analysis_greedy_20260921/measurement_audit.json)、[reports/intervention_analysis_greedy_20260921/EXTRACTION_COMPLETE.json](../../reports/intervention_analysis_greedy_20260921/EXTRACTION_COMPLETE.json)、[reports/intervention_analysis_greedy_20260921/probability_pair/COMPLETE.json](../../reports/intervention_analysis_greedy_20260921/probability_pair/COMPLETE.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| questions=4454，events=43879 | [reports/intervention_analysis_greedy_20260921/headline.json](../../reports/intervention_analysis_greedy_20260921/headline.json) `/pooled_event` |

### intervention_analysis_phase1_20260921

- 证据状态：**数值分析完成；语义/反事实未完**；代码目录：[experiments/intervention_analysis_phase1_20260921](../../experiments/intervention_analysis_phase1_20260921)。
- 配置/范围入口：[experiments/intervention_analysis_phase1_20260921/README.md](../../experiments/intervention_analysis_phase1_20260921/README.md)。
- 已保存阶段完成标记：[reports/intervention_analysis_phase1_20260921/ANALYSIS_COMPLETE.json](../../reports/intervention_analysis_phase1_20260921/ANALYSIS_COMPLETE.json)、[reports/intervention_analysis_phase1_20260921/EXTRACTION_COMPLETE.json](../../reports/intervention_analysis_phase1_20260921/EXTRACTION_COMPLETE.json)、[reports/intervention_analysis_phase1_20260921/continuous_support/COMPLETE.json](../../reports/intervention_analysis_phase1_20260921/continuous_support/COMPLETE.json)、[reports/intervention_analysis_phase1_20260921/probability_pair/COMPLETE.json](../../reports/intervention_analysis_phase1_20260921/probability_pair/COMPLETE.json)、[reports/intervention_analysis_phase1_20260921/teacher_uncertainty/COMPLETE.json](../../reports/intervention_analysis_phase1_20260921/teacher_uncertainty/COMPLETE.json)。
- 结果入口：[reports/intervention_analysis_phase1_20260921/headline.json](../../reports/intervention_analysis_phase1_20260921/headline.json)、[reports/intervention_analysis_phase1_20260921/ANALYSIS_COMPLETE.json](../../reports/intervention_analysis_phase1_20260921/ANALYSIS_COMPLETE.json)、[reports/intervention_analysis_phase1_20260921/measurement_audit.json](../../reports/intervention_analysis_phase1_20260921/measurement_audit.json)、[reports/intervention_analysis_phase1_20260921/EXTRACTION_COMPLETE.json](../../reports/intervention_analysis_phase1_20260921/EXTRACTION_COMPLETE.json)、[reports/intervention_analysis_phase1_20260921/probability_pair/COMPLETE.json](../../reports/intervention_analysis_phase1_20260921/probability_pair/COMPLETE.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| questions=4549，events=399601 | [reports/intervention_analysis_phase1_20260921/headline.json](../../reports/intervention_analysis_phase1_20260921/headline.json) `/pooled_event` |

### intervention_entropy_20260914

- 证据状态：**已有结果汇总**；代码目录：[experiments/intervention_entropy_20260914](../../experiments/intervention_entropy_20260914)。
- 配置/范围入口：[experiments/intervention_entropy_20260914/PLAN.md](../../experiments/intervention_entropy_20260914/PLAN.md)。
- 已保存阶段完成标记：[experiments/intervention_entropy_20260914/results/COMPLETE.json](../../experiments/intervention_entropy_20260914/results/COMPLETE.json)、[experiments/intervention_entropy_20260914/results/student_COMPLETE.json](../../experiments/intervention_entropy_20260914/results/student_COMPLETE.json)、[experiments/intervention_entropy_20260914/results/teacher_COMPLETE.json](../../experiments/intervention_entropy_20260914/results/teacher_COMPLETE.json)。
- 结果入口：[experiments/intervention_entropy_20260914/results/summary.csv](../../experiments/intervention_entropy_20260914/results/summary.csv)、[results/westlake/opd_analysis/intervention_entropy_20260914/summary.csv](../../results/westlake/opd_analysis/intervention_entropy_20260914/summary.csv)。

### intervention_logits_bf16_greedy_20260921

- 证据状态：**归档校验完成；大载荷保留西湖**；代码目录：[experiments/intervention_logits_bf16_greedy_20260921](../../experiments/intervention_logits_bf16_greedy_20260921)。
- 配置/范围入口：[experiments/intervention_logits_bf16_greedy_20260921/LAUNCH.json](../../experiments/intervention_logits_bf16_greedy_20260921/LAUNCH.json)、[experiments/intervention_logits_bf16_greedy_20260921/README.md](../../experiments/intervention_logits_bf16_greedy_20260921/README.md)、[experiments/intervention_logits_bf16_greedy_20260921/STATUS.md](../../experiments/intervention_logits_bf16_greedy_20260921/STATUS.md)。
- 已保存阶段完成标记：[experiments/intervention_logits_bf16_greedy_20260921/ALL_LOGITS_VERIFIED.json](../../experiments/intervention_logits_bf16_greedy_20260921/ALL_LOGITS_VERIFIED.json)、[experiments/intervention_logits_bf16_greedy_20260921/INPUTS_COMPLETE.json](../../experiments/intervention_logits_bf16_greedy_20260921/INPUTS_COMPLETE.json)。
- 结果入口：[experiments/intervention_logits_bf16_greedy_20260921/INPUTS_COMPLETE.json](../../experiments/intervention_logits_bf16_greedy_20260921/INPUTS_COMPLETE.json)、[experiments/intervention_logits_bf16_greedy_20260921/ALL_LOGITS_VERIFIED.json](../../experiments/intervention_logits_bf16_greedy_20260921/ALL_LOGITS_VERIFIED.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| events=43879 | [experiments/intervention_logits_bf16_greedy_20260921/INPUTS_COMPLETE.json](../../experiments/intervention_logits_bf16_greedy_20260921/INPUTS_COMPLETE.json) `/` |
- BF16 完成标记记录远端归档已校验；全部张量仍保留西湖，本地只同步小摘要和校验标记。

### intervention_logits_bf16_x8_20260920

- 证据状态：**归档校验完成；大载荷保留西湖**；代码目录：[experiments/intervention_logits_bf16_x8_20260920](../../experiments/intervention_logits_bf16_x8_20260920)。
- 配置/范围入口：[experiments/intervention_logits_bf16_x8_20260920/LAUNCH.json](../../experiments/intervention_logits_bf16_x8_20260920/LAUNCH.json)、[experiments/intervention_logits_bf16_x8_20260920/README.md](../../experiments/intervention_logits_bf16_x8_20260920/README.md)。
- 已保存阶段完成标记：[experiments/intervention_logits_bf16_x8_20260920/ALL_LOGITS_VERIFIED.json](../../experiments/intervention_logits_bf16_x8_20260920/ALL_LOGITS_VERIFIED.json)、[experiments/intervention_logits_bf16_x8_20260920/INPUTS_COMPLETE.json](../../experiments/intervention_logits_bf16_x8_20260920/INPUTS_COMPLETE.json)。
- 结果入口：[experiments/intervention_logits_bf16_x8_20260920/INPUTS_COMPLETE.json](../../experiments/intervention_logits_bf16_x8_20260920/INPUTS_COMPLETE.json)、[experiments/intervention_logits_bf16_x8_20260920/ALL_LOGITS_VERIFIED.json](../../experiments/intervention_logits_bf16_x8_20260920/ALL_LOGITS_VERIFIED.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| events=399601 | [experiments/intervention_logits_bf16_x8_20260920/INPUTS_COMPLETE.json](../../experiments/intervention_logits_bf16_x8_20260920/INPUTS_COMPLETE.json) `/` |
- BF16 完成标记记录远端归档已校验；全部张量仍保留西湖，本地只同步小摘要和校验标记。

### k1_top1_agreement_20260914

- 证据状态：**仅计划/代码证据**；代码目录：[experiments/k1_top1_agreement_20260914](../../experiments/k1_top1_agreement_20260914)。
- 配置/范围入口：[experiments/k1_top1_agreement_20260914/PLAN.md](../../experiments/k1_top1_agreement_20260914/PLAN.md)。
- 已保存阶段完成标记：[experiments/k1_top1_agreement_20260914/results/COMPLETE.json](../../experiments/k1_top1_agreement_20260914/results/COMPLETE.json)、[experiments/k1_top1_agreement_20260914/results/student_COMPLETE.json](../../experiments/k1_top1_agreement_20260914/results/student_COMPLETE.json)、[experiments/k1_top1_agreement_20260914/results/teacher_COMPLETE.json](../../experiments/k1_top1_agreement_20260914/results/teacher_COMPLETE.json)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### logits_replay_20260918

- 证据状态：**仅代码/状态文档**；代码目录：[experiments/logits_replay_20260918](../../experiments/logits_replay_20260918)。
- 配置/范围入口：[experiments/logits_replay_20260918/README.md](../../experiments/logits_replay_20260918/README.md)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### logits_threshold2_replay_20260918

- 证据状态：**仅代码/状态文档**；代码目录：[experiments/logits_threshold2_replay_20260918](../../experiments/logits_threshold2_replay_20260918)。
- 配置/范围入口：[experiments/logits_threshold2_replay_20260918/README.md](../../experiments/logits_threshold2_replay_20260918/README.md)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### main_greedy_k1_t2_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/main_greedy_k1_t2_20260921](../../experiments/main_greedy_k1_t2_20260921)。
- 配置/范围入口：[experiments/main_greedy_k1_t2_20260921/LAUNCH.json](../../experiments/main_greedy_k1_t2_20260921/LAUNCH.json)、[experiments/main_greedy_k1_t2_20260921/README.md](../../experiments/main_greedy_k1_t2_20260921/README.md)。
- 结果入口：[reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json)、[results/westlake/opd_eval_ablation/main_greedy_k1_t2_20260921/summary.json](../../results/westlake/opd_eval_ablation/main_greedy_k1_t2_20260921/summary.json)、[reports/main_greedy_comparison_20260921/four_metrics.json](../../reports/main_greedy_comparison_20260921/four_metrics.json)、[reports/main_greedy_comparison_20260921/comparison.csv](../../reports/main_greedy_comparison_20260921/comparison.csv)、[reports/main_greedy_comparison_20260921/four_metrics.csv](../../reports/main_greedy_comparison_20260921/four_metrics.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=849，accuracy=0.786111 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=394，accuracy=0.4925 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=370，accuracy=0.4625 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=135，accuracy=0.823171 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=50，accuracy=0.285714 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=412，accuracy=0.824 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=276，accuracy=0.730159 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=357，accuracy=0.529674 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/7` |

### main_greedy_score_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/main_greedy_score_20260921](../../experiments/main_greedy_score_20260921)。
- 结果入口：[reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json)、[reports/main_greedy_comparison_20260921/four_metrics.json](../../reports/main_greedy_comparison_20260921/four_metrics.json)、[reports/main_greedy_comparison_20260921/comparison.csv](../../reports/main_greedy_comparison_20260921/comparison.csv)、[reports/main_greedy_comparison_20260921/four_metrics.csv](../../reports/main_greedy_comparison_20260921/four_metrics.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=849，accuracy=0.786111 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=394，accuracy=0.4925 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=370，accuracy=0.4625 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=135，accuracy=0.823171 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=50，accuracy=0.285714 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=412，accuracy=0.824 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=276，accuracy=0.730159 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=357，accuracy=0.529674 | [reports/main_greedy_comparison_20260921/summary.json](../../reports/main_greedy_comparison_20260921/summary.json) `/groups/7` |

### math500_three_20260911

- 证据状态：**已有结果汇总**；代码目录：[experiments/math500_three_20260911](../../experiments/math500_three_20260911)。
- 配置/范围入口：[experiments/math500_three_20260911/STATUS.md](../../experiments/math500_three_20260911/STATUS.md)。
- 已保存阶段完成标记：[experiments/math500_three_20260911/results/ALL_GENERATION_COMPLETE.json](../../experiments/math500_three_20260911/results/ALL_GENERATION_COMPLETE.json)。
- 结果入口：[experiments/math500_three_20260911/results/summary.json](../../experiments/math500_three_20260911/results/summary.json)、[results/westlake/opd_eval_math500/math500_three_20260911/summary.json](../../results/westlake/opd_eval_math500/math500_three_20260911/summary.json)、[experiments/math500_three_20260911/audit.json](../../experiments/math500_three_20260911/audit.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| method=student；questions=500，correct=364，accuracy=0.728，mean_tokens=885.078 | [experiments/math500_three_20260911/results/summary.json](../../experiments/math500_three_20260911/results/summary.json) `/student` |
| method=online；questions=500，correct=407，accuracy=0.814，mean_tokens=882.754 | [experiments/math500_three_20260911/results/summary.json](../../experiments/math500_three_20260911/results/summary.json) `/online` |
| method=prefix256；questions=500，correct=376，accuracy=0.752，mean_tokens=876.026 | [experiments/math500_three_20260911/results/summary.json](../../experiments/math500_three_20260911/results/summary.json) `/prefix256` |

### math_grid_score_029041_20260919

- 证据状态：**已有结果汇总**；代码目录：[experiments/math_grid_score_029041_20260919](../../experiments/math_grid_score_029041_20260919)。
- 配置/范围入口：[experiments/math_grid_score_029041_20260919/LAUNCH.json](../../experiments/math_grid_score_029041_20260919/LAUNCH.json)。
- 结果入口：[reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json)、[results/westlake/opd_rescore/math_grid_score_029041_20260919/summary.json](../../results/westlake/opd_rescore/math_grid_score_029041_20260919/summary.json)、[reports/math_grid_x8_results_20260919/all_metrics.csv](../../reports/math_grid_x8_results_20260919/all_metrics.csv)、[reports/math_grid_x8_results_20260919/replicate_metrics.csv](../../reports/math_grid_x8_results_20260919/replicate_metrics.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| config=student_only / dataset=math500；questions=500，rollouts=4000，correct=2822，accuracy_pct=70.55，pass_at_8_pct=88，mean_output_tokens=880.94，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/0` |
| config=student_only / dataset=olympiad_bench；questions=674，rollouts=5392，correct=2197，accuracy_pct=40.7455，pass_at_8_pct=64.3917，mean_output_tokens=1467.85，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/1` |
| config=student_only / dataset=bbh；questions=1080，rollouts=8640，correct=3718，accuracy_pct=43.0324，pass_at_8_pct=74.8148，mean_output_tokens=481.056，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/2` |
| config=teacher_only / dataset=math500；questions=500，rollouts=4000，correct=3352，accuracy_pct=83.8，pass_at_8_pct=93，mean_output_tokens=932.448，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/3` |
| config=teacher_only / dataset=olympiad_bench；questions=674，rollouts=5392，correct=3166，accuracy_pct=58.7166，pass_at_8_pct=77.7448，mean_output_tokens=1528.68，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/4` |
| config=teacher_only / dataset=bbh；questions=1080，rollouts=8640，correct=6902，accuracy_pct=79.8843，pass_at_8_pct=90.463，mean_output_tokens=287.422，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/5` |
| config=k1_t2 / dataset=math500；questions=500，rollouts=4000，correct=3265，accuracy_pct=81.625，pass_at_8_pct=92.4，mean_output_tokens=832.829，mean_interventions=9.8125 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/6` |
| config=k1_t2 / dataset=olympiad_bench；questions=674，rollouts=5392，correct=2946，accuracy_pct=54.6365，pass_at_8_pct=73.5905，mean_output_tokens=1422.76，mean_interventions=17.1567 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/7` |

本源另有 73 条聚合记录；完整数值见 `result_records.csv.gz`。

### math_grid_score_all_2000_20260919

- 证据状态：**已有结果汇总**；代码目录：[experiments/math_grid_score_all_2000_20260919](../../experiments/math_grid_score_all_2000_20260919)。
- 配置/范围入口：[experiments/math_grid_score_all_2000_20260919/LAUNCH.json](../../experiments/math_grid_score_all_2000_20260919/LAUNCH.json)、[experiments/math_grid_score_all_2000_20260919/README.md](../../experiments/math_grid_score_all_2000_20260919/README.md)。
- 结果入口：[reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json)、[results/westlake/opd_rescore/math_grid_score_all_2000_20260919/summary.json](../../results/westlake/opd_rescore/math_grid_score_all_2000_20260919/summary.json)、[reports/math_grid_x8_results_20260919/all_metrics.csv](../../reports/math_grid_x8_results_20260919/all_metrics.csv)、[reports/math_grid_x8_results_20260919/replicate_metrics.csv](../../reports/math_grid_x8_results_20260919/replicate_metrics.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| config=student_only / dataset=math500；questions=500，rollouts=4000，correct=2822，accuracy_pct=70.55，pass_at_8_pct=88，mean_output_tokens=880.94，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/0` |
| config=student_only / dataset=olympiad_bench；questions=674，rollouts=5392，correct=2197，accuracy_pct=40.7455，pass_at_8_pct=64.3917，mean_output_tokens=1467.85，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/1` |
| config=student_only / dataset=bbh；questions=1080，rollouts=8640，correct=3718，accuracy_pct=43.0324，pass_at_8_pct=74.8148，mean_output_tokens=481.056，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/2` |
| config=teacher_only / dataset=math500；questions=500，rollouts=4000，correct=3352，accuracy_pct=83.8，pass_at_8_pct=93，mean_output_tokens=932.448，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/3` |
| config=teacher_only / dataset=olympiad_bench；questions=674，rollouts=5392，correct=3166，accuracy_pct=58.7166，pass_at_8_pct=77.7448，mean_output_tokens=1528.68，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/4` |
| config=teacher_only / dataset=bbh；questions=1080，rollouts=8640，correct=6902，accuracy_pct=79.8843，pass_at_8_pct=90.463，mean_output_tokens=287.422，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/5` |
| config=k1_t2 / dataset=math500；questions=500，rollouts=4000，correct=3265，accuracy_pct=81.625，pass_at_8_pct=92.4，mean_output_tokens=832.829，mean_interventions=9.8125 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/6` |
| config=k1_t2 / dataset=olympiad_bench；questions=674，rollouts=5392，correct=2946，accuracy_pct=54.6365，pass_at_8_pct=73.5905，mean_output_tokens=1422.76，mean_interventions=17.1567 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/7` |

本源另有 73 条聚合记录；完整数值见 `result_records.csv.gz`。

### math_reasoning_grid_x8_20260916

- 证据状态：**已有结果汇总**；代码目录：[experiments/math_reasoning_grid_x8_20260916](../../experiments/math_reasoning_grid_x8_20260916)。
- 配置/范围入口：[experiments/math_reasoning_grid_x8_20260916/LAUNCH.md](../../experiments/math_reasoning_grid_x8_20260916/LAUNCH.md)、[experiments/math_reasoning_grid_x8_20260916/PLAN.json](../../experiments/math_reasoning_grid_x8_20260916/PLAN.json)、[experiments/math_reasoning_grid_x8_20260916/configs.json](../../experiments/math_reasoning_grid_x8_20260916/configs.json)、[experiments/math_reasoning_grid_x8_20260916/submission.json](../../experiments/math_reasoning_grid_x8_20260916/submission.json)。
- 结果入口：[reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json)、[reports/math_grid_x8_results_20260919/all_metrics.csv](../../reports/math_grid_x8_results_20260919/all_metrics.csv)、[reports/math_grid_x8_results_20260919/replicate_metrics.csv](../../reports/math_grid_x8_results_20260919/replicate_metrics.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| config=student_only / dataset=math500；questions=500，rollouts=4000，correct=2822，accuracy_pct=70.55，pass_at_8_pct=88，mean_output_tokens=880.94，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/0` |
| config=student_only / dataset=olympiad_bench；questions=674，rollouts=5392，correct=2197，accuracy_pct=40.7455，pass_at_8_pct=64.3917，mean_output_tokens=1467.85，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/1` |
| config=student_only / dataset=bbh；questions=1080，rollouts=8640，correct=3718，accuracy_pct=43.0324，pass_at_8_pct=74.8148，mean_output_tokens=481.056，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/2` |
| config=teacher_only / dataset=math500；questions=500，rollouts=4000，correct=3352，accuracy_pct=83.8，pass_at_8_pct=93，mean_output_tokens=932.448，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/3` |
| config=teacher_only / dataset=olympiad_bench；questions=674，rollouts=5392，correct=3166，accuracy_pct=58.7166，pass_at_8_pct=77.7448，mean_output_tokens=1528.68，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/4` |
| config=teacher_only / dataset=bbh；questions=1080，rollouts=8640，correct=6902，accuracy_pct=79.8843，pass_at_8_pct=90.463，mean_output_tokens=287.422，mean_interventions=0 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/5` |
| config=k1_t2 / dataset=math500；questions=500，rollouts=4000，correct=3265，accuracy_pct=81.625，pass_at_8_pct=92.4，mean_output_tokens=832.829，mean_interventions=9.8125 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/6` |
| config=k1_t2 / dataset=olympiad_bench；questions=674，rollouts=5392，correct=2946，accuracy_pct=54.6365，pass_at_8_pct=73.5905，mean_output_tokens=1422.76，mean_interventions=17.1567 | [reports/math_grid_x8_results_20260919/summary.json](../../reports/math_grid_x8_results_20260919/summary.json) `/results/7` |

本源另有 73 条聚合记录；完整数值见 `result_records.csv.gz`。

### nojudge_grid_20260912

- 证据状态：**已有结果汇总**；代码目录：[experiments/nojudge_grid_20260912](../../experiments/nojudge_grid_20260912)。
- 配置/范围入口：[experiments/nojudge_grid_20260912/STATUS.md](../../experiments/nojudge_grid_20260912/STATUS.md)、[experiments/nojudge_grid_20260912/artifacts/configs.json](../../experiments/nojudge_grid_20260912/artifacts/configs.json)、[experiments/nojudge_grid_20260912/configs.json](../../experiments/nojudge_grid_20260912/configs.json)。
- 已保存阶段完成标记：[experiments/nojudge_grid_20260912/artifacts/ALL_GENERATION_COMPLETE.json](../../experiments/nojudge_grid_20260912/artifacts/ALL_GENERATION_COMPLETE.json)、[experiments/nojudge_grid_20260912/artifacts/k16_t2/VERIFIED.json](../../experiments/nojudge_grid_20260912/artifacts/k16_t2/VERIFIED.json)、[experiments/nojudge_grid_20260912/artifacts/k16_t2p5/VERIFIED.json](../../experiments/nojudge_grid_20260912/artifacts/k16_t2p5/VERIFIED.json)。
- 结果入口：[experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json](../../experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json)、[experiments/nojudge_grid_20260912/artifacts/combined_completed_metrics.json](../../experiments/nojudge_grid_20260912/artifacts/combined_completed_metrics.json)、[results/westlake/opd_eval_hard/nojudge_grid_20260912/combined_completed_metrics.json](../../results/westlake/opd_eval_hard/nojudge_grid_20260912/combined_completed_metrics.json)、[results/westlake/opd_eval_hard/nojudge_grid_20260912/scoring_batches/batch_001/summary.json](../../results/westlake/opd_eval_hard/nojudge_grid_20260912/scoring_batches/batch_001/summary.json)、[results/westlake/opd_eval_hard/nojudge_grid_20260912/scoring_batches/batch_002/summary.json](../../results/westlake/opd_eval_hard/nojudge_grid_20260912/scoring_batches/batch_002/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| config=k1_t2 / dataset=olympiad_bench；n=674，correct=299，accuracy=0.44362，mean_interventions=17.9407 | [experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json](../../experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json) `/groups/k1_t2/olympiad_bench` |
| config=k1_t2 / dataset=bbh；n=1080，correct=830，accuracy=0.768519，mean_interventions=18.6056 | [experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json](../../experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json) `/groups/k1_t2/bbh` |
| config=k1_t2 / dataset=math500；n=500，correct=419，accuracy=0.838，mean_interventions=9.71 | [experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json](../../experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json) `/groups/k1_t2/math500` |
| config=k1_t2p5 / dataset=olympiad_bench；n=674，correct=301，accuracy=0.446588，mean_interventions=10.9614 | [experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json](../../experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json) `/groups/k1_t2p5/olympiad_bench` |
| config=k1_t2p5 / dataset=bbh；n=1080，correct=835，accuracy=0.773148，mean_interventions=15.7806 | [experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json](../../experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json) `/groups/k1_t2p5/bbh` |
| config=k1_t2p5 / dataset=math500；n=500，correct=411，accuracy=0.822，mean_interventions=6.29 | [experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json](../../experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json) `/groups/k1_t2p5/math500` |
| config=k1_t3 / dataset=olympiad_bench；n=674，correct=279，accuracy=0.413947，mean_interventions=7.32938 | [experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json](../../experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json) `/groups/k1_t3/olympiad_bench` |
| config=k1_t3 / dataset=bbh；n=1080，correct=836，accuracy=0.774074，mean_interventions=13.7972 | [experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json](../../experiments/nojudge_grid_20260912/reports/combined_completed_metrics.json) `/groups/k1_t3/bbh` |

本源另有 43 条聚合记录；完整数值见 `result_records.csv.gz`。

### online_comparison_20260911

- 证据状态：**已有结果汇总**；代码目录：[experiments/online_comparison_20260911](../../experiments/online_comparison_20260911)。
- 结果入口：[experiments/online_comparison_20260911/results.json](../../experiments/online_comparison_20260911/results.json)、[experiments/online_comparison_20260911/comparison.json](../../experiments/online_comparison_20260911/comparison.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| questions=1000，correct=476，accuracy=0.476 | [experiments/online_comparison_20260911/results.json](../../experiments/online_comparison_20260911/results.json) `/summary` |

### prefix_takeover_caps11_20_20260917

- 证据状态：**已有结果汇总**；代码目录：[experiments/prefix_takeover_caps11_20_20260917](../../experiments/prefix_takeover_caps11_20_20260917)。
- 配置/范围入口：[experiments/prefix_takeover_caps11_20_20260917/LAUNCH.json](../../experiments/prefix_takeover_caps11_20_20260917/LAUNCH.json)、[experiments/prefix_takeover_caps11_20_20260917/README.md](../../experiments/prefix_takeover_caps11_20_20260917/README.md)。
- 结果入口：[reports/prefix_caps0_20_analysis_20260918/summary.json](../../reports/prefix_caps0_20_analysis_20260918/summary.json)、[results/westlake/opd_eval_prefix/prefix_takeover_caps11_20_20260917/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps11_20_20260917/summary.json)、[results/westlake/opd_eval_prefix/prefix_takeover_caps11_20_20260917/summary.csv](../../results/westlake/opd_eval_prefix/prefix_takeover_caps11_20_20260917/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=800，correct=485，accuracy=0.60625 | [reports/prefix_caps0_20_analysis_20260918/summary.json](../../reports/prefix_caps0_20_analysis_20260918/summary.json) `/groups/0` |
| dataset=bbh；n=800，correct=495，accuracy=0.61875 | [reports/prefix_caps0_20_analysis_20260918/summary.json](../../reports/prefix_caps0_20_analysis_20260918/summary.json) `/groups/1` |
| dataset=bbh；n=800，correct=509，accuracy=0.63625 | [reports/prefix_caps0_20_analysis_20260918/summary.json](../../reports/prefix_caps0_20_analysis_20260918/summary.json) `/groups/2` |
| dataset=bbh；n=800，correct=511，accuracy=0.63875 | [reports/prefix_caps0_20_analysis_20260918/summary.json](../../reports/prefix_caps0_20_analysis_20260918/summary.json) `/groups/3` |
| dataset=bbh；n=800，correct=519，accuracy=0.64875 | [reports/prefix_caps0_20_analysis_20260918/summary.json](../../reports/prefix_caps0_20_analysis_20260918/summary.json) `/groups/4` |
| dataset=bbh；n=800，correct=523，accuracy=0.65375 | [reports/prefix_caps0_20_analysis_20260918/summary.json](../../reports/prefix_caps0_20_analysis_20260918/summary.json) `/groups/5` |
| dataset=bbh；n=800，correct=543，accuracy=0.67875 | [reports/prefix_caps0_20_analysis_20260918/summary.json](../../reports/prefix_caps0_20_analysis_20260918/summary.json) `/groups/6` |
| dataset=bbh；n=800，correct=531，accuracy=0.66375 | [reports/prefix_caps0_20_analysis_20260918/summary.json](../../reports/prefix_caps0_20_analysis_20260918/summary.json) `/groups/7` |

本源另有 72 条聚合记录；完整数值见 `result_records.csv.gz`。

### prefix_takeover_caps25_40_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/prefix_takeover_caps25_40_20260918](../../experiments/prefix_takeover_caps25_40_20260918)。
- 配置/范围入口：[experiments/prefix_takeover_caps25_40_20260918/LAUNCH.json](../../experiments/prefix_takeover_caps25_40_20260918/LAUNCH.json)、[experiments/prefix_takeover_caps25_40_20260918/README.md](../../experiments/prefix_takeover_caps25_40_20260918/README.md)。
- 结果入口：[results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json)、[results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/code_summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/code_summary.json)、[results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.csv](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=800，correct=580，accuracy=0.725 | [results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json) `/groups/0` |
| dataset=bbh；n=800，correct=599，accuracy=0.74875 | [results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json) `/groups/1` |
| dataset=bbh；n=800，correct=603，accuracy=0.75375 | [results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json) `/groups/2` |
| dataset=bbh；n=800，correct=612，accuracy=0.765 | [results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json) `/groups/3` |
| dataset=cruxeval_i；n=800，correct=393，accuracy=0.49125 | [results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json) `/groups/4` |
| dataset=cruxeval_i；n=800，correct=393，accuracy=0.49125 | [results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json) `/groups/5` |
| dataset=cruxeval_i；n=800，correct=393，accuracy=0.49125 | [results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json) `/groups/6` |
| dataset=cruxeval_i；n=800，correct=393，accuracy=0.49125 | [results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.json) `/groups/7` |

本源另有 24 条聚合记录；完整数值见 `result_records.csv.gz`。

### prefix_takeover_caps25_40_split_20260918

- 证据状态：**仅定位部署记录**；代码目录：[experiments/prefix_takeover_caps25_40_split_20260918](../../experiments/prefix_takeover_caps25_40_split_20260918)。
- 配置/范围入口：[experiments/prefix_takeover_caps25_40_split_20260918/LAUNCH.json](../../experiments/prefix_takeover_caps25_40_split_20260918/LAUNCH.json)、[experiments/prefix_takeover_caps25_40_split_20260918/README.md](../../experiments/prefix_takeover_caps25_40_split_20260918/README.md)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### prefix_takeover_caps5_10_20260917

- 证据状态：**已有结果汇总**；代码目录：[experiments/prefix_takeover_caps5_10_20260917](../../experiments/prefix_takeover_caps5_10_20260917)。
- 配置/范围入口：[experiments/prefix_takeover_caps5_10_20260917/README.md](../../experiments/prefix_takeover_caps5_10_20260917/README.md)。
- 已保存阶段完成标记：[reports/prefix_takeover_caps5_10_20260917/ALL_GENERATION_COMPLETE.json](../../reports/prefix_takeover_caps5_10_20260917/ALL_GENERATION_COMPLETE.json)、[reports/prefix_takeover_caps5_10_20260917/ALL_SCORING_COMPLETE.json](../../reports/prefix_takeover_caps5_10_20260917/ALL_SCORING_COMPLETE.json)、[reports/prefix_takeover_caps5_10_20260917/INPUTS_COMPLETE.json](../../reports/prefix_takeover_caps5_10_20260917/INPUTS_COMPLETE.json)。
- 结果入口：[reports/prefix_takeover_caps5_10_20260917/summary.json](../../reports/prefix_takeover_caps5_10_20260917/summary.json)、[results/westlake/opd_eval_prefix/prefix_takeover_caps5_10_20260917/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps5_10_20260917/summary.json)、[reports/prefix_takeover_caps5_10_20260917/summary.csv](../../reports/prefix_takeover_caps5_10_20260917/summary.csv)、[results/westlake/opd_eval_prefix/prefix_takeover_caps5_10_20260917/summary.csv](../../results/westlake/opd_eval_prefix/prefix_takeover_caps5_10_20260917/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=800，correct=391，accuracy=0.48875 | [reports/prefix_takeover_caps5_10_20260917/summary.json](../../reports/prefix_takeover_caps5_10_20260917/summary.json) `/groups/0` |
| dataset=bbh；n=800，correct=417，accuracy=0.52125 | [reports/prefix_takeover_caps5_10_20260917/summary.json](../../reports/prefix_takeover_caps5_10_20260917/summary.json) `/groups/1` |
| dataset=bbh；n=800，correct=431，accuracy=0.53875 | [reports/prefix_takeover_caps5_10_20260917/summary.json](../../reports/prefix_takeover_caps5_10_20260917/summary.json) `/groups/2` |
| dataset=bbh；n=800，correct=448，accuracy=0.56 | [reports/prefix_takeover_caps5_10_20260917/summary.json](../../reports/prefix_takeover_caps5_10_20260917/summary.json) `/groups/3` |
| dataset=bbh；n=800，correct=459，accuracy=0.57375 | [reports/prefix_takeover_caps5_10_20260917/summary.json](../../reports/prefix_takeover_caps5_10_20260917/summary.json) `/groups/4` |
| dataset=bbh；n=800，correct=499，accuracy=0.62375 | [reports/prefix_takeover_caps5_10_20260917/summary.json](../../reports/prefix_takeover_caps5_10_20260917/summary.json) `/groups/5` |
| dataset=cruxeval_i；n=800，correct=363，accuracy=0.45375 | [reports/prefix_takeover_caps5_10_20260917/summary.json](../../reports/prefix_takeover_caps5_10_20260917/summary.json) `/groups/6` |
| dataset=cruxeval_i；n=800，correct=377，accuracy=0.47125 | [reports/prefix_takeover_caps5_10_20260917/summary.json](../../reports/prefix_takeover_caps5_10_20260917/summary.json) `/groups/7` |

本源另有 40 条聚合记录；完整数值见 `result_records.csv.gz`。

### prefix_takeover_caps_20260917

- 证据状态：**已有结果汇总**；代码目录：[experiments/prefix_takeover_caps_20260917](../../experiments/prefix_takeover_caps_20260917)。
- 配置/范围入口：[experiments/prefix_takeover_caps_20260917/README.md](../../experiments/prefix_takeover_caps_20260917/README.md)。
- 已保存阶段完成标记：[reports/prefix_takeover_caps_20260917/ALL_GENERATION_COMPLETE.json](../../reports/prefix_takeover_caps_20260917/ALL_GENERATION_COMPLETE.json)、[reports/prefix_takeover_caps_20260917/ALL_SCORING_COMPLETE.json](../../reports/prefix_takeover_caps_20260917/ALL_SCORING_COMPLETE.json)、[reports/prefix_takeover_caps_20260917/INPUTS_COMPLETE.json](../../reports/prefix_takeover_caps_20260917/INPUTS_COMPLETE.json)。
- 结果入口：[reports/prefix_takeover_caps_20260917/summary.json](../../reports/prefix_takeover_caps_20260917/summary.json)、[results/westlake/opd_eval_prefix/prefix_takeover_caps_20260917/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps_20260917/summary.json)、[reports/prefix_caps_analysis_20260917/analysis.json](../../reports/prefix_caps_analysis_20260917/analysis.json)、[reports/prefix_caps_analysis_20260917/position_distribution.json](../../reports/prefix_caps_analysis_20260917/position_distribution.json)、[reports/prefix_takeover_caps_20260917/summary.csv](../../reports/prefix_takeover_caps_20260917/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=800，correct=344，accuracy=0.43 | [reports/prefix_takeover_caps_20260917/summary.json](../../reports/prefix_takeover_caps_20260917/summary.json) `/groups/0` |
| dataset=bbh；n=800，correct=340，accuracy=0.425 | [reports/prefix_takeover_caps_20260917/summary.json](../../reports/prefix_takeover_caps_20260917/summary.json) `/groups/1` |
| dataset=bbh；n=800，correct=358，accuracy=0.4475 | [reports/prefix_takeover_caps_20260917/summary.json](../../reports/prefix_takeover_caps_20260917/summary.json) `/groups/2` |
| dataset=bbh；n=800，correct=347，accuracy=0.43375 | [reports/prefix_takeover_caps_20260917/summary.json](../../reports/prefix_takeover_caps_20260917/summary.json) `/groups/3` |
| dataset=bbh；n=800，correct=377，accuracy=0.47125 | [reports/prefix_takeover_caps_20260917/summary.json](../../reports/prefix_takeover_caps_20260917/summary.json) `/groups/4` |
| dataset=cruxeval_i；n=800，correct=173，accuracy=0.21625 | [reports/prefix_takeover_caps_20260917/summary.json](../../reports/prefix_takeover_caps_20260917/summary.json) `/groups/5` |
| dataset=cruxeval_i；n=800，correct=68，accuracy=0.085 | [reports/prefix_takeover_caps_20260917/summary.json](../../reports/prefix_takeover_caps_20260917/summary.json) `/groups/6` |
| dataset=cruxeval_i；n=800，correct=153，accuracy=0.19125 | [reports/prefix_takeover_caps_20260917/summary.json](../../reports/prefix_takeover_caps_20260917/summary.json) `/groups/7` |

本源另有 32 条聚合记录；完整数值见 `result_records.csv.gz`。

### quartile_greedy_baselines_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/quartile_greedy_baselines_20260918](../../experiments/quartile_greedy_baselines_20260918)。
- 配置/范围入口：[experiments/quartile_greedy_baselines_20260918/LAUNCH.json](../../experiments/quartile_greedy_baselines_20260918/LAUNCH.json)、[experiments/quartile_greedy_baselines_20260918/README.md](../../experiments/quartile_greedy_baselines_20260918/README.md)。
- 已保存阶段完成标记：[reports/quartile_support_k_results_20260919/ALL_SEVEN_GROUPS_COMPLETE.json](../../reports/quartile_support_k_results_20260919/ALL_SEVEN_GROUPS_COMPLETE.json)。
- 结果入口：[reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json)、[reports/quartile_support_k_results_20260919/paired_changes_vs_k0.csv](../../reports/quartile_support_k_results_20260919/paired_changes_vs_k0.csv)、[reports/quartile_support_k_results_20260919/summary_with_baselines.csv](../../reports/quartile_support_k_results_20260919/summary_with_baselines.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh / group=k0 / quartile=1；n=25，correct=9，accuracy=0.36 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/0` |
| dataset=bbh / group=k0 / quartile=2；n=25，correct=10，accuracy=0.4 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/1` |
| dataset=bbh / group=k0 / quartile=3；n=25，correct=7，accuracy=0.28 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/2` |
| dataset=bbh / group=k0 / quartile=4；n=25，correct=16，accuracy=0.64 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/3` |
| dataset=bbh / group=k0 / quartile=all；n=100，correct=42，accuracy=0.42 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/4` |
| dataset=bbh / group=k1 / quartile=1；n=25，correct=6，accuracy=0.24 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/5` |
| dataset=bbh / group=k1 / quartile=2；n=25，correct=11，accuracy=0.44 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/6` |
| dataset=bbh / group=k1 / quartile=3；n=25，correct=9，accuracy=0.36 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/7` |

本源另有 272 条聚合记录；完整数值见 `result_records.csv.gz`。

### quartile_support_k_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/quartile_support_k_20260918](../../experiments/quartile_support_k_20260918)。
- 配置/范围入口：[experiments/quartile_support_k_20260918/LAUNCH.json](../../experiments/quartile_support_k_20260918/LAUNCH.json)、[experiments/quartile_support_k_20260918/README.md](../../experiments/quartile_support_k_20260918/README.md)。
- 已保存阶段完成标记：[reports/quartile_support_k_results_20260919/ALL_SEVEN_GROUPS_COMPLETE.json](../../reports/quartile_support_k_results_20260919/ALL_SEVEN_GROUPS_COMPLETE.json)。
- 结果入口：[reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json)、[results/westlake/opd_eval_prefix/quartile_support_k_20260918/summary.json](../../results/westlake/opd_eval_prefix/quartile_support_k_20260918/summary.json)、[results/westlake/opd_eval_prefix/quartile_support_k_20260918/summary_with_baselines.json](../../results/westlake/opd_eval_prefix/quartile_support_k_20260918/summary_with_baselines.json)、[results/westlake/opd_eval_prefix/quartile_support_k_20260918/k0_summary.json](../../results/westlake/opd_eval_prefix/quartile_support_k_20260918/k0_summary.json)、[results/westlake/opd_eval_prefix/quartile_support_k_20260918/teacher_greedy_summary.json](../../results/westlake/opd_eval_prefix/quartile_support_k_20260918/teacher_greedy_summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh / group=k0 / quartile=1；n=25，correct=9，accuracy=0.36 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/0` |
| dataset=bbh / group=k0 / quartile=2；n=25，correct=10，accuracy=0.4 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/1` |
| dataset=bbh / group=k0 / quartile=3；n=25，correct=7，accuracy=0.28 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/2` |
| dataset=bbh / group=k0 / quartile=4；n=25，correct=16，accuracy=0.64 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/3` |
| dataset=bbh / group=k0 / quartile=all；n=100，correct=42，accuracy=0.42 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/4` |
| dataset=bbh / group=k1 / quartile=1；n=25，correct=6，accuracy=0.24 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/5` |
| dataset=bbh / group=k1 / quartile=2；n=25，correct=11，accuracy=0.44 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/6` |
| dataset=bbh / group=k1 / quartile=3；n=25，correct=9，accuracy=0.36 | [reports/quartile_support_k_results_20260919/summary_with_baselines.json](../../reports/quartile_support_k_results_20260919/summary_with_baselines.json) `/groups/7` |

本源另有 272 条聚合记录；完整数值见 `result_records.csv.gz`。

### qwen3_greedy_hard_three_20261005

- 证据状态：**已有结果汇总**；代码目录：[experiments/qwen3_greedy_hard_three_20261005](../../experiments/qwen3_greedy_hard_three_20261005)。
- 配置/范围入口：[experiments/qwen3_greedy_hard_three_20261005/README.md](../../experiments/qwen3_greedy_hard_three_20261005/README.md)。
- 结果入口：[results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_4b/summary.json)、[results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_8b/summary.json](../../results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_8b/summary.json)、[reports/qwen3_1p7b_4b_8b_14b_all_results_20261005.json](../../reports/qwen3_1p7b_4b_8b_14b_all_results_20261005.json)、[experiments/qwen3_greedy_hard_three_20261005/results_summary.json](../../experiments/qwen3_greedy_hard_three_20261005/results_summary.json)、[experiments/qwen3_greedy_hard_three_20261005/qwen3_4b_summary.json](../../experiments/qwen3_greedy_hard_three_20261005/qwen3_4b_summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=aime24；questions=30，correct=5，accuracy=0.166667 | [results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_4b/summary.json) `/aime24` |
| dataset=aime25；questions=30，correct=4，accuracy=0.133333 | [results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_4b/summary.json) `/aime25` |
| dataset=bbh；questions=1080，correct=732，accuracy=0.677778 | [results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_4b/summary.json) `/bbh` |
| dataset=olympiad_bench；questions=674，correct=309，accuracy=0.458457 | [results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_4b/summary.json) `/olympiad_bench` |

### qwen3_greedy_remaining_20261005

- 证据状态：**已有结果汇总**；代码目录：[experiments/qwen3_greedy_remaining_20261005](../../experiments/qwen3_greedy_remaining_20261005)。
- 配置/范围入口：[experiments/qwen3_greedy_remaining_20261005/README.md](../../experiments/qwen3_greedy_remaining_20261005/README.md)。
- 结果入口：[experiments/qwen3_greedy_remaining_20261005/full_results_summary.json](../../experiments/qwen3_greedy_remaining_20261005/full_results_summary.json)、[results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json)、[results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_8b/summary.json](../../results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_8b/summary.json)、[reports/qwen3_1p7b_4b_8b_14b_all_results_20261005.json](../../reports/qwen3_1p7b_4b_8b_14b_all_results_20261005.json)、[experiments/qwen3_greedy_remaining_20261005/remote_results/qwen3_4b_summary.json](../../experiments/qwen3_greedy_remaining_20261005/remote_results/qwen3_4b_summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=math500；questions=500，correct=412，accuracy=0.824，mean_tokens=2307.85 | [results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json) `/math500` |
| dataset=cruxeval_o；questions=800，correct=297，accuracy=0.37125，mean_tokens=65.5738 | [results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json) `/cruxeval_o` |
| dataset=cruxeval_i；questions=800，correct=139，accuracy=0.17375，mean_tokens=216.776 | [results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json) `/cruxeval_i` |
| dataset=humaneval_plus；questions=164，correct=121，accuracy=0.737805，mean_tokens=165.689 | [results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json) `/humaneval_plus` |
| dataset=mbpp_plus；questions=378，correct=227，accuracy=0.600529，mean_tokens=104.955 | [results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json) `/mbpp_plus` |
| dataset=livecodebench；questions=175，correct=47，accuracy=0.268571，mean_tokens=5871.43 | [results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json) `/livecodebench` |

### qwen3vl_greedy_bench_20261016

- 证据状态：**仅代码/状态文档**；代码目录：[experiments/qwen3vl_greedy_bench_20261016](../../experiments/qwen3vl_greedy_bench_20261016)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### r1_math_score_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/r1_math_score_20260918](../../experiments/r1_math_score_20260918)。
- 配置/范围入口：[experiments/r1_math_score_20260918/LAUNCH.json](../../experiments/r1_math_score_20260918/LAUNCH.json)。
- 结果入口：[results/westlake/opd_rescore/r1_math_score_20260918/summary.json](../../results/westlake/opd_rescore/r1_math_score_20260918/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=864，accuracy=0.8 | [results/westlake/opd_rescore/r1_math_score_20260918/summary.json](../../results/westlake/opd_rescore/r1_math_score_20260918/summary.json) `/groups/0` |
| dataset=math500；n=500，correct=415，accuracy=0.83 | [results/westlake/opd_rescore/r1_math_score_20260918/summary.json](../../results/westlake/opd_rescore/r1_math_score_20260918/summary.json) `/groups/1` |
| dataset=olympiad_bench；n=674，correct=365，accuracy=0.541543 | [results/westlake/opd_rescore/r1_math_score_20260918/summary.json](../../results/westlake/opd_rescore/r1_math_score_20260918/summary.json) `/groups/2` |

### random_k8_max8_20260911

- 证据状态：**已有结果汇总**；代码目录：[experiments/random_k8_max8_20260911](../../experiments/random_k8_max8_20260911)。
- 配置/范围入口：[experiments/random_k8_max8_20260911/STATUS.md](../../experiments/random_k8_max8_20260911/STATUS.md)。
- 已保存阶段完成标记：[experiments/random_k8_max8_20260911/artifacts/ALL_GENERATION_COMPLETE.json](../../experiments/random_k8_max8_20260911/artifacts/ALL_GENERATION_COMPLETE.json)。
- 结果入口：[experiments/random_k8_max8_20260911/artifacts/summary.json](../../experiments/random_k8_max8_20260911/artifacts/summary.json)、[results/westlake/opd_eval_hard/random_k8_max8_20260911/summary.json](../../results/westlake/opd_eval_hard/random_k8_max8_20260911/summary.json)、[experiments/random_k8_max8_20260911/audit.json](../../experiments/random_k8_max8_20260911/audit.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| method=online / dataset=aime25；questions=30，correct=5，accuracy=0.166667 | [experiments/random_k8_max8_20260911/artifacts/summary.json](../../experiments/random_k8_max8_20260911/artifacts/summary.json) `/online/aime25` |
| method=online / dataset=aime24；questions=30，correct=3，accuracy=0.1 | [experiments/random_k8_max8_20260911/artifacts/summary.json](../../experiments/random_k8_max8_20260911/artifacts/summary.json) `/online/aime24` |
| method=online / dataset=olympiad_bench；questions=674，correct=127，accuracy=0.188427 | [experiments/random_k8_max8_20260911/artifacts/summary.json](../../experiments/random_k8_max8_20260911/artifacts/summary.json) `/online/olympiad_bench` |
| method=online / dataset=bbh；questions=1080，correct=385，accuracy=0.356481 | [experiments/random_k8_max8_20260911/artifacts/summary.json](../../experiments/random_k8_max8_20260911/artifacts/summary.json) `/online/bbh` |
| method=online / dataset=math500；questions=500，correct=290，accuracy=0.58 | [experiments/random_k8_max8_20260911/artifacts/summary.json](../../experiments/random_k8_max8_20260911/artifacts/summary.json) `/online/math500` |

### random_quartile_greedy_k1_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/random_quartile_greedy_k1_20260921](../../experiments/random_quartile_greedy_k1_20260921)。
- 结果入口：[experiments/random_quartile_greedy_k1_20260921/results/summary.json](../../experiments/random_quartile_greedy_k1_20260921/results/summary.json)、[results/westlake/opd_eval_ablation/random_quartile_greedy_k1_20260921/summary.json](../../results/westlake/opd_eval_ablation/random_quartile_greedy_k1_20260921/summary.json)、[experiments/random_quartile_greedy_k1_20260921/results/baseline_greedy_summary.json](../../experiments/random_quartile_greedy_k1_20260921/results/baseline_greedy_summary.json)、[experiments/random_quartile_greedy_k1_20260921/results/summary.csv](../../experiments/random_quartile_greedy_k1_20260921/results/summary.csv)、[results/westlake/opd_eval_ablation/random_quartile_greedy_k1_20260921/summary.csv](../../results/westlake/opd_eval_ablation/random_quartile_greedy_k1_20260921/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=421，accuracy=0.389815 | [experiments/random_quartile_greedy_k1_20260921/results/summary.json](../../experiments/random_quartile_greedy_k1_20260921/results/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=72，accuracy=0.09 | [experiments/random_quartile_greedy_k1_20260921/results/summary.json](../../experiments/random_quartile_greedy_k1_20260921/results/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=196，accuracy=0.245 | [experiments/random_quartile_greedy_k1_20260921/results/summary.json](../../experiments/random_quartile_greedy_k1_20260921/results/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=51，accuracy=0.310976 | [experiments/random_quartile_greedy_k1_20260921/results/summary.json](../../experiments/random_quartile_greedy_k1_20260921/results/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=18，accuracy=0.102857 | [experiments/random_quartile_greedy_k1_20260921/results/summary.json](../../experiments/random_quartile_greedy_k1_20260921/results/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=283，accuracy=0.566 | [experiments/random_quartile_greedy_k1_20260921/results/summary.json](../../experiments/random_quartile_greedy_k1_20260921/results/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=87，accuracy=0.230159 | [experiments/random_quartile_greedy_k1_20260921/results/summary.json](../../experiments/random_quartile_greedy_k1_20260921/results/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=228，accuracy=0.338279 | [experiments/random_quartile_greedy_k1_20260921/results/summary.json](../../experiments/random_quartile_greedy_k1_20260921/results/summary.json) `/groups/7` |

### random_quartile_k1_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/random_quartile_k1_20260918](../../experiments/random_quartile_k1_20260918)。
- 配置/范围入口：[experiments/random_quartile_k1_20260918/LAUNCH.json](../../experiments/random_quartile_k1_20260918/LAUNCH.json)、[experiments/random_quartile_k1_20260918/README.md](../../experiments/random_quartile_k1_20260918/README.md)。
- 结果入口：[reports/random_quartile_comparison_20260918/summary.json](../../reports/random_quartile_comparison_20260918/summary.json)、[results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json)、[reports/random_quartile_comparison_20260918/r1_summary.json](../../reports/random_quartile_comparison_20260918/r1_summary.json)、[reports/random_quartile_comparison_20260918/r1_summary_with_teacher.json](../../reports/random_quartile_comparison_20260918/r1_summary_with_teacher.json)、[results/westlake/opd_eval_ablation/random_quartile_k1_20260918/code_summary.json](../../results/westlake/opd_eval_ablation/random_quartile_k1_20260918/code_summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=440，accuracy=0.407407 | [results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=83，accuracy=0.10375 | [results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=196，accuracy=0.245 | [results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=65，accuracy=0.396341 | [results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=9，accuracy=0.0514286 | [results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=306，accuracy=0.612 | [results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=101，accuracy=0.267196 | [results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=214，accuracy=0.317507 | [results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.json) `/groups/7` |

### reverse_greedy_score_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/reverse_greedy_score_20260921](../../experiments/reverse_greedy_score_20260921)。
- 配置/范围入口：[experiments/reverse_greedy_score_20260921/LAUNCH.json](../../experiments/reverse_greedy_score_20260921/LAUNCH.json)、[experiments/reverse_greedy_score_20260921/README.md](../../experiments/reverse_greedy_score_20260921/README.md)。
- 结果入口：[reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=470，accuracy=0.435185 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=192，accuracy=0.24 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=199，accuracy=0.24875 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=108，accuracy=0.658537 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=31，accuracy=0.177143 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=384，accuracy=0.768 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=219，accuracy=0.579365 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=339，accuracy=0.502967 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/7` |

### reverse_greedy_t2_k1_8gpu_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/reverse_greedy_t2_k1_8gpu_20260921](../../experiments/reverse_greedy_t2_k1_8gpu_20260921)。
- 配置/范围入口：[experiments/reverse_greedy_t2_k1_8gpu_20260921/LAUNCH.json](../../experiments/reverse_greedy_t2_k1_8gpu_20260921/LAUNCH.json)、[experiments/reverse_greedy_t2_k1_8gpu_20260921/README.md](../../experiments/reverse_greedy_t2_k1_8gpu_20260921/README.md)。
- 结果入口：[reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json)、[results/westlake/opd_eval_ablation/reverse_greedy_t2_k1_8gpu_20260921/summary.json](../../results/westlake/opd_eval_ablation/reverse_greedy_t2_k1_8gpu_20260921/summary.json)、[results/westlake/opd_eval_ablation/reverse_greedy_t2_k1_8gpu_20260921/summary.csv](../../results/westlake/opd_eval_ablation/reverse_greedy_t2_k1_8gpu_20260921/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=470，accuracy=0.435185 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=192，accuracy=0.24 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=199，accuracy=0.24875 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=108，accuracy=0.658537 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=31，accuracy=0.177143 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=384，accuracy=0.768 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=219，accuracy=0.579365 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=339，accuracy=0.502967 | [reports/reverse_greedy_results_20260921/summary.json](../../reports/reverse_greedy_results_20260921/summary.json) `/groups/7` |

### reverse_t2_k1_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/reverse_t2_k1_20260918](../../experiments/reverse_t2_k1_20260918)。
- 配置/范围入口：[experiments/reverse_t2_k1_20260918/LAUNCH.json](../../experiments/reverse_t2_k1_20260918/LAUNCH.json)、[experiments/reverse_t2_k1_20260918/README.md](../../experiments/reverse_t2_k1_20260918/README.md)。
- 结果入口：[results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json)、[reports/reverse_t2_k1_baseline_comparison_20260918.json](../../reports/reverse_t2_k1_baseline_comparison_20260918.json)、[results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/code_summary.json](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/code_summary.json)、[results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.csv](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=459，accuracy=0.425 | [results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=186，accuracy=0.2325 | [results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=188，accuracy=0.235 | [results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=104，accuracy=0.634146 | [results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=30，accuracy=0.171429 | [results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=382，accuracy=0.764 | [results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=217，accuracy=0.574074 | [results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=318，accuracy=0.47181 | [results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.json) `/groups/7` |

### reverse_t2_k1_score_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/reverse_t2_k1_score_20260918](../../experiments/reverse_t2_k1_score_20260918)。
- 配置/范围入口：[experiments/reverse_t2_k1_score_20260918/LAUNCH.json](../../experiments/reverse_t2_k1_score_20260918/LAUNCH.json)、[experiments/reverse_t2_k1_score_20260918/README.md](../../experiments/reverse_t2_k1_score_20260918/README.md)。
- 结果入口：[reports/reverse_t2_k1_baseline_comparison_20260918.json](../../reports/reverse_t2_k1_baseline_comparison_20260918.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=459，accuracy=0.425 | [reports/reverse_t2_k1_baseline_comparison_20260918.json](../../reports/reverse_t2_k1_baseline_comparison_20260918.json) `/reverse_summary/groups/0` |
| dataset=cruxeval_i；n=800，correct=186，accuracy=0.2325 | [reports/reverse_t2_k1_baseline_comparison_20260918.json](../../reports/reverse_t2_k1_baseline_comparison_20260918.json) `/reverse_summary/groups/1` |
| dataset=cruxeval_o；n=800，correct=188，accuracy=0.235 | [reports/reverse_t2_k1_baseline_comparison_20260918.json](../../reports/reverse_t2_k1_baseline_comparison_20260918.json) `/reverse_summary/groups/2` |
| dataset=humaneval_plus；n=164，correct=104，accuracy=0.634146 | [reports/reverse_t2_k1_baseline_comparison_20260918.json](../../reports/reverse_t2_k1_baseline_comparison_20260918.json) `/reverse_summary/groups/3` |
| dataset=livecodebench；n=175，correct=30，accuracy=0.171429 | [reports/reverse_t2_k1_baseline_comparison_20260918.json](../../reports/reverse_t2_k1_baseline_comparison_20260918.json) `/reverse_summary/groups/4` |
| dataset=math500；n=500，correct=382，accuracy=0.764 | [reports/reverse_t2_k1_baseline_comparison_20260918.json](../../reports/reverse_t2_k1_baseline_comparison_20260918.json) `/reverse_summary/groups/5` |
| dataset=mbpp_plus；n=378，correct=217，accuracy=0.574074 | [reports/reverse_t2_k1_baseline_comparison_20260918.json](../../reports/reverse_t2_k1_baseline_comparison_20260918.json) `/reverse_summary/groups/6` |
| dataset=olympiad_bench；n=674，correct=318，accuracy=0.47181 | [reports/reverse_t2_k1_baseline_comparison_20260918.json](../../reports/reverse_t2_k1_baseline_comparison_20260918.json) `/reverse_summary/groups/7` |

### shift5_greedy_k1_t2_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/shift5_greedy_k1_t2_20260921](../../experiments/shift5_greedy_k1_t2_20260921)。
- 配置/范围入口：[experiments/shift5_greedy_k1_t2_20260921/LAUNCH.json](../../experiments/shift5_greedy_k1_t2_20260921/LAUNCH.json)、[experiments/shift5_greedy_k1_t2_20260921/README.md](../../experiments/shift5_greedy_k1_t2_20260921/README.md)。
- 结果入口：[reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json)、[results/westlake/opd_eval_ablation/shift5_greedy_k1_t2_20260921/summary.json](../../results/westlake/opd_eval_ablation/shift5_greedy_k1_t2_20260921/summary.json)、[reports/shift5_greedy_score_20260921/summary.csv](../../reports/shift5_greedy_score_20260921/summary.csv)、[reports/shift5_greedy_score_20260921/comparison_greedy_baselines.csv](../../reports/shift5_greedy_score_20260921/comparison_greedy_baselines.csv)、[results/westlake/opd_eval_ablation/shift5_greedy_k1_t2_20260921/summary.csv](../../results/westlake/opd_eval_ablation/shift5_greedy_k1_t2_20260921/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=234，accuracy=0.216667 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=104，accuracy=0.13 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=133，accuracy=0.16625 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=32，accuracy=0.195122 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=14，accuracy=0.08 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=326，accuracy=0.652 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=85，accuracy=0.224868 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=256，accuracy=0.379822 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/7` |

### shift5_greedy_score_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/shift5_greedy_score_20260921](../../experiments/shift5_greedy_score_20260921)。
- 配置/范围入口：[experiments/shift5_greedy_score_20260921/LAUNCH.json](../../experiments/shift5_greedy_score_20260921/LAUNCH.json)。
- 结果入口：[reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json)、[reports/shift5_greedy_score_20260921/summary.csv](../../reports/shift5_greedy_score_20260921/summary.csv)、[reports/shift5_greedy_score_20260921/comparison_greedy_baselines.csv](../../reports/shift5_greedy_score_20260921/comparison_greedy_baselines.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=234，accuracy=0.216667 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=104，accuracy=0.13 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=133，accuracy=0.16625 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=32，accuracy=0.195122 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=14，accuracy=0.08 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=326，accuracy=0.652 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=85，accuracy=0.224868 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=256，accuracy=0.379822 | [reports/shift5_greedy_score_20260921/summary.json](../../reports/shift5_greedy_score_20260921/summary.json) `/groups/7` |

### shift5_k1_t2_20260918

- 证据状态：**已有结果汇总**；代码目录：[experiments/shift5_k1_t2_20260918](../../experiments/shift5_k1_t2_20260918)。
- 配置/范围入口：[experiments/shift5_k1_t2_20260918/LAUNCH.json](../../experiments/shift5_k1_t2_20260918/LAUNCH.json)、[experiments/shift5_k1_t2_20260918/README.md](../../experiments/shift5_k1_t2_20260918/README.md)。
- 结果入口：[reports/shift5_comparison_20260919/summary.json](../../reports/shift5_comparison_20260919/summary.json)、[results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json)、[results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/code_summary.json](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/code_summary.json)、[results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.csv](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=284，accuracy=0.262963 | [results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=97，accuracy=0.12125 | [results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=140，accuracy=0.175 | [results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=53，accuracy=0.323171 | [results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=10，accuracy=0.0571429 | [results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=336，accuracy=0.672 | [results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=77，accuracy=0.203704 | [results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=270，accuracy=0.400593 | [results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.json) `/groups/7` |

### signed_gap_audit_20260911

- 证据状态：**仅代码/状态文档**；代码目录：[experiments/signed_gap_audit_20260911](../../experiments/signed_gap_audit_20260911)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### strong4_weak17_reprompt_greedy_t2_k1_20261005

- 证据状态：**已有结果汇总**；代码目录：[experiments/strong4_weak17_reprompt_greedy_t2_k1_20261005](../../experiments/strong4_weak17_reprompt_greedy_t2_k1_20261005)。
- 配置/范围入口：[experiments/strong4_weak17_reprompt_greedy_t2_k1_20261005/LAUNCH.json](../../experiments/strong4_weak17_reprompt_greedy_t2_k1_20261005/LAUNCH.json)、[experiments/strong4_weak17_reprompt_greedy_t2_k1_20261005/README.md](../../experiments/strong4_weak17_reprompt_greedy_t2_k1_20261005/README.md)、[experiments/strong4_weak17_reprompt_greedy_t2_k1_20261005/configs.json](../../experiments/strong4_weak17_reprompt_greedy_t2_k1_20261005/configs.json)。
- 已保存阶段完成标记：[experiments/strong4_weak17_reprompt_greedy_t2_k1_20261005/INPUTS_COMPLETE.json](../../experiments/strong4_weak17_reprompt_greedy_t2_k1_20261005/INPUTS_COMPLETE.json)、[experiments/strong_reprompt_score_20261005/results/ALL_SCORING_COMPLETE.json](../../experiments/strong_reprompt_score_20261005/results/ALL_SCORING_COMPLETE.json)。
- 结果入口：[reports/strong_main_logit_analysis_20261006/summary.json](../../reports/strong_main_logit_analysis_20261006/summary.json)、[reports/strong_reprompt_vs_greedy_20261005/comparison.json](../../reports/strong_reprompt_vs_greedy_20261005/comparison.json)、[experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json)、[reports/strong_reprompt_vs_greedy_20261005/comparison.csv](../../reports/strong_reprompt_vs_greedy_20261005/comparison.csv)、[experiments/strong_reprompt_score_20261005/results/summary.csv](../../experiments/strong_reprompt_score_20261005/results/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=aime24 / experiment=strong4_weak17；n=30，correct=5，accuracy=0.166667 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/0` |
| dataset=aime25 / experiment=strong4_weak17；n=30，correct=4，accuracy=0.133333 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/1` |
| dataset=bbh / experiment=strong4_weak17；n=1080，correct=658，accuracy=0.609259 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/2` |
| dataset=cruxeval_i / experiment=strong4_weak17；n=800，correct=62，accuracy=0.0775 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/3` |
| dataset=cruxeval_o / experiment=strong4_weak17；n=800，correct=172，accuracy=0.215 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/4` |
| dataset=humaneval_plus / experiment=strong4_weak17；n=164，correct=112，accuracy=0.682927 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/5` |
| dataset=livecodebench / experiment=strong4_weak17；n=175，correct=39，accuracy=0.222857 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/6` |
| dataset=math500 / experiment=strong4_weak17；n=500，correct=384，accuracy=0.768 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/7` |

本源另有 12 条聚合记录；完整数值见 `result_records.csv.gz`。

### strong8_weak17_reprompt_greedy_t2_k1_20261005

- 证据状态：**已有结果汇总**；代码目录：[experiments/strong8_weak17_reprompt_greedy_t2_k1_20261005](../../experiments/strong8_weak17_reprompt_greedy_t2_k1_20261005)。
- 配置/范围入口：[experiments/strong8_weak17_reprompt_greedy_t2_k1_20261005/LAUNCH.json](../../experiments/strong8_weak17_reprompt_greedy_t2_k1_20261005/LAUNCH.json)、[experiments/strong8_weak17_reprompt_greedy_t2_k1_20261005/README.md](../../experiments/strong8_weak17_reprompt_greedy_t2_k1_20261005/README.md)、[experiments/strong8_weak17_reprompt_greedy_t2_k1_20261005/configs.json](../../experiments/strong8_weak17_reprompt_greedy_t2_k1_20261005/configs.json)。
- 已保存阶段完成标记：[experiments/strong8_weak17_reprompt_greedy_t2_k1_20261005/INPUTS_COMPLETE.json](../../experiments/strong8_weak17_reprompt_greedy_t2_k1_20261005/INPUTS_COMPLETE.json)、[experiments/strong_reprompt_score_20261005/results/ALL_SCORING_COMPLETE.json](../../experiments/strong_reprompt_score_20261005/results/ALL_SCORING_COMPLETE.json)。
- 结果入口：[reports/strong_main_logit_analysis_20261006/summary.json](../../reports/strong_main_logit_analysis_20261006/summary.json)、[reports/strong_reprompt_vs_greedy_20261005/comparison.json](../../reports/strong_reprompt_vs_greedy_20261005/comparison.json)、[experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json)、[reports/strong_reprompt_vs_greedy_20261005/comparison.csv](../../reports/strong_reprompt_vs_greedy_20261005/comparison.csv)、[experiments/strong_reprompt_score_20261005/results/summary.csv](../../experiments/strong_reprompt_score_20261005/results/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=aime24 / experiment=strong4_weak17；n=30，correct=5，accuracy=0.166667 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/0` |
| dataset=aime25 / experiment=strong4_weak17；n=30，correct=4，accuracy=0.133333 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/1` |
| dataset=bbh / experiment=strong4_weak17；n=1080，correct=658，accuracy=0.609259 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/2` |
| dataset=cruxeval_i / experiment=strong4_weak17；n=800，correct=62，accuracy=0.0775 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/3` |
| dataset=cruxeval_o / experiment=strong4_weak17；n=800，correct=172，accuracy=0.215 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/4` |
| dataset=humaneval_plus / experiment=strong4_weak17；n=164，correct=112，accuracy=0.682927 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/5` |
| dataset=livecodebench / experiment=strong4_weak17；n=175，correct=39，accuracy=0.222857 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/6` |
| dataset=math500 / experiment=strong4_weak17；n=500，correct=384，accuracy=0.768 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/7` |

本源另有 12 条聚合记录；完整数值见 `result_records.csv.gz`。

### strong_reprompt_score_20261005

- 证据状态：**已有结果汇总**；代码目录：[experiments/strong_reprompt_score_20261005](../../experiments/strong_reprompt_score_20261005)。
- 已保存阶段完成标记：[experiments/strong_reprompt_score_20261005/results/ALL_SCORING_COMPLETE.json](../../experiments/strong_reprompt_score_20261005/results/ALL_SCORING_COMPLETE.json)。
- 结果入口：[experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json)、[results/westlake/opd_eval_ablation/strong_reprompt_score_20261005/summary.json](../../results/westlake/opd_eval_ablation/strong_reprompt_score_20261005/summary.json)、[experiments/strong_reprompt_score_20261005/results/summary.csv](../../experiments/strong_reprompt_score_20261005/results/summary.csv)、[results/westlake/opd_eval_ablation/strong_reprompt_score_20261005/summary.csv](../../results/westlake/opd_eval_ablation/strong_reprompt_score_20261005/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=aime24 / experiment=strong4_weak17；n=30，correct=5，accuracy=0.166667 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/0` |
| dataset=aime25 / experiment=strong4_weak17；n=30，correct=4，accuracy=0.133333 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/1` |
| dataset=bbh / experiment=strong4_weak17；n=1080，correct=658，accuracy=0.609259 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/2` |
| dataset=cruxeval_i / experiment=strong4_weak17；n=800，correct=62，accuracy=0.0775 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/3` |
| dataset=cruxeval_o / experiment=strong4_weak17；n=800，correct=172，accuracy=0.215 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/4` |
| dataset=humaneval_plus / experiment=strong4_weak17；n=164，correct=112，accuracy=0.682927 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/5` |
| dataset=livecodebench / experiment=strong4_weak17；n=175，correct=39，accuracy=0.222857 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/6` |
| dataset=math500 / experiment=strong4_weak17；n=500，correct=384，accuracy=0.768 | [experiments/strong_reprompt_score_20261005/results/summary.json](../../experiments/strong_reprompt_score_20261005/results/summary.json) `/groups/7` |

本源另有 12 条聚合记录；完整数值见 `result_records.csv.gz`。

### student_only64_20260910

- 证据状态：**已有结果汇总**；代码目录：[experiments/student_only64_20260910](../../experiments/student_only64_20260910)。
- 配置/范围入口：[experiments/student_only64_20260910/STATUS.md](../../experiments/student_only64_20260910/STATUS.md)、[experiments/student_only64_20260910/config.json](../../experiments/student_only64_20260910/config.json)。
- 已保存阶段完成标记：[experiments/student_only64_20260910/COMPLETE.json](../../experiments/student_only64_20260910/COMPLETE.json)。
- 结果入口：[experiments/student_only64_20260910/summary.json](../../experiments/student_only64_20260910/summary.json)、[results/westlake/opd_eval_deepmath/student_only64_20260910/summary.json](../../results/westlake/opd_eval_deepmath/student_only64_20260910/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| questions=1000，rollouts=64000，correct=17914，accuracy=0.279906，pass_at_64=0.626，mean_tokens=1321.06 | [experiments/student_only64_20260910/summary.json](../../experiments/student_only64_20260910/summary.json) `/` |

### takeover_ablation_20260911

- 证据状态：**仅代码/状态文档**；代码目录：[experiments/takeover_ablation_20260911](../../experiments/takeover_ablation_20260911)。
- 配置/范围入口：[experiments/takeover_ablation_20260911/STATUS.md](../../experiments/takeover_ablation_20260911/STATUS.md)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### teacher8_student8_20260909

- 证据状态：**已有结果汇总**；代码目录：[experiments/teacher8_student8_20260909](../../experiments/teacher8_student8_20260909)。
- 配置/范围入口：[experiments/teacher8_student8_20260909/STATUS.md](../../experiments/teacher8_student8_20260909/STATUS.md)。
- 已保存阶段完成标记：[experiments/teacher8_student8_20260909/SCORING_COMPLETE.json](../../experiments/teacher8_student8_20260909/SCORING_COMPLETE.json)。
- 结果入口：[experiments/teacher8_student8_20260909/summary.json](../../experiments/teacher8_student8_20260909/summary.json)、[results/westlake/opd_eval_deepmath/teacher8_student8_20260909/summary.json](../../results/westlake/opd_eval_deepmath/teacher8_student8_20260909/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| method=teacher；rows=8000，correct=3399，accuracy=0.424875，pass_any=0.615 | [experiments/teacher8_student8_20260909/summary.json](../../experiments/teacher8_student8_20260909/summary.json) `/teacher` |
| method=student；rows=64000，correct=18109，accuracy=0.282953，pass_any=0.512875 | [experiments/teacher8_student8_20260909/summary.json](../../experiments/teacher8_student8_20260909/summary.json) `/student/32/conditional_on_teacher_prefix` |
| method=student；rows=64000，correct=18109，accuracy=0.282953，pass_any=0.641 | [experiments/teacher8_student8_20260909/summary.json](../../experiments/teacher8_student8_20260909/summary.json) `/student/32/per_question_across_8_teacher_prefixes` |
| method=student；rows=64000，correct=18109，accuracy=0.282953，pass_any=0.512875 | [experiments/teacher8_student8_20260909/summary.json](../../experiments/teacher8_student8_20260909/summary.json) `/student/32/active_student_prefixes` |
| method=student；rows=64000，correct=18215，accuracy=0.284609，pass_any=0.511875 | [experiments/teacher8_student8_20260909/summary.json](../../experiments/teacher8_student8_20260909/summary.json) `/student/64/conditional_on_teacher_prefix` |
| method=student；rows=64000，correct=18215，accuracy=0.284609，pass_any=0.648 | [experiments/teacher8_student8_20260909/summary.json](../../experiments/teacher8_student8_20260909/summary.json) `/student/64/per_question_across_8_teacher_prefixes` |
| method=student；rows=64000，correct=18215，accuracy=0.284609，pass_any=0.511875 | [experiments/teacher8_student8_20260909/summary.json](../../experiments/teacher8_student8_20260909/summary.json) `/student/64/active_student_prefixes` |
| method=student；rows=64000，correct=18292，accuracy=0.285813，pass_any=0.5105 | [experiments/teacher8_student8_20260909/summary.json](../../experiments/teacher8_student8_20260909/summary.json) `/student/128/conditional_on_teacher_prefix` |

本源另有 5 条聚合记录；完整数值见 `result_records.csv.gz`。

### teacher_only_sample100_x8_20260917

- 证据状态：**已有结果汇总**；代码目录：[experiments/teacher_only_sample100_x8_20260917](../../experiments/teacher_only_sample100_x8_20260917)。
- 配置/范围入口：[experiments/teacher_only_sample100_x8_20260917/LAUNCH.json](../../experiments/teacher_only_sample100_x8_20260917/LAUNCH.json)、[experiments/teacher_only_sample100_x8_20260917/README.md](../../experiments/teacher_only_sample100_x8_20260917/README.md)。
- 结果入口：[results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json](../../results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json)、[results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.csv](../../results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| condition=teacher_only / dataset=bbh；n=800，correct=617，accuracy=0.77125 | [results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json](../../results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json) `/groups/0` |
| condition=teacher_only / dataset=cruxeval_i；n=800，correct=400，accuracy=0.5 | [results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json](../../results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json) `/groups/1` |
| condition=teacher_only / dataset=cruxeval_o；n=800，correct=359，accuracy=0.44875 | [results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json](../../results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json) `/groups/2` |
| condition=teacher_only / dataset=humaneval_plus；n=800，correct=650，accuracy=0.8125 | [results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json](../../results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json) `/groups/3` |
| condition=teacher_only / dataset=livecodebench；n=800，correct=248，accuracy=0.31 | [results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json](../../results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json) `/groups/4` |
| condition=teacher_only / dataset=math500；n=800，correct=702，accuracy=0.8775 | [results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json](../../results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json) `/groups/5` |
| condition=teacher_only / dataset=mbpp_plus；n=800，correct=580，accuracy=0.725 | [results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json](../../results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json) `/groups/6` |
| condition=teacher_only / dataset=olympiad_bench；n=800，correct=508，accuracy=0.635 | [results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json](../../results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json) `/groups/7` |

### teacher_prefix_handoff_20260909

- 证据状态：**已有结果汇总**；代码目录：[experiments/teacher_prefix_handoff_20260909](../../experiments/teacher_prefix_handoff_20260909)。
- 配置/范围入口：[experiments/teacher_prefix_handoff_20260909/STATUS.md](../../experiments/teacher_prefix_handoff_20260909/STATUS.md)。
- 结果入口：[experiments/teacher_prefix_handoff_20260909/summary.json](../../experiments/teacher_prefix_handoff_20260909/summary.json)、[results/westlake/opd_eval_deepmath/teacher_prefix_handoff_20260909/summary.json](../../results/westlake/opd_eval_deepmath/teacher_prefix_handoff_20260909/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| h=32；n=1000，correct=293，accuracy=0.293 | [experiments/teacher_prefix_handoff_20260909/summary.json](../../experiments/teacher_prefix_handoff_20260909/summary.json) `/0` |
| h=64；n=1000，correct=283，accuracy=0.283 | [experiments/teacher_prefix_handoff_20260909/summary.json](../../experiments/teacher_prefix_handoff_20260909/summary.json) `/1` |
| h=128；n=1000，correct=293，accuracy=0.293 | [experiments/teacher_prefix_handoff_20260909/summary.json](../../experiments/teacher_prefix_handoff_20260909/summary.json) `/2` |
| h=256；n=1000，correct=276，accuracy=0.276 | [experiments/teacher_prefix_handoff_20260909/summary.json](../../experiments/teacher_prefix_handoff_20260909/summary.json) `/3` |

### threshold2_luna_k1_20260911

- 证据状态：**仅代码/状态文档**；代码目录：[experiments/threshold2_luna_k1_20260911](../../experiments/threshold2_luna_k1_20260911)。
- 配置/范围入口：[experiments/threshold2_luna_k1_20260911/STATUS.md](../../experiments/threshold2_luna_k1_20260911/STATUS.md)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### threshold3_mathteacher_20260911

- 证据状态：**已有结果汇总**；代码目录：[experiments/threshold3_mathteacher_20260911](../../experiments/threshold3_mathteacher_20260911)。
- 配置/范围入口：[experiments/threshold3_mathteacher_20260911/STATUS.md](../../experiments/threshold3_mathteacher_20260911/STATUS.md)。
- 已保存阶段完成标记：[experiments/threshold3_mathteacher_20260911/artifacts/ALL_GENERATION_COMPLETE.json](../../experiments/threshold3_mathteacher_20260911/artifacts/ALL_GENERATION_COMPLETE.json)、[experiments/threshold3_mathteacher_20260911/artifacts/THRESHOLD_GENERATION_COMPLETE.json](../../experiments/threshold3_mathteacher_20260911/artifacts/THRESHOLD_GENERATION_COMPLETE.json)。
- 结果入口：[experiments/threshold3_mathteacher_20260911/artifacts/summary.json](../../experiments/threshold3_mathteacher_20260911/artifacts/summary.json)、[results/westlake/opd_eval_hard/threshold3_mathteacher_20260911/summary.json](../../results/westlake/opd_eval_hard/threshold3_mathteacher_20260911/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| method=online / dataset=aime25；questions=30，correct=6，accuracy=0.2 | [experiments/threshold3_mathteacher_20260911/artifacts/summary.json](../../experiments/threshold3_mathteacher_20260911/artifacts/summary.json) `/online/aime25` |
| method=online / dataset=aime24；questions=30，correct=7，accuracy=0.233333 | [experiments/threshold3_mathteacher_20260911/artifacts/summary.json](../../experiments/threshold3_mathteacher_20260911/artifacts/summary.json) `/online/aime24` |
| method=online / dataset=olympiad_bench；questions=674，correct=286，accuracy=0.424332 | [experiments/threshold3_mathteacher_20260911/artifacts/summary.json](../../experiments/threshold3_mathteacher_20260911/artifacts/summary.json) `/online/olympiad_bench` |
| method=online / dataset=bbh；questions=1080，correct=828，accuracy=0.766667 | [experiments/threshold3_mathteacher_20260911/artifacts/summary.json](../../experiments/threshold3_mathteacher_20260911/artifacts/summary.json) `/online/bbh` |
| method=online / dataset=math500；questions=500，correct=412，accuracy=0.824 | [experiments/threshold3_mathteacher_20260911/artifacts/summary.json](../../experiments/threshold3_mathteacher_20260911/artifacts/summary.json) `/online/math500` |
| method=teacher / dataset=math500；questions=500，correct=437，accuracy=0.874 | [experiments/threshold3_mathteacher_20260911/artifacts/summary.json](../../experiments/threshold3_mathteacher_20260911/artifacts/summary.json) `/teacher/math500` |

### threshold_k8_nojudge_20260911

- 证据状态：**已有结果汇总**；代码目录：[experiments/threshold_k8_nojudge_20260911](../../experiments/threshold_k8_nojudge_20260911)。
- 配置/范围入口：[experiments/threshold_k8_nojudge_20260911/STATUS.md](../../experiments/threshold_k8_nojudge_20260911/STATUS.md)。
- 已保存阶段完成标记：[experiments/threshold_k8_nojudge_20260911/artifacts/ALL_GENERATION_COMPLETE.json](../../experiments/threshold_k8_nojudge_20260911/artifacts/ALL_GENERATION_COMPLETE.json)。
- 结果入口：[experiments/threshold_k8_nojudge_20260911/artifacts/summary.json](../../experiments/threshold_k8_nojudge_20260911/artifacts/summary.json)、[results/westlake/opd_eval_hard/threshold_k8_nojudge_20260911/summary.json](../../results/westlake/opd_eval_hard/threshold_k8_nojudge_20260911/summary.json)、[experiments/threshold_k8_nojudge_20260911/audit.json](../../experiments/threshold_k8_nojudge_20260911/audit.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| method=online / dataset=aime25；questions=30，correct=6，accuracy=0.2 | [experiments/threshold_k8_nojudge_20260911/artifacts/summary.json](../../experiments/threshold_k8_nojudge_20260911/artifacts/summary.json) `/online/aime25` |
| method=online / dataset=aime24；questions=30，correct=3，accuracy=0.1 | [experiments/threshold_k8_nojudge_20260911/artifacts/summary.json](../../experiments/threshold_k8_nojudge_20260911/artifacts/summary.json) `/online/aime24` |
| method=online / dataset=olympiad_bench；questions=674，correct=311，accuracy=0.461424 | [experiments/threshold_k8_nojudge_20260911/artifacts/summary.json](../../experiments/threshold_k8_nojudge_20260911/artifacts/summary.json) `/online/olympiad_bench` |
| method=online / dataset=bbh；questions=1080，correct=818，accuracy=0.757407 | [experiments/threshold_k8_nojudge_20260911/artifacts/summary.json](../../experiments/threshold_k8_nojudge_20260911/artifacts/summary.json) `/online/bbh` |
| method=online / dataset=math500；questions=500，correct=421，accuracy=0.842 | [experiments/threshold_k8_nojudge_20260911/artifacts/summary.json](../../experiments/threshold_k8_nojudge_20260911/artifacts/summary.json) `/online/math500` |

### token_entropy_distribution_20260914

- 证据状态：**已有结果汇总**；代码目录：[experiments/token_entropy_distribution_20260914](../../experiments/token_entropy_distribution_20260914)。
- 配置/范围入口：[experiments/token_entropy_distribution_20260914/PLAN.md](../../experiments/token_entropy_distribution_20260914/PLAN.md)。
- 已保存阶段完成标记：[experiments/token_entropy_distribution_20260914/results/COMPLETE.json](../../experiments/token_entropy_distribution_20260914/results/COMPLETE.json)、[experiments/token_entropy_distribution_20260914/results/student_COMPLETE.json](../../experiments/token_entropy_distribution_20260914/results/student_COMPLETE.json)、[experiments/token_entropy_distribution_20260914/results/teacher_COMPLETE.json](../../experiments/token_entropy_distribution_20260914/results/teacher_COMPLETE.json)。
- 结果入口：[experiments/token_entropy_distribution_20260914/results/k1_entropy_correctness.json](../../experiments/token_entropy_distribution_20260914/results/k1_entropy_correctness.json)、[experiments/token_entropy_distribution_20260914/results/k1_high_student_low_teacher_analysis.json](../../experiments/token_entropy_distribution_20260914/results/k1_high_student_low_teacher_analysis.json)、[experiments/token_entropy_distribution_20260914/results/k1_entropy_correctness.csv](../../experiments/token_entropy_distribution_20260914/results/k1_entropy_correctness.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| events=32360 | [experiments/token_entropy_distribution_20260914/results/k1_high_student_low_teacher_analysis.json](../../experiments/token_entropy_distribution_20260914/results/k1_high_student_low_teacher_analysis.json) `/` |

### weak1p7_strong4_reprompt_greedy_t2_k1_20261005

- 证据状态：**仅定位部署记录**；代码目录：[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_20261005](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_20261005)。
- 配置/范围入口：[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_20261005/LAUNCH.json](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_20261005/LAUNCH.json)、[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_20261005/README.md](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_20261005/README.md)、[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_20261005/configs.json](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_20261005/configs.json)。
- 已保存阶段完成标记：[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_20261005/INPUTS_COMPLETE.json](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_20261005/INPUTS_COMPLETE.json)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005

- 证据状态：**仅定位部署记录**；代码目录：[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005)。
- 配置/范围入口：[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005/LAUNCH.json](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005/LAUNCH.json)、[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005/README.md](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005/README.md)、[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005/configs.json](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005/configs.json)。
- 已保存阶段完成标记：[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005/INPUTS_COMPLETE.json](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_mainminus_20261005/INPUTS_COMPLETE.json)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。
- 不把命名中的 mainminus/postlogit 当成已有结果；应另行核对完成输出。

### weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005

- 证据状态：**仅定位部署记录**；代码目录：[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005)。
- 配置/范围入口：[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005/LAUNCH.json](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005/LAUNCH.json)、[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005/README.md](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005/README.md)、[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005/configs.json](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005/configs.json)。
- 已保存阶段完成标记：[experiments/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005/INPUTS_COMPLETE.json](../../experiments/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005/INPUTS_COMPLETE.json)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。
- 不把命名中的 mainminus/postlogit 当成已有结果；应另行核对完成输出。

### weak1p7_strong8_reprompt_greedy_t2_k1_20261005

- 证据状态：**仅定位部署记录**；代码目录：[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_20261005](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_20261005)。
- 配置/范围入口：[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_20261005/LAUNCH.json](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_20261005/LAUNCH.json)、[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_20261005/README.md](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_20261005/README.md)、[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_20261005/configs.json](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_20261005/configs.json)。
- 已保存阶段完成标记：[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_20261005/INPUTS_COMPLETE.json](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_20261005/INPUTS_COMPLETE.json)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005

- 证据状态：**仅定位部署记录**；代码目录：[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005)。
- 配置/范围入口：[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005/LAUNCH.json](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005/LAUNCH.json)、[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005/README.md](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005/README.md)、[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005/configs.json](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005/configs.json)。
- 已保存阶段完成标记：[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005/INPUTS_COMPLETE.json](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_mainminus_20261005/INPUTS_COMPLETE.json)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。
- 不把命名中的 mainminus/postlogit 当成已有结果；应另行核对完成输出。

### weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005

- 证据状态：**仅定位部署记录**；代码目录：[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005)。
- 配置/范围入口：[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005/LAUNCH.json](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005/LAUNCH.json)、[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005/README.md](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005/README.md)、[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005/configs.json](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005/configs.json)。
- 已保存阶段完成标记：[experiments/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005/INPUTS_COMPLETE.json](../../experiments/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005/INPUTS_COMPLETE.json)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。
- 不把命名中的 mainminus/postlogit 当成已有结果；应另行核对完成输出。

### weak_reprompt_16gpu_2000s_20260920

- 证据状态：**仅定位部署记录**；代码目录：[experiments/weak_reprompt_16gpu_2000s_20260920](../../experiments/weak_reprompt_16gpu_2000s_20260920)。
- 配置/范围入口：[experiments/weak_reprompt_16gpu_2000s_20260920/LAUNCH.json](../../experiments/weak_reprompt_16gpu_2000s_20260920/LAUNCH.json)、[experiments/weak_reprompt_16gpu_2000s_20260920/README.md](../../experiments/weak_reprompt_16gpu_2000s_20260920/README.md)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### weak_reprompt_8gpu_2000s_20260920

- 证据状态：**仅定位部署记录**；代码目录：[experiments/weak_reprompt_8gpu_2000s_20260920](../../experiments/weak_reprompt_8gpu_2000s_20260920)。
- 配置/范围入口：[experiments/weak_reprompt_8gpu_2000s_20260920/LAUNCH.json](../../experiments/weak_reprompt_8gpu_2000s_20260920/LAUNCH.json)、[experiments/weak_reprompt_8gpu_2000s_20260920/README.md](../../experiments/weak_reprompt_8gpu_2000s_20260920/README.md)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### weak_reprompt_clean_greedy_score_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_reprompt_clean_greedy_score_20260921](../../experiments/weak_reprompt_clean_greedy_score_20260921)。
- 配置/范围入口：[experiments/weak_reprompt_clean_greedy_score_20260921/LAUNCH.json](../../experiments/weak_reprompt_clean_greedy_score_20260921/LAUNCH.json)。
- 结果入口：[reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=368，accuracy=0.340741 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=58，accuracy=0.0725 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=174，accuracy=0.2175 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=106，accuracy=0.646341 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=25，accuracy=0.142857 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=381，accuracy=0.762 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=184，accuracy=0.486772 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=333，accuracy=0.494065 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/7` |

### weak_reprompt_clean_greedy_t2_k1_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_reprompt_clean_greedy_t2_k1_20260921](../../experiments/weak_reprompt_clean_greedy_t2_k1_20260921)。
- 配置/范围入口：[experiments/weak_reprompt_clean_greedy_t2_k1_20260921/LAUNCH.json](../../experiments/weak_reprompt_clean_greedy_t2_k1_20260921/LAUNCH.json)、[experiments/weak_reprompt_clean_greedy_t2_k1_20260921/README.md](../../experiments/weak_reprompt_clean_greedy_t2_k1_20260921/README.md)。
- 结果入口：[reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json)、[results/westlake/opd_eval_ablation/weak_reprompt_clean_greedy_t2_k1_20260921/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_clean_greedy_t2_k1_20260921/summary.json)、[results/westlake/opd_eval_ablation/weak_reprompt_clean_greedy_t2_k1_20260921/summary.csv](../../results/westlake/opd_eval_ablation/weak_reprompt_clean_greedy_t2_k1_20260921/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=368，accuracy=0.340741 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=58，accuracy=0.0725 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=174，accuracy=0.2175 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=106，accuracy=0.646341 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=25，accuracy=0.142857 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=381，accuracy=0.762 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=184，accuracy=0.486772 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=333，accuracy=0.494065 | [reports/weak_reprompt_clean_greedy_results_20260921/summary.json](../../reports/weak_reprompt_clean_greedy_results_20260921/summary.json) `/groups/7` |

### weak_reprompt_clean_score_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_reprompt_clean_score_20260921](../../experiments/weak_reprompt_clean_score_20260921)。
- 配置/范围入口：[experiments/weak_reprompt_clean_score_20260921/LAUNCH.json](../../experiments/weak_reprompt_clean_score_20260921/LAUNCH.json)。
- 结果入口：[reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=366，accuracy=0.338889 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=51，accuracy=0.06375 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=137，accuracy=0.17125 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=87，accuracy=0.530488 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=30，accuracy=0.171429 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=380，accuracy=0.76 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=178，accuracy=0.470899 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=322，accuracy=0.477745 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/7` |

### weak_reprompt_clean_t2_k1_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_reprompt_clean_t2_k1_20260921](../../experiments/weak_reprompt_clean_t2_k1_20260921)。
- 配置/范围入口：[experiments/weak_reprompt_clean_t2_k1_20260921/LAUNCH.json](../../experiments/weak_reprompt_clean_t2_k1_20260921/LAUNCH.json)、[experiments/weak_reprompt_clean_t2_k1_20260921/README.md](../../experiments/weak_reprompt_clean_t2_k1_20260921/README.md)。
- 结果入口：[reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json)、[results/westlake/opd_eval_ablation/weak_reprompt_clean_t2_k1_20260921/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_clean_t2_k1_20260921/summary.json)、[results/westlake/opd_eval_ablation/weak_reprompt_clean_t2_k1_20260921/summary.csv](../../results/westlake/opd_eval_ablation/weak_reprompt_clean_t2_k1_20260921/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=366，accuracy=0.338889 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=51，accuracy=0.06375 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=137，accuracy=0.17125 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=87，accuracy=0.530488 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=30，accuracy=0.171429 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=380，accuracy=0.76 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=178，accuracy=0.470899 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=322，accuracy=0.477745 | [reports/weak_reprompt_clean_results_20260921/summary.json](../../reports/weak_reprompt_clean_results_20260921/summary.json) `/groups/7` |

### weak_reprompt_nodraft_greedy_score_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_reprompt_nodraft_greedy_score_20260921](../../experiments/weak_reprompt_nodraft_greedy_score_20260921)。
- 配置/范围入口：[experiments/weak_reprompt_nodraft_greedy_score_20260921/LAUNCH.json](../../experiments/weak_reprompt_nodraft_greedy_score_20260921/LAUNCH.json)。
- 结果入口：[reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=761，accuracy=0.70463 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=71，accuracy=0.08875 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=131，accuracy=0.16375 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=121，accuracy=0.737805 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=39，accuracy=0.222857 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=393，accuracy=0.786 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=246，accuracy=0.650794 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=344，accuracy=0.510386 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/7` |

### weak_reprompt_nodraft_greedy_t2_k1_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_reprompt_nodraft_greedy_t2_k1_20260921](../../experiments/weak_reprompt_nodraft_greedy_t2_k1_20260921)。
- 配置/范围入口：[experiments/weak_reprompt_nodraft_greedy_t2_k1_20260921/LAUNCH.json](../../experiments/weak_reprompt_nodraft_greedy_t2_k1_20260921/LAUNCH.json)、[experiments/weak_reprompt_nodraft_greedy_t2_k1_20260921/README.md](../../experiments/weak_reprompt_nodraft_greedy_t2_k1_20260921/README.md)。
- 结果入口：[reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json)、[results/westlake/opd_eval_ablation/weak_reprompt_nodraft_greedy_t2_k1_20260921/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_nodraft_greedy_t2_k1_20260921/summary.json)、[results/westlake/opd_eval_ablation/weak_reprompt_nodraft_greedy_t2_k1_20260921/summary.csv](../../results/westlake/opd_eval_ablation/weak_reprompt_nodraft_greedy_t2_k1_20260921/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=761，accuracy=0.70463 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=71，accuracy=0.08875 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=131，accuracy=0.16375 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=121，accuracy=0.737805 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=39，accuracy=0.222857 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=393，accuracy=0.786 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=246，accuracy=0.650794 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=344，accuracy=0.510386 | [reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json](../../reports/weak_reprompt_nodraft_greedy_results_20260921/summary.json) `/groups/7` |

### weak_reprompt_nodraft_score_20260920

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_reprompt_nodraft_score_20260920](../../experiments/weak_reprompt_nodraft_score_20260920)。
- 配置/范围入口：[experiments/weak_reprompt_nodraft_score_20260920/LAUNCH.json](../../experiments/weak_reprompt_nodraft_score_20260920/LAUNCH.json)、[experiments/weak_reprompt_nodraft_score_20260920/README.md](../../experiments/weak_reprompt_nodraft_score_20260920/README.md)。
- 结果入口：[reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json)、[reports/weak_reprompt_nodraft_results_20260920/baseline_comparison.json](../../reports/weak_reprompt_nodraft_results_20260920/baseline_comparison.json)、[reports/weak_reprompt_nodraft_results_20260920/summary.csv](../../reports/weak_reprompt_nodraft_results_20260920/summary.csv)、[reports/weak_reprompt_nodraft_results_20260920/baseline_comparison.csv](../../reports/weak_reprompt_nodraft_results_20260920/baseline_comparison.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=702，accuracy=0.65 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=52，accuracy=0.065 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=125，accuracy=0.15625 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=122，accuracy=0.743902 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=40，accuracy=0.228571 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=396，accuracy=0.792 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=240，accuracy=0.634921 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=337，accuracy=0.5 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/7` |

### weak_reprompt_nodraft_t2_k1_20260920

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_reprompt_nodraft_t2_k1_20260920](../../experiments/weak_reprompt_nodraft_t2_k1_20260920)。
- 配置/范围入口：[experiments/weak_reprompt_nodraft_t2_k1_20260920/LAUNCH.json](../../experiments/weak_reprompt_nodraft_t2_k1_20260920/LAUNCH.json)、[experiments/weak_reprompt_nodraft_t2_k1_20260920/README.md](../../experiments/weak_reprompt_nodraft_t2_k1_20260920/README.md)。
- 结果入口：[reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json)、[results/westlake/opd_eval_ablation/weak_reprompt_nodraft_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_nodraft_t2_k1_20260920/summary.json)、[reports/weak_reprompt_nodraft_results_20260920/baseline_comparison.json](../../reports/weak_reprompt_nodraft_results_20260920/baseline_comparison.json)、[reports/weak_reprompt_nodraft_results_20260920/summary.csv](../../reports/weak_reprompt_nodraft_results_20260920/summary.csv)、[reports/weak_reprompt_nodraft_results_20260920/baseline_comparison.csv](../../reports/weak_reprompt_nodraft_results_20260920/baseline_comparison.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=702，accuracy=0.65 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=52，accuracy=0.065 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=125，accuracy=0.15625 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=122，accuracy=0.743902 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=40，accuracy=0.228571 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=396，accuracy=0.792 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=240，accuracy=0.634921 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=337，accuracy=0.5 | [reports/weak_reprompt_nodraft_results_20260920/summary.json](../../reports/weak_reprompt_nodraft_results_20260920/summary.json) `/groups/7` |

### weak_reprompt_score_20260921

- 证据状态：**仅定位部署记录**；代码目录：[experiments/weak_reprompt_score_20260921](../../experiments/weak_reprompt_score_20260921)。
- 配置/范围入口：[experiments/weak_reprompt_score_20260921/LAUNCH.json](../../experiments/weak_reprompt_score_20260921/LAUNCH.json)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

### weak_reprompt_t2_k1_20260920

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_reprompt_t2_k1_20260920](../../experiments/weak_reprompt_t2_k1_20260920)。
- 配置/范围入口：[experiments/weak_reprompt_t2_k1_20260920/LAUNCH.json](../../experiments/weak_reprompt_t2_k1_20260920/LAUNCH.json)、[experiments/weak_reprompt_t2_k1_20260920/README.md](../../experiments/weak_reprompt_t2_k1_20260920/README.md)。
- 结果入口：[results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json)、[results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.csv](../../results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=261，accuracy=0.241667 | [results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=35，accuracy=0.04375 | [results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=77，accuracy=0.09625 | [results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=41，accuracy=0.25 | [results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=11，accuracy=0.0628571 | [results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=360，accuracy=0.72 | [results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=111，accuracy=0.293651 | [results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=305，accuracy=0.452522 | [results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json) `/groups/7` |

### weak_resample_greedy_8gpu_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_resample_greedy_8gpu_20260921](../../experiments/weak_resample_greedy_8gpu_20260921)。
- 配置/范围入口：[experiments/weak_resample_greedy_8gpu_20260921/LAUNCH.json](../../experiments/weak_resample_greedy_8gpu_20260921/LAUNCH.json)。
- 结果入口：[reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=401，accuracy=0.371296 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=154，accuracy=0.1925 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=199，accuracy=0.24875 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=126，accuracy=0.768293 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=35，accuracy=0.2 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=390，accuracy=0.78 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=243，accuracy=0.642857 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=337，accuracy=0.5 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/7` |

### weak_resample_greedy_score_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_resample_greedy_score_20260921](../../experiments/weak_resample_greedy_score_20260921)。
- 配置/范围入口：[experiments/weak_resample_greedy_score_20260921/LAUNCH.json](../../experiments/weak_resample_greedy_score_20260921/LAUNCH.json)。
- 结果入口：[reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=401，accuracy=0.371296 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=154，accuracy=0.1925 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=199，accuracy=0.24875 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=126，accuracy=0.768293 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=35，accuracy=0.2 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=390，accuracy=0.78 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=243，accuracy=0.642857 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=337，accuracy=0.5 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/7` |

### weak_resample_greedy_t2_k1_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_resample_greedy_t2_k1_20260921](../../experiments/weak_resample_greedy_t2_k1_20260921)。
- 配置/范围入口：[experiments/weak_resample_greedy_t2_k1_20260921/LAUNCH.json](../../experiments/weak_resample_greedy_t2_k1_20260921/LAUNCH.json)、[experiments/weak_resample_greedy_t2_k1_20260921/README.md](../../experiments/weak_resample_greedy_t2_k1_20260921/README.md)。
- 结果入口：[reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json)、[results/westlake/opd_eval_ablation/weak_resample_greedy_t2_k1_20260921/summary.json](../../results/westlake/opd_eval_ablation/weak_resample_greedy_t2_k1_20260921/summary.json)、[results/westlake/opd_eval_ablation/weak_resample_greedy_t2_k1_20260921/summary.csv](../../results/westlake/opd_eval_ablation/weak_resample_greedy_t2_k1_20260921/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=401，accuracy=0.371296 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=154，accuracy=0.1925 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=199，accuracy=0.24875 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=126，accuracy=0.768293 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=35，accuracy=0.2 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=390，accuracy=0.78 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=243，accuracy=0.642857 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=337，accuracy=0.5 | [reports/weak_resample_greedy_results_20260921/summary.json](../../reports/weak_resample_greedy_results_20260921/summary.json) `/groups/7` |

### weak_resample_score_20260921

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_resample_score_20260921](../../experiments/weak_resample_score_20260921)。
- 配置/范围入口：[experiments/weak_resample_score_20260921/LAUNCH.json](../../experiments/weak_resample_score_20260921/LAUNCH.json)。
- 结果入口：[reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=410，accuracy=0.37963 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=142，accuracy=0.1775 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=198，accuracy=0.2475 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=122，accuracy=0.743902 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=38，accuracy=0.217143 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=391，accuracy=0.782 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=245，accuracy=0.648148 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=341，accuracy=0.505935 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/7` |

### weak_resample_t2_k1_20260920

- 证据状态：**已有结果汇总**；代码目录：[experiments/weak_resample_t2_k1_20260920](../../experiments/weak_resample_t2_k1_20260920)。
- 配置/范围入口：[experiments/weak_resample_t2_k1_20260920/LAUNCH.json](../../experiments/weak_resample_t2_k1_20260920/LAUNCH.json)、[experiments/weak_resample_t2_k1_20260920/README.md](../../experiments/weak_resample_t2_k1_20260920/README.md)。
- 结果入口：[reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json)、[results/westlake/opd_eval_ablation/weak_resample_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_resample_t2_k1_20260920/summary.json)、[results/westlake/opd_eval_ablation/weak_resample_t2_k1_20260920/summary.csv](../../results/westlake/opd_eval_ablation/weak_resample_t2_k1_20260920/summary.csv)。

| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |
|---|---|
| dataset=bbh；n=1080，correct=410，accuracy=0.37963 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/0` |
| dataset=cruxeval_i；n=800，correct=142，accuracy=0.1775 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/1` |
| dataset=cruxeval_o；n=800，correct=198，accuracy=0.2475 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/2` |
| dataset=humaneval_plus；n=164，correct=122，accuracy=0.743902 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/3` |
| dataset=livecodebench；n=175，correct=38，accuracy=0.217143 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/4` |
| dataset=math500；n=500，correct=391，accuracy=0.782 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/5` |
| dataset=mbpp_plus；n=378，correct=245，accuracy=0.648148 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/6` |
| dataset=olympiad_bench；n=674，correct=341，accuracy=0.505935 | [reports/weak_resample_results_20260921/summary.json](../../reports/weak_resample_results_20260921/summary.json) `/groups/7` |

### weak_to_8b_prefix_handoff_20261005

- 证据状态：**仅代码/状态文档**；代码目录：[experiments/weak_to_8b_prefix_handoff_20261005](../../experiments/weak_to_8b_prefix_handoff_20261005)。
- 配置/范围入口：[experiments/weak_to_8b_prefix_handoff_20261005/README.md](../../experiments/weak_to_8b_prefix_handoff_20261005/README.md)。
- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。

## 不在 experiments/ 的早期与跨目录系列

### 早期 AIME24 单次评测与提取审计

状态：**已有结果汇总**。入口：[EXPERIMENT_AUDIT.json](../../EXPERIMENT_AUDIT.json)、[EXPERIMENT_AUDIT.md](../../EXPERIMENT_AUDIT.md)。

保存数值的产物：[EXPERIMENT_AUDIT.json](../../EXPERIMENT_AUDIT.json)。

### DeepMath train25% / Top16 Reverse-KL 训练

状态：**已有结果汇总**。入口：[analysis/deepmath_train25_top16_rkl_logs](../../analysis/deepmath_train25_top16_rkl_logs)。

保存数值的产物：[analysis/deepmath_train25_top16_rkl_logs/training_curve.csv](../../analysis/deepmath_train25_top16_rkl_logs/training_curve.csv)、[analysis/deepmath_train25_top16_rkl_logs/training_summary.json](../../analysis/deepmath_train25_top16_rkl_logs/training_summary.json)。

### DeepMath thinking 测试 1000 题与五个 checkpoint

状态：**已有结果汇总**。入口：[results/legacy/deepmath-heldout-1000](../../results/legacy/deepmath-heldout-1000)。

保存数值的产物：[results/legacy/deepmath-heldout-1000/baseline.json](../../results/legacy/deepmath-heldout-1000/baseline.json)、[results/legacy/deepmath-heldout-1000/step106.json](../../results/legacy/deepmath-heldout-1000/step106.json)、[results/legacy/deepmath-heldout-1000/step133.json](../../results/legacy/deepmath-heldout-1000/step133.json)、[results/legacy/deepmath-heldout-1000/step27.json](../../results/legacy/deepmath-heldout-1000/step27.json)、[results/legacy/deepmath-heldout-1000/step53.json](../../results/legacy/deepmath-heldout-1000/step53.json)、[results/legacy/deepmath-heldout-1000/step80.json](../../results/legacy/deepmath-heldout-1000/step80.json)。

### DeepMath nonthinking Student / Traditional / Bridge / Good-Bridge OPD

状态：**已有结果汇总**。入口：[figures/testset_model_comparison_20260908.json](../../figures/testset_model_comparison_20260908.json)。

保存数值的产物：[figures/testset_model_comparison_20260908.json](../../figures/testset_model_comparison_20260908.json)。

### DeepMath thinking / nonthinking 逐 token teacher-student gap

状态：**已有结果汇总**。入口：[reports/deepmath_baseline_sample_gap/summary.json](../../reports/deepmath_baseline_sample_gap/summary.json)、[reports/deepmath_nonthinking_sample_gap/summary.json](../../reports/deepmath_nonthinking_sample_gap/summary.json)、[reports/deepmath_baseline_gap_position_analysis.json](../../reports/deepmath_baseline_gap_position_analysis.json)、[reports/deepmath_student_top16_trace_summary.json](../../reports/deepmath_student_top16_trace_summary.json)。

保存数值的产物：[reports/deepmath_baseline_gap_position_analysis.json](../../reports/deepmath_baseline_gap_position_analysis.json)、[reports/deepmath_baseline_sample_gap/summary.json](../../reports/deepmath_baseline_sample_gap/summary.json)、[reports/deepmath_nonthinking_sample_gap/summary.json](../../reports/deepmath_nonthinking_sample_gap/summary.json)、[reports/deepmath_student_top16_trace_summary.json](../../reports/deepmath_student_top16_trace_summary.json)。

### Bridge horizon 与长度/准确率关系分析

状态：**已有结果汇总**。入口：[analysis/bridge_horizon_summary_20260904.csv](../../analysis/bridge_horizon_summary_20260904.csv)、[analysis/bridge_horizon_summary_20260906.csv](../../analysis/bridge_horizon_summary_20260906.csv)、[analysis/rollout_shortening_accuracy_20260907](../../analysis/rollout_shortening_accuracy_20260907)、[analysis/teacher_student_length_accuracy_correlation_20260908.json](../../analysis/teacher_student_length_accuracy_correlation_20260908.json)、[analysis/bridge_pass_metrics_20260906.csv](../../analysis/bridge_pass_metrics_20260906.csv)。

保存数值的产物：[analysis/bridge_horizon_summary_20260904.csv](../../analysis/bridge_horizon_summary_20260904.csv)、[analysis/bridge_horizon_summary_20260906.csv](../../analysis/bridge_horizon_summary_20260906.csv)、[analysis/bridge_pass_metrics_20260906.csv](../../analysis/bridge_pass_metrics_20260906.csv)、[analysis/rollout_shortening_accuracy_20260907/buckets.csv](../../analysis/rollout_shortening_accuracy_20260907/buckets.csv)、[analysis/rollout_shortening_accuracy_20260907/summary.json](../../analysis/rollout_shortening_accuracy_20260907/summary.json)、[analysis/teacher_student_length_accuracy_correlation_20260908.json](../../analysis/teacher_student_length_accuracy_correlation_20260908.json)。

### H256 分叉锚点语义判定（Luna）

状态：**仅文档证据**。入口：[outputs/anchor_bifurcation_h256_openlux_luna_seed20260908/full/run_summary.json](../../outputs/anchor_bifurcation_h256_openlux_luna_seed20260908/full/run_summary.json)、[outputs/anchor_bifurcation_h256_openlux_luna_seed20260908/pilot8/run_summary.json](../../outputs/anchor_bifurcation_h256_openlux_luna_seed20260908/pilot8/run_summary.json)。

### 八轮主结果绘图数据与校验

状态：**仅文档证据**。入口：[figures/main_grid_x8_20260920](../../figures/main_grid_x8_20260920)。

### 介入位置、长度、分布与三元结果分析

状态：**已有结果汇总**。入口：[reports/intervention_positions_20260923](../../reports/intervention_positions_20260923)、[reports/k1_t2_positions_20260916](../../reports/k1_t2_positions_20260916)、[reports/k1_t2_length_vs_student_20260916](../../reports/k1_t2_length_vs_student_20260916)、[reports/intervention_distribution_20260911](../../reports/intervention_distribution_20260911)、[reports/current_greedy_triplet_analysis_20260923](../../reports/current_greedy_triplet_analysis_20260923)。

保存数值的产物：[reports/current_greedy_triplet_analysis_20260923/per_dataset.csv](../../reports/current_greedy_triplet_analysis_20260923/per_dataset.csv)、[reports/intervention_distribution_20260911/frequency.csv](../../reports/intervention_distribution_20260911/frequency.csv)、[reports/intervention_positions_20260923/raw_summary.json](../../reports/intervention_positions_20260923/raw_summary.json)、[reports/k1_t2_length_vs_student_20260916/summary.csv](../../reports/k1_t2_length_vs_student_20260916/summary.csv)、[reports/k1_t2_length_vs_student_20260916/summary.json](../../reports/k1_t2_length_vs_student_20260916/summary.json)、[reports/k1_t2_positions_20260916/summary.csv](../../reports/k1_t2_positions_20260916/summary.csv)、[reports/k1_t2_positions_20260916/summary.json](../../reports/k1_t2_positions_20260916/summary.json)。

### 早期单次 rollout 跨 benchmark 比较

状态：**已有结果汇总**。入口：[reports/all_benchmark_comparison_20260911](../../reports/all_benchmark_comparison_20260911)。

保存数值的产物：[reports/all_benchmark_comparison_20260911/remote_aggregate_evidence.json](../../reports/all_benchmark_comparison_20260911/remote_aggregate_evidence.json)、[reports/all_benchmark_comparison_20260911/single_rollout_results.csv](../../reports/all_benchmark_comparison_20260911/single_rollout_results.csv)、[reports/all_benchmark_comparison_20260911/single_rollout_results.json](../../reports/all_benchmark_comparison_20260911/single_rollout_results.json)。

### 全实验可视化数据与混合策略三元表

状态：**已有结果汇总**。入口：[outputs/benchmark-viz](../../outputs/benchmark-viz)、[outputs/triple-outcomes](../../outputs/triple-outcomes)、[outputs/prefix-triple-outcomes](../../outputs/prefix-triple-outcomes)、[outputs/01a085ee-f483-7b63-83aa-0ac0dbf0fac4/merged_benchmarks](../../outputs/01a085ee-f483-7b63-83aa-0ac0dbf0fac4/merged_benchmarks)。

保存数值的产物：[outputs/01a085ee-f483-7b63-83aa-0ac0dbf0fac4/merged_benchmarks/math_metrics.json](../../outputs/01a085ee-f483-7b63-83aa-0ac0dbf0fac4/merged_benchmarks/math_metrics.json)。

### B3/B4 同前缀反事实与机制收益归属（计划）

状态：**仅计划**。入口：[refine-logs/EXPERIMENT_PLAN.md](../../refine-logs/EXPERIMENT_PLAN.md)、[refine-logs/EXPERIMENT_TRACKER.md](../../refine-logs/EXPERIMENT_TRACKER.md)。

### 早期方法与训练、介入长度、锚点确认实验规划

状态：**仅计划**。入口：[refine-logs](../../refine-logs)、[idea-stage](../../idea-stage)。

## 西湖共享存储运行目录覆盖

已同步 995 个小文件，覆盖 106 个运行/数据目录；远端顶层清点 33 项。这里只对已下载的小摘要建立索引，不复制数据集、模型权重和大型输出。远端清点见 [artifacts/storage_inventory/westlake_inventory_20261019.json](../../artifacts/storage_inventory/westlake_inventory_20261019.json)。

| 远端分类/运行 | 证据状态 | 同步文件数 | 数值记录数 | 结果节选（源单位） | 汇总入口 |
|---|---|---:|---:|---|---|
| analysis/rollout_shortening_accuracy_20260907 | 已有结果汇总 | 1 | 13 | rollouts=847616，accuracy=0.404995<br>rollouts=847616，accuracy=0.410011 | [results/westlake/analysis/rollout_shortening_accuracy_20260907/summary.json](../../results/westlake/analysis/rollout_shortening_accuracy_20260907/summary.json) |
| analysis/rollout_shortening_significance_20260908 | 已有结果汇总 | 1 | 29 | rollouts=847616，accuracy=0.404995<br>rollouts=847616，accuracy=0.410011 | [results/westlake/analysis/rollout_shortening_significance_20260908/summary.json](../../results/westlake/analysis/rollout_shortening_significance_20260908/summary.json) |
| opd_analysis/anchor_bifurcation_h256_openrouter_luna_seed20260908 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_analysis/anchor_bifurcation_h256_openrouter_luna_seed20260908](../../results/westlake/opd_analysis/anchor_bifurcation_h256_openrouter_luna_seed20260908) |
| opd_analysis/intervention_analysis_greedy_20260921 | 已有结果汇总 | 10 | 234 | dataset=ALL；events=43879<br>dataset=bbh；events=19181 | [results/westlake/opd_analysis/intervention_analysis_greedy_20260921/continuous_support/summary.csv](../../results/westlake/opd_analysis/intervention_analysis_greedy_20260921/continuous_support/summary.csv)、[results/westlake/opd_analysis/intervention_analysis_greedy_20260921/probability_pair/paired_summary.csv](../../results/westlake/opd_analysis/intervention_analysis_greedy_20260921/probability_pair/paired_summary.csv) |
| opd_analysis/intervention_analysis_phase1_20260921 | 已有结果汇总 | 9 | 284 | dataset=ALL；events=399601<br>dataset=bbh；events=159435 | [results/westlake/opd_analysis/intervention_analysis_phase1_20260921/continuous_support/summary.csv](../../results/westlake/opd_analysis/intervention_analysis_phase1_20260921/continuous_support/summary.csv)、[results/westlake/opd_analysis/intervention_analysis_phase1_20260921/probability_pair/paired_summary.csv](../../results/westlake/opd_analysis/intervention_analysis_phase1_20260921/probability_pair/paired_summary.csv) |
| opd_analysis/intervention_entropy_20260914 | 已有结果汇总 | 1 | 740 | 未定位准确率/主指标汇总 | [results/westlake/opd_analysis/intervention_entropy_20260914/summary.csv](../../results/westlake/opd_analysis/intervention_entropy_20260914/summary.csv) |
| opd_analysis/intervention_logits_bf16_greedy_20260921 | 仅元数据/阶段标记 | 1 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_analysis/intervention_logits_bf16_greedy_20260921](../../results/westlake/opd_analysis/intervention_logits_bf16_greedy_20260921) |
| opd_analysis/intervention_logits_bf16_x8_20260920 | 仅元数据/阶段标记 | 1 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_analysis/intervention_logits_bf16_x8_20260920](../../results/westlake/opd_analysis/intervention_logits_bf16_x8_20260920) |
| opd_analysis/k1_top1_agreement_20260914 | 仅元数据/阶段标记 | 1 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_analysis/k1_top1_agreement_20260914](../../results/westlake/opd_analysis/k1_top1_agreement_20260914) |
| opd_analysis/token_entropy_distribution_20260914 | 仅元数据/阶段标记 | 1 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_analysis/token_entropy_distribution_20260914](../../results/westlake/opd_analysis/token_entropy_distribution_20260914) |
| opd_data/attempts | 仅元数据/阶段标记 | 1 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_data/attempts](../../results/westlake/opd_data/attempts) |
| opd_data/deepmath_103k | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_data/deepmath_103k](../../results/westlake/opd_data/deepmath_103k) |
| opd_data/deepmath_ge6_opd | 仅元数据/阶段标记 | 1 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_data/deepmath_ge6_opd](../../results/westlake/opd_data/deepmath_ge6_opd) |
| opd_data/deepmath_ge6_opd_train15pct_seed20260811 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_data/deepmath_ge6_opd_train15pct_seed20260811](../../results/westlake/opd_data/deepmath_ge6_opd_train15pct_seed20260811) |
| opd_data/deepmath_ge6_opd_train25pct_perdifficulty_seed20260823 | 仅元数据/阶段标记 | 1 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_data/deepmath_ge6_opd_train25pct_perdifficulty_seed20260823](../../results/westlake/opd_data/deepmath_ge6_opd_train25pct_perdifficulty_seed20260823) |
| opd_data/model_downloads | 仅元数据/阶段标记 | 3 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_data/model_downloads](../../results/westlake/opd_data/model_downloads) |
| opd_eval/qwen3_gsm8k_math_512 | 已有结果汇总 | 3 | 6 | correct=110，accuracy=0.429688<br>correct=95，accuracy=0.371094 | [results/westlake/opd_eval/qwen3_gsm8k_math_512/qwen3_14b_base.summary.json](../../results/westlake/opd_eval/qwen3_gsm8k_math_512/qwen3_14b_base.summary.json)、[results/westlake/opd_eval/qwen3_gsm8k_math_512/qwen3_4b_base.summary.json](../../results/westlake/opd_eval/qwen3_gsm8k_math_512/qwen3_4b_base.summary.json) |
| opd_eval_ablation/main_greedy_k1_t2_20260921 | 已有结果汇总 | 5 | 16 | dataset=bbh；n=1080，correct=849，accuracy=0.786111<br>dataset=cruxeval_i；n=800，correct=394，accuracy=0.4925 | [results/westlake/opd_eval_ablation/main_greedy_k1_t2_20260921/summary.csv](../../results/westlake/opd_eval_ablation/main_greedy_k1_t2_20260921/summary.csv)、[results/westlake/opd_eval_ablation/main_greedy_k1_t2_20260921/summary.json](../../results/westlake/opd_eval_ablation/main_greedy_k1_t2_20260921/summary.json) |
| opd_eval_ablation/random_quartile_greedy_k1_20260921 | 已有结果汇总 | 6 | 16 | dataset=bbh；n=1080，correct=421，accuracy=0.389815<br>dataset=cruxeval_i；n=800，correct=72，accuracy=0.09 | [results/westlake/opd_eval_ablation/random_quartile_greedy_k1_20260921/summary.csv](../../results/westlake/opd_eval_ablation/random_quartile_greedy_k1_20260921/summary.csv)、[results/westlake/opd_eval_ablation/random_quartile_greedy_k1_20260921/summary.json](../../results/westlake/opd_eval_ablation/random_quartile_greedy_k1_20260921/summary.json) |
| opd_eval_ablation/random_quartile_k1_20260918 | 已有结果汇总 | 5 | 21 | dataset=cruxeval_i；n=800，correct=83，accuracy=0.10375<br>dataset=cruxeval_o；n=800，correct=196，accuracy=0.245 | [results/westlake/opd_eval_ablation/random_quartile_k1_20260918/code_summary.json](../../results/westlake/opd_eval_ablation/random_quartile_k1_20260918/code_summary.json)、[results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.csv](../../results/westlake/opd_eval_ablation/random_quartile_k1_20260918/summary.csv) |
| opd_eval_ablation/reverse_greedy_t2_k1_8gpu_20260921 | 已有结果汇总 | 6 | 16 | dataset=bbh；n=1080，correct=470，accuracy=0.435185<br>dataset=cruxeval_i；n=800，correct=192，accuracy=0.24 | [results/westlake/opd_eval_ablation/reverse_greedy_t2_k1_8gpu_20260921/summary.csv](../../results/westlake/opd_eval_ablation/reverse_greedy_t2_k1_8gpu_20260921/summary.csv)、[results/westlake/opd_eval_ablation/reverse_greedy_t2_k1_8gpu_20260921/summary.json](../../results/westlake/opd_eval_ablation/reverse_greedy_t2_k1_8gpu_20260921/summary.json) |
| opd_eval_ablation/reverse_t2_k1_20260918 | 已有结果汇总 | 6 | 21 | dataset=cruxeval_i；n=800，correct=186，accuracy=0.2325<br>dataset=cruxeval_o；n=800，correct=188，accuracy=0.235 | [results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/code_summary.json](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/code_summary.json)、[results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.csv](../../results/westlake/opd_eval_ablation/reverse_t2_k1_20260918/summary.csv) |
| opd_eval_ablation/shift5_greedy_k1_t2_20260921 | 已有结果汇总 | 5 | 16 | dataset=bbh；n=1080，correct=234，accuracy=0.216667<br>dataset=cruxeval_i；n=800，correct=104，accuracy=0.13 | [results/westlake/opd_eval_ablation/shift5_greedy_k1_t2_20260921/summary.csv](../../results/westlake/opd_eval_ablation/shift5_greedy_k1_t2_20260921/summary.csv)、[results/westlake/opd_eval_ablation/shift5_greedy_k1_t2_20260921/summary.json](../../results/westlake/opd_eval_ablation/shift5_greedy_k1_t2_20260921/summary.json) |
| opd_eval_ablation/shift5_k1_t2_20260918 | 已有结果汇总 | 5 | 21 | dataset=cruxeval_i；n=800，correct=97，accuracy=0.12125<br>dataset=cruxeval_o；n=800，correct=140，accuracy=0.175 | [results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/code_summary.json](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/code_summary.json)、[results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.csv](../../results/westlake/opd_eval_ablation/shift5_k1_t2_20260918/summary.csv) |
| opd_eval_ablation/strong4_weak17_reprompt_greedy_t2_k1_20261005 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_ablation/strong4_weak17_reprompt_greedy_t2_k1_20261005](../../results/westlake/opd_eval_ablation/strong4_weak17_reprompt_greedy_t2_k1_20261005) |
| opd_eval_ablation/strong8_weak17_reprompt_greedy_t2_k1_20261005 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_ablation/strong8_weak17_reprompt_greedy_t2_k1_20261005](../../results/westlake/opd_eval_ablation/strong8_weak17_reprompt_greedy_t2_k1_20261005) |
| opd_eval_ablation/strong_reprompt_score_20261005 | 已有结果汇总 | 3 | 40 | dataset=aime24 / experiment=strong4_weak17；n=30，correct=5，accuracy=0.166667<br>dataset=aime25 / experiment=strong4_weak17；n=30，correct=4，accuracy=0.133333 | [results/westlake/opd_eval_ablation/strong_reprompt_score_20261005/summary.csv](../../results/westlake/opd_eval_ablation/strong_reprompt_score_20261005/summary.csv)、[results/westlake/opd_eval_ablation/strong_reprompt_score_20261005/summary.json](../../results/westlake/opd_eval_ablation/strong_reprompt_score_20261005/summary.json) |
| opd_eval_ablation/weak1p7_strong4_reprompt_greedy_t2_k1_20261005 | 仅元数据/阶段标记 | 1 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_ablation/weak1p7_strong4_reprompt_greedy_t2_k1_20261005](../../results/westlake/opd_eval_ablation/weak1p7_strong4_reprompt_greedy_t2_k1_20261005) |
| opd_eval_ablation/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_ablation/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005](../../results/westlake/opd_eval_ablation/weak1p7_strong4_reprompt_greedy_t2_k1_postlogit_20261005) |
| opd_eval_ablation/weak1p7_strong8_reprompt_greedy_t2_k1_20261005 | 仅元数据/阶段标记 | 1 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_ablation/weak1p7_strong8_reprompt_greedy_t2_k1_20261005](../../results/westlake/opd_eval_ablation/weak1p7_strong8_reprompt_greedy_t2_k1_20261005) |
| opd_eval_ablation/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_ablation/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005](../../results/westlake/opd_eval_ablation/weak1p7_strong8_reprompt_greedy_t2_k1_postlogit_20261005) |
| opd_eval_ablation/weak_reprompt_clean_greedy_t2_k1_20260921 | 已有结果汇总 | 5 | 16 | dataset=bbh；n=1080，correct=368，accuracy=0.340741<br>dataset=cruxeval_i；n=800，correct=58，accuracy=0.0725 | [results/westlake/opd_eval_ablation/weak_reprompt_clean_greedy_t2_k1_20260921/summary.csv](../../results/westlake/opd_eval_ablation/weak_reprompt_clean_greedy_t2_k1_20260921/summary.csv)、[results/westlake/opd_eval_ablation/weak_reprompt_clean_greedy_t2_k1_20260921/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_clean_greedy_t2_k1_20260921/summary.json) |
| opd_eval_ablation/weak_reprompt_clean_t2_k1_20260921 | 已有结果汇总 | 5 | 16 | dataset=bbh；n=1080，correct=366，accuracy=0.338889<br>dataset=cruxeval_i；n=800，correct=51，accuracy=0.06375 | [results/westlake/opd_eval_ablation/weak_reprompt_clean_t2_k1_20260921/summary.csv](../../results/westlake/opd_eval_ablation/weak_reprompt_clean_t2_k1_20260921/summary.csv)、[results/westlake/opd_eval_ablation/weak_reprompt_clean_t2_k1_20260921/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_clean_t2_k1_20260921/summary.json) |
| opd_eval_ablation/weak_reprompt_nodraft_greedy_t2_k1_20260921 | 已有结果汇总 | 5 | 16 | dataset=bbh；n=1080，correct=761，accuracy=0.70463<br>dataset=cruxeval_i；n=800，correct=71，accuracy=0.08875 | [results/westlake/opd_eval_ablation/weak_reprompt_nodraft_greedy_t2_k1_20260921/summary.csv](../../results/westlake/opd_eval_ablation/weak_reprompt_nodraft_greedy_t2_k1_20260921/summary.csv)、[results/westlake/opd_eval_ablation/weak_reprompt_nodraft_greedy_t2_k1_20260921/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_nodraft_greedy_t2_k1_20260921/summary.json) |
| opd_eval_ablation/weak_reprompt_nodraft_t2_k1_20260920 | 已有结果汇总 | 5 | 16 | dataset=bbh；n=1080，correct=702，accuracy=0.65<br>dataset=cruxeval_i；n=800，correct=52，accuracy=0.065 | [results/westlake/opd_eval_ablation/weak_reprompt_nodraft_t2_k1_20260920/summary.csv](../../results/westlake/opd_eval_ablation/weak_reprompt_nodraft_t2_k1_20260920/summary.csv)、[results/westlake/opd_eval_ablation/weak_reprompt_nodraft_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_nodraft_t2_k1_20260920/summary.json) |
| opd_eval_ablation/weak_reprompt_t2_k1_20260920 | 已有结果汇总 | 7 | 16 | dataset=bbh；n=1080，correct=261，accuracy=0.241667<br>dataset=cruxeval_i；n=800，correct=35，accuracy=0.04375 | [results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.csv](../../results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.csv)、[results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_reprompt_t2_k1_20260920/summary.json) |
| opd_eval_ablation/weak_resample_greedy_t2_k1_20260921 | 已有结果汇总 | 5 | 16 | dataset=bbh；n=1080，correct=401，accuracy=0.371296<br>dataset=cruxeval_i；n=800，correct=154，accuracy=0.1925 | [results/westlake/opd_eval_ablation/weak_resample_greedy_t2_k1_20260921/summary.csv](../../results/westlake/opd_eval_ablation/weak_resample_greedy_t2_k1_20260921/summary.csv)、[results/westlake/opd_eval_ablation/weak_resample_greedy_t2_k1_20260921/summary.json](../../results/westlake/opd_eval_ablation/weak_resample_greedy_t2_k1_20260921/summary.json) |
| opd_eval_ablation/weak_resample_t2_k1_20260920 | 已有结果汇总 | 5 | 16 | dataset=bbh；n=1080，correct=410，accuracy=0.37963<br>dataset=cruxeval_i；n=800，correct=142，accuracy=0.1775 | [results/westlake/opd_eval_ablation/weak_resample_t2_k1_20260920/summary.csv](../../results/westlake/opd_eval_ablation/weak_resample_t2_k1_20260920/summary.csv)、[results/westlake/opd_eval_ablation/weak_resample_t2_k1_20260920/summary.json](../../results/westlake/opd_eval_ablation/weak_resample_t2_k1_20260920/summary.json) |
| opd_eval_baselines/full_teacher_student_greedy_20260920 | 已有结果汇总 | 9 | 78 | dataset=cruxeval_i / role=student；n=800，correct=159，accuracy=0.19875<br>dataset=cruxeval_i / role=teacher；n=800，correct=393，accuracy=0.49125 | [results/westlake/opd_eval_baselines/full_teacher_student_greedy_20260920/code_summary.csv](../../results/westlake/opd_eval_baselines/full_teacher_student_greedy_20260920/code_summary.csv)、[results/westlake/opd_eval_baselines/full_teacher_student_greedy_20260920/code_summary.json](../../results/westlake/opd_eval_baselines/full_teacher_student_greedy_20260920/code_summary.json) |
| opd_eval_baselines/full_teacher_student_x8_20260918 | 已有结果汇总 | 5 | 160 | dataset=bbh / role=student；n=8640，correct=3718，accuracy=0.430324<br>dataset=bbh / role=teacher；n=8640，correct=6902，accuracy=0.798843 | [results/westlake/opd_eval_baselines/full_teacher_student_x8_20260918/summary.csv](../../results/westlake/opd_eval_baselines/full_teacher_student_x8_20260918/summary.csv)、[results/westlake/opd_eval_baselines/full_teacher_student_x8_20260918/summary.json](../../results/westlake/opd_eval_baselines/full_teacher_student_x8_20260918/summary.json) |
| opd_eval_baselines/qwen3_greedy_remaining_20261005 | 已有结果汇总 | 4 | 12 | dataset=math500；questions=500，correct=412，accuracy=0.824，mean_tokens=2307.85<br>dataset=cruxeval_o；questions=800，correct=297，accuracy=0.37125，mean_tokens=65.5738 | [results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_4b/summary.json)、[results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_8b/summary.json](../../results/westlake/opd_eval_baselines/qwen3_greedy_remaining_20261005/qwen3_8b/summary.json) |
| opd_eval_code/code_grid_x8_20260916 | 已有结果汇总 | 178 | 1250 | config=k16_t2 / dataset=cruxeval_i；n=6400，correct=3153，accuracy=0.492656<br>config=k16_t2.5 / dataset=cruxeval_i；n=6400，correct=3135，accuracy=0.489844 | [results/westlake/opd_eval_code/code_grid_x8_20260916/x8_score_summary.csv](../../results/westlake/opd_eval_code/code_grid_x8_20260916/x8_score_summary.csv)、[results/westlake/opd_eval_code/code_grid_x8_20260916/x8_score_summary.json](../../results/westlake/opd_eval_code/code_grid_x8_20260916/x8_score_summary.json) |
| opd_eval_code/code_reasoning_grid_20260913 | 已有结果汇总 | 86 | 135 | config=student_only / dataset=cruxeval_o；n=800，correct=175，acc=0.21875，mean_output_tokens=402.012，mean_interventions=0，teacher_token_fraction=0<br>config=student_only / dataset=cruxeval_i；n=800，correct=158，acc=0.1975，mean_output_tokens=349.055，mean_interventions=0，teacher_token_fraction=0 | [results/westlake/opd_eval_code/code_reasoning_grid_20260913/FINAL_COMPARISON.csv](../../results/westlake/opd_eval_code/code_reasoning_grid_20260913/FINAL_COMPARISON.csv) |
| opd_eval_deepmath/full_seed20260809 | 已有结果汇总 | 18 | 286 | dataset=deepmath_103k；accuracy=0.18<br>dataset=deepmath_103k；accuracy=0.192 | [results/westlake/opd_eval_deepmath/full_seed20260809/diagnostic/shard_0/summary.json](../../results/westlake/opd_eval_deepmath/full_seed20260809/diagnostic/shard_0/summary.json)、[results/westlake/opd_eval_deepmath/full_seed20260809/diagnostic/shard_1/summary.json](../../results/westlake/opd_eval_deepmath/full_seed20260809/diagnostic/shard_1/summary.json) |
| opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910 | 已有结果汇总 | 44 | 762 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/analysis_n8x8/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/analysis_n8x8/summary.json)、[results/westlake/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/anchors_luna/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/anchors_luna/summary.json) |
| opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811 | 已有结果汇总 | 12 | 42 | model_label=qwen3_14b；rows=1000，correct=436，accuracy=0.436，mean_response_tokens=1439.99<br>model_label=qwen3_14b；rows=307，correct=162，accuracy=0.527687，mean_response_tokens=1499.81 | [results/westlake/opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811/comparison/comparison.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811/comparison/comparison.json)、[results/westlake/opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811/qwen3_14b/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811/qwen3_14b/merged/summary.json) |
| opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811 | 已有结果汇总 | 11 | 10 | rows=1000，correct=625，accuracy=0.625，mean_response_tokens=10118.2<br>rows=307，correct=211，accuracy=0.687296，mean_response_tokens=10025.9 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811/merged/summary.json)、[results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811/sampled_token_gap_visualization/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811/sampled_token_gap_visualization/summary.json) |
| opd_eval_deepmath/ge6_test1000_qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_seed20260827_eval_seed20260811 | 已有结果汇总 | 5 | 40 | model_label=qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_step118；rows=1000，correct=271，accuracy=0.271，mean_response_tokens=1634.73<br>model_label=qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_step118；rows=307，correct=102，accuracy=0.332248，mean_response_tokens=1708.17 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_seed20260827_eval_seed20260811/step_118/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_seed20260827_eval_seed20260811/step_118/merged/summary.json)、[results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_seed20260827_eval_seed20260811/step_236/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_seed20260827_eval_seed20260811/step_236/merged/summary.json) |
| opd_eval_deepmath/ge6_test1000_qwen3_1_7b_bridge_h256_n8_student_top16_rkl_seed20260827_eval_seed20260811 | 已有结果汇总 | 5 | 40 | model_label=qwen3_1_7b_bridge_h256_n8_student_top16_rkl_step1324；rows=1000，correct=313，accuracy=0.313，mean_response_tokens=1799.53<br>model_label=qwen3_1_7b_bridge_h256_n8_student_top16_rkl_step1324；rows=307，correct=130，accuracy=0.423453，mean_response_tokens=1808 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_bridge_h256_n8_student_top16_rkl_seed20260827_eval_seed20260811/step_1324/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_bridge_h256_n8_student_top16_rkl_seed20260827_eval_seed20260811/step_1324/merged/summary.json)、[results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_bridge_h256_n8_student_top16_rkl_seed20260827_eval_seed20260811/step_1656/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_bridge_h256_n8_student_top16_rkl_seed20260827_eval_seed20260811/step_1656/merged/summary.json) |
| opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step106_seed20260811 | 已有结果汇总 | 1 | 16 | checkpoint_step=106；rows=1000，correct=706，accuracy=0.706，mean_response_tokens=12687.4 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step106_seed20260811/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step106_seed20260811/merged/summary.json) |
| opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step133_seed20260811 | 已有结果汇总 | 1 | 16 | checkpoint_step=133；rows=1000，correct=674，accuracy=0.674，mean_response_tokens=12595.2 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step133_seed20260811/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step133_seed20260811/merged/summary.json) |
| opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step27_seed20260811 | 已有结果汇总 | 1 | 16 | checkpoint_step=27；rows=1000，correct=650，accuracy=0.65，mean_response_tokens=13218.6 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step27_seed20260811/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step27_seed20260811/merged/summary.json) |
| opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step53_seed20260811 | 已有结果汇总 | 1 | 16 | rows=1000，correct=695，accuracy=0.695，mean_response_tokens=13261.7 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step53_seed20260811/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step53_seed20260811/merged/summary.json) |
| opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step80_seed20260811 | 已有结果汇总 | 1 | 16 | checkpoint_step=80；rows=1000，correct=701，accuracy=0.701，mean_response_tokens=12765.2 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step80_seed20260811/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step80_seed20260811/merged/summary.json) |
| opd_eval_deepmath/ge6_test1000_qwen3_1_7b_train25pct_nonthinking_top16_rkl_seed20260811 | 已有结果汇总 | 5 | 40 | model_label=qwen3_1_7b_train25pct_nonthinking_top16_rkl_step133；rows=1000，correct=302，accuracy=0.302，mean_response_tokens=1753.73<br>model_label=qwen3_1_7b_train25pct_nonthinking_top16_rkl_step133；rows=307，correct=125，accuracy=0.407166，mean_response_tokens=1799.35 | [results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_train25pct_nonthinking_top16_rkl_seed20260811/step_133/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_train25pct_nonthinking_top16_rkl_seed20260811/step_133/merged/summary.json)、[results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_train25pct_nonthinking_top16_rkl_seed20260811/step_178/merged/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_train25pct_nonthinking_top16_rkl_seed20260811/step_178/merged/summary.json) |
| opd_eval_deepmath/ge6_test1000_student_only_nonthinking_x8_seed20260811 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_deepmath/ge6_test1000_student_only_nonthinking_x8_seed20260811](../../results/westlake/opd_eval_deepmath/ge6_test1000_student_only_nonthinking_x8_seed20260811) |
| opd_eval_deepmath/ge6_test1000_three_ckpts_nonthinking_x8_seed20260811 | 已有结果汇总 | 4 | 3 | model_label=bridge_opd_step1656；questions=1000，rollouts=8000，correct_rollouts=2382，accuracy=0.29775，pass_at_8=0.533，mean_response_tokens=1789.81<br>model_label=good_bridge_opd_step590；questions=1000，rollouts=8000，correct_rollouts=2418，accuracy=0.30225，pass_at_8=0.546，mean_response_tokens=1696.94 | [results/westlake/opd_eval_deepmath/ge6_test1000_three_ckpts_nonthinking_x8_seed20260811/bridge_opd_step1656/scored_v2/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_three_ckpts_nonthinking_x8_seed20260811/bridge_opd_step1656/scored_v2/summary.json)、[results/westlake/opd_eval_deepmath/ge6_test1000_three_ckpts_nonthinking_x8_seed20260811/good_bridge_opd_step590/scored_v2/summary.json](../../results/westlake/opd_eval_deepmath/ge6_test1000_three_ckpts_nonthinking_x8_seed20260811/good_bridge_opd_step590/scored_v2/summary.json) |
| opd_eval_deepmath/maxlen_seed20260809 | 已有结果汇总 | 31 | 296 | dataset=deepmath_103k；accuracy=0.852<br>dataset=deepmath_103k；accuracy=0.872 | [results/westlake/opd_eval_deepmath/maxlen_seed20260809/diagnostic/shard_0/summary.json](../../results/westlake/opd_eval_deepmath/maxlen_seed20260809/diagnostic/shard_0/summary.json)、[results/westlake/opd_eval_deepmath/maxlen_seed20260809/diagnostic/shard_1/summary.json](../../results/westlake/opd_eval_deepmath/maxlen_seed20260809/diagnostic/shard_1/summary.json) |
| opd_eval_deepmath/maxlen_smoke_86139 | 已有结果汇总 | 4 | 2 | dataset=deepmath_103k；accuracy=0<br>dataset=deepmath_103k；accuracy=1 | [results/westlake/opd_eval_deepmath/maxlen_smoke_86139/Qwen3-1.7B/summary.json](../../results/westlake/opd_eval_deepmath/maxlen_smoke_86139/Qwen3-1.7B/summary.json)、[results/westlake/opd_eval_deepmath/maxlen_smoke_86139/Qwen3-4B/summary.json](../../results/westlake/opd_eval_deepmath/maxlen_smoke_86139/Qwen3-4B/summary.json) |
| opd_eval_deepmath/online_chunk64_k32_test1000_h100_seed20260911 | 已有结果汇总 | 3 | 1 | questions=1000，correct=504，accuracy=0.504 | [results/westlake/opd_eval_deepmath/online_chunk64_k32_test1000_h100_seed20260911/results/summary.json](../../results/westlake/opd_eval_deepmath/online_chunk64_k32_test1000_h100_seed20260911/results/summary.json) |
| opd_eval_deepmath/online_chunk64_k8_smoke20260911 | 已有结果汇总 | 4 | 1 | questions=8，correct=0，accuracy=0 | [results/westlake/opd_eval_deepmath/online_chunk64_k8_smoke20260911/results/summary.json](../../results/westlake/opd_eval_deepmath/online_chunk64_k8_smoke20260911/results/summary.json) |
| opd_eval_deepmath/online_chunk64_k8_test1000_2nodes_seed20260911 | 已有结果汇总 | 5 | 1 | questions=1000，correct=476，accuracy=0.476 | [results/westlake/opd_eval_deepmath/online_chunk64_k8_test1000_2nodes_seed20260911/results/summary.json](../../results/westlake/opd_eval_deepmath/online_chunk64_k8_test1000_2nodes_seed20260911/results/summary.json) |
| opd_eval_deepmath/online_chunk64_k8_test1000_seed20260911 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_deepmath/online_chunk64_k8_test1000_seed20260911](../../results/westlake/opd_eval_deepmath/online_chunk64_k8_test1000_seed20260911) |
| opd_eval_deepmath/random_k8_cap8_test1000_h100_seed20260911 | 已有结果汇总 | 3 | 1 | questions=1000，correct=426，accuracy=0.426 | [results/westlake/opd_eval_deepmath/random_k8_cap8_test1000_h100_seed20260911/results/summary.json](../../results/westlake/opd_eval_deepmath/random_k8_cap8_test1000_h100_seed20260911/results/summary.json) |
| opd_eval_deepmath/smoke_1_7b_86081 | 已有结果汇总 | 2 | 1 | dataset=deepmath_103k；accuracy=0 | [results/westlake/opd_eval_deepmath/smoke_1_7b_86081/summary.json](../../results/westlake/opd_eval_deepmath/smoke_1_7b_86081/summary.json) |
| opd_eval_deepmath/smoke_86074 | 已有结果汇总 | 2 | 1 | dataset=deepmath_103k；accuracy=0 | [results/westlake/opd_eval_deepmath/smoke_86074/qwen3_4b/summary.json](../../results/westlake/opd_eval_deepmath/smoke_86074/qwen3_4b/summary.json) |
| opd_eval_deepmath/student_only64_20260910 | 已有结果汇总 | 4 | 1 | questions=1000，rollouts=64000，correct=17914，accuracy=0.279906，pass_at_64=0.626，mean_tokens=1321.06 | [results/westlake/opd_eval_deepmath/student_only64_20260910/summary.json](../../results/westlake/opd_eval_deepmath/student_only64_20260910/summary.json) |
| opd_eval_deepmath/takeover_baseline_comparison | 已有结果汇总 | 1 | 19 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_deepmath/takeover_baseline_comparison/summary.json](../../results/westlake/opd_eval_deepmath/takeover_baseline_comparison/summary.json) |
| opd_eval_deepmath/teacher8_student8_20260909 | 已有结果汇总 | 1 | 14 | method=teacher；rows=8000，correct=3399，accuracy=0.424875，pass_any=0.615<br>method=student；rows=64000，correct=18109，accuracy=0.282953，pass_any=0.512875 | [results/westlake/opd_eval_deepmath/teacher8_student8_20260909/summary.json](../../results/westlake/opd_eval_deepmath/teacher8_student8_20260909/summary.json) |
| opd_eval_deepmath/teacher8_student8_20260909_smoke | 已有结果汇总 | 1 | 14 | method=teacher；rows=8，correct=7，accuracy=0.875，pass_any=1<br>method=student；rows=64，correct=29，accuracy=0.453125，pass_any=1 | [results/westlake/opd_eval_deepmath/teacher8_student8_20260909_smoke/summary.json](../../results/westlake/opd_eval_deepmath/teacher8_student8_20260909_smoke/summary.json) |
| opd_eval_deepmath/teacher_prefix_handoff_20260909 | 已有结果汇总 | 1 | 4 | h=32；n=1000，correct=293，accuracy=0.293<br>h=64；n=1000，correct=283，accuracy=0.283 | [results/westlake/opd_eval_deepmath/teacher_prefix_handoff_20260909/summary.json](../../results/westlake/opd_eval_deepmath/teacher_prefix_handoff_20260909/summary.json) |
| opd_eval_deepmath/teacher_prefix_handoff_20260909_smoke | 已有结果汇总 | 5 | 4 | h=32；n=8，correct=2，accuracy=0.25<br>h=64；n=8，correct=5，accuracy=0.625 | [results/westlake/opd_eval_deepmath/teacher_prefix_handoff_20260909_smoke/summary.json](../../results/westlake/opd_eval_deepmath/teacher_prefix_handoff_20260909_smoke/summary.json) |
| opd_eval_dynamic/dynamic_k_entropy_20260914 | 已有结果汇总 | 10 | 16 | dataset=bbh；n=1080，correct=834，acc=0.772222，mean_output_tokens=263.095，mean_interventions=17.4907，teacher_token_fraction=0.0714711<br>dataset=cruxeval_i；n=800，correct=392，acc=0.49，mean_output_tokens=18.345，mean_interventions=3.785，teacher_token_fraction=0.227719 | [results/westlake/opd_eval_dynamic/dynamic_k_entropy_20260914/summary.csv](../../results/westlake/opd_eval_dynamic/dynamic_k_entropy_20260914/summary.csv)、[results/westlake/opd_eval_dynamic/dynamic_k_entropy_20260914/summary.json](../../results/westlake/opd_eval_dynamic/dynamic_k_entropy_20260914/summary.json) |
| opd_eval_hard/bbh_official_20260911 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_hard/bbh_official_20260911](../../results/westlake/opd_eval_hard/bbh_official_20260911) |
| opd_eval_hard/hard_k32_20260911 | 已有结果汇总 | 4 | 4 | method=online / dataset=aime25；questions=30，correct=3，accuracy=0.1<br>method=online / dataset=aime24；questions=30，correct=4，accuracy=0.133333 | [results/westlake/opd_eval_hard/hard_k32_20260911/summary.json](../../results/westlake/opd_eval_hard/hard_k32_20260911/summary.json) |
| opd_eval_hard/hard_three_20260911 | 已有结果汇总 | 5 | 12 | method=online / dataset=aime25；questions=30，correct=7，accuracy=0.233333<br>method=online / dataset=aime24；questions=30，correct=3，accuracy=0.1 | [results/westlake/opd_eval_hard/hard_three_20260911/summary.json](../../results/westlake/opd_eval_hard/hard_three_20260911/summary.json) |
| opd_eval_hard/math_reasoning_grid_x8_20260916 | 仅元数据/阶段标记 | 176 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_hard/math_reasoning_grid_x8_20260916](../../results/westlake/opd_eval_hard/math_reasoning_grid_x8_20260916) |
| opd_eval_hard/nojudge_grid_20260912 | 已有结果汇总 | 54 | 276 | config=k1_t2 / dataset=math500；n=500，correct=419，accuracy=0.838，mean_tokens=842.026<br>config=k1_t2 / dataset=olympiad_bench；n=674，correct=299，accuracy=0.44362，mean_tokens=1510.33 | [results/westlake/opd_eval_hard/nojudge_grid_20260912/audit_summary.json](../../results/westlake/opd_eval_hard/nojudge_grid_20260912/audit_summary.json)、[results/westlake/opd_eval_hard/nojudge_grid_20260912/combined_completed_metrics.json](../../results/westlake/opd_eval_hard/nojudge_grid_20260912/combined_completed_metrics.json) |
| opd_eval_hard/qwen3_greedy_hard_three_20261005 | 已有结果汇总 | 2 | 8 | dataset=aime24；questions=30，correct=5，accuracy=0.166667<br>dataset=aime25；questions=30，correct=4，accuracy=0.133333 | [results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_4b/summary.json](../../results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_4b/summary.json)、[results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_8b/summary.json](../../results/westlake/opd_eval_hard/qwen3_greedy_hard_three_20261005/qwen3_8b/summary.json) |
| opd_eval_hard/random_k8_max8_20260911 | 已有结果汇总 | 4 | 5 | method=online / dataset=aime25；questions=30，correct=5，accuracy=0.166667<br>method=online / dataset=aime24；questions=30，correct=3，accuracy=0.1 | [results/westlake/opd_eval_hard/random_k8_max8_20260911/summary.json](../../results/westlake/opd_eval_hard/random_k8_max8_20260911/summary.json) |
| opd_eval_hard/takeover_ablation_20260911 | 仅元数据/阶段标记 | 6 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_hard/takeover_ablation_20260911](../../results/westlake/opd_eval_hard/takeover_ablation_20260911) |
| opd_eval_hard/threshold2_luna_k1_20260911 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_hard/threshold2_luna_k1_20260911](../../results/westlake/opd_eval_hard/threshold2_luna_k1_20260911) |
| opd_eval_hard/threshold3_mathteacher_20260911 | 已有结果汇总 | 4 | 6 | method=online / dataset=aime25；questions=30，correct=6，accuracy=0.2<br>method=online / dataset=aime24；questions=30，correct=7，accuracy=0.233333 | [results/westlake/opd_eval_hard/threshold3_mathteacher_20260911/summary.json](../../results/westlake/opd_eval_hard/threshold3_mathteacher_20260911/summary.json) |
| opd_eval_hard/threshold_k8_nojudge_20260911 | 已有结果汇总 | 3 | 5 | method=online / dataset=aime25；questions=30，correct=6，accuracy=0.2<br>method=online / dataset=aime24；questions=30，correct=3，accuracy=0.1 | [results/westlake/opd_eval_hard/threshold_k8_nojudge_20260911/summary.json](../../results/westlake/opd_eval_hard/threshold_k8_nojudge_20260911/summary.json) |
| opd_eval_math500/data | 仅元数据/阶段标记 | 1 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_eval_math500/data](../../results/westlake/opd_eval_math500/data) |
| opd_eval_math500/math500_three_20260911 | 已有结果汇总 | 3 | 3 | method=student；questions=500，correct=364，accuracy=0.728，mean_tokens=885.078<br>method=online；questions=500，correct=407，accuracy=0.814，mean_tokens=882.754 | [results/westlake/opd_eval_math500/math500_three_20260911/summary.json](../../results/westlake/opd_eval_math500/math500_three_20260911/summary.json) |
| opd_eval_prefix/prefix_takeover_caps11_20_20260917 | 已有结果汇总 | 4 | 160 | dataset=bbh；n=800，correct=485，accuracy=0.60625<br>dataset=bbh；n=800，correct=495，accuracy=0.61875 | [results/westlake/opd_eval_prefix/prefix_takeover_caps11_20_20260917/summary.csv](../../results/westlake/opd_eval_prefix/prefix_takeover_caps11_20_20260917/summary.csv)、[results/westlake/opd_eval_prefix/prefix_takeover_caps11_20_20260917/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps11_20_20260917/summary.json) |
| opd_eval_prefix/prefix_takeover_caps25_40_20260918 | 已有结果汇总 | 5 | 84 | dataset=cruxeval_i；n=800，correct=393，accuracy=0.49125<br>dataset=cruxeval_i；n=800，correct=393，accuracy=0.49125 | [results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/code_summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/code_summary.json)、[results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.csv](../../results/westlake/opd_eval_prefix/prefix_takeover_caps25_40_20260918/summary.csv) |
| opd_eval_prefix/prefix_takeover_caps5_10_20260917 | 已有结果汇总 | 4 | 96 | dataset=bbh；n=800，correct=391，accuracy=0.48875<br>dataset=bbh；n=800，correct=417，accuracy=0.52125 | [results/westlake/opd_eval_prefix/prefix_takeover_caps5_10_20260917/summary.csv](../../results/westlake/opd_eval_prefix/prefix_takeover_caps5_10_20260917/summary.csv)、[results/westlake/opd_eval_prefix/prefix_takeover_caps5_10_20260917/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps5_10_20260917/summary.json) |
| opd_eval_prefix/prefix_takeover_caps_20260917 | 已有结果汇总 | 4 | 80 | dataset=bbh；n=800，correct=344，accuracy=0.43<br>dataset=bbh；n=800，correct=340，accuracy=0.425 | [results/westlake/opd_eval_prefix/prefix_takeover_caps_20260917/summary.csv](../../results/westlake/opd_eval_prefix/prefix_takeover_caps_20260917/summary.csv)、[results/westlake/opd_eval_prefix/prefix_takeover_caps_20260917/summary.json](../../results/westlake/opd_eval_prefix/prefix_takeover_caps_20260917/summary.json) |
| opd_eval_prefix/quartile_support_k_20260918 | 已有结果汇总 | 11 | 1392 | dataset=bbh / quartile=all；n=100，correct=42，accuracy=0.42<br>dataset=bbh / quartile=1；n=25，correct=9，accuracy=0.36 | [results/westlake/opd_eval_prefix/quartile_support_k_20260918/k0_summary.csv](../../results/westlake/opd_eval_prefix/quartile_support_k_20260918/k0_summary.csv)、[results/westlake/opd_eval_prefix/quartile_support_k_20260918/k0_summary.json](../../results/westlake/opd_eval_prefix/quartile_support_k_20260918/k0_summary.json) |
| opd_eval_prefix/teacher_only_sample100_x8_20260917 | 已有结果汇总 | 4 | 16 | condition=teacher_only / dataset=bbh；n=800，correct=617，accuracy=0.77125<br>condition=teacher_only / dataset=cruxeval_i；n=800，correct=400，accuracy=0.5 | [results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.csv](../../results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.csv)、[results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json](../../results/westlake/opd_eval_prefix/teacher_only_sample100_x8_20260917/summary.json) |
| opd_logs/cache_queue_cleanup_20260920_164150 | 仅元数据/阶段标记 | 37 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_logs/cache_queue_cleanup_20260920_164150](../../results/westlake/opd_logs/cache_queue_cleanup_20260920_164150) |
| opd_rescore/baseline_math_gpt4o_mini_openlux_20260916 | 已有结果汇总 | 4 | 24 | 未定位准确率/主指标汇总 | [results/westlake/opd_rescore/baseline_math_gpt4o_mini_openlux_20260916/baseline_comparison.csv](../../results/westlake/opd_rescore/baseline_math_gpt4o_mini_openlux_20260916/baseline_comparison.csv)、[results/westlake/opd_rescore/baseline_math_gpt4o_mini_openlux_20260916/baseline_comparison.json](../../results/westlake/opd_rescore/baseline_math_gpt4o_mini_openlux_20260916/baseline_comparison.json) |
| opd_rescore/grid_math_gpt4o_mini_openlux_20260916 | 已有结果汇总 | 4 | 162 | dataset=bbh / config=k4_t2；accuracy=77.7778<br>dataset=bbh / config=k16_t2；accuracy=80 | [results/westlake/opd_rescore/grid_math_gpt4o_mini_openlux_20260916/comparison/summary.json](../../results/westlake/opd_rescore/grid_math_gpt4o_mini_openlux_20260916/comparison/summary.json)、[results/westlake/opd_rescore/grid_math_gpt4o_mini_openlux_20260916/summary.csv](../../results/westlake/opd_rescore/grid_math_gpt4o_mini_openlux_20260916/summary.csv) |
| opd_rescore/math_grid_score_029041_20260919 | 已有结果汇总 | 1 | 45 | config=r1_k1_t2 / dataset=bbh；n=1080，correct=864，accuracy=0.8<br>config=r1_k1_t2 / dataset=math500；n=500，correct=415，accuracy=0.83 | [results/westlake/opd_rescore/math_grid_score_029041_20260919/summary.json](../../results/westlake/opd_rescore/math_grid_score_029041_20260919/summary.json) |
| opd_rescore/math_grid_score_all_2000_20260919 | 已有结果汇总 | 1 | 525 | config=r1_k1_t2 / dataset=olympiad_bench；n=674，correct=365，accuracy=0.541543<br>config=r1_k1_t2 / dataset=bbh；n=1080，correct=864，accuracy=0.8 | [results/westlake/opd_rescore/math_grid_score_all_2000_20260919/summary.json](../../results/westlake/opd_rescore/math_grid_score_all_2000_20260919/summary.json) |
| opd_rescore/r1_math_score_20260918 | 已有结果汇总 | 1 | 3 | dataset=bbh；n=1080，correct=864，accuracy=0.8<br>dataset=math500；n=500，correct=415，accuracy=0.83 | [results/westlake/opd_rescore/r1_math_score_20260918/summary.json](../../results/westlake/opd_rescore/r1_math_score_20260918/summary.json) |
| opd_rollouts/deepmath_ge6_train25pct_qwen3_14b_native_nonthinking_seed20260811 | 仅元数据/阶段标记 | 1 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_rollouts/deepmath_ge6_train25pct_qwen3_14b_native_nonthinking_seed20260811](../../results/westlake/opd_rollouts/deepmath_ge6_train25pct_qwen3_14b_native_nonthinking_seed20260811) |
| opd_rollouts/deepmath_ge6_train25pct_qwen3_14b_native_nonthinking_seed20260811_single_node_8h800 | 已有结果汇总 | 19 | 39 | rows=14208，correct=6062，accuracy=0.426661<br>rows=1，correct=0，accuracy=0 | [results/westlake/opd_rollouts/deepmath_ge6_train25pct_qwen3_14b_native_nonthinking_seed20260811_single_node_8h800/scoring_rule_v1/summary.json](../../results/westlake/opd_rollouts/deepmath_ge6_train25pct_qwen3_14b_native_nonthinking_seed20260811_single_node_8h800/scoring_rule_v1/summary.json)、[results/westlake/opd_rollouts/deepmath_ge6_train25pct_qwen3_14b_native_nonthinking_seed20260811_single_node_8h800/summary.json](../../results/westlake/opd_rollouts/deepmath_ge6_train25pct_qwen3_14b_native_nonthinking_seed20260811_single_node_8h800/summary.json) |
| opd_rollouts/deepmath_ge6_train25pct_qwen3_1_7b_native_nonthinking_seed20260811 | 已有结果汇总 | 19 | 39 | rows=14208，correct=3953，accuracy=0.278224<br>rows=1，correct=0，accuracy=0 | [results/westlake/opd_rollouts/deepmath_ge6_train25pct_qwen3_1_7b_native_nonthinking_seed20260811/scoring_rule_v1/summary.json](../../results/westlake/opd_rollouts/deepmath_ge6_train25pct_qwen3_1_7b_native_nonthinking_seed20260811/scoring_rule_v1/summary.json)、[results/westlake/opd_rollouts/deepmath_ge6_train25pct_qwen3_1_7b_native_nonthinking_seed20260811/summary.json](../../results/westlake/opd_rollouts/deepmath_ge6_train25pct_qwen3_1_7b_native_nonthinking_seed20260811/summary.json) |
| opd_runs/qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_seed20260827 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_runs/qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_seed20260827](../../results/westlake/opd_runs/qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_seed20260827) |
| opd_runs/qwen3_1_7b_bridge_h256_n8_student_top16_rkl_seed20260827 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_runs/qwen3_1_7b_bridge_h256_n8_student_top16_rkl_seed20260827](../../results/westlake/opd_runs/qwen3_1_7b_bridge_h256_n8_student_top16_rkl_seed20260827) |
| opd_runs/qwen3_1_7b_from_14b_deepmath_ge6_full_vocab_rkl | 仅元数据/阶段标记 | 1 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_runs/qwen3_1_7b_from_14b_deepmath_ge6_full_vocab_rkl](../../results/westlake/opd_runs/qwen3_1_7b_from_14b_deepmath_ge6_full_vocab_rkl) |
| opd_runs/qwen3_1_7b_from_14b_deepmath_ge6_train15pct_student_top16_rkl_16h800 | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_runs/qwen3_1_7b_from_14b_deepmath_ge6_train15pct_student_top16_rkl_16h800](../../results/westlake/opd_runs/qwen3_1_7b_from_14b_deepmath_ge6_train15pct_student_top16_rkl_16h800) |
| opd_runs/qwen3_1_7b_from_14b_deepmath_ge6_train25pct_nonthinking_student_top16_rkl_8h800_single_node | 仅元数据/阶段标记 | 2 | 0 | 未定位准确率/主指标汇总 | [results/westlake/opd_runs/qwen3_1_7b_from_14b_deepmath_ge6_train25pct_nonthinking_student_top16_rkl_8h800_single_node](../../results/westlake/opd_runs/qwen3_1_7b_from_14b_deepmath_ge6_train25pct_nonthinking_student_top16_rkl_8h800_single_node) |

历史 DeepMath 评分存在布尔答案归一化修订；源汇总与 2026-09-11 统一比较表可能分别保存旧、新口径。本目录保留历史值和版本路径，不覆盖旧数值；跨协议/评分版本比较须回到原报告的 scoring 定义。较新的统一比较见 [reports/all_benchmark_comparison_20260911/REPORT.md](../../reports/all_benchmark_comparison_20260911/REPORT.md)。

## 结果解读与后续工作

1. Nonthinking held-out 1000 中，Good-Bridge OPD 相对 Student 的 rollout 准确率为 30.225% 对 28.0875%（+2.1375 个百分点，相对 +7.61%），但未保存多训练种子证据；不应仅凭八次采样解释训练稳定性。
2. 主方法的 greedy 表已覆盖八个 benchmark；随机介入、延迟 shift5、reverse 与 reprompt/resample 均有独立结果源，不能仅保留有利子组。
3. 同前缀熵和支持度结果只支持统计描述。要测试“路径修复”的解释，下一步是按已冻结计划补完语义盲标、反事实 pilot 和确认集；本次目录整理未启动新实验。
4. 西湖已可连接；后续按需读取原始 rollouts、评分逐题文件、训练 checkpoints 和 logits payload，再核对 source SHA256、manifest 行数和完成标记。大型原始输出不随代码推送 GitHub。

## 再生成与校验

```bash
python3 tools/organization/build_experiment_catalog.py
```

脚本只读取本地证据并覆写本目录的生成文件；不访问网络、不发起评分、不提交训练。源路径使用整理前兼容路径，目录规整后由项目保留的 symlink 解析。机器清单保留每个索引源文件的 SHA256 便于核对版本。
本轮解析失败文件数：1。 失败入口：[reports/intervention_analysis_20260923.json](../../reports/intervention_analysis_20260923.json)（空文件或无效 JSON，未补造内容）。
