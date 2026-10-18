# 通用脚本分类索引

本目录保存通用数据准备、训练、生成、评分、汇总、验证、可视化和 Slurm 入口。脚本保持同级位置，避免破坏同目录导入、模板查找与历史作业路径。各实验自己的代码和启动记录位于根目录 `experiments/`；pipeline 位于 `ocrd_pipeline/`。

入口：[实验与结果总表](../docs/experiments/ALL_EXPERIMENTS.md)、[代码地图](../docs/project/CODE_MAP.md)、[西湖存储](../docs/storage/WESTLAKE.md)。本索引按 [code_inventory.csv](../docs/project/code_inventory.csv) 的 `scripts/` 条目生成；完整清单还包括 pipeline 和分析目录，类别用于查找职责。

## 运行约定

在仓库根目录执行本地分析或显式指定路径；长时间 GPU 任务通过 Slurm。旧作业文件大量使用 `/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/`，部署前核对该路径、模型、数据与 conda 环境。共享模型/数据只读，新增输出写入西湖项目根目录的分类子目录，遵循 [AGENT.md](../AGENT.md)。

同名 `run_*`、`submit_*`、`verify_*` 往往组成一个流程；smoke、formal、resume、merge 和 retry 文件记录不同阶段，不按相似名称视为重复脚本删除。`run_bridge_top16_reverse_kl_train.py` 导入同目录的 `run_deepmath_full_vocab_reverse_kl_train.py`，部署时应保持两者相邻。

部分训练入口在导入时读取数据并检查配置，先阅读该实验的 launcher 和配置再执行。当前没有统一且经过验证的根级 Python 环境锁文件，依赖与环境边界见代码地图。

## 类别总览

| 类别 | 文件数 | 用途 |
| --- | ---: | --- |
| 数据准备 | 13 | 数据下载、分层子集、评测分片、训练输入与 bridge selection。 |
| 训练与蒸馏 | 10 | Reverse-KL / student-top-k / bridge 训练及多 GPU launcher。 |
| 生成与运行编排 | 14 | baseline、checkpoint rollout、benchmark generation 与服务 launcher。 |
| 评分 | 4 | 数学最终答案及已生成 rollout 的评分。 |
| 分析与汇总 | 17 | shard 合并、样本/错误分析与结果比较。 |
| 可视化与导出 | 8 | 离线 HTML、token heatmap、bridge 浏览器数据和本地查看入口。 |
| 验证与预检查 | 14 | 输入、checkpoint、输出完整性、环境和评分行为检查。 |
| Slurm 启动文件 | 99 | 提交、重试、评分、合并和 smoke 作业配置。 |
| 维护工具 | 3 | 文件整理与其他维护脚本，执行前检查输入和修改范围。 |

## 数据准备

- [build_anchor_bifurcation_openrouter_inputs.py](build_anchor_bifurcation_openrouter_inputs.py)
- [build_deepmath_ge6_opd_data.py](build_deepmath_ge6_opd_data.py)
- [build_deepmath_ge6_train15pct.py](build_deepmath_ge6_train15pct.py)
- [build_deepmath_ge6_train25pct_perdifficulty.py](build_deepmath_ge6_train25pct_perdifficulty.py)
- [build_good_bridge_selection.py](build_good_bridge_selection.py)
- [build_gsm8k_math_opd_data.py](build_gsm8k_math_opd_data.py)
- [download_deepmath_103k.sh](download_deepmath_103k.sh)
- [make_opd_smoke_data.py](make_opd_smoke_data.py)
- [prepare_deepmath_ge6_eval_shards.py](prepare_deepmath_ge6_eval_shards.py)
- [prepare_deepmath_stratified_sample.py](prepare_deepmath_stratified_sample.py)
- [prepare_deepmath_train25_nonthinking_rollout.py](prepare_deepmath_train25_nonthinking_rollout.py)
- [scan_deepmath_full_prompt_lengths.py](scan_deepmath_full_prompt_lengths.py)
- [test1000_anchor_bridge_20260910/prepare_anchor_shards.py](test1000_anchor_bridge_20260910/prepare_anchor_shards.py)

## 训练与蒸馏

- [run_bridge_top16_8h800_single_node_launcher.sh](run_bridge_top16_8h800_single_node_launcher.sh)
- [run_bridge_top16_reverse_kl_train.py](run_bridge_top16_reverse_kl_train.py)
- [run_deepmath_full_vocab_reverse_kl_train.py](run_deepmath_full_vocab_reverse_kl_train.py)
- [run_deepmath_top16_16h800_launcher.sh](run_deepmath_top16_16h800_launcher.sh)
- [run_deepmath_top16_8h800_single_node_launcher.sh](run_deepmath_top16_8h800_single_node_launcher.sh)
- [run_deepmath_train25_nonthinking_rollout.py](run_deepmath_train25_nonthinking_rollout.py)
- [run_full_vocab_reverse_kl_smoke.py](run_full_vocab_reverse_kl_smoke.py)
- [run_qwen3_1_7b_from_14b_reverse_kl_deepmath_ge6.sh](run_qwen3_1_7b_from_14b_reverse_kl_deepmath_ge6.sh)
- [run_qwen3_4b_from_14b_gkd_gsm8k_math.sh](run_qwen3_4b_from_14b_gkd_gsm8k_math.sh)
- [run_qwen3_4b_from_14b_gkd_smoke.sh](run_qwen3_4b_from_14b_gkd_smoke.sh)

## 生成与运行编排

- [launch_anchor_bifurcation_openrouter.sh](launch_anchor_bifurcation_openrouter.sh)
- [run_anchor_bifurcation_openrouter.py](run_anchor_bifurcation_openrouter.py)
- [run_deepmath_baseline_sample_gap.py](run_deepmath_baseline_sample_gap.py)
- [run_deepmath_checkpoint_nonthinking_x8.py](run_deepmath_checkpoint_nonthinking_x8.py)
- [run_deepmath_student_top16_trace_diagnostic.py](run_deepmath_student_top16_trace_diagnostic.py)
- [run_opd_logit_diagnostic.py](run_opd_logit_diagnostic.py)
- [run_qwen3_1_7b_math500_baseline.sh](run_qwen3_1_7b_math500_baseline.sh)
- [run_qwen3_hard_eval.py](run_qwen3_hard_eval.py)
- [run_qwen3_hard_eval.sh](run_qwen3_hard_eval.sh)
- [run_qwen3_hard_eval_nonthinking.py](run_qwen3_hard_eval_nonthinking.py)
- [run_qwen3_hard_eval_nonthinking.sh](run_qwen3_hard_eval_nonthinking.sh)
- [run_qwen3_math500_baseline.py](run_qwen3_math500_baseline.py)
- [run_verl_val_only_qwen.sh](run_verl_val_only_qwen.sh)
- [run_vllm_api_no_metrics.py](run_vllm_api_no_metrics.py)

## 评分

- [eval_qwen_opd_models.py](eval_qwen_opd_models.py)
- [score_deepmath_checkpoint_nonthinking_x8.py](score_deepmath_checkpoint_nonthinking_x8.py)
- [score_deepmath_train25_nonthinking_rollout.py](score_deepmath_train25_nonthinking_rollout.py)
- [score_format_free_final_answer.py](score_format_free_final_answer.py)

## 分析与汇总

- [analyze_anchor_bifurcation_openlux.py](analyze_anchor_bifurcation_openlux.py)
- [analyze_deepmath_baseline_gap_position.py](analyze_deepmath_baseline_gap_position.py)
- [analyze_deepmath_nonthinking_unrecognized_tokens.py](analyze_deepmath_nonthinking_unrecognized_tokens.py)
- [analyze_format_free_failures.py](analyze_format_free_failures.py)
- [analyze_opd_eval.py](analyze_opd_eval.py)
- [analyze_sampled_reverse_kl.py](analyze_sampled_reverse_kl.py)
- [analyze_verl_val_only.py](analyze_verl_val_only.py)
- [compare_deepmath_ge6_native_nonthinking.py](compare_deepmath_ge6_native_nonthinking.py)
- [compare_response_length_evals.py](compare_response_length_evals.py)
- [merge_deepmath_checkpoint_nonthinking_x8.py](merge_deepmath_checkpoint_nonthinking_x8.py)
- [merge_deepmath_eval_diagnostic.py](merge_deepmath_eval_diagnostic.py)
- [merge_deepmath_ge6_baseline.py](merge_deepmath_ge6_baseline.py)
- [merge_deepmath_ge6_native_nonthinking.py](merge_deepmath_ge6_native_nonthinking.py)
- [merge_deepmath_ge6_step53_eval.py](merge_deepmath_ge6_step53_eval.py)
- [merge_deepmath_student_top16_trace_diagnostic.py](merge_deepmath_student_top16_trace_diagnostic.py)
- [merge_deepmath_teacher_rollouts.py](merge_deepmath_teacher_rollouts.py)
- [merge_deepmath_train25_nonthinking_scores.py](merge_deepmath_train25_nonthinking_scores.py)

## 可视化与导出

- [build_deepmath_17b_miss14b_token_vis.py](build_deepmath_17b_miss14b_token_vis.py)
- [build_deepmath_baseline_gap_vis.py](build_deepmath_baseline_gap_vis.py)
- [build_deepmath_student_top16_trace_vis.py](build_deepmath_student_top16_trace_vis.py)
- [build_deepmath_token_vis_payload.py](build_deepmath_token_vis_payload.py)
- [export_bridge_100pp_browser_data.py](export_bridge_100pp_browser_data.py)
- [export_opd_token_heatmap.py](export_opd_token_heatmap.py)
- [serve_deepmath_baseline_gap.py](serve_deepmath_baseline_gap.py)
- [sync_deepmath_baseline_gap_artifact.sh](sync_deepmath_baseline_gap_artifact.sh)

## 验证与预检查

- [inspect_math_final_answers.py](inspect_math_final_answers.py)
- [test_nonthinking_symbolic_timeout.py](test_nonthinking_symbolic_timeout.py)
- [validate_verl_opd_env.py](validate_verl_opd_env.py)
- [verify_bridge_top16_reverse_kl_train.py](verify_bridge_top16_reverse_kl_train.py)
- [verify_deepmath_eval.py](verify_deepmath_eval.py)
- [verify_deepmath_full_vocab_formal_preflight.py](verify_deepmath_full_vocab_formal_preflight.py)
- [verify_deepmath_ge6_train15pct.py](verify_deepmath_ge6_train15pct.py)
- [verify_deepmath_ge6_train25pct_perdifficulty.py](verify_deepmath_ge6_train25pct_perdifficulty.py)
- [verify_deepmath_maxlen_eval.py](verify_deepmath_maxlen_eval.py)
- [verify_deepmath_top16_smoke.py](verify_deepmath_top16_smoke.py)
- [verify_deepmath_train25_14b_nonthinking_rollout.py](verify_deepmath_train25_14b_nonthinking_rollout.py)
- [verify_deepmath_train25_nonthinking_rollout.py](verify_deepmath_train25_nonthinking_rollout.py)
- [verify_hf_checkpoint.py](verify_hf_checkpoint.py)
- [verify_qwen3_hard_eval_nonthinking.py](verify_qwen3_hard_eval_nonthinking.py)

## Slurm 启动文件

- [resume_three_checkpoint_nonthinking_x8_generation_8h800.slurm](resume_three_checkpoint_nonthinking_x8_generation_8h800.slurm)
- [run_student_only_nonthinking_x8_4h800.slurm](run_student_only_nonthinking_x8_4h800.slurm)
- [score_student_only_nonthinking_x8_cpu.slurm](score_student_only_nonthinking_x8_cpu.slurm)
- [score_three_checkpoint_nonthinking_x8_cpu.slurm](score_three_checkpoint_nonthinking_x8_cpu.slurm)
- [smoke_three_checkpoint_nonthinking_x8_h800.slurm](smoke_three_checkpoint_nonthinking_x8_h800.slurm)
- [submit_bridge_top16_rkl_formal_8h800.slurm](submit_bridge_top16_rkl_formal_8h800.slurm)
- [submit_bridge_top16_rkl_smoke_8h800.slurm](submit_bridge_top16_rkl_smoke_8h800.slurm)
- [submit_build_deepmath_17b_miss14b_vis.slurm](submit_build_deepmath_17b_miss14b_vis.slurm)
- [submit_build_deepmath_ge6_opd_data.slurm](submit_build_deepmath_ge6_opd_data.slurm)
- [submit_build_deepmath_ge6_train15pct.slurm](submit_build_deepmath_ge6_train15pct.slurm)
- [submit_build_deepmath_ge6_train25pct_perdifficulty.slurm](submit_build_deepmath_ge6_train25pct_perdifficulty.slurm)
- [submit_cuda_probe_h800.slurm](submit_cuda_probe_h800.slurm)
- [submit_deepmath_17b_diagnostic_full_h100.slurm](submit_deepmath_17b_diagnostic_full_h100.slurm)
- [submit_deepmath_17b_diagnostic_merge.slurm](submit_deepmath_17b_diagnostic_merge.slurm)
- [submit_deepmath_17b_diagnostic_smoke_h100.slurm](submit_deepmath_17b_diagnostic_smoke_h100.slurm)
- [submit_deepmath_17b_top16_merge_vis.slurm](submit_deepmath_17b_top16_merge_vis.slurm)
- [submit_deepmath_17b_top16_rescore_h800.slurm](submit_deepmath_17b_top16_rescore_h800.slurm)
- [submit_deepmath_baseline_gap_position_analysis.slurm](submit_deepmath_baseline_gap_position_analysis.slurm)
- [submit_deepmath_baseline_sample_gap_h800.slurm](submit_deepmath_baseline_sample_gap_h800.slurm)
- [submit_deepmath_baseline_sample_gap_merge.slurm](submit_deepmath_baseline_sample_gap_merge.slurm)
- [submit_deepmath_eval_diagnostic_full_h100.slurm](submit_deepmath_eval_diagnostic_full_h100.slurm)
- [submit_deepmath_eval_diagnostic_merge.slurm](submit_deepmath_eval_diagnostic_merge.slurm)
- [submit_deepmath_eval_diagnostic_smoke_h100.slurm](submit_deepmath_eval_diagnostic_smoke_h100.slurm)
- [submit_deepmath_ge6_17b_baseline_h100.slurm](submit_deepmath_ge6_17b_baseline_h100.slurm)
- [submit_deepmath_ge6_17b_baseline_h800.slurm](submit_deepmath_ge6_17b_baseline_h800.slurm)
- [submit_deepmath_ge6_17b_baseline_merge.slurm](submit_deepmath_ge6_17b_baseline_merge.slurm)
- [submit_deepmath_ge6_full_vocab_reverse_kl_formal.slurm](submit_deepmath_ge6_full_vocab_reverse_kl_formal.slurm)
- [submit_deepmath_ge6_full_vocab_reverse_kl_smoke.slurm](submit_deepmath_ge6_full_vocab_reverse_kl_smoke.slurm)
- [submit_deepmath_ge6_native_nonthinking_compare.slurm](submit_deepmath_ge6_native_nonthinking_compare.slurm)
- [submit_deepmath_ge6_native_nonthinking_h800.slurm](submit_deepmath_ge6_native_nonthinking_h800.slurm)
- [submit_deepmath_ge6_native_nonthinking_merge.slurm](submit_deepmath_ge6_native_nonthinking_merge.slurm)
- [submit_deepmath_ge6_top16_rkl_formal_16h800.slurm](submit_deepmath_ge6_top16_rkl_formal_16h800.slurm)
- [submit_deepmath_ge6_top16_rkl_remaining_eval_h100.slurm](submit_deepmath_ge6_top16_rkl_remaining_eval_h100.slurm)
- [submit_deepmath_ge6_top16_rkl_remaining_eval_h800.slurm](submit_deepmath_ge6_top16_rkl_remaining_eval_h800.slurm)
- [submit_deepmath_ge6_top16_rkl_remaining_eval_merge.slurm](submit_deepmath_ge6_top16_rkl_remaining_eval_merge.slurm)
- [submit_deepmath_ge6_top16_rkl_smoke2_16h800.slurm](submit_deepmath_ge6_top16_rkl_smoke2_16h800.slurm)
- [submit_deepmath_ge6_top16_rkl_smoke_16h800.slurm](submit_deepmath_ge6_top16_rkl_smoke_16h800.slurm)
- [submit_deepmath_ge6_top16_rkl_step53_eval_h100.slurm](submit_deepmath_ge6_top16_rkl_step53_eval_h100.slurm)
- [submit_deepmath_ge6_top16_rkl_step53_eval_merge.slurm](submit_deepmath_ge6_top16_rkl_step53_eval_merge.slurm)
- [submit_deepmath_maxlen_merge.slurm](submit_deepmath_maxlen_merge.slurm)
- [submit_deepmath_maxlen_smoke_h800.slurm](submit_deepmath_maxlen_smoke_h800.slurm)
- [submit_deepmath_maxlen_student_h800.slurm](submit_deepmath_maxlen_student_h800.slurm)
- [submit_deepmath_maxlen_teacher_h800.slurm](submit_deepmath_maxlen_teacher_h800.slurm)
- [submit_deepmath_maxlen_teacher_merge.slurm](submit_deepmath_maxlen_teacher_merge.slurm)
- [submit_deepmath_nonthinking_sample_gap_h800.slurm](submit_deepmath_nonthinking_sample_gap_h800.slurm)
- [submit_deepmath_nonthinking_sample_gap_merge.slurm](submit_deepmath_nonthinking_sample_gap_merge.slurm)
- [submit_deepmath_student_top16_trace_diagnostic_h800.slurm](submit_deepmath_student_top16_trace_diagnostic_h800.slurm)
- [submit_deepmath_student_top16_trace_merge.slurm](submit_deepmath_student_top16_trace_merge.slurm)
- [submit_deepmath_train25_14b_nonthinking_rollout_8h800_single_node.slurm](submit_deepmath_train25_14b_nonthinking_rollout_8h800_single_node.slurm)
- [submit_deepmath_train25_14b_nonthinking_rollout_h800.slurm](submit_deepmath_train25_14b_nonthinking_rollout_h800.slurm)
- [submit_deepmath_train25_nonthinking_ckpt_eval_8h800_single_node.slurm](submit_deepmath_train25_nonthinking_ckpt_eval_8h800_single_node.slurm)
- [submit_deepmath_train25_nonthinking_ckpt_eval_merge.slurm](submit_deepmath_train25_nonthinking_ckpt_eval_merge.slurm)
- [submit_deepmath_train25_nonthinking_rollout_h800.slurm](submit_deepmath_train25_nonthinking_rollout_h800.slurm)
- [submit_deepmath_train25_nonthinking_top16_rkl_formal_8h800.slurm](submit_deepmath_train25_nonthinking_top16_rkl_formal_8h800.slurm)
- [submit_deepmath_train25_nonthinking_top16_rkl_smoke_8h800.slurm](submit_deepmath_train25_nonthinking_top16_rkl_smoke_8h800.slurm)
- [submit_good_bridge_top16_rkl_formal_8h800.slurm](submit_good_bridge_top16_rkl_formal_8h800.slurm)
- [submit_good_bridge_top16_rkl_smoke_8h800.slurm](submit_good_bridge_top16_rkl_smoke_8h800.slurm)
- [submit_merge_deepmath_train25_14b_nonthinking_scores.slurm](submit_merge_deepmath_train25_14b_nonthinking_scores.slurm)
- [submit_merge_deepmath_train25_nonthinking_scores.slurm](submit_merge_deepmath_train25_nonthinking_scores.slurm)
- [submit_opd_logit_diagnostic_aime24.slurm](submit_opd_logit_diagnostic_aime24.slurm)
- [submit_opd_logit_diagnostic_dataset.slurm](submit_opd_logit_diagnostic_dataset.slurm)
- [submit_prepare_deepmath_ge6_17b_baseline.slurm](submit_prepare_deepmath_ge6_17b_baseline.slurm)
- [submit_prepare_deepmath_train25_14b_nonthinking_rollout.slurm](submit_prepare_deepmath_train25_14b_nonthinking_rollout.slurm)
- [submit_prepare_deepmath_train25_14b_nonthinking_rollout_8h800_single_node.slurm](submit_prepare_deepmath_train25_14b_nonthinking_rollout_8h800_single_node.slurm)
- [submit_prepare_deepmath_train25_nonthinking_rollout.slurm](submit_prepare_deepmath_train25_nonthinking_rollout.slurm)
- [submit_qwen3_1_7b_from_14b_reverse_kl_deepmath_ge6.slurm](submit_qwen3_1_7b_from_14b_reverse_kl_deepmath_ge6.slurm)
- [submit_qwen3_1_7b_from_14b_reverse_kl_deepmath_ge6_smoke.slurm](submit_qwen3_1_7b_from_14b_reverse_kl_deepmath_ge6_smoke.slurm)
- [submit_qwen3_1_7b_math500_baseline_h100.slurm](submit_qwen3_1_7b_math500_baseline_h100.slurm)
- [submit_qwen3_4b_from_14b_gkd_gsm8k_math.slurm](submit_qwen3_4b_from_14b_gkd_gsm8k_math.slurm)
- [submit_qwen3_4b_from_14b_gkd_smoke.slurm](submit_qwen3_4b_from_14b_gkd_smoke.slurm)
- [submit_qwen3_hard_eval_full.slurm](submit_qwen3_hard_eval_full.slurm)
- [submit_qwen3_hard_eval_nonthinking_full.slurm](submit_qwen3_hard_eval_nonthinking_full.slurm)
- [submit_qwen3_hard_eval_nonthinking_resume_4b.slurm](submit_qwen3_hard_eval_nonthinking_resume_4b.slurm)
- [submit_qwen3_hard_eval_nonthinking_smoke.slurm](submit_qwen3_hard_eval_nonthinking_smoke.slurm)
- [submit_qwen3_hard_eval_smoke.slurm](submit_qwen3_hard_eval_smoke.slurm)
- [submit_qwen_opd_baseline_eval.slurm](submit_qwen_opd_baseline_eval.slurm)
- [submit_score_deepmath_train25_14b_nonthinking_rollout.slurm](submit_score_deepmath_train25_14b_nonthinking_rollout.slurm)
- [submit_score_deepmath_train25_nonthinking_rollout.slurm](submit_score_deepmath_train25_nonthinking_rollout.slurm)
- [submit_smoke_deepmath_train25_14b_nonthinking_rollout.slurm](submit_smoke_deepmath_train25_14b_nonthinking_rollout.slurm)
- [submit_smoke_deepmath_train25_nonthinking_rollout.slurm](submit_smoke_deepmath_train25_nonthinking_rollout.slurm)
- [submit_three_checkpoint_nonthinking_x8_8h800.slurm](submit_three_checkpoint_nonthinking_x8_8h800.slurm)
- [submit_verify_bridge_top16_rkl_formal.slurm](submit_verify_bridge_top16_rkl_formal.slurm)
- [submit_verify_deepmath_ge6_train25pct_perdifficulty.slurm](submit_verify_deepmath_ge6_train25pct_perdifficulty.slurm)
- [submit_verify_deepmath_maxlen.slurm](submit_verify_deepmath_maxlen.slurm)
- [submit_verify_deepmath_train25_14b_nonthinking_rollout.slurm](submit_verify_deepmath_train25_14b_nonthinking_rollout.slurm)
- [submit_verify_deepmath_train25_14b_nonthinking_rollout_8h800_single_node.slurm](submit_verify_deepmath_train25_14b_nonthinking_rollout_8h800_single_node.slurm)
- [submit_verify_deepmath_train25_nonthinking_rollout.slurm](submit_verify_deepmath_train25_nonthinking_rollout.slurm)
- [submit_verify_deepmath_train25_nonthinking_top16_rkl_8h800.slurm](submit_verify_deepmath_train25_nonthinking_top16_rkl_8h800.slurm)
- [submit_verify_good_bridge_top16_rkl_formal.slurm](submit_verify_good_bridge_top16_rkl_formal.slurm)
- [submit_verl_val_only_qwen.slurm](submit_verl_val_only_qwen.slurm)
- [submit_verl_val_only_qwen_4096.slurm](submit_verl_val_only_qwen_4096.slurm)
- [submit_verl_val_only_qwen_max.slurm](submit_verl_val_only_qwen_max.slurm)
- [test1000_anchor_bridge_20260910/aggregate.slurm](test1000_anchor_bridge_20260910/aggregate.slurm)
- [test1000_anchor_bridge_20260910/anchor_candidates_node8.slurm](test1000_anchor_bridge_20260910/anchor_candidates_node8.slurm)
- [test1000_anchor_bridge_20260910/bridges_node8.slurm](test1000_anchor_bridge_20260910/bridges_node8.slurm)
- [test1000_anchor_bridge_20260910/judge_anchors_openlux.slurm](test1000_anchor_bridge_20260910/judge_anchors_openlux.slurm)
- [test1000_anchor_bridge_20260910/rollout_replicate_node8.slurm](test1000_anchor_bridge_20260910/rollout_replicate_node8.slurm)
- [test1000_anchor_bridge_20260910/score_replicate_cpu.slurm](test1000_anchor_bridge_20260910/score_replicate_cpu.slurm)
- [validate_verl_opd_env.slurm](validate_verl_opd_env.slurm)

## 维护工具

- [compact_json_files.py](compact_json_files.py)
- [test1000_anchor_bridge_20260910/aggregate_n8x8.py](test1000_anchor_bridge_20260910/aggregate_n8x8.py)
- [test1000_anchor_bridge_20260910/judge_and_submit_downstream_login.sh](test1000_anchor_bridge_20260910/judge_and_submit_downstream_login.sh)

## 常用可视化示例

在仓库根执行：

```bash
python scripts/export_opd_token_heatmap.py /path/to/diagnostic/tokens.parquet \
  --output reports/example_token_heatmap.html --title "Token KL diagnostic"
```

其他导出与浏览器启动方式见 [可视化说明](../visualization/README.md)。结果解释与实际完成状态以实验总表、源汇总和 audit 标记为准。
