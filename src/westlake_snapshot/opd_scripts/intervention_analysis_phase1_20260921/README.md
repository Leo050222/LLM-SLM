# 第一阶段：冻结介入 logits 的描述性分析

范围：k=1、π=2，8 数据集×8 轮；B0–B2。只读既有 logits 和历史评分，不生成新 rollout，不启动 B3 反事实。

## 入口与来源

- `extract.py`：全词表 FP32 指标，BF16 排名并列上下界，原始及禁止原 token 后的分布，历史 logprob 校准，逐事件/轨迹表。
- `summarize.py`：2000 次题目聚类 bootstrap，数据集/权重/阈值敏感性、图表、盲标样本包。
- `validate.py`：选择各数据集首尾、概率极值和最大并列数事件，以独立 NumPy FP64 对原始文件复算。
- `gpu.slurm` / `summary.slurm`：集群提交入口。

远端脚本：`/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/intervention_analysis_phase1_20260921`。

远端输入：`/storage/yukaichengLab/wangxin/test_eiloeo/opd_analysis/intervention_logits_bf16_x8_20260920`。

远端输出：`/storage/yukaichengLab/wangxin/test_eiloeo/opd_analysis/intervention_analysis_phase1_20260921`。

冻结 manifest SHA256：`ced4ae7505ffea1e401b3e262b68c5bec38574ede3217d658396817033b4f75a`。

## 作业记录

- 107928、107929：在第 6 个分片写 Parquet 时遇到历史评分 status 的字符串/字典混合类型。统一序列化元数据后从头运行，失败输出保留为带作业编号的 sibling 目录，不混入最终分析。
- 107932：全量提取，正常排队分配至 gnho008，单张 H800；未固定节点，拒绝已知故障 GPU UUID。
- 107933：依赖提取成功的 CPU 汇总。
- 107934：依赖提取成功的 CPU 独立数值验证。
- 107940：修正汇总展示中跨数据集同名 problem_id 的题数去重（统计 bootstrap 已使用数据集/题目复合键）。
- 107943：最终汇总，增加评分异常敏感性并调整图例位置；COMPLETED 0:0，23 秒。107932 COMPLETED 0:0，4分39秒；107934 COMPLETED 0:0，22秒。

最终结果：399601 个事件全部覆盖，4549 个有介入题目，36568 条轨迹，全部评分 hash 匹配；40 事件的 560 项独立 FP64 核验全部通过。报告和紧凑产物位于 `reports/intervention_analysis_phase1_20260921/`。人工语义标注仍待完成。

已通过合成测试：并列排名、极小概率、屏蔽高概率原 token 后的稳定归一化及熵。题目/轮次权重 toy fixture：事件均值 0.6、题目均值 0.25。

解释约束：这是历史共同前缀上的 BF16 重放分布，不是历史 vLLM 张量；单 token 的弱支持度不代表逻辑新路径，低熵不代表正确。盲标样本包不代表人工标注已完成。
