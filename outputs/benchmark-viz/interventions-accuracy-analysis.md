# 介入次数与准确率：配置级相关分析

数据：本地 data.json，25 个固定 k 混合配置/数据集。排除 Student/Teacher-only；未包含 adaptive。前3项基准准确率为2026-09-16 GPT-4o-mini重评分。

Pearson 衡量线性关联；Spearman 为平均秩相关；within-k 为分别对两变量减去所属 k 组均值后计算 Pearson，描述同一 k 内改变阈值时的合并关联。每个 k 只有5个点。

|数据集|Pearson|Spearman|Within-k Pearson|
|---|---:|---:|---:|
|MATH500|0.801|0.878|0.873|
|OlympiadBench|0.838|0.939|0.843|
|BBH|0.583|0.628|0.629|
|CRUXEval-O|0.458|0.234|-0.351|
|CRUXEval-I|-0.298|-0.450|0.854|
|HumanEval+|-0.409|-0.402|0.029|
|MBPP+|0.183|0.186|-0.250|
|LiveCodeBench|-0.358|-0.469|0.075|

限制：配置共用评测题，非独立重复实验；未报告显著性或因果效应。停止规则及评分误差可能影响趋势。汇总均值不能推断题目级介入与答对的关系。常量组的相关系数记为空值。

原始数据见 data.csv / data.json / source.md；各 k、各阈值的分组结果见 interventions-accuracy-correlation.json。