# 固定 prefix 续写与历史 Teacher-only 的三列组合

每数据集100题，每题固定prefix续写8次。k=1、阈值=2，分别统计已核验完整的介入上限1–10；Student列为同prefix、同replicate编号的cap=0续写。

Teacher-only列来自原始整题生成的同题历史结果，每题只有一个布尔标签，重复用于8个replicate作为参照。不是同prefix下Teacher续写8次，不能解读为严格三模式反事实对照。数学与BBH使用GPT-4o-mini重评分口径；代码用测试评分。

每列分母800，100个独立prefix簇；百分比是逐次rollout组合占比，不是题目级成功率或pass@8。同种子不保证逐token轨迹一致；仅作编号配对描述。样本限于存在首次触发点的条件化队列。

源数据：reports/prefix_caps_analysis_20260917/paired_scores.csv；outputs/triple-outcomes/baselines-paired-source.json。

已检查70400条记录唯一键、各档cap=0完整配对、每数据集100prefix/800rollout、Teacher历史标签完整覆盖及重复8次的边际计数。
