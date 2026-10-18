# 全题集 baseline ×8

每格为 pass@8 / pass all 8；分母为题目数。已逐题核验 73,136 条评分的 role、题目 ID、replicate 0–7、评分签名；通过次数与现有准确率汇总完全一致。

| 数据集 | 题数 | Student-only | Teacher-only |
|---|---:|---:|---:|
| CRUXEval-O | 800 | 30.75% / 12.62% | 52.50% / 39.00% |
| CRUXEval-I | 800 | 34.50% / 8.12% | 61.88% / 37.62% |
| HumanEval+ | 164 | 77.44% / 44.51% | 91.46% / 75.00% |
| MBPP+ | 378 | 68.25% / 36.77% | 78.57% / 64.29% |
| LiveCodeBench | 175 | 27.43% / 9.71% | 40.00% / 20.57% |
| MATH500 | 500 | 88.00% / 47.40% | 93.00% / 67.00% |
| OlympiadBench | 674 | 64.39% / 18.55% | 77.74% / 36.35% |
| BBH | 1080 | 74.81% / 9.81% | 90.46% / 63.43% |

pass@8：8 次至少通过 1 次；pass all 8：8 次全部通过。生成超时计错。Code 为执行测试评分；math/reasoning 为既有 GPT-4o-mini judge 评分。

来源：opd_eval_baselines/full_teacher_student_x8_20260918 各组 scores 与 summary.json。
