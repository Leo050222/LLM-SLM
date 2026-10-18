| 数据集 | Student greedy | Teacher greedy | 本次 k=1, π=2 | Δ Student | Δ Teacher |
|---|---:|---:|---:|---:|---:|
| CRUXEval-O | 23.25% | 46.25% | 46.25% | +23.00 | +0.00 |
| CRUXEval-I | 19.88% | 49.12% | 49.25% | +29.38 | +0.12 |
| HumanEval+ | 64.63% | 84.76% | 82.32% | +17.68 | -2.44 |
| MBPP+ | 52.38% | 72.75% | 73.02% | +20.63 | +0.26 |
| LiveCodeBench | 19.43% | 28.57% | 28.57% | +9.14 | +0.00 |
| OlympiadBench | 41.99% | 59.50% | 52.97% | +10.98 | -6.53 |
| BBH | 43.89% | 80.19% | 78.61% | +34.72 | -1.57 |
| MATH500 | 72.00% | 83.60% | 82.40% | +10.40 | -1.20 |
| 总体（按题数加权） | 39.38% | 63.71% | 62.20% | +22.82 | -1.51 |

Delta unit: percentage points. Single rollout per question; 4571 frozen samples; 900s cumulative deadline, timeout count wrong. Existing audited greedy baselines from reports/greedy_vs_temp1_20260921/greedy/summary.json; current output summary downloaded from main_greedy_k1_t2_20260921. Hardware/concurrency differ across baseline deployments, so this does not isolate method-only causal effects or establish statistical significance.
