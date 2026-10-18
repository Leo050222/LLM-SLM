# Qwen3-1.7B / Qwen3-14B 全部基线对比

主表使用同一份 4,571 题、同一输入和官方评分口径的单次 greedy 基线；AIME24/25 没有纳入这份 4,571 题 matched-greedy 清单，因此附上 hard_three 的历史 direct、temperature=1 结果并标明协议。不要把不同协议的总准确率当作严格同协议比较。

| 数据集 | 题数 | Qwen3-1.7B | Qwen3-14B | 14B−1.7B | 协议 |
|---|---:|---:|---:|---:|---|
| aime24 | 30 | 5/30 (16.67%) | 8/30 (26.67%) | +10.00 pp | hard_three direct, T=1 |
| aime25 | 30 | 4/30 (13.33%) | 8/30 (26.67%) | +13.33 pp | hard_three direct, T=1 |
| math500 | 500 | 360/500 (72.00%) | 418/500 (83.60%) | +11.60 pp | matched greedy |
| olympiad_bench | 674 | 283/674 (41.99%) | 401/674 (59.50%) | +17.51 pp | matched greedy |
| bbh | 1080 | 474/1080 (43.89%) | 866/1080 (80.19%) | +36.30 pp | matched greedy |
| cruxeval_i | 800 | 159/800 (19.88%) | 393/800 (49.12%) | +29.25 pp | matched greedy |
| cruxeval_o | 800 | 186/800 (23.25%) | 370/800 (46.25%) | +23.00 pp | matched greedy |
| humaneval_plus | 164 | 106/164 (64.63%) | 139/164 (84.76%) | +20.12 pp | matched greedy |
| mbpp_plus | 378 | 198/378 (52.38%) | 275/378 (72.75%) | +20.37 pp | matched greedy |
| livecodebench | 175 | 34/175 (19.43%) | 50/175 (28.57%) | +9.14 pp | matched greedy |
| **matched greedy 总计** | **4571** | **1800/4571 (39.38%)** | **2912/4571 (63.71%)** | **+24.33 pp** | **同协议** |

历史 AIME 行合并后的展示总计为 4631 题：1.7B 1809/4631 (39.06%)，14B 2928/4631 (63.23%)；由于 AIME 两行使用 hard_three T=1，不把它作为 matched-greedy 总结。

来源：`reports/greedy_vs_temp1_20260921/REPORT.md`、`reports/greedy_vs_temp1_20260921/greedy/summary.json`、`experiments/hard_three_20260911/REPORT.md`.
