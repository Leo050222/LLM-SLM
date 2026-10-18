# 当前主实验 greedy 逐题组合分析（k=1, π=2）

样本数：4571 道题；三种结果按 `(dataset, problem_id)` 对齐。

|指标|成功/机会|成功率|
|---|---:|---:|
|Rescue|1145/1298|88.21%|
|Preservation|49/186|26.34%|
|Potential synergy|98/1473|6.65%|
|Breakdown|63/1614|3.90%|

|比较|Improved|Regressed|净变化|双侧 exact p|
|---|---:|---:|---:|---:|
|Hybrid vs Student|1243|200|1043|4.58e-184|
|Hybrid vs Teacher|147|216|-69|0.000345|

## 结论

Hybrid 相对 Student 在 1243 道题上由错变对，在 200 道题上由对变错，净增加 1043 道。
在 Teacher 正确、Student 错误的 1298 道题中，Hybrid 救回 1145 道，Rescue rate 为 88.21%。
在 Student 正确、Teacher 错误的 186 道题中，Hybrid 保留 49 道，Preservation rate 为 26.34%。
两者都错时，Hybrid 在 98/1473 道题上答对；两者都对时，Hybrid 在 63 道题上退化。

完整分数据集结果见 `per_dataset.csv`。