# 在线单次rollout对比

同一1000题，SHA与problem_id核验通过。统一布尔归一化评分。

|单次实验|1000题正确率|共同958题正确率|
|---|---:|---:|
|online|47.60%|46.45%|
|qwen3_1_7b|39.90%|38.41%|
|qwen3_14b|61.00%|60.33%|

在线：每64个student token检查，teacher每次接续8token，可多次介入。平均每题介入5.896次，teacher写入47.156token，judge调用25.825次；58题没有介入。teacher写入量不含其候选检查等计算成本。

共同958题：在线单次46.45%；历史采样平均正确率student64=39.46%、prefix256=41.78%、离线anchor256=43.33%、teacher8=59.06%。单次与多次采样平均均是回答正确率，但抽样方式不同，不能据此声称统计显著或总成本等同。不得与pass@64混淆。

新评分未出现missing/error/timeout。原始配置与逐题评分保存于results.json，汇总在comparison.json。
