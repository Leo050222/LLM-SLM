# 单条 rollout 长度缩短与正确率

本次从原始 scores 重新统计，分析单位为单条 after-bridge rollout。每个 prefix 的 baseline 长度为其 8 次直接 student rollout 的整体 trace 长度均值。整体 trace 包含 prefix、bridge 和 student continuation。缩短比例 = (baseline 长度 − 当前长度) / baseline 长度。

覆盖 13,244 个 prefix；每种 horizon 有 8 个 source bridge × 8 次 continuation，即 847,616 条 rollout。四种 horizon 共 3,390,464 条。核验 baseline 的每个 prefix 恰有 8 条，after 的每个 replicate 覆盖全部 `(prefix, source bridge)` 且无重复；分档计数与正确数之和均与缩短组一致，总正确率与此前报告一致。

## 主要发现

| Bridge tokens | 缩短组 N | 缩短组正确率 | 未缩短组正确率 | 两组差值 |
|---:|---:|---:|---:|---:|
| 32 | 471,607 | 42.435% | 38.072% | +4.363 pp |
| 64 | 467,126 | 43.187% | 38.317% | +4.870 pp |
| 128 | 459,015 | 44.356% | 39.254% | +5.102 pp |
| 256 | 441,094 | 46.918% | 41.264% | +5.654 pp |

缩短组整体正确率更高，但组内不是普遍的“缩短越多、绝对正确率越高”。

- 按节省 token 数分档，四个 horizon 的绝对正确率均随档位上升而下降。例如 h128 从节省 0–64 tokens 档的 48.926%，下降到 >1024 tokens 档的 30.730%。对应的 baseline 正确率从 45.736% 降至 21.349%，说明各档样本构成不同。
- 按相对缩短比例分档，h32/h64 大体下降，h128 大体持平，h256 在中等缩短幅度内略上升，但均不严格单调。>75% 档仅 622–713 条，且对 baseline 截断排除较敏感，不能据此建立稳定尾部规律。
- 将缩短组按相对缩短幅度分为五个近等量分位组，各 horizon 的“该组正确率减去匹配 baseline 正确率”均逐档增加。匹配 baseline 按该档每条 rollout 所对应 prefix 的 baseline 正确率加权。此差值是描述性统计，不是随机化干预效应。
- 在缩短组内部，将缩短比例与 0/1 正确性分别在 prefix 内去均值，拟合共同线性斜率：每多缩短 baseline 长度的 10 个百分点，正确率的拟合关联为 h32 +0.500 pp、h64 +0.693 pp、h128 +0.995 pp、h256 +1.442 pp。它控制了 prefix 间固定差异，但不证明逐档单调或因果关系，也不排除 bridge 内容的影响。
- 排除 baseline 任一条被长度截断的 prefix，并排除单条 after rollout 的长度截断样本后，主要趋势不变。同 prefix 斜率分别为 +0.500、+0.693、+0.996、+1.442 pp。

## 解释边界

“是否缩短”“跨 prefix 的绝对正确率”“相对 baseline 的差值”“同 prefix 内的关联”是不同问题。本次数据支持缩短组平均更准确、同 prefix 内缩短程度与正确性正向关联；不支持跨 prefix 普遍的缩短越多绝对正确率越高。

baseline 仅由 8 次采样估计；按生成后的长度分组存在选择效应；同 prefix/source bridge 下的多次续写并非独立题目。本报告未进行置信区间或显著性检验，未声称因果性。若需要推断性结论，应按题目/prefix 聚类重采样，并进一步比较同 source bridge 内的关联。

## 产物

- `buckets.csv`：所有分档计数、正确数、正确率、匹配 baseline 与长度统计。
- `summary.json`：总体指标、同 prefix 斜率、输入路径与文件元数据。
- `report.md`：完整 all/clean 分档表。
- `shortening_accuracy.png` / `.svg`：三种比较的图。
- 上级目录的 `rollout_shortening_accuracy_20260907.py` 与 `plot_rollout_shortening_accuracy_20260907.py`：统计与绘图脚本。
