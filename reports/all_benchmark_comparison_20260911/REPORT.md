# 当前介入实验总对比（2026-09-11）

范围：原始Qwen3-1.7B Student与Qwen3-14B Teacher的推理介入实验。最后一类按“无Luna判定”解释。主表全部non-thinking、每题单次回答正确率，不是pass@8/64。未混入4B学生、训练后checkpoint或thinking模式基线。

## 1. 完成实验：单次回答正确率

| 方法 | DeepMath (n=1000) | MATH500 (n=500) | OlympiadBench (n=674) | BBH (n=1080) | AIME24 (n=30) | AIME25 (n=30) |
|---|---:|---:|---:|---:|---:|---:|
| Student直接到底 | 39.90% (399/1000) | 72.80% (364/500) | 29.97% (202/674) | 35.00% (378/1080) | 16.67% (5/30) | 13.33% (4/30) |
| Teacher直接到底 | 61.00% (610/1000) | 87.40% (437/500) | 46.59% (314/674) | 77.31% (835/1080) | 26.67% (8/30) | 26.67% (8/30) |
| Teacher开头32→Student到底 | 41.70% (417/1000) | — | — | — | — | — |
| Teacher开头64→Student到底 | 41.40% (414/1000) | — | — | — | — | — |
| Teacher开头128→Student到底 | 42.40% (424/1000) | — | — | — | — | — |
| Teacher开头256→Student到底 | 40.00% (400/1000) | 75.20% (376/500) | — | — | — | — |
| 无限介入，Luna，绝对差>2，k=8 | 47.60% (476/1000) | 81.40% (407/500) | 34.42% (232/674) | 44.81% (484/1080) | 10.00% (3/30) | 23.33% (7/30) |
| 无限介入，Luna，绝对差>2，k=32 | 50.40% (504/1000) | — | 35.91% (242/674) | 52.59% (568/1080) | 13.33% (4/30) | 10.00% (3/30) |
| 无限介入，无Luna，绝对差>2，k=8 | — | 84.20% (421/500) | 46.14% (311/674) | 75.74% (818/1080) | 10.00% (3/30) | 20.00% (6/30) |
| 无限介入，无Luna，绝对差>3，k=8 | — | 82.40% (412/500) | 42.43% (286/674) | 76.67% (828/1080) | 23.33% (7/30) | 20.00% (6/30) |
| 随机位置，k=8，最多8次（额外对照） | 42.60% (426/1000) | 58.00% (290/500) | 18.84% (127/674) | 35.65% (385/1080) | 10.00% (3/30) | 16.67% (5/30) |

—表示本次盘点未找到该配置的完成结果，不代表0%。已跑但未评分的配置另列。BBH为官方27个子任务各40题、共1080题；OlympiadBench为英文数学OE_TO_maths_en_COMP全674题。AIME每年仅30题。

所有上表阈值介入都是历史绝对差 `abs(logp_student−logp_teacher)>τ`，不是后来更正的正向差规则。在线student chunk=64；Luna只判断候选是否接管，不生成最终答案、不担任答案评分器。历史在线组保留teacher首token不得等于student原token的规则。

DeepMath统一布尔答案归一化：原student/teacher单次28.3%/43.6%已统一为39.9%/61.0%。本次只读重汇总了开头接管单次实验：k32/64/128/256旧分数29.3/28.3/29.3/27.6%，统一后41.7/41.4/42.4/40.0%；分别新增124/131/131/124个正确、没有反向改判。原输出和原评分未改写。新规则不是对证明过程的审核。

DeepMath直接/开头接管的历史seed为20260811，在线seed为20260911；其余主要20260911，temperature=1/top_p=1。阈值3实验用H100，许多基线用H800。单次结果不能据此声称仅一个因素的严格因果差异。

k32的OlympiadBench与BBH各有1条被用户提前停止，按错误纳入原分母；与其它组跑到既定结束条件的协议不同。随机DeepMath组允许原token，其它五个benchmark的随机组强制首token不同，两组随机结果不能当作同一协议。

## 2. 单次anchor支持：同一DeepMath 958题子集

1000题中958题找到了离线anchor；剩余42题未纳入这张表，不能把43.33%直接放进全1000题排行榜。Teacher开头与单次anchor均为8个teacher分支×8个student continuation，每题64个回答；Student直接为64个回答，Teacher直接为8个回答。以下为回答平均正确率。

| 方法 | 回答数 | 正确率 | 相对同子集Student |
|---|---:|---:|---:|
| Student直接×64 | 61,312 | 39.4572% | +0.0000 pp |
| Teacher直接×8 | 7,664 | 59.0553% | +19.5981 pp |
| Teacher开头32→Student到底 | 61,312 | 39.8307% | +0.3735 pp |
| Teacher开头64→Student到底 | 61,312 | 39.7899% | +0.3327 pp |
| Teacher开头128→Student到底 | 61,312 | 40.2026% | +0.7454 pp |
| Teacher开头256→Student到底 | 61,312 | 41.7765% | +2.3193 pp |
| 单次anchor处Teacher256→Student到底 | 61,312 | 43.3341% | +3.8769 pp |

同题子集中，anchor256相对开头256提高1.5576个百分点，相对Student直接提高3.8769个百分点。它使用原始student前缀、anchor筛选和分支采样成本，不能视为等算力比较。

全1000题的多次采样平均正确率（同一新评分口径）：

| 方法 | 回答数 | 正确率 |
|---|---:|---:|
| student64 | 64,000 | 40.5641% |
| teacher | 8,000 | 59.6500% |
| h32 | 64,000 | 40.9438% |
| h64 | 64,000 | 40.9391% |
| h128 | 64,000 | 41.3047% |
| h256 | 64,000 | 42.9328% |

单次1×1与8×8结果须分开：例如开头256在单次实验40.0%，在8×8采样平均42.9328%。不能只挑更高者或把pass@64当正确率。

## 3. 更早的train25已选anchor集合：支持长度实验

此表为13,244个已选prefix上的continuation实验，不是held-out1000题。Student基线从同一个anchor前缀继续，不是从问题开头直接rollout。每个bridge长度847,616条continuation，基线105,952条。此处保留既有报告评分口径，不直接合并到上述布尔归一化测试表。

| 条件 | 回答平均正确率 | 相对prefix Student |
|---|---:|---:|
| 同prefix Student到底 | 39.2668% | — |
| anchor处Teacher32→Student | 40.4995% | +1.2326 pp |
| anchor处Teacher64→Student | 41.0011% | +1.7343 pp |
| anchor处Teacher128→Student | 42.0167% | +2.7498 pp |
| anchor处Teacher256→Student | 44.2065% | +4.9396 pp |
| 同prefix Teacher到底 | 57.3118% | +18.0450 pp |

长度增加的主要收益是提高回答成功概率和重复成功稳定性；其pass@8约69.8%–70.0%，不应解释为解题覆盖随k同幅增加。

## 4. 有无Luna与阈值的teacher用量

下列每题teacher保留token不是总计算量，不含候选logit评分、丢弃student草稿和Luna请求。

| Benchmark | Luna τ2 k8平均介入次数 | 无Luna τ2 k8平均次数 | 有Luna平均teacher token | 无Luna平均teacher token |
|---|---:|---:|---:|---:|
| MATH500 | 2.25 | 9.25 | 17.99 | 73.85 |
| OlympiadBench | 4.54 | 16.10 | 36.29 | 128.03 |
| BBH | 3.31 | 10.49 | 25.49 | 80.66 |
| AIME24 | 8.17 | 27.67 | 65.33 | 221.10 |
| AIME25 | 6.27 | 23.90 | 50.13 | 191.20 |

DeepMath Luna k8/k32分别平均5.896/5.042次介入、47.156/160.817个teacher token。MATH500固定开头256平均251.96个teacher token，在线Luna k8平均17.99个，但后者另付候选评分与Luna成本。

无Luna τ3 k8：MATH/Olympiad/BBH平均介入3.924/6.789/8.217次，teacher汇总token占比3.80%/3.70%/23.22%；τ2对应8.56%/8.60%/29.72%。阈值3节省介入，但MATH/Olympiad正确率降低1.80/3.71个百分点，BBH增加0.93个百分点；不存在所有benchmark一致最优阈值。

## 5. 尚未有完整评分的配置（远程18:16左右快照）

| 配置 | 数据集 | 当前证据 |
|---|---|---|
| 绝对差>2 + Luna，k1，不限次数，首token排除 | MATH500/Olympiad/BBH，共2254题 | GPU102597运行，1542/2254生成、0评分；不能填正确率 |
| 正向差student−teacher>2 + Luna，k8，最多8次 | 同2254题 | GPU102611运行，502/2254生成；0评分 |
| 正向差>2，无Luna，k8，最多8次 | 同2254题 | 已安排，顺序等待前组 |
| 正向差>2 + Luna，k8，不限次数，允许原token | 同2254题 | 已安排，顺序等待前组 |
| 随机Bernoulli介入，k8，不限次数，允许原token | 同2254题 | 已安排，按各数据集历史Luna平均间隔设概率；仅期望频率匹配 |

未找到完成结果：无Luna k32及其它k支持长度扫点、有Luna τ3扫点、MATH500有Luna k32、DeepMath无Luna阈值扫点、其它benchmark的单次anchor与开头k长度扫点。不得把最多8次的组归入无限次数，亦不得把正向差新实验与绝对差历史实验混为一谈。

## 6. 可以支持的结论

1. 完成的在线介入中，无Luna τ2 k8在MATH500/OlympiadBench/BBH取得84.20%/46.14%/75.74%，比有Luna k8高2.80/11.72/30.93 pp；teacher用量同时增加，尚不能解释为等预算下Luna有害。
2. Luna k32相对k8在DeepMath/Olympiad/BBH提高2.8/1.48/7.78 pp，teacher写入量更高；AIME波动且AIME25下降，不能宣称所有benchmark受益。
3. 单次anchor256在相同958题、相同64回答结构下比开头256高1.56 pp；这支持介入位置有价值，但不是等计算预算因果消融。
4. 当前结果未超过相应Teacher直接rollout：BBH无Luna τ3接近Teacher（76.67 vs77.31），Olympiad无Luna τ2接近Teacher（46.14 vs46.59）。单次小样本不支持稳定等价声明。
5. 优先完成正在运行的正向差/首token规则消融，再做同teacher预算的Luna开关对照与多seed；AIME需更多采样。当前KV复用只有功能/性能验证，没有新全量正确率实验，不能把其加速数混入上述准确率证据。

## 来源与可复核产物

- single_rollout_results.csv / .json：每个完成配置的正确数、分母、阈值、次数上限、相对提升、源路径。
- normalize_single_prefix.py：只读重汇总旧单次开头介入的脚本；仅采用现有归一化函数，不改原输出。
- experiments/anchor_comparison_20260910/{comparison.json,SCORING_AUDIT.md}：多次采样同题与评分口径。
- experiments/{hard_three,hard_k32,threshold_k8_nojudge,threshold3_mathteacher,random_k8_max8}_20260911/artifacts/summary.json。
- experiments/math500_three_20260911/results/summary.json；experiments/online_comparison_20260911/comparison.json。
- analysis/prefix_bridge_teacher_average_accuracy_20260908.csv：train25前缀支持力度。
- experiments/{threshold2_luna_k1,takeover_ablation}_20260911/STATUS.md与本次远程只读快照。
