# OPD 项目：证据、后续实验与论文叙事建议

日期：2026-09-11。范围：本地已完成实验、实现与相关论文；没有启动新 GPU 实验、调用付费 judge 或修改算法。以下是研究判断与建议，不是新颖性认证。

## 1. 建议的主问题

**在有限强模型辅助预算下，何时接管、接管多久，才能提高弱模型后续推理的成功率？**

当前冻结模型在线实验最直接支持这个问题。最初的 OPD 监督过滤是研究起点；如果论文主实验没有蒸馏训练，不应以“解决 OPD 监督质量”作为论文承诺。可以在背景中简述其启发，把训练应用留给有实验证据的后续工作。

目前最强的候选版本是无 judge 的分歧阈值触发策略。语义 judge 的必要性尚未成立。“概率分歧”“语义路径改变”“介入使 student 更容易做对”是三个不同概念，不能互相代替。论文应把区分这三者作为实验组织原则，而不是预先宣布它们相同。

已有文献覆盖了分歧触发、稀疏接管、语义过滤和并行验证。仅将这些拼接起来，或将接管长度从 1 改为 8，不足以建立有力的独立贡献。最值得补充的证据是：**统一预算下的干预效用、student 能接续的短接管长度，以及计入验证开销后的实际效率边界。**

## 2. 已完成实验支持什么

### 2.1 原始分歧观察

定义 `g_t = log p_S(y_t | prefix) - log p_T(y_t | prefix)`。本地 non-thinking 分析覆盖 1,000 条轨迹、1,303,870 个 response token。

- `g > 2` 的比例：正确轨迹平均 1.602%，错误轨迹 2.425%；差 +0.823 个百分点，轨迹 bootstrap 95% CI [+0.659, +0.984]。
- 用该比例区分对错的 AUROC 为 0.679：有用的风险信号，但不是精准错误探测器。
- 峰值在相对位置 20–30%，宽峰约 20–40%；称为“前中段”比笼统称“中期”更准确。
- 这仍是观察相关性，不能证明所选 token 导致了错误，或对应真正的逻辑转折。
- 相对位置 `t / 最终轨迹长度` 用到了事后长度，适合描述图，不能原样成为在线定位规则。
- 报告使用可视化压缩数据；正式论文应以原始浮点分数复算。

来源：[固定阈值报告](/Users/leolee/Documents/OPD/reports/deepmath_nonthinking_gt2_analysis.md)、[多阈值报告](/Users/leolee/Documents/OPD/reports/deepmath_nonthinking_unrecognized_token_analysis.md)。

### 2.2 当前在线结果

相同 Qwen3-1.7B student / Qwen3-14B teacher，native non-thinking，temperature=1，top_p=1；每题每策略单次采样。数字为正确率百分数。

| 数据集 | Student | 开头 teacher256 | 随机8，最多8次 | 分歧+Luna8 | 仅分歧8 | Teacher |
|---|---:|---:|---:|---:|---:|---:|
| MATH500 | 72.80 | 75.20 | 58.00 | 81.40 | 84.20 | 未完成同表对照 |
| OlympiadBench | 29.97 | — | 18.84 | 34.42 | 46.14 | 46.59 |
| BBH | 35.00 | — | 35.65 | 44.81 | 75.74 | 77.31 |
| AIME2024 | 16.67 | — | 10.00 | 10.00 | 10.00 | 26.67 |
| AIME2025 | 13.33 | — | 16.67 | 23.33 | 20.00 | 26.67 |

原报告：[MATH500](/Users/leolee/Documents/OPD/experiments/math500_three_20260911/REPORT.md)、[困难集](/Users/leolee/Documents/OPD/experiments/hard_three_20260911/REPORT.md)、[随机策略](/Users/leolee/Documents/OPD/experiments/random_k8_max8_20260911/REPORT.md)、[无 judge](/Users/leolee/Documents/OPD/experiments/threshold_k8_nojudge_20260911/REPORT.md)。

关键解释：

1. 在线协作优于 student 的证据较强，特别是无 judge 的 OlympiadBench 和 BBH。其正确率点估计接近 teacher，但“无显著差异”不能证明等效。
2. MATH500 的 prefix256 有 +2.4pp，不能写成“好的开头没有用”；只能说该设置下在线策略点估计更高。
3. 随机组是“随机位置 + 强制首 token 改写”，而且最多 8 次，anchor 组不限次数。它不是仅替换位置、其余完全相同的因果对照。
4. 去除 judge 同时放宽了实际介入量：MATH500 从约 18 到 74 teacher token/题，OlympiadBench 从约 36 到 128，BBH 从约 25 到 81。不能断言相同预算下 judge 有害。
5. AIME 每年 30 题，单轮的几道题变化不适合支撑普遍结论。
6. teacher32 相比 teacher8 在 OlympiadBench/BBH 更好，但 teacher 写入量更高；32 组有两条用户中止并计错的轨迹，协议不同。

### 2.3 本次从原始逐题结果追加的检查

重读了 4,628 条完成轨迹，按规范化 problem_id 对齐了 2,314 道题的 student、judge、nojudge 评分。保留原协议中 student/OlympiadBench 的一条答案提取为空记录，计错且未剔除。

以下区间为 20,000 次逐题配对 bootstrap，条件于已有单 seed 输出；不覆盖重新生成、跨模型或新 BBH 子任务的不确定性，也未做多重比较校正。

| 无 judge 相对 Student | 新做对 | 原先对而现在错 | 差值 | 95% CI |
|---|---:|---:|---:|---:|
| MATH500 | 70 | 13 | +11.40pp | [+8.0, +14.8] |
| OlympiadBench | 152 | 43 | +16.17pp | [+12.3, +20.0] |
| BBH | 475 | 35 | +40.74pp | [+37.4, +44.0] |

无 judge 相对有 judge：MATH500 +2.8pp，CI [-0.2, +5.8]；OlympiadBench +11.72pp，CI 约 [+8.2, +15.4]；BBH +30.93pp，CI 约 [+27.6, +34.3]。这些比较仍然没有控制介入预算。

无 judge 相对 teacher：OlympiadBench -0.45pp，CI 约 [-3.9, +3.0]；BBH -1.57pp，CI 约 [-3.5, +0.4]。若要主张性能保持，需要预先指定非劣界和有相应功效的验证设计。

可复算：[recompute.py](/Users/leolee/Documents/OPD/analysis/research_direction_review_20260911/recompute.py)，完整结果：[reanalysis.json](/Users/leolee/Documents/OPD/analysis/research_direction_review_20260911/reanalysis.json)。

### 2.4 离线 bridge 与 bifurcation 证据

13,244 个 prefix 的平均 student 正确率 39.27%；接 32/64/128/256-token bridge 后分别为 40.50/41.00/42.02/44.21%。这说明一些 student 状态可以被 teacher continuation 改善，但不是“在线 8-token 精准定位”本身的验证。共享 prefix、题目、bridge 和续写之间存在层级相关性，正式统计应按 problem_id 聚类。

900 个 question–bridge 组的语义评估中，0/8-lift 组仍有 61.6% 被判逻辑分岔，8/8-lift 组为 77.4%；相关系数约 0.194。这支持“语义改变和收益有一定关联”，也直接说明“发生逻辑分岔”不是充分条件。按已经观测到的 lift 分桶有选择噪声；需要独立 continuation 估计来确认极端组。

来源：[bridge 长度报告](/Users/leolee/Documents/OPD/analysis/bridge_h32_h64_h128_h256_report_20260906.md)、[bifurcation 报告](/Users/leolee/Documents/OPD/outputs/anchor_bifurcation_h256_openlux_luna_seed20260908/full/analysis/analysis_report.md)。

## 3. 写论文前必须讲清的实现问题

### 3.1 有方向分歧与绝对分歧

原始观察检验 `g > 2`，在线 [build_candidates](/Users/leolee/Documents/OPD/ocrd_pipeline/anchor_selector/core.py) 实现 `abs(g) > 2`。当 `g < -2` 时，teacher 实际比 student 更认可被抽中的 token，不能称为 teacher 拒绝。

本次统计无 judge 介入中负 gap 占比：MATH500 10.29%，OlympiadBench 11.81%，BBH 3.83%，AIME2024 11.33%，AIME2025 10.74%。这不是可以忽略不提的定义差异。

建议比较 signed-positive、absolute、negative-only；均在验证集调节触发率以对齐预算。保留 historical absolute 结果，不要改公式后继续把旧结果当新方法结果。

`g > 2` 表示 teacher/student 概率比低于 exp(-2)，不等于 teacher 认为该 token 绝对概率低，也不是当前 token 的真错误标签。单个 sampled-token log ratio 可以为负，不应直接叫“该位置的 KL divergence”；在 student 分布下的期望才对应 KL。

### 3.2 强制首 token 不同

当前 teacher 接管禁止生成原 student token。这是一个独立的策略设计，可能造成正常位置的有害扰动，也可能强迫 teacher 放弃自己认可的答案。至少需要无 ban 对照，以及在同位置的 student 自重采样对照。

推荐隔离两个因素：位置选择 × 是否强制改写。先做四格实验（随机/分歧 × ban/no-ban），固定 horizon 和预算；再对重要格子添加 student 自重采样，辨别收益是否来自一般探索。

### 3.3 “少量 teacher token”必须明确分母

下表是 pooled teacher retained tokens / pooled final output tokens，以及 discarded student draft / all generated student draft，不是 FLOPs 比例。

| 无 judge | Teacher 占最终输出 | 被丢弃的 student 草稿 |
|---|---:|---:|
| MATH500 | 8.56% | 31.08% |
| OlympiadBench | 8.60% | 31.70% |
| BBH | 29.72% | 71.39% |

BBH 虽然平均约 81 个 teacher token，但平均最终输出约 271 token；不能把它描述为几乎全由小模型完成。其高草稿浪费率提示 chunk64 在频繁接管任务上可能过长，值得实测。

## 4. Chunk64 的效率论证

直觉成立的部分：已知的 64 个 draft token 可在因果 attention 下由 teacher 一次前向并行评分，减少 teacher 必须串行 decode 的步骤。chunk 不会自动将 teacher 算力变为逐 token 的 1/64；teacher 仍处理每个 draft token，接管后还要丢弃并重算分岔后缀。

本地旧在线结果默认关闭 prefix caching，因此重复 prefill 有额外成本。最新 [KV reuse 报告](/Users/leolee/Documents/OPD/reports/kv_reuse_20260911/IMPLEMENTATION.md) 已有真实微基准：warm batch-one teacher score call 在 1024/4096/8192 初始 prefix 上，相对重复 full-prefill 为 9.05×/27.84×/47.73×。这是评分调用的局部加速，不能写成相对 teacher-only 或整条推理的倍数。

该验证同时存在 BF16 数值漂移：原 0.05 logprob tolerance 未通过，cached/reference 最大差约 0.341。应该检查阈值两侧的触发决策变化率及端到端 accuracy，不只检查均值误差。缓存淘汰、并发变化、回滚、边界修复也应进入系统实验。

推荐成本记录：

`总成本 = student draft生成 + teacher候选评分 + teacher接管生成 + judge + 缓存/通信/调度成本`。

对 latency 使用关键路径实测，对 GPU 工作量避免把共享 batch 时间重复相加。至少报告端到端 median/p95 latency、吞吐、GPU秒/题、teacher score/decode 工作量、teacher最终写入量、草稿接纳率、judge输入输出token和时间。与 teacher-only 比较时要统一总 GPU 资源和并发；外部 judge 成本不可省略。

关键系统对照：student-only、teacher-only、exact speculative decoding、我们的 chunk 8/16/32/64/128；分别验证 warm/cold cache 和 batch1/常用并发。先做小规模可重复测量，不需要立刻把所有模型和数据集全组合跑完。

若全位置 teacher 评分仍占主导，可尝试 student entropy/margin 先筛 chunk，再请求 teacher；这个变体应评价触发召回率和 accuracy-cost frontier，不能默认免费保留原策略收益。

## 5. 相关工作与定位

检索覆盖本地四篇 PDF、官方论文页面与全文。Zotero/Obsidian 未配置；arXiv helper canonical chain 未解析到，使用官方 arXiv 网页补足。下表区分正式发表与预印本；不是所有训练方法都值得完整重训，优先复现机制接近且成本可控的对照。

| 论文 | 作者、年份、发表状态 | 核心方法/结果 | 与本项目的关系 | 来源 |
|---|---|---|---|---|
| Reasoning Can Be Restored by Correcting a Few Decision Tokens | Shen et al., 2026, ICML 2026 | 分布级 CE/spike gate、单 token takeover；已有失败预测和 early-only/random 对照 | 最直接竞争：原始“分歧→失败→稀疏接管”证据链已有；需比 CE/rKL/entropy 与 1-token | [全文](https://arxiv.org/html/2605.16874v1) |
| R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing | Fu et al., 2025, NeurIPS 2025 | 区分 neutral 与 divergent；外部语义标注训练 router；报告 2.8× wall-clock 加速 | 风格不等于逻辑差异已有明确研究；其离线 teacher-prefix 后缀标注不同于这里在线 student-prefix judge | [正式版](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b39cef2ef90591cffdc9c674cd55bebe-Abstract-Conference.html) |
| FutureBridge: Token Selection Beyond Local Preference in Collaborative Decoding | Li et al., 2026, arXiv preprint | 学习候选 token 对接收方 student 的未来适用性；统一请求策略和预算 | 直接触及“teacher偏好不等于student效用”；其需要训练，本项目当前无需训练 | [全文](https://arxiv.org/html/2608.06819v1) |
| Pass the Baton: Trajectory-Relayed On-Policy Distillation | Xu et al., 2026, arXiv preprint | 反思词方向分歧触发teacher leg；单引擎block verification；推理pilot 27.73→34.96 | 短接管、归还及chunk验证均有前史；可复现反思词触发做机制对照 | [本地/官方](https://arxiv.org/abs/2607.26057) |
| Look Ahead Before You Distill: Future Trajectory Validation of Teacher Guidance for Agentic On-Policy Distillation | Chen et al., 2026, arXiv preprint | 高分歧action桥接后让student接续；分析中仍约19%高分歧干预退化 | 需分开teacher局部偏好与后续效用；这是多轮agent训练，不是同一token推理策略 | [本地/官方](https://arxiv.org/abs/2608.01953) |
| Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM’s Reasoning Capability | Lin et al., 2025, ICML 2025 | rollout sampling 定位影响最终结果的critical token，结合contrastive估计/cDPO | 为同prefix反事实测试提供直接方法背景；不是仅看token对错 | [正式版](https://proceedings.mlr.press/v267/lin25j.html) |
| What makes Reasoning Models Different? Follow the Reasoning Leader for Efficient Decoding | Li et al., 2025, arXiv；有ACL workshop版本 | 强模型给句首，弱模型完成句子；报告理论FLOPs降30–50% | 比整条trace开头prefix更强的便宜位置对照：句首/段首 | [论文](https://arxiv.org/abs/2506.06998) |
| RelayLLM: Efficient Reasoning via Collaborative Decoding | Huang et al., 2026, arXiv preprint | 学习主动求助；报告1.07% teacher生成token | student-only gate 对照；其成本节约参照对象要核对，不能当teacher-only节约 | [论文](https://arxiv.org/abs/2601.05167) |
| CITER: Collaborative Inference for Efficient Large Language Model Decoding with Token-Level Routing | Zheng et al., 2025, COLM 2025 | quality与cost共同训练router | 预算感知learned routing背景；应计训练成本，不能直接拼原文数值 | [论文](https://arxiv.org/abs/2502.01976) |
| Fast Inference from Transformers via Speculative Decoding | Leviathan et al., 2023, ICML 2023 | draft+并行target验证；精确修正维持target分布，T5-XXL报告2–3× | chunk评分的基础系统对照；我们的混合策略没有精确分布保持保证 | [正式版](https://proceedings.mlr.press/v202/leviathan23a.html) |
| SpecCoT: Accelerating Chain-of-Thought Reasoning through Speculative Exploration | Shi et al., 2025, Findings of EMNLP | 多候选推理步并行验证；报告1.7–4.1×延迟缩减 | 说明teacher-only不是唯一效率基线 | [正式版](https://aclanthology.org/2025.findings-emnlp.1326/) |
| Speculate, then Collaborate: Fusing Knowledge of Language Models during Decoding | Wang et al., 2025, ICML 2025 | speculative式草稿与选择性协作融合知识 | 训练自由协作与验证结合已有研究，应比较具体机制和目标 | [正式版](https://proceedings.mlr.press/v267/wang25e.html) |
| Learning to Decode Collaboratively with Multiple Language Models | Shen et al., 2024, ACL main | latent selector学习逐token自主生成或求助 | 更早的逐token协作背景；需训练，与当前阈值法区分 | [正式版](https://aclanthology.org/2024.acl-long.701/) |
| Efficient Thought Space Exploration Through Strategic Intervention | Li et al., 2026, AAAI 2026 | 强hinter指导弱practitioner，树状探索和分布不一致减少 | 多路径策略背景；必须按全部探索成本比较 | [正式版](https://ojs.aaai.org/index.php/AAAI/article/view/40459) |
| ADWIN: Adaptive Windows for Horizon-Aware On-Policy Distillation | Liang et al., 2026, arXiv preprint | 自适应训练窗口，报告最高4.1×训练FLOPs减少 | 最初OPD动机相关；训练窗口不同于在线chunk | [本地/官方](https://arxiv.org/abs/2605.28396) |
| DASH: Divergence-Adaptive Supervision Horizons for On-Policy Self-Distillation of Reasoning Models | Hou et al., 2026, arXiv preprint | divergence控制时序监督聚合；不增加模型forward | 支持局部分歧不充分，但这是训练loss方法，不是takeover | [本地/官方](https://arxiv.org/abs/2608.06243) |

可以将文献组织成四组：token级协作路由；critical-token/分歧诊断；以student后续表现校验teacher指导；speculative并行验证。它们共同说明位置、动作和成本需要分别建模。

与 Few Decision Tokens 的“早期峰值”差异尤其不能直接解释成新发现：其主要分析 base→thinking、CE和每轨迹top1%；当前是 non-thinking小→大模型、sampled signed gap和固定阈值。应交叉改变模型模式、指标、阈值和位置归一化后再讨论机制差异。

另需区分两个名字接近的 FutureBridge：2608.06819 是 token collaborative decoding；2608.01953 是 agentic OPD。论文中使用完整标题或明确别名，避免合并为一篇。

## 6. 按优先级补实验

### P0：决定主方法与主要结论

**E1：公平的位置与改写对照。** 先在开发集比较 random/分歧 × ban/no-ban，固定 k=8，预算上限 M=1/2/4/8/16；加入句首、固定间隔，必要时增加位置分布匹配random。相同上限不等于相同实际用量，应画实际teacher写入量与完整成本的accuracy frontier，在相近成本处比较。随机策略用在线hazard/间隔，不使用当前测试题未来真实轨迹长度。离线长度匹配可以保留为诊断，但必须标明不可部署。共同预算与位置占用可在开发集校准，不用test调参。

**E2：judge是否选中了更有价值的候选。** 在相同分歧候选流上比较：不筛、Luna筛、随机接受（匹配Luna接受率）、更严格阈值（开发集匹配用量）。预算统一后若Luna仍提高成功率或减少伤害，可称为效用过滤；否则将其放到消融，主方法采用更简单版本。BBH原judge提示偏数学，需区分通用效用目标与领域错配。

**E3：signed/absolute与便宜替代信号。** 比较 g>τ、|g|>τ、g<−τ、teacher NLL、student entropy/margin；加 Few Decision Tokens 的 CE/rKL gate。每个信号按开发集预算校准，评估干预后的收益而非只评估对错AUROC。teacher NLL和g消融能判断student概率项究竟提供什么。

**E4：小规模端到端成本试验。** 固定方法，测chunk/KV/并发矩阵，纳入所有评分、重算和judge费用，与teacher-only及标准speculative对照。不能用训练作业总时长或单个warm评分call替代公平服务latency。

这四项优先于继续增加许多随机horizon值或大量新数据集；它们会决定最终论文是否有预算优势，以及哪些模块值得保留。

### P1：把“相关的候选”升级为“有价值的介入状态”

**E5：同prefix反事实分支。** 从未介入student轨迹中，在看到任何后续结果前采样候选状态，分层包括高g+judge接受、高g+judge拒绝、低g、匹配位置random。每个状态都运行：直接student、teacher短接管后student、student自重采样后student；使用多个独立continuation和teacher bridge，剩余介入策略一致，第一轮可设为不再介入以隔离局部效应。

目标量：

`U_h(s) = P(最终正确 | 在状态s由teacher接管h后student续写) − P(最终正确 | 从s由student续写)`。

它是给定动作生成分布和后续策略下的平均效用，而不是token的永恒属性。比较候选筛选器的正效用比例、平均效用、伤害率和预算内累计效用。独立确认集重新估计，不用同一批8次采样同时挑“最好anchor”并报告效应。

可将当前256-token离线bridge集合用作pilot和案例来源，但确认实验需要新的续写，不能从“未来收益很高”反推线上judge已能找到它们。

**E6：位置局部性与horizon。** 同prefix在anchor、附近可比位置（例如提前/推迟8或32 token）介入；h=1/2/4/8/16/32。比较同位置的效用增长，并在总teacher预算B下比较少次长接管与多次短接管。若1-token已足够，8-token不是必要设计；若若干token后收益饱和且student仍能继续完成，才有“最小有效接管跨度”的证据。

延迟位置可能让错误已经进入prefix，要说明所估计的是整套延迟策略的效果；不能声称所有位置共享完全相同的状态。用人工抽样或独立judge检查算术修正、策略改变、答案直接代写、格式修正等类别，尤其核验BBH高teacher占比时的贡献。

### P2：泛化和部署边界

- 第二个模型规模/家族，分别测试双方 non-thinking、双方 thinking；冻结阈值/在独立dev校准，不逐benchmark调最优test阈值。token级比较首先保证兼容tokenizer。
- 多seed评估主要数据集；AIME每题多次并按题聚类，不能把30题×采样次数当独立题数。
- 更多预算分配策略：固定horizon、句/步骤边界归还、基于分歧下降的归还。额外检查teacher当前认可不等于student已能继续。
- 若服务成本高，训练轻量student-side gate或做cheap prefilter；单独计数据构造与训练开销，不把“training-free”名称沿用到训练后变体。
- 如果回到OPD，单独做student-only测试的蒸馏实验，证明训练收益与监督转移；不要仅凭在线协作收益推断训练收益。

## 7. 建议论文叙事与mini-outline

### 7.1 Introduction的六个段落职责

1. **Opening：** 强模型推理质量高但串行生成昂贵；希望有限辅助提高小模型的最终正确率。
2. **Challenge：** 关键问题是为当前student状态分配teacher计算，局部概率不一致未必是错误，语义差异也未必值得修正；定位本身有代价。
3. **Evidence：** 用精确、有限定的观察说明分歧是候选风险信号：错误轨迹更常见，前中段富集；明确这只是动机。
4. **Method：** 描述最终经公平实验选定的机制：chunk draft、causal teacher评分、触发、回滚至所选位置、短接管、student继续。judge若无净价值，不写成必要组件；signed/absolute必须与实验一致。
5. **Advantage：** 阐明待检验的优势：把teacher串行生成集中到高效用状态，评分批处理并复用未改变prefix；不预先承诺总FLOPs减少。
6. **Evidence and scope：** 给相同预算与完整成本下的核心结果、反事实效用分析、适用/失败边界。只用实际完成且通过验证的数字。

不要按个人探索时间线写成“OPD问题→A不行→B不行→加judge的C成功”。A/B/C适合作为回答竞争假设的实验，而非引言的故事骨架。

### 7.2 正文结构

1. 问题定义：可观测状态、共享tokenizer、预算、teacher token成本与总成本。
2. 诊断观察：风险相关性，位置分布及混杂控制。
3. 方法：选择信号、接管动作、归还、chunk与缓存。
4. 主效果与budget frontier：先对照，再报告不同数据集。
5. 机制实验：位置/改写分离，judge效用，同prefix反事实，horizon。
6. 系统实验与限制：延迟、吞吐、草稿浪费、数值漂移、teacher接近但不等效。

### 7.3 建议图表

- Teaser：同一student状态的直接续写与短teacher接管后续写；展示后续行为和最终结果，不只画流程图。
- 主图：accuracy 对完整成本的Pareto图；另附teacher写入量横轴，避免两者混同。
- 诊断图：分歧风险、控制长度/难度后的效应、不同position桶的干预效用；把出现频率与效用画成不同panel。
- 机制图：位置策略×horizon的效用/伤害率，judge接受/拒绝的反事实收益分布。
- 系统图：chunk大小对latency与草稿接纳率；用任务间差异解释最优chunk不一定相同。

## 8. Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| 高正gap在错误轨迹中更常见 | 1000条non-thinking轨迹，差值CI与AUROC | supported，限该观察设置 |
| 强分歧在前中段更常见 | 固定g>2的20–40%富集 | supported，描述性，需跨设置复核 |
| 这些token是逻辑错误的因果起点 | 现有分布图与语义判定不足 | needs evidence |
| 在线分歧接管能改善student | 多个benchmark单seed及逐题配对检查 | supported，限已测设置 |
| Luna能在同预算下挑出更有价值的位置 | 目前judge/nojudge用量不同 | needs evidence |
| anchor优于随机仅因位置更好 | 首token ban、次数上限、分段与长度信息不匹配 | needs evidence |
| 很少teacher生成即可接近teacher质量 | Olympiad、BBH点估计；比例随任务显著不同 | partial，需完整预算及非劣检验 |
| chunk/KV减少重复评分时间 | warm局部微基准 | supported，限该局部测试 |
| 整体比teacher-only/speculative更快更便宜 | 尚无公平端到端对照 | needs evidence |
| 改进OPD监督或蒸馏后student能力 | 当前主表是冻结模型协作 | needs separate training evidence |

## 9. 五维自审与写作检查

| 维度 | 当前回答 | 下一步 |
|---|---|---|
| Contribution | 分歧与稀疏接管已有直接工作；具体效用与成本优势尚待验证 | 同预算基线与同prefix效用实验 |
| Writing clarity | 原动机与实现有signed/absolute差异，OPD与inference需要分清 | 冻结最终定义，使用稳定术语 |
| Experimental strength | 多个数据集提升明显，AIME不稳定，nojudge更强但用量高 | 多seed、完整失败案例、horizon预算对照 |
| Evaluation completeness | 缺无ban random、cheap位置、直接文献baseline和完整cost | 优先E1–E4 |
| Design soundness | chunk回滚逻辑清楚；高丢弃率、judge领域错配、缓存漂移需检验 | E4、负gap消融、决策漂移测试 |

清晰度：每段只讲一个问题；段首给主旨。衔接：风险观察只引出候选机制，不越过因果证据。术语：固定使用disagreement/候选位置/语义分岔/干预效用四个不同概念。无支持主张：已在上表标出，写作时弱化或移除。缺失证据：优先补预算公平性和端到端成本，之后再决定是否以“逻辑转折点”作为标题或核心贡献。
