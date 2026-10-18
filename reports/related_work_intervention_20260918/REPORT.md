# Training-free 强模型介入：现有工作与本地结果对比

核查日期：2026-09-18。结论：当前支持“少量强模型写入token即可显著提高小模型表现”，不支持“现有方法中性能最好”或“首次training-free单token介入”。未复现外部方法，未启动GPU实验。

## 本方法及数据口径

当前核心信号为实际student采样token y 上的 g=log p_student(y|prefix)−log p_teacher(y|prefix)，当 g>τ 时接管。τ=2等价于p_teacher(y)<p_student(y)/exp(2)。它是对实际动作的相对不认可度；不是单独的teacher surprisal，也不是全分布KL。若y严格采自p_student，则E[g]=KL(p_student||p_teacher)；对单次g取阈值与对全分布KL取阈值并不等价。

本次选固定k=1、τ=2、无Luna配置，而非逐数据集挑最优。Qwen3-1.7B与14B，non-thinking，temperature=1、top_p=1。实现还包含teacher首token排除原student token；这是独立于门控信号的额外机制，消融时必须控制。

准确率取 outputs/benchmark-viz/data.csv：数学/BBH及其baseline统一2026-09-16 GPT-4o-mini重评分，代码用测试评分。token与介入数量从 reports/k1_t2_length_vs_student_20260916/paired.csv 重新聚合，不使用其中可能属于旧口径的正确性字段。原始数学judge存在确认的等价表达误判，分数不能视为人工核验真值。停止规则和baseline分块方式亦非完全一致。

完整8数据集表见 own_table.md 与 ours_fixed_k1_t2.csv。固定配置MATH500/OlympiadBench/BBH分别写入9.71/17.94/18.61 token每题，占1.15%/1.19%/6.91%，准确率提升9.60/12.61/36.11pp；各自填补student–teacher准确率差距约73.8%/94.5%/98.3%。均值接近不代表统计等效。

“介入占比”必须区分：最终输出teacher token总数/最终输出token总数；逐题比例的均值；至少发生一次介入的题目比例；强模型生成步骤比例；强模型探测/评分比例。当前4470/4571题有介入，约97.79%，因此是token稀疏，非题目稀疏。

CRUXEval-O总量加权teacher占比为2.41%，但双方正常结束且无超时的784对样本为16.43%；少数超长student来源输出膨胀总分母。CRUXEval-I只有4.07强token/题，但总输出很短，占比22.14%。建议同时报告逐题比例中位数/均值、总量加权占比、结束状态。

## 直接相关工作

| 方法 | 是否training-free | 强模型使用量 | 已核实性能 | 比较限制 |
|---|---|---|---|---|
| Shen等，Reasoning Can Be Restored by Correcting a Few Decision Tokens，2026预印本 | 是；有离线阈值校准 | 每次1 token；1.7B组报告平均实际token比例约6%或16%；每题绝对数量未列 | MATH500 54.2→66.8/81.5，+12.6/+27.3pp | 1.7B-Base+8B Thinking；比例是跨任务汇总，不是MATH500专属；Accuracy和Pass@8须分开 |
| Yang等，Speculative Thinking，2025，读取2026 v2 | 是 | MATH500强token19%，总输出4582.8；粗估约871强token/题，非确认的精确均值 | R1-Distill 1.5B+32B：83.2→89.4，+6.2pp | 长思维模型，不能以绝对分数直接排本方法之上/下；比例聚合方式未明确 |
| Wang等，MentorCollab-Free，2026预印本v3 | 是；MLP版需要训练 | 探测率25%，候选16 token；该Free配置平均强token/题及比例未明确列出 | Qwen3-1.7B+14B，MATH40.0→43.2，+3.2pp | 512-token greedy限制；MATH非本地MATH500协议；不可把摘要18.4%当Free这一配置的数值 |
| Zeng等，GlimpRouter，2026 | 是 | AIME25强模型生成27%的步骤，每次接管完整一步；非token占比 | Qwen3-4B+R1-Distill32B，45.00→51.67；teacher46.67；延迟220→163秒 | 相对teacher提升10.7%是相对百分比，实际+5.00pp；非本地模型/解码设置 |
| RelayLLM，2026预印本 | 否，SFT+GRPO | 1.7B Simple/Difficulty六任务平均teacher比例0.43%/1.07%；未列强token每题 | 1.7B+8B，MATH50074.6→83.4/81.4；六任务均分42.50→49.30/49.52 | 数值包括训练收益；比例非MATH500专属；其余greedy、AIME多采样，与本地不同 |

来源：[Decision Tokens Tables3/7及§5](https://arxiv.org/html/2605.16874v1)、[Speculative Thinking Table2](https://arxiv.org/html/2504.12329v2#S4.T2)、[MentorCollab v3](https://arxiv.org/html/2602.05307v3)、[GlimpRouter Tables1/5](https://arxiv.org/html/2601.05110v1)、[RelayLLM Table1](https://arxiv.org/html/2601.05167v1#S4.T1)。

Decision Tokens是最直接概念先行：全分布CE/rKL、校准与局部窗口门控、单token接管，不改权重。我们的实际采样token概率比与它的期望分布分歧不同，但training-free、disagreement gate、one-token takeover都不能单独作为首创。其主方法也需逐token强模型评分，并明确区分诊断控制与优化推理系统。

补充效率类先例：[Pan等SpecReason](https://arxiv.org/abs/2504.07891)以强模型验证/纠正小模型步骤，报告1.5–2.5倍加速及性能改善；这类结果应按端到端延迟对照，不能把step接受/拒绝率当本地token写入率。更早训练式token路由见[Shen等Co-LLM](https://arxiv.org/html/2403.03870v2)，不应只与训练方法比较而忽略上述training-free方法。

## 能否说最好

目前不能。绝对准确率已有其他模型组合达到更高值，低于1%的teacher token占比亦已有训练方法做到；且论文间任务集合、模型规模、thinking、长度上限、采样和评分不一致，不构成统一榜单。training-free是部署优点，不能替代同条件比较。

当前适合表达：无需更新模型或训练路由器，基于小模型实际采样token的相对概率分歧触发短时强模型接管；在本地固定Qwen3-1.7B/14B非思维配置上，少量teacher写入显著提高八项任务的单次回答准确率，其中数学任务约1.2%写入即可取得较大增益。性能提升仍以当前评分口径及单次实验为限。

强模型写入量不等于强模型算力。需另记teacher scored tokens、实际decode token、重复prefill、student discarded drafts、GPU-seconds、端到端时延/吞吐。64-token分块并行评分可能摊薄打分时间，但不能仅凭写入1%推断只消耗1%teacher计算。

最小有辨别力补实验：同模型同题集同输出上限/评分/采样，复现Decision Tokens CE/rKL及entropy、MentorCollab-Free、Speculative Thinking，并放入随机/开头接管对照；阈值用独立校准集确定；比较0.5/1/2/5/10%实际teacher写入预算与端到端成本曲线。保持首token排除规则一致，或单独做开关消融。多seed按题目聚类bootstrap准确率差与置信区间，检查等价答案翻转样本。只有同成本曲线证实占优后，才适合写“在所评测training-free基线中取得最佳准确率–成本权衡”。

## 本地证据

- outputs/benchmark-viz/source.md、data.csv：统一当前分数及协议。
- experiments/nojudge_grid_20260912/ablation_core.py：正向概率差、首token排除、draft丢弃。
- reports/k1_t2_length_vs_student_20260916/paired.csv：逐题token数量。
- reports/k1_t2_positions_20260916/REPORT.md：4470/4571题有介入。
- reports/grid_math_rescore_comparison_20260916/REPORT.md：已确认的judge误判。
- reports/teacher_vs_prefix_caps_20260918/REPORT.md属于固定prefix continuation实验，不混入在线端到端本表。
