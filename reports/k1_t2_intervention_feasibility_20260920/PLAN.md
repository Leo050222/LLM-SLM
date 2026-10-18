# k=1、π=2，八次主实验：介入 token 的弱模型支持度

## 研究问题与可识别范围

研究强模型实际选择的 token 是弱模型的高支持候选，还是弱模型低支持的替代选项。概率和排名能描述这个区别，但不能单独证明该 token 对应新的逻辑路径，也不能证明它是正确选择。介入点来自严格 signed logP_student(x)-logP_teacher(x)>2，属于分歧筛选点，不应直接等同于经过语义验证的逻辑分岔点。

## 数据与边界

全量只读核查已完成：16个来源组（两类任务×八轮），36568条轨迹，399601次介入。399601次全部通过k=1、token/位置/来源一致、强弱token不同、gap>2和绝对位置对齐检查；全部启用first-token ban。35752条轨迹有介入，816条没有，227条生成超时。0次事件保存弱top-token列表，0次保存entropy字段。397361次强token在已保存teacher top16内，2240次不在其中（即便强模型自身的实际token概率也不能完全从top16读取）。这些检查是字段/对齐核查，不是重新执行评分或验证每个逻辑选择正确性。

- 范围：主表全部八个数据集，4571 题 × 8 次 = 36568 条 k1_t2 轨迹；r0 保留历史数据，r1–r7 使用新增目录。完整来源和字段覆盖见 audit.json；audit.py 仅在登录节点读取 JSON，不调用 GPU。
- 事件保存 bridge_token_ids、student_entry_token、anchor_response_position、selected_candidate；轨迹保存 prompt_token_ids、response_token_ids、token_sources。可无损重建每个事件的 token 前缀。
- candidate 中 student_probability 指的是被拒绝的弱模型 token x，不是强模型 token y。teacher_top_tokens 是强模型对原弱草稿评分时的 top16，不是弱模型分布。
- 生成 worker 返回 token IDs 和 finish_reason；未把完整生成 logits 存入完成轨迹。后七轮更明确省略 student top16，旧候选构造接口同样只消费弱模型已选 token 的概率。
- 主实验 T=1、top_p=1，强模型在第一个替换 token 上屏蔽 x，因此 y!=x 是协议强制结果。
- 历史全 token entropy replay 覆盖每数据集100题、25配置、仅历史 r0，不能作为八次完整主实验结果。其 intervention_points.csv 为计算 before/after 对而丢弃末尾事件，复用时应从 NPZ 取 before 熵以保留末尾/EOS事件。

## 第一阶段：冻结轨迹，前向补充测量

对事件位置 j，固定 c = prompt_token_ids + response_token_ids[:j]；x=student_entry_token，y=bridge_token_ids[0]。前缀必须同时排除 x、y；禁止 decode 后再 tokenize。

两模型在完全相同 c 上计算完整 logits，以 FP32 log_softmax 得到未屏蔽、T=1 的 p_w、p_s。按整条已接受轨迹进行 causal teacher forcing，在 j 对应的 hidden state 投影词表；不必每个事件单独重复 prefill。按序列长度分块前向、保留 KV 并分块词表投影，避免长序列激活和 sequence×vocabulary 张量溢出。只保存每个事件的统计和 top32/64，不保存全部巨大 logits 张量。

每行保存：dataset、problem_id、replicate、seed、source SHA256、event_index、j、prefix SHA256、x/y IDs 与文本、finish_reason、最终正确性，以及：

1. p_w(y)、log p_w(y)、弱模型排名 r_w(y)=1+数量{z_v>z_y}；另存并列数/排名区间。
2. p_w(x)、p_s(x)、p_s(y)，p_w(y)/p_w(x)、p_w(y)/max_v p_w(v)，以及 log p_s(y)-log p_w(y)。
3. 累积更高概率质量 M_w(y)=Σ_{v:p_w(v)>p_w(y)}p_w(v)，以及 top5/top10/top50 和 nucleus 90%/95% 归属（固定并列处理）。排名与绝对概率必须联合解释。
4. H_w(c)=-Σp_w log p_w、H_s(c)=-Σp_s log p_s、ΔH=H_w-H_s，单位 nats，完整词表。不得用 top16 归一化熵代替。
5. 实际强模型屏蔽后的 q_s(y)=p_s(y)/(1-p_s(x))，以及公平的弱模型重采样对照 q_w(y)=p_w(y)/(1-p_w(x))；用 logsumexp 稳定计算。额外保存两者屏蔽后的熵。模型原始不确定性主要比较 H_w/H_s，实际替换策略比较屏蔽后的分布。
6. 同一前缀独立重采样 m 次弱模型遇到 y 的概率 1-(1-p_w(y))^m（例如 m=8），仅解释同一前缀的重试成本，不得解释为八条自由 rollout 遇到该路径的概率。

数值校准：优先匹配历史模型权重、tokenizer、vLLM版本、dtype和评分设置。先抽取覆盖数据集、轮次、位置长度的事件，将 replay p_w(x)、p_s(x)、teacher top16 与历史记录比较，报告 logprob误差、排名边界变化及阈值翻转率；不会用重放 gap 重新筛选历史事件。历史记录是原打分结果，重放新增列明确命名 replay_*。无法恢复未保存的历史生成 logits 或保证 bitwise 一致。已有单点诊断 reports/logits_threshold2_replay_20260918/RESULTS.md 出现最大概率差约1.62个百分点，不能外推为全数据误差。

## 第二阶段：描述性统计与图

- 主结果同时给出 p_w(y) 与 r_w(y) 的经验CDF、概率×排名二维分布，以及累计质量分布。报告 top1/5/10/50 比例，概率 ≥10%、1–10%、0.1–1%、<0.1% 的比例；这些是操作性分箱，不是能力边界。
- 高支持例：top5且p_w(y)≥1%；低支持例：rank>50且p_w(y)<0.1%；不满足两端者保留中间类别。扫描top5/10/50、概率0.1%/1%等阈值，避免单一人为边界支配结论。
- H_w–H_s 配对散点/差值分布，分强token的弱支持度、数据集、最终对错报告。低熵只表示自信，不表示正确。
- 主结果按题等权（题内先合并八轮事件），另给事件等权，防止少数长轨迹多事件主导结论。无介入轨迹单独计数，不能人为补零支持度。
- 区间按题 cluster bootstrap，八轮和同一轨迹所有事件一起重采样；各数据集单独报告，总体使用数据集宏平均。保留r0和r1–r7分层、超时敏感性分析。
- 将空白/标点、EOS、格式token、数字/运算符、变量与内容词分层。抽取高/低支持两端的上下文进行语义审查，确认是否真正改变推理；仅一个token的低概率可能只是表述差异。
- 最终正确性只能做相关分析：一条轨迹最终正确不意味着每一次介入都正确或有益。

## 第三阶段：局部反事实，判断是否选对分路

在预先分层抽样的同一批冻结介入前缀上，分别固定下一token为 A=实际强token y，B=原弱token x，C=屏蔽x后的弱模型新采样token。随后全部仅让弱模型继续，不再介入，统一剩余token/时间预算、评分器，并用多组配对随机种子估计每条分支正确率。

核心效应 ΔV(c)=P(correct | c+y, weak continuation)-P(correct | c+x, weak continuation)。把 ΔV 与 p_w(y)/排名联合：高支持且ΔV>0支持“已有候选中的有益选择”；低支持且ΔV>0支持“有益的低支持替代路径”；ΔV≤0属于无明确收益或有害替换，不能丢弃。对多次续写噪声报告区间，不以单次成败硬标签判定。

C 区分强模型选择与简单禁止原token后重试的收益。当前 weak_resample_t2_k1_20260920 是有价值的整条策略对照，但其前缀随运行分叉、只有一轮且2000秒预算不同于历史主实验，不能替代同前缀局部因果比较。新的GPU实验应独立安排；本次仅做数据核查和实验设计。

## 可复用的历史熵数据（非本次八轮结论）

来源：experiments/token_entropy_distribution_20260914/results/intervention_points.csv，筛选k1_t2；每数据集100题的历史r0样本，事件等权，排除了缺少after位置的末尾事件。单位nats。

| 数据集 | 事件数 | 弱熵均值 | 强熵均值 | H_w>H_s比例 |
|---|---:|---:|---:|---:|
| MATH500 | 902 | 0.9291 | 0.8457 | 59.53% |
| OlympiadBench | 1824 | 1.0112 | 0.9592 | 58.33% |
| BBH | 1757 | 0.6184 | 0.5457 | 56.63% |
| CRUXEval-O | 264 | 0.4781 | 0.2129 | 71.59% |
| CRUXEval-I | 440 | 0.3408 | 0.1912 | 57.05% |
| HumanEval+ | 644 | 0.4468 | 0.2179 | 71.43% |
| MBPP+ | 620 | 0.3849 | 0.1644 | 68.87% |
| LiveCodeBench | 3047 | 0.8220 | 0.8041 | 54.09% |

历史样本中强模型平均熵较低，但不是每个介入点都更低。此表不能回答强token在弱分布中的位置，也不能推出局部正确性。
