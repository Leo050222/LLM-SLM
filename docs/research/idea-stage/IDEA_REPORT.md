# 研究想法报告：从 Logit 差异转向 Teacher 的因果训练价值

**方向**：在数学推理的 on-policy distillation（OPD）中，只训练 teacher logits 真正能提升 student 最终答题能力的位置或片段。  
**生成时间**：2026-08-09  
**状态**：仅做实验规划；未运行模型、未启动 GPU pilot。  
**评审口径**：same-family provisional，正式新颖性结论仍需独立 novelty check。

## 核心结论

原始想法的科学问题是对的，但“高 KL + 强制一个 teacher token + 一次 student continuation 答对”不是稳定的效用定义：

1. KL 只说明 teacher 和 student 不同，不说明 teacher 更好。现有 AIME24 reverse-KL 极端 token 中已经出现 `*`、` e`、` factors` 等符号或子词，说明大差异可能来自措辞、分词或分布尖锐度。
2. 单次 continuation 的二元正确性方差太大；一个有效 token 后面仍可能被 student 的新错误毁掉。
3. 单 token 往往不是完整的推理动作；真正有效的干预单位可能是一个短片段或完整推理步。
4. “替换后行为变好”仍不等于“对这个位置做参数更新会泛化”。共享参数会影响其他位置，forward KL 和 reverse KL 产生的更新方向也不同。

因此建议把研究问题分成三层：

- **方向有效性**：teacher 分布是否把概率质量移向更可能答对的动作？
- **可交接性**：teacher 需要接管多长片段，student 才能接着完成？
- **可训练性**：所选 KL 目标产生的梯度，是否与提高最终奖励的方向一致？

## 文献位置

- [GKD](https://arxiv.org/abs/2306.13649) 建立了在 student 自身生成分布上进行蒸馏的基本框架。
- [PW-OPSD](https://arxiv.org/abs/2605.21606) 已经提出 branch-viability diagnostic：在 student prefix 后强制 teacher 的候选 token，再让 student 继续，并检查是否恢复正确答案；它发现位置比局部不确定性更能预测可靠性。因此，原始“单 token 替换后能否答对”的想法本身已经高度相近，不能单独作为主要新意。
- [BRTS](https://arxiv.org/abs/2605.09725) 说明单个 teacher rollout 可能错误或高方差，并用 correctness-first 的 Best-of-N teacher trajectory selection 改善监督可靠性。
- [RLAD](https://arxiv.org/abs/2602.22495) 强调只有 teacher guidance 改善当前策略更新时才模仿，并引入 advantage-aware trust-region distillation。
- [AOPD](https://arxiv.org/abs/2605.06387) 指出 token-level OPD 可能存在高方差、零优势区梯度消失和探索不足，并在非正优势区采用局部 divergence minimization。
- [Rethinking Selective Knowledge Distillation](https://arxiv.org/abs/2602.01395) 说明选择性蒸馏需要区分样本、位置和词表类别等不同选择维度。
- [Trajectory-Refined Distillation (TRD)](https://arxiv.org/abs/2606.08432) 已把问题从 token reweighting 推向 teacher-guided trajectory correction，并把错误 student prefix 称为 prefix failure。这是 minimal rescue span 最大的新颖性威胁：如果只做“更短的 trajectory refinement”，很容易被视为 TRD 的局部化版本。
- [Rethinking On-Policy Self-Distillation for Thinking Models](https://arxiv.org/abs/2607.05184) 报告 privileged-context distillation 在长推理预算下可能压制分叉、自我检查和回溯，甚至降低正确率。因此“teacher 更确定”不能自动视为好信号，还要检查是否破坏 student 的探索和自纠。

## 推荐主线：Causal Handoff OPD

### 1. Distribution-level Teacher Utility

对 student rollout 的某个 prefix `s`，定义：

`Q_S(s,a) = P(最终答案正确 | prefix=s, 下一 token=a, 此后全部由 student 生成)`

teacher 在该位置的一步因果效用为：

`U_1(s) = E_{a~T}[Q_S(s,a)] - E_{a~S}[Q_S(s,a)]`

等价地：

`U_1(s) = sum_a (T(a|s)-S(a|s)) Q_S(s,a)`

它直接回答：**把下一步的动作分布从 student 换成 teacher，是否提高预期正确率？**

这比强制 teacher top-1 更好，因为它评价整个 teacher 分布；也比“teacher 分支答对”更严格，因为它减去了同一 prefix 下 student 自身的反事实基线。

最低成本估计方式：取 teacher/student top-token 的并集，在少量候选动作上用多次 student continuation 估计 `Q`，并对未覆盖的 tail mass 给出上下界。KL 只用于提出候选 prefix，不作为有效性的标签。

### 2. Minimal Rescue Span / Handoff Curve

对同一 prefix 测量：

`U_h(s) = P(correct | teacher 接管未来 h 个 token，然后交还 student) - P(correct | student 一直继续)`

其中 `h ∈ {1, 4, 16, 一个完整推理步, full-teacher}`。由此得到 teacher-to-student handoff curve，并定义最短有效接管长度 `h*`。

这个曲线会自然把位置分成四类：

| 类型 | 观测模式 | 对训练的含义 |
|---|---|---|
| Token-local | `U_1 > 0` | 适合 token-level selective OPD |
| Span-local | `U_1 ≈ 0`，但 `U_step > 0` | 应对完整推理片段训练，而不是单 token |
| Teacher-only | 只有 full-teacher 有效 | 普通局部 OPD 很可能无法传递，应考虑轨迹级训练或 curriculum |
| Harmful / unrecoverable | full-teacher 也无增益或为负 | 不应蒸馏该 prefix |

这一部分明显超出 PW-OPSD 的固定单-token branch viability：研究对象从“某 token 能否幸存”变为“teacher 知识需要多大干预剂量才能被 student 接住”。

### 3. Divergence-specific Causal Gradient Alignment

行为干预有效仍不代表 KL 参数更新有效。可以利用上面估计的 `Q`，检查具体蒸馏损失的 logit 更新方向与奖励梯度是否一致。

对 forward KL `KL(T||S)`，student logits 的梯度下降方向是：

`d_F(s) = T(.|s) - S(.|s)`

而 student value 的 logit 梯度为：

`g_R(a|s) = S(a|s) [Q_S(s,a) - V_S(s)]`

定义：

`G_F(s) = <d_F(s), g_R(s)>`

若 `G_F(s) > 0`，一个足够小的 forward-KL 更新在局部上与提高最终正确率同向。reverse KL 使用它自身真实的 logit 梯度方向计算 `G_RKL(s)`。这样同一套 `Q` 数据可以直接比较：

- teacher 行为本身是否有用；
- forward KL 是否会朝有用方向更新；
- reverse KL 是否会朝有用方向更新；
- 两种散度在哪类 prefix 上发生分歧。

这是最贴近“teacher logits 是否真的值得被训练”的量，因为它把 teacher 的行为价值和实际 loss 的更新方向连接起来。

### 4. 训练目标

最简单的主方法可以写为：

`L = sum_s w(s) D(T(.|s), S(.|s))`

其中：

`w(s) = max(0, LCB(U_h(s))) × 1[G_D(s) > 0]`

- `LCB` 是多次 paired branch 得到的效用下置信界，避免一次幸运 rollout 造成假阳性。
- `h` 取该位置的最短有效 span；权重施加在该 span 内，而不是只施加在命中的单 token。
- `D` 可以分别使用 forward KL 和 reverse KL；门控效用本身与散度类型解耦。

如果可靠性标签太稀疏，可在 discovery 数据上训练一个 cross-fitted utility gate，输入 forward/reverse KL、teacher/student entropy、top-1 disagreement、相对位置和推理步边界，预测 `P(U_h>0)`。必须按题目划分训练和验证，不能让同一道题的多个 prefix 跨集合。

## 两个备选方向

### 备选 A：Counterfactual Branch Preference Distillation

在同一 prefix 下，如果 teacher-guided span 的 continuation 显著优于 student span，就把两段组成局部 preference pair，用片段级 DPO/hinge loss，而不是拟合 teacher 的整个词表分布。

优势是减少对 teacher 表达风格和低价值概率尾部的模仿；风险是有效正负对可能稀疏，并且模型可能利用长度、标点等捷径。它适合作为主方法的 objective ablation，不建议一开始就作为唯一主线。

### 备选 B：Causal Utility × Micro-update Trainability Test

先用 branch 证明某片段行为上有效，再只对该片段做一次很小的 adapter/functional update，并在未用于更新的同类题上检查 value 或正确率是否提高。它是“值得训练”的最直接证据，但计算和噪声都更高，适合后续机制实验，而不是第一轮 discovery。

综合评审后，更强的主方法应把该测试作为第三个必要条件，而不是永久停留在可选分析：实际实现可以先用梯度对齐作为便宜近似，只对少量候选做真实 micro-update audit。

## 分阶段实验规划（不执行）

### 阶段 A：只做因果诊断

1. 在 AIME24、AIME25、HMMT25 中按 `dataset × 原始是否正确 × 相对位置 × forward/reverse KL 分位数 × top-1是否一致` 分层抽 prefix。
2. 每条 rollout 同时抽高 KL 和位置匹配的低 KL control；不能只看极端 KL。
3. 每个 prefix 比较 `student-only`、`teacher-1-token`、`teacher-step`、`teacher-full` 四种条件；discovery 至少 4 个 paired continuations，确认阶段至少 8 个。
4. 使用共同随机数或共享 Gumbel noise 降低方差；同 token 的 sham 分支必须逐 token 一致。
5. 主指标是按问题聚类的 paired ATE，而不是把几十万 token 当独立样本。
6. 截断样本记为 censored，不能直接当作答错；最终答案解析器对实验臂保持盲法。

第一阶段要回答的不是“能否找出几个漂亮案例”，而是：

- 高 KL 对正 `U_h` 的 AUROC/AUPRC 有多少？
- 高 KL token 中有多少实际上无效或有害？
- 低 KL 中是否存在高效用位置？
- `h*` 的分布是什么；大多数知识是 token-local 还是 span-local？
- forward/reverse KL 的梯度对齐分数，谁更能预测真实 branch uplift？
- teacher 干预或更新后，verification、backtracking、hedging 等自纠 marker 和 reasoning fork 是否下降？若正确率的短期增益伴随长预算探索能力损失，不能认定为值得训练。

### 阶段 B：学习可复用 gate

只用 discovery 集训练一个简单、可校准的 utility gate；在未见题目上固定阈值后验证。比较：

- position-only（对应 PW-OPSD 思路）；
- forward-KL top-q；
- reverse-KL top-q；
- entropy/top-1 disagreement；
- causal utility gate；
- 相同覆盖率的随机位置。

门控的确认指标是固定覆盖率下相对随机门的 uplift，并报告 Brier/ECE；不能只报告 AUROC。

### 阶段 C：以后再做的训练消融

只有阶段 A/B 证明 gate 在未见题上有因果 uplift 后，才值得真正参数更新。至少需要：

1. no-OPD / continued training；
2. all-token OPD；
3. KL-top-q selective OPD；
4. position-weighted OPD；
5. causal handoff OPD；
6. 相同 token 数、相同位置分布的 random-mask OPD。

所有方法必须同时比较等 optimizer steps 和等 teacher-supervised tokens 两种预算口径。若 causal 方法只胜 no-OPD、不胜 all-token，最多能声称“稀疏蒸馏可行”；只有在等预算下胜 all-token、random-mask 和 KL-only，才能声称因果筛选真的更好。

## 必要的 negative controls

- `teacher = student` 的 sham intervention，估计应为零。
- 与 teacher 方向具有相同 KL 范数，但随机旋转或跨题置换的 logit 方向。
- 正确答案 privileged teacher 与错误答案/遮蔽答案 teacher；区分真实知识和单纯更自信的分布。
- teacher/student entropy 匹配，排除只是温度或探索度改变。
- 同长度、同位置的随机 span。
- 训练题和评测题严格按 problem 分离；不能在 AIME/HMMT 同一批题的 rollout 上训练后再报告这些题的提升。

## 暂缓的想法

- 多位置 Shapley interaction：能研究“多个单点都无效但联合有效”，但组合成本高，先证明 span-level 信号存在再做。
- 多 teacher 共识方向：科学价值高，但需要多个教师或 prompt view，第一轮不必要。
- bilevel meta-gate：最直接优化验证奖励，但离散数学奖励和二阶更新噪声大。
- FDR 序贯采样、doubly robust selection：是降低分支成本和选择偏差的好工具，适合在主效应出现后工程化。

## 推荐执行顺序

1. **先验证 Causal Handoff Curve**：判断单 token 假设是否成立，以及最小有效 span。
2. **再估计 Distribution-level Utility**：证明 teacher 确实把概率质量移向高价值动作，而不是只产生偶然正确分支。
3. **然后比较 forward/reverse KL 的 Gradient Alignment**：选择真正值得更新的 loss 和位置。
4. **用少量独立题做 Micro-update audit**：验证局部梯度分数能否预测真实的小更新收益。
5. **最后才训练 gate 和 OPD 模型**。

最终建议的主 claim 不是“多 token 比单 token 好”，而是：**行为 uplift、handoff 后的最小充分接管长度、以及参数可写回性是三个不同量；只在三者同时成立时蒸馏。** 相对 PW-OPSD 这是清晰差异；相对 PW-OPSD、TRD、RLAD 的联合文献则属于“有条件成立的中强新意”，需要逐 prefix 随机干预、严格 handoff、等计算基线和三模块消融共同坐实。

本轮按用户要求没有启动任何 pilot。由于当前 completion 可长达约 28k token，完整的多分支确认实验很可能超过单 GPU 2 小时，正式预算需要在确定 prefix 数、分支数和 suffix 截断策略后单独估算。
