# Anchor 逻辑分岔验证实验计划

**问题**：256-token teacher bridge 带来的 rollout 成功率提升，是否源于 anchor 处发生了可解释的推理方向分岔？
**方法主张**：成功率提升越大的 bridge，其 anchor 越可能是严格、局部且能持续影响后续推理的逻辑分岔点。
**日期**：2026-09-08

## Claim Map

| Claim | 最低可信证据 |
|---|---|
| C1：高提升 bridge 更常对应真实逻辑分岔 | 在控制题目、结果转移类型和长度后，独立 judge 的 strict-bifurcation 概率随独立复测的 bridge effect 单调上升 |
| C2：趋势不是“看到一个错答案和一个对答案”的机械结果 | 在 Student错→Bridge对（F→S）这一固定结果类型内仍存在正趋势；placebo anchor 与 Student-vs-Student 对照显著更低 |
| Anti-claim：LLM judge 只是偏爱 bridge、更长文本或正确答案 | 路径身份、正确性与提升档对 judge 隐藏；A/B 顺序随机；长度匹配；多 judge 与人工盲审一致 |

## 核心设计

### 1. 实验单位与提升档

- 单位是 `(question, prefix/anchor, bridge_id)`，不是单条 rollout。
- 已有计数：Student-only 为 `k_S/8`，该 bridge 后为 `k_B/8`，原始提升 `Δ=(k_B-k_S)/8`。
- 每档抽 100 个不同题目；同一道题最多进入一个档、只选一个 bridge，避免同一组 Student-only 8 条被重复计算。
- 若 1/8 的精细档位不足 100，预注册合并为五档：`Δ≤0`、`(0,.25]`、`(.25,.5]`、`(.5,.75]`、`(.75,1]`。负/零档必须保留为控制。
- 分层抽样时匹配 difficulty、topic、Student baseline success、anchor token position、prefix length；报告抽样权重，不能把每档等量样本当作总体分布。
- 现有 8×8 只用于 discovery 分档。对抽中的 bridge 独立再生成至少 8 条 Student-only 与 8 条 bridge rollout，得到 confirmatory `Δ`；主趋势使用 confirmatory 值，避免 winner's curse。预算允许时推荐 16+16。

### 2. 8×8 rollout 的比较方式

不能把 replicate 0 对 replicate 0 当成天然配对。两组 rollout 是独立随机样本，每个 bridge 有 64 个可能的 Student/Bridge 路径对。

主方案：

1. 按评分把 64 对分成四个 cell：F→F、F→S、S→F、S→S。
2. 每个非空 cell 均匀无放回抽 2 对；另抽 2 个完全均匀随机对。一个 bridge 通常得到 6–10 个 judge item。
3. judge 不看到 cell、正确性、参考答案、bridge 身份或提升档。
4. 分别报告：
   - 随机路径对上的总体 strict-bifurcation rate；
   - 固定 F→S cell 内的 conditional bifurcation rate（最关键，消除结果类型的机械混杂）；
   - 四个 cell 的分解结果。
5. 统计时 pair 是观测，bridge 是核心单位，question 是 cluster；不能把数千个 pair 当作完全独立样本。

### 3. Prompt 表示与盲法

每个 judge prompt 包含：

- 原题；
- 到 anchor 为止完全相同的 prefix，并插入唯一标记 `<ANCHOR/>`；
- Path A：从 anchor 后继续的完整轨迹；
- Path B：从 anchor 后继续的完整轨迹；
- 对 bridge 路径额外标出 256-token intervention 的结束边界，但使用中性名称 `controlled span`，不写 Teacher/Student；
- A/B 顺序随机并保存 permutation；不提供 reward、正确答案、提升档和模型名。

完整 trace 可能超过 judge context。必须冻结截断规则：保留完整公共 prefix 的尾部（例如 anchor 前 1,024 tokens）、anchor 后前 512 tokens、两条轨迹结尾各 1,024 tokens；另做 10% 全文敏感性分析。若 judge 支持足够长上下文，主实验使用全文，并记录实际 token 数。

### 4. “逻辑分岔点”的操作定义

judge 不能只回答 yes/no，必须输出结构化 JSON：

- `strict_anchor_bifurcation`: yes / no / uncertain
- `divergence_onset`: first_32_tokens / 33_128 / 129_256 / after_bridge / no_semantic_divergence
- `direction_change`: error_correction / new_subgoal / changed_assumption / changed_strategy / stylistic_only / none
- `persistent_effect`: yes / no / uncertain
- `evidence_a`, `evidence_b`: 各引用一个短证据片段
- `confidence`: 1–5

严格阳性定义为：方向变化在 anchor 后 32 tokens（或第一个完整语义步骤）内出现，属于前四类逻辑变化，并持续影响后续推理；仅在 256-token span 后期才分开只能算 loose bifurcation，不能证明 anchor 定位准确。

### 5. 必须加入的对照

- Placebo anchor：同一对轨迹把标记平移到真实 anchor 前/后，测试 judge 是否只因“两条轨迹不同”就判 yes。
- Student-vs-Student：从同一 prefix 的 8 条 Student rollout 中抽两条，估计自然采样造成的背景分岔率。
- Bridge-vs-Bridge：同一 bridge 的两条 rollout，估计 bridge 后随机分歧。
- A/B swap：至少 20% item 交换顺序，检测位置偏差。
- 可选 same-length placebo span：用长度相同但无提升的 bridge 匹配高提升 bridge。

### 6. Judge 可靠性

- 使用至少两个独立于 Student/Teacher 的强 judge；temperature=0，同一冻结 rubric。
- 每个 item 每个 judge 只判一次；10% item 用措辞等价的第二版 prompt 测 prompt 稳定性。
- 人工盲审每档至少 20 个 item，并覆盖所有对照；报告 Krippendorff's alpha 或 Cohen's kappa、precision/recall。
- 主标签采用多 judge 多数票；分歧项保留 uncertain，不强制二值化。单一 judge 结果只能作为 exploratory。

## 统计分析

1. 每档画 strict/loose bifurcation rate 及 question-cluster bootstrap 95% CI。
2. 主回归：mixed-effects logistic / GEE：
   `judge_yes ~ confirmatory_Δ + outcome_cell + baseline_success + difficulty + topic + anchor_position + prefix_length + trace_length + A/B_order`，
   question 与 judge 作为聚类/随机效应。
3. 核心检验是 confirmatory `Δ` 系数是否为正，以及 F→S 子集内是否仍为正。
4. 辅助报告 Spearman 单调相关、预注册线性趋势检验和 isotonic curve；不要只做相邻档显著性。
5. 对 8 次 rollout 的二项噪声，用 beta-binomial/hierarchical shrinkage 估计潜在 `p_B-p_S`；同时报告原始离散档结果。
6. 成功标准：整体趋势为正、F→S 条件趋势仍为正、真实 anchor 明显高于 placebo，且多 judge/人工审查方向一致。

## Run Order

| 阶段 | 工作 | 决策门 |
|---|---|---|
| M0 | 冻结档位、去重/匹配、rubric 与 JSON schema | 每档数量足够且协变量平衡 |
| M1 | 30 个 bridge pilot，含全部 controls | JSON 成功率>99%，人工一致性可接受 |
| M2 | 对抽中 bridge 做独立 confirmatory 8+8 rollout | 完整率与 scorer 校验通过 |
| M3 | 多 judge 批量判定 | 无身份/答案泄漏，A/B swap 偏差可控 |
| M4 | cluster-aware 分析与图表 | 主趋势、F→S 条件趋势、placebo 三项共同解释 |

## 预算

若使用五档×100 bridge、平均 8 个 pair/bridge、2 个 judges，约 8,000 次 judge 调用；若保留九个正提升档，则约 14,400 次。先做 30-bridge pilot 估算上下文 token 与成本。最主要新增 GPU 成本来自抽中 bridge 的独立 confirmatory rollout，而不是 judge。

## Failure Interpretation

- 仅总体趋势存在、F→S 内消失：结论只是高提升档含更多“错→对”组合，不能声称 anchor 更像逻辑分岔。
- loose 上升但 strict 不升：256-token span 有用，但 anchor 点定位不准。
- 真实 anchor 与 placebo 相近：judge 在识别轨迹差异，而非定位分岔点。
- judge 趋势强但人工不一致：只能视为模型偏好，不作为机制证据。
- confirmatory Δ 不复现：原 8 次分档噪声过大，应增加 rollout 或使用层级模型，不继续解释 anchor。

