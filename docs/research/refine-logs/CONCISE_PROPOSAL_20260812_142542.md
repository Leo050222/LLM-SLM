# 简要研究 Proposal：Outcome-Calibrated Recovery-Span Distillation

## 1. 背景与动机

传统 On-Policy Distillation（OPD）让 student 先生成自己的 rollout，再在 student 访问到的每个 prefix 上匹配 teacher 的输出分布。近期方法进一步提出，只训练 teacher–student 差异较大的关键 token，从而减少无效监督。

但这类方法隐含了一个假设：**只要纠正关键位置上的一个 token，student 就能够回到正确轨迹。** 对复杂推理而言，这一假设可能不成立。某些错误不是单个 token 的选择错误，而是错误假设、错误子目标或错误推理方向造成的连续状态偏移。即使在高差异 token 上施加 KL loss，后续监督仍可能条件化在已经损坏的轨迹上，无法形成一条连贯的纠偏路径。

因此，我们关心的核心问题不是“哪个 token 最重要”，而是：

> 对于一个 student failure prefix，teacher 至少需要提供多长的连续纠偏片段，才能让 student 在 teacher 退出后继续完成问题？

我们将这个长度称为 **recovery span**。

## 2. 核心假设

不同 failure prefix 具有不同的 recovery span：

- 简单的局部错误可能只需要一个或少量 teacher token；
- 逻辑转向或中间结论错误可能需要一个完整推理片段；
- 部分问题即使 teacher 长时间介入，student 仍无法接手。

因此，单 token 监督可能不足，而固定长度的 teacher bridge 又可能对部分样本过短、对另一些样本过长。**按 prefix 选择 outcome-effective recovery span，可能是比单 token 或全局固定 chunk 更合适的训练单位。**

## 3. 方法

### 3.1 获取 student failure prefix

使用当前 student 对训练题进行 rollout。主要在稳定错误或低成功率题目上运行 bridge pipeline，同时保留一部分正确题用于检测能力退化和训练 replay。

在错误 rollout 上，根据预先固定的规则提出 anchor，例如第一个显著的 teacher–student divergence peak，并将其对齐到完整句子或推理步骤边界。Anchor 规则在观察后续成功率之前冻结。

### 3.2 估计 teacher handoff response curve

对于固定 prefix (s)：

1. Student-only 多次 rollout，估计基线成功率；
2. Teacher 独立采样 (B) 条最长 bridge；
3. 在 (h\in\{1,16,64,128,256\}) 处将生成权交还给 frozen student；
4. 对每个 \((b,h)\) 采样 (M) 条 student continuation；
5. 比较最终可验证成功率。

定义：

\[
\Delta_h(s)=
P(\mathrm{success}\mid T_h\rightarrow S,s)
-P(\mathrm{success}\mid S,s).
\]

选择最短的、能够可靠提升成功率的测试跨度 (h^\dagger)，称为 **smallest tested outcome-effective span**。诊断实验可使用 (B=8,M=8)；大规模数据构造采用较小预算初筛，再对高价值样本确认。

需要排除 teacher 已经直接给出最终答案的样本，确保成功来自 teacher bridge 后 student 的实质接手，而不是 teacher 替 student 完成整道题。

### 3.3 构造训练数据

每条训练样本包括：

\[
(x,\;u,\;z^{1:h^\dagger}),
\]

其中 (x) 是题目，(u) 是到 anchor 为止的 student prefix，(z^{1:h^\dagger}) 是 teacher corrective bridge。

用于选择 horizon 的 teacher samples 与真正用于训练的 bridge 应相互独立，或采用 cross-fitting，避免把偶然成功的 lucky bridge 直接作为监督。

### 3.4 训练目标

Anchor 之前的 student prefix 只作为条件，不计算 loss；在 teacher bridge 上计算长度归一化的 KL 或交叉熵：

\[
\mathcal L_{\mathrm{bridge}}
=\frac{1}{h^\dagger}
\sum_{j=1}^{h^\dagger}
D\!\left(
\pi_S(\cdot\mid x,u,z^{<j}),
\pi_T(\cdot\mid x,u,z^{<j})
\right).
\]

训练时混入原本正确的数据，防止模型只学习“犯错后恢复”而损害正常能力。模型更新后重新进行 student rollout 并刷新 failure prefixes，以减轻旧数据造成的状态分布漂移。

## 4. 方法定位

该方法不是 fully on-policy distillation。准确描述是：

> **On-policy student roll-in with counterfactual corrective supervision.**

Anchor 入口来自当前 student 的 rollout，因此 roll-in 是 on-policy；但 teacher bridge 内的状态由 teacher intervention 产生，不属于纯 student occupancy。如果每轮训练后重新收集 student rollout，可以称为 iterative on-policy roll-in。

## 5. 核心验证

需要验证两个关键问题：

1. **Recovery-span heterogeneity**：不同 prefix 的有效跨度是否确实不同？自适应跨度在相同 teacher-token 预算下是否优于全局最优固定跨度？
2. **Intervention-to-update transfer**：一个 bridge 能救回 frozen student，是否意味着蒸馏它以后 student 能够自行完成纠偏？

第二个问题需要在相同训练预算下比较：

- outcome-effective bridge；
- 相同长度但无效的 bridge；
- 相同长度的未筛选 bridge。

训练后分别评估：

- 原训练 prefix 上的 fresh free-run，检验纠偏是否能够写入参数；
- 完全未见问题上的 pure-student rollout，检验是否能够泛化。

主要基线包括 token-only selective OPD、随机/未筛选 chunk、global best fixed bridge，以及等监督 token 预算的标准 OPD。

## 6. 预期贡献与结论边界

本工作的核心贡献不是“使用 teacher chunk”，而是：

> 将 OPD 的训练单位从局部差异 token，重新定义为由下游结果校准的、prefix-specific recovery span。

如果实验成立，我们可以主张：不同推理失败需要不同长度的连续 teacher 支持；在相同 teacher 监督预算下，蒸馏最短有效 recovery span 比单 token 监督和固定长度 bridge 更有效。

如果自适应跨度不能优于全局固定跨度，则方法贡献不成立，最多保留 recovery-span 的现象分析；如果 intervention success 不能预测训练后的提升，则 teacher bridge 只能作为有效的外部推理干预，不能被视为有效的训练数据选择信号。
