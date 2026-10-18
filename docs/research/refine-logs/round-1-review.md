# Round 1 Review

# 顶会审稿式评审

**CALIBRATION: anchored**  
**review_independence: same-family**  
**acceptance_status: provisional**

## 总评

该提案抓住了一个真实且重要的问题：标准 OPD 虽然在 student 访问的 prefix 上获得 teacher 监督，但它依然把局部 token-level discrepancy 当作基本纠偏单位；对于已经进入错误推理分支的状态，单 token 更新可能不足以跨越实际的“恢复障碍”。

但当前版本尚未形成足够锐利的顶会贡献。最大问题不是方法模块不够，而是关键逻辑链仍然缺了一环：

\[
\text{teacher bridge 在上下文中能救回 frozen student}
\not\Rightarrow
\text{蒸馏该 bridge 后 student 能自己完成同样的纠偏}.
\]

此外，teacher takeover、future continuation validation、multi-token trajectory guidance 都已被近期工作覆盖。当前真正可能成立的独特贡献只剩下：

> 对每个 student failure prefix 估计 intervention dose-response，并在预算约束下选择最短有效恢复跨度。

这个差异是清晰的，但目前显得偏增量，而且训练规模下的 \(B=2,M=2\) 不足以可靠地估计 prefix-specific horizon。若不能证明恢复跨度具有显著异质性，并且 adaptive horizon 在等 token 预算下稳定优于全局固定 horizon，论文核心贡献将直接坍塌。

**Verdict: RETHINK**

---

## 评分

| 维度 | 权重 | 得分 | 加权分 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 8/10 | 1.20 |
| Method Specificity | 25% | 6/10 | 1.50 |
| Contribution Quality | 25% | 5/10 | 1.25 |
| Frontier Leverage | 15% | 6/10 | 0.90 |
| Feasibility | 10% | 5/10 | 0.50 |
| Validation Focus | 5% | 7/10 | 0.35 |
| Venue Readiness | 5% | 4/10 | 0.20 |
| **Weighted Composite** | 100% |  | **5.90/10** |

---

## 1. Problem Fidelity — 8/10

提案总体忠实于 Problem Anchor：

- 没有把高 KL 直接等价为因果重要性；
- 明确区分局部分布差异和需要连续状态转移的纠偏；
- 没有把 reverse KL、judge、anchor selector 包装成并列创新；
- 训练后推理只使用 student；
- 将预算匹配写入 success condition。

值得警惕的是，当前方法实际测到的可能是“teacher 向上下文注入了一段关键中间解答后，student 能否完成剩余部分”，而不一定是“student 需要跨越一个持续 \(h\) token 的恢复障碍”。过滤 teacher-complete 样本不足以完全消除这个歧义：teacher 可能没有输出最终答案，但已经完成了题目最难的子问题。

因此，“recovery barrier”在实验结果出来前应作为假设，而不是已成立的机制解释。

---

## 2. Method Specificity — 6/10

### Weakness

方法在三个关键统计对象上仍不明确。

第一，\(h^\dagger\) 是针对 prefix 的 teacher-policy-average horizon，还是针对某一条 teacher bridge 的 horizon？

当前定义：

\[
\Delta_h(s)
=
P(\text{success}\mid T_h\rightarrow S,s)
-
P(\text{success}\mid S,s)
\]

隐含对 teacher bridge 随机性的平均。但实际训练目标使用具体序列 \(z^{1:h^\dagger}\)。一个 horizon 在平均意义上有效，不代表其中每一条 teacher bridge 都适合训练；反过来，按观测成功率筛选具体 bridge 又会产生 lucky-bridge selection bias。

第二，训练集阶段的 \(B=2,M=2\) 与“LCB 超过阈值”的表述不相容。每个 \(h\) 只有两个 teacher cluster，每个 cluster 只有两个 suffix，无法给出可信的 prefix-level lower confidence bound。

第三，\(h\in\{1,16,64,128,256\}\) 只能支持“最小通过测试的离散 horizon”，不能支持一般意义上的“最小充分跨度”。如果响应曲线非单调，还需要 simultaneous confidence control，而不是逐个 \(h\) 独立判断。

### Concrete method-level fix

明确选择一个干净的 estimand：

\[
\Delta_h(s)
=
\mathbb E_{z\sim\pi_T(\cdot\mid s)}
\left[
P_{y\sim\pi_S}
\bigl(R(s,z_{1:h},y)=1\bigr)
\right]
-
P_{y\sim\pi_S}\bigl(R(s,y)=1\bigr).
\]

然后：

1. 在 prefix 层面选择 \(h_s^\dagger\)，而不是为每条幸运 bridge 单独选长度。
2. 用一组 teacher samples 估计 \(h_s^\dagger\)。
3. 用独立 teacher samples，或在 \(B\) 条 bridge 间 cross-fit，生成真正用于训练的 bridge。
4. 训练时对这些独立 bridge 均匀采样，避免直接蒸馏“因采样运气被选中”的具体序列。
5. 将术语改成 **smallest tested effective horizon**，除非统计上确实排除了所有更短跨度。

训练规模下不要声称有可靠的 per-prefix LCB。可以使用简单的经验阈值或跨 prefix shrinkage，把严格区间估计留给独立诊断集。

**Priority: P0**

---

## 3. Contribution Quality — 5/10

### Weakness

相关工作已经覆盖了贡献叙事的大部分组成：

- [TOPD](https://arxiv.org/abs/2606.00305) 已明确提出“trajectory-sampled but token-learned”的问题，并用 near-future trajectory information 识别真实分叉、向多个未来 token 分配指导。
- [Relay-OPD](https://arxiv.org/abs/2607.26057) 已提出失败 prefix 上的短 teacher takeover、student resume、teacher leg 长度干预实验，以及对 relay trajectory 的蒸馏。
- [FutureBridge-OPD](https://arxiv.org/abs/2608.01953) 已在高 disagreement 状态执行 teacher bridge，并通过随后 frozen-student continuation 验证 bridge 是否值得保留。

因此以下表述都不能作为主贡献：

- chunk 比 token 更好；
- teacher 应短暂接管；
- 应查看 teacher intervention 后的 student future；
- 只训练被未来轨迹验证过的 bridge。

当前仅存的清晰增量是：

\[
\text{固定短 bridge}
\quad\longrightarrow\quad
\text{prefix-specific dose-response 和最小有效跨度}.
\]

这个增量能否达到顶会标准，完全取决于两个经验事实：

1. 不同 prefix 的有效跨度确实高度异质；
2. 这种异质性足以让 adaptive horizon 在等 teacher-token 预算下显著优于 global best fixed horizon。

若二者任一不成立，方法就退化为 FutureBridge/Relay-OPD 的昂贵 horizon tuning。

### Concrete method-level fix

将论文压缩到一个单一命题：

> Reasoning failures have heterogeneous recovery horizons; estimating the shortest outcome-effective teacher intervention span yields a better distillation unit than either isolated tokens or a globally fixed bridge.

不要把“outcome validation”单独当创新，因为 FutureBridge 已经覆盖。不要把“bridge distillation”当创新，因为 Relay-OPD 已经覆盖。

最合理的命名和定位应围绕：

- recovery horizon；
- intervention dose-response；
- budgeted adaptive span。

例如 **Outcome-Calibrated Recovery-Span Distillation**，而不是宽泛的 Adaptive Bridge OPD。

**Priority: P0**

---

## 4. Frontier Leverage — 6/10

### Weakness

方法使用了当前有价值的元素：student roll-in、teacher intervention、verifiable outcome、多分支 continuation。但当前 horizon 估计本质上仍是固定网格上的 brute-force Monte Carlo。它需要大量 suffix generation，却没有充分利用：

- 不同 \(h\) 来自同一条嵌套 teacher path；
- 不同 prefix 之间可以共享恢复跨度的统计结构；
- 目标本质上是预算约束的 contextual dose selection，而不是对每个 \(h\) 独立做假设检验。

固定阈值 \(\delta\) 也缺少直接的资源解释。

### Concrete method-level fix

把选择规则写成预算化的单一目标：

\[
h^*(s)
=
\arg\max_{h\in\mathcal H}
\left[
\hat\Delta_h(s)-\lambda h
\right],
\]

或保持“最短有效”形式，但将 \(\delta\) 与 teacher-token budget 明确关联。

估计层面使用：

- 同一 teacher trajectory 的 nested horizons；
- bridge 为 cluster 的层级估计；
- 跨 prefix shrinkage；
- successive elimination，只在短 horizon 不够时继续测试更长 horizon。

这些是估计器和采样策略的现代化，不需要新增 learned controller 或 reward model。

**Priority: P1**

---

## 5. Feasibility — 5/10

### Weakness

计算流程存在结构性矛盾：

- 完整 \(B=8,M=8\) 可以做较可信的诊断，但无法覆盖约5000个错误候选；
- \(B=2,M=2\) 可以覆盖训练集，却不足以可靠地判断 prefix-specific recovery horizon；
- 如果 noisy horizon label 无法优于 global fixed horizon，昂贵的数据构造没有实际价值。

此外，最大开满 token 的 student suffix 是主要成本，teacher generation 本身反而不是瓶颈。若每个 prefix 评估五个 \(h\)，多数计算可能浪费在明显无效或已经充分有效的跨度上。

### Concrete method-level fix

在全量训练前先做一个“kill experiment”，而不是直接扩展 pipeline：

- 200–300个失败 prefix；
- 完整 \(B=8,M=8\)；
- 只比较三个跨度，例如 \(h=\{16,64,256\}\)；
- 检查恢复跨度异质性，以及 oracle/adaptive horizon 相对 global fixed horizon 的预算优势。

若 oracle adaptive 在这个高质量实验上都不能明显优于固定 horizon，应停止该方向。

若现象成立，训练集采用逐级测试：

\[
16\rightarrow64\rightarrow256,
\]

一旦达到经验效果阈值便停止，不为每个 prefix 无条件扫描全部长度。严格的置信区间只用于诊断集，不用于包装每条训练样本。

**Priority: P0**

---

## 6. Validation Focus — 7/10

验证方案整体是当前提案最强的部分：

- 只有两个核心 claim；
- 明确规定等 teacher-supervised token、等 optimizer step；
- 包含 global best fixed \(h\) 和 same-length unfiltered chunk；
- 将 intervention utility 与 parameter-update utility 区分开；
- 规定如果固定 \(h\) 与 adaptive 相同则主 claim 失败。

最重要的实验不是完整训练菜单，而是下面这个最小三组测试：

1. outcome-effective bridge；
2. 相同长度但 intervention-ineffective bridge；
3. 相同长度 unfiltered bridge。

对三组做相同数量的小规模参数更新，然后从原始问题进行全新 pure-student rollout。若第一组的实际训练收益没有显著更高，就说明 intervention outcome 不是有效的训练数据选择信号。

不要用更多 loss、handoff window 或 selector ablation 掩盖这一核心检验。

---

## 7. Venue Readiness — 4/10

### Weakness

当前还是一个逻辑完整的研究假设，而不是已经足够支撑投稿的方法。

两个 blocking issue 尚未解决：

1. 与 FutureBridge/Relay-OPD 相比，adaptive recovery horizon 是否足够重要；
2. behavioral intervention utility 是否能预测 parameter-update utility。

目前 supporting contribution“检验前者能否预测后者”实际上是主方法成立的必要条件，而不是附属分析。如果预测关系不存在，training method 就缺少机制基础。

### Concrete method-level fix

投稿准备顺序应该是：

1. 先验证 horizon heterogeneity；
2. 再验证 intervention-to-update transfer；
3. 只有两者都成立，才实现全量 adaptive bridge training；
4. 若只有第一项成立，应转为 measurement/diagnostic paper；
5. 若两项都不成立，应终止该方法，而不是添加 controller、DPO 或 micro-update gate 补救。

**Priority: P0**

---

## On-policy 定位审查

称其为 **fully on-policy distillation** 在技术上不诚实。

设 student rollout 到 anchor：

\[
u\sim d_{\pi_S}.
\]

这个 anchor 状态确实来自当前或近当前 student policy。但 teacher 生成第一个 token 后：

\[
(u,z_1), (u,z_1,z_2),\ldots
\]

这些 prefix 来自 teacher intervention-induced occupancy，而不是 student occupancy。训练 loss 又恰恰主要计算在这些 teacher-forced states 上。因此不能说整个监督状态分布都是 student-state on-policy。

周期性 refresh 只能减少 anchor prefix 的 staleness，不能让 teacher bridge 内部状态变成 on-policy。

以下称法技术上诚实：

- **on-policy student roll-in with counterfactual corrective supervision**；
- **student-roll-in corrective distillation**；
- **intervention-augmented distillation**；
- **hybrid corrective distillation**。

“student-state on-policy”仍容易误导，因为只有 bridge 入口状态严格来自 student。更准确的是 **on-policy entry state** 或 **on-policy roll-in**。

若名称继续使用 “OPD”，应明确写成“an OPD extension that preserves student roll-in but relaxes on-policy occupancy after intervention”。否则 reviewer 会认为是在重新定义 on-policy 来保留热门标签。

---

## Intervention success 与 parameter training 的逻辑缺口

当前 intervention 测量的是：

\[
P(R=1\mid do(z_{1:h}),s),
\]

即把一段 teacher token 强制放入上下文后，frozen student 是否能成功。

训练需要的是：

\[
P(R=1\mid \pi_{\theta'}),
\qquad
\theta'=\theta-\eta\nabla_\theta
\mathcal L_{\text{bridge}}.
\]

两者之间至少缺少三个假设：

1. student 能通过梯度更新提高 teacher bridge 首部的生成概率；
2. bridge 内 teacher-forcing 的条件分布更新能在 free-running 时保持；
3. bridge 的效果不是依赖某个极低概率的精确 token 序列。

长 bridge 尤其容易违反这些假设：teacher forcing 可以让每个局部预测看起来容易，但 student free-run 时一个早期偏差就会使后续训练过的 states 全部不可达。

因此，“outcome-effective”最多表示这段 bridge 是有效的外部干预，不能直接称为有效训练单位。最小的修复不是设计新 loss，而是把 intervention-to-update transfer 设为论文的先决实证事实。

---

## GAP

当前工作距离顶会就绪的差距，不在于缺少 anchor selector、handoff loss、learned controller 或更复杂的 confidence rule，而在于尚未建立一条连续的证据链：

\[
\text{prefix 存在异质恢复跨度}
\rightarrow
\text{估计出的跨度能泛化到新 teacher bridge}
\rightarrow
\text{蒸馏该跨度能改变 pure-student rollout}
\rightarrow
\text{等预算下优于全局固定跨度}.
\]

当前提案主要定义了第一步的测量方案，并假定后面三步会自然成立。对顶会审稿人而言，这个假定不可接受。尤其是 FutureBridge 和 Relay-OPD 已覆盖“未来验证”和“teacher takeover”后，parameter writability 是该工作剩余差异能否转化为实际贡献的决定性问题。

---

## Simplification Opportunities

1. **移除核心方法中的语义 anchor selector。**  
   主实验使用一个固定、可复现的 anchor 规则，例如首个显著 divergence peak。selector 只作为附录敏感性分析。

2. **删除主方法中的 reacquisition window。**  
   第一版只蒸馏 bridge。否则无法判断收益来自有效跨度还是额外 student suffix SFT。

3. **不把 replay 写成方法贡献。**  
   正确数据 replay 是训练稳定性措施，不进入方法名称和 contribution list。

4. **删除 per-prefix LCB 的强表述。**  
   训练时使用经验 horizon；严格统计只在独立诊断集报告。

5. **只保留一个中心变量：\(h\)。**  
   固定 anchor、固定 teacher、固定现有 KL、固定训练配比。不要同时优化 anchor、bridge、handoff 和权重。

6. **将主 claim 从“causal bridge selection”收缩为“outcome-calibrated recovery span selection”。**  
   因果性只适用于 frozen-student intervention effect，不适用于尚未验证的参数更新效果。

---

## Modernization Opportunities

1. 将 horizon selection 表述成 **contextual intervention dose-response**，而非 chunk-length tuning。
2. 使用 teacher trajectory 内的 nested horizons 和 cluster-aware uncertainty。
3. 用层级 shrinkage 共享跨 prefix 信息，避免假装 \(B=2,M=2\) 能独立估计可靠区间。
4. 用 budget-penalized utility 或 successive elimination 代替全网格穷举。
5. 将 intervention-to-micro-update correlation 作为一次关键诊断，不将 micro-update gate 产品化为新模块。
6. 对 token cutoff 使用完整句子/推理步边界，否则所谓 horizon 可能主要测量“是否把 teacher 截断在半句话中”。

---

## Drift Warning

提案目前尚未明显偏离 Problem Anchor，但存在三个高风险漂移方向：

- 从“token 纠偏不足”漂移成复杂的 causal data-selection pipeline；
- 从“最小恢复跨度”漂移成 anchor selector、judge、handoff loss 的多模块系统；
- 为保留 OPD 标签而弱化 teacher-induced states 的 off-policy 本质。

应始终保持论文唯一主线：

> Student failure prefixes differ in how much continuous corrective support they require; estimate that recovery span and test whether it is a better budget-matched training unit.

如果实验最终只证明“筛掉坏 bridge 有帮助”，那是 FutureBridge 式结论；如果只证明“teacher takeover 有帮助”，那是 Relay-OPD 式结论；如果只证明“多 token guidance 优于单 token”，那是 TOPD 式结论。只有 **prefix-specific recovery horizon 本身带来不可被固定跨度替代的训练收益**，才是这项工作的独立贡献。

# Verdict

**RETHINK**

问题本身值得继续，但当前不应直接进入全量训练。首先完成两个决定生死的小规模实验：

1. recovery horizon 是否显著异质，且 oracle adaptive 在等预算下明显优于 global fixed horizon；
2. intervention utility 是否能预测独立参数更新后的 pure-student improvement。

两项都成立后，方案可进入 **REVISE**；否则应转为更窄的现象研究，或放弃 adaptive bridge training 这一主张。
