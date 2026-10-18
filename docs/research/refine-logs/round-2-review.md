# Round 2 Re-evaluation

**CALIBRATION: anchored**  
**review_independence: same-family**  
**acceptance_status: provisional**

## 总评

本轮修订是实质性收敛，而不是表面重命名。原 Problem Anchor 被完整保留；方法从“anchor selector + teacher bridge + horizon scan + handoff loss + replay”的潜在多模块系统，压缩为单一中心变量：**prefix-specific recovery horizon**。对 on-policy 的表述也已基本技术诚实：student 只负责 on-policy roll-in，teacher bridge 内部明确属于 intervention-induced occupancy，不再把整个训练分布包装为 fully on-policy。

本轮最重要的进步是把两条原本被默认成立的逻辑明确改成生死条件：

\[
\text{horizon heterogeneity}
\quad\text{和}\quad
\text{intervention-to-update transfer}.
\]

两个 kill experiments 在设计层面确实瞄准了上一轮的两个 blocking gaps。但是它们目前仍是实验计划，而不是已获得的证据；并且第二个 kill experiment 仍需区分“同一训练 prefix 上是否可写回”与“对未见问题是否能泛化”。因此方案已从 **RETHINK** 提升为一份值得执行的、边界清楚的研究计划，但还没有达到 READY。

**Weighted Composite: 7.25/10**  
**Verdict: REVISE**

---

## 评分

| 维度 | 权重 | 得分 | 加权分 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9/10 | 1.35 |
| Method Specificity | 25% | 8/10 | 2.00 |
| Contribution Quality | 25% | 6/10 | 1.50 |
| Frontier Leverage | 15% | 7/10 | 1.05 |
| Feasibility | 10% | 7/10 | 0.70 |
| Validation Focus | 5% | 8/10 | 0.40 |
| Venue Readiness | 5% | 5/10 | 0.25 |
| **Overall** | **100%** |  | **7.25/10** |

---

## 1. Problem Fidelity — 9/10

原 Problem Anchor 被逐项保留，而且修订没有用额外模块转移问题：

- Bottom-line problem 仍是 token-local supervision 不一定能纠正未来轨迹；
- bottleneck 仍是识别需要连续 teacher support 的跨度；
- 不把高 KL、selector、judge 或新 loss 主张为创新；
- 最终评估仍要求未见题目上的 pure-student inference；
- 等 teacher-supervised token 与训练预算仍是成功条件。

修订还正确降低了“recovery barrier”的认识论地位：它现在是待检验假设，而不是预设机制。提案承认 teacher bridge 可能只是替 student 完成了最难的子问题，这是必要且诚实的边界。

仍需注意：过滤 teacher-complete 或 near-complete 只能排除最明显的答案泄漏，不能完全证明 bridge 的作用是“跨越状态吸引域”而非“提供关键中间结论”。不过这一点不必再扩展成新检测模块；只要论文把机制语言限制为 **recovery span as an operational measurement**，而不是声称发现了内部认知障碍即可。

---

## 2. Method Specificity — 8/10

本轮已经明确了此前最模糊的统计对象：

\[
\Delta_h(s)
=
\mathbb E_{z\sim\pi_T}
\left[
P_{y\sim\pi_S}(R(s,z_{1:h},y)=1)
\right]
-
P_{y\sim\pi_S}(R(s,y)=1).
\]

这是 teacher-policy-average intervention effect，而不是 lucky bridge 的个体效果。估计 horizon 的 teacher samples 与训练 bridge 分开或 cross-fit，也直接回应了 selection bias。训练目标保持现有 KL/CE，并把 anchor 前 prefix mask 掉，方法已经足够简洁。

仍有三个实现层面的定义需要在真正运行前固定：

1. “第一个显著 divergence peak”中的显著阈值、平滑窗口与 peak tie-breaking 尚未给出；如果规则在观察 outcome 后调整，会重新引入 selector overfitting。
2. 训练集上的“经验 horizon 或跨 prefix shrinkage”仍是二选一，没有形成唯一可复现算法。
3. smallest tested effective horizon 的 acceptance threshold 尚未与 baseline success rate、teacher-token cost 或最小效果量绑定。

这些问题不需要新增 learned controller。最小修复是预注册一个固定 anchor rule、一个三档 horizon grid，以及一个唯一的经验选择规则，然后在整个 kill experiment 中冻结。

---

## 3. Contribution Quality — 6/10

### Weakness

dominant contribution 已经明显集中，现在可以用一句话准确表述：

> 不同 failure prefix 具有不同的 outcome-effective teacher support span；估计该 span 可能比 isolated token 或全局固定 bridge 提供更好的预算匹配训练单位。

但是贡献仍处于拥挤前沿中的窄增量。TOPD 已覆盖 multi-token future guidance，Relay-OPD 已覆盖 teacher takeover 与 student resume，FutureBridge-OPD 已覆盖由 future student trajectory 验证 teacher bridge。当前独立增量几乎完全依赖 **prefix-specific dose-response / recovery-horizon heterogeneity**。

因此，“更集中”不等于“已经足够强”。如果实验只发现短、中、长 bridge 有一些性能差异，但 global best fixed span 在等预算下基本追平 adaptive span，这个贡献仍不足以支撑顶会主方法论文。即使 oracle adaptive 有优势，如果廉价的可部署估计器无法恢复该优势，也只能形成 measurement finding。

### Concrete method-level fix

不要再增加任何模块。把贡献强度绑定到两个预先声明的经验门槛：

1. cross-fitted adaptive/oracle horizon 在等 teacher-token 预算下应稳定优于 global best fixed horizon；
2. 廉价训练阶段的 horizon estimator 必须保留 oracle advantage 的实质比例，而不是只在昂贵 \(B=8,M=8\) 诊断条件下成立。

论文的主图应是 prefix-level response curves 与 budget–performance frontier，而不是系统框图。若第二个条件不成立，应诚实改写成恢复跨度的测量论文，而非声称提出可扩展训练方法。

**Priority: P0**

---

## 4. Frontier Leverage — 7/10

修订合理使用了当前前沿工具，但没有为了显得先进而堆叠模型：student roll-in、teacher counterfactual intervention、nested horizons、cluster-aware uncertainty、cross-fitting 和 pure-student evaluation 共同形成了一个现代且克制的测量框架。

最值得保留的是 cross-fitting，因为它解决的不是工程问题，而是核心 estimand 与训练样本之间的选择偏差。相对而言，跨 prefix shrinkage、successive elimination 和 semantic step boundary 都应被视为规模化手段，而不是主贡献。

当前的 frontier leverage 仍受限于固定网格 brute-force scan。只要 kill experiment 先证明现象成立，后续可以用逐级 \(16\rightarrow64\rightarrow256\) 的 successive testing 降低成本；在现象尚未成立前，不应提前引入复杂 contextual bandit 或 learned horizon predictor。

---

## 5. Feasibility — 7/10

200–300 个失败 prefix、\(B=8,M=8\)、三个 horizon 的 kill experiment 是可执行的，且足以先判断方向是否值得扩展。相较上一轮直接面向约5000条错误题构建大规模 pipeline，本轮将高质量估计限制在诊断集，显著降低了计算风险。

主要剩余可行性问题是从诊断设置到训练设置的落差：

\[
B=8,M=8
\quad\longrightarrow\quad
B=2,M=2\text{ 或其他廉价估计}.
\]

如果 prefix-specific horizon 只能在昂贵条件下稳定估计，方法在一万题规模上仍不可用。解决它不需要先覆盖全量数据；应在同一诊断集上对子采样 \(B,M\)，绘制估计准确率、固定-span regret 与计算成本的曲线，选出最小可接受配置。

第二个 kill experiment 的“相同小规模参数更新”也必须按数据组训练独立 student clones，而不是为每条 bridge 做一个完整 checkpoint；否则计算和梯度干扰分析都会失控。

---

## 6. Validation Focus — 8/10

两个 kill experiments 已经抓住最关键的证据链，没有扩张成巨型 ablation menu：

### Kill experiment 1：horizon heterogeneity

它能够回答：

- prefix-level response curve 是否真实异质；
- global fixed span 是否已经足够；
- adaptive/oracle span 是否存在预算优势。

这里必须把 **oracle** 与 **deployable adaptive estimator** 分开。Oracle 使用完整 outcome 信息，是上界和方向生死测试；adaptive estimator 必须只用允许的训练阶段观测，并在独立 teacher samples 或 cross-fit folds 上评估。不能把 oracle gain 直接当作方法 gain。

### Kill experiment 2：intervention-to-update transfer

它直接检验此前最关键的逻辑缺口：有效外部干预是否也是更好的参数监督。但当前措辞把两个不同问题混在一起：

1. **Writability**：在某些训练 prefix 上蒸馏 bridge 后，student 能否从这些原始 prefix 自己生成纠偏路径？
2. **Generalization**：在一组训练题上蒸馏 effective bridges 后，是否改善未参与更新的新问题？

最小且干净的实现是：构造 effective、same-length ineffective、same-length unfiltered 三个等规模训练集合，分别从同一初始 checkpoint 训练独立 clones；然后同时报告训练 prefix 的 fresh free-run 与完全 disjoint problem set 的 pure-student rollout。前者测试可写回性，后者测试训练价值。无需增加新的 loss 或 gate。

因此，两个 kill experiments **在设计意图上补齐了逻辑链，但尚未由结果闭环**。这一结论应明确写入项目决策，而不能因为实验已经被列出就视为 blocking issue 已解决。

---

## 7. Venue Readiness — 5/10

### Weakness

方案现在具有投稿级的问题定义、诚实定位和可证伪实验，但还没有任何结果证明两个生死条件成立。顶会 readiness 不能仅依据实验设计评分。

剩余 blocking issues 是：

1. recovery horizon heterogeneity 是否足以产生相对 global fixed span 的实际预算收益；
2. expensive diagnostic horizon 是否能转化为廉价、可规模化的训练时选择；
3. intervention effect 是否能预测 parameter update utility；
4. 该关系是否只在训练 prefix 上成立，还是能改善未见问题。

### Concrete method-level fix

按顺序运行两个 kill experiments，并使用预注册的停止规则。只有在 Kill 1 显示可利用的 adaptive advantage、Kill 2 同时显示 writability 和 held-out improvement 后，才进入全量训练。不要在结果不支持时通过 handoff loss、DPO、RL 或 learned selector 挽救故事。

**Priority: P0**

---

## On-policy 定位复审

本轮定位是诚实的：

> on-policy student roll-in with counterfactual corrective supervision

以及：

> an OPD extension that preserves student roll-in but relaxes on-policy occupancy after teacher intervention.

这两种说法都准确区分了：

\[
s\sim d_{\pi_S}
\quad\text{versus}\quad
(s,z_{1:j})\sim d_{\pi_T\rightarrow\pi_S}.
\]

“iterative on-policy roll-in”也可以接受，前提是每轮更新后确实重新从当前 student 收集 prefix。需要保留两个措辞护栏：

- 不称 training objective 本身为 fully on-policy；
- 不把 teacher bridge 内部 prefix 称为 student-visited states。

论文标题最好不强行包含 OPD。正文可以把它定位为 OPD extension，但主名称使用 Outcome-Calibrated Recovery-Span Distillation 更能避免术语争论。

---

## Dominant Contribution 与方法简化复审

dominant contribution 已经从多个组件收敛为：

\[
\text{measure prefix-specific recovery horizon}
\rightarrow
\text{distill an independently sampled bridge at that horizon}.
\]

这是一个足够清楚、可以被单独证伪的命题。方法也已达到“smallest adequate intervention”的合理形态：固定 anchor、三个候选 span、nested teacher samples、cross-fit、现有 KL。reacquisition、replay、selector 和新 loss 均没有继续污染主贡献。

但是需要避免一句话 claim 中的条件从句掩盖问题：

> when intervention effectiveness transfers to parameter updates

这不是一般性的适用条件，而是该方法必须通过实验证明的核心事实。正式论文若成立，应直接报告该 transfer；若不成立，则不能保留训练方法 claim。

---

## GAP

与上一轮相比，概念与方法设计上的 GAP 已显著缩小。当前剩余差距主要不是“还缺什么模块”，而是三段实证桥梁：

\[
\text{heterogeneous response curves}
\rightarrow
\text{cross-fitted/selectable horizons}
\rightarrow
\text{parameter-writeable and generalizable gains}.
\]

Kill 1 覆盖第一段并部分覆盖第二段；Kill 2 覆盖第三段。真正需要修正的是评估单位，而不是添加组件：oracle 与 adaptive 要分开，same-prefix writability 与 held-out generalization 也要分开。完成这两个区分后，实验计划已经足够，不需要扩展。

---

## Drift Warning

当前修订没有偏离原 Problem Anchor，且比上一轮更忠实。后续最危险的漂移有四种：

1. **把 oracle horizon 当成可部署 adaptive method。** 完整 \(B=8,M=8\) 扫描得到的上界不能替代训练时廉价选择器。
2. **把训练 prefix 上的记忆改善当成泛化。** same-prefix free-run 只能证明 writability，不能证明未见题收益。
3. **把 intervention-induced states 重新包装为 on-policy states。** 刷新 student roll-in 不改变 bridge 内部 occupancy 的性质。
4. **在 kill experiment 失败后增加模块。** 若 fixed span 足够或 intervention utility 不可写回，应按预定规则收缩/终止，而不是引入 handoff loss、judge、DPO 或 controller。

必须继续维持唯一主线：**prefix-specific recovery horizon 是否是比 fixed bridge 更好的预算匹配训练单位。**

---

## Simplification Opportunities

1. **核心实验只保留三个跨度。** 使用 \(h\in\{16,64,256\}\) 或三个预先定义的完整 reasoning-step boundary，不同时展开五档 token horizon 和多种语义 horizon。
2. **固定一个 anchor rule。** 不把“高 KL + 语义边界”做成组合选择器；先用首个冻结阈值的 divergence peak。
3. **只用一个训练 objective。** 主实验固定 bridge-only KL/CE；handoff suffix loss 仅在主结果成立后做一个消融。
4. **cross-fitting 只解决 sample reuse bias。** 不再同时加入连续权重、LCB gating 和 learned horizon prediction。
5. **将正确 replay 移出方法框图。** 它是所有方法一致采用的稳定措施，不是 recovery-span method 的组成。
6. **将“shortest”限定为 tested grid。** 全文使用 smallest tested outcome-effective span，避免过度声称连续最优。

---

## Modernization Opportunities

1. 使用 bridge-cluster bootstrap 或层级二项模型估计诊断集 response curves，避免把同一 teacher bridge 下的 \(M\) 个 suffix 当独立样本。
2. 使用 cross-fitting：一半 teacher bridges 选 horizon，另一半估计 intervention utility/构造训练样本，再交换 folds。
3. 在诊断结果成立后，使用 successive elimination 逐级测试 \(16\rightarrow64\rightarrow256\)，而不是全量扫描。
4. 报告 budget–utility frontier：横轴为 teacher-generated/supervised tokens，纵轴分别为 intervention success 与训练后 pure-student gain。
5. 对廉价 \(B,M\) 子采样绘制 estimator regret，而不是直接训练 learned horizon predictor。
6. token 截断点优先吸附到最近完整句子或 reasoning-step boundary，但该规则必须预先固定，不能由 outcome 反向选择。

这些机会应严格按顺序采用：只有 kill experiments 证明主现象后，才实施规模化优化。

---

## Remaining Action Items

### P0 — 必须先完成

1. **冻结 operational definitions。** 明确 anchor threshold、三个 horizon、success verifier、teacher-complete/near-complete 过滤、baseline rollout 数和经验 horizon selection rule。
2. **运行 Kill 1，并区分 oracle 与 adaptive。** 使用 cluster-aware uncertainty 和 cross-fitting；比较 global fixed span、oracle span、训练时可用的 cheap adaptive span，全部匹配 teacher-token budget。
3. **预先定义 Kill 1 的停止条件。** 如果 oracle adaptive 无实质优势，终止训练方法；如果 oracle 有优势但 cheap estimator 无法恢复，则转 measurement paper。
4. **运行 Kill 2 的三组 clone 实验。** Effective、same-length ineffective、same-length unfiltered 使用相同题目数、监督 token、更新步数和初始 checkpoint。
5. **分开报告 writability 与 generalization。** 同一训练 prefix 上 fresh free-run 测可写回性；完全未参与更新的问题测泛化。

### P1 — 仅在两个 kill experiments 通过后

6. 在同一诊断集对子采样 \(B,M\)，确定一万题规模下的最小可用估计预算。
7. 实现逐级 horizon testing 和一次 on-policy roll-in refresh。
8. 进行完整等预算比较：token-only、same-length unfiltered bridge、global best fixed span、adaptive span。无需先扩展更多基线。

### P2 — 主结果成立后才考虑

9. 做一次 handoff suffix loss 消融和一次 semantic boundary 敏感性分析。
10. 扩展第二个 student scale 或第二类任务，检验 recovery-horizon finding 是否具有外部有效性。

---

# Verdict

**REVISE — 7.25/10**

本轮已经完成了研究设计层面的关键修正：Problem Anchor 保留，dominant contribution 集中，方法更简单，on-policy 定位诚实，两个 kill experiments 也对准了此前缺失的逻辑链。当前不需要再发明新方法，而需要执行这两个实验并严格区分：

\[
\text{oracle vs. deployable adaptive}
\quad\text{以及}\quad
\text{same-prefix writability vs. held-out generalization}.
\]

在缺少这些结果前仍有 blocking issues，因此不能判定 READY。若 Kill 1 和 Kill 2 均通过，下一轮可进一步提升至强 REVISE；只有在跨模型/任务的等预算主结果稳定成立、且没有未解决的可扩展性问题后，才可能达到 READY 所要求的 \(\ge 9\)。
