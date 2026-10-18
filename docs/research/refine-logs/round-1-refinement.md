# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem**：现有 OPD 在 student 自己访问到的状态上提供 teacher 的逐 token 分布监督，但把局部 teacher–student 差异默认当成足以纠正未来轨迹的训练单位；对于已经进入错误推理分支的 prefix，一个 token 的局部更新可能无法将 student 带回可解轨迹。
- **Must-solve bottleneck**：区分“局部 logits 不同”和“需要跨越若干连续状态才能产生可保持的下游收益”，并找到最短且足够的纠偏监督跨度。
- **Non-goals**：不把高 KL 本身当作因果重要性；不追求让 teacher 在推理时长期接管；不把新 loss、anchor selector 和 judge 同时包装成并列创新。
- **Constraints**：约一万道训练题；昂贵的 B=8、M=8 仅用于确认或高价值子集；训练后推理必须仅使用 student；与基线匹配 teacher-supervised token 和训练预算。
- **Success condition**：在未见题目、纯 student 推理下，按 prefix 自适应选择的有效 bridge 监督在等 teacher token 预算下优于 token-only、随机/未筛选 chunk 和全局最优固定长度；干预阶段估计的有效性还能预测真实参数更新后的收益。

## Anchor Check

- 原始 bottleneck 仍是 token-local supervision 无法表达一段连续纠偏状态转换。
- 修订后只保留 recovery horizon 这一个中心变量，不优化 selector、judge 或新 loss。
- 不将 on-policy 标签扩张到 teacher bridge 内部状态。

## Simplicity Check

- **Dominant contribution**：prefix-specific recovery horizon 的测量，以及其是否构成比固定跨度更好的预算匹配训练单位。
- **移除/降级**：语义 selector 改为固定 anchor 规则；reacquisition window 移至消融；replay 仅为稳定措施；per-prefix LCB 仅用于严格诊断集。
- **不新增**：learned controller、DPO、RL、reward model、micro-update gate。

## Revised Proposal

### 定位

方法称为 **Outcome-Calibrated Recovery-Span Distillation**。它是：

> on-policy student roll-in with counterfactual corrective supervision

或：

> an OPD extension that preserves student roll-in but relaxes on-policy occupancy after teacher intervention.

它不是 fully on-policy distillation。只有 teacher bridge 的入口状态严格来自当前 student；bridge 内部是 intervention-induced states。若数据每轮刷新，可称 iterative on-policy roll-in；单轮静态数据只能称 near-current/behavior-student roll-in。

### 核心假设

不同 student failure prefix 需要不同长度的连续 teacher 支持，才能让 student 在 handoff 后维持成功。固定跨度会对一部分 failure 过短、对另一部分过长。

这是待检验假设，不预先把它解释成 recovery barrier。teacher 可能只是完成了最难子问题，因此需要报告 bridge 对后续 student 实质推理量和 teacher-complete/near-complete 的过滤。

### estimand

对 prefix `s`：

`Delta_h(s) = E_{z~pi_T}[ P_{y~pi_S}(R(s,z_1:h,y)=1) ] - P_{y~pi_S}(R(s,y)=1)`。

它是 teacher-policy-average intervention effect，而不是某一条 lucky bridge 的效果。严格诊断集上定义 smallest tested effective horizon；训练集只使用经验 horizon 或跨 prefix shrinkage，不声称 B=2,M=2 能产生可靠 per-prefix LCB。

### 数据与训练

1. 固定一个可复现 anchor rule，例如第一个显著 divergence peak。
2. 用多条 teacher trajectory 的 nested horizons 估计 `Delta_h(s)`。
3. 估计 horizon 的 teacher samples 与实际训练 bridge 分开，或对 bridge 做 cross-fitting，避免 lucky-bridge selection bias。
4. 只蒸馏独立采样的有效长度 bridge；anchor 前 student prefix 不计算 loss。
5. 第一版固定现有 KL/CE，不加入 handoff suffix loss；正确数据 replay 仅作稳定措施。
6. 更新 student 后重新收集 failure prefixes，以保持 roll-in 接近当前策略。

训练 loss：

`L_bridge = (1/h_s) sum_j D(pi_S(.|x,u,z_<j), pi_T(.|x,u,z_<j))`。

### 两个生死条件

1. **Horizon heterogeneity**：高质量诊断集上，oracle/adaptive recovery span 在等 teacher-token 预算下明显优于 global best fixed span。
2. **Intervention-to-update transfer**：outcome-effective bridge 在相同训练 token、相同更新步数下，比 ineffective/unfiltered matched bridge 带来更高的 pure-student rollout 改善。

只有两项都成立，才进入全量 adaptive training。若只有第一项成立，转为 measurement/diagnostic paper；若第一项不成立，固定 bridge 足够，主方法终止；若第二项不成立，说明 external rescue 不是可写回的训练单位。

### 最小 kill experiments

- 200–300 个失败 prefix，B=8、M=8，先用 `h={16,64,256}` 或语义完整的 step boundary。
- 报告 teacher bridge 为 cluster 的不确定性、非单调性和 prefix 异质性。
- 在 held-out prefix/问题上，对 effective、same-length ineffective、same-length unfiltered bridge 做相同的小规模参数更新，随后从原题做全新 pure-student rollout。

### 唯一主 claim

> Reasoning failures have heterogeneous recovery horizons; when intervention effectiveness transfers to parameter updates, distilling the shortest outcome-effective span yields a better budget-matched training unit than isolated tokens or a globally fixed bridge.

如果实证不能支持后半句，只保留 finding-driven claim，不把方法包装成新的 OPD framework。
