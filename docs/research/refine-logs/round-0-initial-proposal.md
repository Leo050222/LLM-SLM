# Research Proposal: Outcome-Calibrated Adaptive Bridge Distillation

## Problem Anchor

- **Bottom-line problem**：现有 OPD 在 student 自己访问到的状态上提供 teacher 的逐 token 分布监督，但把局部 teacher–student 差异默认当成足以纠正未来轨迹的训练单位；对于已经进入错误推理分支的 prefix，一个 token 的局部更新可能无法将 student 带回可解轨迹。
- **Must-solve bottleneck**：区分“局部 logits 不同”和“需要跨越若干连续状态才能产生可保持的下游收益”，并找到最短且足够的纠偏监督跨度。
- **Non-goals**：不把高 KL 本身当作因果重要性；不追求让 teacher 在推理时长期接管；不把新 loss、anchor selector 和 judge 同时包装成并列创新。
- **Constraints**：约一万道训练题；昂贵的 B=8、M=8 仅用于确认或高价值子集；训练后推理必须仅使用 student；与基线匹配 teacher-supervised token 和训练预算。
- **Success condition**：在未见题目、纯 student 推理下，按 prefix 自适应选择的有效 bridge 监督在等 teacher token 预算下优于 token-only、随机/未筛选 chunk 和全局最优固定长度；干预阶段估计的有效性还能预测真实参数更新后的收益。

## Technical Gap

“On-policy”应按状态来源而不是监督动作来源定义。标准 OPD 的关键优势是 prefix 来自 student roll-in，teacher 只在 student 访问的状态上给标签。这里仍从当前 student rollout 中选择 anchor，因此起始状态分布是 on-policy；但 teacher bridge 及其内部状态来自反事实 teacher intervention，训练目标是 hybrid corrective supervision，而不是纯 student-trajectory OPD。

现有局部选择默认单 token 信号可以写回模型；固定 chunk 方法又默认所有错误需要相同的纠偏跨度。两者都忽略 prefix-specific recovery barrier：短干预可能无法离开错误吸引域，过长干预则浪费 teacher token、放大风格模仿并削弱 student 的接管能力。

## Method Thesis

- **One-sentence thesis**：在 student 访问的失败状态上，teacher 纠偏具有 prefix-dependent temporal support；估计 teacher-to-student handoff response curve，并只蒸馏最短可靠有效 bridge，可将局部 OPD 的训练单位从 proxy-selected token 改为 outcome-validated corrective transition。
- **Smallest adequate intervention**：一个反事实长度扫描、一个 acceptance rule、一个现有 KL/CE 训练目标；不新增可训练 judge 或复杂 controller。
- **Positioning**：student-state on-policy, intervention-augmented distillation；简称 adaptive bridge OPD 时必须明确它不是 fully on-policy trajectory training。

## Contribution Focus

- **Dominant contribution**：最小充分 teacher handoff span 的可测定义和预算匹配的自适应 bridge 训练机制。
- **Supporting contribution**：证明 intervention utility 与 parameter-update utility 是不同量，并检验前者能否预测后者。
- **Explicit non-contributions**：高 KL anchor、LLM judge、reverse KL、短 handoff loss 均是可替换组件，不单独主张创新。

## Proposed Method

### Complexity Budget

- 冻结/复用：student、teacher、verifier、现有 OPD loss。
- 新增：离线/周期性 handoff curve 估计与 bridge acceptance。
- 暂不引入：learned horizon controller、DPO、RL、可训练 reward model。

### System Overview

1. 当前 student 对题目 rollout，得到 student-visited prefix `s`。
2. 用高 divergence 局部峰值加语义边界 selector 提出少量 anchor；selector 在效果评估前冻结。
3. 从同一 `s` 采样多个 teacher bridges，并在 `h ∈ {1,16,64,128,256}` 交还给冻结 student，多次 continuation。
4. 估计 `Delta_h(s)=P(success|T_h→S,s)-P(success|S,s)`，选择最短 demonstrated-effective horizon `h†`，要求其 LCB 超过效应阈值；若声称统计意义的“最小”，还要求所有更短 h 的 UCB 未超过阈值。
5. 只保留 teacher 未完成答案且 student 在 handoff 后仍需实质推理的 bridge。
6. 从原始 student prefix 条件化，蒸馏被接受 bridge；可选训练很短的 reacquisition window。
7. 更新 student 后重新 rollout，周期性刷新数据。

### Training Objective

对样本 `(x,u,z^{1:h†},v)`，anchor 前的旧 student prefix `u` 只作条件，不计算 loss：

`L_repair = mean_j D(pi_student(.|x,u,z^{<j}), pi_teacher(.|x,u,z^{<j}))`

可选：

`L = L_repair + lambda_hand L_reacquire(v^{1:q}) + lambda_keep L_correct-replay`

第一版用二值 acceptance；长度内取平均，避免长 bridge 天然获得更大梯度。每题每 epoch 最多采一条有效 bridge，避免多 bridge 题目过权。正确轨迹 replay 用于保持原能力。

### On-policy Status

- `u ~ d_{pi_S}`：anchor 前 roll-in 是 current/near-current student on-policy。
- `z ~ pi_T`：bridge 动作和 bridge 内状态是 intervention-induced，不属于纯 student occupancy。
- minibatch 在 student 更新后会变 stale，因此只能称 iterative/approximately on-policy。
- 训练时 teacher forcing 仍有 bridge 内 exposure mismatch；周期性 refresh 和纯 student evaluation 用来检验 bridge 是否真正写回 student。

所以准确表述是 **on-policy roll-in with counterfactual corrective supervision**，而不是 fully on-policy distillation。

### Failure Modes and Diagnostics

- 干预有效但训练无效：说明 behavioral rescue 不等于 parameter writability；转为诊断结论或加入 micro-update gate。
- 最佳固定 h 与自适应相同：说明 horizon heterogeneity 不足，主方法不成立。
- 只有 teacher-complete 有效：student 无法接管，属于 capability gap，不作为 bridge 成功。
- 长 bridge 损害探索/回溯：报告 Pass@k、自纠行为和正确题 harm；优先最短有效 span。
- 数据刷新后 anchor 消失：这是期望的修复；在新 on-policy failures 上继续构造数据。

## Novelty and Elegance Argument

相对 token-selective OPD，本方法不以局部 divergence 直接定义训练价值，而以 handoff 后 frozen-student 的下游 outcome 验证整段纠偏。相对固定 teacher takeover/bridge 方法，核心差异不是“使用 chunk”，而是估计每个 prefix 的 intervention dose-response，并在等 teacher token 预算下选择最短可靠有效跨度。论文应以该测量发现和预算适配为核心，避免声称 teacher bridge 本身新颖。

## Claim-Driven Validation Sketch

### Claim 1：纠偏监督存在异质的最小时间跨度

- 在独立问题上随机化 h，使用多 teacher bridge 和多 student suffix。
- 报告 token-local、span-local、harmful、unresolved 分布、置信区间、跨 seed/模型/任务稳定性和非单调曲线。
- 若大多数 prefix 均被同一固定 h 支配，claim 失败。

### Claim 2：自适应有效 bridge 是更好的训练单位

- 等 teacher-supervised token、optimizer step 和数据题目数比较：all-token OPD、KL-top token、same-length random/unfiltered chunk、global best fixed h、adaptive h†、oracle h。
- 主指标是未见题目上纯 student greedy/Pass@k；同时测 intervention uplift 对 micro-update/完整训练 uplift 的预测能力。
- 只有 adaptive h† 在等预算下优于 global best fixed h 和 same-length unfiltered chunk，才能声称 adaptive causal selection 的训练价值。

## Compute & Timeline Estimate

- 先在小规模独立集完整运行 B=8、M=8 建立现象；训练集用 B=2、M=2 初筛后对少量样本确认。
- 第一轮训练采用静态 bridge 数据加一次 refresh；不把 fully online collection 作为必要条件。
- 主要瓶颈是多分支 student suffix generation，而不是反向传播。
