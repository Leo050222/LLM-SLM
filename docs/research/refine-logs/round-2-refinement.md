# Round 2 Refinement

## Problem Anchor

- **Bottom-line problem**：现有 OPD 在 student 自己访问到的状态上提供 teacher 的逐 token 分布监督，但把局部 teacher–student 差异默认当成足以纠正未来轨迹的训练单位；对于已经进入错误推理分支的 prefix，一个 token 的局部更新可能无法将 student 带回可解轨迹。
- **Must-solve bottleneck**：区分“局部 logits 不同”和“需要跨越若干连续状态才能产生可保持的下游收益”，并找到最短且足够的纠偏监督跨度。
- **Non-goals**：不把高 KL 本身当作因果重要性；不追求让 teacher 在推理时长期接管；不把新 loss、anchor selector 和 judge 同时包装成并列创新。
- **Constraints**：约一万道训练题；昂贵的 B=8、M=8 仅用于确认或高价值子集；训练后推理必须仅使用 student；与基线匹配 teacher-supervised token 和训练预算。
- **Success condition**：在未见题目、纯 student 推理下，按 prefix 自适应选择的有效 bridge 监督在等 teacher token 预算下优于 token-only、随机/未筛选 chunk 和全局最优固定长度；干预阶段估计的有效性还能预测真实参数更新后的收益。

## Anchor Check

- 原问题保持不变；本轮只冻结 operational definitions。
- oracle horizon、cheap adaptive horizon、writability 与 generalization 被明确分开。

## Simplicity Check

- 唯一自变量仍是 recovery span `h`。
- 不新增 learned selector、judge、loss 或 controller。
- cross-fitting 只用于避免同一 teacher bridge 同时参与选择与训练/评价。

## Revised Proposal

### 技术定位

**Outcome-Calibrated Recovery-Span Distillation (OCRD)** 是一个采用 on-policy student roll-in 与 counterfactual corrective supervision 的 OPD extension。它不声称 bridge 内 occupancy 是 on-policy；只有反复从当前 student 收集 anchor 时，才能说 iterative on-policy roll-in。

### 冻结的 pilot 定义

- Anchor：在错误 student rollout 上计算 token-level teacher–student divergence；取第一个超过该轨迹 divergence 第90百分位的局部峰值，并吸附到其前一个完整句子/推理步边界。阈值和吸附规则不看 outcome。
- Horizon grid：`H={16,64,256}`；正文称 smallest tested outcome-effective span。
- Baseline：同一 anchor 下 student-only continuation 16次；所有 arm 的总回答 token budget相同。
- Diagnostic sampling：每个 prefix `B=8` teacher bridges，每个 `(bridge,h)` 有 `M=8` frozen-student suffixes；teacher bridge 是 cluster。
- Success：可程序验证任务使用 exact verifier；开放任务不进入第一轮 kill experiment。
- Exclusion：teacher bridge 在 handoff 前已给出最终答案/EOS，或留下少于预设 `r_min` 个 student 生成的实质 reasoning tokens，则标记 teacher-complete/near-complete，不计 bridge-and-handoff 成功。
- Effect threshold：pilot 使用预注册的最小绝对成功率提升 `delta=0.10`；同时报告完整 response curve，不用阈值掩盖非单调性。

### 三种 horizon

1. **Oracle diagnostic horizon**：使用完整 cross-fitted outcome data，在测试网格中选择预算惩罚效用最大的 `h`，只作为上界。
2. **Cross-fitted adaptive horizon**：一半 teacher bridges 选择 `h`，另一半独立估计效果并提供训练 bridge，交换 folds；这是 kill experiment 的可验证 adaptive rule。
3. **Cheap training horizon**：从完整数据子采样更小 `B,M`，按固定经验规则选择；其价值用相对 oracle 的 regret 和相对 global fixed span 的 budget advantage 衡量。

不得用 oracle gain 代替 cheap adaptive method gain。

### Kill 1：recovery-horizon value

- 200–300个失败 prefix。
- 比较 global fixed span、oracle diagnostic span、cross-fitted adaptive span，以及不同 `B,M` 子采样下的 cheap span。
- 主图是 teacher-token budget–intervention-utility frontier。
- Stop：若 oracle adaptive 在等预算下不能稳定优于 global fixed span，终止训练方法；若 oracle 有优势但 cheap selector 无法保留实质比例，则转为 measurement paper。

### Kill 2：intervention-to-update transfer

构造三个按问题数、bridge长度、监督 token 和 anchor位置匹配的训练集：effective、ineffective、unfiltered。每组从同一 checkpoint 训练独立 clones，使用相同 KL/CE objective、更新步数和 replay。

- **Writability**：在训练 prefix 上做全新 free-running student rollout，检查是否能在没有强制 teacher token 时重新进入正确轨迹。
- **Generalization**：在完全 disjoint 的问题上进行 pure-student evaluation。

只有 effective 组同时在 writability 与 held-out generalization 上优于 matched controls，才支持 bridge 是更好的训练单位。

### 唯一主 claim

> Failure prefixes exhibit heterogeneous outcome-effective recovery spans; when those spans are estimable under a practical budget and their intervention value transfers to parameter updates, recovery-span supervision outperforms token-local and globally fixed-span distillation under matched teacher-token budgets.

这是待实证验证的 claim，不是当前已成立结论。
