# Research Proposal: Outcome-Calibrated Recovery-Span Distillation

## Problem Anchor

现有 OPD 在 student rollout 的状态上提供 teacher 的逐 token 分布监督，但局部 teacher–student 差异未必是足以纠正未来推理轨迹的训练单位。一个 failure prefix 可能需要连续若干 token 的纠偏状态转换，student 才能在 teacher 退出后保持收益。

## Method Thesis

不同 failure prefix 具有不同的 outcome-effective recovery span。方法从当前 student rollout 中选择固定规则定义的 anchor，通过多长度 teacher-to-student handoff 估计 response curve，再蒸馏独立采样的、处于所选 recovery span 的 teacher bridge。核心比较是在等 teacher-supervised token 预算下，相对 token-local supervision 和 global fixed bridge 是否更有效。

## Technical Positioning

该方法是 **on-policy student roll-in with counterfactual corrective supervision**：

- bridge 入口 `s` 来自当前 student 的 occupancy；
- teacher 第一个 token 之后的 bridge states 是 intervention-induced，而非 student on-policy states；
- 每轮更新后重新采集 student failure prefixes 时，可称 iterative on-policy roll-in；
- 不能称 fully on-policy distillation。

它可以作为 OPD extension，但论文标题和主方法名不依赖 OPD 标签。

## Intervention Estimand

对 student prefix `s` 和候选跨度 `h`：

`Delta_h(s) = E_{z~pi_T}[P_{y~pi_S}(R(s,z_1:h,y)=1)] - P_{y~pi_S}(R(s,y)=1)`。

该量对 teacher policy 的 bridge 随机性取平均。训练时不能直接使用因偶然成功而被选中的同一条 bridge；应使用独立 teacher samples 或 cross-fitting，避免 lucky-bridge selection bias。

## Minimal Method

1. 当前 student 对训练题 rollout，只处理其 failure prefixes；另保留正确题作为 harm control 和 replay。
2. 使用预先冻结的 anchor rule；第一版不训练 selector。
3. 同一 teacher trajectory 使用 nested horizons `H={16,64,256}`，由 frozen student 接手并多次 rollout。
4. 区分 oracle diagnostic horizon、cross-fitted adaptive horizon 和 cheap training horizon。
5. 排除 teacher-complete/near-complete 样本，确保 student 在 handoff 后仍需实质推理。
6. anchor 前的 student prefix 只作条件；对独立 teacher bridge 计算长度归一化 KL/CE：

   `L_bridge = (1/h) sum_j D(pi_S(.|x,u,z_<j), pi_T(.|x,u,z_<j))`。

7. 第一版不加入 handoff suffix loss；正确数据 replay 是所有系统统一使用的稳定措施，而非方法贡献。
8. 更新 student 后刷新 failure prefix，使 roll-in 接近当前策略。

## Claim Boundaries

“teacher bridge 能救回 frozen student”只证明 external intervention utility，不自动证明 parameter writability。训练方法必须先通过：

1. **Horizon heterogeneity**：oracle/cross-fitted adaptive span 在等预算下优于 global best fixed span；
2. **Scalability**：cheap horizon estimator 能保留 oracle advantage 的实质比例；
3. **Writability**：训练后 student 在原训练 prefix 上无需强制 teacher token即可重新进入正确轨迹；
4. **Generalization**：在完全未参与更新的问题上改善 pure-student performance。

若只满足第一项，工作应转为 recovery-span measurement paper；若第一项不成立，固定 bridge 已足够；若 intervention utility 不可写回，则不应声称 recovery span 是有效训练单位。

## Kill Experiments

### Kill 1: Recovery-horizon value

- 200–300 个失败 prefix；`B=8,M=8`；三个预注册跨度；bridge-cluster-aware uncertainty。
- 比较 global fixed、oracle、cross-fitted adaptive，以及不同小 `B,M` 子采样下的 cheap adaptive。
- 主结果为 teacher-token budget–utility frontier。
- 若 oracle adaptive 无实质预算优势，终止训练方法。

### Kill 2: Intervention-to-update transfer

构造 effective、same-length ineffective、same-length unfiltered 三个匹配训练集，从同一 checkpoint 训练独立 clones。匹配题目数、监督 token、更新步数、anchor位置和 replay。

- 在训练 prefix 上 fresh free-run 测 writability；
- 在 disjoint problems 上 pure-student rollout 测 generalization。

只有 effective group 同时优于两个 matched controls，才支持训练 claim。

## Primary Claim

> Failure prefixes exhibit heterogeneous outcome-effective recovery spans; when those spans are estimable under a practical budget and their intervention value transfers to parameter updates, recovery-span supervision outperforms token-local and globally fixed-span distillation under matched teacher-token budgets.

该 claim 在 kill experiments 得到结果之前只是可证伪假设。
