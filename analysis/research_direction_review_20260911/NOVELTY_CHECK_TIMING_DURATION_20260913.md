# 强模型介入时机与动态支持长度：查新报告

检索日期：2026-09-13。范围包括 2024–2026 先例与 2026-03-13 至检索日的定向检索。网页检索之外使用了 SciSpace；Hugging Face paper_search 返回工具不存在。结论基于核对的原始论文，不代表穷尽检索或首创性认证。

## Proposed Method

冻结强弱模型，在 student 生成轨迹中决定是否由 teacher 接管，并选择支持长度及交还时刻。用同状态分支的 student 最终成功率估计不同介入动作的价值，在有限 teacher 生成预算下优化策略。无 Luna，无端到端效率主张。

## Core Claims

| 主张 | 新颖性 | 最近先例及边界 |
|---|---|---|
| 分布分歧或不确定性识别介入时机 | LOW | R2R、Few Decision Tokens、R-Stitch、GlimpRouter 已覆盖广义问题 |
| 联合学习何时调用与调用多少 token | LOW | RelayLLM 直接学习调用时机及长度；冻结模型是设置差异，不能单独构成贡献 |
| 基于 student 状态动态交还 | LOW（简单规则）；MEDIUM（未验证的具体新发现） | FoReaL 连续一致性；SFD 的 student NLL/entropy readiness；Tandem 的 handoff robustness；FutureBridge 的接续适用性 |
| 同状态识别介入与延长接管的效应差异，并据此改进预算分配 | MEDIUM，待证 | 组成部分都有先例；尚未在本轮已核对资料中找到完全相同的实证证据链。不足以声称首次 |

## Closest Prior Work

| Paper | Year / Venue | Overlap | Key Difference |
|---|---|---|---|
| [RelayLLM](https://arxiv.org/html/2601.05167)，Chengsong Huang et al. | 2026，arXiv | student 通过调用命令决定时机和 token 数；有动态与固定长度比较 | 训练 student；当前可研究冻结模型同状态的长度作用，不应把冻结本身当创新 |
| [FoReaL](https://arxiv.org/html/2506.06998)，Ming Li et al. | 2025，按 arXiv 版核对 | 句级随机选择介入，最短 teacher span 后连续 top1 一致才交还 | 一致性启发式；可检验其是否真实预测额外支持的收益 |
| [R-Stitch](https://arxiv.org/html/2507.17307)，Zhuokun Chen et al. | 2025；2026-02 v6，arXiv | student 高熵触发；teacher 低熵交还；另有 RL router | 已有双向动态控制；不能声称首次分开进入与退出信号 |
| [RelayGen](https://arxiv.org/html/2602.06454)，Jiwon Song et al. | 2026，arXiv | 无训练、按话语线索切换推理片段 | 片段分配，不是当前同状态 teacher 短 span 响应研究 |
| [Tandem Training for Language Models](https://aclanthology.org/2026.eacl-long.386/)，Robert West et al. | EACL 2026 | 用弱伙伴接手续推成功定义 handoff robustness | 训练强模型、随机交还；不是冻结双方学习有限支持预算的控制器 |
| [Making Complex Reasoning Student-Friendly: A Hybrid LLM-to-SLM Distillation Framework](https://openreview.net/pdf?id=6jP5PDOmqN)，Yongjin Yang et al. | ICLR 2026 Scaling Post-training Workshop | SFD 根据平滑后的 student NLL 或 entropy 判定 readiness，动态截取 teacher prefix 后由 student 完成 | 用于构造蒸馏数据，主要从 teacher 初始前缀交还；其 readiness 已直接覆盖简单接续能力规则 |
| [FutureBridge](https://arxiv.org/html/2608.06819v1)，Quanquan Li et al. | 2026-08，arXiv | 用 student 后续适用性监督 token 选择 | 在线单 token；扩展动作到 span 本身可能只是自然延伸 |
| [R2R](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b39cef2ef90591cffdc9c674cd55bebe-Abstract-Conference.html)，Tianyu Fu et al. | NeurIPS 2025 | 语义分岔标签和 token 路由 | 当前要识别额外支持收益，而非只判语义分歧 |
| [Few Decision Tokens](https://arxiv.org/html/2605.16874v1)，Changshuo Shen et al. | ICML 2026 | 分歧触发稀疏纠正，机制分析定位 | 单 token 接管；无效率主张也不能与之区分 |

其他已核对/沿用前轮原文核对的邻近工作：SplitReason、CITER、Select to Think、Sparse but Critical、Critical Tokens Matter、SPOT、Counterfactual Recoverability in OPD、Pass the Baton、TrigReason、TOPD。它们分别覆盖学习分段卸载、长期价值路由、候选充分性、限制纠正次数、同 prefix 自重采样、结果探测及训练时干预。详见同目录 LITERATURE_POSITIONING_20260912.md。

SFD 直接 PDF/论坛遇到浏览器校验；本轮核对官方 PDF 的检索正文（摘要、§3.1）及作者主页，未读取全部附录。R-Stitch 的 OpenReview 检索记录显示 submitted，未据此宣称录用。

## Overall Novelty Assessment

- Recommendation: **PROCEED WITH CAUTION**。
- 宽泛“动态时机+长度”方法主张：约 **3/10**，主观判断，不是统计量。
- 缩窄后的“支持的边际价值及其可预测性”：具有中等潜力，当前未形成已验证的新发现，不能提前赋予高新颖性。
- 最大审稿风险：RelayLLM 已解决 when/how long；FoReaL、R-Stitch、SFD 已有交还规则；Tandem/FutureBridge 已考虑接续成功。不能靠把这些概念组合、重新命名或换模型形成首创主张。

## Suggested Positioning

建议问题：**在什么状态下介入能够改善最终结果，以及继续由强模型生成何时仍有额外价值？如何把有限支持预算留给真正受益的状态？**

区分三个量：失败风险 P(student失败|s)、介入收益 V(s,k)-V(s,0)、延长收益 V(s,k2)-V(s,k1)。这里 V 是固定后续策略下的最终成功概率，必须明确是否禁止未来 teacher。高风险不意味着可救，高介入收益也不意味着应延长。这些逻辑区别本身不是理论创新；需要证明当前常用信号在这些决策上的具体失效及替代信号的泛化效果。

当前 25 组网格可作动机：收益非单调、任务间不同。但轨迹在不同 k 下分叉、种子不足、长 k 停止政策不同，不能直接识别状态级效应或过度介入伤害。

## 决定是否继续投入的三个实验

### 1. 同状态的时机×长度响应

除既有 anchors 外，要纳入阈值附近和低于阈值的随机位置，否则只能学习已选位置的长度，无法检验时机。按问题分开开发/验证/测试。

动作包括保持 student、student 禁原 token 重采样、teacher 1/4/8/16 token 后 student 独立完成。使用嵌套 teacher continuation 前缀并重复 teacher/student 采样。然后增加“现在介入 vs 先接受若干 student token 再介入”的随机分支，对比完整延迟策略；延迟后的状态不同是时机效应的一部分，不能伪称固定了后续 prefix。

先验证收益异质性是否超出采样噪声，再问 gap/entropy/一致性是否足以预测最合适动作。把测试未来续写当在线特征属于泄漏。

### 2. 接管中的交还实验

在 teacher 已生成到某 prefix 时，随机分为立即交还或继续 1/4/8 token 再交还。估计 student 成功率增量；比较 FoReaL 一致性、R-Stitch 熵、SFD readiness 风格规则与预测增量的策略。

不要预设收益随长度单调或递减。短期没有增益不表示跨过一个完整计算/表达单元后仍无增益。因此逐步停止策略需测试多个继续跨度，不能只用下一个 token 的即时价值决定终止。

### 3. 固定预算闭环验证

四格消融：固定/简单触发×固定 k；动态触发×固定 k；固定/简单触发×动态 k；动态触发×动态 k。补匹配动作频率/长度分布的随机策略及调优固定基线。同生成预算并报告实际 teacher token 与调用次数；统一停止政策。

无效率主张时可优先做强机制对照。若声称新方法优于已有路由，则需适当比较 RelayLLM、R-Stitch 等完整算法；启发式适配不得标成原论文完整复现。

继续投入条件：效应差异跨重复存在、可从当前状态预测、闭环留出表现优于固定和已有简单交还规则、能迁移到另一模型对。若只有随机波动或简单阈值已解释全部收益，则收缩为诊断发现；不应继续包装为新控制方法。

## Tracing

同模型家族独立代理复核：gpt-5.6-sol，xhigh，结论为 provisional，不代表外部专家共识。完整记录位于 .aris/traces/novelty-check/2026-09-13_run01/；未运行新生成实验。

独立复核结论：当前方法约 3.5/10，有条件继续，与主报告方向一致。审阅者特别指出“FutureBridge 的价值思想 + RelayLLM 的长度控制”可能被视为自然组合；贡献必须落在新的可重复规律及其决策价值。其建议的数值 go/no-go 门槛仅供项目规划，不是领域公认标准。探索性 oracle 上界需通过独立续写估值与评估，避免逐状态挑最大采样均值的选择偏差；正式策略用留出集闭环结果判断。
