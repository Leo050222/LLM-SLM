# 文献定位更新：短时teacher干预与student交还后的表现

检索日期：2026-09-12。定向核对论文正文、附录与官方发表记录，并对照当前17组已完成的signed-gap无judge网格。不是系统综述，也不是首创性认证；“本次未见”不代表所有文献中不存在。未运行新模型实验。

## 判断

当前结果能支持一个有价值的研究项目，但尚不足以把“分歧触发、稀疏1-token纠正、无需训练”当作独立的新方法贡献。这些元素以及diagnostic-first论文定位均已有直接先例。

较值得争取的生态位是：**在固定teacher生成预算下，研究干预粒度与后续协助依赖，区分短暂干预后student能够自主完成的状态，以及必须持续或延长teacher接管的状态。**

这是待验证的具体研究目标。仅增加长度/阈值网格，或把已有策略重命名为闭环控制，都不足以建立该贡献。需要同状态反事实、等预算策略对比、独立确认和跨模型复现。

## 一、最直接的文献覆盖

| 工作、作者与发表状态 | 已核对的方法/结果 | 对本项目的限制与可比较处 |
|---|---|---|
| Reasoning Can Be Restored by Correcting a Few Decision Tokens；Changshuo Shen et al.，ICML 2026 | CE/rKL分歧触发单token接管并反复执行；random/early-only预算对照；明确自称diagnostic-first | 高分歧→失败相关→稀疏接管，以及暂不追求端到端效率，均不是独有定位。其主算法直接从teacher采样，未见首token ban步骤。[§5、§6、附录C](https://arxiv.org/html/2605.16874v1) |
| Sparse but Critical: A Token-Level Analysis of Distributional Shifts in RLVR Fine-Tuning of LLMs；Haoming Meng et al./Qwen Pilot Team，ICLR 2026 | JS触发base/RL双向cross-sampling；限制前k次干预后交回primary；预算增加时性能渐变 | “前几次纠正能影响后续”“重复小纠正恢复性能”已有。其主要解释RLVR前后变化，不是本项目同预算跨接管跨度的状态级研究。[§3](https://arxiv.org/html/2603.22446v1) |
| Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM’s Reasoning Capability；Zicheng Lin et al.，ICML 2025 | rollout定位关键位置后，保留prefix，mask原token，由原模型重采样alternative再续写 | 强制离开原分支本身已有明显收益；student自重采样是必要对照。其位置定位使用大量未来rollout，不能直接等同于在线gap gate。[正文§2](https://arxiv.org/html/2411.19943)、[正式版](https://proceedings.mlr.press/v267/lin25j.html) |
| R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing；Tianyu Fu et al.，NeurIPS 2025 | 区分neutral/divergent；teacher续写两分支，由verifier做语义标注，再训练router | 风格与逻辑差异已有直接研究；标签的续写者是teacher，不能自动代表student接续效用。[§3及附录B](https://arxiv.org/html/2505.21600v2) |
| Select to Think: Unlocking SLM Potential with Local Sufficiency；Wenxuan Ye et al.，ICML 2026 | 同trigger与预算下比较teacher自由生成下一token和从student top-K选择；进一步蒸馏selector | “小模型有候选，只是不善选择”也已有；候选池大小K不等于teacher接管长度h。[§5及附录F](https://arxiv.org/html/2604.26940v2) |
| FutureBridge: Token Selection Beyond Local Preference in Collaborative Decoding；Quanquan Li et al.，2026 arXiv预印本 | 用student后续适用性训练候选reranker；推理追加1 token后交还；最多8次事件 | teacher偏好不等于student效用已有直接定位。H=16/32/64/128是离线监督future长度，不能当在线horizon消融。[§5与附录](https://arxiv.org/html/2608.06819v1) |
| Dense Supervision but Sparse Guidance: A Token-Level Analysis of On-Policy Self-Distillation；Taeckyung Lee et al.，ICML 2026 AI for Math Workshop | 代码任务中peak-KL常落在非代码逻辑文本；局部错误原因标签少，但单token替换后student续写可恢复部分teacher续写效果 | 不能首创性声称“看起来不重要的token也有轨迹作用”。是参数共享、privileged-context自蒸馏诊断，与大小模型冻结协作有设置差异。[官方PDF](https://openreview.net/pdf?id=IfAQFQ56Zi) |
| SPOT: Sparse Probing and Outcome Calibration for On-Policy Distillation；Zikun Qu et al.，2026 arXiv预印本 | 分配位置探测预算，用verifier-scored student continuations给teacher候选估值，再构造蒸馏目标 | “按student最终结果评价teacher指导”已有；其最终任务是训练，而当前应聚焦冻结策略的介入组织方式。[§3–4](https://arxiv.org/html/2608.04419v1) |
| Not Every Divergence Should Be Suppressed: Counterfactual Recoverability in On-Policy Distillation；De Jiang et al.，2026 arXiv预印本 | matched-budget continuation vs rollback，区分可恢复/不可恢复/不确定状态，oracle控制训练 | “分歧不等于可恢复”“反事实决定干预”也非空白。其主要比较teacher延续与回滚，不能直接替代teacher短干预后student自主续写实验。[方法与实验](https://arxiv.org/html/2608.04408v1) |
| Pass the Baton: Trajectory-Relayed On-Policy Distillation；Haolei Xu et al.，2026 arXiv预印本 | 反思词方向分歧触发；冻结推理pilot单反思token已有收益；训练扫teacher段长和次数 | 单token有效、短teacher leg、交还及block验证已有。主要L/M消融是训练结果，未见冻结策略等生成预算的长度×次数交叉设计。[§3–4](https://arxiv.org/html/2607.26057v1) |
| RelayLLM: Efficient Reasoning via Collaborative Decoding；Chengsong Huang et al.，2026 arXiv预印本 | 训练student主动请求n个teacher token；Fixed20/100/500分别重新训练；更换teacher测试协作依赖 | 可变接管长度和teacher兼容性有前史。不能用其训练消融替代同冻结student的因果长度实验。[§3、§5](https://arxiv.org/html/2601.05167) |
| What makes Reasoning Models Different? Follow the Reasoning Leader for Efficient Decoding；Ming Li et al.，2025 arXiv，有ACL workshop版本 | FoReaL：首段强制teacher，之后句级选择；teacher至少生成n token，再连续top1一致才交还 | handoff本身不是新对象；该停止规则可以作为对照。实际不是简单固定句首n-token接管。[§3–4](https://arxiv.org/html/2506.06998) |
| TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models；Yi Zhao et al.，Findings of ACL 2026 | initial planning、overload、recovery三类触发；扫priming/recovery步骤；包括BBH | A/B/C的宏观竞争假设也有前史。其reasoning step不等于token，且独立扫参不同于等总预算粒度实验。[正式版](https://aclanthology.org/2026.findings-acl.333/) |
| CITER: Collaborative Inference for Efficient Large Language Model Decoding with Token-Level Routing；Wenhao Zheng et al.，COLM 2025 | 学习token级router，目标考虑后续质量与成本，有去除长期影响的消融 | 不能声称首次用最终utility指导router。若后续增加可学习控制器，需作为相关方法。[正文](https://arxiv.org/html/2502.01976) |
| Bridging Reasoning Trajectories in On-Policy Distillation via Near-Future Guidance；Yuxuan Jiang、Francis Ferraro，2026 arXiv预印本 | TOPD用未来trajectory差异过滤局部false alarm，再做多token训练指导 | “单token不足”在这里是参数更新机制，不是冻结模型token替换；不能把两者写成互相矛盾。[§2、§5](https://arxiv.org/html/2606.00305) |

Dense Supervision的官方PDF直开遇到browser challenge，本次使用搜索工具返回的官方PDF正文/首页，覆盖限于可读部分；未声称核对全部附录。其他项按各自正文/附录记录范围判断。arXiv与自动交叉验证helper此前在canonical路径下未解析到，本次以官方网页核对；没有冒充自动citation audit通过。

## 二、文献版图的含义

第一组研究已经回答了稀疏模型分歧是否具有功能意义。尤其Few Decision Tokens和Sparse but Critical直接说明，高分歧处的少量token交换可以显著改变结果。因此，新论文如果只展示相同现象、更换模型对、再扫一些阈值，很容易被视为已有发现的扩展验证。

第二组研究已经细化了何时和如何选择token。R2R讨论语义分岔，S2T讨论本地候选覆盖，FutureBridge/CITER讨论student后续效用。不能把“局部teacher偏好不够”当成领域尚未意识到的问题；这应成为新工作的已知出发点。

第三组研究把干预与训练、recoverability或handoff联系起来。Relay-OPD、RelayLLM和FoReaL已经存在长度与交还设计；SPOT和反事实recoverability已经使用结果探测。因此，单独加入反事实实验是提高证据质量，不自动构成新方法创新。

本次尚未定位到一套直接覆盖下述完整设计的研究：**同一批预先采样的student状态，同冻结模型，同teacher生成总预算，联合比较连续接管跨度与分散纠正，并随机关闭未来帮助以测量交还后的独立成功。** 这是较有希望的研究对象，但目前只能称为有依据的候选空间。

## 三、与当前结果最贴合的候选定位

工作问题可以写为：

> 短时teacher干预何时足以支持student独立续推，何时仍需持续协助？有限teacher生成预算应该分散到多个状态，还是集中用于完成局部推理？

现有[17组网格](/Users/leolee/Documents/OPD/experiments/nojudge_grid_20260912/reports/combined_completed_metrics.md)给出了动机：正向阈值2时，k1/k8在MATH500为83.8/86.0，在OlympiadBench为44.36/45.70，在BBH为76.85/76.76；但k1每题仍介入约9.71/17.94/18.61次。

这些结果说明“单次很短”和“整条轨迹不依赖teacher”必须分开。目前的高accuracy可能来自许多关键状态的持续协作，不能直接解读为一次纠正激活了小模型的完整推理能力。

建议以三个可证伪问题构成论文，而不是预先写成结论：

1. **粒度：** 相同teacher生成预算下，多次1-token是否优于少次长span？是否只在一部分任务/状态上成立？
2. **持续性：** 完成某次短干预后关闭后续teacher，收益保留多少？在哪些状态上延长当前bridge能提高之后的独立成功？
3. **解释：** 这种差异能否由student对新prefix的接续能力解释，而不是单纯gap大小、语义转折标签或强制重采样？

这是对现有研究的具体延伸，不宜宣称“首次提出闭环纠正”“首次关注交还”或“首次考虑student能力”。

## 四、最小证据包

### A. 固定预算的策略实验

用开发集预先选定的信号，在teacher生成预算8/16/32等档位，比较每次1/4/8 token；预算耗尽后student独立完成。报告实际预算、EOS缩短、accuracy和伤害率。还需同gate的无ban与student重采样对照，不能让强制改写的影响混入粒度结论。

### B. 同状态的停止协助实验

在干预前抽取状态，随机分支到：student继续；teacher1后student独立继续；teacher8后student独立继续；teacher1后允许后续预算内再次介入。跨分支保持后续生成上限等条件一致，使用多个续写seed。对于完整teacher continuation只作为能力参照，不把它当数学意义上的上界。

局部长度实验和整条轨迹策略实验回答不同问题，应分别报告。不要先根据已经观测到的高lift筛状态，再用相同续写评价效应；旧bridge集合可做pilot，确认需要独立续写。

### C. 接续行为和答案直接代写检查

分开统计teacher是否直接写出了最终答案或关键选项，以及student是否实际完成了剩余推理。BBH尤其需要检查第一个token干预、答案字段覆盖和具体子任务。主结果保留全部题；机制表单独区分这些类型，避免通过删掉不利样本制造自治结论。

语义judge只作为辅助标签来源。可抽样比较judge接受/拒绝的实际效用，但“表面不重要却有用”已有文献，不能独自承担新颖性。

### D. 独立确认和泛化

现有网格是探索性证据，不能从test挑最优点后宣称普遍规律。冻结少量代表配置，在新seed、第二模型组合上复现；新增模型最好能改变student能力，而不只是同一模型重复采样。原DeepMath关联还需统一答案评分后重算。

## 五、基线优先级和写作边界

冻结模型诊断的最低直接对照：Few Decision Tokens的CE/rKL gate、Sparse but Critical的JS gate、student entropy、匹配随机/句首；同gate下比较teacher自由替换、ban替换和student自重采样。机制复现可以明确写成统一框架下的适配版本，不能冒充原论文完整重现。

如果文章进一步主张“新router优于已有方法”，则需加入R2R/CITER/RelayLLM等适合的训练基线及训练预算；如果主张“候选选择更好”，则需对照S2T/FutureBridge。诊断论文无需为了列表完整而重训所有模型，但必须直接对照最接近的可复现机制。

现阶段可以不做端到端速度主张，使用“teacher生成预算”描述控制资源，同时明确teacher仍对草稿评分。应记录开销以便复现，但不必强行证明系统speedup。不能把暂不研究效率当作与Few Decision Tokens的差异，因为它同样明确采取diagnostic-first定位。

暂定标题可用 **When Is One Token Enough? Intervention Granularity and Post-Handoff Reasoning in Small Language Models**。问句代表待回答的问题，不预先宣称1-token普遍充分。

## 六、结论强度与决策

| 拟主张 | 当前状态 | 决定是否成立的证据 |
|---|---|---|
| 多次1-token协作能明显改善小模型 | 已有项目结果，也有直接前史 | 新seed确认；不作为唯一贡献 |
| 固定预算下分散纠正更有效 | 尚未证明 | 等预算粒度比较，允许存在任务反例 |
| 短干预后student获得持久独立接续收益 | 尚未证明，当前允许反复帮助 | 随机停止未来帮助、同状态多分支 |
| 存在需要更长teacher span的状态 | 合理假说 | 同状态长度曲线与独立确认 |
| 可以依据状态选择接管长度 | 可选后续方法 | 离线标签不能泄漏到测试；在线预测与固定策略对照 |
| 整体更快、更省算力 | 当前不提出 | 另需完整端到端实验 |

如果这些机制实验只复现“多纠正几次通常更好”，贡献会比较薄；如果能识别短纠正充分与持续辅助必要的边界，并证明等预算策略可以利用该差异，才更接近一篇独立的机制/方法论文。
