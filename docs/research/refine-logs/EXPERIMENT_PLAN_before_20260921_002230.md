# 两项实验计划：逐题策略互补与同状态长度效应

日期：2026-09-14。状态：设计完成，未启动生成、训练或远端任务。现有网格是探索性资料，新采样用于确认。

## 主张与证据

| 主张 | 最低证据 | 实验 |
|---|---|---|
| C1：短、长接管策略对题目的效果可能具有稳定差异 | 逐题交叉 + 独立种子；互补超出偶然采样，BBH差异不只来自任务构成 | E1 |
| C2：在同一状态和相同首个teacher纠正token下，后续teacher支持长度可能改变student独立完成概率 | 同prefix、嵌套teacher片段、多次teacher/student采样、后续禁止帮助；保留无效/负效应和不确定状态 | E2 |

排除的解释：不同题集/评分，首次teacher token不同，原token ban本身的效应，训练更新，未来teacher援助，答案直接代写，EOS/截断及截止政策差异。

E1研究完整策略，E2研究选定状态的局部动作。E1不提供同状态因果证据，E2不直接保证允许后续援助的完整动态策略更好。

## 公共协议

- 冻结 Qwen3-1.7B student / Qwen3-14B teacher；沿用实际模型revision、tokenizer、prompt、native non-thinking、temperature=1、top_p=1、seed派生规则和scorer版本。
- E1仍采用chunk64，signed logpS-logpT>τ，teacher首token ban，后续介入不设次数上限，最大response40264 token。E2只做指定的这一次介入。
- 固定prompt token IDs，不通过解码再编码重建prefix。保存模型/代码/scorer/prompt SHA及原始输出。
- 新实验统一token上限；不采用混合900秒截止。基础设施故障保留并重试相同逻辑请求；无法完成标缺失，不静默删除或当普通错误。最终缺失样本报告成功率上下界。若要采用wall-time截止，需在全部新分支统一预注册并另外报告。
- 长度耗尽计为模型行为并按现有评分规则评分，同时报告length/EOS比例。答案判分异常与模型答错分开记录。
- 不以同一整数seed宣称token级共同随机数配对；不同轨迹消耗的随机数不同。配对单位是问题或固定anchor。
- 当前只论teacher生成预算，记录评分、探测、丢弃token以便复现，不主张端到端效率。

## E1：逐题交叉分析

### E1-A：直接分析已有结果（MUST，无新增模型生成）

数据路径：`experiments/nojudge_grid_20260912/artifacts/` 下的 `prompts.jsonl`、配置的 `completed/`、`scores/`。原始分数来源hash须匹配输出；输入应为MATH500=500、Olympiad=674、BBH=1080。配对键使用(dataset,subset,problem_id)，dataset_index只作一致性校验。

主阈值预定τ=2.5；主比较k=1 vs8。次比较1vs4、1vs16；τ=2、3、3.5重复描述性分析，不能观察结果后更换主比较。k32及k16/τ4仅附录，因原停止规则不同。

逐题宽表字段：question key、BBH subset、每配置correct/score_status、实际teacher token、teacher次数、总输出token、finish_reason、原始文件hash。各配置都必须包含相同题集，无法配对者列清单。

对每组短a/长b，构造n11(都对)、n10(短对长错)、n01(短错长对)、n00(都错)。必须同时报原始计数和占总题量比例：

- 准确率差Δ=(n01-n10)/N；差异总量D=(n01+n10)/N。
- 相对短策略的改正比例n01/(n00+n01)、退化比例n10/(n10+n11)，分母为零时报NA。
- 探索性oracle联合成功=(n11+n10+n01)/N；相对较好单策略的增量=min(n10,n01)/N。明确这是看了答案后选策略的乐观诊断，不是可部署准确率，也不是稳定互补的证明。
- 主比较按题配对bootstrap 10000次给95%区间；主τ下3个数据集的1vs8比较用Holm控制多重检验。其他比较标exploratory。可补exact McNemar，不能仅用独立比例检验。
- 逐题列出短独有/长独有样本，但只作诊断，不直接作为E2主样本池。

### BBH子任务

使用prompts中的subset，不从题目文本猜标签。按实际清点子任务数/样本数报告。每个subset输出同样四格表、Δ及教师token/次数差。主表micro acc；另报子任务macro acc。

各子任务样本少，不按原始Δ排名即下结论。27个子任务若做发现检验，BH-FDR校正并标探索性；确认阶段冻结选中的子任务，独立seed复核。随机抽取各结果类型案例，检查是否格式/选项直接代写/局部计算/推理方向；身份和结果对标注者隐藏到可行程度，保留无法判断。

### E1-B：独立种子确认（MUST，分阶段）

先固定一个600题面板：MATH随机200、Olympiad随机200；BBH按子任务近似等额抽200并保存实际权重。不得按E1-A正确性结果抽样。抽样seed20260914；生成seed建议20260915、20260916、20260917，均为建议的新独立种子，不覆盖旧结果。

Pilot主τ=2.5，k∈{1,4,8}，3新seed：600×3×3=5400条完整rollout。旧seed20260911只作discovery，不混成三次独立确认。

每题估计各策略在seed上的平均成功率；总体Δ先对每题求种子平均差，再按题bootstrap。另报每seed Δ及方向，不把题×seed当作互不相关样本。3seed不足以可靠给每道题贴稳定偏好标签。

稳定互补加测：用discovery选固定最多100题（覆盖短独有/长独有，单独标明富集样本），对k1/k8各采16条全新rollout，最多3200条；若从pilot选择则加测seed必须独立。报告独立复测的题级概率差和不确定性，不能把富集队列结果外推为总体发生率。可同时增加少量随机题对照并预先修订预算。

正式确认可扩到全2254题，仍主τ和k1/4/8、3seed：共20286条（包含pilot；增量14886）。k16是有条件扩展：全量再加6762；τ2为稳健性扩展，避免默认把全部网格重跑。

### E1输出与判断

输出：question_outcomes.parquet/csv；pairwise_transition.csv；bbh_subset_transition.csv；seed_confirmation.csv；案例索引；四格堆叠图、子任务Δ图、准确率-实际teacher token散点。

若独有成功题在独立采样中仍显示方向一致且有实质概率差，支持题级异质性。若只有n10/n01很大而独立复测趋近0，主要是随机波动。没有总体平均提升不否定互补，存在互补也不证明可预测或同anchor长度效应。

## E2：同anchor长度分支

### E2-A：状态采样与重建（MUST）

参考策略固定τ2.5/k1。主队列从参考轨迹的实际触发事件取样，先随机抽有anchor的问题，再在该题预设分层中随机选一个事件，每题最多一个。每个数据集按首次/后续介入约1:1分层（不足保留真实计数）；记录gap和已生成位置。无需知道原trace最终对错；没有anchor的问题记录覆盖率，不强行造anchor。

这一队列的结论限定为参考策略到达的触发状态，非所有推理状态。每题抽一个估计的是题均衡状态分布，不是事件频率分布；跨数据集均衡也不等于自然任务总体。未来研究触发时机时再加低gap/random状态，本计划不把它作为必须生成项。

前缀=原prompt_token_ids + response_token_ids[:anchor_response_position]。候选x取event.student_entry_token；当前位置原teacher bridge及之后文本不能进入prefix。用prefix hash、候选ID、gap、reference seed/event index唯一标识状态。prefix允许包含此前teacher输出，但必须标首次/后续，分别报告。

### E2-B：生成动作

| 动作 | 操作 | 作用 |
|---|---|---|
| S-keep | 接受原student候选x，再由student完成 | 原候选条件下的基线 |
| S-resample | 禁x，由student重采样1 token，再由student完成 | 隔离强制离开原分支 |
| T1 | teacher禁x，生成共同片段的前1 token，student完成 | 首次纠正收益 |
| T4/T8/T16 | 使用同一个teacher片段的前4/8/16 token，student完成 | 固定首token后增加支持的收益 |

对每个anchor生成J条独立teacher continuation，每条最长16 token，仅第一个位置禁x。同一条片段截取1/4/8/16作为嵌套分支，因此同一个teacher replicate内所有k共享第一token以及重叠前缀。不同replicate允许不同第一token，便于对teacher内容随机性取平均，不挑最好片段。

每个分支再采R条独立student continuation；所有分支交还后永久关闭teacher/judge/候选重排。学生继续使用相同原始prompt模板，不能增加解释性帮助指令。剩余输出预算=40264-已生成prefix响应token数-实际本次teacher/强制token数，保证同一最大总响应长度。

S-keep条件在已采样x，不是无条件student从prefix采样的成功率。需要无条件估值时另加S-free，不能混用名称。S-resample每个replicate重新抽替代token，除第一步ban外后续不限制。

teacher提前EOS：保留这个结果，实际长度记账；所有覆盖EOS的k共享同一终止答案，不复制成独立证据。离线生成的未用teacher后缀也记录生成开销，不假称这些分支实际只付出了k token计算。一次采16再截断与线上max_tokens=k的前k token分布一致性需先通过采样参数/停止逻辑小检查，禁止max_tokens影响内容的额外提示。

### E2-C：预算与阶段

- 工程sanity：每数据集3 anchor，共9；J=1,R=2，4种teacher长度各2+两种student控制各2，共108个续写/终止记录，teacher片段9条。用于协议检查，不作统计证据。
- Pilot：90 anchor，3数据集各30且来自不同题；J=2,R=4。每anchor 4×2×4+2×4=40条，共3600条student续写/终止记录，teacher片段180条。
- 确认：独立120 anchor（各40，问题与pilot不重叠）；J=3,R=8。每anchor 4×3×8+2×8=112条，共13440条student续写/终止记录，teacher片段360条。
- 对边界机制需要更精确的少量状态，另立独立复测队列；不是观察p值后反复补采直到显著。先由pilot做功效模拟，确定正式样本量并冻结。上述规模是启动预算，不保证识别5个百分点的小效应。

### E2-D：估计、统计与标签

令状态s包含prefix及原候选x。V(s,k)是按预定teacher片段分布平均、本次接管后仅student完成的成功率。teacher长度组先对R条student平均，再对J片段平均，最后按anchor平均。

主对比：Δ8,1=V(s,8)-V(s,1)；支持性对比：V(s,1)-V(s,keep)、V(s,1)-V(s,resample)。长度曲线全部展示1/4/8/16，不能只展示最佳k。主推断按问题/anchor聚类bootstrap，跨k保留同teacher片段配对；不能将JR条后续当JR个独立anchor。可补分层模型处理任务、teacher片段和student采样层级。

候选分类：短足够、延长有益、延长有害、当前动作集无法有效恢复、不确定。绝不能将“不显著”当成等效。若需定义短足够，预设实际容忍ε（建议先以5个百分点作为规划值），要求长相对短的增益区间上界低于ε，并且T1对基线有实际收益；样本不足则保持不确定。J=3,R=8每长度只有24条，状态级小差异通常难以定性。不得用同批数据选最大V(s,k)再把它当无偏最佳长度。

### E2-E：机制角度（先小样本，MUST抽样检查，深入扩展OPTIONAL）

从全部响应类型中预先随机抽案例，标注teacher片段是否完成词/数值/公式/子目标，以及是否直接输出最终答案。尽量对标注者隐藏分支成功率及模型身份，两人独立标注后仲裁，保留无法判定。

统计成功率跃升是否出现在某个可验证中间结果完成附近。若teacher已经写出最终答案，保留主结果但单独标记；不能称student恢复后续推理。将这些内容类别作为机制描述而非按结果筛样本。

T1后student生成到第4/8/16 token处可记录局部步骤是否完成；它只是T1轨迹上的观测检查点，不是新增独立算法或随机化对照。与teacher相同位置的内容差异是探索性机制证据，最终作用仍由嵌套teacher长度实验估计。

可选：在部分状态比较固定token长度与词/表达式/步骤边界交还，按实际token匹配或绘前沿；检查长度收益是否只是语法/表达完整性。语义边界可能只能事后看到，不直接当免费在线策略。

### E2输出与判断

anchor_manifest.jsonl（问题、prefix hash、token IDs、参考轨迹及来源SHA、采样权重）；branches.jsonl（branch ID/J/R/seed、片段及实际长度、剩余预算、输出、评分、EOS/timeout/error）；anchor_effects.csv（各V、Δ和区间）；mechanism_annotations.csv。

主图：各任务平均V(k)及配对差区间；状态级Δ8,1与Δ1,keep散点（带不确定性）；若干随机/预选机制案例的嵌套片段和成功率。状态最优k直方图只能标探索性，不能用小样本argmax直接证明多峰需求。

支持C2：同状态固定首token后，长短效应在独立队列中稳定；部分状态短介入已经有效，而另一部分对额外teacher内容有可信增益；效果不由ban/最终答案/EOS独占。若全体都由T1解释，长支持需求假说不成立；若大部分均不确定，先评估功效，不能直接否定也不能强行正面结论。

## 执行顺序与预算

| 顺序 | 内容 | 新生成规模 | 决策 |
|---|---|---:|---|
| M0 | E1文件/hash/题目对齐；E2 sanity | 108个续写记录+9teacher片段 | 确认prefix/ban/EOS/scorer正确 |
| M1 | 全量E1已有数据交叉与BBH子任务 | 0 | 冻结主对照及随机面板 |
| M2 | E1独立种子pilot | 5400整轨迹 | 是否有超出偶然采样的异质性 |
| M3 | E2 pilot | 3600续写+180teacher片段 | 判断方差、状态覆盖和功效；冻结确认方案 |
| M4 | E2独立确认 | 13440续写+360teacher片段 | 同状态额外支持是否有可信作用 |
| M5可选 | E1全2254确认 | 相比pilot增14886整轨迹 | 支撑主文完整策略稳健性 |
| M6可选 | E1富集问题独立复测；k16/第二阈值；边界实验 | 分别单列，富集最多3200 | 只补改变结论的检查 |

M2和M3无数据依赖，可独立排队，但本轮仅规划。E2主样本不依赖E1的成功/失败筛选。

GPU小时与墙钟时长目前不能可靠填写：E1是完整协作且含评分，E2主要是student后续生成，二者每条成本差异很大。用sanity/pilot实测平均续写长度、student吞吐、teacher prefill/decode和有效并发，再按剩余token外推并报告尾部。预算应分别记录teacher片段/评分、student续写、评分CPU和人工标注量，不能由条数直接承诺完成天数。

## 论文布局与范围

主文：E1主阈值交叉及独立seed摘要；E2固定首token后的V(k)和作用差异。附录：完整阈值、BBH全部子任务、停止/缺失敏感性、边界及机制案例。暂不做：动态router训练、端到端speedup、将新方法与所有训练基线重跑。这两项先决定动态控制是否有可利用的空间。

## 完成检查

- [ ] E1题集/评分hash匹配，主比较与停止政策明确。
- [ ] BBH subset正确使用，微/宏平均及多重比较处理一致。
- [ ] 独立采样不以旧正确性选主队列；富集队列单独标记。
- [ ] E2相同prefix与候选，嵌套首token一致，无未来teacher泄漏。
- [ ] S-keep/S-resample含义明确，EOS与预算正确，教师未用后缀开销记录。
- [ ] 统计聚类到问题/anchor，不把后续rollout当独立问题。
- [ ] 不显著不等于短足够；不确定状态保留；独立确认不挑最优结果。
