# 采样、训练与介入协议补充核查

2026-09-18，基于论文原文。未进行外部方法复现实验。

## Decision Tokens

Appendix A.1明确：每题8次独立采样，temperature=0.6、top_p=0.95、上限16384 token。Accuracy是8次正确性的均值；Pass@8是8次至少一次成功。前次汇报的66.8/81.5等是Accuracy，非Pass@8。

Table3包含Qwen3-8B Thinking直接生成：MATH500 97.4/100.0、Olympiad60.0/69.0、AIME24 70.0/83.3、AIME25 52.5/73.3、六任务均值78.1/87.3（Accuracy/Pass@8）。1.7B高介入组平均62.1/83.8，仍低于8B；其Recovery=112%以同尺寸1.7B Thinking的Pass@8为参照，不代表超过8B teacher。

两预算采用校准tail ratio r=0.03/0.05。在500道GSM8K训练题上收集分歧，τ=Q_(1-r)，λ=E[s|s>τ]/E[s]。线上要求s>τ且s>λ×前64token局部均值。二者每次都只接管1token；改变的是门控频率。校准尾比例不是线上实际替换比例，主表1.7B组实际约6%/16%。

来源：https://arxiv.org/html/2605.16874v1

## RelayLLM

训练对象是student LM的生成策略，包括生成推理内容、求助命令和长度。teacher作为外部生成器；不是只训练一个外置分类器。先用student自采样序列随机插入<call>n</call>做SFT warm-up，再GRPO。训练group size=8，不能与测试采样数混淆。

Simple reward=I(correct)−ρ。Difficulty-aware按训练采样组分情况：存在不求助正确回答时奖励独立正确1.5、求助正确1−ρ、错误0；只有求助回答做对时，不求助罚−1、求助正确1−ρ；整组全错时，求助轨迹奖励ρ以促进探索。后者不是统一更严的费用惩罚，而是依题目状态调整求助激励。

测试：AIME24/25 avg@32，其余主表任务单次greedy。正文/附录表主要报强生成token比例，未找到平均调用次数/题的明确统计。由于请求长度n可变，不能从0.43%/1.07%反推调用次数。Fixed20/100/500是另行重训变体。

来源：https://arxiv.org/html/2601.05167v1

## GlimpRouter

§4.2明确将超过teacher归因于self-correction：高首token熵可能暴露前文逻辑不一致，切入teacher时teacher会重新评估并纠正前文。AppendixF案例：student把四次转向写成四段，teacher修为五段，然后student继续。

这是作者机制解释加案例，不能独立证明相较teacher从头生成更强的因果原因。Table2对照首token熵/整步熵/整步perplexity，AIME25准确率51.67/46.67/47.50支持路由信号有用，但不是隔离自纠机制的消融。论文4次运行均值不能当已完成显著性检验。

来源：https://arxiv.org/html/2601.05110v1

## MentorCollab-Free

随机Bernoulli探测ρ=25%，实际在空白边界比较模型输出词；两个建议不同才让双方生成L=16候选片段；将A/B候选匿名提供给student，用prompt让student选择；若选择mentor则采用其片段，然后回student。探测次数、分歧次数和实际强片段采纳次数不同。

官方MATH test split，4-shot、512-token cap、greedy；1.7B+14B 40.0→43.2，14B直接仅1.8（论文讨论短预算截断）。本地MATH500 student71.6→81.2，teacher84.6的实验条件显著不同，不能称直接碾压。比较需要统一题集、prompt、thinking、预算、评分和teacher总开销。我们无需student A/B验证和双分支lookahead，但需要teacher草稿评分，不能未测就断言更快。

来源：https://arxiv.org/html/2602.05307v3
