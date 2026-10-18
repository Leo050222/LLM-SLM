# 随机位置 teacher8、最多8次介入实验

每题单次non-thinking，Qwen3-1.7B student/Qwen3-14B teacher，temperature1/top_p1/seed20260911。2314题全部完成，无丢弃，评分2314条全部ok。

| 数据集 | Student直接 | Anchor在线8 | Anchor在线32 | 随机8/最多8次 | 随机相对student |
|---|---:|---:|---:|---:|---:|
| MATH500 | 72.80% | 81.40% | — | 58.00%（290/500） | -14.80个百分点 |
| OlympiadBench | 29.97% | 34.42% | 35.91% | 18.84%（127/674） | -11.13个百分点 |
| AIME2024 | 16.67% | 10.00% | 13.33% | 10.00%（3/30） | -6.67个百分点 |
| AIME2025 | 13.33% | 23.33% | 10.00% | 16.67%（5/30） | +3.33个百分点 |
| BBH | 35.00% | 44.81% | 52.59% | 35.65%（385/1080） | +0.65个百分点 |

结果：随机介入在MATH500和OlympiadBench明显低于student直接生成；BBH接近student；AIME各30题波动较大。上述所有数据集上随机组均未超过原anchor在线8组（AIME2024相同）。这是当前协议单次采样的观察，不能单独归因为anchor选择质量。

## 随机协议与比较限制

1. 沿用同一批题目和prompt token。每题按已有student-only trace长度，在token位置中均匀抽取最多8个相距至少8token的起点（等概率抽取所有可行位置集合），固定seed可复现。只使用参考长度，不使用参考答案或judge分数。
2. 新rollout达到起点时交给teacher生成最多8token，再由student续写；EOS或总长度/上下文上限结束。提前结束可能少于8次，长于参考轨迹的尾部不会新增介入。
3. 保留原在线算法的首token排除规则：先采样student在该位置的token，再禁止teacher首token复用它。该对照是“随机位置＋强制首token不同”，不是无约束随机teacher续写。对随机的正常位置，这条规则也可能干扰本来正确的输出。
4. 原anchor组不限介入次数，本随机组上限8；实际介入次数、teacher token成本和student生成分段也不同。因此不是只改变anchor位置、其它预算完全匹配的严格消融。参考student长度也是离线信息。
5. Anchor32此前有2题按用户要求中止并计错，本次没有中止题。MATH500未跑anchor32或teacher直接对照，不补造缺失结果。

## 实际用量

| 数据集 | 平均介入次数 | 平均teacher token | 平均总输出token |
|---|---:|---:|---:|
| MATH500 | 7.39 | 59.05 | 1032.8 |
| OlympiadBench | 7.02 | 56.07 | 1537.0 |
| AIME2024 | 6.73 | 53.87 | 2596.7 |
| AIME2025 | 6.70 | 53.60 | 1979.0 |
| BBH | 6.06 | 47.89 | 398.3 |

GPU102477：2节点16H800，13分42秒，正常完成；CPU102478：64CPU/48进程，9秒，正常完成。全部生成结束后才评分，资源已释放。BBH2题和MATH5001题触发既定长度上限，原样纳入评分，其余自然结束。

远程audit.json核验2314题基线来源hash、提示、随机位置重现、最多8次、每次最多8token、首token排除规则、judge_calls=0、评分源hash和先生成后评分时序。原始输出/逐题评分/BBH子任务汇总见artifacts。建议下一步匹配每题介入次数/teacher预算，并另设去掉首token排除的随机组，以分离随机位置、强制改写和预算差异的影响。


## BBH子任务（随机组）

| 子任务 | 正确/40 | 正确率 |
|---|---:|---:|
| boolean_expressions | 6/40 | 15.0% |
| causal_judgement | 7/40 | 17.5% |
| date_understanding | 27/40 | 67.5% |
| disambiguation_qa | 5/40 | 12.5% |
| dyck_languages | 0/40 | 0.0% |
| formal_fallacies | 3/40 | 7.5% |
| geometric_shapes | 20/40 | 50.0% |
| hyperbaton | 27/40 | 67.5% |
| logical_deduction_five_objects | 24/40 | 60.0% |
| logical_deduction_seven_objects | 19/40 | 47.5% |
| logical_deduction_three_objects | 30/40 | 75.0% |
| movie_recommendation | 6/40 | 15.0% |
| multistep_arithmetic_two | 2/40 | 5.0% |
| navigate | 17/40 | 42.5% |
| object_counting | 16/40 | 40.0% |
| penguins_in_a_table | 25/40 | 62.5% |
| reasoning_about_colored_objects | 20/40 | 50.0% |
| ruin_names | 14/40 | 35.0% |
| salient_translation_error_detection | 12/40 | 30.0% |
| snarks | 25/40 | 62.5% |
| sports_understanding | 2/40 | 5.0% |
| temporal_sequences | 16/40 | 40.0% |
| tracking_shuffled_objects_five_objects | 20/40 | 50.0% |
| tracking_shuffled_objects_seven_objects | 18/40 | 45.0% |
| tracking_shuffled_objects_three_objects | 20/40 | 50.0% |
| web_of_lies | 2/40 | 5.0% |
| word_sorting | 2/40 | 5.0% |
