# 仅阈值触发teacher8（无Luna judge）结果

2314题，每题单次non-thinking；Student Qwen3-1.7B，teacher Qwen3-14B，temperature1/top_p1，seed20260911。全部题完成，没有舍弃。

| 数据集 | Student直接 | 随机8/最多8次 | 阈值+Luna8 | 阈值+Luna32 | 仅阈值8 | 无judge相对judge8 |
|---|---:|---:|---:|---:|---:|---:|
| MATH500 | 72.80% | 58.00% | 81.40% | — | 84.20%（421/500） | +2.80个百分点 |
| OlympiadBench | 29.97% | 18.84% | 34.42% | 35.91% | 46.14%（311/674） | +11.72个百分点 |
| AIME2024 | 16.67% | 10.00% | 10.00% | 13.33% | 10.00%（3/30） | +0.00个百分点 |
| AIME2025 | 13.33% | 16.67% | 23.33% | 10.00% | 20.00%（6/30） | -3.33个百分点 |
| BBH | 35.00% | 35.65% | 44.81% | 52.59% | 75.74%（818/1080） | +30.93个百分点 |

无judge组在MATH500、OlympiadBench和BBH分别比judge8多对14、79、334题；AIME2024相同，AIME2025少对1题。单次采样结果，AIME各30题，不宣称多seed稳定性。Olympiad无judge311/674、teacher直接314/674；BBH无judge818/1080、teacher直接835/1080。

## 严格协议

复用原64-token student草稿，按token时间顺序选择第一个满足abs(student logprob − teacher logprob)>2的位置，直接交给teacher生成最多8token，然后student继续。不限介入次数，保留首token不得与student原token相同的规则。只有外部Luna判断筛选被移除；没有读取judge密钥、实例化网络judge或执行judge调用。每个事件judgments为空、judge_calls=0，候选严格gap>2且candidate_index=0已逐条核验。

## teacher用量

| 数据集 | Judge8平均介入次数 | 无judge平均次数 | Judge8平均teacher token | 无judge平均teacher token | 无judge平均总输出token |
|---|---:|---:|---:|---:|---:|
| MATH500 | 2.25 | 9.25 | 17.99 | 73.85 | 862.5 |
| OlympiadBench | 4.54 | 16.10 | 36.29 | 128.03 | 1489.4 |
| AIME2024 | 8.17 | 27.67 | 65.33 | 221.10 | 2468.0 |
| AIME2025 | 6.27 | 23.90 | 50.13 | 191.20 | 2036.8 |
| BBH | 3.31 | 10.49 | 25.49 | 80.66 | 271.4 |

结论范围：当前配置下，去掉Luna筛选后的整体性能更强，尤其BBH；但teacher介入更频繁、teacher写入量更大，不能据此断言相同teacher预算下judge一定有害。建议下一步匹配teacher token预算或介入次数上限，检验Luna是否拒绝了有益候选。直接查看Luna历史拒绝记录也只能提供线索；配对分支和多seed才能更可靠地验证。随机组还改变了位置采样、次数上限和student分段；32-token组有2题按用户要求舍弃并计错，均需保留比较限制。

## 执行与核验

初始GPU102506于14:12:24—14:13:14因vLLM端口冲突失败，没有生成任何题。修复为每rank不同起始端口后，GPU102508于14:16:24—15:16:47运行1小时23秒成功；始终最多2节点16H800，64CPU/节点。CPU102507于15:17:27—15:17:38运行11秒成功，无GPU。最后长轨迹00452达到既定40264token上限，其余2313题自然结束；没有generation异常重试，也未舍弃题目。

远程audit.json核验2314题与基线的提示/参考答案、输出评分哈希、全部生成早于任何评分、候选阈值/首候选、teacher段<=8token、首token排除和零judge调用。所有2314条评分status=ok。原始输出和评分见artifacts，BBH子任务见artifacts/bbh_subtasks.json。


## BBH子任务（无judge）

| 子任务 | 正确/40 | 正确率 |
|---|---:|---:|
| boolean_expressions | 38/40 | 95.0% |
| causal_judgement | 23/40 | 57.5% |
| date_understanding | 37/40 | 92.5% |
| disambiguation_qa | 18/40 | 45.0% |
| dyck_languages | 0/40 | 0.0% |
| formal_fallacies | 36/40 | 90.0% |
| geometric_shapes | 21/40 | 52.5% |
| hyperbaton | 39/40 | 97.5% |
| logical_deduction_five_objects | 37/40 | 92.5% |
| logical_deduction_seven_objects | 37/40 | 92.5% |
| logical_deduction_three_objects | 39/40 | 97.5% |
| movie_recommendation | 22/40 | 55.0% |
| multistep_arithmetic_two | 38/40 | 95.0% |
| navigate | 39/40 | 97.5% |
| object_counting | 34/40 | 85.0% |
| penguins_in_a_table | 40/40 | 100.0% |
| reasoning_about_colored_objects | 29/40 | 72.5% |
| ruin_names | 26/40 | 65.0% |
| salient_translation_error_detection | 23/40 | 57.5% |
| snarks | 27/40 | 67.5% |
| sports_understanding | 20/40 | 50.0% |
| temporal_sequences | 39/40 | 97.5% |
| tracking_shuffled_objects_five_objects | 40/40 | 100.0% |
| tracking_shuffled_objects_seven_objects | 35/40 | 87.5% |
| tracking_shuffled_objects_three_objects | 40/40 | 100.0% |
| web_of_lies | 35/40 | 87.5% |
| word_sorting | 6/40 | 15.0% |
