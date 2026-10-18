# 三组单次 non-thinking rollout 评测

2026-09-11。Student 为 Qwen3-1.7B；teacher 为 Qwen3-14B。每题每组仅生成一次，表中是单次正确率。

| 数据集 | 题数 | Student 直接 | 在线 teacher 8 token | Teacher 直接 | 在线相对 student |
|---|---:|---:|---:|---:|---:|
| AIME2024 | 30 | 5/30（16.67%） | 3/30（10.00%） | 8/30（26.67%） | -6.67 个百分点 |
| AIME2025 | 30 | 4/30（13.33%） | 7/30（23.33%） | 8/30（26.67%） | +10.00 个百分点 |
| OlympiadBench 英文数学 | 674 | 202/674（29.97%） | 232/674（34.42%） | 314/674（46.59%） | +4.45 个百分点 |
| BBH（27×40） | 1080 | 378/1080（35.00%） | 484/1080（44.81%） | 835/1080（77.31%） | +9.81 个百分点 |

在线介入在 OlympiadBench、AIME2025 和 BBH 上高于 student，AIME2024 低于 student；四组数据集上都低于 teacher 直接生成。AIME 每年仅30题，单次采样不能据此断言稳定提升或下降。

## 设置与评分口径

- OlympiadBench 沿用 OE_TO_maths_en_COMP 全674题；AIME2024/2025各30题；BBH官方27个子任务各固定随机抽40题，共1080题。抽样索引及来源哈希见 artifacts/data_manifest.json。
- 三组共享题目和提示 token，原生 non-thinking，0-shot，temperature=1、top_p=1，seed=20260911。输出上限40264 token，并受40960上下文限制。
- 在线算法保持原实现：student chunk64，teacher每次最多8 token，gap2，不设介入次数上限。BBH也沿用原有数学推理判定提示词，未针对BBH调整算法。
- AIME采用EvalScope数学提取、规范化与grader；OlympiadBench采用其官方adapter提取和MathJudger；BBH采用adapter答案提取后保留标点的规范化精确匹配，dyck仅忽略空白。
- 评分不存在异常或超时。有1条student/Olympiad提取为空：00309.json中的答案换行使adapter提取为空；实际输出33、标准11，因此仍属错误，不影响正确题数。

生成长度限制触发数：student 7题、在线组2题、teacher 0题。这些题保留原输出并按同一规则评分，没有丢弃或缩短实验范围。

## 在线介入用量

累计 7068 次介入，teacher贡献 55451 token；平均每题 3.90 次、30.57 teacher token，单题最多 90 次。

## 执行与核验

- 初始GPU作业102199：04:11:19—05:18:47；2节点16 H800。1812题完成，2题请求异常。
- 自动恢复GPU作业102213：05:19:41—05:25:02，原2节点16 H800，补齐两题；两次作业没有重叠。
- 评分作业102200：05:25:32—05:25:48，1节点64 CPU、48评分进程，无GPU。全部生成结束后启动。
- 全部5442条输出与评分哈希、题目身份、共享提示和先生成后评分时间顺序已核验；原始输出、逐题评分已同步本地。
- 详见 audit.json、local_verification.json、artifacts/summary.json 和 artifacts/scores/。

## BBH 子任务（每项40题）

| 子任务 | Student | 在线8 token | Teacher |
|---|---:|---:|---:|
| boolean_expressions | 2/40（5.0%） | 3/40（7.5%） | 38/40（95.0%） |
| causal_judgement | 17/40（42.5%） | 24/40（60.0%） | 25/40（62.5%） |
| date_understanding | 26/40（65.0%） | 29/40（72.5%） | 37/40（92.5%） |
| disambiguation_qa | 7/40（17.5%） | 9/40（22.5%） | 22/40（55.0%） |
| dyck_languages | 1/40（2.5%） | 0/40（0.0%） | 0/40（0.0%） |
| formal_fallacies | 7/40（17.5%） | 4/40（10.0%） | 28/40（70.0%） |
| geometric_shapes | 14/40（35.0%） | 20/40（50.0%） | 25/40（62.5%） |
| hyperbaton | 28/40（70.0%） | 26/40（65.0%） | 39/40（97.5%） |
| logical_deduction_five_objects | 25/40（62.5%） | 19/40（47.5%） | 38/40（95.0%） |
| logical_deduction_seven_objects | 25/40（62.5%） | 22/40（55.0%） | 36/40（90.0%） |
| logical_deduction_three_objects | 29/40（72.5%） | 30/40（75.0%） | 38/40（95.0%） |
| movie_recommendation | 7/40（17.5%） | 14/40（35.0%） | 23/40（57.5%） |
| multistep_arithmetic_two | 2/40（5.0%） | 5/40（12.5%） | 40/40（100.0%） |
| navigate | 10/40（25.0%） | 13/40（32.5%） | 39/40（97.5%） |
| object_counting | 21/40（52.5%） | 33/40（82.5%） | 36/40（90.0%） |
| penguins_in_a_table | 28/40（70.0%） | 35/40（87.5%） | 40/40（100.0%） |
| reasoning_about_colored_objects | 18/40（45.0%） | 32/40（80.0%） | 33/40（82.5%） |
| ruin_names | 11/40（27.5%） | 15/40（37.5%） | 22/40（55.0%） |
| salient_translation_error_detection | 9/40（22.5%） | 21/40（52.5%） | 28/40（70.0%） |
| snarks | 21/40（52.5%） | 21/40（52.5%） | 27/40（67.5%） |
| sports_understanding | 0/40（0.0%） | 1/40（2.5%） | 19/40（47.5%） |
| temporal_sequences | 15/40（37.5%） | 29/40（72.5%） | 39/40（97.5%） |
| tracking_shuffled_objects_five_objects | 18/40（45.0%） | 24/40（60.0%） | 40/40（100.0%） |
| tracking_shuffled_objects_seven_objects | 21/40（52.5%） | 29/40（72.5%） | 37/40（92.5%） |
| tracking_shuffled_objects_three_objects | 16/40（40.0%） | 21/40（52.5%） | 40/40（100.0%） |
| web_of_lies | 0/40（0.0%） | 0/40（0.0%） | 34/40（85.0%） |
| word_sorting | 0/40（0.0%） | 5/40（12.5%） | 12/40（30.0%） |
