# 答案提取审计（2026-09-16）

仅分析提取，不使用标准答案、不执行正确性判分。trace_timeout 单独统计，不计入 attempted。
nonempty 仅表示返回非空候选，不表示候选就是最终答案。syntax_valid 仅表示 Python 语法有效，不表示代码完整、可运行或正确。BBH shape 检查为启发式：选择题检查单个 A-Z；自由回答检查子任务对应的答案形态，不能作为经过人工标注的提取成功率。
OlympiadBench 的 nonempty 是第一层提取非空；has_boxed / last_line_math / whole_text_fallback 描述第二层提取路径。multiple_boxes 不一定是错误，多答案题也可能合法。

## 按实验、数据集合并

|实验|数据集|总数|生成超时|实际提取|非空候选|非空率|
|---|---|---:|---:|---:|---:|---:|
|aime_historical|aime24|90|0|90|90|100.000%|
|aime_historical|aime25|90|0|90|90|100.000%|
|code_grid|cruxeval_i|21600|38|21562|21562|100.000%|
|code_grid|cruxeval_o|21600|117|21483|21483|100.000%|
|code_grid|humaneval_plus|4428|10|4418|4418|100.000%|
|code_grid|livecodebench|4725|134|4591|4591|100.000%|
|code_grid|mbpp_plus|10206|18|10188|10188|100.000%|
|dynamic|bbh|1080|0|1080|1080|100.000%|
|dynamic|cruxeval_i|800|0|800|800|100.000%|
|dynamic|cruxeval_o|800|5|795|795|100.000%|
|dynamic|humaneval_plus|164|0|164|164|100.000%|
|dynamic|livecodebench|175|2|173|173|100.000%|
|dynamic|math500|500|0|500|500|100.000%|
|dynamic|mbpp_plus|378|0|378|378|100.000%|
|dynamic|olympiad_bench|674|83|591|591|100.000%|
|math_grid|bbh|27000|60|26940|26940|100.000%|
|math_grid|math500|12500|59|12441|12441|100.000%|
|math_grid|olympiad_bench|16850|183|16667|16666|99.994%|

## 所有诊断计数

```json
[
  {
    "experiment": "aime_historical",
    "dataset": "aime24",
    "n": 90,
    "attempted": 90,
    "boxed_branch": 90,
    "nonempty": 90
  },
  {
    "experiment": "aime_historical",
    "dataset": "aime25",
    "n": 90,
    "attempted": 90,
    "boxed_branch": 90,
    "nonempty": 90
  },
  {
    "experiment": "code_grid",
    "dataset": "cruxeval_i",
    "n": 21600,
    "attempted": 21562,
    "tag_match": 20618,
    "syntax_valid": 19773,
    "nonempty": 21562,
    "syntax_invalid": 1789,
    "no_tag_fallback": 944,
    "trace_timeout": 38
  },
  {
    "experiment": "code_grid",
    "dataset": "cruxeval_o",
    "n": 21600,
    "attempted": 21483,
    "tag_match": 21162,
    "syntax_valid": 20122,
    "nonempty": 21483,
    "no_tag_fallback": 321,
    "syntax_invalid": 1361,
    "trace_timeout": 117
  },
  {
    "experiment": "code_grid",
    "dataset": "humaneval_plus",
    "n": 4428,
    "attempted": 4418,
    "code_block_match": 4418,
    "syntax_valid": 4415,
    "nonempty": 4418,
    "trace_timeout": 10,
    "syntax_invalid": 3,
    "multiple_code_blocks": 11
  },
  {
    "experiment": "code_grid",
    "dataset": "livecodebench",
    "n": 4725,
    "attempted": 4591,
    "code_block_match": 4591,
    "syntax_valid": 4544,
    "nonempty": 4591,
    "multiple_code_blocks": 555,
    "syntax_invalid": 47,
    "trace_timeout": 134
  },
  {
    "experiment": "code_grid",
    "dataset": "mbpp_plus",
    "n": 10206,
    "attempted": 10188,
    "code_block_match": 10188,
    "syntax_valid": 10187,
    "nonempty": 10188,
    "trace_timeout": 18,
    "syntax_invalid": 1,
    "multiple_code_blocks": 18
  },
  {
    "experiment": "dynamic",
    "dataset": "bbh",
    "n": 1080,
    "attempted": 1080,
    "exact_answer_marker": 1079,
    "ff_n": 400,
    "ff_shape_valid": 323,
    "nonempty": 1080,
    "mc_n": 680,
    "mc_letter": 672,
    "ff_shape_suspicious": 77,
    "mc_not_letter": 8,
    "bbh_no_exact_marker": 1
  },
  {
    "experiment": "dynamic",
    "dataset": "cruxeval_i",
    "n": 800,
    "attempted": 800,
    "tag_match": 775,
    "syntax_valid": 734,
    "nonempty": 800,
    "no_tag_fallback": 25,
    "syntax_invalid": 66
  },
  {
    "experiment": "dynamic",
    "dataset": "cruxeval_o",
    "n": 800,
    "attempted": 795,
    "tag_match": 792,
    "syntax_valid": 750,
    "nonempty": 795,
    "syntax_invalid": 45,
    "no_tag_fallback": 3,
    "trace_timeout": 5
  },
  {
    "experiment": "dynamic",
    "dataset": "humaneval_plus",
    "n": 164,
    "attempted": 164,
    "code_block_match": 164,
    "syntax_valid": 164,
    "nonempty": 164
  },
  {
    "experiment": "dynamic",
    "dataset": "livecodebench",
    "n": 175,
    "attempted": 173,
    "code_block_match": 173,
    "syntax_valid": 172,
    "nonempty": 173,
    "trace_timeout": 2,
    "multiple_code_blocks": 15,
    "syntax_invalid": 1
  },
  {
    "experiment": "dynamic",
    "dataset": "math500",
    "n": 500,
    "attempted": 500,
    "boxed_branch": 500,
    "nonempty": 500
  },
  {
    "experiment": "dynamic",
    "dataset": "mbpp_plus",
    "n": 378,
    "attempted": 378,
    "code_block_match": 378,
    "syntax_valid": 378,
    "nonempty": 378
  },
  {
    "experiment": "dynamic",
    "dataset": "olympiad_bench",
    "n": 674,
    "trace_timeout": 83,
    "attempted": 591,
    "has_boxed": 591,
    "nonempty": 591,
    "exact_final_marker": 51,
    "multiple_boxes": 151
  },
  {
    "experiment": "math_grid",
    "dataset": "bbh",
    "n": 27000,
    "attempted": 26940,
    "exact_answer_marker": 26866,
    "ff_n": 9984,
    "ff_shape_valid": 8037,
    "nonempty": 26940,
    "mc_n": 16956,
    "mc_letter": 16714,
    "ff_shape_suspicious": 1947,
    "mc_not_letter": 242,
    "bbh_no_exact_marker": 74,
    "trace_timeout": 60
  },
  {
    "experiment": "math_grid",
    "dataset": "math500",
    "n": 12500,
    "attempted": 12441,
    "boxed_branch": 12441,
    "nonempty": 12441,
    "trace_timeout": 59
  },
  {
    "experiment": "math_grid",
    "dataset": "olympiad_bench",
    "n": 16850,
    "attempted": 16667,
    "has_boxed": 16658,
    "nonempty": 16666,
    "multiple_boxes": 4977,
    "exact_final_marker": 1510,
    "last_line_math": 1,
    "trace_timeout": 183,
    "whole_text_fallback": 8,
    "empty": 1
  }
]
```

## 数据与复现

- audit.py：逐条扫描数学网格 25 组、代码网格（含基线）、dynamic 和历史 AIME。
- baseline_audit.py：补充历史学生／教师／online，以及 MATH500 基线。
- deepmath_audit.py：两组 DeepMath，各 1000 条。
- results.json、baseline_results.json、deepmath_results.json：原始计数和诊断样例。
- by_config.csv：逐配置计数。
- bbh_examples.json：若干原文结尾与实际提取结果，未比较标准答案。
- 本次未修改原始 rollout 或既有评分结果。
