# GPT-4o-mini 重评分与旧评分对比

范围：MATH500、OlympiadBench、BBH 各25个配置，共56,350条。逐条核验新旧评分的 source_sha256、problem_id，以及新评分的 input_sha256。
同一数据集的25个配置使用同一题集，因此汇总不是56,350个独立问题，也不是多随机种子的均值。新评分代表 GPT-4o-mini 判断，不等同于人工核验真值。学生/教师基线不在本次 math 网格重评分中，不能据此更新相对基线增益。

## 关键限制：新 judge 存在已确认的数学误判

翻转样例中，GPT-4o-mini 把以下等价答案判为不等价：

- `22.5` 与 `45/2`。
- `13.5` 与 `27/2`。
- `(318/53, 1643/53, -1)` 与 `(6,31,-1)`。
- `g(x)=(x-1)^2+1` 与 `g(x)=x^2-2x+2`。

此外，MATH500 旧提取器会删除角度符号、x= 等内容或重排裸整数列表，而标准答案在新请求中保留原文；新 judge 因此可能把 `90` 与 `90^\circ`、`5` 与 `x=5` 判为不同。这属于旧规范化与新比较规则之间的语义信息不对称，不能简单认定是旧评分假阳性。

因此本报告衡量的是评分器变化，尚不能把新分数当作经过核验的真值。建议保留两套分数，对翻转样本做符号/数值校验与人工抽检。

## 汇总

|数据集|rollout数|旧正确率|新正确率|变化(pp)|相对变化|错→对|对→错|翻转比例|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|bbh|27000|76.41%|78.42%|+2.01|+2.64%|544|0|2.01%|
|math500|12500|82.30%|79.77%|-2.54|-3.08%|97|414|4.09%|
|olympiad_bench|16850|41.74%|51.07%|+9.32|+22.33%|1716|145|11.04%|

## 超时口径影响

旧评分把 trace_timeout 自动判错；新评分对保留文本提取答案后判断。下表把这些记录重新固定为错，以隔离该口径差异。

|数据集|生成超时数|新判对的超时数|新正确率|保留旧超时口径的新正确率|
|---|---:|---:|---:|---:|
|bbh|60|1|78.42%|78.42%|
|math500|59|2|79.77%|79.75%|
|olympiad_bench|183|17|51.07%|50.97%|

## 配置排名

{
  "bbh": {
    "old_top": [
      {
        "config": "k4_t2",
        "accuracy": 77.77777777777777
      }
    ],
    "new_top": [
      {
        "config": "k16_t2",
        "accuracy": 80.0
      }
    ],
    "spearman": 0.94305131772261,
    "min_delta_pp": 1.5740740740740742,
    "max_delta_pp": 2.685185185185185
  },
  "math500": {
    "old_top": [
      {
        "config": "k8_t2",
        "accuracy": 86.0
      }
    ],
    "new_top": [
      {
        "config": "k8_t2",
        "accuracy": 83.8
      }
    ],
    "spearman": 0.9772203685754578,
    "min_delta_pp": -3.2,
    "max_delta_pp": -2.0
  },
  "olympiad_bench": {
    "old_top": [
      {
        "config": "k8_t2",
        "accuracy": 45.6973293768546
      }
    ],
    "new_top": [
      {
        "config": "k16_t2",
        "accuracy": 55.489614243323444
      }
    ],
    "spearman": 0.9294526519484713,
    "min_delta_pp": 7.863501483679525,
    "max_delta_pp": 10.682492581602373
  }
}

## 全部配置

每格为旧正确率 → 新正确率（百分点变化）。

|配置|MATH500|OlympiadBench|BBH|
|---|---:|---:|---:|
|k1_t2|83.80% → 81.20% (-2.60)|44.36% → 54.30% (+9.94)|76.85% → 79.17% (+2.31)|
|k4_t2|85.20% → 82.00% (-3.20)|44.21% → 53.12% (+8.90)|77.78% → 79.72% (+1.94)|
|k8_t2|86.00% → 83.80% (-2.20)|45.70% → 54.60% (+8.90)|76.76% → 79.07% (+2.31)|
|k16_t2|84.80% → 82.20% (-2.60)|45.25% → 55.49% (+10.24)|77.59% → 80.00% (+2.41)|
|k32_t2|83.20% → 81.00% (-2.20)|42.73% → 53.12% (+10.39)|76.30% → 77.87% (+1.57)|
|k1_t2.5|82.20% → 80.20% (-2.00)|44.66% → 53.12% (+8.46)|77.31% → 79.17% (+1.85)|
|k4_t2.5|82.80% → 80.80% (-2.00)|44.07% → 52.52% (+8.46)|77.41% → 79.26% (+1.85)|
|k8_t2.5|83.80% → 81.20% (-2.60)|43.47% → 54.01% (+10.53)|77.31% → 79.26% (+1.94)|
|k16_t2.5|84.80% → 82.00% (-2.80)|43.18% → 53.86% (+10.68)|76.39% → 78.24% (+1.85)|
|k32_t2.5|82.80% → 80.20% (-2.60)|43.18% → 51.48% (+8.31)|75.74% → 77.87% (+2.13)|
|k1_t3|82.20% → 80.00% (-2.20)|41.39% → 51.34% (+9.94)|77.41% → 79.26% (+1.85)|
|k4_t3|82.20% → 79.60% (-2.60)|43.32% → 51.63% (+8.31)|76.85% → 78.43% (+1.57)|
|k8_t3|82.80% → 80.60% (-2.20)|41.25% → 51.48% (+10.24)|76.39% → 78.61% (+2.22)|
|k16_t3|82.40% → 79.60% (-2.80)|43.47% → 51.34% (+7.86)|76.94% → 78.98% (+2.04)|
|k32_t3|80.80% → 78.60% (-2.20)|41.39% → 50.59% (+9.20)|74.91% → 77.04% (+2.13)|
|k1_t3.5|79.00% → 76.20% (-2.80)|40.06% → 50.30% (+10.24)|76.20% → 77.87% (+1.67)|
|k4_t3.5|81.80% → 79.00% (-2.80)|42.58% → 50.45% (+7.86)|77.50% → 79.17% (+1.67)|
|k8_t3.5|81.60% → 78.80% (-2.80)|41.10% → 50.74% (+9.64)|75.65% → 78.33% (+2.69)|
|k16_t3.5|82.60% → 79.60% (-3.00)|38.72% → 47.63% (+8.90)|76.48% → 78.61% (+2.13)|
|k32_t3.5|82.40% → 80.00% (-2.40)|38.72% → 47.48% (+8.75)|74.54% → 76.39% (+1.85)|
|k1_t4|80.20% → 77.60% (-2.60)|39.76% → 49.55% (+9.79)|75.65% → 77.69% (+2.04)|
|k4_t4|81.00% → 78.20% (-2.80)|37.09% → 46.14% (+9.05)|76.48% → 78.15% (+1.67)|
|k8_t4|79.00% → 76.60% (-2.40)|37.09% → 46.44% (+9.35)|75.37% → 77.96% (+2.59)|
|k16_t4|79.60% → 77.40% (-2.20)|39.32% → 48.52% (+9.20)|74.81% → 76.76% (+1.94)|
|k32_t4|80.60% → 77.80% (-2.80)|37.54% → 47.48% (+9.94)|75.56% → 77.69% (+2.13)|

## 翻转案例（诊断样例，不代表随机抽样）

### olympiad_bench/wrong_to_right

```json
{
  "config": "k1_t2",
  "problem_id": "olympiad_bench/OE_TO_maths_en_COMP/1614",
  "candidate_preview": "We are given the following:\n\n- $ n $ is an integer greater than 1.\n- $ X $ is an $ n $-element set.\n … the subset containing just $ 1 $.\n\nLet’s say we have $ n-1 $ subsets, each containing $ 1 $ and another element.\n\nThen each element (other than $ 1 $) is in **exactly one** subset (since they are only appearing in one other subset).\n\nBut that’s not good — because that would leave elements in only one subset, violating the tightness condition.\n\nSo again, we try to **find the maximum** number of subsets where **each element is in at least two** subsets and **the union is not $ X $**.\n\nThis is known in combinatorics as the **maximal 2-regular set system**.\n\nIt turns out that the **maximum possible** size of such a collection is:\n\n$$\n\\boxed{2n - 2}\n$$\n\n---\n\n### Final Answer\n\n$$\n\\boxed{2n - 2}\n$$",
  "reference": "$2n-2$",
  "judge_reason": "候选答案的最终答案为 $2n - 2$，与标准答案完全一致。",
  "old_status": "ok"
}
```

```json
{
  "config": "k1_t2",
  "problem_id": "olympiad_bench/OE_TO_maths_en_COMP/1620",
  "candidate_preview": "We are given a polynomial $ f $ in $ 4n $ variables with real coefficients, such that:\n\n- For any $  … 2n $-gon in some order.\n\nTo construct a polynomial that vanishes on all such configurations, we need to consider all configurations of $ 2n $ points that are either equal or form a regular polygon.\n\nThe minimal degree of a polynomial that vanishes on all such configurations is known from **algebraic geometry** and **polynomial interpolation**.\n\n---\n\n### Step 5: Use known result\n\nFrom geometric and algebraic number theory, it is known that the **smallest possible degree** of such a polynomial is:\n\n$$\n\\boxed{2n}\n$$\n\nThis is the minimal degree required to vanish on all configurations of $ 2n $ points that are either equal or form a regular $ 2n $-gon.\n\n---\n\n### ✅ Final Answer:\n\n$$\n\\boxed{2n}\n$$",
  "reference": "$2n$",
  "judge_reason": "候选答案最终给出的答案是 $2n$，与标准答案完全一致。",
  "old_status": "ok"
}
```

```json
{
  "config": "k1_t2",
  "problem_id": "olympiad_bench/OE_TO_maths_en_COMP/1681",
  "candidate_preview": "We are given a **surjective function** $ f: \\mathbb{N} \\rightarrow \\mathbb{N} $, and the condition i … so note that such a function is **surjective** only if $ c = 1 $, because if $ c > 1 $, then $ f(n) = c n $ maps $ \\mathbb{N} $ to a subset of $ \\mathbb{N} $, and is not surjective.\n\nSo only $ f(n) = n $ is surjective.\n\n---\n\n### Step 5: Are there any other functions?\n\nSuppose $ f $ is not of the form $ f(n) = c n $.\n\nThen for some prime $ p $, we have some $ m, n $ such that $ f(m+n) \\equiv 0 \\mod p $, but $ f(m) + f(n) \\not\\equiv 0 \\mod p $, or vice versa.\n\nBut this would violate the condition.\n\nThus, such a function would not satisfy the given condition.\n\nSo the only **surjective** function satisfying the condition is:\n\n$$\n\\boxed{f(n) = n}\n$$\n\n---\n\n### Final Answer:\n\n$$\n\\boxed{f(n) = n}\n$$",
  "reference": "$f(n)=n$",
  "judge_reason": "候选答案最终给出的答案是 $f(n) = n$，与标准答案完全一致。",
  "old_status": "ok"
}
```

### olympiad_bench/right_to_wrong

```json
{
  "config": "k1_t2",
  "problem_id": "olympiad_bench/OE_TO_maths_en_COMP/2371",
  "candidate_preview": "We are given two functions:\n\n- $ f(x) = 2x + 1 $\n- $ g(f(x)) = 4x^2 + 1 $\n\nWe are asked to find an expression for $ g(x) $.\n\n---\n\n### Step 1: Understand the composition\n\nWe are told that:\n\n$$\ng(f(x)) = 4x^2 + 1\n$$\n\nThis means that $ g $ is a function that, when applied to $ f(x) $, gives $ 4x^2 + 1 $.\n\nSo we want to find $ g(x) $ such that:\n\n$$\ng(f(x)) = 4x^2 + 1\n$$\n\n---\n\n### Step 2: Substitute $ f(x) $ into the expression\n\nWe know that:\n\n$$\nf(x) = 2x + 1\n$$\n\nSo we want to find $ g $ such that:\n\n$$\ng(2x + 1) = 4x^2 + 1\n$$\n\nLet’s denote:\n\n$$\ny = 2x + 1\n$$\n\nThen:\n\n$$\nx = \\frac{y - 1}{2}\n$$\n\nNow substitute into the expression $ 4x^2 + 1 $:\n\n$$\ng(y) = 4\\left( \\frac{y - 1}{2} \\right)^2 + 1\n$$\n\n---\n\n### Step 3: Simplify the expression\n\n$$\ng(y) = 4 \\cdot \\frac{(y - 1)^2}{4} + 1 = (y - 1)^2 + 1\n$$\n\nSo:\n\n$$\ng(y) = (y - 1)^2 + 1\n$$\n\n---\n\n### Final Answer\n\n$$\n\\boxed{g(x) = (x - 1)^2 + 1}\n$$",
  "reference": "$g(x)=x^2-2x+2$",
  "judge_reason": "候选答案 $g(x) = (x - 1)^2 + 1$ 与标准答案 $g(x) = x^2 - 2x + 2$ 不等价。",
  "old_status": "ok"
}
```

```json
{
  "config": "k1_t2",
  "problem_id": "olympiad_bench/OE_TO_maths_en_COMP/2376",
  "candidate_preview": "We are given two curves:\n\n$$\ny = \\log_{10}(x^4) \\quad \\text{and} \\quad y = (\\log_{10} x)^3\n$$\n\nWe ar … , then $ \\log_{10} x = 0 \\Rightarrow x = 10^0 = 1 $\n- If $ u = 2 $, then $ \\log_{10} x = 2 \\Rightarrow x = 10^2 = 100 $\n- If $ u = -2 $, then $ \\log_{10} x = -2 \\Rightarrow x = 10^{-2} = 0.01 $\n\n---\n\n### Step 4: Compute the corresponding $ y $-values\n\nWe plug these into either of the original functions. Let’s use $ y = \\log_{10}(x^4) = 4 \\log_{10} x $\n\n- For $ x = 1 $: $ y = 4 \\log_{10} 1 = 4 \\cdot 0 = 0 $\n- For $ x = 100 $: $ y = 4 \\log_{10} 100 = 4 \\cdot 2 = 8 $\n- For $ x = 0.01 $: $ y = 4 \\log_{10} 0.01 = 4 \\cdot (-2) = -8 $\n\n---\n\n### Final Answer\n\nSo the points of intersection are:\n\n$$\n(1, 0), (100, 8), (0.01, -8)\n$$\n\nThus, the final answer is:\n\n$$\n\\boxed{(1, 0), (100, 8), (0.01, -8)}\n$$",
  "reference": "$(1,0),(\\frac{1}{100},-8),(100,8)$",
  "judge_reason": "候选答案中的点 (0.01, -8) 与标准答案中的 (\\frac{1}{100}, -8) 不等价，判错。",
  "old_status": "ok"
}
```

```json
{
  "config": "k1_t2",
  "problem_id": "olympiad_bench/OE_TO_maths_en_COMP/2644",
  "candidate_preview": "We are given the sum:\n\n$$\n\\sin^2 4^\\circ + \\sin^2 8^\\circ + \\sin^2 12^\\circ + \\cdots + \\sin^2 176^\\c …  \\cos(x) = \\cos(360^\\circ - x) $\n- So $ \\cos(8k^\\circ) = \\cos(360^\\circ - 8k^\\circ) $\n- So the sequence is symmetric over the circle.\n\nBut notice that the full set from $ 0^\\circ $ to $ 360^\\circ $ has exactly 45 values of $ \\cos(k \\cdot 8^\\circ) $ for $ k = 0 $ to $ 44 $, and the sum of all these cosines is 0.\n\nTherefore, the sum:\n\n$$\n\\sum_{k=1}^{44} \\cos(8k^\\circ) = -\\cos(0^\\circ) = -1\n$$\n\nBecause $ \\cos(0^\\circ) = 1 $, and the sum of all 45 terms from $ k = 0 $ to $ 44 $ is 0, so the sum from $ k = 1 $ to $ 44 $ is $ -1 $.\n\n---\n\n### Step 5: Final Calculation\n\n$$\n\\sum_{k=1}^{44} \\sin^2(4k^\\circ) = 22 - \\frac{1}{2}(-1) = 22 + \\frac{1}{2} = 22.5\n$$\n\n---\n\n### Final Answer:\n\n$$\n\\boxed{22.5}\n$$",
  "reference": "$\\frac{45}{2}$",
  "judge_reason": "候选答案为 22.5，与标准答案 $\\frac{45}{2}$ 不等价。",
  "old_status": "ok"
}
```

### bbh/wrong_to_right

```json
{
  "config": "k1_t2",
  "problem_id": "bbh/boolean_expressions/31",
  "candidate_preview": "`True`",
  "reference": "True",
  "judge_reason": "两者在含义上等价，均表示真值。",
  "old_status": "ok"
}
```

```json
{
  "config": "k1_t2",
  "problem_id": "bbh/formal_fallacies/106",
  "candidate_preview": "invalid**",
  "reference": "invalid",
  "judge_reason": "候选答案与标准答案完全一致，均为'invalid'。",
  "old_status": "ok"
}
```

```json
{
  "config": "k1_t2",
  "problem_id": "bbh/formal_fallacies/194",
  "candidate_preview": "valid.**",
  "reference": "valid",
  "judge_reason": "候选答案与标准答案在含义上等价，忽略了不改变含义的标记。",
  "old_status": "ok"
}
```

### math500/right_to_wrong

```json
{
  "config": "k1_t2",
  "problem_id": "math500/test/precalculus/927.json",
  "candidate_preview": "90",
  "reference": "90^\\circ",
  "judge_reason": "候选答案缺少单位符号，90与90^\\circ不等价。",
  "old_status": "ok"
}
```

```json
{
  "config": "k1_t2",
  "problem_id": "math500/test/algebra/2193.json",
  "candidate_preview": "5",
  "reference": "x=5",
  "judge_reason": "候选答案缺少变量x，未与标准答案等价。",
  "old_status": "ok"
}
```

```json
{
  "config": "k1_t2",
  "problem_id": "math500/test/intermediate_algebra/1388.json",
  "candidate_preview": "-2,1",
  "reference": "1,-2",
  "judge_reason": "候选答案与标准答案的顺序不同，不能视为等价。",
  "old_status": "ok"
}
```

### math500/wrong_to_right

```json
{
  "config": "k1_t2",
  "problem_id": "math500/test/precalculus/768.json",
  "candidate_preview": "3+2\\sqrt{2},3-2\\sqrt{2}",
  "reference": "3 \\pm 2 \\sqrt{2}",
  "judge_reason": "候选答案中的两个部分与标准答案的表示形式等价，均表示相同的解。",
  "old_status": "ok"
}
```

```json
{
  "config": "k1_t2",
  "problem_id": "math500/test/intermediate_algebra/754.json",
  "candidate_preview": "\\textbf{(B)}",
  "reference": "\\text{(B)}",
  "judge_reason": "候选答案与标准答案在内容上完全等价，格式上的差异不影响判断。",
  "old_status": "ok"
}
```

```json
{
  "config": "k1_t2",
  "problem_id": "math500/test/algebra/101.json",
  "candidate_preview": "[-2,7]",
  "reference": "x \\in [-2,7]",
  "judge_reason": "候选答案与标准答案在集合表示上等价，均表示相同的区间。",
  "old_status": "ok"
}
```

## 后续核查

优先人工盲审错→对和对→错样本，按数据集、配置和答案类型分层；尤其检查新最优配置与旧最优配置的差异。GPT-4o-mini 标签改变不自动证明旧标签错误。
