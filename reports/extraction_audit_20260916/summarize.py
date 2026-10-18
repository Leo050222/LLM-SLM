import collections,csv,json
from pathlib import Path
P=Path(__file__).parent
d=json.loads((P/'results.json').read_text())
groups=collections.defaultdict(collections.Counter)
for r in d['rows']:
 for k,v in r.items():
  if isinstance(v,int):groups[(r['experiment'],r['dataset'])][k]+=v
rows=[dict(experiment=e,dataset=ds,**v) for (e,ds),v in sorted(groups.items())]
(P/'summary.json').write_text(json.dumps(rows,indent=2))
fields=sorted(set().union(*(r.keys() for r in d['rows'])))
with (P/'by_config.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(d['rows'])
lines=['# 答案提取审计（2026-09-16）','',
'仅分析提取，不使用标准答案、不执行正确性判分。trace_timeout 单独统计，不计入 attempted。',
'nonempty 仅表示返回非空候选，不表示候选就是最终答案。syntax_valid 仅表示 Python 语法有效，不表示代码完整、可运行或正确。BBH shape 检查为启发式：选择题检查单个 A-Z；自由回答检查子任务对应的答案形态，不能作为经过人工标注的提取成功率。',
'OlympiadBench 的 nonempty 是第一层提取非空；has_boxed / last_line_math / whole_text_fallback 描述第二层提取路径。multiple_boxes 不一定是错误，多答案题也可能合法。','',
'## 按实验、数据集合并','',
'|实验|数据集|总数|生成超时|实际提取|非空候选|非空率|',
'|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 a=r.get('attempted',0);n=r.get('nonempty',0)
 lines.append(f"|{r['experiment']}|{r['dataset']}|{r['n']}|{r.get('trace_timeout',0)}|{a}|{n}|{100*n/a if a else 0:.3f}%|")
lines+=['','## 所有诊断计数','','```json',json.dumps(rows,ensure_ascii=False,indent=2),'```','',
'## 数据与复现','',
'- audit.py：逐条扫描数学网格 25 组、代码网格（含基线）、dynamic 和历史 AIME。',
'- baseline_audit.py：补充历史学生／教师／online，以及 MATH500 基线。',
'- deepmath_audit.py：两组 DeepMath，各 1000 条。',
'- results.json、baseline_results.json、deepmath_results.json：原始计数和诊断样例。',
'- by_config.csv：逐配置计数。',
'- bbh_examples.json：若干原文结尾与实际提取结果，未比较标准答案。',
'- 本次未修改原始 rollout 或既有评分结果。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n')
for r in rows:print(json.dumps(r))
