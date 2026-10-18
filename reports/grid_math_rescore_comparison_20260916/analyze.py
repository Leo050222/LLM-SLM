import collections,csv,json
from pathlib import Path
P=Path(__file__).parent
data=json.loads((P/'paired_results.json').read_text())
rows=data['rows']
def enrich(r):
 n=r['n'];a=r['old_correct'];b=r['new_correct']
 return dict(r,old_accuracy=100*a/n,new_accuracy=100*b/n,delta_pp=100*(b-a)/n,
             relative_change_percent=100*(b-a)/a if a else None,
             flipped=r.get('wrong_to_right',0)+r.get('right_to_wrong',0),
             flipped_percent=100*(r.get('wrong_to_right',0)+r.get('right_to_wrong',0))/n,
             new_accuracy_preserving_timeout_wrong=100*(b-r.get('timeout_new_correct',0))/n)
rows=[enrich(r) for r in rows]
agg=collections.defaultdict(collections.Counter)
for r in data['rows']:
 for k,v in r.items():
  if isinstance(v,int):agg[r['dataset']][k]+=v
totals=[enrich(dict(dataset=ds,**v)) for ds,v in sorted(agg.items())]
rankings={}
for ds in agg:
 rr=[r for r in rows if r['dataset']==ds]
 old=sorted(rr,key=lambda r:(-r['old_correct'],r['config']))
 new=sorted(rr,key=lambda r:(-r['new_correct'],r['config']))
 # Average ranks for ties; correlations describe configuration ordering, not statistical evidence.
 def ranks(field):return {r['config']:1+sum(q[field]>r[field] for q in rr)+(sum(q[field]==r[field] for q in rr)-1)/2 for r in rr}
 a=ranks('old_correct');b=ranks('new_correct');am=sum(a.values())/len(a);bm=sum(b.values())/len(b)
 cov=sum((a[k]-am)*(b[k]-bm) for k in a)
 denom=(sum((v-am)**2 for v in a.values())*sum((v-bm)**2 for v in b.values()))**.5
 rankings[ds]=dict(old_top=[dict(config=r['config'],accuracy=r['old_accuracy']) for r in old if r['old_correct']==old[0]['old_correct']],new_top=[dict(config=r['config'],accuracy=r['new_accuracy']) for r in new if r['new_correct']==new[0]['new_correct']],spearman=cov/denom if denom else None,min_delta_pp=min(r['delta_pp'] for r in rr),max_delta_pp=max(r['delta_pp'] for r in rr))
for name,rs in [('per_config.csv',rows),('per_dataset.csv',totals)]:
 keys=list(dict.fromkeys(k for r in rs for k in r))
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
out=dict(verification=data['verification'],totals=totals,rankings=rankings,k1_t2=[r for r in rows if r['config']=='k1_t2'])
(P/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
lines=['# GPT-4o-mini 重评分与旧评分对比','',
'范围：MATH500、OlympiadBench、BBH 各25个配置，共56,350条。逐条核验新旧评分的 source_sha256、problem_id，以及新评分的 input_sha256。',
'同一数据集的25个配置使用同一题集，因此汇总不是56,350个独立问题，也不是多随机种子的均值。新评分代表 GPT-4o-mini 判断，不等同于人工核验真值。学生/教师基线不在本次 math 网格重评分中，不能据此更新相对基线增益。',
'','## 汇总','',
'|数据集|rollout数|旧正确率|新正确率|变化(pp)|相对变化|错→对|对→错|翻转比例|',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in totals:lines.append(f"|{r['dataset']}|{r['n']}|{r['old_accuracy']:.2f}%|{r['new_accuracy']:.2f}%|{r['delta_pp']:+.2f}|{r['relative_change_percent']:+.2f}%|{r.get('wrong_to_right',0)}|{r.get('right_to_wrong',0)}|{r['flipped_percent']:.2f}%|")
lines+=['','## 超时口径影响','',
'旧评分把 trace_timeout 自动判错；新评分对保留文本提取答案后判断。下表把这些记录重新固定为错，以隔离该口径差异。','',
'|数据集|生成超时数|新判对的超时数|新正确率|保留旧超时口径的新正确率|','|---|---:|---:|---:|---:|']
for r in totals:lines.append(f"|{r['dataset']}|{r.get('trace_timeouts',0)}|{r.get('timeout_new_correct',0)}|{r['new_accuracy']:.2f}%|{r['new_accuracy_preserving_timeout_wrong']:.2f}%|")
lines+=['','## 配置排名','',json.dumps(rankings,ensure_ascii=False,indent=2),'','## 全部配置','',
'每格为旧正确率 → 新正确率（百分点变化）。','',
'|配置|MATH500|OlympiadBench|BBH|','|---|---:|---:|---:|']
def sort_config(g):
 k,t=g[1:].split('_t');return float(t),int(k)
for config in sorted({r['config'] for r in rows},key=sort_config):
 cells=[]
 for ds in ['math500','olympiad_bench','bbh']:
  r=next(r for r in rows if r['dataset']==ds and r['config']==config)
  cells.append(f"{r['old_accuracy']:.2f}% → {r['new_accuracy']:.2f}% ({r['delta_pp']:+.2f})")
 lines.append('|'+config+'|'+'|'.join(cells)+'|')
lines+=['','## 翻转案例（诊断样例，不代表随机抽样）','']
for key,examples in data['examples'].items():
 lines+=['### '+key,'']
 for e in examples[:3]:
  candidate=e['candidate_answer'] or '';preview=candidate if len(candidate)<1000 else candidate[:100]+' … '+candidate[-700:]
  lines+=['```json',json.dumps(dict(config=e['config'],problem_id=e['problem_id'],candidate_preview=preview,reference=e['reference_answer'],judge_reason=e['reason'],old_status=e['old_status']),ensure_ascii=False,indent=2),'```','']
lines+=['## 后续核查','',
'优先人工盲审错→对和对→错样本，按数据集、配置和答案类型分层；尤其检查新最优配置与旧最优配置的差异。GPT-4o-mini 标签改变不自动证明旧标签错误。','']
(P/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(out,ensure_ascii=False,indent=2))
