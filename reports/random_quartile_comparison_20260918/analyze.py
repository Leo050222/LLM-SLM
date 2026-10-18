import json,csv,collections
from pathlib import Path
p=Path(__file__).parent;d=json.loads((p/'evidence.json').read_text());rows=d['rows'];groups=collections.defaultdict(list)
for r in rows:groups[r['dataset']].append(r)
res=[]
for ds,rs in sorted(groups.items()):
 n=len(rs);v=dict(dataset=ds,n=n)
 for k in ['student','threshold_r0','random']:
  v[k+'_correct']=sum(r[k]['policy_correct'] for r in rs);v[k+'_accuracy']=100*v[k+'_correct']/n
  v[k+'_answer_correct']=sum(r[k]['correct'] for r in rs)
 v['delta_random_student_pp']=v['random_accuracy']-v['student_accuracy'];v['delta_random_threshold_pp']=v['random_accuracy']-v['threshold_r0_accuracy']
 v['relative_random_student_percent']=100*(v['random_accuracy']/v['student_accuracy']-1)
 v['planned']=sum(r['random']['planned'] for r in rs);v['actual']=sum(r['random']['actual'] for r in rs)
 v['r1_accuracy']=100*sum(r['threshold_r1']['correct'] for r in rs)/n if all('threshold_r1' in r for r in rs) else None
 res.append(v)
totals={k:dict(correct=sum(r[k]['policy_correct'] for r in rows),micro=100*sum(r[k]['policy_correct'] for r in rows)/len(rows),macro=sum(v[k+'_accuracy'] for v in res)/len(res)) for k in ['student','random','threshold_r0']}
(p/'summary.json').write_text(json.dumps(dict(groups=res,total=totals,audit=d['audit']),indent=2)+'\n')
with (p/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(res[0]));w.writeheader();w.writerows(res)
lines=['# 四段随机位置介入对比','', '同一4571题，逐题prompt及评分源SHA256已核验。主表统一生成超时计错。阈值主列为r0（seed20260911），随机实验seed20260912、计划来源r1；r1数学尚无评分，不能把主表当作严格相同源rollout对照。代码补充r1如下。数学judge提示逐字节相同；student历史生成组织不同。','', '|数据集|n|弱模型%|随机%|阈值r0%|随机−弱模型pp|随机−阈值pp|阈值r1%|','|---|---:|---:|---:|---:|---:|---:|---:|']
for v in res:
 r1='—' if v['r1_accuracy'] is None else f"{v['r1_accuracy']:.2f}"
 lines.append(f"|{v['dataset']}|{v['n']}|{v['student_accuracy']:.2f}|{v['random_accuracy']:.2f}|{v['threshold_r0_accuracy']:.2f}|{v['delta_random_student_pp']:+.2f}|{v['delta_random_threshold_pp']:+.2f}|{r1}|")
lines+=['',json.dumps(totals,ensure_ascii=False),'',f"计划介入{sum(v['planned'] for v in res)}次，实际{sum(v['actual'] for v in res)}次。正常EOS或其他停止导致实际次数未全部兑现，不能宣称实际预算严格相等。",'', '结论：本次随机介入8个数据集中的7个低于弱模型，CRUXEval-O提高2.625pp，阈值组所有数据集高于随机组。结果与介入位置重要相符，但单随机种子、r0/r1差异和实际次数差异限制因果结论。下一步建议优先补齐原r1数学评分，再考虑更多随机种子。']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
