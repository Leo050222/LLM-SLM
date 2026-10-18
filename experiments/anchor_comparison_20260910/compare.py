import json
from pathlib import Path
P=Path(__file__).parent;d=json.loads((P/'aligned.json').read_text());groups={k:{x['dataset_index']:x for x in v} for k,v in d['groups'].items()};ids=set(groups['anchor']);out={'same_subset':{},'full1000':{},'overlap':{},'scoring_changes':d['changes']}
def stats(rows):
 n=sum(x['n'] for x in rows);c=sum(x['correct'] for x in rows);solved=sum(x['correct']>0 for x in rows)
 return {'questions':len(rows),'samples':n,'correct':c,'accuracy':c/n,'solved':solved,'pass_any':solved/len(rows)}
for k,g in groups.items():
 out['same_subset'][k]=stats([g[i] for i in sorted(ids)])
 if len(g)==1000:out['full1000'][k]=stats(list(g.values()))
a={i for i in ids if groups['anchor'][i]['correct']>0}
for k in groups:
 if k=='anchor':continue
 b={i for i in ids if groups[k][i]['correct']>0}
 out['overlap'][k]={'both':len(a&b),'anchor_only':len(a-b),'baseline_only':len(b-a),'neither':len(ids-(a|b))}
out['no_anchor_indices']=sorted(set(range(1000))-ids)
(P/'comparison.json').write_text(json.dumps(out,indent=2))
report=['# Anchor支持与固定prefix对比','', '所有组按anchor pipeline的布尔归一化规则统一评分；使用相同958道找到anchor的测试题。其余42题不纳入这张表。','', '|实验|回答数|正确数|平均正确率|至少一次做对|pass@k|','|---|---:|---:|---:|---:|---:|']
for k,s in out['same_subset'].items():report.append(f"|{k}|{s['samples']}|{s['correct']}|{100*s['accuracy']:.4f}%|{s['solved']}|{100*s['pass_any']:.4f}%|")
report+=['','Teacher使用8次，其余各组64次。Anchor是一个student原始trace上的anchor，8条teacher 256-token bridge，每条student继续8次。它还包含anchor寻找和原始student prefix成本，不等同于从头64次的总计算预算。','', '原始结果未改写。aligned.json保存统一后的逐题统计、评分变化数量以及原始source manifest；comparison.json保存同题对比、全1000题基线和做对题目重合。']
(P/'REPORT.md').write_text('\n'.join(report)+'\n');print(json.dumps(out,indent=2))
