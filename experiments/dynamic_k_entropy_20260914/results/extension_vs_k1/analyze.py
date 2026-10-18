from pathlib import Path
import json,csv,math
from collections import Counter
P=Path(__file__).parent
rows=json.loads((P/'paired.json').read_text());assert len(rows)==4571
order=['math500','olympiad_bench','bbh','cruxeval_o','cruxeval_i','humaneval_plus','mbpp_plus','livecodebench']

def timedout(r):
 return any('timeout' in str(r[k]).lower() for k in ['dynamic_finish','k1_finish','dynamic_score_status','k1_score_status'])

def stat(rs,ds,scope):
 n=len(rs);b=sum(not r['k1_correct'] and r['dynamic_correct'] for r in rs);c=sum(r['k1_correct'] and not r['dynamic_correct'] for r in rs);d=b+c
 base=sum(r['k1_correct'] for r in rs);dyn=sum(r['dynamic_correct'] for r in rs)
 return dict(dataset=ds,scope=scope,n=n,k1_correct=base,dynamic_correct=dyn,k1_acc=base/n if n else None,dynamic_acc=dyn/n if n else None,delta_pp=100*(dyn-base)/n if n else None,wrong_to_right=b,right_to_wrong=c,both_right=sum(r['k1_correct'] and r['dynamic_correct']for r in rs),both_wrong=sum(not r['k1_correct'] and not r['dynamic_correct']for r in rs),timeout_either=sum(timedout(r)for r in rs),mcnemar_exact_p=min(1,2*sum(math.comb(d,i) for i in range(min(b,c)+1))/2**d) if d else 1)
results=[]
for scope in ['extended','extended_no_timeout','not_extended']:
 for ds in order+['ALL']:
  rr=[r for r in rows if (ds=='ALL' or r['dataset']==ds) and (not r['extended'] if scope=='not_extended' else r['extended']) and (scope!='extended_no_timeout' or not timedout(r))]
  results.append(stat(rr,ds,scope))
primary=[r for r in results if r['scope']=='extended' and r['dataset']!='ALL'];last=0
for i,r in enumerate(sorted(primary,key=lambda r:r['mcnemar_exact_p'])):
 last=max(last,min(1,(8-i)*r['mcnemar_exact_p']));r['holm_p_8_datasets']=last
# Cross-check all-question baseline/dynamic counts against previously reported table.
compare={r['dataset']:r for r in csv.DictReader((P.parent/'comparison.csv').open())}
for ds in order:
 rr=[r for r in rows if r['dataset']==ds];assert abs(sum(r['k1_correct'] for r in rr)/len(rr)-float(compare[ds]['fixed_k1_acc']))<1e-12
 assert sum(r['dynamic_correct'] for r in rr)==int(compare[ds]['correct'])
(P/'summary.json').write_text(json.dumps(results,indent=2))
with (P/'paired.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fields=list(results[0]);fields+=['holm_p_8_datasets'] if 'holm_p_8_datasets' not in fields else []
with (P/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
for scope in ['extended','extended_no_timeout','not_extended']:
 print(scope)
 for r in results:
  if r['scope']==scope:print(r['dataset'],r['n'],r['k1_correct'],r['dynamic_correct'],round(r['delta_pp'],3),r['wrong_to_right'],r['right_to_wrong'],'timeout',r['timeout_either'],'p',round(r['mcnemar_exact_p'],4))
