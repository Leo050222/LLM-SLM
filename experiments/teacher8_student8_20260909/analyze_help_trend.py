import json
from pathlib import Path
from collections import defaultdict
P=Path(__file__).resolve().parent
q=json.loads((P/'per_question.json').read_text());base=json.loads((P/'student_only_correctness.json').read_text())
b=defaultdict(int)
for r in base:b[str(r['problem_id'])]+=int(r['correct'])
hs=[32,64,128,256]
cs={h:[sum(r[f'h{h}_per_teacher_correct']) for r in q] for h in hs}
s=[b[str(r['problem_id'])] for r in q]
result={'vs_baseline':{},'by_baseline_correct':{},'adjacent':{}}
for h in hs:
 vals=cs[h]
 result['vs_baseline'][h]={'improved':sum(v>8*x for v,x in zip(vals,s)),'equal':sum(v==8*x for v,x in zip(vals,s)),'worse':sum(v<8*x for v,x in zip(vals,s)),'delta_pp':100*sum(v/64-x/8 for v,x in zip(vals,s))/1000}
for x in range(9):
 ids=[i for i,a in enumerate(s) if a==x]
 result['by_baseline_correct'][x]={'n':len(ids),'h':{h:{'accuracy':100*sum(cs[h][i] for i in ids)/(64*len(ids)),'at_least_one':sum(cs[h][i]>0 for i in ids),'at_least_half':sum(cs[h][i]>=32 for i in ids),'improved':sum(cs[h][i]>x*8 for i in ids),'worse':sum(cs[h][i]<x*8 for i in ids)} for h in hs}}
for a,z in zip(hs,hs[1:]):
 result['adjacent'][f'{a}->{z}']={'improved':sum(y>x for x,y in zip(cs[a],cs[z])),'equal':sum(y==x for x,y in zip(cs[a],cs[z])),'worse':sum(y<x for x,y in zip(cs[a],cs[z]))}
result['monotonic_non_decreasing']=sum(all(cs[a][i]<=cs[z][i] for a,z in zip(hs,hs[1:])) for i in range(1000))
result['monotonic_with_gain']=sum(all(cs[a][i]<=cs[z][i] for a,z in zip(hs,hs[1:])) and cs[256][i]>cs[32][i] for i in range(1000))
# Paired question bootstrap, avoids treating all 64 continuations as independent.
import numpy as np
rng=np.random.default_rng(20260910)
a=np.asarray(cs[32])/64;z=np.asarray(cs[256])/64
for label,mask in [('all',np.ones(1000,dtype=bool)),('baseline_zero',np.asarray(s)==0),('baseline_le2',np.asarray(s)<=2)]:
 d=(z-a)[mask];boots=np.mean(d[rng.integers(0,len(d),size=(10000,len(d)))],axis=1)*100
 result[label+'_h256_minus_h32']={'n':len(d),'delta_pp':float(d.mean()*100),'ci95':np.quantile(boots,[.025,.975]).tolist(),'improved':int((d>0).sum()),'equal':int((d==0).sum()),'worse':int((d<0).sum())}
(P/'help_trend.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
