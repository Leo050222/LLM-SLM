import json,numpy as np
from pathlib import Path
p=Path('/Users/leolee/Documents/OPD/reports/k1_t2_length_vs_student_20260916')
rows=json.loads((p/'paired_evidence.json').read_text())['rows'];rng=np.random.default_rng(20260916)
def clean(r):return all(r[m]['finish']=='stop' and 'timeout' not in str(r[m]['score_status']).lower() for m in ['student','k1'])
def test(rs,mode='log',B=20000):
 groups=[]
 for ds in sorted(set(r['dataset']for r in rs)):
  a=[r for r in rs if r['dataset']==ds]
  x=np.array([np.log((r['student']['tokens']+1)/(r['k1']['tokens']+1)) if mode=='log' else float(r['k1']['tokens']<r['student']['tokens'])for r in a]); y=np.array([int(r['k1']['correct'])-int(r['student']['correct'])for r in a],float)
  groups.append((x-x.mean(),y-y.mean()))
 obs=sum(np.sum(x*y) for x,y in groups);den=np.sqrt(sum(np.sum(x*x) for x,y in groups)*sum(np.sum(y*y) for x,y in groups));hits=0
 for _ in range(B):
  perm=sum(np.sum(x*rng.permutation(y))for x,y in groups)
  hits+=abs(perm)>=abs(obs)-1e-12
 return {'n':len(rs),'dataset_centered_pearson_r':obs/den,'permutation_two_sided_p':(hits+1)/(B+1),'permutations':B}
out={}
for label,rs in [('all',rows),('both_normal',[r for r in rows if clean(r)])]:
 out[label]={'log_shortening':test(rs),'any_shortening':test(rs,'binary'),'groups':{}}
 for short in [True,False]:
  a=[r for r in rs if (r['k1']['tokens']<r['student']['tokens'])==short]
  out[label]['groups'][str(short)]={'n':len(a),'student_acc':sum(r['student']['correct']for r in a)/len(a),'k1_acc':sum(r['k1']['correct']for r in a)/len(a)}
out['per_dataset']={}
for ds in sorted(set(r['dataset']for r in rows)):out['per_dataset'][ds]=test([r for r in rows if r['dataset']==ds])
(p/'association.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
