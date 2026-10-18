import json,csv
from pathlib import Path
import numpy as np
p=Path('/Users/leolee/Documents/OPD/outputs/benchmark-viz')
d=json.loads((p/'data.json').read_text());rows=d['rows'][2:]
def rank(a):
 a=np.array(a);return np.array([np.sum(a<v)+(np.sum(a==v)+1)/2 for v in a])
def corr(x,y):
 if np.ptp(x)==0 or np.ptp(y)==0:return None
 return float(np.corrcoef(x,y)[0,1])
result=[]
for j,b in enumerate(d['bench']):
 x=np.array([r['v'][j][1] for r in rows]);y=np.array([r['v'][j][0] for r in rows])
 byk={str(k):corr([r['v'][j][1] for r in rows if r['k']==k],[r['v'][j][0] for r in rows if r['k']==k]) for k in [1,4,8,16,32]}
 byt={str(t):corr([r['v'][j][1] for r in rows if r['t']==t],[r['v'][j][0] for r in rows if r['t']==t]) for t in [2,2.5,3,3.5,4]}
 # Remove k-specific means (within-k pooled correlation).
 xx=x.copy();yy=y.copy()
 for k in [1,4,8,16,32]:
  m=np.array([r['k']==k for r in rows]);xx[m]-=x[m].mean();yy[m]-=y[m].mean()
 o=dict(benchmark=b,pearson=corr(x,y),spearman=corr(rank(x),rank(y)),within_k=corr(xx,yy),by_k=byk,by_threshold=byt,interventions_range=[min(x),max(x)],accuracy_range=[min(y),max(y)])
 result.append(o)
 print(json.dumps(o))
(p/'interventions-accuracy-correlation.json').write_text(json.dumps(result,indent=2))
with (p/'interventions-accuracy-correlation.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['benchmark','pearson','spearman','within_k',*[f'k{k}' for k in [1,4,8,16,32]]])
 for o in result:w.writerow([o['benchmark'],o['pearson'],o['spearman'],o['within_k'],*o['by_k'].values()])
