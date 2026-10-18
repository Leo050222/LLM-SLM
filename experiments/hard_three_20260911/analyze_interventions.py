from pathlib import Path
import json,collections,csv
import numpy as np
P=Path(__file__).parent;A=P/'artifacts';rows=[]
for f in sorted((A/'online/completed').glob('*.json')):
 v=json.loads(f.read_text());s=json.loads((A/'scores/online'/f.name).read_text());base=json.loads((A/'scores/student'/f.name).read_text());teacher=json.loads((A/'teacher/completed'/f.name).read_text());student=json.loads((A/'student/completed'/f.name).read_text())
 rows.append(dict(dataset=v['dataset'],index=v['dataset_index'],takeovers=sum('bridge_token_ids' in e for e in v['events']),tokens=len(v['response_token_ids']),teacher_tokens=v['token_sources'].count('teacher'),student_only_tokens=len(student['response_token_ids']),teacher_only_tokens=len(teacher['response_token_ids']),correct=int(s['correct']),baseline_correct=int(base['correct'])))
with (P/'intervention_rows.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def corr(x,y):
 return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
def describe(rr):
 x=np.array([r['takeovers'] for r in rr]);y=np.array([r['correct'] for r in rr]);length=np.array([r['tokens'] for r in rr]);out=dict(n=len(rr),mean_tokens=float(length.mean()),median_tokens=float(np.median(length)),teacher_tokens=np.mean([r['teacher_tokens'] for r in rr]),student_only_tokens=np.mean([r['student_only_tokens'] for r in rr]),teacher_only_tokens=np.mean([r['teacher_only_tokens'] for r in rr]),mean_takeovers=float(x.mean()),correct_n=int(y.sum()),correct_takeovers=float(x[y==1].mean()),wrong_takeovers=float(x[y==0].mean()),point_biserial=corr(x,y),takeovers_length_r=corr(x,np.log1p(length)))
 # Descriptive partial correlation, linear residualization on log output length.
 z=np.column_stack([np.ones(len(rr)),np.log1p(length)]);out['partial_r_loglength']=corr(x-np.einsum('ij,j->i',z,np.linalg.lstsq(z,x,rcond=None)[0]),y-np.einsum('ij,j->i',z,np.linalg.lstsq(z,y,rcond=None)[0]))
 rng=np.random.default_rng(20260911);cc=[]
 for _ in range(2000):
  ids=rng.integers(0,len(rr),len(rr));c=corr(x[ids],y[ids]);
  if c is not None:cc.append(c)
 out['r_bootstrap95']=np.percentile(cc,[2.5,97.5]).tolist()
 out['bins']={}
 for name,lo,hi in [('0',0,0),('1-2',1,2),('3-5',3,5),('6-10',6,10),('11+',11,9999)]:
  q=[r for r in rr if lo<=r['takeovers']<=hi]
  if q:out['bins'][name]=dict(n=len(q),correct=sum(r['correct'] for r in q),accuracy=np.mean([r['correct'] for r in q]),rescued=sum(r['correct'] and not r['baseline_correct'] for r in q),lost=sum(not r['correct'] and r['baseline_correct'] for r in q))
 return out
out={ds:describe([r for r in rows if r['dataset']==ds]) for ds in ['aime24','aime25','olympiad_bench','bbh']};out['all']=describe(rows)
x=np.array([r['takeovers'] for r in rows]);y=np.array([r['correct'] for r in rows]);z=np.column_stack([np.ones(len(rows)),np.log1p([r['tokens'] for r in rows])]+[np.array([r['dataset']==ds for r in rows]) for ds in ['aime24','aime25','olympiad_bench']]);out['all']['partial_r_dataset_loglength']=corr(x-np.einsum('ij,j->i',z,np.linalg.lstsq(z,x,rcond=None)[0]),y-np.einsum('ij,j->i',z,np.linalg.lstsq(z,y,rcond=None)[0]))
(P/'intervention_analysis.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
