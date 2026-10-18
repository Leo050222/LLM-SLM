import os,json,time,collections,statistics
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');out={}
for name,rel,j in [('code','opd_eval_code/code_grid_x8_20260916',104980),('math','opd_eval_hard/math_reasoning_grid_x8_20260916',104982)]:
 p=R/rel;s=p/'slots'/f'job_{j}';groups={};ts=[]
 for c in json.loads((p/'configs.json').read_text()):
  d=p/c['name']/'completed'
  if d.exists():
   mt=[x.stat().st_mtime for x in os.scandir(d) if x.name.endswith('.json')];groups[c['name']]=len(mt);ts+=mt
 now=time.time(); e={'timestamp':now,'completed':len(ts),'groups':groups,'last5m':sum(t>=now-300 for t in ts),'last10m':sum(t>=now-600 for t in ts),'queues':[]}
 log=(s/'logs/dispatch.log').read_text();e['errors']=log.count('TRACE_ERROR');e['timeout_completed']=sum(x.startswith('TRACE_DONE') and 'trace_timeout' in x for x in log.splitlines())
 for i in range(4):
  for role in ['student','teacher']:
   b=s/'queue'/f'replica_{i}'/role; pending=[]
   for f in (b/'requests').glob('*.json'):
    try:
     st=f.stat();v=json.loads(f.read_text());pending.append({'id':f.stem,'mtime':st.st_mtime,'op':v['op'],'prefix_tokens':len(v['prefix'])})
    except FileNotFoundError:pass
   t=time.time()
   for x in pending:x['age']=t-x.pop('mtime')
   e['queues'].append({'pair':i,'role':role,'time':t,'pending':len(pending),'oldest':sorted(pending,key=lambda x:-x['age'])[:3]})
 out[name]=e
print(json.dumps(out))
