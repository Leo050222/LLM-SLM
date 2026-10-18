import time,json,os,heapq,statistics
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');out={}
def stats(v):
 v=sorted(v)
 return {'n':len(v),'mean':statistics.mean(v),'p95':v[min(len(v)-1,int(.95*len(v)))],'max':max(v)} if v else {}
for name,rel,job in [('code','opd_eval_code/code_grid_x8_20260916',104996),('math','opd_eval_hard/math_reasoning_grid_x8_20260916',104997)]:
 p=R/rel;s=p/'slots'/f'job_{job}';log=(s/'logs/dispatch.log').read_text();lines=log.splitlines();e={'time':time.time(),'new_done':sum(x.startswith('TRACE_DONE') for x in lines),'new_timeouts':sum(x.startswith('TRACE_DONE') and 'trace_timeout' in x for x in lines),'errors':[x for x in lines if x.startswith('TRACE_ERROR')],'workers':[]}
 for i in range(4):
  for role in ['student','teacher']:
   b=s/'queue'/f'replica_{i}'/role
   latest=heapq.nlargest(120,((f.stat().st_mtime,Path(f.path)) for f in os.scandir(b/'results')))
   values=[json.loads(f.read_text()) for _,f in latest]
   pending=[]
   for f in (b/'requests').glob('*.json'):
    try:pending.append(time.time()-f.stat().st_mtime)
    except FileNotFoundError:pass
   e['workers'].append({'pair':i,'role':role,'pending':len(pending),'oldest_pending':max(pending,default=0),'wait_seconds':stats([v['queue_wait_seconds'] for v in values if 'queue_wait_seconds' in v]),'batch_seconds':stats([v['worker_batch_seconds'] for v in values if 'worker_batch_seconds' in v]),'batch_size':stats([v['worker_batch_size'] for v in values if 'worker_batch_size' in v])})
 out[name]=e
print(json.dumps(out))
