"""Fixed trace-derived RPC replay; two replicas per mode, fresh request keys."""
import os,json,time,math,statistics,concurrent.futures
from pathlib import Path
from queue_rpc import RPC,atomic
from deadline import Budget
import sys
sys.path.insert(0,'/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/nojudge_grid_20260912')
from ablation_core import digest
P=Path(os.environ['GRID_ROOT']);S=Path(os.environ['OPD_ONLINE_RUN']);Q=S/'queue'
for _ in range(900):
 if len(list(Q.glob('replica_*/*/model.json')))==8:break
 time.sleep(2)
else:raise RuntimeError('Benchmark models not ready')
models=[{r:json.loads((Q/f'replica_{i}'/r/'model.json').read_text()) for r in ['student','teacher']} for i in range(4)]
rows=json.loads((P/'recovery_async_retry_v3/benchmark_inputs.json').read_text())
assert len(rows)>=8
atomic(S/'benchmark_inputs.json',rows)
def call(i,role,payload):
 rpc=RPC(Q/f'replica_{i}',Budget(S/'bench_timers'/f'{i}_{time.time_ns()}.json',180))
 return rpc.call(role,dict(model_identity=digest(models[i][role]),**payload))
# Prewarm with requests separate from timed requests.
def warm(i):
 for r in ['student','teacher']:call(i,r,dict(op='generate',prefix=rows[0]['prefix'],n=1,seed=88991))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(warm,range(4)))
results=[]
for trial in range(3):
 # Fresh seeds & token prefixes each trial prevent RPC-result-cache speedups.
 payloads=[]
 for j,row in enumerate(rows):
  prefix=row['prefix']+([row['tokens'][0]]*trial)
  payloads.extend([dict(role=r,op='score',prefix=prefix,tokens=row['tokens']) for r in ['student','teacher']])
  payloads.append(dict(role='student',op='generate',prefix=prefix,n=[1,8,32,64][j%4],seed=1731+trial*100+j))
  payloads.append(dict(role='teacher',op='generate',prefix=prefix,n=1,seed=3731+trial*100+j,banned_first=row['tokens'][0]))
 def replay(i):
  start=time.monotonic()
  with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
   futures=[pool.submit(call,i,x['role'],{k:v for k,v in x.items() if k!='role'}) for x in payloads]
   values=[f.result() for f in futures]
  return {'replica':i,'seconds':time.monotonic()-start,'values':values}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:run=list(pool.map(replay,range(4)))
 numeric=[];baseline=[];generation_matches=0;generation_total=0
 for j,req in enumerate(payloads):
  vals=[x['values'][j] for x in run]
  if req['op']=='score':
   assert all(len(v['chosen'])==len(req['tokens']) and all(math.isfinite(x) for x in v['chosen']) for v in vals)
   if req['role']=='teacher':assert all(all(len(ids)==16 for ids in v['top_ids']) for v in vals)
   baseline.append(max(abs(a-b) for a,b in zip(vals[0]['chosen'],vals[1]['chosen'])))
   for v in vals[2:]:numeric.append(max(abs(a-b) for a,b in zip(vals[0]['chosen'],v['chosen'])))
  else:
   for v in vals:
    assert 0<len(v['token_ids'])<=req['n']
    if 'banned_first' in req:assert v['token_ids'][0]!=req['banned_first']
   generation_total+=2;generation_matches+=sum(v['token_ids']==vals[0]['token_ids'] for v in vals[2:])
 # Exact gap-threshold decisions on the same forced tokens across both roles.
 decisions=True
 for j in range(0,len(payloads),4):
  gaps=[[a-b for a,b in zip(x['values'][j]['chosen'],x['values'][j+1]['chosen'])] for x in run]
  for threshold in [2,2.5,3,3.5,4]:
   decisions &= all([a>threshold for a in g]==[a>threshold for a in gaps[0]] for g in gaps[1:])
 sync=statistics.mean(x['seconds'] for x in run[:2]);asyn=statistics.mean(x['seconds'] for x in run[2:])
 result={'trial':trial,'seconds':[x['seconds'] for x in run],'speedup':sync/asyn,'max_score_delta':max(numeric),'baseline_score_delta':max(baseline),'threshold_decisions_equal':decisions,'generation_matches':generation_matches,'generation_comparisons':generation_total}
 results.append(result);atomic(S/f'trial{trial}.json',{'summary':result,'payloads':payloads,'runs':run});print('BENCHMARK',json.dumps(result),flush=True)
# Numerical tolerance is explicit and baseline variability is reported, not hidden.
passed=all(r['max_score_delta']<=1e-4 and r['threshold_decisions_equal'] for r in results) and statistics.median(r['speedup'] for r in results)>=1.05
atomic(S/'benchmark_summary.json',{'passed':passed,'trials':results,'note':'RPC replay, not end-to-end rollout throughput; seeds/batching may change sampled token outcomes.'})
if not passed:raise RuntimeError('Benchmark acceptance gate failed; retain FIFO production')
(S/'BENCHMARK_PASSED').touch()
