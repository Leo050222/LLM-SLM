"""Matched teacher scoring only, warm prefix, batch one. No end-to-end claim."""
import os,json,sys,statistics,time
from pathlib import Path
sys.path.insert(0,os.environ['OPD_CODE_ROOT']+'/ocrd_pipeline')
from online_rollout.core import FileRPC,digest,atomic_json
root=Path(os.environ['OPD_KV_RUN']); role='teacher'
models={m:json.loads((root/m/role/'model.json').read_text()) for m in ['cached','reference']}
rpcs={m:FileRPC(root/m,timeout=600) for m in models}
def score(mode,prefix,tokens):
 return rpcs[mode].call(role,dict(model_identity=digest(models[mode]),op='score',prefix=prefix,tokens=tokens,benchmark_nonce=time.time_ns()))
rows=[]
for length in [1024,4096,8192]:
 prefix=([151644,8948,198]+[100,200,300,400,500,600,700,800]*1100)[:length]
 for mode in models:score(mode,prefix,[901,902,903])
 prefix += [901,902,903]
 data={m:[] for m in models}
 for step in range(8):
  tokens=list(range(1000+step*64,1064+step*64))
  for mode in (['cached','reference'] if step%2 else ['reference','cached']):
   out=score(mode,prefix,tokens)
   assert out['worker_batch_size']==1,'Concurrent work invalidates timing'
   data[mode].append(dict(seconds=out['worker_batch_seconds']+out.get('boundary_repair_seconds',0),
      computed=out['computed_prompt_tokens']+sum(x['computed_prompt_tokens'] for x in out.get('boundary_repairs',[])),
      cached=out['num_cached_tokens'],sidecar=out['logprob_sidecar_hit']))
  prefix+=tokens
 row=dict(initial_prefix=length,chunk=64,rounds=8,measurements=data)
 row['score_gpu_time_ratio']=sum(x['seconds'] for x in data['reference'])/sum(x['seconds'] for x in data['cached'])
 rows.append(row)
 print(json.dumps(dict(initial_prefix=length,chunk=64,score_gpu_time_ratio=row['score_gpu_time_ratio'],cached_computed=sum(x['computed'] for x in data['cached']),reference_computed=sum(x['computed'] for x in data['reference']))),flush=True)
atomic_json(root/'benchmark.json',dict(scope='warm teacher scoring only, synthetic sequences, batch1; excludes student/judge/bridge latency',rows=rows))
