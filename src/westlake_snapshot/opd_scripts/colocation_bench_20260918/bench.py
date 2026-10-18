import os,sys,json,time,hashlib,queue,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from ablation_core import Settings,rollout,digest,atomic_json
from queue_rpc import RPC
from deadline import Budget
sys.path.append('/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/nojudge_grid_20260912')
from run import Backend,tokenizer_fingerprint

def main():
 from transformers import AutoTokenizer
 P=Path(os.environ['BENCH_ROOT']);slot=Path(os.environ['BENCH_SLOT']);pairs=int(os.environ['BENCH_PAIRS'])
 models=[{r:json.loads((slot/f'queue/replica_{i}'/r/'model.json').read_text()) for r in ['student','teacher']} for i in range(pairs)]
 assert all(m==models[0] for m in models)
 tok=AutoTokenizer.from_pretrained(models[0]['student']['model'],local_files_only=True)
 assert all(m['tokenizer_sha256']==tokenizer_fingerprint(tok) for m in models[0].values())
 raw=(P/'prompts.jsonl').read_bytes();meta=json.loads((P/'INPUTS_COMPLETE.json').read_text());assert hashlib.sha256(raw).hexdigest()==meta['sha256']
 rows=list(map(json.loads,raw.splitlines()));tasks=queue.Queue();errors=[]
 # Service warmup excluded from measured rollout interval; independent seeds.
 for i in range(pairs):
  b=Backend(RPC(slot/f'queue/replica_{i}',Budget(slot/f'warm{i}.json',600)),models[i],tok)
  d=b.call('student',op='generate',prefix=rows[0]['prompt_token_ids'],n=8,seed=1234)
  b.candidates(rows[0]['prompt_token_ids'],d['token_ids'],2)
  b.call('teacher',op='generate',prefix=rows[0]['prompt_token_ids'],n=8,seed=5678)
 # Synthetic repeated-token contexts test capacity only, not answer quality.
 # Excluded from all rollout-throughput timing and recorded separately.
 stress=[];unit=rows[0]['prompt_token_ids'][1:] or rows[0]['prompt_token_ids']
 for i in range(pairs):
  rpc=RPC(slot/f'queue/replica_{i}',Budget(slot/f'stress{i}.json',1800))
  for length in [8192,16384,32768,40896]:
   prefix=(unit*((length+len(unit)-1)//len(unit)))[:length]
   for role in ['student','teacher']:
    t=time.monotonic();out=rpc.call(role,dict(model_identity=digest(models[i][role]),op='generate',prefix=prefix,n=8,seed=901+length))
    assert 0<len(out['token_ids'])<=8
    stress.append(dict(pair=i,role=role,context_tokens=length,elapsed=time.monotonic()-t))
 atomic_json(slot/'CONTEXT_STRESS.json',dict(passed=True,synthetic=True,concurrency=1,not_in_rollout_metrics=True,results=stress))
 for r in rows:
  for k in meta['ks']:tasks.put((r,k))
 class Timed:
  def __init__(self,rpc):self.rpc=rpc;self.events=[]
  def call(self,role,payload):
   start=time.monotonic();v=self.rpc.call(role,payload)
   self.events.append(dict(role=role,op=payload['op'],context=len(payload['prefix']),wall=time.monotonic()-start,worker_batch_seconds=v.get('worker_batch_seconds'),worker_batch_size=v.get('worker_batch_size'),queue_wait_seconds=v.get('queue_wait_seconds')));return v
 started=time.time();atomic_json(slot/'MEASURE_START.json',dict(time=started))
 def consume(i):
  while not errors:
   try:r,k=tasks.get_nowait()
   except queue.Empty:return
   key=f"{r['dataset_index']:05d}_k{k}";t=time.monotonic();budget=Budget(slot/'timings'/f'{key}.json',3*3600);rpc=Timed(RPC(slot/f'queue/replica_{i}',budget));settings=Settings(mode='nojudge_grid',max_takeovers=None,teacher_tokens=k,gap_threshold=2,seed=meta['seed'])
   def save(s):budget.persist();atomic_json(slot/'traces'/f'{key}.json',s)
   try:
    result=rollout(r,settings,Backend(rpc,models[i],tok),None,None,save)
    assert result['status']=='COMPLETE' and result['judge_calls']==0 and result['identity']==digest(dict(row=r,settings=asdict(settings)))
    rec=dict(dataset=r['dataset'],index=r['dataset_index'],bin=r['profile_bin'],k=k,elapsed=time.monotonic()-t,tokens=len(result['response_token_ids']),finish_reason=result['finish_reason'],response_sha256=digest(result['response_token_ids']),calls=rpc.events)
    atomic_json(slot/'completed'/f'{key}.json',rec);print('DONE',key,round(rec['elapsed'],2),flush=True)
   except Exception as e:
    errors.append(dict(key=key,type=type(e).__name__,message=str(e)[:400]));atomic_json(slot/'ERRORS.json',errors);raise
 with ThreadPoolExecutor(max_workers=32*pairs) as pool:list(pool.map(consume,[i%pairs for i in range(32*pairs)]))
 elapsed=time.time()-started;results=[json.loads(f.read_text()) for f in (slot/'completed').glob('*.json')];assert len(results)==54 and not errors
 times=sorted(r['elapsed'] for r in results)
 summary=dict(n=54,elapsed=elapsed,rollouts_per_hour=54/elapsed*3600,response_tokens=sum(r['tokens'] for r in results),tokens_per_second=sum(r['tokens'] for r in results)/elapsed,median_latency=times[len(times)//2],p95_latency=times[int(.95*(len(times)-1))],measurement_end=time.time(),model_info=models[0],scientific_settings=dict(ks=meta['ks'],threshold=2,max_tokens=40264,seed=meta['seed']),outputs='no accuracy scoring; infrastructure timeout is failure, not incorrect answer')
 atomic_json(slot/'SUMMARY.json',summary);print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
