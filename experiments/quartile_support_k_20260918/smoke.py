import os,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ablation_core import digest,atomic_json
from queue_rpc import RPC
from deadline import Budget

def run(i):
 slot=Path(os.environ['OPD_ONLINE_RUN']);p=Path(os.environ['GRID_ROOT']);q=slot/'queue'/f'replica_{i}'
 for _ in range(720):
  if all((q/r/'model.json').exists() for r in ['student','teacher']):break
  time.sleep(2)
 else:raise RuntimeError('Models not ready')
 models={r:json.loads((q/r/'model.json').read_text()) for r in ['student','teacher']}
 assert models['student']['tokenizer_sha256']==models['teacher']['tokenizer_sha256']
 row=json.loads((p/'prompts.jsonl').open().readline());prefix=row['prompt_token_ids']
 rpc=RPC(q,Budget(slot/'smoke'/f'timer{i}.json',600))
 def call(role,n,temp,prefix,seed):return rpc.call(role,dict(op='generate',model_identity=digest(models[role]),prefix=prefix,n=n,temperature=temp,seed=seed))
 bridge=call('teacher',32,1.,prefix,731+i);assert 0<len(bridge['token_ids'])<=32
 a=call('student',8,0.,prefix+bridge['token_ids'][:1],1)
 b=call('student',8,0.,prefix+bridge['token_ids'][:1],9876)
 assert a['token_ids']==b['token_ids'],'Greedy seed invariance failed'
 atomic_json(slot/'smoke'/f'pair{i}.json',dict(passed=True,teacher_tokens=len(bridge['token_ids']),greedy_seed_invariance=True))
 print('SERVICE_SMOKE_PASSED',i,flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,range(4)))
