import os,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ablation_core import digest,atomic_json
from queue_rpc import RPC
from deadline import Budget
def run(i):
 slot=Path(os.environ['OPD_ONLINE_RUN']);p=Path(os.environ['GRID_ROOT']);q=slot/'queue'/f'replica_{i}';role=os.environ['BASELINE_ROLE']
 for _ in range(720):
  if (q/role/'model.json').exists():break
  time.sleep(2)
 else:raise RuntimeError('Model not ready')
 model=json.loads((q/role/'model.json').read_text());row=json.loads((p/'prompts.jsonl').open().readline());prefix=row['prompt_token_ids']+row['anchor']['response_prefix_token_ids']
 rpc=RPC(q,Budget(slot/'smoke'/f'timer{i}.json',600))
 def call(seed):return rpc.call(role,dict(op='generate',model_identity=digest(model),prefix=prefix,n=8,temperature=0.,seed=seed))
 a=call(1);b=call(9876);assert a['token_ids']==b['token_ids']
 atomic_json(slot/'smoke'/f'student{i}.json',dict(passed=True));print('SERVICE_SMOKE_PASSED',i,flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(run,range(8)))
