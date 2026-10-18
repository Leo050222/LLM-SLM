import os,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue_rpc import RPC,atomic
from deadline import Budget
from ablation_core import digest

def run(i):
 role=os.environ['BASELINE_ROLE'];slot=Path(os.environ['OPD_ONLINE_RUN']);p=Path(os.environ['GRID_ROOT']);q=slot/'queue'/f'replica_{i}'
 for _ in range(720):
  if (q/role/'model.json').exists():break
  time.sleep(2)
 else:raise RuntimeError('Teacher not ready')
 model=json.loads((q/role/'model.json').read_text());row=json.loads((p/'code_prompts.jsonl').open().readline())
 out=RPC(q,Budget(slot/'smoke'/f'timer{i}.json',300)).call(role,dict(model_identity=digest(model),op='generate',prefix=row['prompt_token_ids'],n=8,seed=731+i))
 assert 0<len(out['token_ids'])<=8
 atomic(slot/'smoke'/f'teacher{i}.json',dict(passed=True));print('TEACHER_SMOKE_PASSED',i,flush=True)
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(run,range(8)))
