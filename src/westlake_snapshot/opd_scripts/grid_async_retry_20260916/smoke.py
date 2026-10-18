import os,json,time,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue_rpc import RPC,atomic
from deadline import Budget
import sys
sys.path.insert(0,'/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/nojudge_grid_20260912')
from ablation_core import digest

def run(i):
 slot=Path(os.environ['OPD_ONLINE_RUN']);p=Path(os.environ['GRID_ROOT']);q=slot/'queue'/f'replica_{i}'
 for _ in range(720):
  if all((q/r/'model.json').exists() for r in ['student','teacher']):break
  time.sleep(2)
 else:raise RuntimeError('Models not ready')
 models={r:json.loads((q/r/'model.json').read_text()) for r in ['student','teacher']}
 row=json.loads((p/'prompts.jsonl').open().readline());prefix=row['prompt_token_ids']
 rpc=RPC(q,Budget(slot/'smoke'/f'timer{i}.json',300))
 def call(role,**x):return rpc.call(role,dict(model_identity=digest(models[role]),**x))
 draft=call('student',op='generate',prefix=prefix,n=8,seed=731+i)
 assert 0<len(draft['token_ids'])<=8
 for role in ['student','teacher']:
  x=call(role,op='score',prefix=prefix,tokens=draft['token_ids'])
  assert len(x['chosen'])==len(draft['token_ids']) and all(math.isfinite(a) for a in x['chosen'])
  if role=='teacher':assert all(len(a)==16 for a in x['top_ids'])
 entry=draft['token_ids'][0]
 bridge=call('teacher',op='generate',prefix=prefix,n=1,seed=918+i,banned_first=entry)
 assert len(bridge['token_ids'])==1 and bridge['token_ids'][0]!=entry
 atomic(slot/'smoke'/f'pair{i}.json',{'passed':True,'roles':list(models)})
 print('SERVICE_SMOKE_PASSED',i,flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,range(4)))
