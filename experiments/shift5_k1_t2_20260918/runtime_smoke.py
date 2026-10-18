import os,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from ablation_core import Settings,rollout,atomic_json
from run import Backend
from queue_rpc import RPC
from deadline import Budget
slot=Path(os.environ['OPD_ONLINE_RUN']);P=Path(os.environ['GRID_ROOT'])
row=json.loads((P/'prompts.jsonl').open().readline())
def smoke(i):
 q=slot/'queue'/f'replica_{i}';models={r:json.loads((q/r/'model.json').read_text()) for r in ['student','teacher']}
 tok=AutoTokenizer.from_pretrained(models['student']['model'],local_files_only=True)
 b=Backend(RPC(q,Budget(slot/'smoke'/f'rollout_timer{i}.json',300)),models,tok)
 x=rollout(row,Settings(mode='shift5',teacher_tokens=1,max_takeovers=None,max_tokens=64,seed=20260912),b)
 assert x['status']=='COMPLETE' and len(x['response_token_ids'])==len(x['token_sources'])
 assert all(abs(e['actual_shift'])<=5 for e in x['events'] if 'bridge_token_ids' in e)
 atomic_json(slot/'smoke'/f'rollout{i}.json',x);print('SHIFT_ROLLOUT_SMOKE_PASSED',i,len(x['response_token_ids']),flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(smoke,range(4)))
