import os,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from ablation_core import Settings,rollout,atomic_json
from resample import build_prompt
from run import Backend
from queue_rpc import RPC
from deadline import Budget
slot=Path(os.environ['OPD_ONLINE_RUN']);P=Path(os.environ['GRID_ROOT'])
row=json.loads((P/'prompts.jsonl').open().readline())
def smoke(i):
 q=slot/'queue'/f'replica_{i}';models={r:json.loads((q/r/'model.json').read_text()) for r in ['student','teacher']}
 tok=AutoTokenizer.from_pretrained(models['student']['model'],local_files_only=True)
 b=Backend(RPC(q,Budget(slot/'smoke'/f'rollout_timer{i}.json',300)),models,tok)
 # Force a trigger only in this isolated smoke to exercise the new path on every replica.
 original=b.candidates
 def forced(prefix,ids,threshold):
  original(prefix,ids,threshold)
  return [dict(response_position=0,student_token_id=ids[0],signed_logprob_gap_student_minus_teacher=3.)]
 b.candidates=forced
 x=rollout(row,Settings(mode='weak_resample',teacher_tokens=1,max_takeovers=1,max_tokens=4,seed=20260912),b)
 events=[e for e in x['events'] if 'bridge_token_ids' in e]
 assert x['status']=='COMPLETE' and len(events)==1
 e=events[0];assert len(e['bridge_token_ids'])==1 and e['bridge_token_ids'][0]!=e['student_entry_token']
 assert x['token_sources'][0]=='student_resample'
 ids=build_prompt(tok,row,[],e['draft_token_ids'],0)
 atomic_json(slot/'smoke'/f'prompt{i}.json',dict(token_ids=ids,text=tok.decode(ids)))
 atomic_json(slot/'smoke'/f'rollout{i}.json',x);print('RESAMPLE_ROLLOUT_SMOKE_PASSED',i,len(x['response_token_ids']),flush=True)
with ThreadPoolExecutor(max_workers=int(os.environ['GRID_REPLICAS'])) as pool:list(pool.map(smoke,range(int(os.environ['REPLICA_START']),int(os.environ['REPLICA_START'])+int(os.environ['GRID_REPLICAS']))))
