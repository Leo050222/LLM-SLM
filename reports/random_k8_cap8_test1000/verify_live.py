import os,sys,json
from pathlib import Path
sys.path.insert(0,os.environ['OPD_CODE_ROOT']+'/ocrd_pipeline')
from online_rollout.core import FileRPC,Settings
from online_rollout.run_random import Backend
from online_rollout.random_policy import random_rollout
from transformers import AutoTokenizer
r=Path(os.environ['OPD_ONLINE_RUN']);q=r/'queue/replica_0'
tok=AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B',local_files_only=True)
models={role:json.loads((q/role/'model.json').read_text()) for role in ['student','teacher']}
b=Backend(FileRPC(q),models,tok)
row=json.loads((r/'prompts1000.jsonl').read_text().splitlines()[0]);row['random_positions']=[0,8,16,24,32,40,48,56]
s=random_rollout(row,Settings(max_tokens=64),b)
es=[e for e in s['events'] if 'bridge_token_ids' in e]
assert 1<=len(es)<=8 and all(len(e['bridge_token_ids'])<=8 for e in es)
assert all(e['anchor_response_position']==i*8 for i,e in enumerate(es))
assert s['judge_calls']==0 and all(t=='teacher' for t in s['token_sources'])
print(json.dumps(dict(status='RANDOM_LIVE_OK',takeovers=len(es),teacher_tokens=len(s['response_token_ids']),finish_reason=s['finish_reason'])))
