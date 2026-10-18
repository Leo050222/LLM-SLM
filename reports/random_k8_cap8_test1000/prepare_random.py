import os,sys,json
from pathlib import Path
sys.path.insert(0,os.environ['OPD_CODE_ROOT']+'/ocrd_pipeline')
from online_rollout.core import digest
from online_rollout.random_policy import sample_positions
root=Path(os.environ['OPD_ONLINE_RUN']);base=root.parent
rows=[json.loads(l) for l in (base/'online_chunk64_k32_test1000_h100_seed20260911/prompts1000.jsonl').read_text().splitlines()]
baseline={r['dataset_index']:r for l in (base/'ge6_test1000_native_nonthinking_seed20260811/qwen3_1_7b/merged/rollouts.jsonl').open() for r in [json.loads(l)]}
for row in rows:
 b=baseline[row['dataset_index']]
 assert row['prompt_token_ids']==b['prompt_token_ids'] and str(row['problem_id'])==str(b['problem_id'])
 length=len(b['response_token_ids'])
 row['random_schedule_horizon']=length
 row['random_positions']=sample_positions(length,8,8,int(digest(['random_positions',20260911,row['dataset_index']])[:16],16))
(root/'prompts1000.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
print('RANDOM_SCHEDULE_READY',len(rows),sum(len(r['random_positions']) for r in rows))
