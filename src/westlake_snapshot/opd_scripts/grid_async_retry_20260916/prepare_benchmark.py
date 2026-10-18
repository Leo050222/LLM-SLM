import json,hashlib
from pathlib import Path
from collections import defaultdict
from queue_rpc import atomic
p=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/math_reasoning_grid_x8_20260916');groups=defaultdict(list)
for row in map(json.loads,(p/'prompts.jsonl').read_text().splitlines()):groups[row['dataset']].append(row)
rows=[]
for dataset,problems in sorted(groups.items()):
 for pos in [0,len(problems)//5,2*len(problems)//5,3*len(problems)//5,4*len(problems)//5]:
  row=problems[pos];idx=row['dataset_index'];x=json.loads((p/'r1_k1_t2/completed'/f'{idx:05d}.json').read_text())
  assert x['problem_id']==row['problem_id'] and x['status']=='COMPLETE'
  tokens=x['response_token_ids']
  if len(tokens)<8 or x.get('finish_reason')=='trace_timeout':continue
  offset=min(len(tokens)-8,[0,128,512,1024][len(rows)%4])
  rows.append({'dataset':dataset,'dataset_index':idx,'prefix':row['prompt_token_ids']+tokens[:offset],'tokens':tokens[offset:offset+8]})
assert len(rows)>=8
atomic(p/'recovery_async_retry_v3/benchmark_inputs.json',rows)
print('BENCHMARK_INPUTS_VERIFIED',len(rows),[(r['dataset'],r['dataset_index']) for r in rows],flush=True)
