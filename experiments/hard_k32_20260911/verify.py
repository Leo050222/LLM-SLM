import json,hashlib
from pathlib import Path
P=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/hard_k32_20260911')
rows=[json.loads(l) for l in (P/'prompts.jsonl').open()]
assert len(rows)==1814 and len(list((P/'online/completed').glob('*.json')))==1814
for r in rows:
 v=json.loads((P/'online/completed'/f"{r['dataset_index']:05d}.json").read_text())
 assert v['status']==('USER_ABORTED' if r['dataset_index'] in [436,900] else 'COMPLETE') and v['problem_id']==r['problem_id'] and v['prompt_token_ids']==r['prompt_token_ids'] and v['solution']==r['solution']
 assert len(v['token_sources'])==len(v['response_token_ids'])
 assert all(0<len(e['bridge_token_ids'])<=32 for e in v['events'] if 'bridge_token_ids' in e)
p=P/'ALL_GENERATION_COMPLETE.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(rows=1814,groups=1,completed=1812,user_aborted=[436,900])));t.replace(p)
