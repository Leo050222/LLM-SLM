import json,hashlib,time
from pathlib import Path
O=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/threshold2_luna_k1_20260911')
rows=[json.loads(l) for l in (O/'prompts.jsonl').read_text().splitlines()]
assert len(rows)==2254
manifest=json.loads((O/'online/manifest.json').read_text())
assert manifest['settings']['teacher_tokens']==1 and manifest['settings']['gap_threshold']==2
assert manifest['expected_rows']==2254
assert len(list((O/'online/completed').glob('*.json')))==2254
for row in rows:
 r=json.loads((O/'online/completed'/f"{row['dataset_index']:05d}.json").read_text())
 assert r['problem_id']==row['problem_id'] and r['prompt_token_ids']==row['prompt_token_ids']
 assert r['status']=='COMPLETE'
p=O/'ALL_GENERATION_COMPLETE.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(rows=2254,completed_at=time.time())));tmp.replace(p)
