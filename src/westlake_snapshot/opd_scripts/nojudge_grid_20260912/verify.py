import json,sys,time
from pathlib import Path
from ablation_core import atomic_json
P=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/nojudge_grid_20260912')
configs=json.loads((P/'configs.json').read_text());rows=[json.loads(l) for l in (P/'prompts.jsonl').read_text().splitlines()]
assert len(rows)==2254 and all(r['dataset'] in ('math500','olympiad_bench','bbh') for r in rows)
def verify(c):
 p=P/c['name'];m=json.loads((p/'manifest.json').read_text());s=m['settings']
 assert s['teacher_tokens']==c['k'] and s['gap_threshold']==c['threshold'] and s['max_takeovers'] is None and s['first_token_ban']
 assert s['gap_rule']=='student_minus_teacher_gt_positive_threshold' and m['judge_config'] is None
 assert len(list((p/'completed').glob('*.json')))==2254
 for row in rows:
  r=json.loads((p/'completed'/f"{row['dataset_index']:05d}.json").read_text());assert r['status']=='COMPLETE' and r['judge_calls']==0
  assert r['prompt_token_ids']==row['prompt_token_ids'] and r['problem_id']==row['problem_id']
  for e in r['events']:
   assert not e['judgments']
   if 'bridge_token_ids' in e:
    assert 0<len(e['bridge_token_ids'])<=c['k'] and e['bridge_token_ids'][0]!=e['student_entry_token']
    assert e['selected_candidate']['signed_logprob_gap_student_minus_teacher']>c['threshold']
 atomic_json(p/'VERIFIED.json',dict(rows=2254,verified_at=time.time()))
if sys.argv[1]=='all':
 for c in configs:verify(c)
 atomic_json(P/'ALL_GENERATION_COMPLETE.json',dict(rows=56350,configs=25,completed_at=time.time()))
else:verify(next(c for c in configs if c['name']==sys.argv[1]))
