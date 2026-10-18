import json,sys,time
from pathlib import Path
from ablation_core import atomic_json
O=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/takeover_ablation_20260911')
MODES=['luna_cap8','nojudge_cap8','luna_allow','random_allow']
rows=[json.loads(l) for l in (O/'prompts.jsonl').read_text().splitlines()]
assert len(rows)==2254 and all(r['dataset'] not in ('aime24','aime25') for r in rows)
def verify(mode):
 p=O/mode;manifest=json.loads((p/'manifest.json').read_text());s=manifest['settings']
 assert s['mode']==mode and s['teacher_tokens']==8 and s['gap_threshold']==2
 assert s['max_takeovers']==(8 if mode.endswith('cap8') else None)
 assert s['first_token_ban']==mode.endswith('cap8')
 assert len(list((p/'completed').glob('*.json')))==2254
 for row in rows:
  r=json.loads((p/'completed'/f"{row['dataset_index']:05d}.json").read_text())
  assert r['status']=='COMPLETE' and r['problem_id']==row['problem_id'] and r['prompt_token_ids']==row['prompt_token_ids']
  ev=[e for e in r['events'] if 'bridge_token_ids' in e]
  if mode.endswith('cap8'):assert len(ev)<=8
  if mode in ('nojudge_cap8','random_allow'):assert r['judge_calls']==0
  assert len(r['token_sources'])==len(r['response_token_ids'])
  for e in ev:
   assert 0<len(e['bridge_token_ids'])<=8
   if mode.endswith('cap8'):assert e['bridge_token_ids'][0]!=e['student_entry_token']
   if mode!='random_allow':assert e['selected_candidate']['signed_logprob_gap_student_minus_teacher']>2
   if mode.startswith('luna'):assert e['judgments'][-1]['judgment']['select_anchor']
 atomic_json(p/'VERIFIED.json',dict(rows=2254,verified_at=time.time()))
if sys.argv[1]=='all':
 for mode in MODES:verify(mode)
 atomic_json(O/'ALL_GENERATION_COMPLETE.json',dict(rows=9016,groups=4,completed_at=time.time()))
else:verify(sys.argv[1])
