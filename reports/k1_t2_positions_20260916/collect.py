from pathlib import Path
import json,hashlib
root=Path('/storage/yukaichengLab/wangxin/test_eiloeo');rows=[]
for rel,g in [('opd_eval_hard/nojudge_grid_20260912','math'),('opd_eval_code/code_reasoning_grid_20260913','code')]:
 r=root/rel
 for p in sorted((r/'k1_t2/completed').glob('*.json')):
  b=p.read_bytes();x=json.loads(b);sp=r/'scores/k1_t2'/p.name if g=='math' else r/'k1_t2/scores'/p.name;s=json.loads(sp.read_text());assert hashlib.sha256(b).hexdigest()==s['source_sha256']
  pos=[i for i,v in enumerate(x['token_sources'])if v=='teacher'];ev=[e for e in x.get('events',[])if 'bridge_token_ids'in e];assert all(len(e['bridge_token_ids'])==1 for e in ev)
  rows.append(dict(dataset=x['dataset'],problem_id=x['problem_id'],length=len(x['response_token_ids']),source_length=len(x['token_sources']),positions=pos,event_positions=[e.get('anchor_response_position')for e in ev],finish=x['finish_reason'],correct=bool(s['correct'])))
print(json.dumps(rows))
