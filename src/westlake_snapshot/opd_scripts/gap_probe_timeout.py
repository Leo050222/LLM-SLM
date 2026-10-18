from pathlib import Path
import json,hashlib
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation')
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
for name,job,idx in [('weak_reprompt_t2_k1_20260920',107873,0),('weak_resample_t2_k1_20260920',107874,19)]:
 p=R/name;d=p/('r1_'+name.removesuffix('_20260920'));x=json.loads((d/'completed'/f'{idx:05d}.json').read_text());e=x['events'][-1];offset=e['response_offset'];prefix=x['prompt_token_ids']+x['response_token_ids'][:offset];tokens=x['response_token_ids'][offset:offset+e['draft_tokens']];found={}
 for b in (p/'slots'/f'job_{job}'/'queue').glob('replica_*'):
  for role in ['student','teacher']:
   model=json.loads((b/role/'model.json').read_text());payload=dict(model_identity=digest(model),op='score',prefix=prefix,tokens=tokens);key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();f=b/role/'results'/(key+'.json')
   if f.exists():found[role]=json.loads(f.read_text())
 if len(found)==2:
  gaps=[a-b for a,b in zip(found['student']['chosen'],found['teacher']['chosen'])]
  print(json.dumps(dict(job=job,id=idx,last_round_gap_min=min(gaps),last_round_gap_max=max(gaps),student_first_logp=found['student']['chosen'][0],teacher_first_logp=found['teacher']['chosen'][0],trigger_count=sum(g>2 for g in gaps))))
 else:print(job,'cached_scores_not_found',list(found))
