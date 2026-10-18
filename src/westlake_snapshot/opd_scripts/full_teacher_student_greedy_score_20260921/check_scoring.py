from pathlib import Path
import json,collections,hashlib
r=Path('/storage/yukaichengLab/wangxin/test_eiloeo');result={}
for label,name in [('greedy','full_teacher_student_greedy_20260920'),('temp1','full_teacher_student_x8_20260918')]:
 p=r/'opd_eval_baselines'/name;summary=json.loads((p/'summary.json').read_text());status=collections.defaultdict(collections.Counter);counts=collections.Counter();correct=collections.Counter()
 for c in json.loads((p/'configs.json').read_text()):
  for f in (p/c['name']/'scores').glob('*.json'):
   x=json.loads(f.read_text());assert x['scoring_signature']==summary['scoring_signature'];assert type(x['correct']) is bool
   key=x['dataset']+'/'+c['role'];status[key][x['status'] if isinstance(x['status'],str) else 'execution_details']+=1;counts[key]+=1;correct[key]+=x['correct']
 for g in summary['combined']:
  k=g['dataset']+'/'+g['role'];assert (counts[k],correct[k])==(g['n'],g['correct'])
 s=r/'opd_scripts'/('full_teacher_student_greedy_score_20260921' if label=='greedy' else name)
 files={n:hashlib.sha256((s/n).read_bytes()).hexdigest() for n in ['score.py','code_scorer.py','score_one.py','judge_client.py','system_prompt.txt']}
 result[label]=dict(statuses={k:dict(v) for k,v in status.items()},files=files,score_rows=sum(counts.values()),verified_summary=True)
print(json.dumps(result))
