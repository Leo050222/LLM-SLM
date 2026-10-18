import json,hashlib,collections
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P=R/'opd_eval_ablation/random_quartile_k1_20260918'
rows=list(map(json.loads,(P/'prompts.jsonl').read_text().splitlines()))
base={};old={};r1={};out=[]
for rel,mode in [('opd_eval_hard/hard_three_20260911','student'),('opd_eval_math500/math500_three_20260911','student'),('opd_eval_code/code_reasoning_grid_20260913','student_only')]:
 root=R/rel
 for f in (root/mode/'completed').glob('*.json'):
  x=json.loads(f.read_text());ds=x.get('dataset','math500')
  if ds.startswith('aime'):continue
  pid=x['problem_id'] if ds!='math500' else 'math500/'+x['problem_id']
  base[ds,pid]=(f,x)
for rel in ['opd_eval_hard/nojudge_grid_20260912','opd_eval_code/code_reasoning_grid_20260913']:
 root=R/rel
 for f in (root/'k1_t2/completed').glob('*.json'):
  x=json.loads(f.read_text());old[x['dataset'],x['problem_id']]=(f,x)
mathscores={}
for rel,config,label in [('baseline_math_gpt4o_mini_openlux_20260916','student_only','student'),('grid_math_gpt4o_mini_openlux_20260916','k1_t2','threshold_r0')]:
 root=R/'opd_rescore'/rel
 for f in (root/'scores').glob(f'*/{config}/*.json'):
  s=json.loads(f.read_text());mathscores[label,s['dataset'],s['problem_id']]=s
# Validate identical external judge instruction across comparison groups.
prompt=(P/'judge/system_prompt.txt').read_bytes()
for rel in ['baseline_math_gpt4o_mini_openlux_20260916','grid_math_gpt4o_mini_openlux_20260916']:
 assert (R/'opd_rescore'/rel/'system_prompt.txt').read_bytes()==prompt
for row in rows:
 key=(row['dataset'],row['problem_id']);record=dict(dataset=key[0],problem_id=key[1])
 for label,source in [('student',base),('threshold_r0',old)]:
  f,x=source[key];assert x['prompt_token_ids']==row['prompt_token_ids']
  if key[0] in ['bbh','math500','olympiad_bench']:
   # Legacy MATH500 baseline score IDs may be unprefixed.
   s=mathscores.get((label,*key)) or mathscores[(label,key[0],key[1].removeprefix('math500/'))]
   correct=s['is_correct']
  else:
   s=json.loads((f.parent.parent/'scores'/f.name).read_text());correct=s['correct']
  assert type(correct) is bool
  assert hashlib.sha256(f.read_bytes()).hexdigest()==s['source_sha256']
  record[label]=dict(correct=correct,policy_correct=correct and x['finish_reason']!='trace_timeout',finish=x['finish_reason'],source=str(f))
 f=P/'r1_random_quartile_k1/completed'/f"{row['dataset_index']:05d}.json";x=json.loads(f.read_text());s=json.loads((f.parent.parent/'scores'/f.name).read_text())
 assert hashlib.sha256(f.read_bytes()).hexdigest()==s['source_sha256']
 record['random']=dict(correct=s['answer_correct'],policy_correct=s['correct'],finish=x['finish_reason'],actual=x['intervention_audit']['actual_total'],planned=x['intervention_audit']['planned_total'])
 f=Path(row['quartile_plan']['source']);assert hashlib.sha256(f.read_bytes()).hexdigest()==row['quartile_plan']['source_sha256']
 sf=f.parent.parent/'scores'/f.name
 if sf.exists():
  s=json.loads(sf.read_text());assert s['source_sha256']==row['quartile_plan']['source_sha256'];record['threshold_r1']=dict(correct=s['correct'])
 out.append(record)
print(json.dumps(dict(rows=out,random_summary=json.loads((P/'summary.json').read_text()),complete=json.loads((P/'ALL_SCORING_COMPLETE.json').read_text()),audit=dict(paired_prompts=len(out),r1_math_scores_available=sum('threshold_r1' in x and x['dataset'] in ['bbh','math500','olympiad_bench'] for x in out),same_judge_prompt=True))))
