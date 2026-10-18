import json,collections,hashlib,concurrent.futures,sys,time
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');G=R/'opd_eval_hard/math_reasoning_grid_x8_20260916';B=R/'opd_eval_baselines/full_teacher_student_x8_20260918'
load=lambda p:json.loads(p.read_bytes())
marker=load(G/'ALL_SCORING_COMPLETE.json');assert marker['rows']==450800
F=Path(marker['first_round_scores']);A=Path(marker['additional_scores']);summary=load(A/'summary.json');bs=load(B/'summary.json')
rows=list(map(json.loads,(G/'prompts.jsonl').read_text().splitlines()));brows=list(map(json.loads,(B/'math_prompts.jsonl').read_text().splitlines()));assert rows==brows
rowmap={x['dataset_index']:x for x in rows};assert len(rowmap)==2254
configs=[dict(name='r0_'+a['config'],base_config=a['config'],replicate_index=0,source_dir=a['path'],kind='old') for a in load(G/'replicate0_audit.json')]
configs += [dict(c,source_dir=str(G/c['name']),kind='grid') for c in load(G/'configs.json')]
configs += [dict(name=c['name'],base_config=c['role']+'_only',replicate_index=c['replicate'],source_dir=str(B/c['name']),kind='baseline') for c in load(B/'configs.json') if c['family']=='math']
assert len(configs)==216
expected={(r['config'],r['dataset']):r for r in summary['groups']}
def scan(c):
 d=Path(c['source_dir']);files=list((d/'completed').glob('*.json'));assert len(files)==2254
 groups=collections.defaultdict(collections.Counter);question=[];sigs=collections.Counter();statuses=collections.Counter()
 for f in files:
  raw=f.read_bytes();x=json.loads(raw);ds=x['dataset'];row=rowmap[x['dataset_index']];assert x['problem_id']==row['problem_id'] and x['prompt_token_ids']==row['prompt_token_ids'] and x['status']=='COMPLETE'
  cfg=c['base_config'].replace('p','.');timeout=x.get('finish_reason')=='trace_timeout'
  sf=F/'scores'/ds/cfg/f.name if c['kind']=='old' else d/'scores'/f.name
  y=load(sf);assert y['source_sha256']==hashlib.sha256(raw).hexdigest() and y['problem_id']==x['problem_id'] and y['dataset']==ds
  if c['kind']=='old':
   answer=y['is_correct'];correct=answer and not timeout;sigs[y['judge_signature']]+=1
  else:
   correct=y['correct'];answer=y.get('answer_correct',correct);sig=y['scoring_signature'];sigs[sig]+=1
   assert sig==bs['scoring_signature'] if c['kind']=='baseline' else sig in [summary['scoring_signature'],summary.get('repair_signature')]
   assert not timeout or not correct
  assert type(correct) is bool and type(answer) is bool
  sources=x['token_sources'];n=len(x['response_token_ids']);assert len(sources)==n
  teacher=sources.count('teacher');events=sum(bool(e.get('bridge_token_ids')) for e in x['events'])
  if c['kind']=='baseline':assert events==0 and teacher==(n if c['base_config']=='teacher_only' else 0)
  else:assert teacher==sum(len(e.get('bridge_token_ids',[])) for e in x['events'])
  z=groups[ds];z.update(n=1,correct=int(correct),answer_correct=int(answer),interventions=events,teacher_tokens=teacher,output_tokens=n,generation_timeouts=int(timeout),timeout_answer_correct=int(timeout and answer),length_stops=int(x.get('finish_reason')=='length'))
  statuses[y.get('status','unspecified')]+=1
  question.append([ds,x['problem_id'],int(correct)])
 for ds,z in groups.items():
  if c['kind']=='grid':
   e=expected[c['name'],ds];assert z['n']==e['n'] and z['correct']==e['correct']
 return dict(config=c['base_config'].replace('p','.'),replicate=c['replicate_index'],kind=c['kind'],groups=dict(groups),questions=question,signatures=dict(sigs),statuses=dict(statuses))
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for result in pool.map(scan,configs):
  results.append(result)
  if len(results)%8==0:print('Verified groups',len(results),'/216',file=sys.stderr,flush=True)
assert sum(z['n'] for g in results for z in g['groups'].values())==486864
for r in bs['combined']:
 if r['dataset'] not in ['math500','olympiad_bench','bbh']:continue
 gs=[g['groups'][r['dataset']] for g in results if g['config']==r['role']+'_only'];assert sum(g['correct'] for g in gs)==r['correct']
print(json.dumps(dict(completion=marker,baseline_rows=36064,grid_rows=450800,verified_rows=486864,groups=results,repair=load(A/'REPAIR.json') if (A/'REPAIR.json').exists() else None,created_at=time.time())))
