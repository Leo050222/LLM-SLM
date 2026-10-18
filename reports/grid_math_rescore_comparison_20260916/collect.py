import collections,csv,json,sys
from pathlib import Path
ROOT=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
NEW=ROOT/'opd_rescore/grid_math_gpt4o_mini_openlux_20260916'
OLD=ROOT/'opd_eval_hard/nojudge_grid_20260912/scores'
assert json.loads((NEW/'COMPLETE.json').read_text())['rows']==56350
counts=collections.defaultdict(collections.Counter)
examples=collections.defaultdict(list)
paired=[]
with (NEW/'inputs.jsonl').open() as f:
 for i,line in enumerate(f,1):
  x=json.loads(line);filename=Path(x['relative_path']).name
  old=json.loads((OLD/x['source_config']/filename).read_text())
  new=json.loads((NEW/'scores'/x['dataset']/x['config']/filename).read_text())
  assert old['source_sha256']==new['source_sha256']==x['source_sha256']
  assert new['input_sha256']==x['input_sha256']
  assert old['problem_id']==new['problem_id']==x['problem_id']
  assert type(old['correct']) is bool and type(new['is_correct']) is bool
  assert new['status'] in ['ok','extraction_missing']
  a=old['correct'];b=new['is_correct'];key=(x['dataset'],x['config']);c=counts[key]
  c['n']+=1;c['old_correct']+=a;c['new_correct']+=b
  change='both_correct' if a and b else 'wrong_to_right' if b else 'right_to_wrong' if a else 'both_wrong'
  c[change]+=1
  c['old_status_'+old['status']]+=1;c['new_status_'+new['status']]+=1
  timeout=x.get('finish_reason')=='trace_timeout'
  if timeout:c['trace_timeouts']+=1;c['timeout_new_correct']+=b
  paired.append(dict(dataset=x['dataset'],config=x['config'],problem_id=x['problem_id'],old_correct=a,new_correct=b,finish_reason=x.get('finish_reason')))
  if a!=b and len(examples[(x['dataset'],change)])<8:
   examples[(x['dataset'],change)].append(dict(config=x['config'],problem_id=x['problem_id'],candidate_answer=x['candidate_answer'],reference_answer=x['reference_answer'],reason=new['reason'],old_status=old['status'],finish_reason=x.get('finish_reason'),old_score_path=str(OLD/x['source_config']/filename),new_score_path=str(NEW/'scores'/x['dataset']/x['config']/filename)))
  if i%10000==0:print('COMPARED',i,file=sys.stderr,flush=True)
assert len(paired)==56350
print(json.dumps(dict(rows=[dict(dataset=d,config=g,**c) for (d,g),c in sorted(counts.items())],paired=paired,examples={d+'/'+change:v for (d,change),v in examples.items()},verification=dict(rows=56350,source_hashes_matched=True,input_hashes_matched=True,problem_ids_matched=True)),ensure_ascii=False))
