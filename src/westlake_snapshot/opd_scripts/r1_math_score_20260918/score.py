import collections,concurrent.futures,fcntl,hashlib,importlib.util,json,time
from pathlib import Path
import judge_client
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_rescore/r1_math_score_20260918';S=Path(__file__).parent
import sys
sys.path.insert(0,str(R/'opd_scripts/random_quartile_k1_20260918'))
spec=importlib.util.spec_from_file_location('paired_scorer',R/'opd_scripts/random_quartile_k1_20260918/score.py');paired=importlib.util.module_from_spec(spec);spec.loader.exec_module(paired)
from ablation_core import atomic_json,digest

def main():
 lock=(P/'scoring.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows=list(map(json.loads,(R/'opd_eval_ablation/random_quartile_k1_20260918/prompts.jsonl').read_text().splitlines()))
 rows=[x for x in rows if x['dataset'] in ['bbh','math500','olympiad_bench']];assert len(rows)==2254
 prompt=(P/'judge/system_prompt.txt').read_text();assert prompt==(R/'opd_eval_ablation/random_quartile_k1_20260918/judge/system_prompt.txt').read_text()
 client=judge_client.Client(json.loads((P/'judge/relay.json').read_text())['port'],32,prompt)
 signature=digest(dict(model='gpt-4o-mini',prompt=prompt,code={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in S.glob('*.py')},extractor=hashlib.sha256((R/'opd_scripts/random_quartile_k1_20260918/score.py').read_bytes()).hexdigest()))
 inputs=[]
 for row in rows:
  f=Path(row['quartile_plan']['source']);raw=f.read_bytes();x=json.loads(raw)
  assert hashlib.sha256(raw).hexdigest()==row['quartile_plan']['source_sha256']
  assert x['status']=='COMPLETE' and x['problem_id']==row['problem_id'] and x['prompt_token_ids']==row['prompt_token_ids']
  m=json.loads((f.parent.parent/'manifest.json').read_text())['settings'];assert m['seed']==20260912 and m['teacher_tokens']==1 and m['gap_threshold']==2
  predicted=paired.candidate(x);reference=x['metadata']['final_answer'][0] if x['dataset']=='olympiad_bench' else x['solution']
  assert predicted is None or isinstance(predicted,str)
  assert isinstance(reference,str) and reference.strip()
  inputs.append(dict(source=str(f),source_sha256=hashlib.sha256(raw).hexdigest(),dataset=x['dataset'],problem_id=x['problem_id'],dataset_index=x['dataset_index'],finish_reason=x['finish_reason'],predicted=predicted,reference=reference))
 atomic_json(P/'inputs.json',inputs)
 def action(r):
  f=Path(r['source']);dest=f.parent.parent/'scores'/f.name
  if dest.exists():
   old=json.loads(dest.read_text());assert old['source_sha256']==r['source_sha256'] and old['scoring_signature']==signature
   if old['correct'] is not None:return old
  user=json.dumps(dict(candidate_answer=r['predicted'],reference_answer=r['reference']),ensure_ascii=False)
  result=client.call(user) if r['predicted'] and r['predicted'].strip() else dict(status='extraction_missing',is_correct=False,attempts=0,reason='empty extracted answer')
  result.update(r,answer_correct=result['is_correct'],correct=(False if r['finish_reason']=='trace_timeout' else result['is_correct']) if result['is_correct'] is not None else None,input_sha256=hashlib.sha256(user.encode()).hexdigest(),scoring_signature=signature,completed_at=time.time())
  atomic_json(dest,result);return result
 pending=inputs
 for attempt in range(3):
  failed=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
   futures={pool.submit(action,r):r for r in pending}
   for n,f in enumerate(concurrent.futures.as_completed(futures),1):
    x=f.result()
    if x['correct'] is None:failed.append(futures[f])
    if n%100==0:print('PROGRESS',attempt,n,len(pending),'errors',len(failed),flush=True)
  if not failed:break
  pending=failed
 assert not failed,'Unresolved API failures; resume supported'
 groups=collections.defaultdict(collections.Counter)
 for r in inputs:
  f=Path(r['source']);s=json.loads((f.parent.parent/'scores'/f.name).read_text());assert type(s['correct']) is bool and s['source_sha256']==r['source_sha256'] and s['scoring_signature']==signature
  g=groups[r['dataset']];g['n']+=1;g['correct']+=s['correct'];g['answer_correct']+=s['answer_correct'];g['timeouts']+=r['finish_reason']=='trace_timeout'
 summary=[dict(dataset=ds,**g,accuracy=g['correct']/g['n'],answer_accuracy=g['answer_correct']/g['n']) for ds,g in sorted(groups.items())]
 atomic_json(P/'summary.json',dict(rows=2254,groups=summary,scoring_signature=signature))
 atomic_json(P/'ALL_SCORING_COMPLETE.json',dict(rows=2254,completed_at=time.time()))
 (P/'judge/relay.stop').touch();print('ALL_SCORING_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
