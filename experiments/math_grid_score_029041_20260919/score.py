import collections,concurrent.futures,fcntl,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import judge_client
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');G=R/'opd_eval_hard/math_reasoning_grid_x8_20260916';P=R/'opd_rescore/math_grid_score_029041_20260919';S=Path(__file__).parent
OLD=R/'opd_scripts/r1_math_score_20260918';EX=R/'opd_scripts/random_quartile_k1_20260918/score.py'
sys.path.insert(0,str(EX.parent))
spec=importlib.util.spec_from_file_location('paired_scorer',EX);paired=importlib.util.module_from_spec(spec);spec.loader.exec_module(paired)
from ablation_core import atomic_json,digest

def main():
 lock=(G/'scoring.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (P/'judge/EXTERNAL_SCORING_APPROVED.json').exists()
 ownership=json.loads((G/'recovery_colocated22_20260918/migration.json').read_text())['ownership']
 configs=[c for c in json.loads((G/'configs.json').read_text()) if ownership[c['name']] in (0,1)]
 assert len(configs)==145 and all((G/c['name']/'VERIFIED.json').exists() for c in configs)
 rows=list(map(json.loads,(G/'prompts.jsonl').read_text().splitlines()));assert len(rows)==2254
 prompt=(P/'judge/system_prompt.txt').read_text();assert prompt==(OLD/'system_prompt.txt').read_text()
 # Preserve the exact existing r1 scoring protocol/signature for compatible reuse.
 signature=digest(dict(model='gpt-4o-mini',prompt=prompt,code={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OLD.glob('*.py')},extractor=hashlib.sha256(EX.read_bytes()).hexdigest()))
 atomic_json(P/'RUN_CONFIG.json',dict(rows=326830,configs=[c['name'] for c in configs],shards=[0,1],workers=256,scoring_signature=signature,implementation_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in S.glob('*.py')},job_id=os.getenv('SLURM_JOB_ID'),started_at=time.time()))
 if '--validate-only' in sys.argv:
  for row in rows:
   f=G/'r1_k1_t2'/'completed'/f"{row['dataset_index']:05d}.json";raw=f.read_bytes();old=json.loads((f.parent.parent/'scores'/f.name).read_text());assert old['source_sha256']==hashlib.sha256(raw).hexdigest() and old['scoring_signature']==signature and type(old['correct']) is bool
  print('VALIDATED145_CONFIGS_AND2254_REUSABLE_SCORES',flush=True);return
 client=judge_client.Client(json.loads((P/'judge/relay.json').read_text())['port'],256,prompt)
 started=time.time();processed=0;reused=0;summaries=[]
 for c in configs:
  group=c['name'];d=G/group;manifest=json.loads((d/'manifest.json').read_text());settings=manifest['settings'];assert settings['seed']==c['seed'] and settings['teacher_tokens']==c['k'] and settings['gap_threshold']==float(c['threshold'])
  def action(row):
   f=d/'completed'/f"{row['dataset_index']:05d}.json";raw=f.read_bytes();x=json.loads(raw);source=hashlib.sha256(raw).hexdigest();dest=d/'scores'/f.name
   assert x['status']=='COMPLETE' and x['identity']==digest(dict(row=row,settings=settings)) and x['problem_id']==row['problem_id'] and x['prompt_token_ids']==row['prompt_token_ids']
   if dest.exists():
    old=json.loads(dest.read_text());assert old['source_sha256']==source and old['scoring_signature']==signature
    if type(old['correct']) is bool:return old,True
   predicted=paired.candidate(x);reference=x['metadata']['final_answer'][0] if x['dataset']=='olympiad_bench' else x['solution'];assert predicted is None or isinstance(predicted,str);assert isinstance(reference,str) and reference.strip()
   user=json.dumps(dict(candidate_answer=predicted,reference_answer=reference),ensure_ascii=False)
   result=client.call(user) if predicted and predicted.strip() else dict(status='extraction_missing',is_correct=False,attempts=0,reason='empty extracted answer')
   result.update(source=str(f),source_sha256=source,dataset=x['dataset'],problem_id=x['problem_id'],dataset_index=x['dataset_index'],finish_reason=x['finish_reason'],predicted=predicted,reference=reference,answer_correct=result['is_correct'],correct=(False if x['finish_reason']=='trace_timeout' else result['is_correct']) if result['is_correct'] is not None else None,input_sha256=hashlib.sha256(user.encode()).hexdigest(),scoring_signature=signature,completed_at=time.time())
   atomic_json(dest,result);return result,False
  pending=rows
  for attempt in range(3):
   failed=[]
   with concurrent.futures.ThreadPoolExecutor(max_workers=256) as pool:
    futures={pool.submit(action,row):row for row in pending}
    for future in concurrent.futures.as_completed(futures):
     result,was_reused=future.result()
     if result['correct'] is None:failed.append(futures[future])
     else:processed+=1;reused+=was_reused
     if processed%100==0:
      z=dict(processed=processed,total=326830,reused=reused,group=group,attempt=attempt,unresolved_this_pass=len(failed),active_limit=client.gate.limit,elapsed_seconds=round(time.time()-started,1),updated_at=time.time());atomic_json(P/'progress.json',z);print('PROGRESS',json.dumps(z),flush=True)
   if not failed:break
   pending=failed
  assert not failed,'Unresolved API failures; rerun resumes saved scores'
  counts=collections.defaultdict(collections.Counter)
  for row in rows:
   f=d/'scores'/f"{row['dataset_index']:05d}.json";v=json.loads(f.read_text());assert type(v['correct']) is bool and v['scoring_signature']==signature
   z=counts[v['dataset']];z['n']+=1;z['correct']+=v['correct'];z['answer_correct']+=v['answer_correct'];z['timeouts']+=v['finish_reason']=='trace_timeout'
  summary=dict(config=group,rows=2254,groups=[dict(dataset=ds,**z,accuracy=z['correct']/z['n'],answer_accuracy=z['answer_correct']/z['n']) for ds,z in sorted(counts.items())],scoring_signature=signature)
  summaries.append(summary);atomic_json(P/'groups'/f'{group}.json',summary);atomic_json(P/'summary.json',dict(rows=2254*len(summaries),total=326830,groups=summaries));print('GROUP_COMPLETE',group,flush=True)
 assert processed==326830
 atomic_json(P/'ALL_SCORING_COMPLETE.json',dict(rows=processed,reused=reused,completed_at=time.time(),scoring_signature=signature))
 (P/'judge/relay.stop').touch();print('ALL_SCORING_COMPLETE',processed,flush=True)
if __name__=='__main__':main()
