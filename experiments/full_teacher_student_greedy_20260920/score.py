import collections,concurrent.futures,csv,fcntl,hashlib,json,sys,time,statistics
from pathlib import Path
from types import SimpleNamespace
from ablation_core import atomic_json,digest
from code_scorer import sandbox,extract
import judge_client
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_baselines/full_teacher_student_greedy_20260920';S=Path(__file__).parent
sys.path.insert(0,str(R/'eval_repos/evalscope'))
from evalscope.metrics.math.parser import extract_answer
from evalscope.benchmarks.bbh.bbh_adapter import BBHAdapter
from evalscope.benchmarks.olympiad_bench.olympiad_bench_adapter import OlympiadBenchAdapter

def candidate(x):
 response=x['response'].rsplit('</think>',1)[-1].strip();ds=x['dataset']
 if ds=='math500':return extract_answer(response)
 if ds=='bbh':return (BBHAdapter._extract_mc_answer if x['metadata']['multiple_choice'] else BBHAdapter._extract_ff_answer)(response)
 if ds=='olympiad_bench':return OlympiadBenchAdapter.extract_answer(None,response,SimpleNamespace(metadata=x['metadata']))
 raise ValueError(ds)

def main():
 assert json.loads((P/'ALL_GENERATION_COMPLETE.json').read_text())['rows']==9142
 lock=(P/'scoring.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 code_only='--code-only' in sys.argv;client=None;prompt=(P/'judge/system_prompt.txt').read_text()
 if not code_only:
  assert (P/'judge/EXTERNAL_SCORING_APPROVED.json').exists()
  client=judge_client.Client(json.loads((P/'judge/relay.json').read_text())['port'],32,prompt)
 data={};datahash={}
 for ds,name,key in [('cruxeval_i','cruxeval.jsonl','id'),('cruxeval_o','cruxeval.jsonl','id'),('humaneval_plus','HumanEvalPlus.jsonl','task_id'),('mbpp_plus','MbppPlus.jsonl','task_id'),('livecodebench','livecodebench_v6_incremental.jsonl','question_id')]:
  f=R/'opd_eval_code/code_reasoning_grid_20260913/data'/name;datahash[name]=hashlib.sha256(f.read_bytes()).hexdigest();items=map(json.loads,f.read_text().splitlines());data[ds]={(x['platform']+'/'+x[key] if ds=='livecodebench' else x[key]):x for x in items}
 sig=digest(dict(judge_model='gpt-4o-mini',prompt=prompt,data=datahash,files={n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['score.py','code_scorer.py','score_one.py','judge_client.py']},timeout_policy='source_trace_timeout_count_wrong'))
 problem=next(iter(data['cruxeval_i'].values()))
 assert sandbox(dict(dataset='cruxeval_i',problem=problem,solution=problem['input']))['correct']
 assert not sandbox(dict(dataset='cruxeval_i',problem=problem,solution='__opd_bad_input__'))['correct']
 atomic_json(P/'SCORING_SMOKE.json',dict(passed=True,signature=sig))
 configs=json.loads((P/'configs.json').read_text());files={'code':[],'math':[]}
 for c in configs:
  n=2317 if c['family']=='code' else 2254
  assert json.loads((P/c['name']/'VERIFIED.json').read_text())['rows']==n
  fs=list((P/c['name']/'completed').glob('*.json'));assert len(fs)==n
  files[c['family']].extend((c,f) for f in fs)
 assert len(files['code'])==4634 and len(files['math'])==4508
 def action(item):
  c,f=item;raw=f.read_bytes();h=hashlib.sha256(raw).hexdigest();dest=f.parent.parent/'scores'/f.name
  if dest.exists():
   x=json.loads(dest.read_text());assert x['source_sha256']==h and x['scoring_signature']==sig
   if type(x.get('correct')) is bool:return x
  x=json.loads(raw);ds=x['dataset'];assert x['status']=='COMPLETE'
  if x['finish_reason']=='trace_timeout':
   assert x['timeout_count_as_wrong'];result=dict(correct=False,status='trace_timeout')
  elif c['family']=='code':result=sandbox(dict(dataset=ds,problem=data[ds][x['problem_id']],solution=extract(ds,x['response'])))
  else:
   predicted=candidate(x);reference=x['metadata']['final_answer'][0] if ds=='olympiad_bench' else x['solution'];assert isinstance(reference,str) and reference.strip()
   if predicted is None or not predicted.strip():result=dict(correct=False,status='extraction_missing')
   else:
    result=client.call(json.dumps(dict(candidate_answer=predicted,reference_answer=reference),ensure_ascii=False))
    if result.get('status')=='api_error':raise RuntimeError('API error; do not count wrong')
    result['correct']=result['is_correct']
   result.update(predicted=predicted,reference=reference)
  assert type(result['correct']) is bool
  result.update(dataset=ds,problem_id=x['problem_id'],dataset_index=x['dataset_index'],role=c['role'],replicate=c['replicate'],source_sha256=h,scoring_signature=sig,completed_at=time.time())
  atomic_json(dest,result);return result
 def group(family,workers):
  pending=files[family]
  for attempt in range(3):
   failed=[]
   with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    ff={pool.submit(action,item):item for item in pending}
    for n,future in enumerate(concurrent.futures.as_completed(ff),1):
     try:future.result()
     except Exception:failed.append(ff[future])
     if n%500==0:print('SCORE_PROGRESS',family,attempt,n,len(pending),'errors',len(failed),flush=True)
   if not failed:return
   pending=failed
  atomic_json(P/f'{family}_SCORING_ERRORS.json',[dict(group=c['name'],file=f.name) for c,f in pending]);raise RuntimeError('Unresolved scoring errors')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  fs=[pool.submit(group,'code',16)]
  if not code_only:fs.append(pool.submit(group,'math',32))
  for f in fs:f.result()
 counts=collections.defaultdict(collections.Counter)
 for family in (['code'] if code_only else ['code','math']):
  for c,f in files[family]:
   x=json.loads((f.parent.parent/'scores'/f.name).read_text());assert type(x['correct']) is bool and x['scoring_signature']==sig
   g=counts[(x['dataset'],c['role'],c['replicate'])];g['n']+=1;g['correct']+=x['correct']
 groups=[dict(dataset=ds,role=role,replicate=rep,**v,accuracy=v['correct']/v['n']) for (ds,role,rep),v in sorted(counts.items())]
 assert len(groups)==(10 if code_only else 16)
 combined=[]
 for ds,role in sorted({(g['dataset'],g['role']) for g in groups}):
  gs=[g for g in groups if (g['dataset'],g['role'])==(ds,role)];assert len(gs)==1
  n=sum(g['n'] for g in gs);correct=sum(g['correct'] for g in gs);combined.append(dict(dataset=ds,role=role,n=n,correct=correct,accuracy=correct/n,replicate_accuracy_sd=0.0))
 expected=4634 if code_only else 9142;assert sum(g['n'] for g in groups)==expected
 stem='code_summary' if code_only else 'summary'
 atomic_json(P/f'{stem}.json',dict(rows=expected,groups=groups,combined=combined,scoring_signature=sig))
 with (P/f'{stem}.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
 atomic_json(P/('ALL_CODE_SCORING_COMPLETE.json' if code_only else 'ALL_SCORING_COMPLETE.json'),dict(rows=expected,completed_at=time.time()))
 if not code_only:(P/'judge/relay.stop').touch()
 print('SCORING_COMPLETE',expected,flush=True)
if __name__=='__main__':main()
