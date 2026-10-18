import collections,concurrent.futures,csv,fcntl,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from ablation_core import atomic_json,digest
from code_scorer import sandbox,extract
import judge_client
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_prefix/prefix_takeover_caps11_20_20260917';S=Path(__file__).parent
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
 assert (P/'ALL_GENERATION_COMPLETE.json').exists()
 lock=(P/'scoring.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 code_only='--code-only' in sys.argv;prompt=(P/'judge/system_prompt.txt').read_text();client=None
 if not code_only:
  assert (P/'judge/EXTERNAL_SCORING_APPROVED.json').exists()
  port=json.loads((P/'judge/relay.json').read_text())['port'];client=judge_client.Client(port,32,prompt)
 sig=digest({'judge_model':'gpt-4o-mini','prompt':prompt,'files':{n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['score.py','code_scorer.py','score_one.py','judge_client.py']}})
 data={}
 for ds,name,key in [('cruxeval_o','cruxeval.jsonl','id'),('cruxeval_i','cruxeval.jsonl','id'),('humaneval_plus','HumanEvalPlus.jsonl','task_id'),('mbpp_plus','MbppPlus.jsonl','task_id'),('livecodebench','livecodebench_v6_incremental.jsonl','question_id')]:
  items=map(json.loads,(R/'opd_eval_code/code_reasoning_grid_20260913/data'/name).read_text().splitlines())
  data[ds]={(x['platform']+'/'+x[key] if ds=='livecodebench' else x[key]):x for x in items}
 rows=list(map(json.loads,(P/'prompts.jsonl').read_text().splitlines()));code=[];math=[]
 for rep in range(8):
  for cap in range(11,21):
   for row in rows:
    f=P/f'cap{cap}_r{rep}'/'completed'/f"{row['dataset_index']:05d}.json"
    (code if row['dataset'] in data else math).append(f)
 assert len(code)==40000 and len(math)==24000
 def action(f):
  raw=f.read_bytes();source=hashlib.sha256(raw).hexdigest();dest=f.parent.parent/'scores'/f.name
  if dest.exists():
   old=json.loads(dest.read_text());assert old['source_sha256']==source and old['scoring_signature']==sig
   if old['status'] not in ['api_error','harness_error']:return old
  x=json.loads(raw);ds=x['dataset']
  if ds in data:
   try:result=sandbox(dict(dataset=ds,problem=data[ds][x['problem_id']],solution=extract(ds,x['response'])))
   except Exception as e:result={'correct':None,'status':'harness_error','error_type':type(e).__name__}
  else:
   predicted=candidate(x);reference=x['metadata']['final_answer'][0] if ds=='olympiad_bench' else x['solution']
   assert predicted is None or isinstance(predicted,str)
   assert isinstance(reference,str) and reference.strip()
   user=json.dumps({'candidate_answer':predicted,'reference_answer':reference},ensure_ascii=False)
   if predicted is None or not predicted.strip():result={'status':'extraction_missing','is_correct':False,'reason':'empty extracted answer','attempts':0}
   else:result=client.call(user)
   result.update(correct=result['is_correct'],predicted=predicted,reference=reference,input_sha256=hashlib.sha256(user.encode()).hexdigest())
  result.update(dataset=ds,problem_id=x['problem_id'],sample_index=x['dataset_index'],cap=x['cap'],replicate=x['replicate'],actual_takeovers=x['actual_takeovers'],source_sha256=source,scoring_signature=sig,completed_at=time.time())
  atomic_json(dest,result);return result
 # Separate executor limits prevent code harnesses from starving API throughput.
 def group(files,workers,label):
  pending=files
  for attempt in range(3):
   failed=[]
   with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futures={pool.submit(action,f):f for f in pending}
    for n,future in enumerate(concurrent.futures.as_completed(futures),1):
     result=future.result()
     if result['correct'] is None:failed.append(futures[future])
     if n%100==0:print('SCORE_PROGRESS',label,attempt,n,len(pending),'errors',len(failed),flush=True)
   if not failed:return
   pending=failed
  raise RuntimeError(f'Unresolved {label} errors: {len(pending)}; resumable')
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(group,code,16,'code')]
  if not code_only:futures.append(pool.submit(group,math,32,'math'))
  for f in futures:f.result()
 groups=collections.defaultdict(collections.Counter);usage=collections.Counter()
 for f in code+([] if code_only else math):
  x=json.loads((f.parent.parent/'scores'/f.name).read_text());assert type(x['correct']) is bool
  g=groups[(x['dataset'],x['cap'])];g['n']+=1;g['correct']+=x['correct'];g['actual_takeovers_total']+=x['actual_takeovers']
  for k,v in x.get('usage',{}).items():
   if isinstance(v,int):usage[k]+=v
 summary=[dict(dataset=ds,cap=cap,**g,accuracy=g['correct']/g['n'],mean_actual_takeovers=g['actual_takeovers_total']/g['n']) for (ds,cap),g in sorted(groups.items())]
 assert len(summary)==(50 if code_only else 80) and all(x['n']==800 for x in summary)
 if code_only:
  atomic_json(P/'code_summary.json',{'rows':40000,'groups':summary,'scoring_signature':sig});atomic_json(P/'ALL_CODE_SCORING_COMPLETE.json',{'rows':40000,'completed_at':time.time()});print('CODE_SCORING_COMPLETE',flush=True);return
 atomic_json(P/'summary.json',{'rows':64000,'groups':summary,'usage':dict(usage),'scoring_signature':sig})
 with (P/'summary.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
 atomic_json(P/'ALL_SCORING_COMPLETE.json',{'rows':64000,'completed_at':time.time()});(P/'judge/relay.stop').touch()
 print('ALL_SCORING_COMPLETE',64000,flush=True)
if __name__=='__main__':main()
