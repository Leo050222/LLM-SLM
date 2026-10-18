import os,json,hashlib,collections,csv,sys,subprocess,fcntl,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ablation_core import atomic_json,digest
from code_scorer import sandbox,extract
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=Path(os.environ.get('GRID_ROOT',str(R/'opd_eval_prefix/quartile_support_k_20260918')));S=Path(__file__).parent

def main():
 assert (P/'ALL_GENERATION_COMPLETE.json').exists()
 lock=(P/'scoring.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 data={}
 for ds,name,key in [('cruxeval_o','cruxeval.jsonl','id'),('cruxeval_i','cruxeval.jsonl','id'),('humaneval_plus','HumanEvalPlus.jsonl','task_id'),('mbpp_plus','MbppPlus.jsonl','task_id'),('livecodebench','livecodebench_v6_incremental.jsonl','question_id')]:
  items=map(json.loads,(R/'opd_eval_code/code_reasoning_grid_20260913/data'/name).read_text().splitlines());data[ds]={(x['platform']+'/'+x[key] if ds=='livecodebench' else x[key]):x for x in items}
 sig=digest({n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['score.py','math_rules.py','code_scorer.py','score_one.py']})
 files=[f for k in [1,4,8,16,32] for f in sorted((P/f'k{k}/completed').glob('*.json'))];assert len(files)==4000
 def action(f):
  raw=f.read_bytes();source=hashlib.sha256(raw).hexdigest();x=json.loads(raw);dest=f.parent.parent/'scores'/f.name
  if dest.exists():
   old=json.loads(dest.read_text());assert old['source_sha256']==source and old['scoring_signature']==sig
   if type(old['correct']) is bool:return old
  try:
   if x['dataset'] in data:
    result=sandbox(dict(dataset=x['dataset'],problem=data[x['dataset']][x['problem_id']],solution=extract(x['dataset'],x['response'])))
   else:
    # Bound symbolic grading without conflating infrastructure failures with incorrect answers.
    out=subprocess.run([sys.executable,str(S/'score.py'),'--math-one'],input=json.dumps(x),capture_output=True,text=True,timeout=120,check=True)
    line=next(l[12:] for l in out.stdout.splitlines() if l.startswith('MATH_RESULT='));result=json.loads(line)
   assert type(result['correct']) is bool
  except Exception as e:result=dict(correct=None,status='harness_error',error=repr(e)[:500])
  result.update(dataset=x['dataset'],problem_id=x['problem_id'],sample_index=x['dataset_index'],k=x['k'],quartile=x['quartile'],source_sha256=source,scoring_signature=sig,finish_reason=x['finish_reason'],actual_teacher_tokens=x['actual_teacher_tokens'])
  atomic_json(dest,result);return result
 pending=files
 for attempt in range(3):
  with ThreadPoolExecutor(max_workers=16) as pool:results=list(pool.map(action,pending))
  pending=[f for f,r in zip(pending,results) if r['correct'] is None]
  print('SCORING_PASS',attempt,'unresolved',len(pending),flush=True)
  if not pending:break
 assert not pending,'Unresolved scoring errors; resume after repair'
 groups=collections.defaultdict(list);paired=collections.defaultdict(dict)
 for f in files:
  x=json.loads((f.parent.parent/'scores'/f.name).read_text());groups[x['dataset'],x['k'],0].append(x);groups[x['dataset'],x['k'],x['quartile']].append(x);paired[x['dataset'],x['sample_index']][x['k']]=x
 summary=[]
 for (ds,k,q),rs in sorted(groups.items()):
  assert len(rs)==(100 if q==0 else 25)
  summary.append(dict(dataset=ds,k=k,quartile=q or 'all',n=len(rs),correct=sum(r['correct'] for r in rs),accuracy=sum(r['correct'] for r in rs)/len(rs),teacher_early_stop=sum(r['finish_reason']=='teacher_stop' for r in rs),length_limit=sum(r['finish_reason']=='length_limit' for r in rs)))
 assert len(summary)==200
 changes=[]
 for ds in sorted({key[0] for key in paired}):
  for k in [4,8,16,32]:
   pairs=[v for (d,i),v in paired.items() if d==ds]
   changes.append(dict(dataset=ds,k=k,baseline_k=1,n=100,wrong_to_correct=sum(not v[1]['correct'] and v[k]['correct'] for v in pairs),correct_to_wrong=sum(v[1]['correct'] and not v[k]['correct'] for v in pairs)))
 atomic_json(P/'summary.json',dict(scoring='existing local benchmark code tests and rule-based math/reasoning graders; no external LLM judge',rows=4000,groups=summary,paired_changes=changes,scoring_signature=sig))
 with (P/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 atomic_json(P/'ALL_SCORING_COMPLETE.json',dict(rows=4000,completed_at=time.time()));print('ALL_SCORING_COMPLETE',flush=True)
if __name__=='__main__':
 if '--math-one' in sys.argv:
  from math_rules import score_answer
  x=json.load(sys.stdin);correct,pred=score_answer(x);print('MATH_RESULT='+json.dumps(dict(correct=bool(correct),predicted=pred,status='rule_graded')))
 else:main()
