import json,sys,signal,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter,defaultdict
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');O=R/'opd_eval_hard/threshold3_mathteacher_20260911'
sys.path.insert(0,str(R/'eval_repos/evalscope'))
def score_answer(r):
 response=r['response'].rsplit('</think>',1)[-1].strip();ds=r['dataset']
 if ds=='math500':
  from evalscope.metrics.math.parser import extract_answer,math_equal,strip_answer_string
  pred=extract_answer(response);return bool(math_equal(pred,strip_answer_string(r['solution']))),pred
 if ds=='bbh':
  from evalscope.benchmarks.bbh.bbh_adapter import BBHAdapter
  pred=BBHAdapter._extract_mc_answer(response) if r['metadata']['multiple_choice'] else BBHAdapter._extract_ff_answer(response)
  def canonical(x):
   return ''.join(x.split()) if r['subset']=='dyck_languages' else ' '.join(x.split()).casefold()
  return canonical(pred)==canonical(r['solution']),pred
 if ds in ['aime24','aime25']:
  from evalscope.metrics.math.parser import extract_answer
  from evalscope.benchmarks.aime.math_normalize import normalize_answer
  from evalscope.benchmarks.aime.grader import grade_answer
  pred=normalize_answer(extract_answer(response));return bool(grade_answer(pred,r['solution'])),pred
 if ds=='olympiad_bench':
  from evalscope.benchmarks.olympiad_bench.utils import MathJudger
  from evalscope.benchmarks.olympiad_bench.olympiad_bench_adapter import OlympiadBenchAdapter
  from types import SimpleNamespace
  m=r['metadata'];pred=OlympiadBenchAdapter.extract_answer(None,response,SimpleNamespace(metadata=m));precision=1e-8
  if 'Tuple' not in m['answer_type'] and m.get('error'):
   e=str(m['error']);precision=[float(x) if x else 1e-8 for x in e.split(',')] if ',' in e else float(e)
  return bool(MathJudger().judge(pred,m['final_answer'][0],precision)),pred
 raise ValueError(ds)
def one(arg):
 mode,i=arg;p=O/mode/'completed'/f'{i:05d}.json';r=json.loads(p.read_text());dest=O/'scores'/mode/p.name;digest=hashlib.sha256(p.read_bytes()).hexdigest()
 if dest.exists():
  s=json.loads(dest.read_text());assert s['source_sha256']==digest;return s
 def timeout(*_):raise TimeoutError()
 signal.signal(signal.SIGALRM,timeout);status='ok';correct=False;pred=None
 try:
  signal.alarm(30);correct,pred=score_answer(r)
  if not pred:status='missing'
 except TimeoutError:status='timeout'
 except Exception as e:status=type(e).__name__
 finally:signal.alarm(0)
 s=dict(group=mode,dataset=r['dataset'],subset=r['subset'],dataset_index=i,problem_id=r['problem_id'],correct=correct,predicted=pred,status=status,source_sha256=digest,tokens=len(r['response_token_ids']))
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(s));tmp.replace(dest);return s
if __name__=='__main__':
 assert json.loads((O/'ALL_GENERATION_COMPLETE.json').read_text())['rows']==2814
 assert len(list((O/'online/completed').glob('*.json')))==2314
 assert len(list((O/'teacher/completed').glob('*.json')))==500
 result=[]
 with ProcessPoolExecutor(max_workers=48) as pool:
  tasks=[('online',i) for i in range(2314)]+[('teacher',i) for i in range(1814,2314)]
  fs=[pool.submit(one,t) for t in tasks]
  for k,f in enumerate(as_completed(fs),1):
   result.append(f.result())
   if k%100==0:print('SCORED',k,2314,flush=True)
 groups=defaultdict(list)
 for r in result:groups[(r['group'],r['dataset'])].append(r)
 summary={};subsets={}
 for (mode,ds),rs in groups.items():
  summary.setdefault(mode,{})[ds]=dict(questions=len(rs),correct=sum(r['correct'] for r in rs),accuracy=sum(r['correct'] for r in rs)/len(rs),score_status=dict(Counter(r['status'] for r in rs)))
  if ds=='bbh':
   subsets[mode]={}
   for task in sorted({r['subset'] for r in rs}):
    rr=[r for r in rs if r['subset']==task];assert len(rr)==40;subsets[mode][task]=dict(questions=40,correct=sum(r['correct'] for r in rr),accuracy=sum(r['correct'] for r in rr)/40)
 (O/'summary.json').write_text(json.dumps(summary,indent=2));(O/'bbh_subtasks.json').write_text(json.dumps(subsets,indent=2));(O/'SCORING_COMPLETE').touch();print(json.dumps(summary),flush=True)
