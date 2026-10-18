import os,sys,json,signal,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');O=R/'opd_eval_math500/math500_three_20260911'
sys.path.insert(0,str(R/'eval_repos/evalscope'))
def one(args):
 from evalscope.metrics.math.parser import extract_answer,math_equal,strip_answer_string
 mode,i=args;p=O/mode/'completed'/f'{i:05d}.json';r=json.loads(p.read_text());dest=O/'scores'/mode/p.name
 identity=hashlib.sha256(p.read_bytes()).hexdigest()
 if dest.exists():
  s=json.loads(dest.read_text());assert s['source_sha256']==identity;return s
 def timeout(*_):raise TimeoutError()
 signal.signal(signal.SIGALRM,timeout);pred=None;status='ok';ok=False
 try:
  signal.alarm(30)
  pred=extract_answer(r['response'].rsplit('</think>',1)[-1].strip())
  ok=bool(math_equal(pred,strip_answer_string(r['solution'])))
  if not pred:status='missing'
 except TimeoutError:status='timeout'
 except Exception as e:status=type(e).__name__
 finally:signal.alarm(0)
 s=dict(group=mode,dataset_index=i,problem_id=r['problem_id'],correct=ok,predicted=pred,status=status,source_sha256=identity,tokens=len(r['response_token_ids']))
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(s));tmp.replace(dest);return s
if __name__=='__main__':
 assert json.loads((O/'ALL_GENERATION_COMPLETE.json').read_text())['rows']==1500
 for mode in ['student','online','prefix256']:assert len(list((O/mode/'completed').glob('*.json')))==500
 result=[]
 with ProcessPoolExecutor(max_workers=24) as pool:
  fs=[pool.submit(one,(mode,i)) for mode in ['student','online','prefix256'] for i in range(500)]
  for k,f in enumerate(as_completed(fs),1):
   result.append(f.result())
   if k%100==0:print('SCORED',k,1500,flush=True)
 from collections import Counter
 summary={}
 for mode in ['student','online','prefix256']:
  rs=[r for r in result if r['group']==mode];assert len(rs)==500
  summary[mode]=dict(questions=500,correct=sum(r['correct'] for r in rs),accuracy=sum(r['correct'] for r in rs)/500,score_status=dict(Counter(r['status'] for r in rs)),mean_tokens=sum(r['tokens'] for r in rs)/500)
 (O/'summary.json').write_text(json.dumps(summary,indent=2));(O/'SCORING_COMPLETE').touch();print(json.dumps(summary),flush=True)
