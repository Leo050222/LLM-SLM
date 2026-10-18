import sys,json,hashlib,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
from threshold_core import Settings,FileRPC,atomic_json,digest,rollout
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_hard/threshold3_mathteacher_20260911';C=R/'opd_online_code/20260911_chunk64_k8_2nodes/ocrd_pipeline'
sys.path.insert(0,str(C))
from online_rollout.run import Backend
from anchor_selector.core import tokenizer_fingerprint
N=2314

def main():
 (P/'online').mkdir(exist_ok=True);lock=(P/'online/coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows=[json.loads(l) for l in (P/'prompts.jsonl').open()];assert len(rows)==N
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B',local_files_only=True);fingerprint=tokenizer_fingerprint(tok)
 bs=[];first=None
 for i in range(4):
  q=P/f'queue/replica_{i}';models={role:json.loads((q/role/'model.json').read_text()) for role in ['student','teacher']}
  if first is None:first=models
  assert models==first and all(v['tokenizer_sha256']==fingerprint for v in models.values())
  bs.append(Backend(FileRPC(q,1800),models,tok))
 settings=Settings(gap_threshold=3.)
 manifest=dict(rows=N,input_sha256=digest(rows),code_identity=digest({f.name:f.read_text() for f in Path(__file__).parent.glob('*.py')}),models=first,settings=asdict(settings),sampling=dict(temperature=1.,top_p=1.),thinking=False,judge=False,selection='first chronological candidate with abs(student_lp-teacher_lp)>3',multiple_takeovers=True,first_token_ban=True)
 mp=P/'manifest.json'
 if mp.exists():assert json.loads(mp.read_text())==manifest
 else:atomic_json(mp,manifest)
 def forbidden_judge(*args):raise AssertionError('Judge must never be invoked')
 def one(r):
  i=r['dataset_index'];f=P/'online/traces'/f'{i:05d}.json';ck=json.loads(f.read_text()) if f.exists() else None
  result=rollout(r,settings,bs[i%4],forbidden_judge,ck,save=lambda s:atomic_json(f,s))
  atomic_json(P/'online/completed'/f.name,dict(r,**result,response=tok.decode(result['response_token_ids'],skip_special_tokens=True)))
 failures=[]
 with ThreadPoolExecutor(max_workers=128) as pool:
  pending={pool.submit(one,r):r['dataset_index'] for r in rows}
  for k,f in enumerate(as_completed(pending),1):
   try:f.result()
   except Exception as e:failures.append(dict(index=pending[f],error=type(e).__name__));print('failed',failures[-1],flush=True)
   if k%100==0:print('settled',k,N,flush=True)
 atomic_json(P/'run_status.json',dict(failed=failures))
 if failures:raise SystemExit(1)
 assert len(list((P/'online/completed').glob('*.json')))==N
 for r in rows:
  v=json.loads((P/'online/completed'/f"{r['dataset_index']:05d}.json").read_text());assert v['status']=='COMPLETE' and v['problem_id']==r['problem_id'] and v['prompt_token_ids']==r['prompt_token_ids'];assert v['judge_calls']==0
  for e in v['events']:
   assert not e['judgments']
   if 'bridge_token_ids' in e:
    assert 0<len(e['bridge_token_ids'])<=8 and e['threshold_candidate']['absolute_logprob_gap']>3
 atomic_json(P/'THRESHOLD_GENERATION_COMPLETE.json',dict(rows=N,groups=1))
if __name__=='__main__':main()
