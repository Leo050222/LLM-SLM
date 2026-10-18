import sys,json,hashlib,os,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');O=R/'opd_eval_hard/hard_three_20260911';C=R/'opd_online_code/20260911_chunk64_k8_2nodes'
sys.path.insert(0,str(C/'ocrd_pipeline'))
from online_rollout.core import FileRPC,atomic_json
from online_rollout.run import Backend
N=1814
def rows():return [json.loads(l) for l in (O/'prompts.jsonl').open()]
def generate(mode):
 (O/mode).mkdir(parents=True,exist_ok=True)
 lock=(O/mode/'generator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B',local_files_only=True)
 bs=[]
 for k in [int(x) for x in os.environ.get('OPD_DIRECT_REPLICAS','0,1,2,3,4,5,6,7').split(',')]:
  q=O/f'queue/replica_{k}';models={role:json.loads((q/role/'model.json').read_text()) for role in ['student','teacher']};bs.append(Backend(FileRPC(q,1800),models,tok))
 def one(r):
  p=O/mode/'completed'/f"{r['dataset_index']:05d}.json"
  if p.exists():
   v=json.loads(p.read_text());assert v['problem_id']==r['problem_id'] and v['status']=='COMPLETE';return
  s=bs[r['dataset_index']%len(bs)].generate(mode,r['prompt_token_ids'],min(40264,40960-len(r['prompt_token_ids'])-1),20260911+r['dataset_index']);ids=s['token_ids']
  atomic_json(p,dict(r,status='COMPLETE',response_token_ids=ids,response=tok.decode(ids,skip_special_tokens=True),finish_reason=s['finish_reason']))
 with ThreadPoolExecutor(max_workers=256) as pool:
  fs=[pool.submit(one,r) for r in rows()]
  for k,f in enumerate(as_completed(fs),1):
   f.result()
   if k%50==0:print(mode,k,N,flush=True)
def verify():
 rr=rows();assert len(rr)==N
 for mode in ['online','student','teacher']:
  assert len(list((O/mode/'completed').glob('*.json')))==N
  for r in rr:
   v=json.loads((O/mode/'completed'/f"{r['dataset_index']:05d}.json").read_text());assert v['status']=='COMPLETE' and v['problem_id']==r['problem_id'] and v['solution']==r['solution'] and v['prompt_token_ids']==r['prompt_token_ids']
 atomic_json(O/'ALL_GENERATION_COMPLETE.json',dict(groups=3,rows=N*3))
if __name__=='__main__':
 if sys.argv[1]=='verify':verify()
 else:generate(sys.argv[1])
