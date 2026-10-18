import sys,json,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from threshold_core import atomic_json,FileRPC,digest
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_hard/threshold3_mathteacher_20260911';C=R/'opd_online_code/20260911_chunk64_k8_2nodes/ocrd_pipeline'
sys.path.insert(0,str(C))
from online_rollout.run import Backend
rows=[json.loads(l) for l in (P/'prompts.jsonl').open()];math=[r for r in rows if r['dataset']=='math500'];assert len(math)==500

def verify():
 assert json.loads((P/'THRESHOLD_GENERATION_COMPLETE.json').read_text())['rows']==2314
 for mode,rr in [('online',rows),('teacher',math)]:
  assert len(list((P/mode/'completed').glob('*.json')))==len(rr)
  for r in rr:
   v=json.loads((P/mode/'completed'/f"{r['dataset_index']:05d}.json").read_text());assert v['status']=='COMPLETE' and v['problem_id']==r['problem_id'] and v['prompt_token_ids']==r['prompt_token_ids'] and v['solution']==r['solution']
 atomic_json(P/'ALL_GENERATION_COMPLETE.json',dict(rows=2814,groups=dict(online=2314,teacher=500)))

def main():
 (P/'teacher').mkdir(exist_ok=True);lock=(P/'teacher/generator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B',local_files_only=True)
 bs=[]
 for i in range(4):
  q=P/f'queue/replica_{i}';models={role:json.loads((q/role/'model.json').read_text()) for role in ['student','teacher']};bs.append(Backend(FileRPC(q,1800),models,tok))
 manifest=dict(model=models['teacher'],thinking=False,temperature=1,top_p=1,seed=20260911,seed_rule='20260911+original_math500_index',prompt_identity=digest(math))
 m=P/'teacher/manifest.json'
 if m.exists():assert json.loads(m.read_text())==manifest
 else:atomic_json(m,manifest)
 def one(r):
  f=P/'teacher/completed'/f"{r['dataset_index']:05d}.json"
  if f.exists():
   v=json.loads(f.read_text());assert v['status']=='COMPLETE' and v['problem_id']==r['problem_id'];return
  seed=20260911+r['source_dataset_index'];s=bs[r['source_dataset_index']%4].generate('teacher',r['prompt_token_ids'],min(40264,40960-len(r['prompt_token_ids'])-1),seed)
  atomic_json(f,dict(r,status='COMPLETE',response_token_ids=s['token_ids'],response=tok.decode(s['token_ids'],skip_special_tokens=True),finish_reason=s['finish_reason'],generation_seed=seed))
 for attempt in range(3):
  failed=[]
  with ThreadPoolExecutor(max_workers=128) as pool:
   fs=[pool.submit(one,r) for r in math]
   for k,f in enumerate(as_completed(fs),1):
    try:f.result()
    except Exception as e:failed.append(type(e).__name__)
    if k%100==0:print('teacher settled',k,500,flush=True)
  if not failed:return
  print('teacher retries',failed,flush=True)
 raise SystemExit(1)
if __name__=='__main__':
 if len(sys.argv)>1:verify()
 else:main()
