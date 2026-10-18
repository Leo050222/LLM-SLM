import os,sys,json,hashlib,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');O=R/'opd_eval_math500/math500_three_20260911'
sys.path.insert(0,str(R/'opd_online_code/20260911_chunk64_k8_2nodes/ocrd_pipeline'))
from online_rollout.core import atomic_json,FileRPC
from online_rollout.run import Backend
MODEL='/storage/yukaichengLab/share/model/Qwen3-1.7B'
def rows():return [json.loads(l) for l in (O/'prompts.jsonl').open()]
def prepare():
 from transformers import AutoTokenizer
 p=R/'opd_eval_math500/data/AI-ModelScope_MATH-500/test.jsonl';assert hashlib.sha256(p.read_bytes()).hexdigest()=='35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132'
 data=[json.loads(l) for l in p.open()];assert len(data)==500
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);out=[]
 for i,r in enumerate(data):
  msg=r['problem']+'\nPlease reason step by step, and put your final answer within \\boxed{}.'
  ids=tok.apply_chat_template([{'role':'user','content':msg}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  if hasattr(ids,'keys'):ids=ids['input_ids']
  out.append(dict(dataset_index=i,problem_id=r['unique_id'],problem=r['problem'],solution=r['answer'],level=r['level'],prompt_token_ids=list(ids)))
 O.mkdir(parents=True,exist_ok=True);content=''.join(json.dumps(r)+'\n' for r in out)
 if (O/'prompts.jsonl').exists():assert (O/'prompts.jsonl').read_text()==content
 else:(O/'prompts.jsonl').write_text(content)
 atomic_json(O/'protocol.json',dict(data_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),questions=500,thinking=False,seed=20260911,temperature=1,top_p=1,max_tokens=40264,online_teacher_tokens=8,online_chunk_tokens=64,takeover_limit=None,prefix_tokens=256,scoring='EvalScope MATH500 extract_answer + math_equal; only after all 1500 outputs',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
def baseline(mode):
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 bs=[]
 for k in range(4):
  q=O/f'queue/replica_{k}';models={role:json.loads((q/role/'model.json').read_text()) for role in ['student','teacher']};bs.append(Backend(FileRPC(q,1800),models,tok))
 def one(r):
  p=O/mode/'completed'/f"{r['dataset_index']:05d}.json"
  if p.exists():
   saved=json.loads(p.read_text());assert saved['problem_id']==r['problem_id'] and saved['status']=='COMPLETE';return
  b=bs[r['dataset_index']%4];prompt=r['prompt_token_ids'];prefix=[];seed=20260911+r['dataset_index'];finished=False
  if mode=='prefix256':
   cache=O/'teacher_prefixes'/f"{r['dataset_index']:05d}.json"
   if cache.exists():t=json.loads(cache.read_text())
   else:t=b.generate('teacher',prompt,256,seed);atomic_json(cache,t)
   prefix=t['token_ids'];finished=t['finish_reason']=='stop'
  if finished:ids=prefix;finish='stop'
  else:
   s=b.generate('student',prompt+prefix,min(40264-len(prefix),40960-len(prompt)-len(prefix)),seed);ids=prefix+s['token_ids'];finish=s['finish_reason']
  atomic_json(p,dict(r,status='COMPLETE',response_token_ids=ids,response=tok.decode(ids,skip_special_tokens=True),finish_reason=finish,teacher_tokens=len(prefix),teacher_finished=finished))
 with ThreadPoolExecutor(max_workers=128) as pool:
  fs=[pool.submit(one,r) for r in rows()]
  for k,f in enumerate(as_completed(fs),1):
   f.result()
   if k%20==0:print(mode,k,flush=True)
def verify():
 rr=rows()
 for mode in ['student','online','prefix256']:
  ps=list((O/mode/'completed').glob('*.json'));assert len(ps)==500
  for r in rr:
   s=json.loads((O/mode/'completed'/f"{r['dataset_index']:05d}.json").read_text());assert s['problem_id']==r['problem_id'] and s['status']=='COMPLETE' and s['solution']==r['solution']
 atomic_json(O/'ALL_GENERATION_COMPLETE.json',{'groups':3,'rows':1500})
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='prepare':prepare()
 elif mode=='verify':verify()
 else:baseline(mode)
