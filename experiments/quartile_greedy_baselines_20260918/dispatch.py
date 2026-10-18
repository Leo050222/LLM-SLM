import os,json,hashlib,fcntl,time,queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ablation_core import atomic_json,digest
from queue_rpc import RPC
from deadline import Budget
from branch import run_branch,verify

def main():
 from transformers import AutoTokenizer
 P=Path(os.environ['GRID_ROOT']);slot=Path(os.environ['OPD_ONLINE_RUN']);Q=slot/'queue';role=os.environ['BASELINE_ROLE'];group='k0' if role=='student' else 'teacher_greedy';D=P/group;D.mkdir(exist_ok=True)
 lock=(D/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 models=[json.loads((Q/f'replica_{i}'/role/'model.json').read_text()) for i in range(8)]
 assert all(m==models[0] for m in models)
 original=json.loads((P/'manifest.json').read_text())
 for key in ['model','tokenizer_sha256','max_model_len']:assert models[0][key]==original['models'][role][key]
 tok=AutoTokenizer.from_pretrained(models[0]['model'],local_files_only=True)
 raw=(P/'prompts.jsonl').read_bytes();meta=json.loads((P/'INPUTS_COMPLETE.json').read_text());assert hashlib.sha256(raw).hexdigest()==meta['sha256']
 rows=list(map(json.loads,raw.splitlines()));assert len(rows)==800
 runtime=dict(input_sha256=meta['sha256'],model=models[0],replicas=8,concurrency=64,role=role,group=group,student_temperature=0.,max_tokens=40264,max_model_len=40960,code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
 if (D/'manifest.json').exists():assert json.loads((D/'manifest.json').read_text())==runtime
 else:atomic_json(D/'manifest.json',runtime)
 tasks=queue.Queue()
 for row in rows:
  if not (D/'completed'/f"{row['dataset_index']:05d}.json").exists():tasks.put(row)
 def consume(replica):
  while True:
   try:row=tasks.get_nowait()
   except queue.Empty:return
   i=row['dataset_index'];name=f'{i:05d}.json';budget=Budget(D/'timings'/name,7*86400);rpc=RPC(Q/f'replica_{replica}',budget)
   def call(request_role,prefix,n,temperature,round_index):
    assert request_role==role and temperature==0.
    seed=int(digest([meta['seed'],i,role,round_index])[:8],16)%(2**31)
    return rpc.call(role,dict(op='generate',model_identity=digest(models[replica]),prefix=prefix,n=n,temperature=temperature,seed=seed))
   trace=D/'traces'/name;state=json.loads(trace.read_text()) if trace.exists() else None
   def save(s):budget.persist();atomic_json(trace,s)
   state=run_branch(row,call,state,save,role=role);verify(row,state,role=role)
   prefix_len=len(row['anchor']['response_prefix_token_ids'])
   output=dict(row,**state,k=(0 if role=='student' else 'teacher_greedy'),response=tok.decode(state['response_token_ids'],skip_special_tokens=True),token_sources=row['anchor']['prefix_token_sources']+[role]*(len(state['response_token_ids'])-prefix_len),student_temperature=(0. if role=="student" else None),teacher_temperature=(0. if role=="teacher" else None))
   atomic_json(D/'completed'/name,output);print('TRACE_DONE',i,row['dataset'],row['quartile'],state['finish_reason'],flush=True)
 try:
  with ThreadPoolExecutor(max_workers=64) as pool:list(pool.map(consume,[i%8 for i in range(64)]))
  assert len(list((D/'completed').glob('*.json')))==800
  for row in rows:verify(row,json.loads((D/'completed'/f"{row['dataset_index']:05d}.json").read_text()),role=role)
  atomic_json(P/f'{group}_GENERATION_COMPLETE.json',dict(rows=800,completed_at=time.time()))
  atomic_json(slot/'SEGMENT_DONE.json',dict(pending=0))
 except Exception as e:atomic_json(slot/'FAILED',dict(error=repr(e)));raise
if __name__=='__main__':main()
