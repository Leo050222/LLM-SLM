import os,json,hashlib,fcntl,time,queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ablation_core import atomic_json,digest
from queue_rpc import RPC
from deadline import Budget
from branch import KS,make_bridge,run_branch,verify

def main():
 from transformers import AutoTokenizer
 P=Path(os.environ['GRID_ROOT']);slot=Path(os.environ['OPD_ONLINE_RUN']);Q=slot/'queue'
 lock=(P/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 models=[{r:json.loads((Q/f'replica_{i}'/r/'model.json').read_text()) for r in ['student','teacher']} for i in range(4)]
 assert all(m==models[0] for m in models)
 assert models[0]['student']['tokenizer_sha256']==models[0]['teacher']['tokenizer_sha256']
 tok=AutoTokenizer.from_pretrained(models[0]['student']['model'],local_files_only=True)
 raw=(P/'prompts.jsonl').read_bytes();meta=json.loads((P/'INPUTS_COMPLETE.json').read_text());assert hashlib.sha256(raw).hexdigest()==meta['sha256']
 rows=list(map(json.loads,raw.splitlines()));assert len(rows)==800
 runtime=dict(inputs=meta,models=models[0],replicas=4,concurrency=64,teacher_temperature=1.,student_temperature=0.,ks=list(KS),teacher_request='one n=32 generation shared across k',max_tokens=40264,max_model_len=40960,first_token_ban=False,code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
 if (P/'manifest.json').exists():assert json.loads((P/'manifest.json').read_text())==runtime
 else:atomic_json(P/'manifest.json',runtime)
 tasks=queue.Queue()
 for row in rows:
  if not all((P/f'k{k}/completed'/f"{row['dataset_index']:05d}.json").exists() for k in KS):tasks.put(row)
 print('QUEUED_SAMPLES',tasks.qsize(),flush=True)
 def consume(replica):
  while True:
   try:row=tasks.get_nowait()
   except queue.Empty:return
   i=row['dataset_index'];name=f'{i:05d}.json';budget=Budget(P/'timings'/name,7*86400);rpc=RPC(Q/f'replica_{replica}',budget)
   def call(role,prefix,n,temperature,round_index):
    seed=int(digest([meta['seed'],i,role,round_index])[:8],16)%(2**31)
    return rpc.call(role,dict(op='generate',model_identity=digest(models[replica][role]),prefix=prefix,n=n,temperature=temperature,seed=seed))
   f=P/'bridges'/name
   if f.exists():bridge=json.loads(f.read_text())
   else:
    bridge=make_bridge(row,call);atomic_json(f,bridge)
   assert bridge['identity']==digest(row)
   for k in KS:
    d=P/f'k{k}';out=d/'completed'/name;trace=d/'traces'/name
    if out.exists():continue
    state=json.loads(trace.read_text()) if trace.exists() else None
    def save(s):budget.persist();atomic_json(trace,s)
    state=run_branch(row,k,bridge,call,state,save);verify(row,k,bridge,state)
    prefix_len=len(row['anchor']['response_prefix_token_ids']);t=state['actual_teacher_tokens']
    output=dict(row,**state,k=k,response=tok.decode(state['response_token_ids'],skip_special_tokens=True),token_sources=row['anchor']['prefix_token_sources']+['teacher']*t+['student']*(len(state['response_token_ids'])-prefix_len-t),bridge_sha256=digest(bridge),student_temperature=0.,teacher_temperature=1.)
    atomic_json(out,output);print('TRACE_DONE',i,row['dataset'],'q',row['quartile'],'k',k,state['finish_reason'],flush=True)
 try:
  with ThreadPoolExecutor(max_workers=64) as pool:list(pool.map(consume,[i%4 for i in range(64)]))
  assert len(list((P/'bridges').glob('*.json')))==800
  for k in KS:
   assert len(list((P/f'k{k}/completed').glob('*.json')))==800
   for row in rows:
    name=f"{row['dataset_index']:05d}.json";x=json.loads((P/f'k{k}/completed'/name).read_text());b=json.loads((P/'bridges'/name).read_text());verify(row,k,b,x);assert x['bridge_sha256']==digest(b)
  atomic_json(P/'ALL_GENERATION_COMPLETE.json',dict(rows=4000,bridges=800,completed_at=time.time()))
  atomic_json(slot/'SEGMENT_DONE.json',dict(pending=0))
 except Exception as e:
  atomic_json(slot/'FAILED',dict(error=repr(e)));raise
if __name__=='__main__':main()
