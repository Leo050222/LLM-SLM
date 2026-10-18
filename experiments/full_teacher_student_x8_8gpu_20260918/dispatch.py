import os,json,time,queue,fcntl,hashlib,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ablation_core import Settings,atomic_json,digest
from queue_rpc import RPC
from deadline import Budget,TraceDeadline
from generation import generate,verify

def main():
 from transformers import AutoTokenizer
 P=Path(os.environ['GRID_ROOT']);slot=Path(os.environ['OPD_ONLINE_RUN']);role=os.environ['BASELINE_ROLE'];Q=slot/'queue'
 lock=(P/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 models=[json.loads((Q/f'replica_{i}'/role/'model.json').read_text()) for i in range(8)];assert all(m==models[0] for m in models)
 tok=AutoTokenizer.from_pretrained(models[0]['model'],local_files_only=True)
 meta=json.loads((P/'INPUTS_COMPLETE.json').read_text());rows={}
 for family,info in meta['inputs'].items():
  raw=(P/f'{family}_prompts.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==info['sha256'];rows[family]=list(map(json.loads,raw.splitlines()));assert len(rows[family])==info['rows']
 configs=[c for c in json.loads((P/'configs.json').read_text()) if c['role']==role]
 runtime=dict(role=role,models=models[0],replicas=8,concurrency=256,batch_size=8,trace_limit_seconds=900,inputs=meta['inputs'],seeds=meta['seeds'],code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
 mf=P/f'{role}_manifest.json'
 from migration import migrate
 migrate(mf,runtime)
 tasks=queue.Queue()
 for c in configs:
  for row in rows[c['family']]:
   if not (P/c['name']/'completed'/f"{row['dataset_index']:05d}.json").exists():tasks.put((c,row))
 print('QUEUED',role,tasks.qsize(),flush=True);errors=[];mutex=threading.Lock()
 def consume(i):
  while not errors:
   try:c,row=tasks.get_nowait()
   except queue.Empty:return
   d=P/c['name'];f=d/'traces'/f"{row['dataset_index']:05d}.json";budget=Budget(d/'timings'/f.name,900);cfg=Settings(**c['settings'])
   state=json.loads(f.read_text()) if f.exists() else None
   if state is None:state=dict(identity=digest(dict(row=row,settings=c['settings'])),response_token_ids=[],token_sources=[],events=[],rounds=0,student_draft_tokens=0,discarded_tokens=0,judge_calls=0,status='RUNNING')
   def save(x):budget.persist();atomic_json(f,x)
   try:
    try:
     if state['status']!='COMPLETE':budget.check()
     result=generate(row,cfg,models[i],RPC(Q/f'replica_{i}',budget),state,save)
    except TraceDeadline:
     result=state;result.update(status='COMPLETE',finish_reason='trace_timeout',timeout_count_as_wrong=True)
    result['trace_elapsed_seconds']=budget.elapsed();save(result)
    out=dict(row,**result,response=tok.decode(result['response_token_ids'],skip_special_tokens=True));verify(row,cfg,out)
    atomic_json(d/'completed'/f.name,out);print('TRACE_DONE',c['name'],row['dataset_index'],out['finish_reason'],flush=True)
   except Exception as e:
    with mutex:errors.append(dict(group=c['name'],index=row['dataset_index'],error=type(e).__name__,detail=str(e)[:200]))
 if tasks.qsize():
  with ThreadPoolExecutor(max_workers=256) as pool:list(pool.map(consume,[i%8 for i in range(256)]))
 if errors:atomic_json(slot/'FAILED',errors);raise RuntimeError('Generation errors; checkpoints retained')
 count=0
 for c in configs:
  d=P/c['name'];rr=rows[c['family']];assert len(list((d/'completed').glob('*.json')))==len(rr)
  for row in rr:verify(row,Settings(**c['settings']),json.loads((d/'completed'/f"{row['dataset_index']:05d}.json").read_text()))
  atomic_json(d/'VERIFIED.json',dict(rows=len(rr)));count+=len(rr)
 assert count==36568
 atomic_json(P/f'{role}_GENERATION_COMPLETE.json',dict(rows=count,completed_at=time.time()));atomic_json(slot/'SEGMENT_DONE.json',dict(pending=0))
 if all((P/f'{r}_GENERATION_COMPLETE.json').exists() for r in ['student','teacher']):atomic_json(P/'ALL_GENERATION_COMPLETE.json',dict(rows=73136,completed_at=time.time()))
 print('ROLE_COMPLETE',role,count,flush=True)
if __name__=='__main__':main()
