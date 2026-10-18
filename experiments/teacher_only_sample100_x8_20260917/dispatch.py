import os,json,time,queue,fcntl,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ablation_core import atomic_json,digest
from queue_rpc import RPC
from deadline import Budget
from generation import generate,verify

def main():
 from transformers import AutoTokenizer
 P=Path(os.environ['GRID_ROOT']);slot=Path(os.environ['OPD_ONLINE_RUN']);Q=slot/'queue'
 lock=(P/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 models=[json.loads((Q/f'replica_{i}'/'teacher/model.json').read_text()) for i in range(4)]
 assert all(m==models[0] for m in models)
 tok=AutoTokenizer.from_pretrained(models[0]['model'],local_files_only=True)
 meta=json.loads((P/'INPUTS_COMPLETE.json').read_text());raw=(P/'prompts.jsonl').read_bytes()
 assert hashlib.sha256(raw).hexdigest()==meta['sha256']
 rows=list(map(json.loads,raw.splitlines()));assert len(rows)==800 and meta['expected_outputs']==6400
 runtime=dict(input_sha256=meta['sha256'],model=models[0],replicas=4,concurrency=128,condition='teacher_only',generation_start='original prompt; ignore saved student prefix',branch_seeds=meta['branch_seeds'],code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
 manifest=P/'manifest.json'
 if manifest.exists():assert json.loads(manifest.read_text())==runtime
 else:atomic_json(manifest,runtime)
 tasks=queue.Queue()
 for row in rows:
  for rep,seed in enumerate(meta['branch_seeds']):
   if not (P/f'teacher_only_r{rep}'/'completed'/f"{row['dataset_index']:05d}.json").exists():tasks.put((row,rep,seed))
 print('QUEUED',tasks.qsize(),flush=True)
 def consume(i):
  while True:
   try:row,rep,seed=tasks.get_nowait()
   except queue.Empty:return
   d=P/f'teacher_only_r{rep}';f=d/'traces'/f"{row['dataset_index']:05d}.json";b=Budget(d/'timings'/f.name,7*86400)
   state=json.loads(f.read_text()) if f.exists() else None
   def save(s):b.persist();atomic_json(f,s)
   out=generate(row,seed,models[i],RPC(Q/f'replica_{i}',b),state,save);verify(row,seed,models[i],out)
   out=dict(row,**out,response=tok.decode(out['response_token_ids'],skip_special_tokens=True),condition='teacher_only',replicate=rep,seed=seed,actual_takeovers=0,generation_start='original_prompt',trace_elapsed_seconds=b.elapsed())
   atomic_json(d/'completed'/f.name,out);print('TRACE_DONE',rep,row['dataset_index'],out['finish_reason'],flush=True)
 with ThreadPoolExecutor(max_workers=128) as pool:list(pool.map(consume,[i%4 for i in range(128)]))
 for rep,seed in enumerate(meta['branch_seeds']):
  d=P/f'teacher_only_r{rep}';assert len(list((d/'completed').glob('*.json')))==800
  for row in rows:
   x=json.loads((d/'completed'/f"{row['dataset_index']:05d}.json").read_text());verify(row,seed,models[0],x)
   assert x['prompt_token_ids']==row['prompt_token_ids'] and x['problem_id']==row['problem_id'] and x['seed']==seed and x['replicate']==rep and x['condition']=='teacher_only'
  atomic_json(d/'VERIFIED.json',dict(rows=800))
 atomic_json(P/'ALL_GENERATION_COMPLETE.json',dict(rows=6400,completed_at=time.time()))
 atomic_json(slot/'SEGMENT_DONE.json',dict(pending=0));print('ALL_GENERATION_COMPLETE',6400,flush=True)
if __name__=='__main__':main()
