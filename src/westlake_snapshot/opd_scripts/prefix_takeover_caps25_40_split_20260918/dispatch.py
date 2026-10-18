import os,sys,json,time,queue,threading,fcntl,hashlib
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from ablation_core import Settings,atomic_json,digest
from branch import run_branch,verify
from queue_rpc import RPC
from deadline import Budget
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
sys.path.append(str(R/'opd_scripts/nojudge_grid_20260912'))
from run import Backend,tokenizer_fingerprint

def main():
 from transformers import AutoTokenizer
 P=Path(os.environ['GRID_ROOT']);slot=Path(os.environ['OPD_ONLINE_RUN']);Q=slot/'queue'
 lock=(P/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 models=[{role:json.loads((Q/f'replica_{i}'/role/'model.json').read_text()) for role in ['student','teacher']} for i in range(2)]
 assert all(m==models[0] for m in models)
 tok=AutoTokenizer.from_pretrained(models[0]['student']['model'],local_files_only=True)
 assert all(m['tokenizer_sha256']==tokenizer_fingerprint(tok) for m in models[0].values())
 raw=(P/'prompts.jsonl').read_bytes();meta=json.loads((P/'INPUTS_COMPLETE.json').read_text());assert hashlib.sha256(raw).hexdigest()==meta['sha256']
 rows=list(map(json.loads,raw.splitlines()));assert len(rows)==800
 runtime={'input_sha256':meta['sha256'],'models':models[0],'replicas':2,'concurrency':64,'placement':'teacher and student on separate H800 GPUs','worker_batch_size':8,'teacher_kv_gib':None,'student_kv_gib':None,'code_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},'caps':meta['caps'],'branch_seeds':meta['branch_seeds'],'max_tokens':40264,'threshold':2,'k':1,'time_policy':'No scientific wall-time censoring; infrastructure timeout raises an error and preserves checkpoint','frozen_initial_teacher':'At fixed first anchor, count as first intervention; ban original student entry token','paired_seeds':'Same replicate seed across caps; separate initial_teacher seed namespace'}
 manifest=P/'manifest.json'
 from migration import migrate
 migrate(manifest,runtime)
 tasks=queue.Queue()
 # Interleave caps/replicates for every sample, allowing shared-prefix/cache reuse.
 for row in rows:
  for rep,seed in enumerate(meta['branch_seeds']):
   for cap in meta['caps']:
    name=f'cap{cap}_r{rep}';f=P/name/'completed'/f"{row['dataset_index']:05d}.json"
    if not f.exists():tasks.put((row,rep,seed,cap,name))
 print('QUEUED',tasks.qsize(),flush=True);failures=[];mutex=threading.Lock()
 def action(task,replica):
  row,rep,seed,cap,name=task;d=P/name;i=row['dataset_index'];f=d/'traces'/f'{i:05d}.json'
  settings=Settings(teacher_tokens=1,gap_threshold=2,mode='nojudge_grid',max_takeovers=cap,seed=seed)
  # This is an infrastructure lease, not a rule that marks slow answers incorrect.
  budget=Budget(d/'timings'/f.name,limit=7*86400)
  backend=Backend(RPC(Q/f'replica_{replica}',budget),models[replica],tok)
  state=json.loads(f.read_text()) if f.exists() else None
  def save(s):budget.persist();atomic_json(f,s)
  print('TRACE_START',name,i,flush=True)
  try:
   result=run_branch(row,settings,backend,state,save);used=verify(row,cap,result)
   output=dict(row,**result,response=tok.decode(result['response_token_ids'],skip_special_tokens=True),cap=cap,replicate=rep,seed=seed,actual_takeovers=used,frozen_prefix_sha256=row['anchor']['frozen_prefix_sha256'],continuation_token_count=len(result['response_token_ids'])-len(row['anchor']['response_prefix_token_ids']),trace_elapsed_seconds=budget.elapsed())
   atomic_json(d/'completed'/f.name,output);print('TRACE_DONE',name,i,result['finish_reason'],'takeovers',used,flush=True)
  except Exception as e:
   with mutex:failures.append({'group':name,'index':i,'error':type(e).__name__})
   print('TRACE_ERROR',name,i,type(e).__name__,str(e)[:200],flush=True)
 def consume(i):
  while not failures:
   try:task=tasks.get_nowait()
   except queue.Empty:return
   action(task,i)
 with ThreadPoolExecutor(max_workers=64) as pool:list(pool.map(consume,[i%2 for i in range(64)]))
 if failures:
  atomic_json(slot/'FAILED',failures);raise RuntimeError('Generation errors; preserve checkpoints')
 if not tasks.empty():atomic_json(slot/'SEGMENT_DONE.json',{'pending':tasks.qsize()});return
 count=0
 for rep in range(8):
  for cap in meta['caps']:
   d=P/f'cap{cap}_r{rep}';assert len(list((d/'completed').glob('*.json')))==800
   for row in rows:
    x=json.loads((d/'completed'/f"{row['dataset_index']:05d}.json").read_text());verify(row,cap,x)
    assert x['seed']==meta['branch_seeds'][rep] and x['cap']==cap and x['replicate']==rep
    assert x['identity']==digest({'row':row,'settings':asdict(Settings(teacher_tokens=1,gap_threshold=2,mode='nojudge_grid',max_takeovers=cap,seed=x['seed']))})
    count+=1
   atomic_json(d/'VERIFIED.json',{'rows':800})
 assert count==meta['expected_outputs']==25600
 atomic_json(P/'ALL_GENERATION_COMPLETE.json',{'rows':count,'completed_at':time.time()});atomic_json(slot/'SEGMENT_DONE.json',{'pending':0})
 print('ALL_GENERATION_COMPLETE',count,flush=True)
if __name__=='__main__':main()
