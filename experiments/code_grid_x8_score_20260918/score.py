import collections,concurrent.futures,csv,fcntl,hashlib,json,os,queue,threading,time,statistics,uuid
from pathlib import Path
from code_scorer import sandbox,extract
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');P=R/'opd_eval_code/code_grid_x8_20260916';S=Path(__file__).parent

def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False));t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert json.loads((P/'ALL_GENERATION_COMPLETE.json').read_text())['rows']==405475
 lock=(P/'x8_scoring.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows=list(map(json.loads,(P/'prompts.jsonl').read_text().splitlines()));assert len(rows)==2317
 configs=json.loads((P/'configs.json').read_text());old=Path(json.loads((P/'PLAN.json').read_text())['old_replicate_root'])
 configs=[dict(name='r0_'+a['config'],base_config=a['config'],replicate_index=0,source_dir=a['path']) for a in json.loads((P/'replicate0_audit.json').read_text())]+[dict(c,source_dir=str(P/c['name'])) for c in configs]
 assert len(configs)==200
 data={};datafiles={}
 for ds,name,key in [('cruxeval_i','cruxeval.jsonl','id'),('cruxeval_o','cruxeval.jsonl','id'),('humaneval_plus','HumanEvalPlus.jsonl','task_id'),('mbpp_plus','MbppPlus.jsonl','task_id'),('livecodebench','livecodebench_v6_incremental.jsonl','question_id')]:
  f=old/'data'/name;datafiles[name]=sha(f);items=map(json.loads,f.read_text().splitlines());data[ds]={(x['platform']+'/'+x[key] if ds=='livecodebench' else x[key]):x for x in items}
 signature=hashlib.sha256(json.dumps(dict(files={n:sha(S/n) for n in ['score.py','code_scorer.py','score_one.py']},data=datafiles,timeout_policy='source_trace_timeout_count_wrong'),sort_keys=True).encode()).hexdigest()
 # Exercise actual sandbox with a known-good input and known-invalid input.
 problem=next(iter(data['cruxeval_i'].values()))
 assert sandbox(dict(dataset='cruxeval_i',problem=problem,solution=problem['input']))['correct'] is True
 assert sandbox(dict(dataset='cruxeval_i',problem=problem,solution='__opd_undefined_input__'))['correct'] is False
 atomic(P/'SCORING_SMOKE.json',dict(passed=True,signature=signature));print('SCORING_SMOKE_PASSED',flush=True)
 tasks=queue.Queue()
 for c in configs:
  source=Path(c['source_dir']);assert json.loads((source/'VERIFIED.json').read_text())['rows']==2317
  assert len(list((source/'completed').glob('*.json')))==2317
  for row in rows:tasks.put((c,row))
 assert tasks.qsize()==463400
 progress=collections.Counter();mutex=threading.Lock();errors=[];started=time.time()
 def action(c,row):
  filename=f"{row['dataset_index']:05d}.json";f=Path(c['source_dir'])/'completed'/filename;dest=P/c['name']/'scores'/filename
  raw=f.read_bytes();h=hashlib.sha256(raw).hexdigest()
  if dest.exists():
   result=json.loads(dest.read_text());assert result['source_sha256']==h and result['scoring_signature']==signature
   assert type(result['correct']) is bool
   return True
  x=json.loads(raw);ds=row['dataset'];assert x['dataset']==ds and x['problem_id']==row['problem_id'] and x['prompt_token_ids']==row['prompt_token_ids'] and x['status']=='COMPLETE'
  if x['finish_reason']=='trace_timeout':
   assert x['timeout_count_as_wrong'];result=dict(correct=False,status='trace_timeout')
  else:result=sandbox(dict(dataset=ds,problem=data[ds][x['problem_id']],solution=extract(ds,x['response'])))
  assert type(result['correct']) is bool
  result.update(dataset=ds,problem_id=x['problem_id'],dataset_index=row['dataset_index'],replicate=c['replicate_index'],base_config=c['base_config'],source_sha256=h,scoring_signature=signature,completed_at=time.time())
  atomic(dest,result);return False
 def consume():
  while True:
   try:c,row=tasks.get_nowait()
   except queue.Empty:return
   for attempt in range(3):
    try:cached=action(c,row);break
    except Exception as e:
     if attempt==2:
      with mutex:errors.append(dict(group=c['name'],index=row['dataset_index'],error_type=type(e).__name__))
      cached=None
     else:time.sleep(1)
   with mutex:
    progress['processed']+=1;progress['cached']+=cached is True;progress['errors']+=cached is None
    if progress['processed']%1000==0:
     atomic(P/'SCORING_PROGRESS.json',dict(progress,total=463400,elapsed=time.time()-started,updated_at=time.time(),job_id=os.getenv('SLURM_JOB_ID')));print('SCORE_PROGRESS',dict(progress),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:list(pool.map(lambda _:consume(),range(48)))
 if errors:atomic(P/'SCORING_ERRORS.json',errors);raise RuntimeError(f'{len(errors)} unresolved scoring errors; checkpoints preserved')
 groups=[]
 for c in configs:
  counts=collections.defaultdict(collections.Counter)
  for row in rows:
   x=json.loads((P/c['name']/'scores'/f"{row['dataset_index']:05d}.json").read_text());assert type(x['correct']) is bool and x['scoring_signature']==signature and x['replicate']==c['replicate_index'] and x['problem_id']==row['problem_id']
   g=counts[x['dataset']];g['n']+=1;g['correct']+=x['correct']
  for ds,g in counts.items():groups.append(dict(dataset=ds,config=c['base_config'],replicate=c['replicate_index'],**g,accuracy=g['correct']/g['n']))
 assert len(groups)==1000 and sum(g['n'] for g in groups)==463400
 combined=[]
 for ds in data:
  for cfg in sorted({c['base_config'] for c in configs}):
   gs=[g for g in groups if g['dataset']==ds and g['config']==cfg];assert len(gs)==8 and {g['replicate'] for g in gs}==set(range(8))
   n=sum(g['n'] for g in gs);correct=sum(g['correct'] for g in gs);combined.append(dict(dataset=ds,config=cfg,n=n,correct=correct,accuracy=correct/n,replicate_accuracy_sd=statistics.stdev(g['accuracy'] for g in gs)))
 atomic(P/'x8_score_summary.json',dict(rows=463400,groups=groups,combined=combined,scoring_signature=signature))
 with (P/'x8_score_summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
 atomic(P/'ALL_X8_SCORING_COMPLETE.json',dict(rows=463400,groups=1000,completed_at=time.time(),signature=signature));print('ALL_X8_SCORING_COMPLETE',flush=True)
if __name__=='__main__':main()
