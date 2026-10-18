import asyncio,collections,concurrent.futures,fcntl,hashlib,importlib.util,json,os,random,resource,sys,time
from pathlib import Path
import aiohttp
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');G=R/'opd_eval_hard/math_reasoning_grid_x8_20260916';P=R/'opd_rescore/math_grid_score_all_2000_20260919';S=Path(__file__).parent
OLD=R/'opd_scripts/r1_math_score_20260918';EX=R/'opd_scripts/random_quartile_k1_20260918/score.py'
sys.path.insert(0,str(OLD));import judge_client
sys.path.insert(0,str(EX.parent));spec=importlib.util.spec_from_file_location('paired_scorer',EX);paired=importlib.util.module_from_spec(spec);spec.loader.exec_module(paired)
from ablation_core import atomic_json,digest
sha=lambda b:hashlib.sha256(b).hexdigest()
class Gate:
 def __init__(self):self.limit=256;self.active=0;self.peak=0;self.cv=asyncio.Condition();self.until=0
 async def enter(self):
  async with self.cv:
   while self.active>=self.limit or time.monotonic()<self.until:
    try:await asyncio.wait_for(self.cv.wait(),1)
    except asyncio.TimeoutError:pass
   self.active+=1;self.peak=max(self.peak,self.active)
 async def leave(self):
  async with self.cv:self.active-=1;self.cv.notify_all()
 async def set(self,n):
  async with self.cv:self.limit=n;self.cv.notify_all()

async def main():
 resource.setrlimit(resource.RLIMIT_NOFILE,(min(16384,resource.getrlimit(resource.RLIMIT_NOFILE)[1]),resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
 loop=asyncio.get_running_loop();loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=32))
 lock=(G/'scoring.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (P/'judge/EXTERNAL_SCORING_APPROVED.json').exists() and (G/'ALL_GENERATION_COMPLETE.json').exists()
 configs=json.loads((G/'configs.json').read_text());rows=list(map(json.loads,(G/'prompts.jsonl').read_text().splitlines()));assert len(configs)==175 and len(rows)==2254 and all((G/c['name']/'VERIFIED.json').exists() for c in configs)
 prompt=(P/'judge/system_prompt.txt').read_text();assert prompt==(OLD/'system_prompt.txt').read_text()
 signature=digest(dict(model='gpt-4o-mini',prompt=prompt,code={f.name:sha(f.read_bytes()) for f in OLD.glob('*.py')},extractor=sha(EX.read_bytes())))
 key=(R/'opd_rescore/grid_math_gpt4o_mini_openlux_20260916/.openlux_api_key').read_text().strip()
 relay=json.loads((P/'judge/relay.json').read_text());proxy=f"http://{relay['host']}:{relay['port']}";auth=aiohttp.BasicAuth('opd',(P/'judge/.relay_token').read_text().strip())
 gate=Gate();counts=collections.Counter();latencies=[];started=time.time();queue=asyncio.Queue(maxsize=8192);done_event=asyncio.Event();groups=collections.defaultdict(collections.Counter);failures=[];stages=[]
 settings={c['name']:json.loads((G/c['name']/'manifest.json').read_text())['settings'] for c in configs}
 for c in configs:assert settings[c['name']]['seed']==c['seed'] and settings[c['name']]['teacher_tokens']==c['k'] and settings[c['name']]['gap_threshold']==float(c['threshold'])
 atomic_json(P/'RUN_CONFIG.json',dict(rows=394450,total_with_first_round=450800,max_concurrency=2000,stages=[256,512,1000,2000],stage_seconds=90,scoring_signature=signature,implementation_sha256={f.name:sha(f.read_bytes()) for f in S.glob('*.py')},job_id=os.getenv('SLURM_JOB_ID'),started_at=started))
 def prepare(c,row):
  f=G/c['name']/'completed'/f"{row['dataset_index']:05d}.json";raw=f.read_bytes();x=json.loads(raw);source=sha(raw);dest=f.parent.parent/'scores'/f.name
  assert x['status']=='COMPLETE' and x['identity']==digest(dict(row=row,settings=settings[c['name']])) and x['problem_id']==row['problem_id'] and x['prompt_token_ids']==row['prompt_token_ids']
  if dest.exists():
   old=json.loads(dest.read_text());assert old['source_sha256']==source and old['scoring_signature']==signature
   if type(old['correct']) is bool:return c['name'],dest,old,True
  predicted=paired.candidate(x);reference=x['metadata']['final_answer'][0] if x['dataset']=='olympiad_bench' else x['solution'];assert predicted is None or isinstance(predicted,str);assert isinstance(reference,str) and reference.strip()
  item=dict(source=str(f),source_sha256=source,dataset=x['dataset'],problem_id=x['problem_id'],dataset_index=x['dataset_index'],finish_reason=x['finish_reason'],predicted=predicted,reference=reference)
  return c['name'],dest,item,False
 def record(group,v):
  counts['processed']+=1;g=groups[(group,v['dataset'])];g['n']+=1;g['correct']+=v['correct'];g['answer_correct']+=v['answer_correct'];g['timeouts']+=v['finish_reason']=='trace_timeout'
 async def producer():
  for c in configs:
   for start in range(0,len(rows),64):
    batch=await asyncio.gather(*(asyncio.to_thread(prepare,c,row) for row in rows[start:start+64]))
    for group,dest,item,reused in batch:
     if reused:counts['reused']+=1;record(group,item)
     else:await queue.put((group,dest,item))
  for _ in range(2000):await queue.put(None)
 async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=2000,limit_per_host=2000,keepalive_timeout=60),timeout=aiohttp.ClientTimeout(total=120),trust_env=False) as session:
  async def call(user):
   body=dict(model='gpt-4o-mini',temperature=0,max_tokens=512,stream=False,response_format={'type':'json_object'},messages=[dict(role='system',content=prompt),dict(role='user',content=user)])
   history=[]
   for attempt in range(8):
    await gate.enter();t=time.monotonic();status=None;retry=True;delay=0;counts['attempts']+=1
    try:
     async with session.post('https://api.openlux.ai/v1/chat/completions',json=body,headers={'Authorization':'Bearer '+key,'Accept':'application/json'},proxy=proxy,proxy_auth=auth,allow_redirects=False) as response:
      status=response.status;raw=await response.read();counts['http_'+str(status)]+=1
      if status==200:
       data=json.loads(raw);choice=data['choices'][0];assert choice.get('finish_reason')=='stop';decision=judge_client.validate(choice['message']['content']);counts['api_success']+=1;latencies.append(time.monotonic()-t)
       return dict(status='ok',**decision,api_model=data.get('model'),request_id=data.get('id'),usage=data.get('usage',{}),attempts=attempt+1,retry_history=history,judge_raw_content=choice['message']['content'])
      try:delay=float(response.headers.get('Retry-After','0'))
      except ValueError:delay=0
      retry=status in [408,409,425,429] or status>=500;history.append(dict(attempt=attempt+1,http_status=status,error_body_sha256=sha(raw)))
      if status==429:
       gate.until=max(gate.until,time.monotonic()+max(2,delay));await gate.set(max(16,gate.limit//2))
    except Exception as e:
     counts['transport_or_schema_errors']+=1;history.append(dict(attempt=attempt+1,error_type=type(e).__name__,http_status=status))
    finally:await gate.leave()
    if not retry:break
    await asyncio.sleep(max(delay,min(60,2**attempt))+random.random())
   return dict(status='api_error',is_correct=None,reason=None,retry_history=history,attempts=len(history))
  # Actual evaluator API checks before touching experiment scores.
  for candidate,reference,expected in [('2/4','1/2',True),('1','100',False),('My first guess was3. My final answer is4.','4',True)]:
   x=await call(json.dumps(dict(candidate_answer=candidate,reference_answer=reference)));assert x['status']=='ok' and x['is_correct']==expected
  atomic_json(P/'PROBE.json',dict(passed=True,at=time.time()));counts.clear();latencies.clear();print('API_PROBE_PASSED',flush=True)
  async def consumer():
   while True:
    task=await queue.get()
    if task is None:queue.task_done();return
    group,dest,item=task;user=json.dumps(dict(candidate_answer=item['predicted'],reference_answer=item['reference']),ensure_ascii=False)
    result=await call(user) if item['predicted'] and item['predicted'].strip() else dict(status='extraction_missing',is_correct=False,attempts=0,reason='empty extracted answer')
    result.update(item,answer_correct=result['is_correct'],correct=(False if item['finish_reason']=='trace_timeout' else result['is_correct']) if result['is_correct'] is not None else None,input_sha256=sha(user.encode()),scoring_signature=signature,completed_at=time.time())
    await asyncio.to_thread(atomic_json,dest,result)
    if result['correct'] is None:counts['unresolved']+=1;failures.append(str(dest))
    else:record(group,result)
    queue.task_done()
  async def reporter():
   while not done_event.is_set():
    z=dict(**counts,total=394450,active=gate.active,active_limit=gate.limit,peak_inflight=gate.peak,queue=queue.qsize(),elapsed_seconds=round(time.time()-started,1),updated_at=time.time());await asyncio.to_thread(atomic_json,P/'progress.json',z);print('PROGRESS',json.dumps(z),flush=True);await asyncio.sleep(10)
  async def tune():
   while counts['api_success']<100:await asyncio.sleep(1)
   for target in [256,512,1000,2000]:
    await gate.set(target);gate.peak=gate.active;before=counts.copy();li=len(latencies);t=time.monotonic()
    await asyncio.sleep(90)
    elapsed=time.monotonic()-t;success=counts['api_success']-before['api_success'];limited=counts['http_429']-before['http_429'];errors=counts['transport_or_schema_errors']-before['transport_or_schema_errors'];ls=sorted(latencies[li:]);n=counts['attempts']-before['attempts']
    z=dict(target=target,actual_peak=gate.peak,seconds=elapsed,success=success,success_per_second=success/elapsed,http429=limited,transport_or_schema_errors=errors,attempts=n,p50_seconds=ls[len(ls)//2] if ls else None,p95_seconds=ls[int(len(ls)*.95)] if ls else None,final_limit=gate.limit)
    stages.append(z);await asyncio.to_thread(atomic_json,P/'concurrency_test.json',dict(stages=stages));print('CONCURRENCY_STAGE',json.dumps(z),flush=True)
    if limited/max(1,n)>.01 or errors/max(1,n)>.05:break
   good=[z for z in stages if z['http429']/max(1,z['attempts'])<=.01 and z['transport_or_schema_errors']/max(1,z['attempts'])<=.05]
   if good:
    best=max(z['success_per_second'] for z in good);chosen=min(z['target'] for z in good if z['success_per_second']>=best*.9)
   else:chosen=128
   await gate.set(chosen);await asyncio.to_thread(atomic_json,P/'concurrency_test.json',dict(stages=stages,selected_concurrency=chosen,selected_at=time.time()));print('CONCURRENCY_SELECTED',chosen,flush=True)
  producer_task=asyncio.create_task(producer());report=asyncio.create_task(reporter())
  # Prepare a backlog to make the concurrency sweep measurable, then use one rolling queue.
  while queue.qsize()<4096 and not producer_task.done():await asyncio.sleep(1)
  if producer_task.done():await producer_task
  consumers=[asyncio.create_task(consumer()) for _ in range(2000)];tuner=asyncio.create_task(tune())
  try:await asyncio.gather(producer_task,*consumers)
  finally:
   done_event.set();report.cancel();tuner.cancel();await asyncio.gather(report,tuner,return_exceptions=True)
 if failures:atomic_json(P/'FAILED.json',dict(unresolved=failures));raise RuntimeError('Unresolved API errors; saved results resumable')
 assert counts['processed']==394450
 summary=[dict(config=g,dataset=ds,**z,accuracy=z['correct']/z['n'],answer_accuracy=z['answer_correct']/z['n']) for (g,ds),z in sorted(groups.items())]
 atomic_json(P/'summary.json',dict(rows=394450,groups=summary,scoring_signature=signature))
 first=R/'opd_rescore/grid_math_gpt4o_mini_openlux_20260916';assert json.loads((first/'COMPLETE.json').read_text())['rows']==56350
 atomic_json(P/'ALL_SCORING_COMPLETE.json',dict(rows=394450,first_round_rows=56350,all8_rows=450800,first_round_scores=str(first),completed_at=time.time(),scoring_signature=signature))
 atomic_json(G/'ALL_SCORING_COMPLETE.json',dict(rows=450800,additional_scores=str(P),first_round_scores=str(first),completed_at=time.time()))
 (P/'judge/relay.stop').touch();print('ALL_SCORING_COMPLETE450800',flush=True)
if __name__=='__main__':asyncio.run(main())
