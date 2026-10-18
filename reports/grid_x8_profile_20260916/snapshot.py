import os,json,time,heapq,statistics,collections
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
def stats(v):
 v=sorted(v)
 return {'n':len(v),'mean':statistics.mean(v),'p50':v[len(v)//2],'p95':v[min(len(v)-1,int(len(v)*.95))],'max':max(v)} if v else {}
out={'observed_at':time.time(),'experiments':{}}
for label,rel,job in [('code','opd_eval_code/code_grid_x8_20260916',104980),('math','opd_eval_hard/math_reasoning_grid_x8_20260916',104982)]:
 p=R/rel;slot=p/'slots'/f'job_{job}'; now=time.time();cfg=json.loads((p/'configs.json').read_text());rows={r['dataset_index']:r for r in map(json.loads,(p/'prompts.jsonl').read_text().splitlines())}
 e={'total_new':len(cfg)*len(rows),'groups':{},'workers':[]};allmt=[]
 for c in cfg:
  d=p/c['name'];folder=d/'completed'
  files=[(x.stat().st_mtime,Path(x.path)) for x in os.scandir(folder)] if folder.exists() else []
  if not files:continue
  allmt += [t for t,_ in files]
  # Current outputs are only a few thousand; read to derive actual per-dataset cost.
  dataset=collections.defaultdict(lambda: {'n':0,'timeout':0,'seconds':0.,'tokens':0,'rounds':0})
  for _,f in files:
   x=json.loads(f.read_text());v=dataset[x['dataset']];v['n']+=1;v['timeout']+=x.get('finish_reason')=='trace_timeout';v['seconds']+=x.get('trace_elapsed_seconds',0);v['tokens']+=len(x['response_token_ids']);v['rounds']+=x['rounds']
  e['groups'][c['name']]={'n':len(files),'verified':(d/'VERIFIED.json').exists(),'datasets':dict(dataset)}
 log=(slot/'logs/dispatch.log').read_text();active=set();done_durations=[];errors=[]
 for line in log.splitlines():
  a=line.split()
  if a[:1]==['TRACE_START']: active.add((a[1],int(a[2])))
  if a[:1]==['TRACE_DONE']:
   active.discard((a[1],int(a[2])));done_durations.append(float(a[-1]))
  if a[:1]==['TRACE_ERROR']:errors.append(line)
 e.update(completed=len(allmt),last_completion_age=now-max(allmt) if allmt else None,completed_last_5m=sum(t>now-300 for t in allmt),completed_last_10m=sum(t>now-600 for t in allmt),trace_seconds=stats(done_durations),errors=errors,active_count=len(active))
 active_rows=[]
 for g,i in active:
  try:
   x=json.loads((p/g/'traces'/f'{i:05d}.json').read_text());tim=json.loads((p/g/'timings'/f'{i:05d}.json').read_text())
   active_rows.append({'dataset':rows[i]['dataset'],'seconds':tim['elapsed_seconds'],'tokens':len(x['response_token_ids']),'rounds':x['rounds']})
  except FileNotFoundError:pass
 e['active_by_dataset']=dict(collections.Counter(x['dataset'] for x in active_rows));e['active_seconds']=stats([x['seconds'] for x in active_rows]);e['active_tokens']=stats([x['tokens'] for x in active_rows])
 for pair in range(4):
  for role in ['student','teacher']:
   b=slot/'queue'/f'replica_{pair}'/role; pending=[]
   for f in (b/'requests').glob('*.json'):
    try:
     age=now-f.stat().st_mtime;x=json.loads(f.read_text());pending.append((age,x['op'],len(x['prefix'])))
    except FileNotFoundError:pass
   latest=heapq.nlargest(120,((x.stat().st_mtime,Path(x.path)) for x in os.scandir(b/'results')))
   samples=[]
   for mt,f in latest:
    x=json.loads(f.read_text());samples.append(x)
   e['workers'].append({'pair':pair,'role':role,'pending':len(pending),'pending_age':stats([v[0] for v in pending]),'pending_ops':dict(collections.Counter(v[1] for v in pending)),'pending_prefix':stats([v[2] for v in pending]),'batch_seconds':stats([x['worker_batch_seconds'] for x in samples if 'worker_batch_seconds' in x]),'batch_size':stats([x['worker_batch_size'] for x in samples if 'worker_batch_size' in x]),'result_errors':sum('error' in x for x in samples)})
 out['experiments'][label]=e
print(json.dumps(out))
