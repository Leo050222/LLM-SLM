from pathlib import Path
import json,time,statistics,collections
r=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_dynamic/dynamic_k_entropy_20260914');slot=r/'slots/job_104045'
out={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'witnesses':{},'queues':{},'new_completed':{},'smoke':(slot/'SMOKE_PASSED.json').exists()}
for base in sorted((slot/'queue').glob('*/*')):
 if not base.is_dir():continue
 name=str(base.relative_to(slot/'queue')); w=base/'ENTROPY_NUMERICAL_WITNESS.json'
 if w.exists():out['witnesses'][name]=json.loads(w.read_text())
 ages=[]
 for f in (base/'requests').glob('*.json'):
  try:ages.append(time.time()-f.stat().st_mtime)
  except FileNotFoundError:pass
 files=sorted((base/'results').glob('*.json'),key=lambda f:f.stat().st_mtime)[-160:]
 values=collections.defaultdict(list)
 for f in files:
  x=json.loads(f.read_text())
  if 'error' in x:raise RuntimeError((name,x))
  op='entropy' if 'entropy' in x else 'score' if 'chosen' in x else 'generate'
  values[op].append(x)
 data={}
 for op,rows in values.items():
  data[op]={'n':len(rows)}
  for key in ('queue_wait_seconds','worker_batch_seconds','postprocess_seconds','worker_batch_size'):
   vals=sorted(x[key] for x in rows if key in x)
   if vals:data[op][key]={'mean':round(statistics.mean(vals),4),'p95':round(vals[int(.95*(len(vals)-1))],4),'max':round(max(vals),4)}
 out['queues'][name]={'pending':len(ages),'oldest_seconds':round(max(ages or [0]),2),'recent':data}
archive=json.loads((r/'recovery_104008_performance/migration.json').read_text())['preserved_files']
for group in ('math','code'):
 new=[]
 for f in (r/group/'completed').glob('*.json'):
  if str(f.relative_to(r)) not in archive:new.append(json.loads(f.read_text()))
 out['new_completed'][group]={'n':len(new),'finish':dict(collections.Counter(x['finish_reason'] for x in new))}
print(json.dumps(out,ensure_ascii=False,indent=2))
