"""One-time explicit recovery of the two user-authorized grid jobs."""
import sys,json,subprocess,time,hashlib,fcntl,shutil
from pathlib import Path
from queue_rpc import atomic
from migration import hashes,compatible
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');S=Path(__file__).parent
kind=sys.argv[1];oldjob,rel= {'code':('104996','opd_eval_code/code_grid_x8_20260916'),'math':('104997','opd_eval_hard/math_reasoning_grid_x8_20260916')}[kind]

if kind=='code':
 gate=json.loads((R/'opd_eval_hard/math_reasoning_grid_x8_20260916/recovery_async_v3/benchmark_summary.json').read_text());assert gate['passed']
p=R/rel;rec=p/'recovery_async_v3';rec.mkdir(exist_ok=True)
with (rec/'lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if (rec/'submission.json').exists():print((rec/'submission.json').read_text());sys.exit(0)
 if not (rec/'migration.json').exists():
  job=subprocess.check_output(['/soft/slurm/bin/scontrol','show','job',oldjob],text=True)
  assert f'JobId={oldjob} ' in job and f'/opd_scripts/grid_fifo_20260916/{kind}.slurm' in job
  atomic(rec/'old_job.json',{'detail':job,'stop_requested_at':time.time()})
  subprocess.run(['/soft/slurm/bin/scancel',oldjob],check=True)
  for _ in range(60):
   state=subprocess.check_output(['/soft/slurm/bin/squeue','-h','-j',oldjob,'-o','%T'],text=True).strip()
   if not state:break
   time.sleep(1)
  else:raise RuntimeError('Old job is not terminal; no replacement submitted')
  sources={};preserved={};count=0
  for c in json.loads((p/'configs.json').read_text()):
   d=p/c['name'];f=d/'manifest.json'
   if f.exists():
    sources[c['name']]=json.loads(f.read_text());atomic(rec/'manifests'/f'{c["name"]}.json',sources[c['name']])
   for sub in ['completed','traces','timings']:
    for f in (d/sub).glob('*.json'):
     preserved[str(f.relative_to(p))]=hashlib.sha256(f.read_bytes()).hexdigest();count+=1
     if sub!='completed':
      dest=rec/'checkpoint_backup'/f.relative_to(p);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
   print('ARCHIVED',c['name'],count,flush=True)
  worker_identity=hashlib.sha256(json.dumps((S/'worker.py').read_text(),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  # Validate exact old -> new model migration against every real manifest before submission.
  for source in sources.values():
   target=json.loads(json.dumps(source))
   for v in target['models'].values():v.update(protocol=3,worker_sha256=worker_identity)
   compatible(source,target)
  atomic(rec/'preserved_hashes.json',preserved)
  atomic(rec/'migration.json',{'source_manifests':sources,'target_code_sha256':hashes(S),'target_worker_identity':worker_identity,'old_job':oldjob,'preserved_files':count,'archived_at':time.time(),'timing_note':'Prior cumulative timers retained at their last atomic save; original timer persistence granularity up to approximately2 seconds. Completed timeouts never rerun.'})
 # Replay migration on temporary copies to check canonical JSON and idempotence.
 import tempfile
 from migration import migrate
 record=json.loads((rec/'migration.json').read_text())
 with tempfile.TemporaryDirectory(dir=rec) as tmp:
  t=Path(tmp);atomic(t/'recovery_async_v3/migration.json',record)
  for group,source in record['source_manifests'].items():
   if (p/group/'VERIFIED.json').exists():continue
   target=json.loads(json.dumps(source));target.pop('migration',None)
   for v in target['models'].values():v.update(protocol=3,worker_sha256=record['target_worker_identity'])
   target['runtime_policy']['code_sha256']=record['target_code_sha256']
   f=t/group/'manifest.json';atomic(f,source);migrate(f,target);migrate(f,target)
 print('ALL_MANIFEST_MIGRATION_DRYRUN_PASSED',flush=True)
 extra=['--export=ALL,RUN_ASYNC_BENCHMARK=1'] if kind=='math' else []
 job=subprocess.check_output(['/soft/slurm/bin/sbatch','--parsable',*extra,str(S/f'{kind}.slurm')],text=True).strip()
 atomic(rec/'submission.json',{'job_id':job,'old_job':oldjob,'nodes':1,'gpus':8,'submitted_at':time.time()})
 with (p/'job_ids.txt').open('a') as f:f.write(job+'\n')
 print('SUBMITTED',kind,job,flush=True)
