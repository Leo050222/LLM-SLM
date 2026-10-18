import json, hashlib, subprocess, time
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P=R/'opd_eval_hard/math_reasoning_grid_x8_20260916'
S=R/'opd_scripts/grid_math_8gpu_20260917'
for attempt in range(10):
 q=subprocess.check_output(['/soft/slurm/bin/squeue','-h','-u','wangxin','-o','%i %j']).decode()
 if not any('math_grid' in x for x in q.splitlines()):break
 time.sleep(5)
assert not any('math_grid' in x for x in q.splitlines()),q
rec=P/'recovery_8gpu_20260917'; rec.mkdir(exist_ok=True)
assert not (rec/'submission.json').exists(), 'Already submitted'
sources={c['name']:json.loads((P/c['name']/'manifest.json').read_text()) for c in json.loads((P/'configs.json').read_text()) if (P/c['name']/'manifest.json').exists()}
obj={'source_job':105199,'source_manifests':sources,'target_code_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in S.glob('*.py')},'created_at':time.time(),'replicas':4,'concurrency':128,'gpus':8,'exclusive':True,'preserve_trace_budget':True}
(rec/'migration.json').write_text(json.dumps(obj))
# Migration gate must permit resource-only changes and reject scientific changes.
import sys, copy
sys.path.insert(0,str(S));from migration import migrate
import tempfile
with tempfile.TemporaryDirectory() as td:
 t=Path(td); (t/'recovery_8gpu_20260917').mkdir(); (t/'recovery_8gpu_20260917/migration.json').write_text(json.dumps(obj))
 for g,old in sources.items():
  path=t/g/'manifest.json';path.parent.mkdir();path.write_text(json.dumps(old))
  target=copy.deepcopy(old);target.pop('migration',None);target['execution']={'replicas':4,'concurrency':128};target['runtime_policy'].update(replicas=4,concurrency=128,code_sha256=obj['target_code_sha256'])
  migrate(path,target);migrate(path,target)
  bad=copy.deepcopy(target);bad['settings']['seed']+=1
  try:migrate(path,bad)
  except AssertionError:pass
  else:raise AssertionError('Scientific change accepted')
print('MIGRATION_PREFLIGHT_OK',len(sources),flush=True)
# Trace and timing files are untouched. Capture exact completed content hashes.
hashes={}
for c in json.loads((P/'configs.json').read_text()):
 for f in (P/c['name']/'completed').glob('*.json'):
  hashes[str(f.relative_to(P))]=hashlib.sha256(f.read_bytes()).hexdigest()
(rec/'preserved_completed_sha256.json').write_text(json.dumps(hashes))
print('PRESERVED_COMPLETED',len(hashes),flush=True)
subprocess.run(['bash','-n',str(S/'math.slurm')],check=True)
j=subprocess.check_output(['/soft/slurm/bin/sbatch','--parsable',str(S/'math.slurm')]).decode().strip()
(rec/'submission.json').write_text(json.dumps({'job_id':j,'completed_before':len(hashes),'timestamp':time.time()}))
with (P/'job_ids.txt').open('a') as f:f.write(j+'\n')
print('SUBMITTED',j,flush=True)
