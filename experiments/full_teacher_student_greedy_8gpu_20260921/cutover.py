import os,json,time,subprocess,fcntl,hashlib
from pathlib import Path
from ablation_core import atomic_json
p=Path(os.environ['GRID_ROOT']);s=Path(__file__).parent
rec=p/'recovery_8gpu_20260921.json'
if rec.exists():
 assert json.loads(rec.read_text())['target_code_sha256']=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in s.glob('*.py')}
 raise SystemExit(0)
# Preflight of the new allocation has passed before stopping the original job.
subprocess.run(['/soft/slurm/bin/scancel','107918'],check=False)
for _ in range(120):
 state=subprocess.check_output(['/soft/slurm/bin/squeue','-h','-j','107918','-o','%T'],text=True).strip()
 if not state:break
 time.sleep(2)
else:raise RuntimeError('Original job has not stopped')
locks=[]
for role in ['student','teacher']:
 lock=(p/f'{role}_dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
source={}
for role in ['student','teacher']:
 f=p/f'{role}_manifest.json'
 if f.exists():source[role]=json.loads(f.read_text())
hashes={};counts={}
for c in json.loads((p/'configs.json').read_text()):
 for sub in ['completed','traces','timings']:
  fs=list((p/c['name']/sub).glob('*.json'));counts[f"{c['name']}/{sub}"]=len(fs)
  for f in fs:hashes[str(f.relative_to(p))]=hashlib.sha256(f.read_bytes()).hexdigest()
record=dict(source_job=107918,target_job=os.environ['SLURM_JOB_ID'],source_manifests=source,target_code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in s.glob('*.py')},preserved_hashes=hashes,counts=counts,created_at=time.time(),configs_sha256=hashlib.sha256((p/'configs.json').read_bytes()).hexdigest())
atomic_json(rec,record)
print('CUTOVER_SNAPSHOT',json.dumps(counts),flush=True)
