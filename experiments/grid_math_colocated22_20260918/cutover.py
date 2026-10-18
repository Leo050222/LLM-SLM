"""Snapshot only after the old dispatcher stops; all new jobs share one immutable map."""
import os,json,time,fcntl,hashlib,subprocess
from pathlib import Path
from queue_rpc import atomic
P=Path(os.environ['GRID_ROOT']);S=Path(__file__).parent
record=P/'recovery_colocated22_20260918/migration.json'
# New helper can reserve and preflight its 6 GPUs before the old 16 are released.
atomic(Path(os.environ['OPD_ONLINE_RUN'])/'PREFLIGHT_COMPLETE.json',dict(job=os.environ['SLURM_JOB_ID'],at=time.time()))
for _ in range(360):
 entries=subprocess.check_output(['/soft/slurm/bin/squeue','-h','-j','106743','-o','%T'],text=True).strip()
 if not entries:break
 time.sleep(5)
else:raise RuntimeError('Source job still active; refusing concurrent ownership')
setup_lock=(P/'colocated22_setup.lock').open('a')
fcntl.flock(setup_lock,fcntl.LOCK_EX)
code_hash={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in S.glob('*.py')}
if not record.exists():
 lock=(P/'deadline_scheduler.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX)
 configs=json.loads((P/'configs.json').read_text())
 counts={c['name']:sum(1 for _ in (P/c['name']/'completed').glob('*.json')) for c in configs}
 assert len(configs)==175 and all(n<=2254 for n in counts.values())
 # Deterministic longest-remaining-first, normalized by 8/8/6 GPU capacity.
 capacities=(8,8,6);loads=[0,0,0];ownership={}
 for c in sorted(configs,key=lambda c:(-(2254-counts[c['name']]),c['name'])):
  shard=min(range(3),key=lambda i:loads[i]/capacities[i])
  ownership[c['name']]=shard;loads[shard]+=2254-counts[c['name']]
 rec=dict(source_job=106743,first_target_job=os.environ['SLURM_JOB_ID'],created_at=time.time(),source_manifests={c['name']:json.loads((P/c['name']/'manifest.json').read_text()) for c in configs},target_code_sha256=code_hash,preserved_completed=sum(counts.values()),counts_at_cutover=counts,ownership=ownership,pending_per_shard=loads,capacities=list(capacities))
 atomic(record,rec)
 print('CUTOVER_SNAPSHOT',rec['preserved_completed'],'PENDING',loads,flush=True)
else:
 rec=json.loads(record.read_text());assert rec['target_code_sha256']==code_hash
assert set(rec['ownership'])=={c['name'] for c in json.loads((P/'configs.json').read_text())}
print('CUTOVER_READY',flush=True)
