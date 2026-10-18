import os,json,time,fcntl,subprocess,hashlib
from pathlib import Path
from queue_rpc import atomic
P=Path(os.environ['GRID_ROOT']);S=Path(__file__).parent
# Run only AFTER this two-node job has acquired all16 GPUs.
assert int(os.environ['SLURM_JOB_NUM_NODES'])==2
entries=subprocess.check_output(['/soft/slurm/bin/squeue','-h','-u','wangxin','-o','%i|%j'],text=True).splitlines()
query=next((x for x in entries if x.startswith('106405|')),'')
if query:
 assert query=='106405|math_grid_coloc8',query
 subprocess.run(['/soft/slurm/bin/scancel','106405'],check=True)
lock=(P/'deadline_scheduler.lock').open('a')
for _ in range(120):
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
 except BlockingIOError:time.sleep(1)
else:raise RuntimeError('Old dispatcher did not release lock')
record=P/'recovery_colocated16_20260918/migration.json'
if not record.exists():
 configs=json.loads((P/'configs.json').read_text())
 rec=dict(source_job=106405,target_job=os.environ['SLURM_JOB_ID'],created_at=time.time(),source_manifests={c['name']:json.loads((P/c['name']/'manifest.json').read_text()) for c in configs},target_code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in S.glob('*.py')},preserved_completed=sum(len(list((P/c['name']/'completed').glob('*.json'))) for c in configs))
 atomic(record,rec)
 print('CUTOVER_SNAPSHOT',rec['preserved_completed'],flush=True)
else:
 rec=json.loads(record.read_text());assert rec['target_code_sha256']=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in S.glob('*.py')}
print('CUTOVER_READY',flush=True)
