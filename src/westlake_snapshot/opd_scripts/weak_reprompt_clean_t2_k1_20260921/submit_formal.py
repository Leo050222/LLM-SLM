"""Submit exactly one formal job only after all pilot gates pass."""
from pathlib import Path
import os,json,hashlib,fcntl,subprocess
from ablation_core import atomic_json
P=Path(os.environ['GRID_ROOT']);S=Path(__file__).parent
lock=(P/'formal_submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
receipt=P/'FORMAL_SUBMITTED.json'
if receipt.exists():print('ALREADY_SUBMITTED',receipt.read_text());raise SystemExit(0)
v=json.loads((P/'VALIDATION_PASSED.json').read_text());assert v['passed'] and not v['errors'] and len(v['results'])==12
assert v['code_sha256']=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in S.glob('*.py')},'Code changed after validation'
parent=os.environ['SLURM_JOB_ID']
r=subprocess.run(['/soft/slurm/bin/sbatch','--parsable',f'--dependency=afterok:{parent}',str(S/'gpu.slurm')],check=True,capture_output=True,text=True)
job=int(r.stdout.strip().split(';')[0]);atomic_json(receipt,dict(job_id=job,validation_job=int(parent),nodes=1,gpus=8,rows=4571,trace_limit_seconds=2000));print('FORMAL_SUBMITTED',job,flush=True)
