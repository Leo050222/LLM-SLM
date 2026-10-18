import fcntl,json,subprocess,time
from pathlib import Path
from prepare import P
from ablation_core import atomic_json
S=Path(__file__).parent
assert (P/'PREFLIGHT_PASSED.json').exists()
lock=(P/'submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
receipt=P/'LAUNCH.json'
if receipt.exists():launch=json.loads(receipt.read_text())
else:
 launch=dict(experiment='quartile_support_k_20260918',submitted_at=time.time(),output_root=str(P),resources='one node, 8 H800, 64 CPUs, 512 GiB',generation_job_id=subprocess.check_output(['/soft/slurm/bin/sbatch','--parsable',str(S/'gpu.slurm')],text=True).strip().split(';')[0]);atomic_json(receipt,launch)
if 'scoring_job_id' not in launch:
 launch['scoring_job_id']=subprocess.check_output(['/soft/slurm/bin/sbatch','--parsable','--dependency=afterok:'+launch['generation_job_id'],str(S/'score.slurm')],text=True).strip().split(';')[0];atomic_json(receipt,launch)
print(json.dumps(launch),flush=True)
subprocess.run(['/soft/slurm/bin/squeue','-j',launch['generation_job_id']+','+launch['scoring_job_id'],'-o','%.18i %.24j %.9T %.8M %.5D %R'],check=True)
