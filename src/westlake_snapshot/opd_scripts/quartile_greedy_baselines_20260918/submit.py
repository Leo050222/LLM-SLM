import json,fcntl,subprocess,time
from pathlib import Path
from ablation_core import atomic_json
P=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix/quartile_support_k_20260918');S=Path(__file__).parent
lock=(P/'baseline_submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
f=P/'BASELINES_LAUNCH.json'
if f.exists():launch=json.loads(f.read_text())
else:
 launch=dict(generation_job_id=subprocess.check_output(['/soft/slurm/bin/sbatch','--parsable','--dependency=afterok:106797',str(S/'gpu.slurm')],text=True).strip().split(';')[0],parent_generation_job_id='106797',parent_scoring_job_id='106798',output_root=str(P),groups=['k0','teacher_greedy'],rows_added=1600,total_rows=5600,submitted_at=time.time());atomic_json(f,launch)
if 'scoring_job_id' not in launch:
 launch['scoring_job_id']=subprocess.check_output(['/soft/slurm/bin/sbatch','--parsable','--dependency=afterok:'+launch['generation_job_id']+':106798',str(S/'score.slurm')],text=True).strip().split(';')[0];atomic_json(f,launch)
print(json.dumps(launch),flush=True)
subprocess.run(['/soft/slurm/bin/squeue','-j',launch['generation_job_id']+','+launch['scoring_job_id'],'-o','%.18i %.26j %.9T %.8M %R'],check=True)
