"""Idempotent submission of two single-GPU replay jobs and final CPU verification."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
from run import OUT, atomic_json, load_manifest, sha

script = Path(__file__).resolve().parent
lock = (script/'submission.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
manifest, manifest_hash = load_manifest(OUT)
p = script/'LAUNCH.json'
state = json.loads(p.read_text()) if p.exists() else dict(output=str(OUT), manifest_sha256=manifest_hash,
        jobs={}, requested_gpu_total=2, dtype='BF16', submitted=time.time(),
        script_sha256={n:sha(script/n) for n in ['run.py','replay_base.py','gpu.slurm','verify.slurm']})
assert state['manifest_sha256'] == manifest_hash
if 'combined' in state['jobs']:
    print(json.dumps(state,indent=2))
    raise SystemExit(0)
assert state['script_sha256'] == {n:sha(script/n) for n in ['run.py','replay_base.py','gpu.slurm','verify.slurm']}
for role in ['student','teacher']:
    if role not in state['jobs']:
        job = subprocess.check_output(['/soft/slurm/bin/sbatch','--parsable',
            '--job-name=greedy_logits_'+role,str(script/'gpu.slurm'),role],text=True).strip().split(';')[0]
        assert job.isdigit()
        state['jobs'][role] = job; atomic_json(p,state)
if 'verify' not in state['jobs']:
    dependency = 'afterok:'+':'.join(state['jobs'][r] for r in ['student','teacher'])
    job = subprocess.check_output(['/soft/slurm/bin/sbatch','--parsable','--dependency='+dependency,
                                  str(script/'verify.slurm')],text=True).strip().split(';')[0]
    assert job.isdigit()
    state['jobs']['verify'] = job; atomic_json(p,state)
print(json.dumps(state,indent=2))
