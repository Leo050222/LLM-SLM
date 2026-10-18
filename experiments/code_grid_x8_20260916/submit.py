import fcntl
import json
import subprocess
from pathlib import Path

root = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
p = root / 'opd_eval_code/code_grid_x8_20260916'
with (p / 'submission.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    record = p / 'submission.json'
    if record.exists():
        print(record.read_text())
    else:
        assert len(json.loads((p / 'configs.json').read_text())) == 175
        assert len(json.loads((p / 'replicate0_audit.json').read_text())) == 25
        job = subprocess.check_output(['/soft/slurm/bin/sbatch', '--parsable', str(root / 'opd_scripts/code_grid_x8_20260916/gpu.slurm')], text=True).strip()
        record.write_text(json.dumps(dict(job_id=job, nodes=1, gpus=8, gpu_type='H800')))
        (p / 'job_ids.txt').write_text(job + '\n')
        print(record.read_text())
