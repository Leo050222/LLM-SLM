from pathlib import Path
import fcntl
import json
import os
import secrets
import shutil
import subprocess
import sys
import time

R = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
P = R / 'opd_eval_ablation/strong_reprompt_score_20261005'
J = P / 'judge'
S = Path(__file__).parent
P.mkdir(parents=True, exist_ok=True)
J.mkdir(parents=True, exist_ok=True)
lock = (J / 'authorization_setup.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
shutil.copy2(R / 'opd_eval_ablation/reverse_t2_k1_20260918/judge/system_prompt.txt', J / 'system_prompt.txt')
(J / 'EXTERNAL_SCORING_APPROVED.json').write_text(json.dumps({
    'authorized_by': 'user',
    'instruction': '帮我调用judge，给所有rollout评分，并发可以开大一点',
    'scope': 'Both strong-reprompt rollouts, 9262 total outputs; code via official harness and math/reasoning/AIME via gpt-4o-mini',
    'provider': 'api.openlux.ai',
    'model': 'gpt-4o-mini',
    'created_at': time.time(),
}, ensure_ascii=False, indent=2))
(J / 'relay.stop').unlink(missing_ok=True)
(J / 'relay.json').unlink(missing_ok=True)
if not (J / '.relay_token').exists():
    fd = os.open(J / '.relay_token', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(secrets.token_bytes(32))
log = (J / 'relay.log').open('ab')
proc = subprocess.Popen([sys.executable, str(S / 'relay.py')], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
for _ in range(30):
    if (J / 'relay.json').exists():
        break
    if proc.poll() is not None:
        raise RuntimeError('Relay exited before becoming ready')
    time.sleep(1)
else:
    raise RuntimeError('Relay did not start')
port = json.loads((J / 'relay.json').read_text())['port']
subprocess.run([sys.executable, str(S / 'judge_client.py'), '--tunnel-port', str(port), '--probe'], check=True)
job = subprocess.check_output(['/soft/slurm/bin/sbatch', '--parsable', str(S / 'score.slurm')], text=True).strip()
(P / 'score_job_id.txt').write_text(job + '\n')
print('SCORE_JOB', job, flush=True)
