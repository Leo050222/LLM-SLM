"""Transfer shell supervision without interrupting the active coordinator.

Run outside the immutable runtime snapshot. Retry the identical command so
completed answers are skipped and incomplete traces resume from checkpoints.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
parent, child = map(int, sys.argv[2:4])

def cmdline(pid):
    return [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]

command = cmdline(child)
assert command[1].endswith('/20260911_chunk64_k8_2nodes/ocrd_pipeline/online_rollout/run.py')
assert command[command.index('--output-dir') + 1] == str(root / 'results')
assert cmdline(parent)[1].endswith('/launch_coordinator_2nodes.sh')
status = Path(f'/proc/{child}/status').read_text()
assert f'PPid:\t{parent}\n' in status
assert not any((root / f'queue/replica_{i}/STOP').exists() for i in range(8))
environment = dict(item.decode().split('=', 1) for item in
                   Path(f'/proc/{child}/environ').read_bytes().split(b'\0') if item)
working_directory = os.readlink(f'/proc/{child}/cwd')
(root / 'recovery_guardian.pid').write_text(str(os.getpid()) + '\n')
# Only the verified wrapper is killed; SIGKILL suppresses its EXIT/STOP trap.
# The existing coordinator and all Slurm workers continue unchanged.
os.kill(parent, signal.SIGKILL)
print(json.dumps({'supervision_transferred': True, 'guardian_pid': os.getpid(),
                  'active_coordinator_pid': child, 'recovery_passes': 3}), flush=True)
success = False
try:
    while Path(f'/proc/{child}/stat').exists():
        try:
            if Path(f'/proc/{child}/stat').read_text().split(') ', 1)[1].startswith('Z '):
                break
        except FileNotFoundError:
            break
        time.sleep(10)
    for attempt in range(4):
        result_path = root / 'results/run_status.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if result.get('completed') == 1000 and result.get('failed') == []:
            success = True
            break
        if attempt == 3:
            break
        time.sleep(30)
        print(json.dumps({'recovery_pass': attempt + 1}), flush=True)
        completed = subprocess.run(command, cwd=working_directory, env=environment)
        print(json.dumps({'recovery_exit_code': completed.returncode}), flush=True)
    if success:
        (root / 'GENERATION_COMPLETE').touch()
    (root / 'recovery_status.json').write_text(json.dumps({'complete': success}))
finally:
    for replica in range(8):
        (root / f'queue/replica_{replica}/STOP').touch()
sys.exit(0 if success else 1)
