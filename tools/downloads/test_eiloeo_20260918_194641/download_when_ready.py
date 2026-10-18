import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

local = Path(__file__).resolve().parent
remote = '/storage/yukaichengLab/wangxin/bos_export/test_eiloeo_20260918_194641'
name = 'test_eiloeo_no_weights.tar.gz'
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30', 'gpu-wangxin']

def status(state, **details):
    data = dict(state=state, time=time.strftime('%Y-%m-%d %H:%M:%S'), **details)
    temp = local / 'download_status.tmp'
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(local / 'download_status.json')
    print(json.dumps(data, ensure_ascii=False), flush=True)

os.chdir(local)
while True:
    if (local / 'CANCELLED').exists():
        status('cancelled', reason='User requested deletion of source archive')
        break
    command = (f'cd {remote} && if test -f DONE; then '
               'echo READY; cat size_bytes.txt; cat test_eiloeo_no_weights.tar.gz.sha256; '
               'elif test -f tar.exit; then echo ARCHIVE_NOT_VERIFIED; cat tar.exit; '
               'else echo WAITING; fi')
    try:
        result = subprocess.run(ssh + [command], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        status('waiting_for_connection')
        time.sleep(60)
        continue
    lines = result.stdout.splitlines()
    if result.returncode or not lines:
        status('waiting_for_connection', detail=result.stderr[-500:])
        time.sleep(60)
        continue
    if lines[0] != 'READY':
        status('waiting_for_verified_archive', remote_state=lines)
        time.sleep(60)
        continue
    size = int(lines[1])
    expected = lines[2].split()[0]
    free = shutil.disk_usage(local).free
    if free < size + 2 * 1024**3:
        status('waiting_for_disk_space', archive_bytes=size, free_bytes=free,
               required_free_bytes=size + 2 * 1024**3)
        time.sleep(60)
        continue
    status('downloading', archive_bytes=size)
    part = local / (name + '.partial')
    result = subprocess.run(['rsync', '--partial', '--progress', '-e',
                             'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30',
                             f'gpu-wangxin:{remote}/{name}', str(part)])
    if result.returncode:
        status('transfer_failed', returncode=result.returncode)
        break
    status('verifying', archive_bytes=size)
    digest = hashlib.sha256()
    with part.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected or part.stat().st_size != size:
        status('checksum_failed', expected=expected, actual=actual)
        break
    part.replace(local / name)
    (local / (name + '.sha256')).write_text(f'{expected}  {name}\n')
    status('complete', archive_bytes=size, sha256=expected, path=str(local / name))
    break
