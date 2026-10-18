"""Bind the two Slurm-allocated physical GPUs by UUID, avoiding CUDA ordinal drift."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

assert os.environ['SLURMD_NODENAME']=='gnho008'
allocated=os.environ['SLURM_JOB_GPUS'].split(',')
assert len(allocated)==2 and all(d.isdigit() for d in allocated), allocated
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv'],text=True)
inventory={r['index']:r for r in csv.DictReader(raw.splitlines(),skipinitialspace=True)}
devices=[inventory[d]['uuid'] for d in allocated]
assert 'GPU-c33c45c6-61d6-36bc-e5ac-6d80e3bc4968' not in devices, 'Slurm assigned the unhealthy physical GPU2'
assert len(set(devices))==2
root=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_analysis/intervention_logits_bf16_x8_20260920')
code=Path(__file__).with_name('run.py')
children=[]
try:
    for role,index,uuid in zip(['student','teacher'],allocated,devices):
        print('SLURM_GPU_BINDING',json.dumps(dict(role=role,physical_index=index,uuid=uuid,job=os.environ['SLURM_JOB_ID'])),flush=True)
        log=(root/f'{role}_008_{os.environ["SLURM_JOB_ID"]}.log').open('a')
        p=subprocess.Popen([sys.executable,'-u',str(code),'score','--role',role],
                           env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),stdout=log,stderr=subprocess.STDOUT)
        children.append((role,p,log))
    while any(p.poll() is None for _,p,_ in children):
        for role,p,_ in children:
            if p.poll() not in (None,0):
                raise RuntimeError(f'{role} failed with exit code {p.returncode}; see role log')
        time.sleep(2)
    assert all(p.returncode==0 for _,p,_ in children)
finally:
    for _,p,log in children:
        if p.poll() is None:p.terminate()
    for _,p,log in children:
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:p.kill();p.wait()
        log.close()
