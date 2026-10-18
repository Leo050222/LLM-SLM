#!/bin/bash
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
"$OPD_PYTHON" - <<'CHECK'
import os,subprocess,time,json
ids=os.environ['CUDA_VISIBLE_DEVICES'];n=int(os.environ['GRID_REPLICAS']);assert len(ids.split(','))==n
for attempt in range(13):
 raw=subprocess.check_output(['nvidia-smi','-i',ids,'--query-gpu=index,memory.free,memory.used','--format=csv,noheader,nounits'],text=True)
 rows=[[int(v.strip()) for v in line.split(',')] for line in raw.splitlines()]
 assert len(rows)==n
 if all(free>=74000 for index,free,used in rows):
  print('GPU_PREFLIGHT_PASSED',os.environ.get('SLURMD_NODENAME'),rows,flush=True);break
 if attempt==12:raise RuntimeError('Allocated GPUs already occupied: '+str(rows))
 time.sleep(5)
CHECK
