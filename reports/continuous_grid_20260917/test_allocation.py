import os,subprocess,tempfile,json
from pathlib import Path
root=Path('experiments/grid_fifo_20260916/node.sh').resolve()
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);(p/'logs').mkdir();(p/'smoke.py').write_text('')
 (p/'worker_rank.sh').write_text('''#!/bin/bash
set -eu
rank=$SLURM_PROCID
printf '%s\n' "$$" > "$OPD_ONLINE_RUN/worker_$rank.pid"
mkdir -p "$OPD_ONLINE_RUN/queue/replica_$((rank/2))"
while [[ ! -f "$OPD_ONLINE_RUN/queue/replica_$((rank/2))/STOP" ]]; do sleep 0.1; done
''')
 (p/'dispatch.py').write_text('''import os,json
from pathlib import Path
p=Path(os.environ['OPD_ONLINE_RUN']);f=p/'passes.json';n=json.loads(f.read_text()) if f.exists() else 0
n+=1;f.write_text(json.dumps(n));(p/'SEGMENT_DONE.json').write_text('{}')
if n==3:(p/'ALL_GENERATION_COMPLETE.json').write_text('{}')
''')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',GRID_REPLICAS='4',SCALE_CODE=str(p),GRID_ROOT=str(p),OPD_ONLINE_RUN=str(p),OPD_PYTHON='python3')
 subprocess.run(['bash',str(root)],env=env,check=True,timeout=20)
 assert json.loads((p/'passes.json').read_text())==3
 assert len(list(p.glob('worker_*.pid')))==8
 assert len({f.read_text() for f in p.glob('worker_*.pid')})==8
 assert (p/'ALLOCATION_DONE').exists()
 assert len(list(p.glob('queue/replica_*/STOP')))==4
 print('PASS: three dispatcher passes, same eight worker processes, release only after full completion.')
