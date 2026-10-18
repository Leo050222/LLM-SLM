"""Select only Slurm-allocated, healthy UUIDs; avoid gnho008's failed device."""
import csv,json,os,subprocess
from pathlib import Path
BAD='GPU-c33c45c6-61d6-36bc-e5ac-6d80e3bc4968'
slot=Path(os.environ['OPD_ONLINE_RUN'])
allocated=os.environ.get('SLURM_STEP_GPUS') or os.environ['SLURM_JOB_GPUS']
assert all(x.isdigit() for x in allocated.split(',')), allocated
allocated=set(map(int,allocated.split(',')))
assert len(allocated)==5, allocated
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[dict(index=int(a),uuid=b.strip(),used=int(c),free=int(d)) for a,b,c,d in csv.reader(raw.splitlines())]
# NVML renumbers visible GPUs inside the Slurm cgroup; match physical UUIDs.
physical_by_uuid={'GPU-cfe8b3f9-ba11-8e1b-21dd-74f4f6a17b61': 0, 'GPU-2f9180e8-fa07-fbb2-3030-a9cb50ee5582': 1, 'GPU-c33c45c6-61d6-36bc-e5ac-6d80e3bc4968': 2, 'GPU-24ff6dd2-c945-0876-6d2a-064dd4bed3f7': 3, 'GPU-050edb0f-9a6e-1ad5-ee48-c5ca77592845': 4, 'GPU-4e119421-71b9-06a5-054f-cdf880bb4687': 5, 'GPU-a62b0061-fce8-b11c-d392-df53adab095b': 6, 'GPU-6aa50f92-b9ab-cbfb-f010-6b06dd885d4f': 7}
for r in rows:
 r['physical_index']=physical_by_uuid[r['uuid']]
assert {r['physical_index'] for r in rows}==allocated, (allocated,rows)
healthy=[r for r in rows if r['uuid']!=BAD and r['used']<500 and r['free']>=74000]
required=int(os.environ['GRID_REPLICAS'])*2
assert len(healthy)>=required, rows
selected=healthy[:required]
record=dict(allocated=sorted(allocated),inventory=rows,selected=selected,excluded_uuid=BAD)
(slot/'gpu_selection.json').write_text(json.dumps(record,indent=2))
(slot/'selected_gpu_uuids.txt').write_text(','.join(r['uuid'] for r in selected)+'\n')
print('GPU_PREFLIGHT_PASSED',json.dumps(record),flush=True)
